import cv2
import torch
import numpy as np
import datetime
from ultralytics import YOLO
import supervision as sv

class PeopleTracker:
    def __init__(self, robot_interface):
        self.robot = robot_interface
        self.model = YOLO("yolov8n.pt")
        self.tracker = sv.ByteTrack()
        self.track_history = {}
        self.LINE_Y1 = None
        self.LINE_Y2 = None
        self.width = None
        self.height = None
        
    def initialize_lines(self, height, width):
        self.height = height
        self.width = width
        self.LINE_Y1 = height // 3
        self.LINE_Y2 = 2 * height // 3
        
    def detect_and_track_people(self, frame):
        if self.width is None or self.height is None:
            self.initialize_lines(frame.shape[2], frame.shape[3])
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 4:
                frame_np = ((frame[0].cpu().permute(1,2,0).numpy() + 1) / 2 * 255).astype(np.uint8)
            else:
                frame_np = ((frame.cpu().permute(1,2,0).numpy() + 1) / 2 * 255).astype(np.uint8)
        else:
            frame_np = frame
        results = self.model(frame_np, classes=[0])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        people_info = []
        for box, track_id in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((center_x, center_y))
            person_info = {
                'id': int(track_id),
                'bbox': (x1, y1, x2-x1, y2-y1),
                'center': (center_x, center_y),
                'class_name': f'Person_{track_id}',
                'confidence': 1.0
            }
            people_info.append(person_info)
        return people_info
    
    def draw_battery_icon(self, frame, x, y, width=60, height=25, percentage=87):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        cv2.rectangle(frame, (x + width, y + height//4), (x + width + 15, y + 3*height//4), (255, 255, 255), -1)
        fill_width = int((width - 4) * (percentage / 100))
        if fill_width > 0:
            cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), (0, 255, 0), -1)
    
    def draw_overlay(self, frame, people_info):
        robot_state = self.robot.get_state()
        
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 4:
                frame_np = ((frame[0].cpu().permute(1,2,0).numpy() + 1) / 2 * 255).astype(np.uint8)
                original_height, original_width = frame_np.shape[:2]
            else:
                frame_np = ((frame.cpu().permute(1,2,0).numpy() + 1) / 2 * 255).astype(np.uint8)
                original_height, original_width = frame_np.shape[:2]
        else:
            frame_np = frame.copy()
            original_height, original_width = frame_np.shape[:2]
        overlay = frame_np.copy()
        scale_x = original_width / self.width if self.width else 1.0
        scale_y = original_height / self.height if self.height else 1.0
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for person in people_info:
            color = colors[person['id'] % len(colors)]
            x, y, w, h = person['bbox']
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            cv2.rectangle(overlay, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, 2)
            cv2.putText(overlay, f"ID: {person['id']}", (x_scaled, y_scaled - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        current_time = robot_state['time'].strftime("%H:%M:%S")
        font_scale = 0.5
        thickness = 1
        cv2.putText(overlay, f"Time: {current_time}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        self.draw_battery_icon(overlay, 10, 30, 50, 20, robot_state['battery'])
        cv2.putText(overlay, f"{robot_state['battery']}%", (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay, f"GPS: {robot_state['lat']:.2f}, {robot_state['lon']:.2f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay, f"Temp: {robot_state['temp']:.1f}C", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay, f"Humidity: {robot_state['humidity']}%", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay, f"CPU: {robot_state['cpu_usage']:.1f}%", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(overlay, f"Memory: {robot_state['memory_usage']:.1f}%", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        overlay_tensor = torch.from_numpy(overlay / 255.0 * 2 - 1).permute(2, 0, 1).float()
        if isinstance(frame, torch.Tensor) and frame.dim() == 4:
            overlay_tensor = overlay_tensor.unsqueeze(0).to(frame.device)
        else:
            overlay_tensor = overlay_tensor.to(frame.device if isinstance(frame, torch.Tensor) else 'cpu')
        return overlay_tensor
    
    def get_tracking_regions(self, frame, people_info, padding=10):
        regions = []
        frame_h, frame_w = frame.shape[2], frame.shape[3]
        for person in people_info:
            x, y, w, h = person['bbox']
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame_w, x + w + padding)
            y2 = min(frame_h, y + h + padding)
            regions.append({
                'id': person['id'],
                'bbox': (x1, y1, x2-x1, y2-y1),
                'class_name': person['class_name'],
                'mask': None
            })
        return regions
    
    def extract_secret_from_overlay(self, people_info, frame):
        robot_state = self.robot.get_state()
        secret = []
        for person in people_info:
            secret.append(person['id'])
            secret.extend(person['bbox'])
        secret.extend([
            robot_state['battery'],
            robot_state['lat'],
            robot_state['lon'],
            robot_state['temp'],
            robot_state['humidity'],
            robot_state['cpu_usage'],
            robot_state['memory_usage']
        ])
        secret.extend([robot_state['time'].hour, robot_state['time'].minute, robot_state['time'].second])
        secret_array = np.array(secret, dtype=np.float32)
        secret_tensor = torch.from_numpy(secret_array).float()
        return secret_tensor
    
    def compute_state_vector(self, frame_idx, total_frames, importance_level=0.5):
        robot_state = self.robot.get_state()
        e_prog = robot_state['cpu_usage'] / 100.0
        n_fail = int((1 - robot_state['battery'] / 100.0) * 10)
        n_update = frame_idx / max(total_frames, 1)
        h_mem = np.array([e_prog, n_fail, n_update])
        s_t = importance_level
        x_t = np.concatenate([h_mem, [s_t]])
        return torch.FloatTensor(x_t)