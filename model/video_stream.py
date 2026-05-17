import cv2
import torch
import numpy as np
from torchvision import transforms

class RealTimeVideoStream:
    def __init__(self, camera_id=0, transform=None, clip_length=4, target_size=(192, 192)):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.clip_length = clip_length
        self.target_size = target_size
        self.frame_buffer = []
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        print(f"Real-time video stream initialized, FPS: {self.fps}")

    def get_clip(self):
        while len(self.frame_buffer) < self.clip_length:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) < self.clip_length:
            return None
        clip = torch.stack(self.frame_buffer[:self.clip_length], dim=1)
        self.frame_buffer = self.frame_buffer[self.clip_length:]
        return clip

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb)
        return frame_tensor.unsqueeze(0)

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()