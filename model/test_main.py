import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import random
import warnings
import gc
import psutil
from collections import deque

from config import VideoStegoConfig
from hardware_interface import HardwareInterface
from compiler import CIMCompiler
from robot_interface import RobotInterface
from models import DenseEncoder, DenseDecoder, BasicCritic, PositionPredictor
from aean import AEANConfig, AdaptiveErrorAbsorptionNetwork
from video_stream import RealTimeVideoStream
from tracker import PeopleTracker
from visualizer import VideoStegoVisualizer

warnings.filterwarnings('ignore')

class VideoStegoSystem:
    def __init__(self, config=None):
        if config is None:
            config = VideoStegoConfig()
        self.config = config
        self.device = torch.device(config.device)
        
        self.hardware = HardwareInterface(config)
        self.compiler = CIMCompiler(config)
        self.robot = RobotInterface(config)
        self.robot.connect()
        
        self.encoder = DenseEncoder(config.data_depth, config.hidden_size).to(self.device)
        self.decoder = DenseDecoder(config.data_depth, config.hidden_size).to(self.device)
        self.critic = BasicCritic(config.hidden_size).to(self.device)
        self.position_predictor = PositionPredictor(state_dim=4, hidden_dim=config.pred_hidden_dim, output_size=config.image_size).to(self.device)
        
        self.people_tracker = PeopleTracker(self.robot)
        
        aean_config = AEANConfig(
            input_dim=4,
            output_dim=4,
            learning_rate=config.ae_learning_rate,
            primary_update_ratio=10,
            use_bias=False,
            weight_init='zeros'
        )
        self.aean = AdaptiveErrorAbsorptionNetwork(aean_config)
        
        self.en_de_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=5e-5, betas=(0.5, 0.999)
        )
        self.cr_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=5e-5, betas=(0.5, 0.999)
        )
        self.pred_optimizer = torch.optim.Adam(
            self.position_predictor.parameters(),
            lr=config.pred_learning_rate
        )
        
        self.vgg = models.vgg16(pretrained=True).features[:8].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.prev_frame_info = None
        self.heatmap_cache = None
        self.frame_counter = 0
        self.prev_psnr = 0
        self.prev_acc = 0
        self.sadp_consecutive_counter = 0
        
        self.history = {
            'prev_psnr': [],
            'prev_acc': [],
            'curr_psnr': [],
            'curr_acc': [],
            'aean_weight_norm': [],
            'pred_loss': []
        }
        
        if self.hardware.detect_pcb():
            print("PCB board detected with 512x512 memristor array")
        else:
            print("Using hardware simulator with 512x512 memristor array")
        
        encoder_instructions = self.compiler.compile_model(self.encoder, (1, 3, config.image_size, config.image_size), 'encoder')
        self.compiler.save_instructions(os.path.join('./', 'encoder_instructions.txt'))
        
        predictor_instructions = self.compiler.compile_position_predictor(self.position_predictor, state_dim=4)
        self.compiler.save_instructions(os.path.join('./', 'predictor_instructions.txt'))
        
        aean_torch = self._build_aean_torch_model()
        aean_instructions = self.compiler.compile_aean(aean_torch, input_dim=4)
        self.compiler.save_instructions(os.path.join('./', 'aean_instructions.txt'))
        
        print(f"VideoStegoSystem initialized, device: {self.device}")
    
    def _build_aean_torch_model(self):
        class AEANTorchModel(nn.Module):
            def __init__(self, input_dim=4, output_dim=4):
                super().__init__()
                self.fc = nn.Linear(input_dim, output_dim, bias=False)
                self.fc.weight.data = torch.zeros(output_dim, input_dim)
            
            def forward(self, x):
                return self.fc(x)
        return AEANTorchModel(input_dim=4, output_dim=4).to(self.device)
    
    def clean_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _extract_frame_features(self, frame, tracking_regions):
        features = []
        for region in tracking_regions:
            x, y, w, h = region['bbox']
            if w > 0 and h > 0:
                region_tensor = frame[:, :, y:y+h, x:x+w]
                mean_val = region_tensor.mean().item()
                std_val = region_tensor.std().item()
                grad_x = torch.abs(region_tensor[:, :, :, 1:] - region_tensor[:, :, :, :-1]).mean().item()
                grad_y = torch.abs(region_tensor[:, :, 1:, :] - region_tensor[:, :, :-1, :]).mean().item()
                features.extend([mean_val, std_val, grad_x, grad_y])
        if len(features) < 4:
            features = [0.0] * 4
        return torch.FloatTensor(features[:4]).to(self.device)
    
    def compute_state_vector(self, frame_idx, total_frames, importance_level=0.5):
        return self.people_tracker.compute_state_vector(frame_idx, total_frames, importance_level)
    
    def perceptual_loss(self, img1, img2):
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        return F.mse_loss(feat1, feat2)
    
    def apply_stego_to_tracking_regions(self, frame, payload, tracking_regions, frame_idx=0):
        stego_full = self.encoder(frame, payload)
        mask = torch.zeros_like(frame)
        for region in tracking_regions:
            x, y, w, h = region['bbox']
            x = max(0, min(x, frame.size(3) - 1))
            y = max(0, min(y, frame.size(2) - 1))
            w = min(w, frame.size(3) - x)
            h = min(h, frame.size(2) - y)
            if w > 0 and h > 0:
                mask[:, :, y:y+h, x:x+w] = 1.0
        result = frame * (1 - mask) + stego_full * mask
        return result, mask
    
    def select_stego_positions(self, heatmap, gamma=0.5):
        probs = heatmap.squeeze().detach().cpu().numpy()
        positions = np.argwhere(probs > gamma)
        if positions.ndim == 2 and positions.shape[1] == 3:
            positions = positions[:, 1:]
        return positions
    
    def update_predictor(self, state, prev_heatmap, delta):
        self.pred_optimizer.zero_grad()
        pred_heatmap = self.position_predictor(state)
        if delta > 0:
            target_heatmap = torch.clamp(prev_heatmap + 0.1 * delta, 0, 1)
        else:
            target_heatmap = torch.clamp(prev_heatmap - 0.1 * abs(delta), 0, 1)
        if pred_heatmap.shape != target_heatmap.shape:
            target_heatmap = target_heatmap[:pred_heatmap.size(0)]
        pred_loss = F.binary_cross_entropy(pred_heatmap, target_heatmap)
        pred_loss.backward()
        self.pred_optimizer.step()
        return pred_loss.item()
    
    def _compute_vertical_correlation_loss(self, cover, stego):
        with torch.no_grad():
            cover_v1 = cover[:, :, :-1, :].flatten(1)
            cover_v2 = cover[:, :, 1:, :].flatten(1)
            cover_corr = F.cosine_similarity(cover_v1, cover_v2, dim=1).mean()
        stego_v1 = stego[:, :, :-1, :].flatten(1)
        stego_v2 = stego[:, :, 1:, :].flatten(1)
        stego_corr = F.cosine_similarity(stego_v1, stego_v2, dim=1).mean()
        return F.mse_loss(stego_corr, cover_corr)
    
    def _calculate_ssim(self, img1, img2):
        C1, C2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(img1, 11, 1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, 1, padding=5)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1**2, 11, 1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2**2, 11, 1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1*img2, 11, 1, padding=5) - mu1_mu2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return torch.clamp(ssim_map.mean(), 0, 1)
    
    def train_step(self, video_clip, importance_level=0.5):
        self.clean_gpu_memory()
        batch_size, C, T, H, W = video_clip.shape
        video_clip = video_clip.to(self.device)
        total_metrics = {
            'total_loss': 0, 'recon_loss': 0, 'adv_loss': 0, 
            'percep_loss': 0, 'corr_loss': 0, 'pred_loss': 0,
            'psnr': 0, 'acc': 0, 'ssim': 0,
            'prev_psnr': 0, 'prev_acc': 0,
            'aean_weight_norm': 0
        }
        prev_frame = None
        prev_payload = None
        prev_stego = None
        prev_people_info = None
        prev_heatmap = None
        
        for t in range(T):
            if t % 2 == 0:
                self.clean_gpu_memory()
            current_frame = video_clip[:, :, t, :, :]
            people_info = self.people_tracker.detect_and_track_people(current_frame)
            tracking_regions = self.people_tracker.get_tracking_regions(current_frame, people_info)
            
            state_vec = self.compute_state_vector(t, T, importance_level)
            
            secret_tensor = self.people_tracker.extract_secret_from_overlay(people_info, current_frame)
            payload = torch.zeros(1, self.config.data_depth, H, W).to(self.device)
            payload_flat = payload.view(1, -1)
            secret_len = min(secret_tensor.numel(), payload_flat.numel())
            payload_flat[0, :secret_len] = secret_tensor[:secret_len].to(self.device)
            payload = payload_flat.view(1, self.config.data_depth, H, W)
            
            if prev_heatmap is None:
                pred_heatmap = self.position_predictor(state_vec.unsqueeze(0))
                positions = self.select_stego_positions(pred_heatmap)
                stego_final, mask = self.apply_stego_to_tracking_regions(current_frame, payload, tracking_regions, t)
            else:
                delta = self.aean.get_correction_signal(self.prev_psnr, self.prev_acc)
                pred_loss = self.update_predictor(state_vec.unsqueeze(0), prev_heatmap, delta)
                total_metrics['pred_loss'] += pred_loss
                
                pred_heatmap = self.position_predictor(state_vec.unsqueeze(0))
                positions = self.select_stego_positions(pred_heatmap)
                stego_final, mask = self.apply_stego_to_tracking_regions(current_frame, payload, tracking_regions, t)
            
            decoded_raw = self.decoder(stego_final)
            
            decoded_np = torch.sigmoid(decoded_raw).detach().cpu().numpy().flatten()[:4]
            m_raw = np.array(decoded_np, dtype=np.float32)
            
            if prev_frame is not None:
                x_aean = self._extract_frame_features(current_frame, tracking_regions).cpu().numpy()
                delta_m = self.aean.forward(x_aean)
                m_corrected = m_raw + delta_m
                decoded_corrected = torch.from_numpy(m_corrected).float().to(self.device)
                decoded_corrected = decoded_corrected.view_as(decoded_raw)
                
                residual = (payload - decoded_corrected.detach()).view(batch_size, -1)
                aean_loss = F.mse_loss(torch.from_numpy(delta_m).float().to(self.device), residual[:, :4].mean(dim=0))
                
                delta_psnr = self.config.psnr_target - self.prev_psnr
                delta_acc = self.config.acc_target - self.prev_acc
                delta = 0.3 * delta_psnr + 0.7 * delta_acc
                self.aean.update(x_aean, np.array([delta], dtype=np.float32))
            else:
                decoded_corrected = decoded_raw
                aean_loss = torch.tensor(0.0, device=self.device)
            
            recon_loss = F.binary_cross_entropy_with_logits(decoded_corrected, (payload + 1) / 2)
            perc_loss = self.perceptual_loss(current_frame, stego_final)
            cover_score = self.critic(current_frame)
            stego_score = self.critic(stego_final)
            adv_loss = F.binary_cross_entropy_with_logits(stego_score, torch.ones_like(stego_score))
            correlation_loss = self._compute_vertical_correlation_loss(current_frame, stego_final)
            
            total_loss = (self.config.lambda_recon * recon_loss + 
                         self.config.lambda_adv * adv_loss +
                         self.config.lambda_percep * perc_loss +
                         self.config.lambda_corr * correlation_loss +
                         0.1 * aean_loss)
            
            self.en_de_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            self.en_de_optimizer.step()
            
            with torch.no_grad():
                decoded_binary = (torch.sigmoid(decoded_corrected) > 0.5).float()
                payload_binary = ((payload + 1) / 2 > 0.5).float()
                acc = (decoded_binary == payload_binary).float().mean()
                mse = F.mse_loss(current_frame, stego_final)
                psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
                ssim_val = self._calculate_ssim(current_frame, stego_final)
            
            if prev_frame is not None:
                aean_weight_norm = np.linalg.norm(self.aean.weights)
                total_metrics['prev_psnr'] += self.prev_psnr
                total_metrics['prev_acc'] += self.prev_acc
                total_metrics['aean_weight_norm'] += aean_weight_norm
            
            self.prev_psnr = psnr.item()
            self.prev_acc = acc.item()
            self.history['prev_psnr'].append(self.prev_psnr)
            self.history['prev_acc'].append(self.prev_acc)
            self.history['curr_psnr'].append(psnr.item())
            self.history['curr_acc'].append(acc.item())
            
            total_metrics['total_loss'] += total_loss.item()
            total_metrics['recon_loss'] += recon_loss.item()
            total_metrics['adv_loss'] += adv_loss.item()
            total_metrics['percep_loss'] += perc_loss.item()
            total_metrics['corr_loss'] += correlation_loss.item()
            total_metrics['psnr'] += psnr.item()
            total_metrics['acc'] += acc.item()
            total_metrics['ssim'] += ssim_val.item()
            
            prev_frame = current_frame
            prev_payload = payload
            prev_stego = stego_final
            prev_people_info = people_info
            prev_heatmap = pred_heatmap
            
            hardware_out = self.hardware.run_inference('encoder', current_frame.cpu().numpy())
            mse_hw = self.hardware.get_mse('encoder', current_frame.cpu().numpy(), stego_final.cpu().numpy())
            
            del decoded_raw, decoded_corrected
            if t % 2 == 0:
                torch.cuda.empty_cache()
        
        for key in total_metrics:
            if key not in ['prev_psnr', 'prev_acc', 'aean_weight_norm'] or total_metrics[key] != 0:
                total_metrics[key] /= T
        
        self.clean_gpu_memory()
        return total_metrics
    
    def save_checkpoint(self, iteration, checkpoint_dir, metrics=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'video_model_iteration_{iteration:03d}.pth')
        torch.save({
            'iteration': iteration,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'position_predictor_state_dict': self.position_predictor.state_dict(),
            'en_de_optimizer_state_dict': self.en_de_optimizer.state_dict(),
            'cr_optimizer_state_dict': self.cr_optimizer.state_dict(),
            'pred_optimizer_state_dict': self.pred_optimizer_state_dict(),
            'aean_weights': self.aean.weights,
            'aean_bias': self.aean.bias,
            'metrics': metrics,
            'history': self.history
        }, checkpoint_path)
        print(f"Model saved: {checkpoint_path}")
        return checkpoint_path
    
    def train_on_video(self, save_dir='./video_stego_results'):
        stream = RealTimeVideoStream(
            camera_id=0,
            clip_length=self.config.clip_length,
            target_size=(self.config.image_size, self.config.image_size)
        )
        visualizer = VideoStegoVisualizer(save_dir=save_dir, create_timestamp_subdir=True)
        print(f"\n{'='*60}")
        print("Starting Real-Time Video Stego Training with People Tracking")
        print(f"{'='*60}")
        checkpoint_dir = os.path.join(visualizer.save_dir, 'checkpoints')
        
        original_width = self.config.image_size
        original_height = self.config.image_size
        
        for iteration in range(self.config.num_iterations):
            print(f"\nIteration {iteration+1}/{self.config.num_iterations}")
            iteration_metrics = []
            sample_video_clip = None
            for batch_idx in range(3):
                video_clip = stream.get_clip()
                if video_clip is None:
                    print("End of stream")
                    break
                video_clip = video_clip.unsqueeze(0)
                if batch_idx == 0:
                    sample_video_clip = video_clip
                importance = random.choice([0.3, 0.5, 0.7])
                metrics = self.train_step(video_clip, importance_level=importance)
                iteration_metrics.append(metrics)
                print(f"  Batch {batch_idx+1}: PSNR={metrics['psnr']:.2f}dB, Acc={metrics['acc']:.3f}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not iteration_metrics:
                break
            avg_metrics = {k: np.mean([m[k] for m in iteration_metrics if k in m]) 
                          for k in iteration_metrics[0].keys()}
            print(f"  Average: PSNR={avg_metrics['psnr']:.2f}dB, Acc={avg_metrics['acc']:.3f}")
            self.save_checkpoint(iteration, checkpoint_dir, avg_metrics)
            if len(self.history['curr_psnr']) > 0:
                visualizer.plot_training_curves_with_prev_curr(self.history, 
                                                                f'training_curves_iter_{iteration:04d}.png',
                                                                iteration=iteration)
            if sample_video_clip is not None:
                visualizer.visualize_frame_comparison(self, sample_video_clip, original_width, original_height, 
                                                      iteration, f'frame_comparison_iter_{iteration:04d}.png')
                visualizer.visualize_video_comparison(self, sample_video_clip, original_width, original_height, 
                                                      iteration, f'video_comparison_iter_{iteration:04d}.png')
            
            self.compiler.update_parameter('encoder', 'IT', 100 + iteration * 10)
            self.compiler.update_parameter('position_predictor_fc1', 'IT', 100 + iteration * 10)
            self.compiler.update_parameter('position_predictor_fc2', 'IT', 100 + iteration * 10)
            self.compiler.update_parameter('aean_fc', 'IT', 100 + iteration * 10)
            self.compiler.save_instructions(os.path.join(checkpoint_dir, f'instructions_iter_{iteration}.txt'))
        
        stream.release()
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        if len(self.history['curr_psnr']) > 0:
            visualizer.plot_training_curves_with_prev_curr(self.history, 'final_training_curves.png')
        excel_path = visualizer.save_all_excel_data('video_stego_complete_data.xlsx')
        print(f"\nAll data exported to Excel: {excel_path}")
        if sample_video_clip is not None:
            visualizer.visualize_frame_comparison(self, sample_video_clip, original_width, original_height, 
                                                  self.config.num_iterations, 'frame_comparison_final.png')
            visualizer.visualize_video_comparison(self, sample_video_clip, original_width, original_height, 
                                                  self.config.num_iterations, 'video_comparison_final.png')
        print(f"\n{'='*60}")
        print("Generating final stego video...")
        print(f"{'='*60}")
        self.generate_final_stego_video_from_stream(original_width, original_height, visualizer.subdirs['generated_videos'])
        return visualizer.save_dir
    
    def generate_final_stego_video_from_stream(self, original_width, original_height, save_dir, duration_seconds=10):
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.eval()
        self.decoder.eval()
        self.position_predictor.eval()
        
        import gc
        import psutil
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        stream = RealTimeVideoStream(
            camera_id=0,
            clip_length=1,
            target_size=(self.config.image_size, self.config.image_size)
        )
        fps = int(stream.fps)
        total_frames = duration_seconds * fps
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_video_path = os.path.join(save_dir, f'combined_stego_video_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_width = original_width * 2
        combined_height = original_height * 2
        out_combined = cv2.VideoWriter(combined_video_path, fourcc, fps, (combined_width, combined_height))
        
        if not out_combined.isOpened():
            print("Error: Could not open video writer")
            return
        
        frame_idx = 0
        error_count = 0
        max_errors = 5
        
        try:
            with torch.no_grad():
                while frame_idx < total_frames:
                    try:
                        if frame_idx % 50 == 0:
                            process = psutil.Process()
                            memory_usage = process.memory_info().rss / 1024 / 1024
                            print(f"Memory usage: {memory_usage:.2f} MB, Processed {frame_idx}/{total_frames} frames")
                            if memory_usage > 4000:
                                print("Memory usage too high, forcing garbage collection")
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        
                        clip = stream.get_clip()
                        if clip is None:
                            break
                        frame_tensor = clip.unsqueeze(0).to(self.device)
                        
                        people_info = self.people_tracker.detect_and_track_people(frame_tensor)
                        tracking_regions = self.people_tracker.get_tracking_regions(frame_tensor, people_info)
                        
                        state_vec = self.compute_state_vector(frame_idx, total_frames, 0.5)
                        
                        secret_tensor = self.people_tracker.extract_secret_from_overlay(people_info, frame_tensor)
                        H, W = frame_tensor.shape[2], frame_tensor.shape[3]
                        payload = torch.zeros(1, self.config.data_depth, H, W).to(self.device)
                        payload_flat = payload.view(1, -1)
                        secret_len = min(secret_tensor.numel(), payload_flat.numel())
                        payload_flat[0, :secret_len] = secret_tensor[:secret_len].to(self.device)
                        payload = payload_flat.view(1, self.config.data_depth, H, W)
                        
                        pred_heatmap = self.position_predictor(state_vec.unsqueeze(0))
                        positions = self.select_stego_positions(pred_heatmap)
                        
                        stego_final, mask = self.apply_stego_to_tracking_regions(frame_tensor, payload, tracking_regions, frame_idx)
                        
                        frame_np = ((frame_tensor[0].cpu().permute(1,2,0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                        frame_with_tracking = self.people_tracker.draw_overlay(frame_tensor, people_info)
                        tracking_np = ((frame_with_tracking[0].cpu().permute(1,2,0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                        stego_np = ((stego_final[0].cpu().permute(1,2,0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                        
                        diff_tracking_stego = np.abs(stego_np.astype(np.float32) - tracking_np.astype(np.float32))
                        diff_tracking_stego_1x = np.clip(diff_tracking_stego, 0, 255).astype(np.uint8)
                        diff_tracking_stego_1000x = np.clip(diff_tracking_stego * 1000, 0, 255).astype(np.uint8)
                        
                        frame_np_resized = cv2.resize(frame_np, (original_width, original_height))
                        tracking_np_resized = cv2.resize(tracking_np, (original_width, original_height))
                        stego_np_resized = cv2.resize(stego_np, (original_width, original_height))
                        diff_1x_resized = cv2.resize(diff_tracking_stego_1x, (original_width, original_height))
                        diff_1000x_resized = cv2.resize(diff_tracking_stego_1000x, (original_width, original_height))
                        
                        top_row = np.hstack((tracking_np_resized, stego_np_resized))
                        bottom_row = np.hstack((diff_1x_resized, diff_1000x_resized))
                        combined_frame = np.vstack((top_row, bottom_row))
                        
                        out_combined.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
                        
                        frame_idx += 1
                        
                        del frame_tensor, payload, stego_final, mask, frame_with_tracking
                        del frame_np, tracking_np, stego_np, diff_tracking_stego
                        del frame_np_resized, tracking_np_resized, stego_np_resized, diff_1x_resized, diff_1000x_resized
                        
                        if frame_idx % 30 == 0:
                            gc.collect()
                    
                    except Exception as e:
                        error_count += 1
                        print(f"Error processing frame {frame_idx}: {e}")
                        if error_count > max_errors:
                            print("Too many errors, stopping...")
                            break
                        continue
        
        except Exception as e:
            print(f"Fatal error during video generation: {e}")
        
        finally:
            stream.release()
            out_combined.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\nFinal combined video saved (processed {frame_idx} frames):")
        print(f"Combined video: {combined_video_path}")
        return {'combined': combined_video_path}

def main():
    device = 'cpu'
    print(f"Using device: {device}")
    config = VideoStegoConfig(
        data_depth=4,
        hidden_size=32,
        image_size=192,
        clip_length=4,
        batch_size=1,
        device=device,
        num_iterations=10
    )
    system = VideoStegoSystem(config)
    results_dir = system.train_on_video(save_dir='./video_stego_results')
    print(f"\nAll results saved to: {results_dir}")

if __name__ == '__main__':
    main()