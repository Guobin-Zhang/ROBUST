import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticResonanceLayer(nn.Module):
    def __init__(self, channels, alpha=0.05, beta=5.0, v_th=1):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.beta = beta
        self.v_th = v_th
        self.eta_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, message_bits):
        batch_size, C, H, W = features.shape
        if message_bits.shape[2] != H or message_bits.shape[3] != W:
            msg_flat = F.interpolate(message_bits, size=(H, W), mode='nearest')
        else:
            msg_flat = message_bits
        if msg_flat.size(1) != C:
            if msg_flat.size(1) < C:
                repeat_factor = C // msg_flat.size(1) + 1
                msg_flat = msg_flat.repeat(1, repeat_factor, 1, 1)[:, :C]
            else:
                msg_flat = msg_flat[:, :C]
        msg_mod = torch.tanh(msg_flat) * self.alpha
        noise = torch.randn_like(features) * 0.005
        f_noisy = features + noise * (1 + msg_mod)
        f_noisy_clamped = torch.clamp(f_noisy, -10, 10)
        f_sr = torch.sigmoid(self.beta * (f_noisy_clamped - self.v_th))
        with torch.no_grad():
            grad_x = F.conv2d(f_sr, weight=torch.tensor([[[[-1, 0, 1]]]], dtype=f_sr.dtype, device=f_sr.device).expand(C, 1, 1, 3), padding=(0, 1), groups=C)
            grad_y = F.conv2d(f_sr, weight=torch.tensor([[[[-1], [0], [1]]]], dtype=f_sr.dtype, device=f_sr.device).expand(C, 1, 3, 1), padding=(1, 0), groups=C)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            mean_sq = F.avg_pool2d(f_sr**2, kernel_size=3, stride=1, padding=1)
            mean = F.avg_pool2d(f_sr, kernel_size=3, stride=1, padding=1)
            local_var = torch.abs(mean_sq - mean**2) + 1e-6
            sigma_g, sigma_v = 1.0, 1.0
            mask_val = (grad_mag**2 / (2 * sigma_g**2 + 1e-6)) * (local_var / sigma_v)
            mask = torch.sigmoid(mask_val)
        gate = self.gate(features)
        output = f_sr * mask * gate + features * (1 - gate)
        return output, mask.detach()

class DenseEncoder(nn.Module):
    def __init__(self, data_depth=4, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msg_proj = nn.Conv2d(data_depth, hidden_size, kernel_size=1)
        self.sr_layer = StochasticResonanceLayer(hidden_size)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size + data_depth, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2 + data_depth, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + data_depth, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, image, data):
        x0 = self.conv1(image)
        msg_for_sr = self.msg_proj(data)
        x0_sr, sr_mask = self.sr_layer(x0, msg_for_sr)
        x1_in = torch.cat([x0_sr, data], dim=1)
        x1 = self.conv2(x1_in)
        x2_in = torch.cat([x0_sr, x1, data], dim=1)
        x2 = self.conv3(x2_in)
        x3_in = torch.cat([x0_sr, x1, x2, data], dim=1)
        delta_raw = self.conv4(x3_in) * 0.01
        mask = self.mask_conv(x0_sr)
        delta = delta_raw * mask
        stego = image + delta
        return torch.clamp(stego, -1.0, 1.0)

class DenseDecoder(nn.Module):
    def __init__(self, data_depth=4, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.sr_enhance = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.sr_attention = nn.Sequential(
            nn.Conv2d(hidden_size, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Conv2d(hidden_size * 2, data_depth, kernel_size=3, padding=1)
        
    def forward(self, image, ref_features=None):
        x0 = self.conv1(image)
        x0_sr = self.sr_enhance(x0)
        x1 = self.conv2(x0_sr)
        attn = self.sr_attention(x1)
        x1_attn = x1 * attn
        x2_in = torch.cat([x0_sr, x1_attn], dim=1)
        x2 = self.conv3(x2_in)
        x3_in = torch.cat([x0_sr, x2], dim=1)
        decoded = self.conv4(x3_in)
        return decoded

class BasicCritic(nn.Module):
    def __init__(self, hidden_size=32, input_size=192):
        super().__init__()
        self.global_conv = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_size, hidden_size*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size*2, 1),
        )
        
    def forward(self, image):
        return self.global_conv(image).view(-1)

class PositionPredictor(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=32, output_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size * output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, state):
        h = self.relu(self.fc1(state))
        output = self.fc2(h)
        heatmap = self.sigmoid(output)
        return heatmap.view(-1, 1, self.output_size, self.output_size)