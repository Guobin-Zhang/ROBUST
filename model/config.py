from dataclasses import dataclass

@dataclass
class VideoStegoConfig:
    data_depth: int = 4
    hidden_size: int = 32
    image_size: int = 192
    clip_length: int = 4
    batch_size: int = 1
    device: str = 'cpu'
    num_iterations: int = 10
    lambda_recon: float = 2.0
    lambda_adv: float = 0.1
    lambda_percep: float = 0.1
    lambda_corr: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 42
    
    ae_learning_rate: float = 0.01
    pred_learning_rate: float = 0.005
    ae_hidden_dim: int = 32
    pred_hidden_dim: int = 16
    psnr_target: float = 35.0
    acc_target: float = 0.95
    sadp_trigger_acc_threshold: float = 0.02
    sadp_trigger_consecutive_frames: int = 5
    importance_threshold: float = 0.9
    
    v_std: float = 1.8
    t_std: float = 50.0
    s_fine: float = 0.6
    s_coarse: float = 1.2
    
    hardware_array_size: int = 512
    hardware_num_chips: int = 8
    hardware_detection_path: str = '/dev/pcb_memristor'
    use_hardware_simulator: bool = True
    hardware_noise_std: float = 0.05
    adc_bits: int = 8
    dac_bits: int = 4