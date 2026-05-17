# rpi5_fpga_controller.py
# High-level interface for memristor stego system control

import ctypes
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import struct
import time

# Load C extension
try:
    _gpio = ctypes.CDLL('./rpi5_gpio.so')
    _gpio.rpi5_gpio_init.restype = ctypes.c_int
    _gpio.spi_transfer_frame.argtypes = [
        ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16, 
        ctypes.c_uint16, ctypes.c_uint32, 
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int
    ]
except OSError:
    raise ImportError("Failed to load rpi5_gpio.so, compile with: gcc -O3 -shared -fPIC -o rpi5_gpio.so rpi5_gpio_driver.c")


@dataclass
class MemristorConfig:
    """Configuration for memristor array programming"""
    chip_id: int
    row: int
    col: int
    target_conductance: float  # in uS
    max_iterations: int = 20
    
    def to_fixed_point(self) -> int:
        """Convert conductance to 6-bit fixed point: G_code = (G - 1) * 63 / 99"""
        g_code = int((self.target_conductance - 1.0) * 63.0 / 99.0)
        return max(0, min(63, g_code))


class RPi5FPGAController:
    """
    Raspberry Pi 5 direct GPIO controller for FPGA-based memristor stego system.
    Implements SPI-like protocol for chip configuration, SADP programming, and inference.
    """
    
    # Command codes matching FPGA implementation
    CMD_READ_REG = 0x01
    CMD_WRITE_REG = 0x02
    CMD_SADP_PROGRAM = 0x10
    CMD_LOAD_WEIGHT = 0x20
    CMD_PREDICT_FWD = 0x30
    CMD_ENCODE_STEGO = 0x40
    CMD_DECODE_STEGO = 0x50
    CMD_READ_STATUS = 0x60
    CMD_RESET_FPGA = 0xFF
    
    # Status masks
    STATUS_CB_BUSY = 0x01
    STATUS_SADP_BUSY = 0x02
    STATUS_ENC_BUSY = 0x04
    STATUS_DEC_BUSY = 0x08
    
    def __init__(self):
        self._initialized = False
        self._irq_callback = None
        
    def initialize(self) -> bool:
        """Initialize GPIO interface"""
        if _gpio.rpi5_gpio_init() != 0:
            raise RuntimeError("GPIO initialization failed")
        self._initialized = True
        return True
    
    def close(self):
        """Cleanup GPIO resources"""
        if self._initialized:
            _gpio.rpi5_gpio_cleanup()
            self._initialized = False
    
    def _transfer(self, cmd: int, chip: int, row: int, col: int, 
                  data: int, resp_len: int = 4) -> bytes:
        """Low-level frame transfer"""
        resp = (ctypes.c_uint8 * resp_len)()
        _gpio.spi_transfer_frame(cmd, chip, row, col, data, resp, resp_len)
        return bytes(resp)
    
    def reset_fpga(self) -> bool:
        """Reset FPGA logic to initial state"""
        resp = self._transfer(self.CMD_RESET_FPGA, 0, 0, 0, 0, 1)
        return resp[0] == 0xFF
    
    def read_status(self) -> dict:
        """Read system status registers"""
        resp = self._transfer(self.CMD_READ_STATUS, 0, 0, 0, 0, 8)
        status_word = struct.unpack('>I', resp[:4])[0]
        error_word = struct.unpack('>I', resp[4:8])[0]
        
        return {
            'cb_busy': bool(status_word & self.STATUS_CB_BUSY),
            'sadp_busy': bool(status_word & self.STATUS_SADP_BUSY),
            'enc_busy': bool(status_word & self.STATUS_ENC_BUSY),
            'dec_busy': bool(status_word & self.STATUS_DEC_BUSY),
            'sadp_status': (status_word >> 4) & 0x03,
            'error_code': error_word
        }
    
    def wait_idle(self, timeout_sec: float = 10.0):
        """Poll until system idle"""
        start = time.time()
        while time.time() - start < timeout_sec:
            status = self.read_status()
            if not any([status['cb_busy'], status['sadp_busy'], 
                       status['enc_busy'], status['dec_busy']]):
                return True
            time.sleep(0.001)  # 1ms poll interval
        raise TimeoutError("System busy timeout")
    
    def sadp_program(self, config: MemristorConfig) -> Tuple[bool, int]:
        """
        Execute SADP (Sensitivity-Adaptive Device Programming) on target memristor.
        Performs 3-level sensitivity test, classification, and iterative programming.
        
        Returns:
            (success: bool, iterations: int)
        """
        # Wait for SADP controller idle
        self.wait_idle()
        
        # Send SADP command with target configuration
        g_fixed = config.to_fixed_point()
        resp = self._transfer(
            self.CMD_SADP_PROGRAM,
            config.chip_id,
            config.row,
            config.col,
            g_fixed,
            1
        )
        
        if resp[0] != 0x10:
            raise RuntimeError(f"SADP start failed: 0x{resp[0]:02X}")
        
        # Poll for completion
        start = time.time()
        iterations = 0
        
        while time.time() - start < 30.0:  # 30s timeout for full SADP
            status = self.read_status()
            
            if status['sadp_busy']:
                time.sleep(0.001)
                continue
            
            # SADP done, check status
            sadp_status = status['sadp_status']
            if sadp_status == 0:
                return True, iterations
            elif sadp_status == 1:
                raise RuntimeError("SADP failed: max iterations reached")
            elif sadp_status == 2:
                raise RuntimeError("SADP failed: verification failed")
            else:
                raise RuntimeError("SADP failed: timeout")
        
        raise TimeoutError("SADP programming timeout")
    
    def load_weight(self, chip_id: int, layer_name: str, 
                    weights: np.ndarray, row_offset: int = 0) -> bool:
        """
        Load neural network weights to crossbar array.
        Supports encoder, decoder, critic, predictor layers.
        """
        self.wait_idle()
        
        # Flatten and quantize weights to 6-bit conductance values
        weights_flat = weights.flatten()
        g_values = self._quantize_weights(weights_flat)
        
        # Program row by row
        for i, g in enumerate(g_values):
            row = row_offset + i // 512  # Assuming 512 columns per row
            col = i % 512
            
            config = MemristorConfig(chip_id, row, col, g)
            success, _ = self.sadp_program(config)
            
            if not success:
                return False
            
            if i % 100 == 0:
                print(f"Loaded {i}/{len(g_values)} weights...")
        
        return True
    
    def _quantize_weights(self, weights: np.ndarray) -> List[float]:
        """Quantize float weights to conductance range [1, 100] uS"""
        # Map weight range to conductance range
        w_min, w_max = weights.min(), weights.max()
        g_range = 99.0  # 100 - 1
        
        if abs(w_max - w_min) < 1e-6:
            return [50.5] * len(weights)  # Midpoint
        
        g_values = 1.0 + (weights - w_min) / (w_max - w_min) * g_range
        return g_values.tolist()
    
    def predict_forward(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Execute position predictor forward pass on FPGA.
        Input: 4-dim state vector [cpu_usage, battery, frame_idx, importance]
        Output: 16384-dim probability heatmap (128x128)
        """
        self.wait_idle()
        
        # Pack state vector into 32-bit fixed point
        state_packed = self._pack_state(state_vector)
        
        # Start prediction
        resp = self._transfer(
            self.CMD_PREDICT_FWD,
            7,  # Predictor on Chip 7
            0,
            0,
            state_packed,
            1
        )
        
        if resp[0] != 0x30:
            raise RuntimeError("Predictor start failed")
        
        # Poll for completion
        self.wait_idle()
        
        # Read back heatmap (16384 bytes = 128x128)
        heatmap = self._read_bulk_data(16384)
        return heatmap.reshape(128, 128)
    
    def _pack_state(self, state: np.ndarray) -> int:
        """Pack 4-dim state vector into 32-bit word"""
        # Each component: 8-bit fixed point [0, 255]
        packed = 0
        for i, val in enumerate(state[:4]):
            quantized = int(np.clip(val * 255, 0, 255))
            packed |= (quantized << (24 - i * 8))
        return packed
    
    def _read_bulk_data(self, num_bytes: int) -> np.ndarray:
        """Read large data block from FPGA (DMA-like)"""
        # For large transfers, use optimized burst read
        data = bytearray()
        chunk_size = 256  # Bytes per transfer
        
        for offset in range(0, num_bytes, chunk_size):
            this_chunk = min(chunk_size, num_bytes - offset)
            resp = self._transfer(
                self.CMD_READ_REG,
                0,
                offset >> 8,
                offset & 0xFF,
                0,
                this_chunk
            )
            data.extend(resp)
        
        return np.frombuffer(data, dtype=np.uint8)
    
    def encode_stego(self, frame: np.ndarray, message: np.ndarray,
                     heatmap: np.ndarray) -> np.ndarray:
        """
        Execute steganographic encoding pipeline.
        Inputs: frame [3, H, W], message bits, position heatmap
        Output: stego frame with embedded message
        """
        self.wait_idle()
        
        # Load frame and message to FPGA buffers
        self._load_frame_buffer(frame)
        self._load_message_buffer(message)
        self._load_heatmap_buffer(heatmap)
        
        # Start encoding
        resp = self._transfer(self.CMD_ENCODE_STEGO, 0, 0, 0, 0, 1)
        if resp[0] != 0x40:
            raise RuntimeError("Encoder start failed")
        
        self.wait_idle()
        
        # Read back stego frame
        return self._read_frame_buffer()
    
    def _load_frame_buffer(self, frame: np.ndarray):
        """Load image frame to FPGA input buffer"""
        # Quantize to 8-bit and transfer
        frame_uint8 = (np.clip(frame, -1, 1) * 127.5 + 127.5).astype(np.uint8)
        self._write_bulk_data(0x1000, frame_uint8.tobytes())  # Address 0x1000
    
    def _load_message_buffer(self, message: np.ndarray):
        """Load binary message to FPGA"""
        message_bits = np.packbits(message.astype(np.uint8))
        self._write_bulk_data(0x2000, message_bits.tobytes())  # Address 0x2000
    
    def _load_heatmap_buffer(self, heatmap: np.ndarray):
        """Load position heatmap"""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        self._write_bulk_data(0x3000, heatmap_uint8.tobytes())  # Address 0x3000
    
    def _write_bulk_data(self, base_addr: int, data: bytes):
        """Write large data block to FPGA"""
        for i in range(0, len(data), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            addr = base_addr + i
            self._transfer(
                self.CMD_WRITE_REG,
                0,
                (addr >> 16) & 0xFFFF,
                addr & 0xFFFF,
                word
            )
    
    def _read_frame_buffer(self) -> np.ndarray:
        """Read processed frame from FPGA output buffer"""
        data = self._read_bulk_data(3 * 128 * 128)  # Assuming 128x128 RGB
        frame = np.frombuffer(data, dtype=np.uint8).reshape(3, 128, 128)
        return (frame.astype(np.float32) / 127.5) - 1.0  # Back to [-1, 1]


# Convenience functions for system operation
def initialize_system() -> RPi5FPGAController:
    """Initialize and return FPGA controller"""
    ctrl = RPi5FPGAController()
    ctrl.initialize()
    
    # Reset and check status
    ctrl.reset_fpga()
    time.sleep(0.1)
    
    status = ctrl.read_status()
    print(f"System status: {status}")
    
    return ctrl


def program_encoder_weights(ctrl: RPi5FPGAController, weights_dict: dict):
    """Program all encoder layers to Chip 0-2"""
    layer_mapping = {
        'conv1': (0, 0),
        'conv2': (0, 512),
        'conv3': (1, 0),
        'conv4': (1, 512),
        'conv5': (2, 0),
    }
    
    for layer_name, (chip, offset) in layer_mapping.items():
        if layer_name in weights_dict:
            print(f"Programming {layer_name} to Chip {chip}...")
            ctrl.load_weight(chip, layer_name, weights_dict[layer_name], offset)


def run_inference_pipeline(ctrl: RPi5FPGAController, 
                          video_frame: np.ndarray,
                          secret_message: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Complete inference: predict -> encode -> decode"""
    
    # Step 1: Compute state vector (simplified)
    state = np.array([0.5, 0.8, 0.0, 0.7])  # [cpu, battery, frame, importance]
    
    # Step 2: Predict stego positions
    heatmap = ctrl.predict_forward(state)
    
    # Step 3: Encode stego
    stego_frame = ctrl.encode_stego(video_frame, secret_message, heatmap)
    
    return stego_frame, heatmap


if __name__ == "__main__":
    # Example usage
    ctrl = initialize_system()
    
    try:
        # Load pre-trained weights
        # weights = load_model_weights('stego_model.pth')
        # program_encoder_weights(ctrl, weights)
        
        # Run inference
        frame = np.random.randn(3, 128, 128).astype(np.float32) * 0.5
        message = np.random.randint(0, 2, size=1024)
        
        stego, positions = run_inference_pipeline(ctrl, frame, message)
        print(f"Stego frame shape: {stego.shape}")
        print(f"Position heatmap max: {positions.max()}")
        
    finally:
        ctrl.close()