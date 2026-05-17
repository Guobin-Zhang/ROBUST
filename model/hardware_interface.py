import numpy as np
import time
import os
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class ADCSpec:
    channels: int = 256
    bits: int = 8
    vref: float = 2.0
    sample_rate_hz: float = 1e6


@dataclass
class DACSpec:
    channels: int = 512
    bits: int = 4
    vref: float = 2.0
    settling_time_ns: float = 100.0


@dataclass
class ArraySpec:
    rows: int = 512
    cols: int = 512
    num_subarrays: int = 8
    cell_bits: int = 5
    read_pulse_width_ns: float = 100.0


class MemristorTransport(ABC):

    @abstractmethod
    def open(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def write(self, addr: int, data: np.ndarray) -> int:
        ...

    @abstractmethod
    def read(self, addr: int, count: int) -> np.ndarray:
        ...


class SPITransport(MemristorTransport):

    def __init__(self, bus: int = 0, device: int = 0, speed_hz: int = 20_000_000):
        self.bus = bus
        self.device = device
        self.speed_hz = speed_hz
        self._fd = None
        self._spi_path = f"/dev/spidev{bus}.{device}"

    def open(self) -> bool:
        if not os.path.exists(self._spi_path):
            return False
        try:
            self._fd = os.open(self._spi_path, os.O_RDWR)
            return True
        except OSError:
            self._fd = None
            return False

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def write(self, addr: int, data: np.ndarray) -> int:
        if self._fd is None:
            return 0
        packet = addr.to_bytes(3, 'big') + data.astype(np.uint8).tobytes()
        try:
            return os.write(self._fd, packet)
        except OSError:
            return 0

    def read(self, addr: int, count: int) -> np.ndarray:
        if self._fd is None:
            return np.zeros(count, dtype=np.uint8)
        cmd = (0x80 | (addr & 0x7F)).to_bytes(1, 'big')
        try:
            os.write(self._fd, cmd)
            raw = os.read(self._fd, count)
            return np.frombuffer(raw, dtype=np.uint8)
        except OSError:
            return np.zeros(count, dtype=np.uint8)


class UARTTransport(MemristorTransport):

    def __init__(self, port: str = "/dev/ttyPS0", baudrate: int = 3_000_000):
        self.port = port
        self.baudrate = baudrate
        self._fd = None

    def open(self) -> bool:
        if not os.path.exists(self.port):
            return False
        try:
            self._fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY)
            return True
        except OSError:
            self._fd = None
            return False

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def write(self, addr: int, data: np.ndarray) -> int:
        if self._fd is None:
            return 0
        packet = b'\xAA' + addr.to_bytes(2, 'big') + len(data).to_bytes(2, 'big') + data.astype(np.uint8).tobytes() + b'\x55'
        try:
            return os.write(self._fd, packet)
        except OSError:
            return 0

    def read(self, addr: int, count: int) -> np.ndarray:
        if self._fd is None:
            return np.zeros(count, dtype=np.uint8)
        cmd = b'\xBB' + addr.to_bytes(2, 'big') + count.to_bytes(2, 'big') + b'\x55'
        try:
            os.write(self._fd, cmd)
            time.sleep(0.0001)
            raw = os.read(self._fd, count + 2)
            if len(raw) >= 2:
                return np.frombuffer(raw[1:-1], dtype=np.uint8)
            return np.zeros(count, dtype=np.uint8)
        except OSError:
            return np.zeros(count, dtype=np.uint8)


class SensorBus:

    def __init__(self, i2c_bus: int = 1):
        self.i2c_bus = i2c_bus
        self._i2c_path = f"/dev/i2c-{i2c_bus}"
        self._fd = None
        self._sensor_addrs: Dict[str, int] = {
            "temp": 0x48,
            "current": 0x40,
            "vcore": 0x41,
        }

    def open(self) -> bool:
        if not os.path.exists(self._i2c_path):
            return False
        try:
            self._fd = os.open(self._i2c_path, os.O_RDWR)
            return True
        except OSError:
            self._fd = None
            return False

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def read_temperature(self) -> float:
        if self._fd is None:
            return float('nan')
        return self._read_sensor_float(self._sensor_addrs["temp"], 0x00)

    def read_current_ma(self) -> float:
        if self._fd is None:
            return float('nan')
        raw = self._read_sensor_u16(self._sensor_addrs["current"], 0x01)
        return raw * 0.0125

    def read_vcore_mv(self) -> float:
        if self._fd is None:
            return float('nan')
        raw = self._read_sensor_u16(self._sensor_addrs["vcore"], 0x02)
        return raw * 1.25

    def read_power_mw(self) -> float:
        current_ma = self.read_current_ma()
        vcore_mv = self.read_vcore_mv()
        if np.isnan(current_ma) or np.isnan(vcore_mv):
            return float('nan')
        return current_ma * vcore_mv / 1000.0

    def _read_sensor_float(self, addr: int, reg: int) -> float:
        raw = self._read_sensor_u16(addr, reg)
        if raw == 0xFFFF:
            return float('nan')
        return raw / 256.0

    def _read_sensor_u16(self, addr: int, reg: int) -> int:
        try:
            cmd = (addr << 1).to_bytes(1, 'big') + reg.to_bytes(1, 'big')
            os.write(self._fd, cmd)
            raw = os.read(self._fd, 2)
            return int.from_bytes(raw, 'big')
        except OSError:
            return 0xFFFF


class ADCBank:

    def __init__(self, transport: MemristorTransport, spec: ADCSpec, base_addr: int = 0x2000):
        self.transport = transport
        self.spec = spec
        self.base_addr = base_addr

    def read_column(self, column_idx: int) -> np.ndarray:
        addr = self.base_addr + (column_idx * 4)
        raw = self.transport.read(addr, self.spec.channels * (self.spec.bits // 8))
        if len(raw) < self.spec.channels:
            return np.zeros(self.spec.channels, dtype=np.float32)
        scale = self.spec.vref / (2 ** self.spec.bits - 1)
        values = raw.astype(np.float32) * scale
        return values

    def read_all_columns(self, num_columns: int) -> np.ndarray:
        result = np.zeros((num_columns, self.spec.channels), dtype=np.float32)
        for c in range(num_columns):
            result[c] = self.read_column(c)
        return result.T


class DACBank:

    def __init__(self, transport: MemristorTransport, spec: DACSpec, base_addr: int = 0x1000):
        self.transport = transport
        self.spec = spec
        self.base_addr = base_addr

    def write_row(self, row_idx: int, values: np.ndarray) -> int:
        addr = self.base_addr + (row_idx * 4)
        scale = (2 ** self.spec.bits - 1) / self.spec.vref
        quantized = np.clip(values * scale, 0, 2 ** self.spec.bits - 1).astype(np.uint8)
        return self.transport.write(addr, quantized)

    def write_all_rows(self, values: np.ndarray) -> int:
        total = 0
        for r in range(min(values.shape[0], self.spec.channels)):
            total += self.write_row(r, values[r])
        return total


class GPIOPinController:

    def __init__(self, chip_path: str = "/dev/gpiochip0"):
        self.chip_path = chip_path
        self._lines: Dict[int, object] = {}
        self._chip = None
        try:
            import gpiod
            self._gpiod = gpiod
            if os.path.exists(chip_path):
                self._chip = gpiod.Chip(chip_path)
        except ImportError:
            self._gpiod = None

    def configure_output(self, pin: int, initial: int = 0) -> bool:
        if self._chip is None:
            return False
        try:
            line = self._chip.get_line(pin)
            line.request(consumer="hw_iface", type=self._gpiod.LINE_REQ_DIR_OUT, default_vals=[initial])
            self._lines[pin] = line
            return True
        except Exception:
            return False

    def configure_input(self, pin: int) -> bool:
        if self._chip is None:
            return False
        try:
            line = self._chip.get_line(pin)
            line.request(consumer="hw_iface", type=self._gpiod.LINE_REQ_DIR_IN)
            self._lines[pin] = line
            return True
        except Exception:
            return False

    def write(self, pin: int, value: int) -> None:
        line = self._lines.get(pin)
        if line is not None:
            try:
                line.set_value(value)
            except Exception:
                pass

    def read(self, pin: int) -> int:
        line = self._lines.get(pin)
        if line is not None:
            try:
                return line.get_value()
            except Exception:
                return 0
        return 0

    def release_all(self) -> None:
        for line in self._lines.values():
            try:
                line.release()
            except Exception:
                pass
        self._lines.clear()


class MemristorArrayController:

    def __init__(self, transport: MemristorTransport, array_spec: ArraySpec,
                 adc: ADCBank, dac: DACBank, gpio: GPIOPinController):
        self.transport = transport
        self.spec = array_spec
        self.adc = adc
        self.dac = dac
        self.gpio = gpio
        self._programming_pin = 27
        self._done_pin = 17

    def program_weights(self, layer_name: str, weights: np.ndarray,
                        row_offset: int = 0, col_offset: int = 0) -> bool:
        if len(weights.shape) == 4:
            out_c, in_c, kh, kw = weights.shape
            weights_2d = weights.reshape(out_c, -1)
        else:
            weights_2d = weights

        rows_needed = min(weights_2d.shape[0], self.spec.rows - row_offset)
        cols_needed = min(weights_2d.shape[1], self.spec.cols - col_offset)

        self.gpio.write(self._programming_pin, 1)
        time.sleep(0.00001)

        for r in range(rows_needed):
            row_data = np.zeros(cols_needed, dtype=np.float32)
            actual_cols = min(weights_2d.shape[1], cols_needed)
            row_data[:actual_cols] = weights_2d[r, :actual_cols]
            self.dac.write_row(row_offset + r, row_data)

        self.gpio.write(self._programming_pin, 0)

        timeout = time.time() + 0.1
        while time.time() < timeout:
            if self.gpio.read(self._done_pin) == 1:
                return True
            time.sleep(0.00001)
        return False

    def read_conductance_matrix(self, rows: int, cols: int) -> np.ndarray:
        readout = np.zeros((rows, cols), dtype=np.float32)
        for c in range(cols):
            col_data = self.adc.read_column(c)
            readout[:min(rows, len(col_data)), c] = col_data[:min(rows, len(col_data))]
        return readout

    def set_parameter(self, param_addr: int, value: int) -> None:
        data = np.array([value & 0xFF, (value >> 8) & 0xFF], dtype=np.uint8)
        self.transport.write(0x3000 + param_addr, data)

    def get_parameter(self, param_addr: int) -> int:
        raw = self.transport.read(0x3000 + param_addr, 2)
        if len(raw) >= 2:
            return int(raw[0]) | (int(raw[1]) << 8)
        return 0


class HardwareSimulatorV2:

    def __init__(self, array_size: int, num_subarrays: int, cell_bits: int = 5,
                 adc_bits: int = 8, dac_bits: int = 4):
        self.array_size = array_size
        self.num_subarrays = num_subarrays
        self.cell_bits = cell_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.weights: Dict[str, np.ndarray] = {}
        self.params: Dict[str, Dict[str, float]] = {}

    def load_weights(self, layer_name: str, weights: np.ndarray,
                     row_offset: int = 0, col_offset: int = 0) -> None:
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().detach().numpy()
        quantized = self._quantize_weights(weights.astype(np.float32))
        self.weights[layer_name] = quantized

    def set_parameter(self, layer_name: str, param_name: str, value: float) -> None:
        if layer_name not in self.params:
            self.params[layer_name] = {}
        self.params[layer_name][param_name] = value

    def _quantize_weights(self, w: np.ndarray) -> np.ndarray:
        levels = 2 ** self.cell_bits - 1
        return np.round(np.clip(w, -1.0, 1.0) * (levels / 2) + (levels / 2)) / (levels / 2) * 2.0 - 1.0

    def _dac_quantize(self, x: np.ndarray) -> np.ndarray:
        levels = 2 ** (self.dac_bits - 1) - 1
        return np.round(np.clip(x, -1.0, 1.0) * levels) / max(levels, 1)

    def _adc_quantize(self, x: np.ndarray) -> np.ndarray:
        levels = 2 ** (self.adc_bits - 1) - 1
        return np.round(np.clip(x, -1.0, 1.0) * levels) / max(levels, 1)

    def run_inference(self, layer_name: str, input_data: np.ndarray) -> np.ndarray:
        w = self.weights.get(layer_name)
        if w is None:
            if isinstance(input_data, torch.Tensor):
                return input_data.cpu().numpy()
            return input_data

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        input_4d = input_data.ndim == 4
        if input_4d:
            batch, c, h, w_shape = input_data.shape
            in_flat = input_data.reshape(batch, -1)
        else:
            in_flat = input_data.reshape(input_data.shape[0], -1)

        w_flat = w.reshape(w.shape[0], -1).T

        min_dim = min(w_flat.shape[0], in_flat.shape[1])
        w_flat = w_flat[:min_dim, :]
        in_flat = in_flat[:, :min_dim]

        in_quantized = self._dac_quantize(in_flat)
        out_raw = np.dot(in_quantized, w_flat)

        wcn = self.params.get(layer_name, {}).get('WCN', 1.0)
        if wcn != 1.0:
            out_raw = out_raw * wcn

        out_quantized = self._adc_quantize(out_raw)

        if input_4d:
            out_quantized = out_quantized.reshape(batch, w.shape[0], input_data.shape[2], input_data.shape[3])

        return out_quantized

    def get_mse(self, layer_name: str, input_data: np.ndarray, target_output: np.ndarray) -> float:
        hw_out = self.run_inference(layer_name, input_data)
        if isinstance(target_output, torch.Tensor):
            target_output = target_output.cpu().numpy()
        if hw_out.shape != target_output.shape:
            target_output = target_output.reshape(hw_out.shape)
        return float(np.mean((hw_out - target_output) ** 2))


class HardwareInterface:

    def __init__(self, config):
        self.config = config
        self.array_size = config.hardware_array_size
        self.num_subarrays = config.hardware_num_chips
        self.cell_bits = getattr(config, 'adc_bits', 8)
        self.dac_bits = getattr(config, 'dac_bits', 4)

        self.transport: Optional[MemristorTransport] = None
        self.sensor_bus: Optional[SensorBus] = None
        self.gpio: Optional[GPIOPinController] = None
        self.array_ctrl: Optional[MemristorArrayController] = None
        self.adc: Optional[ADCBank] = None
        self.dac: Optional[DACBank] = None

        self.detected = False

        self.simulator = HardwareSimulatorV2(
            array_size=config.hardware_array_size,
            num_subarrays=config.hardware_num_chips,
            cell_bits=config.adc_bits,
            adc_bits=config.adc_bits,
            dac_bits=config.dac_bits,
        )

        self.weights_map: Dict[str, Tuple[int, int]] = {}
        self.it_values: Dict[str, float] = {}
        self.wcn_values: Dict[str, float] = {}
        self.iem_values: Dict[str, float] = {}

    def detect_pcb(self) -> bool:
        transport = SPITransport(bus=0, device=0, speed_hz=20_000_000)
        if transport.open():
            test_data = transport.read(0x0000, 4)
            valid = len(test_data) == 4 and not all(b == 0 for b in test_data)
            if valid:
                self.transport = transport
                self._initialize_pcb_peripherals()
                self.detected = True
                print(f"Hardware detected via SPI: {transport._spi_path}")
                return True
            transport.close()

        transport = UARTTransport(port="/dev/ttyPS0", baudrate=3_000_000)
        if transport.open():
            self.transport = transport
            self._initialize_pcb_peripherals()
            self.detected = True
            print(f"Hardware detected via UART: {transport.port}")
            return True

        print("Hardware not found, using device-physics simulator")
        self.detected = False
        return False

    def _initialize_pcb_peripherals(self) -> None:
        assert self.transport is not None
        adc_spec = ADCSpec(channels=256, bits=self.cell_bits)
        dac_spec = DACSpec(channels=self.array_size, bits=self.dac_bits)
        array_spec = ArraySpec(rows=self.array_size, cols=self.array_size, num_subarrays=self.num_subarrays)

        self.gpio = GPIOPinController()
        self.gpio.configure_input(17)
        self.gpio.configure_output(27, 0)

        self.adc = ADCBank(self.transport, adc_spec)
        self.dac = DACBank(self.transport, dac_spec)
        self.array_ctrl = MemristorArrayController(self.transport, array_spec, self.adc, self.dac, self.gpio)

        self.sensor_bus = SensorBus(i2c_bus=1)
        self.sensor_bus.open()

    def load_weights(self, layer_name: str, weights, row_offset: int = 0, col_offset: int = 0) -> None:
        if isinstance(weights, torch.Tensor):
            weights_np = weights.cpu().detach().numpy()
        else:
            weights_np = weights

        if self.detected and self.array_ctrl is not None:
            self.array_ctrl.program_weights(layer_name, weights_np, row_offset, col_offset)

        self.simulator.load_weights(layer_name, weights_np, row_offset, col_offset)
        self.weights_map[layer_name] = (row_offset, col_offset)

    def set_parameter(self, layer_name: str, param_name: str, value: float) -> None:
        if param_name == 'IT':
            self.it_values[layer_name] = value
            param_addr = 0x00
        elif param_name == 'WCN':
            self.wcn_values[layer_name] = value
            param_addr = 0x01
        elif param_name == 'IEM':
            self.iem_values[layer_name] = value
            param_addr = 0x02
        else:
            return

        if self.detected and self.array_ctrl is not None:
            quantized = int(np.clip(value, 0, 65535))
            self.array_ctrl.set_parameter(param_addr, quantized)

        self.simulator.set_parameter(layer_name, param_name, value)

    def run_inference(self, layer_name: str, input_data) -> np.ndarray:
        if isinstance(input_data, torch.Tensor):
            input_np = input_data.cpu().numpy()
        else:
            input_np = input_data

        if self.detected and self.array_ctrl is not None and self.adc is not None and self.dac is not None:
            if input_np.ndim >= 2:
                in_flat = input_np.reshape(input_np.shape[0], -1)
                actual_rows = min(in_flat.shape[1], self.array_size)
                for b in range(in_flat.shape[0]):
                    row_input = np.zeros(self.array_size, dtype=np.float32)
                    row_input[:actual_rows] = in_flat[b, :actual_rows]
                    self.dac.write_all_rows(row_input.reshape(1, -1))
                raw_readout = self.adc.read_all_columns(self.array_size)
                out = raw_readout[:min(in_flat.shape[0], raw_readout.shape[0]), :]
                return out
            else:
                row_input = np.zeros(self.array_size, dtype=np.float32)
                actual_vals = min(len(input_np), self.array_size)
                row_input[:actual_vals] = input_np[:actual_vals]
                self.dac.write_all_rows(row_input.reshape(1, -1))
                raw_readout = self.adc.read_all_columns(self.array_size)
                return raw_readout[:1, :]

        return self.simulator.run_inference(layer_name, input_np)

    def get_status(self) -> Dict[str, float]:
        if self.detected and self.sensor_bus is not None:
            temp = self.sensor_bus.read_temperature()
            power = self.sensor_bus.read_power_mw()
            return {
                "temp": temp if not np.isnan(temp) else 0.0,
                "power": power if not np.isnan(power) else 0.0,
            }
        return {"temp": 0.0, "power": 0.0}

    def get_conductance_map(self, layer_name: str) -> Optional[np.ndarray]:
        if self.detected and self.array_ctrl is not None:
            offset = self.weights_map.get(layer_name, (0, 0))
            rows = min(self.array_size, self.simulator.weights.get(layer_name, np.zeros((1, 1))).shape[0])
            cols = min(self.array_size, self.simulator.weights.get(layer_name, np.zeros((1, 1))).shape[1])
            return self.array_ctrl.read_conductance_matrix(rows, cols)
        return None

    def get_mse(self, layer_name: str, input_data, target_output) -> float:
        hw_out = self.run_inference(layer_name, input_data)
        if isinstance(target_output, torch.Tensor):
            target_output = target_output.cpu().numpy()
        if not isinstance(hw_out, np.ndarray):
            return 0.0
        if hw_out.shape != target_output.shape:
            target_output = target_output.reshape(hw_out.shape)
        return float(np.mean((hw_out - target_output) ** 2))

    def shutdown(self) -> None:
        if self.gpio is not None:
            self.gpio.write(27, 0)
            self.gpio.release_all()
        if self.sensor_bus is not None:
            self.sensor_bus.close()
        if self.transport is not None:
            self.transport.close()
        self.detected = False
