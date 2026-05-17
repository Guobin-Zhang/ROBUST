import random
import datetime
import serial
import time
import torch

class RobotInterface:
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.serial_port = None
        self._simulate = True
        self.last_read_time = 0
        self.cached_state = {
            'battery': 87,
            'lat': 120.38,
            'lon': 30.27,
            'temp': 24.5,
            'humidity': 65,
            'cpu_usage': 0,
            'memory_usage': 0,
            'uptime': 0,
            'time': datetime.datetime.now()
        }
    
    def connect(self):
        try:
            self.serial_port = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
            time.sleep(2)
            self.serial_port.write(b'AT+STATUS\r\n')
            response = self.serial_port.readline()
            if response:
                self.connected = True
                print("Connected to Raspberry Pi")
            else:
                self.connected = False
                print("Raspberry Pi not responding, using simulation mode")
        except Exception as e:
            self.connected = False
            print(f"Raspberry Pi connection failed: {e}, using simulation mode")
        return self.connected
    
    def get_state(self):
        current_time = time.time()
        if not self.connected and self._simulate:
            self.cached_state['battery'] = max(60, self.cached_state['battery'] - random.uniform(0, 0.05))
            self.cached_state['lat'] += random.uniform(-0.0005, 0.0005)
            self.cached_state['lon'] += random.uniform(-0.0005, 0.0005)
            self.cached_state['temp'] += random.uniform(-0.05, 0.05)
            self.cached_state['humidity'] = max(40, min(80, self.cached_state['humidity'] + random.uniform(-0.5, 0.5)))
            self.cached_state['cpu_usage'] = random.uniform(5, 30)
            self.cached_state['memory_usage'] = random.uniform(15, 45)
            self.cached_state['uptime'] += 1
            self.cached_state['time'] = datetime.datetime.now()
            return self.cached_state.copy()
        
        if current_time - self.last_read_time < 0.1:
            return self.cached_state.copy()
        
        self.last_read_time = current_time
        try:
            if not self.connected:
                self.connect()
            
            if self.connected:
                self.serial_port.reset_input_buffer()
                self.serial_port.write(b'STATUS\n')
                line = self.serial_port.readline().decode().strip()
                if line and len(line.split(',')) >= 7:
                    parts = line.split(',')
                    self.cached_state['battery'] = int(parts[0])
                    self.cached_state['lat'] = float(parts[1])
                    self.cached_state['lon'] = float(parts[2])
                    self.cached_state['temp'] = float(parts[3])
                    self.cached_state['humidity'] = int(parts[4])
                    self.cached_state['cpu_usage'] = float(parts[5])
                    self.cached_state['memory_usage'] = float(parts[6])
        except Exception as e:
            print(f"Error reading from Raspberry Pi: {e}")
            self.connected = False
        
        self.cached_state['time'] = datetime.datetime.now()
        self.cached_state['uptime'] += 1
        return self.cached_state.copy()
    
    def get_system_state_vector(self):
        state = self.get_state()
        return torch.FloatTensor([
            state['battery'] / 100.0,
            (state['lat'] - 120) / 10,
            (state['lon'] - 30) / 10,
            state['temp'] / 50,
            state['humidity'] / 100,
            state['cpu_usage'] / 100,
            state['memory_usage'] / 100
        ])
    
    def send_command(self, command):
        if not self.connected:
            return False
        try:
            self.serial_port.write(f"{command}\n".encode())
            response = self.serial_port.readline().decode().strip()
            return response
        except:
            return False
    
    def disconnect(self):
        if self.serial_port:
            self.serial_port.close()
            self.connected = False