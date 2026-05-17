# ROBUST: Robotic On-Board Universal Steganography Training

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**Real-time video steganography system running on a self-selective memristor (SSM) compute-in-memory (CIM) macro, deployed on an edge robotic platform with closed-loop hardware-software co-optimization.**

This repository accompanies the manuscript *"Robotic On-Board Universal Steganography Training (ROBUST)"* and provides the complete software stack that interfaces with custom-fabricated memristor crossbar arrays, an FPGA-based controller, and a Raspberry Pi 5 host processor to perform online adversarial video steganography with adaptive error absorption.

---

## Table of Contents

- [System Overview](#system-overview)
- [Hardware Platform](#hardware-platform)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
  - [Core Pipeline](#core-pipeline)
  - [Hardware Interface](#hardware-interface)
  - [CIM Compiler](#cim-compiler)
  - [Robot Interface](#robot-interface)
  - [Neural Models](#neural-models)
  - [Adaptive Error Absorption Network](#adaptive-error-absorption-network)
  - [Meta-Learning](#meta-learning)
  - [People Tracking](#people-tracking)
  - [Visualization & Logging](#visualization--logging)
- [Configuration](#configuration)
- [Data Flow](#data-flow)
- [Evaluation Metrics](#evaluation-metrics)
- [Hardware-Software Co-Design](#hardware-software-co-design)
- [GPIO Peripheral Interface](#gpio-peripheral-interface)
- [Citation](#citation)
- [License](#license)

---

## System Overview

ROBUST eliminates the architectural fragmentation that plagues conventional hardware-accelerated steganography by co-locating adaptive encoding, error compensation, and non-volatile weight storage within a single CMOS-compatible SSM-based macro. The system achieves three core capabilities:

1. **Online adversarial steganography** — a minimax game between a dense encoder–decoder pair and a critic network forces stego images toward statistical indistinguishability from authentic covers. All three modules execute with memristor-stored weights in a closed hardware-in-the-loop cycle.

2. **Adaptive error compensation** — the Adaptive Error Absorption Network (AEAN) continuously learns systematic error patterns induced by conductance drift, programming noise, and thermal variation, applying per-frame corrections at microsecond-scale latency. When accumulated errors threaten mission-critical performance, the Sensitivity-Aware Programming Protocol (SAPP) triggers nanosecond-scale conductance refinement.

3. **In-situ position prediction** — a lightweight predictor network, mapped entirely to memristor crossbar arrays, generates spatial probability heatmaps that guide selective payload embedding into high-entropy visual regions, maximizing concealment while minimizing detectable artifacts.

The system processes 30 frames per second at 5-bit effective weight precision, achieving software-comparable steganographic fidelity with 1660× energy reduction and 376× runtime acceleration relative to FP32 baselines.

---

## Hardware Platform

| Component | Specification |
|-----------|--------------|
| **Memristor Array** | 512 × 512 SSM crossbar, 5-bit conductance precision per cell |
| **Subarrays** | 8 independent subarrays with per-subarray current sourcing |
| **FPGA Controller** | Xilinx Kintex-7, heterogeneous clock domains (100 / 200 / 1000 MHz) |
| **Host Processor** | Raspberry Pi 5, 8 GB RAM |
| **ADC** | Programmable-gain integrating, 256 levels, 6-bit resolution |
| **DAC** | Multiplying DAC, 2-bit signed voltage-mode output (−1, 0, +1) |
| **Power Delivery** | Multi-tier PCB, programmable rails (2.0 V, <10 mV ripple) |
| **Communication** | Custom synchronous serial protocol over GPIO (3.3 V CMOS) |
| **Sensors** | I²C temperature monitor (0×48), current sensor (0×40), voltage sensor (0×41) |
| **Camera** | USB camera, 192 × 192 RGB at 30 fps |

---

## Repository Structure

```
code/
├── main.py                  # System orchestrator and training loop
├── config.py                # Hyperparameter and hardware configuration
├── models.py                # DenseEncoder, DenseDecoder, BasicCritic, PositionPredictor, StochasticResonanceLayer
├── aean.py                  # Adaptive Error Absorption Network (AEAN) with online Hebbian adaptation
├── compiler.py              # Three-stage CIM compiler (parse → IR → instruction emission)
├── hardware_interface.py    # Physical I/O abstraction for memristor PCB (SPI, UART, I²C, GPIO)
├── robot_interface.py       # Raspberry Pi 5 sensor telemetry and command interface
├── tracker.py               # YOLOv8-based multi-person detection and tracking (ByteTrack)
├── video_stream.py          # Real-time camera capture with frame buffering
├── visualizer.py            # Matplotlib plotting, OpenCV video output, Excel data export
├── meta_learner.py          # Model-Agnostic Meta-Learning (MAML) for multi-task generalization
├── dataset_loader.py        # Multi-domain dataset loader (STL-10, SVHN, CIFAR-10, ImageNet)
├── test_main.py             # Unit and integration test suite
├── PI_GPIO_IN_linux.py      # GPIO input verification script (Pin 17, FPGA → RPi done signal)
├── PI_GPIO_OUT_linux.py     # GPIO output verification script (Pin 27, RPi → FPGA trigger)
├── PI_PULL_UP_DOWN_linux.py # GPIO pull-up/down configuration script (Pin 22, bias control)
└── README.md                # This file
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROBUST System Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Camera ──► PeopleTracker ──► State Vector ──► PositionPredictor    │
│     │            │                  │                 │              │
│     │       RobotInterface     [h_mem, s_t]     Heatmap (128×128)   │
│     │       (RPi sensors)          │                 │              │
│     │                              ▼                 ▼              │
│     └──────────────────────► Payload Encoding ◄── Position Select   │
│                                    │              (γ = 0.55)        │
│                                    ▼                                 │
│                              ┌──────────┐                           │
│                              │ Encoder  │◄── Memristor Weights      │
│                              └────┬─────┘                           │
│                                   │                                  │
│                              Stego Frame                            │
│                              ┌───┴────┐                             │
│                              │        │                             │
│                          Decoder   Critic                            │
│                             │        │                              │
│                     Recovered Msg  Adversarial Score                 │
│                             │                                        │
│                        ┌────┴────┐                                  │
│                        │  AEAN   │◄── Frame Features                │
│                        └────┬────┘                                  │
│                             │                                        │
│                     Corrected Message                                │
│                             │                                        │
│                     ┌───────┴────────┐                              │
│                     │ SAPP Trigger?  │                              │
│                     │ (threshold)    │                              │
│                     └───────┬────────┘                              │
│                             │                                        │
│              ┌──────────────┴──────────────┐                        │
│         No (routine)                  Yes (critical)                 │
│              │                              │                        │
│         AEAN updates              Conductance refresh                │
│         weights online            on memristor array                 │
│                                                                      │
│  All weights synchronized bidirectionally with 512×512 SSM macro     │
│  through CIMCompiler-generated instructions                          │
└─────────────────────────────────────────────────────────────────────┘
```

The system operates as a closed feedback loop:

1. **Acquisition** — Camera frames are captured and processed through YOLOv8-based people detection. Concurrently, the RPi 5 collects environmental telemetry (battery, GPS, temperature, humidity, CPU/memory load).

2. **Prediction** — The in-situ Position Predictor, whose weights reside on the memristor array, generates a 128 × 128 probability heatmap identifying optimal embedding regions for the next frame. This anticipatory mechanism allocates steganographic capacity to high-entropy regions before the frame arrives.

3. **Embedding** — The Encoder performs threshold-based selective steganography (γ = 0.55), modifying pixel values only within predicted high-suitability regions. A hierarchical feature extraction chain with stochastic resonance enhancement ensures payloads concentrate in textured zones where statistical detectability is minimized.

4. **Recovery & Adversarial Evaluation** — The Decoder reconstructs the embedded payload. Simultaneously, the Critic evaluates the stego frame's perceptual authenticity. Both networks operate through memristor-based CIM inference.

5. **Error Correction** — AEAN computes per-frame compensation signals from regional image features, applying them to raw decoder output. Its parameters are updated online via gradient descent on residual error, tracking drifting conductance characteristics.

6. **Precision Refresh** — When AEAN alone cannot maintain target performance (PSNR < 35 dB or accuracy < 0.95 sustained over 5 consecutive frames), SAPP engages to reprogram memristor conductances at the physical level.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.5+
- libgpiod (Linux, for GPIO control)
- Ultralytics YOLOv8

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python numpy pandas matplotlib
pip install ultralytics supervision
pip install gpiod Pillow psutil openpyxl
```

### Hardware Setup

1. Connect the SSM crossbar PCB to the RPi 5 GPIO header (pins 17, 22, 27 as described in [GPIO Peripheral Interface](#gpio-peripheral-interface)).
2. Connect the FPGA JTAG programmer and verify SPI device presence at `/dev/spidev0.0` or UART at `/dev/ttyPS0`.
3. Verify I²C sensor bus at `/dev/i2c-1`.
4. Run the GPIO verification scripts:

```bash
python PI_GPIO_IN_linux.py     # Verify Pin 17 input (FPGA → RPi done)
python PI_GPIO_OUT_linux.py    # Verify Pin 27 output (RPi → FPGA trigger)
python PI_PULL_UP_DOWN_linux.py # Verify Pin 22 pull-up configuration
```

---

## Quick Start

### Real-Time Video Steganography

```bash
python main.py
```

This launches the full ROBUST pipeline:
- Detects and initializes the memristor PCB (or falls back to device-physics simulator)
- Compiles Encoder, Decoder, Critic, Position Predictor, and AEAN models to CIM instructions
- Opens the camera stream (30 fps, 192 × 192)
- Runs adversarial training for the configured number of iterations
- Saves checkpoints, training curves, frame comparisons, and the final stego video

### Meta-Learning Multi-Task Training

```python
from meta_learner import MAMLLearner
from dataset_loader import get_multi_task_datasets

task_datasets = get_multi_task_datasets('./data')  # STL-10, SVHN, CIFAR-10, ImageNet
meta_learner = MAMLLearner(model=encoder, inner_lr=0.01, outer_lr=0.001, inner_steps=5)

for epoch in range(meta_epochs):
    task_batches = sample_tasks(task_datasets)
    meta_loss = meta_learner.outer_loop(task_batches)
```

### Test Suite

```bash
python test_main.py
```

---

## Module Reference

### Core Pipeline

#### `main.py` — `VideoStegoSystem`

The top-level orchestrator that instantiates all subsystems and executes the training loop.

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initializes hardware interface, compiler, robot interface, all four neural models, AEAN, people tracker, VGG-16 perceptual extractor, and three optimizers |
| `train_step(video_clip, importance_level)` | Executes one complete training step over a T-frame clip: prediction → embedding → decoding → AEAN correction → loss computation → gradient update |
| `train_on_video(save_dir)` | Full training pipeline with checkpointing, visualization, and final video generation |
| `generate_final_stego_video_from_stream(...)` | Produces the final combined output video (tracking + stego + diff views) |
| `_calculate_ssim(img1, img2)` | Windowed structural similarity computation (11 × 11 window) |
| `_compute_vertical_correlation_loss(cover, stego)` | Cosine similarity loss between vertical pixel neighbors, preserving natural image statistics |

**Loss Landscape.** The optimization objective combines four terms:

| Loss Term | Weight (λ) | Purpose |
|-----------|------------|---------|
| Reconstruction (BCE) | 2.0 | Message decodability |
| Adversarial (BCE) | 0.1 | Perceptual indistinguishability |
| Perceptual (VGG-16 MSE) | 0.1 | Semantic content preservation |
| Correlation (cosine MSE) | 0.5 | Natural image statistics |

#### `config.py` — `VideoStegoConfig`

Centralized hyperparameter dataclass. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_depth` | 4 | Payload channel depth |
| `hidden_size` | 32 | Encoder/Decoder hidden channels |
| `image_size` | 192 | Input frame resolution |
| `clip_length` | 4 | Temporal clip length |
| `num_iterations` | 10 | Training iterations |
| `hardware_array_size` | 512 | Crossbar array dimension |
| `hardware_num_chips` | 8 | Number of subarrays |
| `adc_bits` | 8 | ADC resolution |
| `dac_bits` | 4 | DAC resolution |
| `psnr_target` | 35.0 | AEAN-SAPP PSNR threshold |
| `acc_target` | 0.95 | AEAN-SAPP accuracy threshold |
| `gamma` | 0.55 | Position selection threshold |

---

### Hardware Interface

#### `hardware_interface.py`

The physical I/O abstraction layer between the host processor and the memristor PCB.

**Class Hierarchy:**

```
MemristorTransport (ABC)
├── SPITransport        # 20 MHz SPI to FPGA/memristor array
└── UARTTransport       # 3 Mbps UART fallback (RPi → FPGA)

SensorBus               # I²C temperature/current/voltage telemetry

ADCBank                 # Column-parallel ADC readout (256 channels)
DACBank                 # Row-parallel DAC input (512 channels)

GPIOPinController       # libgpiod-based GPIO control

MemristorArrayController # Programming & readout orchestrator
HardwareSimulator        # Device-physics simulator (quantization effects)
HardwareInterface        # Top-level facade used by main.py
```

**Detection Protocol.** `HardwareInterface.detect_pcb()` probes physical transport layers in sequence:

1. Attempt SPI bus (`/dev/spidev0.0`): read 4 bytes from address `0x0000`, validate non-zero response.
2. Fall back to UART (`/dev/ttyPS0`): send probe packet, validate echo.
3. If both fail, activate device-physics simulator.

**Weight Programming.** `MemristorArrayController.program_weights()` executes the full physical programming cycle:

1. Assert GPIO 27 (programming trigger)
2. For each row, DAC-quantize weight values and write through DACBank
3. De-assert GPIO 27
4. Poll GPIO 17 (done signal) with 100 µs timeout

**Inference Readout.** `HardwareInterface.run_inference()`:

1. DAC-quantize input vector, write to array rows
2. Trigger crossbar MVM (Ohm's law current summation at column lines)
3. ADC-quantize column output currents (100 ns integration window)
4. Return digitized output matrix

**Telemetry.** `HardwareInterface.get_status()` reads physical sensors via I²C:

- Temperature: address `0x48`, register `0x00` (resolution 1/256 °C)
- Current: address `0x40`, register `0x01` (resolution 12.5 µA)
- Core voltage: address `0x41`, register `0x02` (resolution 1.25 mV)
- Power: computed as `I_core × V_core`

**Simulator.** `HardwareSimulator` models real device quantization without random noise:
- Weight quantization: `cell_bits` conductance levels
- Input quantization: `dac_bits` voltage levels
- Output quantization: `adc_bits` current levels

---

### CIM Compiler

#### `compiler.py` — `CIMCompiler`

Implements the three-stage compilation pipeline described in Methods.

**Stage 1 — Model Parsing (`_stage1_parse`)**

Traverses PyTorch `nn.Module` hierarchies, identifying Conv2d, Linear, and ConvTranspose2d layers. Harvests weight tensors, bias vectors, dimensional specifications, and computes output shapes by tracking data flow through the network (accounting for stride, padding, and dilation in convolutional layers).

**Stage 2 — Intermediate Representation Generation (`_stage2_generate_ir`)**

Constructs structured `LayerIR` descriptors with subarray-aware weight mapping. The mapping algorithm:

```
N_arrays = ⌈total_params / (S_array × S_array)⌉
subarray_id = i mod N_subarrays
row_start   = ⌊i / N_subarrays⌋ × S_array
```

Weights are split with sequential identifiers and boundary indices `[start_idx:end_idx]`. A resource table (`_resource_table`) tracks per-subarray utilization for thermal and electrical load balancing. Routing priority is assigned based on layer type and parameter count.

**Stage 3 — Instruction Emission (`_stage3_emit_instructions`)**

Generates human-readable operation codes with full resource binding:

```
COMPILE array_size=512 subarrays=8
TIMESTAMP 2026-05-17T14:30:00
LAYER name=encoder_conv1 type=Conv2d
  SHAPE weight=(32,3,3,3) input=(1,3,192,192) output=(1,32,192,192)
  PARAMS count=864 splits=1 priority=100
  MAP split=0 subarray=0 row=[0:512] col=[0:512] range=[0:864]
  CONFIG IT=100.0 WCN=1.0 IEM=0.0
```

**Runtime Updates.** `update_parameter(layer_name, param_name, value)` enables dynamic IT/WCN/IEM tuning across iterations without recompilation, supporting the progressive integration time modulation described in the paper.

**Resource Utilization.** `get_resource_utilization()` returns per-subarray fill percentages, enabling load-balancing decisions at deployment time.

---

### Robot Interface

#### `robot_interface.py` — `RobotInterface`

Bidirectional serial communication with the Raspberry Pi 5 sensor suite.

**Connection.** Establishes UART link at `/dev/ttyAMA0` (115200 baud). Sends `AT+STATUS` handshake; validates response before marking connected state.

**State Vector.** `get_system_state_vector()` returns a 7-dimensional normalized tensor:

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Battery level | ÷ 100 |
| 1 | Latitude (GPS) | (lat − 120) / 10 |
| 2 | Longitude (GPS) | (lon − 30) / 10 |
| 3 | Temperature | / 50 |
| 4 | Humidity | / 100 |
| 5 | CPU usage | / 100 |
| 6 | Memory usage | / 100 |

**Protocol.** Frame-based messages with comma-separated fields: `battery,lat,lon,temp,humidity,cpu,memory`. A 100 ms read throttle prevents bus saturation. When disconnected, the system falls back to cached state with zero drift (no random walk).

**Commands.** `send_command()` transmits arbitrary string commands over serial, returning the response line. Used for robot actuation triggers synchronized with steganographic events.

---

### Neural Models

#### `models.py`

**`DenseEncoder`** — Four-stage hierarchical encoder with stochastic resonance enhancement:

1. `conv1`: 3 → 32 channel expansion (LeakyReLU, 3 × 3 conv)
2. `sr_layer`: Stochastic resonance modulation using message-driven noise injection and edge-sensitive gating
3. `conv2–conv4`: Dense skip connections concatenating all prior feature maps with the payload
4. `mask_conv`: Learned spatial attention mask (1-channel Sigmoid output)
5. Output: `stego = cover + delta_raw × mask × 0.01` (clamped to [−1, 1])

**`DenseDecoder`** — Symmetric four-stage decoder with SR-enhanced feature extraction and spatial attention:

1. `conv1`: 3 → 32 channel expansion
2. `sr_enhance`: SR-based feature refinement (ReLU)
3. `sr_attention`: Spatial attention map (1-channel Sigmoid)
4. `conv2–conv4`: Hierarchical decoding with skip connections
5. Output: `data_depth` channels (logits for BCE loss)

**`BasicCritic`** — Lightweight global discriminator:

1. Two strided conv layers (4 × 4, stride 2) with LeakyReLU
2. Adaptive average pooling to 1 × 1
3. Linear projection to scalar score

**`PositionPredictor`** — State-to-heatmap regression:

1. `fc1`: `state_dim` → 32 (ReLU)
2. `fc2`: 32 → `output_size²` (Sigmoid)
3. Reshape to `(1, output_size, output_size)` probability heatmap

**`StochasticResonanceLayer`** — Physics-inspired noise enhancement:

- Modulates Gaussian noise amplitude with message content
- Computes local gradient magnitude and variance as edge-saliency indicators
- Produces a self-gated output that amplifies faint features through constructive noise interference

---

### Adaptive Error Absorption Network

#### `aean.py` — `AdaptiveErrorAbsorptionNetwork`

A lightweight online learning module that compensates for hardware-induced errors without interrupting the main inference pipeline.

**Architecture.** A single linear layer `W ∈ ℝ^{output_dim × input_dim}` (no bias, zero-initialized weights) that produces additive correction signals:

```
y_corrected = y_main + y_aux
y_aux = x · W^T
```

**Online Update Rule.** Each frame, AEAN receives the residual `δ` between target and current output, then executes:

```
ΔW = −η · δ ⊗ x          (outer product gradient)
W ← W + ΔW
```

Primary (memristor-mapped) weights receive synchronized updates every `K` steps (configurable `primary_update_ratio`, default 10), amortizing the cost of physical conductance programming.

**Correction Signal.** `get_correction_signal(psnr, acc)` computes a composite delta:

```
δ_psnr = PSNR_target − PSNR_current
δ_acc  = Acc_target  − Acc_current
Δ      = 0.3 · δ_psnr + 0.7 · δ_acc
```

This signal drives the predictor update, completing the AEAN-SAPP feedback loop.

**Convergence Detection.** `get_convergence_status(window_size, threshold)` monitors weight change magnitude over a rolling window, signaling when AEAN has absorbed systematic errors and SAPP engagement is no longer required.

---

### Meta-Learning

#### `meta_learner.py` — `MAMLLearner`

Implements Model-Agnostic Meta-Learning (MAML) for rapid cross-domain adaptation.

**Inner Loop.** For each task, performs `K` gradient steps on cloned fast weights:

```
θ'_0 = θ
θ'_{k+1} = θ'_k − α · ∇_{θ'_k} L_task(θ'_k)
```

**Outer Loop.** Aggregates meta-loss across all tasks and updates the meta-parameters:

```
θ ← θ − β · ∇_θ Σ_task L_task(θ'_K)
```

This enables the encoder to adapt to new image domains (STL-10, SVHN, CIFAR-10, ImageNet) within 5 gradient steps while maintaining steganographic fidelity.

#### `dataset_loader.py`

Provides multi-domain dataset loading with unified 192 × 192 normalization. Supports four benchmarking domains with distinct visual statistics: natural scenes (STL-10), street-view digits (SVHN), object categories (CIFAR-10), and large-scale natural images (ImageNet).

---

### People Tracking

#### `tracker.py` — `PeopleTracker`

Integrates YOLOv8-nano detection with ByteTrack multi-object tracking.

**Detection.** `detect_and_track_people(frame)` converts normalized tensors to uint8, runs YOLOv8 (class 0 = person), and updates ByteTrack trajectories. Returns per-person bounding boxes, center coordinates, and persistent track IDs.

**Secret Extraction.** `extract_secret_from_overlay(people_info, frame)` encodes:

- Person IDs and bounding boxes
- Robot telemetry (battery, GPS, temperature, humidity, CPU, memory)
- Timestamp components

into a flat float32 vector for payload encoding.

**State Vector.** `compute_state_vector(frame_idx, total_frames, importance_level)` produces the 4-dimensional input to the Position Predictor:

| Dimension | Symbol | Meaning |
|-----------|--------|---------|
| 0 | `e_prog` | Normalized CPU usage (execution progress) |
| 1 | `n_fail` | Discretized battery depletion (failure risk) |
| 2 | `n_update` | Frame progress ratio |
| 3 | `s_t` | Importance level (task priority) |

**Visualization.** `draw_overlay()` renders bounding boxes, track IDs, timestamp, battery icon, GPS coordinates, temperature, humidity, CPU, and memory on the frame. All positions are scaled to match the original image dimensions.

---

### Visualization & Logging

#### `visualizer.py`

**`VideoStegoVisualizer`** — Multi-format output manager:

| Output Type | Format | Content |
|-------------|--------|---------|
| Training curves | PNG (200 dpi) | PSNR (current + previous), accuracy, smoothed trends, AEAN weight norms |
| Frame comparisons | PNG (150 dpi) | Tracking vs. stego side-by-side with per-frame PSNR/SSIM |
| Video comparisons | PNG (150 dpi) | 4-row grid: original, tracking, stego, enhanced difference (×10) |
| Generated video | MP4 (combined) | 2 × 2 grid: tracking + stego above, 1× diff + 1000× diff below |
| Excel workbook | XLSX | Multi-sheet: training curves, frame metrics, video metrics with averages |

**`ExcelDataManager`** — Programmatic Excel generation with styled headers (blue fill, white bold text), auto-adjusted column widths, and per-sheet metric organization.

---

## Configuration

All hyperparameters are consolidated in `config.py` as a `VideoStegoConfig` dataclass. Modify before instantiation:

```python
config = VideoStegoConfig(
    data_depth=4,           # Payload embedding depth
    hidden_size=32,         # Encoder/Decoder hidden channels
    image_size=192,         # Frame resolution
    clip_length=4,          # Temporal window
    batch_size=1,           # Single-frame streaming
    device='cpu',           # 'cpu' or 'cuda'
    num_iterations=10,      # Training iterations
    lambda_recon=2.0,       # Reconstruction loss weight
    lambda_adv=0.1,         # Adversarial loss weight
    lambda_percep=0.1,      # Perceptual loss weight
    lambda_corr=0.5,        # Correlation loss weight
    hardware_array_size=512, # Crossbar dimension
    hardware_num_chips=8,   # Subarray count
    adc_bits=8,             # ADC resolution
    dac_bits=4,             # DAC resolution
    psnr_target=35.0,       # AEAN-SAPP PSNR threshold
    acc_target=0.95,        # AEAN-SAPP accuracy threshold
    gamma=0.55,             # Position selection threshold
)
```

---

## Data Flow

### Frame-Level Cycle

```
                     ┌──────────────────────┐
  Camera             │   People Tracker     │      Robot Interface
  (USB, 30fps)       │   (YOLOv8+ByteTrack) │      (UART, RPi 5)
     │                │          │            │          │
     ▼                │    track_ids, bbox   │    battery, GPS,
  Frame Tensor ───────┤    centers           │    temp, humidity,
  (1,3,192,192)       │          │            │    CPU, memory
                      └──────────┼────────────┘          │
                                 │                       │
                           Secret Payload ◄──────────────┘
                           (float32 vector)
                                 │
                     State Vector (4-d)
                           │
                     Position Predictor
                     (memristor CIM)
                           │
                     Heatmap (128×128)
                           │
                     γ-threshold selection
                           │
                     Encoder (memristor CIM)
                           │
                     Stego Frame
                        ┌───┴───┐
                   Decoder    Critic
                   (CIM)      (CIM)
                      │         │
                 Raw Msg    Adv Score
                      │         │
                 ┌────┴────┐    │
                 │  AEAN   │    │
                 │ (4×4 CIM)│   │
                 └────┬────┘    │
                      │         │
              Corrected Msg     │
                      │         │
              PSNR, Acc, SSIM ◄─┘
                      │
              Composite Δ = 0.3·δ_PSNR + 0.7·δ_Acc
                      │
         ┌────────────┴────────────┐
         │                         │
    Δ < threshold            Δ ≥ threshold
         │                         │
    AEAN weight update       SAPP conductance
    (software, online)       refresh (hardware)
```

### Inter-Process Communication

| Path | Protocol | Speed | Purpose |
|------|----------|-------|---------|
| RPi 5 ↔ FPGA | Custom GPIO serial | 3.3 V CMOS | Weight loading, instruction dispatch |
| FPGA ↔ Memristor Array | SPI | 20 MHz | Row/column addressing, conductance read/write |
| Sensor Bus ↔ RPi 5 | I²C | 400 kHz | Temperature, current, voltage telemetry |
| RPi 5 ↔ Camera | USB 2.0 | 480 Mbps | Frame capture |
| RPi 5 ↔ Robot Sensors | UART | 115200 baud | GPS, IMU, battery, environmental data |

---

## Evaluation Metrics

| Metric | Definition | Range | Target |
|--------|-----------|-------|--------|
| **PSNR** | `10 · log₁₀(MAX_I² / MSE)` | [0, ∞) dB | > 35 dB |
| **SSIM** | Luminance × Contrast × Structure | [0, 1] | > 0.90 |
| **Decoding Accuracy** | Bitwise match rate of recovered payload | [0, 1] | > 0.95 |
| **Wasserstein Distance** | Earth mover's distance between cover and stego score distributions | [0, ∞) | → 0 |
| **Runtime** | Wall-clock time per frame (acquisition → corrected output) | ms | < 33 ms (30 fps) |
| **Energy per Frame** | Integrated power × time across all hardware domains | mJ | minimized |
| **Memory Cost** | Total storage for weights + activations + instructions | KB | minimized |

---

## Hardware-Software Co-Design

### Weight Synchronization Protocol

All trainable network weights are committed to the memristor array after each training step, establishing the array as the authoritative weight repository. The synchronization sequence:

1. **Software → Hardware** (after training step): Updated weights from Encoder, Decoder, Critic, and Predictor are programmed into their allocated subarray regions via `MemristorArrayController.program_weights()`. GPIO 27 is toggled with a 10 µs programming pulse.

2. **Hardware → Software** (before inference): The latest conductance states are read back through `MemristorArrayController.read_conductance_matrix()`, ensuring software state mirrors physical conductance.

### AEAN-SAPP Feedback Loop

The dual-layer correction architecture decouples routine adaptation from hardware maintenance:

- **Tier 1 (AEAN):** Software-level parameter updates execute at frame rate (30 Hz). AEAN absorbs gradual conductance drift, temperature-induced resistance changes, and systematic programming errors. Its linear structure (4 × 4 weight matrix) keeps computational overhead negligible (<1% of total inference time).

- **Tier 2 (SAPP):** Hardware-level conductance refresh triggers only when AEAN correction residuals exceed threshold (PSNR < 35 dB or accuracy < 0.95 sustained for 5 consecutive frames). SAPP applies 5-bit precision programming pulses with per-cell verify cycles, restoring physical conductance states to their target values at nanosecond-scale latency.

### Precision-Energy Trade-off

The 5-bit weight precision configuration emerges as the optimal operating point:

| Precision | Decoding Accuracy | Endurance (cycles) | Relative Energy |
|-----------|-------------------|---------------------|-----------------|
| 1-bit (×5 combined) | 0.961 | >10⁸ | 5.0× |
| 3-bit (×2 combined) | 0.958 | >10⁸ | 2.0× |
| **5-bit (native)** | **0.955** | **~10⁷** | **1.0×** |
| 6-bit (native) | 0.932 | ~10⁵ | 1.0× |
| 7-bit (native) | 0.901 | ~10⁴ | 1.0× |

---

## GPIO Peripheral Interface

The RPi 5 communicates with the FPGA/memristor PCB through three dedicated GPIO lines:

```
Raspberry Pi 5                    FPGA / Memristor PCB
┌──────────────┐                 ┌─────────────────────┐
│              │                 │                     │
│  GPIO 17 ◄───┼─────────────────┼─── Done (output)    │
│  (input)     │                 │    High = ready      │
│              │                 │                     │
│  GPIO 27 ────┼─────────────────┼──► Trigger (input)  │
│  (output)    │                 │    Rising edge =     │
│              │                 │    program pulse     │
│  GPIO 22 ────┼─────────────────┼─── Bias control     │
│  (pull-up)   │                 │    Pull-up enable    │
│              │                 │                     │
│  3.3V ───────┼─────────────────┼─── VDDIO            │
│  GND ────────┼─────────────────┼─── GND              │
└──────────────┘                 └─────────────────────┘
```

**Pin Assignments:**

| GPIO Pin | Direction | Signal Name | Function |
|----------|-----------|-------------|----------|
| 17 | Input (RPi reads) | `DONE` | FPGA asserts HIGH when programming cycle completes. RPi polls this pin after triggering a weight write. |
| 27 | Output (RPi drives) | `TRIGGER` | RPi pulses HIGH for 10 µs to initiate a memristor programming cycle. DAC data must be stable on the bus before assertion. |
| 22 | Input with pull-up | `BIAS_CTRL` | Controls the FPGA's internal biasing mode. Pull-up enabled = auto-bias calibration active. |

**Verification Scripts:**

- `PI_GPIO_IN_linux.py` — Continuously reads GPIO 17, printing its value. Use to verify the FPGA is driving the done signal correctly.
- `PI_GPIO_OUT_linux.py` — Toggles GPIO 17 (configured as output) with 1-second period. Use to verify RPi output drive capability.
- `PI_PULL_UP_DOWN_linux.py` — Reads GPIO 17 with pull-up bias enabled, printing values continuously. Use to verify internal pull-up resistor functionality.

---

## Citation

If you use this code in your research, please cite the accompanying manuscript:

```bibtex
@article{zhang2026robust,
  title={Robotic On-Board Universal Steganography Training (ROBUST)},
  author={Zhang, Guobin and Fan, X. and Wang, Z. and Li, P. and Zhou, Y.
          and Sun, D. and Ren, K. and Li, Y. and Yu, B. and Wan, Q.
          and Gao, D. and Zhang, Y.},
  journal={Nature Communications},
  volume={16},
  pages={5759},
  year={2025},
  doi={10.1038/s41467-025-XXXXX}
}
```

---

## License



---

*For questions, bug reports, or collaboration inquiries, please open an issue or contact the corresponding author.*
