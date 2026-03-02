# Multimodal Prediction System for Autonomous Navigation

**Master's Thesis — Centro de Investigación en Computación**  
Gustavo Mandujano Rojas | Maestría en Ciencias en Ingeniería de Cómputo

---

## Overview

This repository contains the implementation and evaluation framework for a multimodal deep learning system designed for autonomous navigation of VTOL (Vertical Take-Off and Landing) drones. The system processes temporal sequences of grayscale images (128×128 px) combined with IMU sensor data to predict flight control commands in real time.

A key contribution of this work is the comparative analysis of multiple neural network architectures under consistent training and evaluation conditions, along with their deployment on embedded hardware.

---

## Architectures Evaluated

| Model | Type | Description |
|---|---|---|
| **SA-ConvLSTM** | Recurrent + Attention | Self-attention spatial-temporal model adapted for flight control |
| **ConvLSTM** | Recurrent | Convolutional LSTM for sequential image processing |
| **PilotNet** | CNN | End-to-end learning baseline inspired by NVIDIA's PilotNet |
| **MobileNetV3** | CNN | Lightweight architecture optimized for embedded inference |
| **ResNet** | CNN | Residual network for feature extraction |

---

## System Architecture

```
Input
├── Image sequence     (T × 128 × 128 grayscale frames)
└── IMU data           (accelerometer + gyroscope readings)
         │
         ▼
    Feature Extraction & Temporal Modeling
         │
         ▼
    Control Command Prediction
    (roll, pitch, yaw, throttle)
```

---

## Evaluation Metrics

Models are benchmarked on the following metrics:

- **MAE** — Mean Absolute Error on predicted control commands
- **RMSE** — Root Mean Square Error
- **Latency (ms)** — Inference time per frame
- **FPS** — Frames processed per second on target hardware

---

## Hardware Deployment

Inference benchmarks were conducted on:

- 🖥️ **Raspberry Pi 5** — Target embedded platform for onboard deployment

---

## Results

> 📌 Results and comparative analysis will be published upon completion of the formal thesis evaluation process.  
> This section will be updated with full metrics, tables, and visualizations.

---

## Repository Structure

```
.
├── models/              # Model architectures implementations
├── data/                # Data preprocessing and loading pipelines
├── training/            # Training scripts and configuration
├── evaluation/          # Benchmarking and metrics scripts
├── deployment/          # Embedded platform optimization (ONNX export)
├── results/             # Figures and metric outputs (post-publication)
└── requirements.txt     # Python dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/GusRojas/Evaluation-and-Comparison-of-Autonomous-Navigation-Models.git
cd Evaluation-and-Comparison-of-Autonomous-Navigation-Models

# Install dependencies
pip install -r requirements.txt
```

---
<!--
## Usage

```bash
# Train a model
python training/train.py --model sa_convlstm --config configs/default.yaml

# Evaluate a model
python evaluation/evaluate.py --model sa_convlstm --checkpoint path/to/checkpoint.pth

# Export to ONNX for embedded deployment
python deployment/export_onnx.py --model sa_convlstm --checkpoint path/to/checkpoint.pth
```
-->
> ⚠️ Note: Dataset is not included in this repository. Instructions for data preparation are available in `data/README.md`.

---

## Technologies

`Python` `PyTorch` `ONNX` `ROS2` `OpenCV` `NumPy` `Raspberry Pi 5`

---

## Author

**Gustavo Mandujano Rojas**  
📧 [gusm.rojas@gmail.com](mailto:gusm.rojas@gmail.com)  
💼 [linkedin.com/in/gustavo-mandujano-rojas](https://www.linkedin.com/in/gustavo-mandujano-rojas/)

---

## Citation

> This work is part of an ongoing master's thesis. Citation details will be added upon formal publication.
