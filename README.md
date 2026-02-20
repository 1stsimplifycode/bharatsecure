# BharatSecure Touchless HCI

> A Zero-Trust, Privacy-Preserving Gesture-Based Media Control System on NVIDIA Jetson Nano

[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-green)](https://developer.nvidia.com/embedded/jetson-nano)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Cost](https://img.shields.io/badge/Infra%20Cost-~%240-brightgreen)]()

---

## Overview

BharatSecure Touchless HCI is a secure, edge-deployed gesture recognition system that replaces
traditional fingerprint biometrics with dynamic behavioral gestures. Built for NVIDIA Jetson Nano
with a **polyglot architecture** (Python · C++ · Shell · YAML · JavaScript), it achieves:

- 🎯 **90–93% gesture accuracy**
- ⚡ **~22 FPS** real-time inference
- 🔒 **Zero raw biometric storage**
- 💰 **~$0 infrastructure cost** (all open-source, edge-only)

---

## Polyglot Architecture

```
bharatsecure/
├── Python        → Core AI/ML, Security layers, Federated Learning, Dashboard
├── C++           → Low-level camera optimisation for Jetson Nano GPU
├── Shell (Bash)  → Jetson setup, deployment automation
├── YAML          → Configuration management
├── JavaScript    → Real-time security dashboard (Vanilla JS, zero dependencies)
└── Makefile      → Unified build & run commands
```

---

## System Architecture

```
Camera Module
     ↓
Hand Landmark Extraction (MediaPipe)
     ↓
Feature Vector Generator (63-Dimensional)
     ↓
Security Validation Layer ──── Liveness | Anomaly | Rate Limit | Hash Check
     ↓
AI Inference Engine (Lightweight MLP)
     ↓
Confidence & Zero-Trust Filter (> 0.90)
     ↓
Secure Command Execution
     ↓
Security Dashboard & Logging
```

---

## Supported Gestures

| Gesture | Command | Description |
|---------|---------|-------------|
| ✋ Open Palm | Stop / Pause | All fingers extended |
| ☝️ One Finger Up | Play | Index finger only |
| 👆 Two Fingers Up | Volume Up | Index + Middle |
| ✌️ Peace Down | Volume Down | Index + Middle downward |
| ✊ Fist | Mute | All fingers curled |

---

## Quick Start

### Prerequisites (Jetson Nano)
```bash
# JetPack 4.6.x recommended
# Python 3.8+, OpenCV 4.x pre-installed in JetPack
```

### Installation
```bash
git clone https://github.com/your-org/bharatsecure-touchless-hci
cd bharatsecure-touchless-hci
chmod +x jetson/setup_jetson.sh
./jetson/setup_jetson.sh       # One-time Jetson setup
make install                   # Python dependencies
```

### Run
```bash
make run                       # Start full system
make dashboard                 # Open security dashboard (port 5000)
make train                     # (Re)train gesture model
make test                      # Run all tests
make federated-server          # Start federated aggregation server
```

### C++ Camera Optimizer (Optional, Jetson GPU)
```bash
make cpp-build                 # Build C++ module
make run-cpp                   # Run with C++ camera backend
```

---

## Security Mechanisms

| Mechanism | Implementation | Cost |
|-----------|---------------|------|
| Liveness Detection | Frame-diff temporal motion | $0 |
| Anomaly Detection | Z-score on landmarks | $0 |
| Zero-Trust Execution | Confidence > 0.90 + multi-gate | $0 |
| Model Integrity | SHA-256 hash verification | $0 |
| Privacy | No raw frame / biometric storage | $0 |
| Federated Learning | Differential privacy + weight aggregation | $0 |
| Transport Security | TLS 1.3 (self-signed for dev) | $0 |

---

## Federated Learning

Each Jetson Nano device trains locally and shares **only differentially-noised weight updates**
with the secure aggregation server. Raw landmark data never leaves the device.

```
Jetson Device 1 → Local Train → Weight + DP Noise ──┐
Jetson Device 2 → Local Train → Weight + DP Noise ──┤ → Secure Server → Global Model
Jetson Device 3 → Local Train → Weight + DP Noise ──┘
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| FPS | 20–25 |
| Gesture Accuracy | 90–93% |
| Model Size | < 5 MB |
| Latency | < 150 ms |
| RAM Usage | ~1.2 GB (Jetson 4GB) |
| Power | 5–10W |

---

## Threat Coverage

| Attack | Mitigation | Status |
|--------|-----------|--------|
| Static Image Replay | Temporal motion delta | ✅ Blocked |
| Video Replay | Liveness Δ < threshold check | ✅ Blocked |
| Adversarial Gloves | Z-score landmark anomaly | ✅ Detected |
| Model Tampering | SHA-256 runtime verification | ✅ Detected |
| MITM | TLS encrypted channel | ✅ Encrypted |
| Frame Flooding / DoS | Rate limiter + frame monitor | ✅ Throttled |
| Biometric Leakage | No storage, ephemeral vectors | ✅ N/A |

---

## Cost Analysis

| Component | Tool | Cost |
|-----------|------|------|
| Hand detection | MediaPipe | Free |
| AI inference | TensorFlow Lite | Free |
| Camera capture | OpenCV + CSI driver | Free |
| Dashboard | Flask + Vanilla JS | Free |
| Logging | SQLite | Free |
| Federated server | Python socket + Flask | Free |
| Hardware | Jetson Nano (one-time) | ~$99 |
| **Recurring infra** | **—** | **$0/month** |

---

## Institution
PES University · Team Lead: Adishree Gupta · Mentor: Dr. Swetha P
