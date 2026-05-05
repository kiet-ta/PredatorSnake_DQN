# PredatorSnake_DQN

[![Fast with Rust](https://img.shields.io/badge/Performance-Rust-orange.svg)](https://www.rust-lang.org/)
[![RL with SB3](https://img.shields.io/badge/RL-Stable--Baselines3-blue.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance custom Snake reinforcement learning environment with a **Rust Engine (Bevy ECS)** and **Stable-Baselines3 (DQN)** control plane.

---

## 🏗️ Architecture

- **Data Plane (Rust):** High-performance game logic, collision detection, and reward calculation using **Bevy ECS**.
- **Control Plane (Python):** Deep Q-Network (DQN) training and evaluation using **Stable-Baselines3**.
- **Bridge:** Zero-copy observation transfer via **PyO3** and **rust-numpy**, ensuring minimal overhead between the engine and the AI.

---

## 🚀 Quick Start

### 1. Requirements

- Rust (latest stable)
- Python 3.10+
- [Maturin](https://github.com/PyO3/maturin)

### 2. Build and Install

```bash
# Install maturin if not already present
pip install maturin

# Compile Rust core and install into your python environment
maturin develop --release
```

---

## 🧠 Training Guide

### A. Training from Scratch

Run the following command to start a new training session:

```bash
python examples/train_sb3.py --train --n-envs 8 --total-timesteps 5000000 --exploration-fraction 0.2 --learning-rate 0.0001
```

**Argument Breakdown:**

- `--n-envs 8`: Spawns 8 parallel workers utilizing multiple CPU cores.
- `--total-timesteps 5000000`: 5M steps for complex strategic emergence.
- `--exploration-fraction 0.2`: 20% of time spent on random exploration.

### B. Resuming Training

```bash
python examples/train_sb3.py --resume --model-path ./models/best_model/best_model.zip --total-timesteps 1000000
```

---

## 📊 Evaluation & Monitoring

### 1. Watch the AI Play (Rendering)

#### Recommended (stable): Python renderer

```bash
python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip --play-renderer python
```

You should see startup logs similar to:

- `[UI Renderer] Mode: python`
- `[UI Renderer] Active backend: QtAgg`

#### Optional (experimental): Bevy renderer

```bash
python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip --play-renderer bevy
```

> Bevy window mode may freeze under manual ticking (`app.update()` from Python) due to Winit event-loop constraints. Use the Python renderer for reliable review.

#### If no UI appears in Python renderer

1. Check backend:

```bash
python -c "import matplotlib; print(matplotlib.get_backend())"
```

2. Install an interactive backend (recommended: Qt):

```bash
pip install PyQt6
```

3. Force Qt backend and run play:

```bash
MPLBACKEND=QtAgg python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip --play-renderer python
```

### 2. Monitor with TensorBoard

```bash
tensorboard --logdir ./tensorboard/
```

---

## 🛠️ Technical Details

### Observation Space

The model receives a **4-channel one-hot encoded tensor** `[4, H, W]`:

- **Channel 0:** Empty space.
- **Channel 1:** Snake head.
- **Channel 2:** Snake body.
- **Channel 3:** Food.

🖼️ **[One-Hot Encoding Visualizer](https://drive.google.com/file/d/1pZwmL65KieHbzkRz3GIVKz13DoQpGB3j/view?usp=drive_link)**

### Reward Signal (configured in Rust)

- **+10.0**: Eating food (Encourages growth).
- **-10.0**: Collision with wall or body (Death penalty).
- **-0.01**: Step penalty (Encourages finding food efficiently).

### Action Space

`Discrete(3)` relative controls:

- `0`: Straight | `1`: Turn Left | `2`: Turn Right

---

## 📈 Visual Results & Analysis

- 🏆 **Evaluation Metrics:** [View Evaluation Chart](https://drive.google.com/file/d/1C1W-OEKyRdZaZwsso4RaIJQAAi9Pb_Sc/view?usp=drive_link)
- 🔄 **Rollout Statistics:** [View Rollout Analysis](https://drive.google.com/file/d/1Rl4wuWV1ZDKOg_8SyjlIM4oH0xm6Fx3c/view?usp=drive_link)
- ⏱️ **Training Time/Performance:** [View Time Training](https://drive.google.com/file/d/12HenuA8LBgAJv0C7ATbW0LeX1Vx0nu3C/view?usp=drive_link)

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
