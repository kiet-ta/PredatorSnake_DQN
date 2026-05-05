# PredatorSnake_DQN

[![Fast with Rust](https://img.shields.io/badge/Performance-Rust-orange.svg)](https://www.rust-lang.org/)
[![RL with SB3](https://img.shields.io/badge/RL-Stable--Baselines3-blue.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance custom Snake reinforcement learning environment with a **Rust Engine (Bevy ECS)** and **Stable-Baselines3 (DQN)** control plane.

---

## 🏗️ Architecture

- **Data Plane (Rust):** High-performance game logic, collision detection, and reward calculation using Bevy ECS.
- **Control Plane (Python):** Deep Q-Network (DQN) training and evaluation.
- **Bridge:** Zero-copy observation transfer via PyO3 and rust-numpy.

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

- `--train`: Activates training mode.
- `--n-envs 8`: Spawns 8 parallel environment workers. This utilizes multiple CPU cores to collect experience 8x faster.
- `--total-timesteps 5000000`: The total number of steps the agent will take. 5M is usually enough for complex strategic emergence.
- `--exploration-fraction 0.2`: The agent will spend 20% of the total time (1M steps) primarily exploring (taking random actions) to discover rewards.
- `--learning-rate 0.0001`: The step size for neural network weight updates. A small value ensures stable convergence.

### B. Resuming Training

To continue training from a saved checkpoint:

```bash
python examples/train_sb3.py --resume --model-path ./models/best_model/best_model.zip --total-timesteps 1000000
```

_Note: Resuming automatically adjusts exploration parameters to focus on fine-tuning._

---

## 📊 Evaluation & Monitoring

### 1. Watch the AI Play (Rendering)

To visualize the agent's performance after training:

```bash
python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip
```

### 2. Monitor with TensorBoard

Track metrics like `mean_reward`, `loss`, and `episode_length` in real-time:

```bash
tensorboard --logdir ./tensorboard/
```

Then visit: `http://localhost:6006` in your browser.

---

## 🛠️ Technical Details

### Observation Space

The model receives a 4-channel one-hot encoded tensor `[4, H, W]`:

- **Channel 0:** Empty space.
- **Channel 1:** Snake head.
- **Channel 2:** Snake body.
- **Channel 3:** Food.

### Reward Signal

- **+1.0**: Eating food.
- **-1.0**: Collision (Death).
- **-0.01**: Step penalty (Encourages efficiency).

### Action Space

`Discrete(3)` relative controls:

- `0`: Straight
- `1`: Turn Left
- `2`: Turn Right
