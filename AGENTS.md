# AGENTS.md

> Agent-facing project context for **PredatorSnake_DQN**.  
> This file follows the open AGENTS.md convention: https://agents.md/

## 1) Project Overview

This repository implements a high-performance Snake RL environment:

- **Data plane (Rust)**: Bevy ECS simulation and game rules
- **Control plane (Python)**: Gymnasium + Stable-Baselines3 (DQN)
- **Bridge**: PyO3 + rust-numpy (Python extension via maturin)

Current observation contract is **one-hot**:

- shape: `[5, H, W]`
- dtype: `uint8`
- channels: `0=empty, 1=head, 2=body, 3=food, 4=reachable_area`

Action space:

- `Discrete(3)` with relative controls: `0=Straight, 1=TurnLeft, 2=TurnRight`

## 2) Key Architecture Facts (must preserve)

1. **Deterministic simulation pipeline** in Rust:
   - `Action -> Move -> Resolve(Eat/Collide/Reward) -> WriteObservation`
2. **Headless scalability fix**:
   - When `render=false`, Bevy must run with single-thread task pool to avoid `SubprocVecEnv` oversubscription.
3. **Python play rendering is default/stable**:
   - `--play-renderer python` is preferred for review loops.
   - `--play-renderer bevy` is experimental due to Winit/manual tick constraints.
4. **Resume training support is implemented**:
   - `--resume` path uses `DQN.load(..., custom_objects=...)` and overrides:
     - `exploration_initial_eps=0.1`
     - `exploration_final_eps=0.01`
     - `learning_starts=1000`

## 3) Source Map

- Rust core:
  - `src/domain/` (rules, config, actions, state)
  - `src/ecs/` (resources/systems)
  - `src/app/` (builder/schedules)
  - `src/ffi/` (PyO3 API)
  - `src/lite_state/` (MCTS-ready lightweight state)
- Python wrapper:
  - `python/snake_env/env.py`
- Training/playing script:
  - `examples/train_sb3.py`
- Learning/architecture docs:
  - `docs/LEARNING.md`
  - `docs/PHASE_6_ALPHAZERO_MCTS.md`

## 4) Setup Commands

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install maturin gymnasium stable-baselines3 tensorboard matplotlib numpy
```

Build/install Rust extension:

```bash
maturin develop --release
```

## 5) Run Commands

Train from scratch:

```bash
python examples/train_sb3.py --train --n-envs 8 --total-timesteps 1000000
```

Resume:

```bash
python examples/train_sb3.py --resume --model-path ./models/best_model/best_model.zip --total-timesteps 500000
```

Play (recommended renderer):

```bash
python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip --play-renderer python
```

Play one episode then stop at endgame (default behavior):

```bash
python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip --play-renderer python
```

Play continuously across episodes:

```bash
python examples/train_sb3.py --play --model-path ./models/best_model/best_model.zip --play-renderer python --play-loop
```

TensorBoard:

```bash
tensorboard --logdir ./tensorboard/
```

## 6) Test & Validation Commands

Rust tests:

```bash
cargo test
```

Phase A parity tests (critical):

```bash
cargo test --test lite_state_parity_tests -- --nocapture
```

Python syntax checks:

```bash
python -m py_compile examples/train_sb3.py python/snake_env/env.py
```

## 7) Coding Rules for Agents

1. **Do not break observation/action contracts** (`[5,H,W]`, `Discrete(3)`).
2. **Preserve deterministic behavior** for seeded runs.
3. **Maintain parity** between `SnakeStateLite` and ECS rules.
4. **No broad error swallowing**; fail fast with actionable messages.
5. **For UI issues**, prefer explicit backend diagnostics (print active renderer/backend).
6. **Keep Rust/Python boundaries minimal and batched** (important for upcoming MCTS integration).

## 8) High-Risk Areas

- `src/ecs/systems.rs`: reward/termination physics
- `src/ecs/resources.rs`: RNG and reset semantics
- `src/lite_state/*`: must stay logic-identical to ECS environment
- `examples/train_sb3.py`: resume path, renderer selection, backend detection

Any changes here require tests and parity checks.

## 9) Known Pitfalls

1. **Matplotlib backend trap**:
   - `QtAgg` is interactive; do not classify backends using substring checks like `"agg" in backend`.
2. **Bevy window freeze in manual tick mode**:
   - Winit event-loop issues can occur when driven via Python `step` loops.
3. **Process/thread oversubscription**:
   - Keep headless Bevy single-thread under vectorized training.

## 10) Roadmap Context (AlphaZero/MCTS)

Planned direction:

- Rust-based MCTS engine
- Lightweight `SnakeStateLite` for fast cloning/stepping
- Async batched inference bridge (Rust queue -> Python model batch eval)
- Strict golden parity tests as a hard gate before scaling MCTS

Reference document: `docs/PHASE_6_ALPHAZERO_MCTS.md`

## 11) Official References

- AGENTS.md convention: https://agents.md/
- Bevy docs: https://bevyengine.org/learn/
- PyO3 docs: https://pyo3.rs/
- Maturin docs: https://www.maturin.rs/
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
