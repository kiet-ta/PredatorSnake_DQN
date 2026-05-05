# PredatorSnake_DQN

High-performance custom Snake reinforcement learning environment:

- **Data Plane**: Rust + Bevy ECS simulation.
- **Control Plane**: Python Gymnasium wrapper for Stable-Baselines3.
- **Bridge**: PyO3 + rust-numpy module built by maturin.

## Build and install

```bash
maturin develop --release
```

## Python usage

```python
from snake_env import SnakeEnv

env = SnakeEnv(width=20, height=20, max_steps=1000, render=False, seed=42)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)  # 0=Straight, 1=Left, 2=Right
```

## Deterministic tick pipeline

Each `env.step(action)` triggers exactly one Bevy update cycle with strict ordering:

1. `apply_action_system`
2. `move_system`
3. `resolve_step_system` (eat/collision/reward/termination)
4. `write_observation_system`

This order is enforced with Bevy `.chain()`.

## Headless threading model

When `render=False`, Bevy is configured with a single-thread task pool. This avoids thread oversubscription when Python scales out with `SubprocVecEnv`.

## Observation and spaces

- Observation: `numpy.ndarray` `uint8` one-hot shape `[4, H, W]`
  - channel `0`: empty
  - channel `1`: snake head
  - channel `2`: snake body
  - channel `3`: food
- Action space: `Discrete(3)` relative controls:
  - `0`: Straight
  - `1`: TurnLeft
  - `2`: TurnRight

## Rewards and termination

- `+1.0` when food is eaten
- `-1.0` on collision death
- `-0.01` living penalty every step
- `terminated=True` on collision
- `truncated=True` when `max_steps` is reached
