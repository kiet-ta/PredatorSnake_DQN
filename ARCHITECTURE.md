# PredatorSnake_DQN Architecture

## 6.1 Observation Encoding

The environment exposes **one-hot encoded observations** as `uint8` tensor shape `[C, H, W]` with `C=4`:

- Channel `0`: Empty cells
- Channel `1`: Snake head
- Channel `2`: Snake body
- Channel `3`: Food

Each cell position `(y, x)` is represented categorically by channel activation (`0/1`), eliminating ordinal bias from scalar-coded grids.

## 6.2 Gym Spaces

- `action_space = Discrete(3)` (relative controls):
  - `0 = Straight`
  - `1 = TurnLeft`
  - `2 = TurnRight`
- `observation_space = Box(low=0, high=1, shape=(4, H, W), dtype=uint8)`

This contract is implemented consistently across Rust FFI outputs and the Python Gymnasium wrapper.

## 9. Trade-off Analysis Matrix

| Criterion                       | Evaluation | Strengths                                                                 | Trade-offs / Risks                                       | Mitigation / Decision                                                                            |
| ------------------------------- | ---------- | ------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **DevEx (Headless RL backend)** | **High**   | Clean Gym API, simple constructor controls, deterministic step pipeline   | Mixed Rust/Python toolchain onboarding cost              | Keep Python API minimal and document build workflow                                              |
| **Performance**                 | **High**   | Rust ECS simulation + low-overhead FFI + one-hot tensors                  | One-hot uses more memory than scalar grid                | Accept memory overhead for better policy learning stability                                      |
| **Security**                    | **High**   | Rust memory safety in core logic, constrained FFI surface                 | FFI boundary misuse remains possible if API is bypassed  | Keep strict typed API and reject invalid action values                                           |
| **Maintainability**             | **High**   | Clear boundaries (`domain/ecs/ffi/bridge/app`) and deterministic schedule | More modules increase navigation surface                 | Preserve separation; update docs with data-flow and lifecycle                                    |
| **Reusability**                 | **High**   | Domain and ECS are independent from SB3 specifics                         | Current Python wrapper targets Gymnasium conventions     | Add additional adapters only at control-plane layer                                              |
| **Scalability**                 | **High**   | Safe process-level scale-out with `SubprocVecEnv`                         | Bevy default task pools can oversubscribe CPU in workers | **Headless mode forces single-thread task pool** so Python process parallelism remains efficient |
