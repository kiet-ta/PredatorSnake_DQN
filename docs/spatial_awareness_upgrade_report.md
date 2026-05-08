# Engineering Report: AlphaZero Spatial Awareness Upgrades

**Date:** May 8, 2026
**Status:** Completed
**Scope:** `src/lite_state`, `src/domain`, `src/mcts`, `python/alpha_zero`

## 1. Executive Summary

This report documents the architectural upgrades applied to the AlphaZero Snake agent to resolve the "Topological Ignorance" and "Greedy Trapper" syndromes. The agent previously plateaued at ~5% board coverage because it lacked the spatial awareness necessary to avoid trapping itself. 

To address this, we integrated a topological state representation into the observation tensor, enabling the Convolutional Neural Network (CNN) to implicitly learn spatial reasoning. We also fortified the Multi-Threading logic within the MCTS engine to ensure rigorous thread-safety during concurrent searches.

## 2. Problem Statement

* **Topological Ignorance:** The previous 4-channel observation tensor (`[empty, head, body, food]`) lacked explicit topological connectivity. The CNN struggled to infer whether a seemingly open area was actually a dead-end corridor.
* **Greedy Trapper Syndrome:** The agent greedily consumed food even if the resulting growth severed its escape routes, leading to premature termination.
* **Performance Constraints:** MCTS simulations must remain extremely fast (~4s/game). Heavy runtime algorithms (e.g., executing multiple BFS traversals per simulation step) would destroy self-play throughput.

## 3. Implemented Features

### 3.1. Topological State Injection (Flood-Fill Channel)
**Status:** ✅ Implemented

We expanded the observation contract from `[4, H, W]` to `[5, H, W]`. The new 5th channel explicitly maps the "Reachable Safe Area".

*   **Implementation Details:** 
    *   A fast Breadth-First Search (BFS) flood-fill algorithm was implemented in Rust (`src/lite_state/encoding.rs` and `src/ecs/systems.rs`).
    *   The BFS originates from the snake's head, treating walls and body segments as impassable obstacles, while treating the food cell as passable.
    *   Reachable cells are encoded as `1` in the 5th observation channel (`CHANNEL_REACHABLE`); unreachable cells are `0`.
*   **Why:** By feeding the reachable area directly into the ResNet, we offload the heavy topological reasoning from the CNN's internal layers. The network can now directly "see" the volume of accessible space for any given state, allowing it to implicitly learn the value of space preservation over greedy food consumption.

### 3.2. MCTS Virtual Loss Hardening
**Status:** ✅ Implemented

We audited and hardened the concurrent MCTS engine's virtual loss implementation to provide formal cross-thread memory visibility guarantees.

*   **Implementation Details:** 
    *   Upgraded the atomic operations on `virtual_loss_active` in `src/mcts/engine.rs` from `Ordering::Relaxed` to `Ordering::AcqRel` during both the Selection (`fetch_add`) and Backpropagation (`fetch_sub`) phases.
*   **Why:** While `Relaxed` ordering often works in practice on x86 architectures when bounded by other synchronization primitives, `AcqRel` provides a formal "happens-before" guarantee. This strictly prevents theoretical instruction reordering bugs where a thread might read a stale virtual loss value during concurrent child node selection, ensuring mathematically sound MCTS exploration.

## 4. Dropped Features

### 4.1. Hamiltonian Reward Shaping (Flood-Fill Delta)
**Status:** ❌ Dropped

The initial plan proposed modifying the step reward to penalize moves that drastically reduce the reachable area (`delta_area < 0`). 

*   **Reason for Rejection:** MCTS requires high-throughput simulations. Computing a BFS flood-fill *before* and *after* every single simulated step would incur approximately 64 Million BFS operations per MCTS move (assuming 800 simulations * 50 steps * 2 BFS ops). This would severely bottleneck the data generation pipeline.
*   **Resolution:** We rely on the CNN (via the 5th observation channel) to map topological traps to lower Value estimates (`Q-value`), achieving spatial avoidance without runtime simulation overhead.

## 5. Architectural Impact & Breaking Changes

> [!WARNING]
> **Observation Contract Change:** The observation tensor is now strictly `[5, H, W]`. 
> **Impact:** All pre-existing model checkpoints and replay buffers are permanently invalidated and cannot be loaded. The agent must be retrained from scratch using the new ResNet architecture (`in_channels=5`).

*   **Rust Parity:** Golden parity between the high-performance `SnakeStateLite` (MCTS) and the `SnakeResource` (Bevy ECS) was successfully maintained. Both engines now generate identical 5-channel outputs.
*   **Test Coverage:** All unit tests and parity fuzz-tests pass successfully. Smoke tests confirm the Python FFI correctly receives and processes the 5-channel NumPy arrays.

## 6. Future Recommendations: O(1) Spatial Reward Shaping

If implicit learning via the 5th channel is insufficient, we propose an O(1) **Neighbor Freedom Bonus** instead of the dropped Hamiltonian shaping. 
*   **Mechanism:** After each move, evaluate the 4 immediate neighbors of the new head position. 
*   **Penalty:** Apply a fixed penalty if the number of free neighbors is `≤ 1` (indicating entry into a corridor or dead-end). 
*   **Advantage:** This requires only 4 array lookups per step instead of a full BFS, maintaining maximum MCTS performance while providing a tactical spatial reward signal.
