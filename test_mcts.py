import numpy as np
from predator_snake_dqn import PyAlphaZeroEngine


def dummy_model(obs: np.ndarray):
    # obs is shape [B, 4, H, W]
    batch_size = obs.shape[0]
    # Return random logits and zero values
    logits = np.random.randn(batch_size, 3).astype(np.float32)
    values = np.zeros(batch_size, dtype=np.float32)
    return logits, values


def main():
    print("Creating PyAlphaZeroEngine...")
    engine = PyAlphaZeroEngine(
        width=10,
        height=10,
        max_steps=100,
        num_simulations=100,
        num_threads=4,
        c_puct=1.0,
        discount_factor=0.99,
        virtual_loss=1.0,
        batch_size=8,
        max_batch_wait_us=5000,
    )

    print("Resetting engine...")
    obs, info = engine.reset()
    print(f"Initial observation shape: {obs.shape}")

    print("Setting model...")
    engine.set_model(dummy_model)

    print("Running MCTS step...")
    action, result = engine.mcts_step()

    print(f"Action chosen: {action}")
    print(f"Policy: {result['policy']}")
    print(f"Root Value: {result['root_value']}")
    print("Test passed successfully!")


if __name__ == "__main__":
    main()
