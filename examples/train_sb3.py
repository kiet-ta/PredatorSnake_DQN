from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch as th
import torch.nn as nn
from gymnasium import spaces
from snake_env import SnakeEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class SnakeCNN(BaseFeaturesExtractor):
    """Compact CNN extractor sized for small grid observations (e.g. 20x20)."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


def make_env(
    rank: int,
    width: int,
    height: int,
    max_steps: int,
    base_seed: int,
    render: bool = False,
) -> Callable[[], Monitor]:
    """Factory for creating a single environment instance."""

    def _init() -> Monitor:
        env = SnakeEnv(
            width=width,
            height=height,
            max_steps=max_steps,
            render=render,
            seed=base_seed + rank,
        )
        return Monitor(env)

    return _init


def resolve_model_path(args: argparse.Namespace, context: str) -> Path:
    """Resolve model path for play/resume and fail fast with clear guidance."""
    default_best_model_path = Path(args.model_dir) / "best_model" / "best_model.zip"
    model_path = Path(args.model_path) if args.model_path else default_best_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot {context}: model file not found at '{model_path}'. "
            "Provide --model-path or train first to generate a checkpoint."
        )
    return model_path


def train(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    tensorboard_dir = Path(args.tensorboard_log)
    eval_dir = model_dir / "eval"
    checkpoint_dir = model_dir / "checkpoints"
    best_model_dir = model_dir / "best_model"
    final_model_path = model_dir / "snake_dqn_final"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Parallel rollout workers for fast data collection.
    train_env = SubprocVecEnv(
        [
            make_env(
                rank=i,
                width=args.width,
                height=args.height,
                max_steps=args.max_steps,
                base_seed=args.seed,
                render=False,
            )
            for i in range(args.n_envs)
        ],
        start_method="fork",
    )

    # Keep evaluation single-process and deterministic.
    eval_env = DummyVecEnv(
        [
            make_env(
                rank=10_000,
                width=args.width,
                height=args.height,
                max_steps=args.max_steps,
                base_seed=args.seed,
                render=False,
            )
        ]
    )

    # SB3 callback frequencies are counted in vec-env calls,
    # so divide by n_envs to preserve "environment step" semantics.
    eval_freq = max(args.eval_freq // args.n_envs, 1)
    checkpoint_freq = max(args.checkpoint_freq // args.n_envs, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="snake_dqn",
    )

    try:
        if args.resume:
            # Resume from checkpoint with lower exploration and shorter warmup
            # because replay buffer is empty but policy is already trained.
            model_path = resolve_model_path(args, context="resume training")
            model = DQN.load(
                str(model_path),
                env=train_env,
                custom_objects={
                    "exploration_initial_eps": 0.1,
                    "exploration_final_eps": 0.01,
                    "learning_starts": 1_000,
                },
            )
            model.tensorboard_log = str(tensorboard_dir)
            model.verbose = 1
        else:
            model = DQN(
                policy="CnnPolicy",
                env=train_env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                gamma=args.gamma,
                tau=1.0,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                target_update_interval=args.target_update_interval,
                exploration_fraction=args.exploration_fraction,
                exploration_initial_eps=args.exploration_initial_eps,
                exploration_final_eps=args.exploration_final_eps,
                policy_kwargs={
                    "normalize_images": False,
                    "features_extractor_class": SnakeCNN,
                    "features_extractor_kwargs": {"features_dim": args.features_dim},
                },
                tensorboard_log=str(tensorboard_dir),
                verbose=1,
                seed=args.seed,
            )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name="snake_dqn",
        )
        model.save(str(final_model_path))
    finally:
        train_env.close()
        eval_env.close()


def play(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(args, context="play")

    env = SnakeEnv(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        render=True,
        seed=args.seed,
    )
    model = DQN.load(str(model_path))

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or play Snake DQN agent.")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--train", action="store_true", help="Run DQN training.")
    mode.add_argument(
        "--play", action="store_true", help="Play with the best saved model."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing model checkpoint.",
    )

    parser.add_argument("--width", type=int, default=20, help="Grid width.")
    parser.add_argument("--height", type=int, default=20, help="Grid height.")
    parser.add_argument(
        "--max-steps", type=int, default=1_000, help="Max steps per episode."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")

    parser.add_argument(
        "--n-envs", type=int, default=8, help="Parallel env workers for training."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps.",
    )

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--features-dim", type=int, default=256)

    parser.add_argument(
        "--eval-freq", type=int, default=10_000, help="Eval every N env steps."
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint every N env steps.",
    )

    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--tensorboard-log", type=str, default="./tensorboard/")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Checkpoint path for --play/--resume. Defaults to ./models/best_model/best_model.zip",
    )

    args = parser.parse_args()
    if args.play and args.resume:
        parser.error("--resume cannot be used with --play.")
    if not (args.train or args.play or args.resume):
        parser.error("Choose one mode: --train, --resume, or --play.")
    return args


def main() -> None:
    args = parse_args()
    if args.play:
        play(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
