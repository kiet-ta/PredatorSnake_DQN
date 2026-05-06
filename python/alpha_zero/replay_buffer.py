from dataclasses import dataclass
from collections import deque
import random
import numpy as np
import torch
from typing import List, Tuple

@dataclass
class SelfPlaySample:
    """A single training sample from self-play."""
    observation: np.ndarray  # [4, H, W] uint8
    target_policy: np.ndarray  # [3] float32 (MCTS visit distribution)
    outcome: float  # z in [-1, 1]

class ReplayBuffer:
    """Fixed-size circular buffer with uniform random sampling."""
    def __init__(self, capacity: int = 200_000):
        self._buffer: deque[SelfPlaySample] = deque(maxlen=capacity)

    def add_game(self, samples: List[SelfPlaySample]) -> None:
        """Add all samples from a completed game."""
        self._buffer.extend(samples)

    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch from the buffer.
        Returns: (obs [B, 4, H, W], target_pi [B, 3], z [B, 1])
        """
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        
        obs_batch = np.stack([sample.observation for sample in batch])
        pi_batch = np.stack([sample.target_policy for sample in batch])
        z_batch = np.array([[sample.outcome] for sample in batch], dtype=np.float32)
        
        # We keep obs as uint8 to save memory transfer bandwidth, the model casts it
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        pi_tensor = torch.from_numpy(pi_batch).to(device)
        z_tensor = torch.from_numpy(z_batch).to(device)
        
        return obs_tensor, pi_tensor, z_tensor

    def __len__(self) -> int:
        return len(self._buffer)
