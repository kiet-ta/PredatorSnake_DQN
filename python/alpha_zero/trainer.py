import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .alpha_zero_net import AlphaZeroNet
from .replay_buffer import ReplayBuffer

class AlphaZeroTrainer:
    def __init__(
        self,
        net: AlphaZeroNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(
        self, buffer: ReplayBuffer, batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Samples random batches from the buffer to train the network.
        In one epoch, we do len(buffer) // batch_size steps, ensuring we see 
        approximately all data.
        """
        self.net.train()
        
        num_batches = max(1, len(buffer) // batch_size)
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0
        
        for _ in range(num_batches):
            obs_batch, pi_batch, z_batch = buffer.sample_batch(batch_size, self.device)
            
            self.optimizer.zero_grad()
            
            pred_logits, pred_value = self.net(obs_batch)
            
            v_loss, p_loss, loss = self._alpha_zero_loss(
                pred_logits=pred_logits,
                pred_value=pred_value,
                target_pi=pi_batch,
                target_z=z_batch
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            
        return {
            "loss/total": total_loss / num_batches,
            "loss/value": total_v_loss / num_batches,
            "loss/policy": total_p_loss / num_batches,
        }

    @staticmethod
    def _alpha_zero_loss(
        pred_logits: torch.Tensor,
        pred_value: torch.Tensor,
        target_pi: torch.Tensor,
        target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        L = MSE(z, v) + CrossEntropy(pi, p)
        L2 is handled by AdamW weight_decay.
        """
        # Value loss: MSE between predicted value and actual game outcome
        value_loss = F.mse_loss(pred_value, target_z)
        
        # Policy loss: Cross Entropy between MCTS target policy and network logits
        # target_pi is a probability distribution [B, 3]
        # pred_logits are raw scores [B, 3]
        # Cross Entropy = -sum(target_pi * log_softmax(pred_logits))
        log_probs = F.log_softmax(pred_logits, dim=1)
        policy_loss = -torch.sum(target_pi * log_probs, dim=1).mean()
        
        total_loss = value_loss + policy_loss
        
        return value_loss, policy_loss, total_loss
