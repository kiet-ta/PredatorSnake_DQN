import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return x + out


class AlphaZeroNet(nn.Module):
    """
    Dual-headed ResNet architecture for AlphaZero Snake.
    Input:  [B, 5, H, W] tensor (uint8 casted to float32)
    Channels: 0=empty, 1=head, 2=body, 3=food, 4=reachable_area
    Output: logits [B, 3], value [B, 1]
    """
    def __init__(self, in_channels: int = 5, num_res_blocks: int = 5, channels: int = 64, board_size: int = 20):
        super().__init__()
        self.board_size = board_size
        
        # Initial convolutional block
        self.initial_conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(channels)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Residual tower
        self.res_tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy head: 1x1 conv -> BN -> ReLU -> Flatten -> Linear
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * board_size * board_size, 3)
        
        # Value head: 1x1 conv -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> Tanh
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 128)
        self.value_fc1_relu = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(128, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cast input uint8 directly to float32 inside the graph
        x = x.float()
        
        # Shared trunk
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.res_tower(x)
        
        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.view(p.size(0), -1)  # Flatten
        logits = self.policy_fc(p)
        
        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_relu(v)
        v = v.view(v.size(0), -1)  # Flatten
        v = self.value_fc1(v)
        v = self.value_fc1_relu(v)
        v = self.value_fc2(v)
        value = self.value_tanh(v)
        
        return logits, value

    @torch.no_grad()
    def predict(self, obs_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interface for the Rust PyAlphaZeroEngine.
        Takes a batch of [B, 5, H, W] numpy arrays of dtype uint8.
        Returns tuple of (logits, values) as float32 numpy arrays.
        """
        self.eval()
        device = next(self.parameters()).device
        obs_tensor = torch.from_numpy(obs_uint8).to(device)
        
        logits, values = self.forward(obs_tensor)
        
        return logits.cpu().numpy(), values.view(-1).cpu().numpy()
