"""CNN encoder: (B, 3, H, W) -> (B, embed_dim)."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class CNNEncoder(nn.Module):
    """Three-layer stride-2 Conv2d stack followed by a linear projection."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        c1, c2, c3 = config.encoder_channels
        in_ch = config.img_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(config.flatten_dim, config.embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations.

        Args:
            obs: ``(B, C, H, W)`` normalized in ``[0, 1]``.

        Returns:
            Embeddings ``(B, embed_dim)``.
        """
        x = self.conv(obs)
        x = x.flatten(start_dim=1)
        return self.fc(x)
