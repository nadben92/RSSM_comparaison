"""CNN decoder: concat(h, z) -> (B, 3, H, W)."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class CNNDecoder(nn.Module):
    """Linear projection followed by three ConvTranspose2d layers (mirror of encoder).

    With kernel=4, stride=2, padding=1: spatial_size → 2× → 2× → 2× (e.g. 4 → 8 → 16 → 32).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        c1, c2, c3 = config.decoder_channels
        state_dim = config.hidden_dim + config.latent_dim
        spatial = config.spatial_size

        self.fc = nn.Linear(state_dim, c1 * spatial * spatial)
        self.spatial = spatial
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, config.img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state into a reconstructed image.

        Args:
            h: Deterministic state ``(B, hidden_dim)``.
            z: Stochastic state ``(B, latent_dim)``.

        Returns:
            Reconstructed observation ``(B, C, H, W)`` in ``[0, 1]``.
        """
        x = torch.cat([h, z], dim=-1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.spatial, self.spatial)
        return self.deconv(x)
