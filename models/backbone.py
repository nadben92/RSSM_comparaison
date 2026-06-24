"""Interchangeable temporal backbone for the RSSM."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from config import Config


class SequenceBackbone(ABC, nn.Module):
    """Abstract recurrent backbone: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}).

    Subclasses (e.g. GRU, Transformer) must implement ``step`` so the rest of
    the RSSM can swap backbones without changing the training loop.
    """

    @abstractmethod
    def step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Advance deterministic state by one time step.

        Args:
            h_prev: Previous hidden state ``(B, hidden_dim)``.
            z_prev: Previous stochastic state ``(B, latent_dim)``.
            a_prev: Previous action ``(B, action_dim)``.

        Returns:
            Next hidden state ``(B, hidden_dim)``.
        """

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero-initialized hidden state for a new sequence."""
        raise NotImplementedError


class GRUBackbone(SequenceBackbone):
    """GRU-based backbone: project [z, a] then apply nn.GRUCell."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        input_dim = config.latent_dim + config.action_dim
        self.hidden_dim = config.hidden_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.gru = nn.GRUCell(config.hidden_dim, config.hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = self.input_proj(x)
        return self.gru(x, h_prev)
