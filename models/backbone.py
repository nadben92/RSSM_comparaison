"""Interchangeable temporal backbone for the RSSM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from config import Config

BackboneState = Any


class SequenceBackbone(ABC, nn.Module):
    """Abstract recurrent backbone with an opaque internal state.

    GRU uses a single tensor ``h``; LSTM uses ``(h, c)``. The RSSM passes the
  state through ``step`` without inspecting its structure and calls ``hidden``
    to obtain ``(B, hidden_dim)`` for prior, posterior, and decoder.
    """

    hidden_dim: int

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> BackboneState:
        """Return zero-initialized backbone state for a new sequence."""

    @abstractmethod
    def step(
        self,
        state: BackboneState,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> BackboneState:
        """Advance backbone state by one time step."""

    @abstractmethod
    def hidden(self, state: BackboneState) -> torch.Tensor:
        """Extract deterministic hidden ``h`` ``(B, hidden_dim)`` from ``state``."""


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

    def hidden(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def step(
        self,
        state: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = self.input_proj(x)
        return self.gru(x, state)


class LSTMBackbone(SequenceBackbone):
    """LSTM-based backbone: project [z, a] then apply nn.LSTMCell."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        input_dim = config.latent_dim + config.action_dim
        self.hidden_dim = config.hidden_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.lstm = nn.LSTMCell(config.hidden_dim, config.hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def hidden(self, state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return state[0]

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = self.input_proj(x)
        h, c = state
        return self.lstm(x, (h, c))


def build_backbone(config: Config) -> SequenceBackbone:
    """Instantiate the temporal backbone selected in ``config.backbone``."""
    name = config.backbone.lower()
    if name == "gru":
        return GRUBackbone(config)
    if name == "lstm":
        return LSTMBackbone(config)
    raise ValueError(f"Unknown backbone '{config.backbone}'. Use 'gru' or 'lstm'.")
