"""Interchangeable temporal backbone for the RSSM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass
class TransformerState:
    """Opaque backbone state: accumulated token sequence + last causal hidden."""

    tokens: torch.Tensor  # (B, T, d_model)
    last_hidden: torch.Tensor  # (B, hidden_dim)


class TransformerBackbone(SequenceBackbone):
    """Transformer backbone with accumulated-sequence state (TSSM / TransDreamer style).

    Unlike GRU/LSTM, state is the growing prefix of projected [z, a] tokens. Each
    ``step`` appends one token, runs the full prefix through a causal Transformer,
    and caches the last-position output for ``hidden``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        input_dim = config.latent_dim + config.action_dim
        self.hidden_dim = config.hidden_dim
        self.d_model = config.transformer_d_model or config.hidden_dim
        ff_dim = config.transformer_ff_dim or (4 * self.d_model)
        self.max_len = config.transformer_max_len

        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoding = nn.Embedding(self.max_len, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.transformer_num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_num_layers,
        )
        self.output_proj = (
            nn.Identity()
            if self.d_model == config.hidden_dim
            else nn.Linear(self.d_model, config.hidden_dim)
        )

    def init_state(self, batch_size: int, device: torch.device) -> TransformerState:
        tokens = torch.zeros(batch_size, 0, self.d_model, device=device)
        last_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return TransformerState(tokens=tokens, last_hidden=last_hidden)

    def hidden(self, state: TransformerState) -> torch.Tensor:
        return state.last_hidden

    def _append_token(
        self,
        tokens: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        pos = tokens.size(1)
        if pos >= self.max_len:
            raise ValueError(
                f"Transformer sequence length {pos + 1} exceeds transformer_max_len={self.max_len}"
            )
        x = x + self.pos_encoding.weight[pos].unsqueeze(0)
        return torch.cat([tokens, x], dim=1)

    def _causal_encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run causal Transformer over the full accumulated prefix."""
        seq_len = tokens.size(1)
        if seq_len == 0:
            return tokens
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=tokens.device,
        )
        return self.encoder(tokens, mask=mask)

    def step(
        self,
        state: TransformerState,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> TransformerState:
        tokens = self._append_token(state.tokens, z_prev, a_prev)
        encoded = self._causal_encode(tokens)
        if encoded.size(1) == 0:
            last_hidden = torch.zeros(
                tokens.size(0),
                self.hidden_dim,
                device=tokens.device,
            )
        else:
            last_hidden = self.output_proj(encoded[:, -1])
        return TransformerState(tokens=tokens, last_hidden=last_hidden)


def build_backbone(config: Config) -> SequenceBackbone:
    """Instantiate the temporal backbone selected in ``config.backbone``."""
    name = config.backbone.lower()
    if name == "gru":
        return GRUBackbone(config)
    if name == "lstm":
        return LSTMBackbone(config)
    if name == "transformer":
        return TransformerBackbone(config)
    raise ValueError(
        f"Unknown backbone '{config.backbone}'. Use 'gru', 'lstm', or 'transformer'."
    )
