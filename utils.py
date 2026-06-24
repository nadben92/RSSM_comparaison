"""Shared utilities: seeding, device selection, logging helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("rssm")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Sample from N(mu, sigma^2) via the reparameterization trick.

    z = mu + sigma * eps,  eps ~ N(0, I)
    """
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


def current_beta(step: int, beta_max: float, anneal_steps: int) -> float:
    """Linear beta-annealing schedule from 0 to ``beta_max``."""
    if anneal_steps <= 0:
        return beta_max
    return min(beta_max, beta_max * step / anneal_steps)
