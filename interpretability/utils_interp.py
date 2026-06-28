"""Shared helpers for RSSM latent interpretability (read-only analysis)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

# Allow running scripts from repo root or from interpretability/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from data import load_dataset  # noqa: E402
from evaluation import ball_position  # noqa: E402
from imagine import load_model  # noqa: E402
from models.rssm import RSSM  # noqa: E402
from utils import get_device, reparameterize, set_seed  # noqa: E402


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> tuple[RSSM, Config]:
    """Load RSSM + config from a training checkpoint."""
    if device is None:
        device = get_device()
    model, config = load_model(checkpoint_path, device)
    return model, config


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


@torch.no_grad()
def reference_state_from_context(
    model: RSSM,
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode context via posterior; return ``(h, z, backbone_state)`` at last context step."""
    state, z = model.encode_context(context_obs, context_actions)
    h = model.backbone.hidden(state)
    return h, z, state


@torch.no_grad()
def decode_frame(model: RSSM, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Decode ``(B, hidden_dim)`` and ``(B, latent_dim)`` to ``(B, C, H, W)``."""
    return model.decoder(h, z).clamp(0, 1)


@torch.no_grad()
def mean_kl_per_dim(
    model: RSSM,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
    free_nats: float = 0.0,
) -> np.ndarray:
    """Mean KL(q||p) per latent dimension over batch and time."""
    output = model(obs_seq, action_seq, free_nats=free_nats)
    return output.kl_per_dim.mean(dim=(0, 1)).cpu().numpy()


def active_dimensions(
    kl_per_dim: np.ndarray,
    *,
    top_k: int = 8,
    min_kl: float = 0.01,
) -> np.ndarray:
    """Return indices of latent dims with meaningful KL (sorted by KL descending)."""
    dims = np.where(kl_per_dim >= min_kl)[0]
    if len(dims) == 0:
        dims = np.argsort(kl_per_dim)[::-1][:top_k]
    else:
        dims = dims[np.argsort(kl_per_dim[dims])[::-1]]
    if len(dims) > top_k:
        dims = dims[:top_k]
    return dims.astype(np.int64)


@torch.no_grad()
def posterior_stats_at_reference(
    model: RSSM,
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior mean/std of z at the last context frame (for traversal ranges)."""
    state, z_prev, a_prev = model._init_states(context_obs.size(0), context_obs.device)
    for t in range(context_obs.size(1)):
        obs_t = context_obs[:, t]
        a_t = context_actions[:, t]
        state = model.backbone.step(state, z_prev, a_prev)
        h = model.backbone.hidden(state)
        embed = model.encoder(obs_t)
        mu_q, sigma_q = model.posterior(torch.cat([h, embed], dim=-1))
        z_prev = mu_q if t == context_obs.size(1) - 1 else reparameterize(mu_q, sigma_q)
        a_prev = a_t
    return mu_q, sigma_q


def ball_position_from_tensor(frame: torch.Tensor, threshold: float = 0.2) -> tuple[float, float] | None:
    """Extract ball position from ``(C, H, W)`` or ``(B, C, H, W)`` tensor."""
    if frame.dim() == 4:
        frame = frame[0]
    return ball_position(frame.cpu().numpy(), threshold=threshold)


@torch.no_grad()
def encode_context_with_ablation(
    model: RSSM,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
    ablate_dim: int | None = None,
) -> tuple[BackboneState, torch.Tensor]:
    """Bootstrap context like ``encode_context``, optionally forcing z[i] ← μ_prior[i]."""
    batch_size, seq_len = obs_seq.shape[:2]
    device = obs_seq.device
    state, z, a_prev = model._init_states(batch_size, device)

    for t in range(seq_len):
        state = model.backbone.step(state, z, a_prev)
        h = model.backbone.hidden(state)
        embed = model.encoder(obs_seq[:, t])
        mu_p, sigma_p = model.prior(h)
        mu_q, sigma_q = model.posterior(torch.cat([h, embed], dim=-1))
        z = mu_q
        if ablate_dim is not None:
            z = z.clone()
            z[:, ablate_dim] = mu_p[:, ablate_dim]
        a_prev = action_seq[:, t]

    return state, z


@torch.no_grad()
def imagine_with_z_hook(
    model: RSSM,
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
    imagine_actions: torch.Tensor,
    z_modifier: callable | None = None,
    ablate_dim: int | None = None,
) -> torch.Tensor:
    """Open-loop imagination after context encoding (optional dim ablation at encode time)."""
    if ablate_dim is None:
        state, z = model.encode_context(context_obs, context_actions)
    else:
        state, z = encode_context_with_ablation(
            model, context_obs, context_actions, ablate_dim=ablate_dim
        )
    a_prev = context_actions[:, -1]
    frames: list[torch.Tensor] = []
    for t in range(imagine_actions.size(1)):
        state = model.backbone.step(state, z, a_prev)
        h = model.backbone.hidden(state)
        mu_p, sigma_p = model.prior(h)
        z = reparameterize(mu_p, sigma_p)
        if z_modifier is not None:
            z = z_modifier(z, mu_p, sigma_p, t)
        frames.append(model.decoder(h, z))
        a_prev = imagine_actions[:, t]
    return torch.stack(frames, dim=1)


@torch.no_grad()
def collect_posterior_latents(
    model: RSSM,
    obs_seq: torch.Tensor,
    action_seq: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect posterior z and h over time. Returns ``z_seq (T, D)``, ``h_seq (T, H)``."""
    batch_size, seq_len = obs_seq.shape[:2]
    device = obs_seq.device
    state, z_prev, a_prev = model._init_states(batch_size, device)
    z_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []

    for t in range(seq_len):
        state = model.backbone.step(state, z_prev, a_prev)
        h = model.backbone.hidden(state)
        embed = model.encoder(obs_seq[:, t])
        mu_q, sigma_q = model.posterior(torch.cat([h, embed], dim=-1))
        z_prev = mu_q  # use mean for structure analysis
        z_list.append(z_prev[0].cpu().numpy())
        h_list.append(h[0].cpu().numpy())
        a_prev = action_seq[:, t]

    return np.stack(z_list, axis=0), np.stack(h_list, axis=0)


def sweep_values_for_dim(
    dim: int,
    *,
    range_mode: str,
    n_values: int,
    z_traj: np.ndarray | None = None,
    mu_q: torch.Tensor | None = None,
    sigma_q: torch.Tensor | None = None,
    sigma_scale: float = 3.0,
    temporal_margin: float = 0.2,
) -> np.ndarray:
    """Build sweep values for one latent dimension."""
    if range_mode == "temporal":
        if z_traj is None:
            raise ValueError("z_traj required for temporal range mode")
        z_i = z_traj[:, dim]
        z_min, z_max = float(z_i.min()), float(z_i.max())
        span = z_max - z_min
        margin = temporal_margin * span if span > 0 else 1.0
        return np.linspace(z_min - margin, z_max + margin, n_values)

    if mu_q is None or sigma_q is None:
        raise ValueError("mu_q and sigma_q required for sigma range mode")
    z_mean = float(mu_q[0, dim].item())
    z_std = max(float(sigma_q[0, dim].item()), 0.05)
    return np.linspace(z_mean - sigma_scale * z_std, z_mean + sigma_scale * z_std, n_values)


def evenly_spaced_starts(episode_len: int, context_len: int, n_refs: int) -> list[int]:
    """Pick ``n_refs`` valid context start indices spread across an episode."""
    if n_refs <= 1:
        return [0]
    max_start = max(0, episode_len - context_len)
    if max_start == 0:
        return [0] * n_refs
    return [int(round(i * max_start / (n_refs - 1))) for i in range(n_refs)]


@torch.no_grad()
def displacement_for_dim(
    model: RSSM,
    h: torch.Tensor,
    z_ref: torch.Tensor,
    dim: int,
    values: np.ndarray,
) -> tuple[float, float]:
    """Max Δx and Δy (pixels) when sweeping ``dim`` over ``values``."""
    xs: list[float] = []
    ys: list[float] = []
    for val in values:
        z_var = z_ref.clone()
        z_var[0, dim] = val
        pos = ball_position_from_tensor(decode_frame(model, h, z_var))
        if pos is not None:
            xs.append(pos[0])
            ys.append(pos[1])
    if len(xs) < 2:
        return 0.0, 0.0
    return max(xs) - min(xs), max(ys) - min(ys)


def positions_from_observations(obs: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Extract ball (x, y) per frame when GT positions are not stored in the dataset."""
    pos = np.zeros((obs.shape[0], 2), dtype=np.float32)
    for t in range(obs.shape[0]):
        p = ball_position(obs[t], threshold=threshold)
        if p is None:
            pos[t] = pos[t - 1] if t > 0 else np.array([16.0, 16.0], dtype=np.float32)
        else:
            pos[t] = p
    return pos


def load_episode(
    data_path: str,
    episode_idx: int = 0,
    start: int = 0,
    length: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``obs (T,C,H,W)``, ``actions (T, A)``, ``positions (T,2)``."""
    obs_all, act_all, lengths, positions = load_dataset(data_path)
    ep_len = int(lengths[episode_idx])
    end = min(start + length, ep_len)
    obs = obs_all[episode_idx, start:end]
    act = act_all[episode_idx, start:end]
    if positions is not None:
        pos = positions[episode_idx, start:end]
    else:
        pos = positions_from_observations(obs)
    return obs, act, pos


def velocity_from_positions(pos: np.ndarray) -> np.ndarray:
    """Finite-difference velocity ``(T, 2)``; first step duplicated."""
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    vel[0] = vel[1]
    return vel
