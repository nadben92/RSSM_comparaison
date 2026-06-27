"""Open-loop imagination rollouts and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data import load_dataset
from models.rssm import RSSM
from utils import ensure_dir, get_device, reparameterize, set_seed, setup_logging


def load_model(checkpoint_path: str, device: torch.device) -> tuple[RSSM, Config]:
    """Load RSSM weights from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: Config = ckpt["config"]
    model = RSSM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


@torch.no_grad()
def imagine_rollout(
    model: RSSM,
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
    imagine_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Imagine a trajectory after a short real context.

    Phase 1 — context (posterior): bootstrap ``h`` and ``z`` from real frames.
    Phase 2 — open loop (prior only): sample ``z ~ p(z|h)``, decode, advance ``h``.

    Args:
        model: Trained RSSM.
        context_obs: ``(B, T_ctx, C, H, W)`` real observations.
        context_actions: ``(B, T_ctx, action_dim)`` continuous context actions.
        imagine_actions: ``(B, H, action_dim)`` continuous actions for imagination.

    Returns:
        ``real_frames`` ``(B, T_ctx, C, H, W)`` and
        ``imagined_frames`` ``(B, H, C, H, W)``.
    """
    state, z = model.encode_context(context_obs, context_actions)
    # Last context action becomes a_{t-1} for the first imagination step
    a_prev = context_actions[:, -1]

    imagined: list[torch.Tensor] = []
    horizon = imagine_actions.shape[1]

    for t in range(horizon):
        # Same temporal order as training: backbone first, then prior
        state = model.backbone.step(state, z, a_prev)
        h = model.backbone.hidden(state)
        mu_p, sigma_p = model.prior(h)
        z = reparameterize(mu_p, sigma_p)
        recon = model.decoder(h, z)
        imagined.append(recon)
        a_prev = imagine_actions[:, t]

    imagined_stack = torch.stack(imagined, dim=1)
    return context_obs, imagined_stack


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert ``(T, C, H, W)`` float tensor to ``(T, H, W, C)`` uint8."""
    x = frames.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return (x * 255).astype(np.uint8)


def save_comparison_gif(
    real_frames: torch.Tensor,
    imagined_frames: torch.Tensor,
    path: str | Path,
    fps: int = 4,
) -> None:
    """Save side-by-side GIF: real context (left) vs imagined rollout (right).

    During the context phase only the real frame is shown; during imagination
    the right panel shows decoded prior samples.
    """
    real = _to_uint8(real_frames)
    imagined = _to_uint8(imagined_frames)

    ctx_len = real.shape[0]
    horizon = imagined.shape[0]
    h, w = real.shape[1], real.shape[2]

    frames: list[np.ndarray] = []

    # Context: real on left, blank on right
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    for t in range(ctx_len):
        side_by_side = np.concatenate([real[t], blank], axis=1)
        frames.append(side_by_side)

    # Imagination: last real frame on left, imagined on right
    last_real = real[-1]
    for t in range(horizon):
        side_by_side = np.concatenate([last_real, imagined[t]], axis=1)
        frames.append(side_by_side)

    ensure_dir(Path(path).parent)
    imageio.mimsave(path, frames, fps=fps, loop=0)


def save_kl_per_dim_plot(
    kl_per_dim: torch.Tensor,
    path: str | Path,
    free_nats: float = 3.0,
) -> None:
    """Bar chart of mean KL per latent dimension (diagnostic for dead dims).

    Args:
        kl_per_dim: ``(B, T, latent_dim)`` from a forward pass.
        path: Output PNG path.
        free_nats: Horizontal line showing the free-bits threshold.
    """
    mean_kl = kl_per_dim.mean(dim=(0, 1)).cpu().numpy()
    dims = np.arange(len(mean_kl))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dims, mean_kl, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.axhline(free_nats, color="crimson", linestyle="--", label=f"free_nats={free_nats}")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title("KL per latent dimension")
    ax.legend()
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSSM imagination rollout")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--context-len", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Offset into the episode where context begins (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)
    device = get_device()

    model, config = load_model(args.checkpoint, device)
    observations, actions, lengths = load_dataset(config.data_path)

    ep_idx = args.episode_idx
    ctx_len = args.context_len
    horizon = args.horizon
    start_frame = args.start_frame
    ep_len = int(lengths[ep_idx])
    seq_len = ctx_len + horizon

    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if start_frame + seq_len > ep_len:
        raise ValueError(
            f"Episode {ep_idx} too short (len={ep_len}) for "
            f"start_frame={start_frame}, context_len={ctx_len}, horizon={horizon} "
            f"(need {start_frame + seq_len} frames)"
        )

    end_frame = start_frame + seq_len
    obs_ep = torch.from_numpy(observations[ep_idx, start_frame:end_frame]).float()
    act_ep = torch.from_numpy(actions[ep_idx, start_frame:end_frame]).float()

    context_obs = obs_ep[:ctx_len].unsqueeze(0).to(device)
    context_actions = act_ep[:ctx_len].unsqueeze(0).to(device)
    imagine_actions = act_ep[ctx_len : ctx_len + horizon].unsqueeze(0).to(device)

    real_ctx, imagined = imagine_rollout(
        model, context_obs, context_actions, imagine_actions
    )

    # KL diagnostic on a short forward pass over context + a few steps
    full_obs = obs_ep[: ctx_len + min(horizon, 10)].unsqueeze(0).to(device)
    full_actions = act_ep[: ctx_len + min(horizon, 10)].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(full_obs, full_actions, free_nats=config.free_nats)

    out_dir = ensure_dir(args.output_dir)
    suffix = f"_s{start_frame}" if start_frame > 0 else ""
    gif_path = out_dir / f"imagine_ep{ep_idx}{suffix}.gif"
    kl_path = out_dir / f"kl_per_dim_ep{ep_idx}{suffix}.png"

    save_comparison_gif(
        real_ctx.squeeze(0),
        imagined.squeeze(0),
        gif_path,
    )
    save_kl_per_dim_plot(output.kl_per_dim, kl_path, free_nats=config.free_nats)

    logger.info("Episode %d | start_frame=%d | context_len=%d | horizon=%d", ep_idx, start_frame, ctx_len, horizon)
    logger.info("Saved comparison GIF: %s", gif_path)
    logger.info("Saved KL plot: %s", kl_path)


if __name__ == "__main__":
    main()
