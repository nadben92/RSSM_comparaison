"""Synthetic bouncing-ball episodes (numpy only, no Gymnasium).

Classic billiard-style world for video prediction / world-model benchmarks.
Pure elastic bounces — no external forces. Actions are always zero (``action_dim=1``)
for RSSM pipeline compatibility; the backbone learns free physics only.

Generates the same on-disk layout as ``data.collect_episodes`` (uint8 obs on disk,
normalized to float32 when loaded via ``data.load_dataset``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import ensure_dir


@dataclass
class BouncingBallConfig:
    """Physics and rendering knobs for synthetic episodes."""

    img_size: int = 32
    ball_radius: int = 7  # r=5→~8%, r=7→~15%, r=8→~19% on 32x32
    initial_speed: float = 2.5
    action_dim: int = 1  # fixed zero action for RSSM compatibility
    bg_color: tuple[int, int, int] = (0, 0, 0)
    ball_color: tuple[int, int, int] = (255, 64, 64)


def default_ball_radius(img_size: int, target_fraction: float = 0.15) -> int:
    """Heuristic radius so disk area ≈ ``target_fraction`` of the image."""
    area = target_fraction * img_size * img_size
    return max(2, int(round((area / np.pi) ** 0.5)))


def render_frame(
    pos: np.ndarray,
    cfg: BouncingBallConfig,
) -> np.ndarray:
    """Draw the ball on a square canvas.

    Args:
        pos: Ball center ``(x, y)`` in pixel coordinates.
        cfg: Rendering configuration.

    Returns:
        ``(C, H, W)`` uint8 RGB.
    """
    s = cfg.img_size
    r = cfg.ball_radius
    frame = np.zeros((s, s, 3), dtype=np.uint8)
    frame[..., :] = np.array(cfg.bg_color, dtype=np.uint8)

    cx, cy = float(pos[0]), float(pos[1])
    y0, y1 = max(0, int(np.floor(cy - r))), min(s, int(np.ceil(cy + r)) + 1)
    x0, x1 = max(0, int(np.floor(cx - r))), min(s, int(np.ceil(cx + r)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    frame[y0:y1, x0:x1][mask] = np.array(cfg.ball_color, dtype=np.uint8)
    return frame.transpose(2, 0, 1)


def ball_pixel_stats(frame_chw: np.ndarray, cfg: BouncingBallConfig) -> tuple[int, float]:
    """Count ball pixels (any channel > 0 on black bg) and image fraction."""
    s = cfg.img_size
    rgb = frame_chw.transpose(1, 2, 0)
    ball_mask = rgb.max(axis=-1) > 0
    count = int(ball_mask.sum())
    fraction = 100.0 * count / (s * s)
    return count, fraction


def _sample_initial_state(rng: np.random.Generator, cfg: BouncingBallConfig) -> tuple[np.ndarray, np.ndarray]:
    r = cfg.ball_radius
    s = cfg.img_size
    margin = r + 1
    pos = rng.uniform(margin, s - margin, size=2).astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    speed = cfg.initial_speed * rng.uniform(0.7, 1.3)
    vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed
    return pos, vel


def _step_physics(
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: BouncingBallConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic bounce step — no external forces."""
    r = cfg.ball_radius
    s = cfg.img_size
    pos = pos + vel

    if pos[0] - r < 0:
        pos[0] = r
        vel[0] = abs(vel[0])
    elif pos[0] + r > s - 1:
        pos[0] = s - 1 - r
        vel[0] = -abs(vel[0])

    if pos[1] - r < 0:
        pos[1] = r
        vel[1] = abs(vel[1])
    elif pos[1] + r > s - 1:
        pos[1] = s - 1 - r
        vel[1] = -abs(vel[1])

    return pos, vel


def roll_out_episode(
    episode_len: int,
    rng: np.random.Generator,
    cfg: BouncingBallConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one episode with pure elastic bounces (zero actions).

    Returns:
        observations ``(T, C, H, W)`` uint8,
        actions ``(T, action_dim)`` float32 (always zero).
    """
    pos, vel = _sample_initial_state(rng, cfg)
    zero_action = np.zeros(cfg.action_dim, dtype=np.float32)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []

    for _ in range(episode_len):
        obs_list.append(render_frame(pos, cfg))
        act_list.append(zero_action)
        pos, vel = _step_physics(pos, vel, cfg)

    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def save_episode_gif(
    path: str | Path,
    episode_len: int = 40,
    seed: int = 42,
    cfg: BouncingBallConfig | None = None,
    fps: int = 8,
    scale: int = 8,
) -> Path:
    """Save one rollout as a GIF (upscaled nearest-neighbor for visibility)."""
    cfg = cfg or BouncingBallConfig()
    rng = np.random.default_rng(seed)
    obs, _ = roll_out_episode(episode_len, rng, cfg)

    frames: list[np.ndarray] = []
    for t in range(episode_len):
        rgb = obs[t].transpose(1, 2, 0)
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        frames.append(rgb)

    out = ensure_dir(Path(path).parent) / Path(path).name
    imageio.mimsave(out, frames, fps=fps, loop=0)

    n_px, frac = ball_pixel_stats(obs[0], cfg)
    print(f"GIF saved: {out}")
    print(f"  ball_radius={cfg.ball_radius} → {n_px}px ({frac:.1f}%) at native 32x32")
    print(f"  pure elastic physics | {episode_len} frames @ {fps} fps")
    return out


def collect_episodes(
    num_episodes: int,
    max_steps: int,
    seed: int = 42,
    cfg: BouncingBallConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``num_episodes`` synthetic rollouts.

    Returns:
        observations ``(N, T, C, H, W)`` uint8,
        actions ``(N, T, action_dim)`` float32 (all zero),
        lengths ``(N,)`` (all equal to ``max_steps``).
    """
    cfg = cfg or BouncingBallConfig()
    rng = np.random.default_rng(seed)
    n, t = num_episodes, max_steps
    c, s = 3, cfg.img_size
    adim = cfg.action_dim

    obs_array = np.zeros((n, t, c, s, s), dtype=np.uint8)
    act_array = np.zeros((n, t, adim), dtype=np.float32)
    lengths = np.full(n, t, dtype=np.int64)

    for ep in tqdm(range(num_episodes), desc="Generating bouncing-ball episodes"):
        obs, acts = roll_out_episode(t, rng, cfg)
        obs_array[ep] = obs
        act_array[ep] = acts

    return obs_array, act_array, lengths


def preview_episode(
    episode_len: int = 20,
    seed: int = 42,
    cfg: BouncingBallConfig | None = None,
    save_path: str | None = None,
) -> None:
    """Visualize one short episode before large-scale collection."""
    cfg = cfg or BouncingBallConfig()
    rng = np.random.default_rng(seed)
    obs, actions = roll_out_episode(episode_len, rng, cfg)

    ncols = min(episode_len, 10)
    nrows = (episode_len + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows))
    axes = np.atleast_2d(axes)
    for t in range(episode_len):
        ax = axes[t // ncols, t % ncols]
        rgb = obs[t].transpose(1, 2, 0)
        ax.imshow(rgb, interpolation="nearest")
        n_px, frac = ball_pixel_stats(obs[t], cfg)
        ax.set_title(f"t={t}\n{n_px}px ({frac:.1f}%)", fontsize=8)
        ax.axis("off")
    for t in range(episode_len, nrows * ncols):
        axes[t // ncols, t % ncols].axis("off")
    fig.suptitle(
        f"Bouncing ball (elastic) | {cfg.img_size}x{cfg.img_size} r={cfg.ball_radius}",
        fontsize=11,
    )
    fig.tight_layout()

    mid = episode_len // 2
    n_px, frac = ball_pixel_stats(obs[mid], cfg)
    print("=" * 60)
    print("INSPECT BEFORE COLLECT / TRAIN")
    print(f"  img_size={cfg.img_size}, ball_radius={cfg.ball_radius}")
    print(f"  frame {mid}: ball pixels = {n_px} ({frac:.2f}% of image)")
    print(f"  actions (always zero): {actions[0]}")
    print("  target ~15-20% → ball_radius=7 or 8")
    print("=" * 60)

    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    ax2.imshow(obs[mid].transpose(1, 2, 0), interpolation="nearest")
    ax2.set_title(f"Zoom t={mid} | {n_px}px ({frac:.1f}%)")
    ax2.axis("off")
    fig2.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved grid: {save_path}")

    plt.show()


if __name__ == "__main__":
    cfg = BouncingBallConfig(ball_radius=7, action_dim=1)
    preview_episode(episode_len=20, seed=42, cfg=cfg)
    save_episode_gif("outputs/bouncing_ball_elastic.gif", episode_len=40, seed=42, cfg=cfg)
