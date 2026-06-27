"""Synthetic bouncing-ball episodes (numpy only, no Gymnasium).

Billiard-style world in a square: elastic wall bounces plus optional scalar actions.
``action_dim=1``: ``+1`` accelerates along velocity, ``-1`` decelerates (random per step).
Set ``zero_actions=True`` for pure free physics with actions stored as ``0``.

Optional vertical occlusion band: the ball is hidden when it passes behind a fixed
opaque rectangle. Ground-truth ``positions`` are stored separately for evaluation
during occlusion (Part 2 memory task).

Generates the same on-disk layout as ``data.collect_episodes`` (uint8 obs on disk,
normalized to float32 when loaded via ``data.load_dataset``).
"""

from __future__ import annotations

import argparse
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
    action_dim: int = 1  # +1 accel / -1 decel along velocity (or 0 if zero_actions)
    zero_actions: bool = False  # True = no force; actions recorded as 0
    action_scale: float = 0.2  # Δ|v| per step when action=±1
    max_speed: float = 6.0
    bg_color: tuple[int, int, int] = (0, 0, 0)
    ball_color: tuple[int, int, int] = (255, 64, 64)
    # Occlusion band (disabled when occlusion_width <= 0 → Part 1 behavior)
    occlusion_width: int = 0
    occlusion_x: int = 0  # left edge of band; 0 = auto-center
    occlusion_color: tuple[int, int, int] = (80, 80, 80)

    @property
    def occlusion_enabled(self) -> bool:
        return self.occlusion_width > 0


def default_ball_radius(img_size: int, target_fraction: float = 0.15) -> int:
    """Heuristic radius so disk area ≈ ``target_fraction`` of the image."""
    area = target_fraction * img_size * img_size
    return max(2, int(round((area / np.pi) ** 0.5)))


def occlusion_rect(cfg: BouncingBallConfig) -> tuple[int, int, int, int] | None:
    """Return ``(x0, x1, y0, y1)`` pixel bounds of the occlusion band, or ``None``."""
    if not cfg.occlusion_enabled:
        return None
    x0 = cfg.occlusion_x if cfg.occlusion_x > 0 else (cfg.img_size - cfg.occlusion_width) // 2
    x1 = min(cfg.img_size, x0 + cfg.occlusion_width)
    return x0, x1, 0, cfg.img_size


def is_ball_occluded(pos: np.ndarray, cfg: BouncingBallConfig) -> bool:
    """True when the ball center passes through the occlusion band (x axis).

    Uses center-in-band rather than disk overlap so occlusion is intermittent
    (ball disappears on crossing, reappears outside) rather than nearly always on.
    """
    rect = occlusion_rect(cfg)
    if rect is None:
        return False
    x0, x1, _, _ = rect
    cx = float(pos[0])
    return x0 <= cx <= x1


def _draw_disk(
    frame: np.ndarray,
    center: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
) -> None:
    """Paint a filled disk onto ``frame`` ``(H, W, 3)`` in place."""
    s = frame.shape[0]
    cx, cy = float(center[0]), float(center[1])
    y0, y1 = max(0, int(np.floor(cy - radius))), min(s, int(np.ceil(cy + radius)) + 1)
    x0, x1 = max(0, int(np.floor(cx - radius))), min(s, int(np.ceil(cx + radius)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    frame[y0:y1, x0:x1][mask] = np.array(color, dtype=np.uint8)


def _draw_occlusion(frame: np.ndarray, cfg: BouncingBallConfig) -> None:
    """Paint the fixed opaque occlusion band on top of the scene."""
    rect = occlusion_rect(cfg)
    if rect is None:
        return
    x0, x1, y0, y1 = rect
    frame[y0:y1, x0:x1] = np.array(cfg.occlusion_color, dtype=np.uint8)


def render_frame(pos: np.ndarray, cfg: BouncingBallConfig) -> np.ndarray:
    """Draw the ball and optional occlusion band on a square canvas.

    The ball is drawn first; the occlusion rectangle is painted on top so the
    ball disappears when it passes behind the obstacle.

    Args:
        pos: Ball center ``(x, y)`` in pixel coordinates.
        cfg: Rendering configuration.

    Returns:
        ``(C, H, W)`` uint8 RGB.
    """
    s = cfg.img_size
    frame = np.zeros((s, s, 3), dtype=np.uint8)
    frame[..., :] = np.array(cfg.bg_color, dtype=np.uint8)
    _draw_disk(frame, pos, cfg.ball_radius, cfg.ball_color)
    _draw_occlusion(frame, cfg)
    return frame.transpose(2, 0, 1)


def ball_pixel_stats(frame_chw: np.ndarray, cfg: BouncingBallConfig) -> tuple[int, float]:
    """Count ball pixels (red-dominant) and image fraction."""
    s = cfg.img_size
    rgb = frame_chw.transpose(1, 2, 0)
    ball_mask = (rgb[..., 0] > 64) & (rgb[..., 0] > rgb[..., 1])
    count = int(ball_mask.sum())
    fraction = 100.0 * count / (s * s)
    return count, fraction


def _sample_initial_state(
    rng: np.random.Generator,
    cfg: BouncingBallConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random ball position and velocity."""
    r = cfg.ball_radius
    s = cfg.img_size
    margin = r + 1
    pos = rng.uniform(margin, s - margin, size=2).astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    speed = cfg.initial_speed * rng.uniform(0.7, 1.3)
    vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed
    return pos, vel


def _sample_action(rng: np.random.Generator, cfg: BouncingBallConfig) -> np.ndarray:
    """Sample action for one step: 0 (free physics) or random ±1 throttle."""
    if cfg.zero_actions:
        return np.zeros(cfg.action_dim, dtype=np.float32)
    value = float(rng.choice([-1.0, 1.0]))
    return np.array([value], dtype=np.float32)


def _apply_action(
    vel: np.ndarray,
    action: float,
    cfg: BouncingBallConfig,
) -> np.ndarray:
    """Change speed along current velocity direction; no effect if speed ≈ 0."""
    speed = float(np.linalg.norm(vel))
    if speed < 1e-6:
        return vel
    v_hat = vel / speed
    new_speed = np.clip(speed + action * cfg.action_scale, 0.0, cfg.max_speed)
    return v_hat * new_speed


def _bounce_walls(
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: BouncingBallConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic reflection off axis-aligned walls."""
    r = cfg.ball_radius
    s = cfg.img_size

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


def _step_physics(
    pos: np.ndarray,
    vel: np.ndarray,
    action: float,
    cfg: BouncingBallConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply action, integrate position, elastic wall bounces."""
    vel = _apply_action(vel, action, cfg)
    pos = pos + vel
    pos, vel = _bounce_walls(pos, vel, cfg)
    return pos, vel


def roll_out_episode(
    episode_len: int,
    rng: np.random.Generator,
    cfg: BouncingBallConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one episode.

    ``obs[t]`` is the frame after step ``t``; ``actions[t]`` is the throttle
    applied before that step (0 if ``zero_actions``, else ``{-1, +1}``).
    ``positions[t]`` is the true ball center ``(x, y)`` at frame ``t``, even
    when the ball is hidden behind the occlusion band.

    Returns:
        observations ``(T, C, H, W)`` uint8,
        actions ``(T, action_dim)`` float32 in ``{-1, +1}``,
        positions ``(T, 2)`` float32.
    """
    pos, vel = _sample_initial_state(rng, cfg)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    pos_list: list[np.ndarray] = []

    for _ in range(episode_len):
        action = _sample_action(rng, cfg)
        pos_list.append(pos.copy())
        obs_list.append(render_frame(pos, cfg))
        act_list.append(action)
        pos, vel = _step_physics(pos, vel, float(action[0]), cfg)

    return (
        np.stack(obs_list, axis=0),
        np.stack(act_list, axis=0),
        np.stack(pos_list, axis=0).astype(np.float32),
    )


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
    obs, actions, positions = roll_out_episode(episode_len, rng, cfg)

    frames: list[np.ndarray] = []
    for t in range(episode_len):
        rgb = obs[t].transpose(1, 2, 0)
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        frames.append(rgb)

    out = ensure_dir(Path(path).parent) / Path(path).name
    imageio.mimsave(out, frames, fps=fps, loop=0)

    n_px, frac = ball_pixel_stats(obs[0], cfg)
    occ_frac = 100.0 * sum(is_ball_occluded(positions[t], cfg) for t in range(episode_len)) / episode_len
    print(f"GIF saved: {out}")
    print(f"  ball_radius={cfg.ball_radius} → {n_px}px ({frac:.1f}%) at native 32x32")
    if cfg.occlusion_enabled:
        rect = occlusion_rect(cfg)
        print(f"  occlusion band x=[{rect[0]}, {rect[1]}) | occluded frames: {occ_frac:.1f}%")
    print(f"  actions ±1 along velocity | {episode_len} frames @ {fps} fps")
    print(f"  sample actions: {actions[0]}, {actions[1]}")
    return out


def collect_episodes(
    num_episodes: int,
    max_steps: int,
    seed: int = 42,
    cfg: BouncingBallConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``num_episodes`` synthetic rollouts.

    Returns:
        observations ``(N, T, C, H, W)`` uint8,
        actions ``(N, T, action_dim)`` float32 in ``{-1, +1}``,
        lengths ``(N,)`` (all equal to ``max_steps``),
        positions ``(N, T, 2)`` float32 ground-truth ball centers.
    """
    cfg = cfg or BouncingBallConfig()
    rng = np.random.default_rng(seed)
    n, t = num_episodes, max_steps
    c, s = 3, cfg.img_size
    adim = cfg.action_dim

    obs_array = np.zeros((n, t, c, s, s), dtype=np.uint8)
    act_array = np.zeros((n, t, adim), dtype=np.float32)
    pos_array = np.zeros((n, t, 2), dtype=np.float32)
    lengths = np.full(n, t, dtype=np.int64)

    for ep in tqdm(range(num_episodes), desc="Generating bouncing-ball episodes"):
        obs, acts, positions = roll_out_episode(t, rng, cfg)
        obs_array[ep] = obs
        act_array[ep] = acts
        pos_array[ep] = positions

    return obs_array, act_array, lengths, pos_array


def preview_episode(
    episode_len: int = 20,
    seed: int = 42,
    cfg: BouncingBallConfig | None = None,
    save_path: str | None = None,
) -> None:
    """Visualize one short episode before large-scale collection."""
    cfg = cfg or BouncingBallConfig()
    rng = np.random.default_rng(seed)
    obs, actions, positions = roll_out_episode(episode_len, rng, cfg)

    ncols = min(episode_len, 10)
    nrows = (episode_len + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows))
    axes = np.atleast_2d(axes)
    for t in range(episode_len):
        ax = axes[t // ncols, t % ncols]
        rgb = obs[t].transpose(1, 2, 0)
        ax.imshow(rgb, interpolation="nearest")
        n_px, frac = ball_pixel_stats(obs[t], cfg)
        occ = is_ball_occluded(positions[t], cfg)
        title = f"t={t}\n{n_px}px ({frac:.1f}%)"
        if occ:
            title += "\n[OCCLUDED]"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for t in range(episode_len, nrows * ncols):
        axes[t // ncols, t % ncols].axis("off")

    occ_title = ""
    if cfg.occlusion_enabled:
        rect = occlusion_rect(cfg)
        occ_frac = 100.0 * sum(is_ball_occluded(positions[t], cfg) for t in range(episode_len)) / episode_len
        occ_title = f" | occlusion x=[{rect[0]},{rect[1]}) {occ_frac:.0f}% occluded"
    fig.suptitle(
        f"Bouncing ball | {cfg.img_size}x{cfg.img_size} r={cfg.ball_radius}{occ_title}",
        fontsize=11,
    )
    fig.tight_layout()

    mid = episode_len // 2
    n_px, frac = ball_pixel_stats(obs[mid], cfg)
    occ_frac = 100.0 * sum(is_ball_occluded(positions[t], cfg) for t in range(episode_len)) / episode_len
    print("=" * 60)
    print("INSPECT BEFORE COLLECT / TRAIN")
    print(f"  img_size={cfg.img_size}, ball_radius={cfg.ball_radius}")
    print(f"  frame {mid}: ball pixels = {n_px} ({frac:.2f}% of image)")
    if cfg.occlusion_enabled:
        rect = occlusion_rect(cfg)
        print(f"  occlusion band: x=[{rect[0]}, {rect[1]}), color={cfg.occlusion_color}")
        print(f"  occluded frames: {occ_frac:.1f}% (target: regular crossings)")
    print(f"  actions: +1=accel / -1=decel along velocity")
    print(f"  sample actions[0:3]: {actions[:3].flatten()}")
    print("  target ~15-20% ball area → ball_radius=7 or 8")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bouncing ball preview / GIF")
    parser.add_argument("--episode-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ball-radius", type=int, default=7)
    parser.add_argument("--occlusion-width", type=int, default=0, help="0 = disabled (Part 1)")
    parser.add_argument("--occlusion-x", type=int, default=0, help="Band left edge; 0 = auto-center")
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--save-grid", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BouncingBallConfig(
        ball_radius=args.ball_radius,
        action_dim=1,
        occlusion_width=args.occlusion_width,
        occlusion_x=args.occlusion_x,
    )
    preview_episode(
        episode_len=min(args.episode_len, 20),
        seed=args.seed,
        cfg=cfg,
        save_path=args.save_grid,
    )
    if args.save_gif:
        save_episode_gif(args.save_gif, episode_len=args.episode_len, seed=args.seed, cfg=cfg)


if __name__ == "__main__":
    main()
