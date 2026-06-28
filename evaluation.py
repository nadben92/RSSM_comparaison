"""Quantitative evaluation: ball position error vs imagination horizon.

Compare RSSM backbones on the same episodes using open-loop imagination.
Ground-truth position comes from the dataset ``positions`` array when available
(reliable under occlusion); imagined position is extracted by pixel centroid.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from data import load_dataset, load_dataset_meta
from imagine import imagine_rollout, load_model
from models.rssm import RSSM
from utils import ensure_dir, get_device, set_seed, setup_logging

logger = logging.getLogger("rssm")


@dataclass(frozen=True)
class PositionErrorResult:
    """Aggregated position-error curves over episodes."""

    mean_error: np.ndarray  # (horizon,)
    std_error: np.ndarray  # (horizon,)
    raw_errors: np.ndarray  # (n_episodes, horizon), NaN when imagined ball not detected
    episode_indices: np.ndarray  # (n_episodes,)
    occluded_mask: np.ndarray | None = None  # (horizon,) fraction occluded per step
    mean_error_occluded: float | None = None
    mean_error_visible: float | None = None


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure ``(C, H, W)`` float32 in ``[0, 1]``."""
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0 + 1e-3:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def occlusion_band_x0(occlusion_width: int, occlusion_x: int, img_size: int) -> int:
    """Left edge of the occlusion band (matches ``bouncing_ball.occlusion_rect``)."""
    if occlusion_width <= 0:
        return 0
    return occlusion_x if occlusion_x > 0 else (img_size - occlusion_width) // 2


def ignore_occlusion_band(
    chw: np.ndarray,
    occlusion_width: int,
    occlusion_x: int,
    img_size: int,
) -> np.ndarray:
    """Return a copy where fixed obstacle pixels are replaced by background.

    Red ball pixels inside the band are kept so an imagined ball can still be
    detected. Used only for error-curve metrics — not for rendering or GIFs.
    """
    if occlusion_width <= 0:
        return chw
    out = chw.copy()
    x0 = occlusion_band_x0(occlusion_width, occlusion_x, img_size)
    x1 = min(img_size, x0 + occlusion_width)
    band = out[:, :, x0:x1]
    r, g = band[0], band[1]
    ball_mask = (r > 0.25) & (r > g)
    for c in range(3):
        band[c] = np.where(ball_mask, band[c], 0.0)
    out[:, :, x0:x1] = band
    return out


def ball_position(
    frame: np.ndarray,
    threshold: float = 0.2,
    occlusion_width: int = 0,
    occlusion_x: int = 0,
    img_size: int = 32,
) -> tuple[float, float] | None:
    """Extract ball center-of-mass from a single frame.

    Args:
        frame: ``(C, H, W)`` float in ``[0, 1]`` (uint8 is normalized first).
        threshold: Max per-pixel channel deviation from background median required
            to count as ball pixel.
        occlusion_width: If > 0, obstacle band pixels are ignored (Part 2 metrics).
        occlusion_x: Band left edge; 0 = auto-center on ``img_size``.

    Returns:
        ``(x, y)`` in pixel coordinates, or ``None`` if no ball pixel is detected.
    """
    chw = normalize_frame(frame)
    if occlusion_width > 0:
        chw = ignore_occlusion_band(chw, occlusion_width, occlusion_x, img_size)
    bg = np.median(chw, axis=(1, 2), keepdims=True)
    diff = np.abs(chw - bg).max(axis=0)
    mask = diff > threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def position_distance_pixels(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    threshold: float = 0.2,
) -> float:
    """Euclidean distance between ball positions extracted from two frames."""
    pos_a = ball_position(frame_a, threshold=threshold)
    pos_b = ball_position(frame_b, threshold=threshold)
    if pos_a is None or pos_b is None:
        return float("nan")
    return float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))


def position_error_vs_gt(
    imagined_frame: np.ndarray,
    gt_pos: np.ndarray,
    threshold: float = 0.2,
    occlusion_width: int = 0,
    occlusion_x: int = 0,
    img_size: int = 32,
) -> float:
    """Error between imagined ball (pixel extraction) and ground-truth position."""
    imagined_pos = ball_position(
        imagined_frame,
        threshold=threshold,
        occlusion_width=occlusion_width,
        occlusion_x=occlusion_x,
        img_size=img_size,
    )
    if imagined_pos is None:
        return float("nan")
    return float(np.hypot(imagined_pos[0] - gt_pos[0], imagined_pos[1] - gt_pos[1]))


def positions_occluded(
    positions: np.ndarray,
    occlusion_width: int,
    occlusion_x: int,
    img_size: int,
    ball_radius: int,
) -> np.ndarray:
    """Boolean mask ``(T,)`` — True when the ball overlaps the occlusion band."""
    if occlusion_width <= 0:
        return np.zeros(len(positions), dtype=bool)
    from bouncing_ball import BouncingBallConfig, is_ball_occluded

    x0 = occlusion_x if occlusion_x > 0 else (img_size - occlusion_width) // 2
    cfg = BouncingBallConfig(
        img_size=img_size,
        ball_radius=ball_radius,
        occlusion_width=occlusion_width,
        occlusion_x=x0,
    )
    return np.array([is_ball_occluded(positions[t], cfg) for t in range(len(positions))])


def select_episode_indices(
    lengths: np.ndarray,
    context_len: int,
    horizon: int,
    n_episodes: int,
    start_frame: int = 0,
) -> np.ndarray:
    """Return the first ``n_episodes`` indices long enough for evaluation."""
    min_len = start_frame + context_len + horizon
    valid = [i for i, length in enumerate(lengths) if int(length) >= min_len]
    if len(valid) < n_episodes:
        raise ValueError(
            f"Need {n_episodes} episodes with length >= {min_len}, "
            f"found {len(valid)} (max length={int(lengths.max())})"
        )
    return np.array(valid[:n_episodes], dtype=np.int64)


@torch.no_grad()
def position_error_curve(
    model: RSSM,
    observations: np.ndarray,
    actions: np.ndarray,
    lengths: np.ndarray,
    episode_indices: Sequence[int],
    context_len: int,
    horizon: int,
    device: torch.device,
    threshold: float = 0.2,
    start_frame: int = 0,
    positions: np.ndarray | None = None,
    occlusion_width: int = 0,
    occlusion_x: int = 0,
    img_size: int = 32,
    ball_radius: int = 7,
) -> PositionErrorResult:
    """Compute mean/std position error vs imagination horizon.

    For each episode, runs open-loop imagination after a real context window.
    Ground truth uses ``positions`` when provided; otherwise falls back to pixel
    extraction on real frames (Part 1 datasets).
    """
    model.eval()
    n_episodes = len(episode_indices)
    raw_errors = np.full((n_episodes, horizon), np.nan, dtype=np.float64)
    occluded_steps = np.zeros(horizon, dtype=np.float64)
    seq_len = context_len + horizon

    for row, ep_idx in enumerate(tqdm(episode_indices, desc="Eval episodes")):
        ep_idx = int(ep_idx)
        ep_len = int(lengths[ep_idx])
        if start_frame + seq_len > ep_len:
            logger.warning("Skipping episode %d (too short)", ep_idx)
            continue

        obs_ep = observations[ep_idx, start_frame : start_frame + seq_len]
        act_ep = actions[ep_idx, start_frame : start_frame + seq_len]

        context_obs = torch.from_numpy(obs_ep[:context_len]).float().unsqueeze(0).to(device)
        context_actions = torch.from_numpy(act_ep[:context_len]).float().unsqueeze(0).to(device)
        imagine_actions = torch.from_numpy(act_ep[context_len : context_len + horizon]).float()
        imagine_actions = imagine_actions.unsqueeze(0).to(device)

        _, imagined = imagine_rollout(model, context_obs, context_actions, imagine_actions)
        imagined_np = imagined.squeeze(0).cpu().numpy()
        real_np = obs_ep[context_len : context_len + horizon]

        gt_positions = None
        if positions is not None:
            gt_positions = positions[ep_idx, start_frame + context_len : start_frame + context_len + horizon]
            if occlusion_width > 0:
                occ = positions_occluded(
                    gt_positions,
                    occlusion_width,
                    occlusion_x,
                    img_size,
                    ball_radius,
                )
                occluded_steps += occ.astype(np.float64)

        for t in range(horizon):
            if gt_positions is not None:
                raw_errors[row, t] = position_error_vs_gt(
                    imagined_np[t],
                    gt_positions[t],
                    threshold=threshold,
                    occlusion_width=occlusion_width,
                    occlusion_x=occlusion_x,
                    img_size=img_size,
                )
            else:
                raw_errors[row, t] = position_distance_pixels(
                    imagined_np[t], real_np[t], threshold=threshold
                )

    mean_error = np.nanmean(raw_errors, axis=0)
    std_error = np.nanstd(raw_errors, axis=0)
    occluded_mask = occluded_steps / max(n_episodes, 1) if occlusion_width > 0 else None

    mean_error_occluded = None
    mean_error_visible = None
    if occluded_mask is not None and np.any(occluded_mask > 0):
        occ_frac = occluded_mask >= 0.5
        if occ_frac.any():
            mean_error_occluded = float(np.nanmean(raw_errors[:, occ_frac]))
        if (~occ_frac).any():
            mean_error_visible = float(np.nanmean(raw_errors[:, ~occ_frac]))

    return PositionErrorResult(
        mean_error=mean_error,
        std_error=std_error,
        raw_errors=raw_errors,
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        occluded_mask=occluded_mask,
        mean_error_occluded=mean_error_occluded,
        mean_error_visible=mean_error_visible,
    )


def _shade_occlusion_regions(
    ax: plt.Axes,
    occluded_mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> None:
    """Grey vertical bands on the plot where imagination steps are often occluded."""
    in_band = False
    start = 1
    for i, frac in enumerate(occluded_mask):
        step = i + 1
        is_occ = frac >= threshold
        if is_occ and not in_band:
            start = step
            in_band = True
        elif not is_occ and in_band:
            ax.axvspan(start - 0.5, step - 0.5, color="grey", alpha=0.15, zorder=0)
            in_band = False
    if in_band:
        ax.axvspan(start - 0.5, len(occluded_mask) + 0.5, color="grey", alpha=0.15, zorder=0)


def plot_error_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    *,
    title: str = "Ball position error vs imagination horizon",
    xlabel: str = "Imagination step",
    ylabel: str = "Position error (pixels)",
    show_std: bool = True,
    occluded_mask: np.ndarray | None = None,
) -> Path:
    """Plot mean error curves for one or more backbones on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = None

    if occluded_mask is not None:
        _shade_occlusion_regions(ax, occluded_mask)

    for label, (mean, std) in curves.items():
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        x = np.arange(1, len(mean) + 1)
        horizons = x
        ax.plot(x, mean, linewidth=2, label=label)
        if show_std:
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    if occluded_mask is not None and np.any(occluded_mask >= 0.5):
        import matplotlib.patches as mpatches

        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color="grey", alpha=0.15, label="occlusion zone"))
        ax.legend(handles=handles)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not (occluded_mask is not None and np.any(occluded_mask >= 0.5)):
        ax.legend()
    ax.grid(True, alpha=0.3)
    if horizons is not None and len(horizons) > 0:
        ax.set_xlim(1, len(horizons))
    fig.tight_layout()

    out = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def debug_position_extraction(
    frame: np.ndarray,
    output_path: str | Path,
    *,
    threshold: float = 0.2,
    title: str | None = None,
) -> tuple[float, float] | None:
    """Save a frame with detected ball position overlaid (red dot).

    Use this to verify ``ball_position`` before trusting evaluation curves.
    """
    chw = normalize_frame(frame)
    pos = ball_position(chw, threshold=threshold)
    rgb = chw.transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    if pos is not None:
        ax.plot(pos[0], pos[1], "o", color="red", markersize=8, markeredgecolor="white")
        ax.set_title(title or f"Detected ball @ ({pos[0]:.1f}, {pos[1]:.1f})")
    else:
        ax.set_title(title or "No ball detected")
    ax.axis("off")
    fig.tight_layout()

    out = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return pos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RSSM ball position error vs imagination horizon"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more model checkpoints (e.g. gru.pt lstm.pt transformer.pt)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (must match number of checkpoints)",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2, help="Ball detection threshold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Plot mean curves only (no shaded std bands)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG filename (default: position_error_curves.png)",
    )
    parser.add_argument(
        "--debug-position",
        action="store_true",
        help="Run position extraction debug on one frame and exit",
    )
    parser.add_argument("--debug-episode-idx", type=int, default=0)
    parser.add_argument("--debug-frame-idx", type=int, default=0)
    parser.add_argument(
        "--debug-output",
        type=str,
        default=None,
        help="PNG path for debug overlay (default: output-dir/position_debug.png)",
    )
    return parser.parse_args()


def _default_labels(checkpoints: list[str]) -> list[str]:
    return [Path(p).stem for p in checkpoints]


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.output_dir)

    data_path = args.data_path
    if data_path is None:
        _, cfg0 = load_model(args.checkpoints[0], device)
        data_path = cfg0.data_path

    observations, actions, lengths, positions = load_dataset(data_path)
    meta = load_dataset_meta(data_path)
    occlusion_width = meta.get("occlusion_width", 0)
    occlusion_x = meta.get("occlusion_x", 0)
    logger.info(
        "Loaded dataset %s: %d episodes, max_len=%d%s",
        data_path,
        len(lengths),
        int(lengths.max()),
        f", positions=yes, occlusion_width={occlusion_width}" if positions is not None else "",
    )

    if args.debug_position:
        ep = args.debug_episode_idx
        frame_idx = args.debug_frame_idx
        if frame_idx >= int(lengths[ep]):
            raise ValueError(f"frame_idx={frame_idx} out of range for episode {ep}")
        debug_out = args.debug_output or str(out_dir / "position_debug.png")
        pos = debug_position_extraction(
            observations[ep, frame_idx],
            debug_out,
            threshold=args.threshold,
            title=f"Episode {ep}, frame {frame_idx}",
        )
        if positions is not None:
            logger.info("Ground-truth position: (%.2f, %.2f)", positions[ep, frame_idx, 0], positions[ep, frame_idx, 1])
        logger.info("Debug position: %s → saved %s", pos, debug_out)
        return

    labels = args.labels or _default_labels(args.checkpoints)
    if len(labels) != len(args.checkpoints):
        raise ValueError(
            f"--labels count ({len(labels)}) must match --checkpoints ({len(args.checkpoints)})"
        )

    episode_indices = select_episode_indices(
        lengths,
        context_len=args.context_len,
        horizon=args.horizon,
        n_episodes=args.n_episodes,
        start_frame=args.start_frame,
    )
    logger.info(
        "Evaluating on episodes %s (context=%d, horizon=%d)",
        episode_indices.tolist(),
        args.context_len,
        args.horizon,
    )

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_results: dict[str, PositionErrorResult] = {}
    shared_occluded_mask: np.ndarray | None = None

    for label, ckpt_path in zip(labels, args.checkpoints):
        logger.info("Loading checkpoint: %s (%s)", label, ckpt_path)
        model, config = load_model(ckpt_path, device)
        if config.data_path != data_path and args.data_path is None:
            logger.warning(
                "Checkpoint %s was trained on %s but evaluating on %s",
                label,
                config.data_path,
                data_path,
            )

        result = position_error_curve(
            model,
            observations,
            actions,
            lengths,
            episode_indices,
            context_len=args.context_len,
            horizon=args.horizon,
            device=device,
            threshold=args.threshold,
            start_frame=args.start_frame,
            positions=positions,
            occlusion_width=occlusion_width,
            occlusion_x=occlusion_x,
            img_size=config.img_size,
            ball_radius=config.ball_radius,
        )
        curves[label] = (result.mean_error, result.std_error)
        all_results[label] = result
        if result.occluded_mask is not None:
            shared_occluded_mask = result.occluded_mask

        valid_frac = np.isfinite(result.raw_errors).mean()
        msg = (
            f"{label} | mean error @ h=1: {result.mean_error[0]:.2f} px | "
            f"@ h={args.horizon}: {result.mean_error[-1]:.2f} px | "
            f"valid detections: {100.0 * valid_frac:.1f}%"
        )
        if result.mean_error_occluded is not None:
            msg += f" | occluded: {result.mean_error_occluded:.2f} px"
        if result.mean_error_visible is not None:
            msg += f" | visible: {result.mean_error_visible:.2f} px"
        logger.info(msg)

    plot_path = out_dir / (args.output or "position_error_curves.png")
    plot_error_curves(
        curves,
        plot_path,
        show_std=not args.no_std,
        occluded_mask=shared_occluded_mask,
    )
    logger.info("Saved comparison plot: %s", plot_path)

    npz_path = out_dir / "position_error_data.npz"
    save_dict: dict[str, np.ndarray] = {
        "episode_indices": episode_indices,
        "horizons": np.arange(1, args.horizon + 1),
    }
    if shared_occluded_mask is not None:
        save_dict["occluded_mask"] = shared_occluded_mask
    for label, result in all_results.items():
        key = label.replace(" ", "_")
        save_dict[f"{key}_mean"] = result.mean_error
        save_dict[f"{key}_std"] = result.std_error
        save_dict[f"{key}_raw"] = result.raw_errors
    np.savez_compressed(npz_path, **save_dict)
    logger.info("Saved raw curves: %s", npz_path)


if __name__ == "__main__":
    main()
