#!/usr/bin/env python3
"""Latent traversal: vary one z dimension at a time and decode to discover what it encodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_dataset
from interpretability.utils_interp import (
    active_dimensions,
    ball_position_from_tensor,
    collect_posterior_latents,
    decode_frame,
    displacement_for_dim,
    ensure_output_dir,
    evenly_spaced_starts,
    load_checkpoint,
    load_episode,
    mean_kl_per_dim,
    posterior_stats_at_reference,
    reference_state_from_context,
    sweep_values_for_dim,
)
from utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent dimension traversal (without_occult checkpoints)")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/without_occult/rssm_gru_epoch050.pt",
        help="Path to trained RSSM checkpoint",
    )
    p.add_argument("--data", default="data/bouncing_ball.npz")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--context-len", type=int, default=10)
    p.add_argument("--start", type=int, default=20, help="Reference context start frame for grid")
    p.add_argument("--output-dir", default="interpretability/outputs/gru")
    p.add_argument("--top-k", type=int, default=8, help="Traverse top-K active dims by KL")
    p.add_argument(
        "--include-dims",
        type=int,
        nargs="*",
        default=[19],
        help="Extra latent dims to include (merged with top-K, default: 19)",
    )
    p.add_argument("--n-values", type=int, default=9, help="Grid columns (z sweep points)")
    p.add_argument(
        "--range-mode",
        choices=("temporal", "sigma"),
        default="temporal",
        help="temporal: min/max over episode z_traj; sigma: mu_q ± sigma_scale·sigma_q",
    )
    p.add_argument("--temporal-margin", type=float, default=0.2, help="Fractional margin beyond observed min/max")
    p.add_argument("--sigma-scale", type=float, default=3.0, help="Used only with --range-mode sigma")
    p.add_argument(
        "--displacement-refs",
        type=int,
        default=5,
        help="Reference contexts averaged for displacement bar chart (1 = single ref)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _format_value(val: float, span: float) -> str:
    if span >= 20:
        return f"{val:.0f}"
    if span >= 2:
        return f"{val:.1f}"
    return f"{val:.2f}"


@torch.no_grad()
def run_traversal(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    _, _, lengths, _ = load_dataset(args.data)
    ep_len = int(lengths[args.episode])

    # Full-episode posterior trajectory for temporal ranges
    obs_ep, act_ep, _ = load_episode(args.data, args.episode, 0, ep_len)
    obs_ep_t = torch.from_numpy(obs_ep).unsqueeze(0).to(device)
    act_ep_t = torch.from_numpy(act_ep).unsqueeze(0).to(device)
    z_traj, _ = collect_posterior_latents(model, obs_ep_t, act_ep_t)

    # Reference context for grid visualization
    obs_ctx, act_ctx, _ = load_episode(args.data, args.episode, args.start, args.context_len)
    obs_t = torch.from_numpy(obs_ctx).unsqueeze(0).to(device)
    act_t = torch.from_numpy(act_ctx).unsqueeze(0).to(device)

    kl = mean_kl_per_dim(model, obs_t, act_t)
    dims = active_dimensions(kl, top_k=args.top_k)
    if args.include_dims:
        extra = [d for d in args.include_dims if d not in dims]
        if extra:
            dims = np.concatenate([dims, np.array(extra, dtype=np.int64)])
        dims = np.array(sorted(dims, key=lambda d: kl[d], reverse=True))

    h, z_ref, _ = reference_state_from_context(model, obs_t, act_t)
    mu_q, sigma_q = posterior_stats_at_reference(model, obs_t, act_t)

    out_dir = ensure_output_dir(args.output_dir)
    stem = Path(args.checkpoint).stem
    n = args.n_values
    positions_log: list[dict] = []
    sweep_ranges: dict[int, tuple[float, float]] = {}

    fig, axes = plt.subplots(len(dims), n, figsize=(1.6 * n, 1.8 * len(dims)))
    if len(dims) == 1:
        axes = np.expand_dims(axes, 0)

    for row, dim in enumerate(dims):
        values = sweep_values_for_dim(
            int(dim),
            range_mode=args.range_mode,
            n_values=n,
            z_traj=z_traj if args.range_mode == "temporal" else None,
            mu_q=mu_q if args.range_mode == "sigma" else None,
            sigma_q=sigma_q if args.range_mode == "sigma" else None,
            sigma_scale=args.sigma_scale,
            temporal_margin=args.temporal_margin,
        )
        sweep_ranges[int(dim)] = (float(values[0]), float(values[-1]))
        span = values[-1] - values[0]
        ref_pos = ball_position_from_tensor(decode_frame(model, h, z_ref))

        for col, val in enumerate(values):
            z_var = z_ref.clone()
            z_var[0, dim] = val
            frame = decode_frame(model, h, z_var)[0]
            pos = ball_position_from_tensor(frame)

            ax = axes[row, col]
            ax.imshow(frame.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"z[{dim}]\nKL={kl[dim]:.3f}", fontsize=8)
            if row == 0:
                ax.set_title(_format_value(float(val), span), fontsize=8)

            positions_log.append(
                {
                    "dim": int(dim),
                    "value": float(val),
                    "ball_x": None if pos is None else pos[0],
                    "ball_y": None if pos is None else pos[1],
                    "ref_x": None if ref_pos is None else ref_pos[0],
                    "ref_y": None if ref_pos is None else ref_pos[1],
                }
            )

    mode_label = "temporal min/max" if args.range_mode == "temporal" else "posterior σ"
    fig.suptitle(
        f"Latent traversal — {stem}\n"
        f"Each row: one dimension z[i] varied ({mode_label}); "
        f"columns: sweep values of z[i] (h and other dims fixed)",
        fontsize=11,
    )
    fig.tight_layout()
    grid_path = out_dir / f"latent_traversal_{stem}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Displacement bar chart — optionally averaged over multiple reference contexts
    ref_starts = evenly_spaced_starts(ep_len, args.context_len, args.displacement_refs)
    delta_x, delta_y, labels = [], [], []
    for dim in dims:
        dxs: list[float] = []
        dys: list[float] = []
        values = sweep_values_for_dim(
            int(dim),
            range_mode=args.range_mode,
            n_values=n,
            z_traj=z_traj if args.range_mode == "temporal" else None,
            mu_q=mu_q if args.range_mode == "sigma" else None,
            sigma_q=sigma_q if args.range_mode == "sigma" else None,
            sigma_scale=args.sigma_scale,
            temporal_margin=args.temporal_margin,
        )
        for ref_start in ref_starts:
            obs_r, act_r, _ = load_episode(args.data, args.episode, ref_start, args.context_len)
            obs_r_t = torch.from_numpy(obs_r).unsqueeze(0).to(device)
            act_r_t = torch.from_numpy(act_r).unsqueeze(0).to(device)
            h_r, z_ref_r, _ = reference_state_from_context(model, obs_r_t, act_r_t)
            dx, dy = displacement_for_dim(model, h_r, z_ref_r, int(dim), values)
            dxs.append(dx)
            dys.append(dy)
        delta_x.append(float(np.mean(dxs)))
        delta_y.append(float(np.mean(dys)))
        labels.append(f"z[{dim}]")

    fig2, ax2 = plt.subplots(figsize=(8, max(3, len(dims) * 0.4)))
    x_pos = np.arange(len(dims))
    w = 0.35
    ax2.bar(x_pos - w / 2, delta_x, w, label="Δx (pixels)", color="steelblue")
    ax2.bar(x_pos + w / 2, delta_y, w, label="Δy (pixels)", color="coral")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Ball displacement across sweep")
    ax2.legend()
    ref_note = f"avg over {len(ref_starts)} refs" if len(ref_starts) > 1 else "single ref"
    ax2.set_title(f"Traversal-induced ball movement ({mode_label}, {ref_note})")
    disp_path = out_dir / f"latent_traversal_displacement_{stem}.png"
    fig2.savefig(disp_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    np.savez(
        out_dir / f"latent_traversal_data_{stem}.npz",
        kl_per_dim=kl,
        active_dims=dims,
        positions=positions_log,
        range_mode=args.range_mode,
        sweep_ranges=sweep_ranges,
    )
    print(f"Saved grid: {grid_path}")
    print(f"Saved displacement chart: {disp_path}")
    print(f"Range mode: {args.range_mode}")
    print(f"Active dims traversed: {dims.tolist()}")
    for dim in dims:
        lo, hi = sweep_ranges[int(dim)]
        print(f"  z[{dim}] sweep: [{lo:.2f}, {hi:.2f}]  (observed [{z_traj[:, dim].min():.2f}, {z_traj[:, dim].max():.2f}])")
    print(f"KL per dim (top): {', '.join(f'z[{d}]={kl[d]:.4f}' for d in dims[:5])}")


if __name__ == "__main__":
    run_traversal(parse_args())
