#!/usr/bin/env python3
"""Latent structure: effective dimensionality, correlations, temporal alignment with physics."""

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

from interpretability.utils_interp import (
    active_dimensions,
    collect_posterior_latents,
    ensure_output_dir,
    load_checkpoint,
    load_episode,
    mean_kl_per_dim,
    velocity_from_positions,
)
from utils import get_device, set_seed


DEFAULT_CHECKPOINTS = {
    "gru": "checkpoints/without_occult/rssm_gru_epoch050.pt",
    "lstm": "checkpoints/without_occult/rssm_lstm_epoch050.pt",
    "transformer": "checkpoints/without_occult/rssm_transformers_epoch050.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent structure analysis")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/without_occult/rssm_gru_epoch050.pt",
        help="Primary checkpoint (correlation + time series)",
    )
    p.add_argument(
        "--compare-backbones",
        action="store_true",
        help="Compare KL allocation across GRU/LSTM/Transformer without_occult checkpoints",
    )
    p.add_argument("--data", default="data/bouncing_ball.npz")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=80)
    p.add_argument("--start", type=int, default=10)
    p.add_argument("--output-dir", default="interpretability/outputs/gru")
    p.add_argument("--kl-threshold", type=float, default=0.01, help="Min KL for 'active' dim")
    p.add_argument("--top-k-trace", type=int, default=4, help="Dims to trace vs physics")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def kl_profile(model, obs_t, act_t, kl_threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    kl = mean_kl_per_dim(model, obs_t, act_t)
    active = np.where(kl >= kl_threshold)[0]
    n_effective = len(active)
    return kl, active, n_effective


@torch.no_grad()
def run_structure(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_output_dir(args.output_dir)

    obs, act, pos = load_episode(args.data, args.episode, args.start, args.seq_len)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    act_t = torch.from_numpy(act).unsqueeze(0).to(device)

    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    stem = Path(args.checkpoint).stem

    kl, active_idx, n_eff = kl_profile(model, obs_t, act_t, args.kl_threshold)
    trace_dims = active_dimensions(kl, top_k=args.top_k_trace)

    z_seq, _ = collect_posterior_latents(model, obs_t, act_t)
    corr_dims = active_idx if len(active_idx) >= 2 else trace_dims
    z_corr = z_seq[:, corr_dims]

    # --- 1. KL bar + effective dim count ---
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.bar(range(len(kl)), kl, color="steelblue", width=0.8)
    ax1.axhline(args.kl_threshold, color="red", ls="--", label=f"threshold={args.kl_threshold}")
    ax1.set_xlabel("Latent dimension")
    ax1.set_ylabel("Mean KL")
    ax1.set_title(f"KL per dimension — {stem} ({n_eff} effective dims)")
    ax1.legend()
    kl_path = out_dir / f"structure_kl_{stem}.png"
    fig1.savefig(kl_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # --- 2. Correlation heatmap (active dims) ---
    if z_corr.shape[1] >= 2:
        corr = np.corrcoef(z_corr.T)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        labels = [f"z[{d}]" for d in corr_dims]
        ax2.set_xticks(range(len(labels)))
        ax2.set_yticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=90, fontsize=7)
        ax2.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax2, fraction=0.046)
        ax2.set_title("Posterior z correlation (active dims)")
        corr_path = out_dir / f"structure_correlation_{stem}.png"
        fig2.savefig(corr_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
    else:
        corr = np.array([[1.0]])
        corr_path = None

    # --- 3. Temporal traces vs physics ---
    vel = velocity_from_positions(pos)
    fig3, axes = plt.subplots(len(trace_dims) + 2, 1, figsize=(10, 2 * (len(trace_dims) + 2)), sharex=True)
    t = np.arange(len(pos))

    axes[0].plot(t, pos[:, 0], label="x", color="steelblue")
    axes[0].plot(t, pos[:, 1], label="y", color="coral")
    axes[0].set_ylabel("Position (px)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Ground-truth ball trajectory")

    axes[1].plot(t, vel[:, 0], label="vx", color="steelblue")
    axes[1].plot(t, vel[:, 1], label="vy", color="coral")
    axes[1].set_ylabel("Velocity (px/step)")
    axes[1].legend(loc="upper right")

    for i, dim in enumerate(trace_dims):
        ax = axes[i + 2]
        ax.plot(t, z_seq[:, dim], color="green")
        ax.set_ylabel(f"z[{dim}]")
        if i == len(trace_dims) - 1:
            ax.set_xlabel("Time step")

    fig3.suptitle(f"Latent vs physics — {stem}", fontsize=11)
    fig3.tight_layout()
    trace_path = out_dir / f"structure_temporal_{stem}.png"
    fig3.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # --- 4. Optional backbone comparison ---
    compare_path = None
    if args.compare_backbones:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        for name, ckpt in DEFAULT_CHECKPOINTS.items():
            if not Path(ckpt).exists():
                print(f"Skipping missing checkpoint: {ckpt}")
                continue
            m, _ = load_checkpoint(ckpt, device)
            m.eval()
            k, _, ne = kl_profile(m, obs_t, act_t, args.kl_threshold)
            ax4.plot(range(len(k)), k, label=f"{name} ({ne} active)", alpha=0.85)
        ax4.axhline(args.kl_threshold, color="gray", ls="--")
        ax4.set_xlabel("Latent dimension")
        ax4.set_ylabel("Mean KL")
        ax4.set_title("KL allocation across backbones (without_occult)")
        ax4.legend()
        compare_path = out_dir / "structure_kl_backbone_comparison.png"
        fig4.savefig(compare_path, dpi=150, bbox_inches="tight")
        plt.close(fig4)

    np.savez(
        out_dir / f"structure_data_{stem}.npz",
        kl_per_dim=kl,
        active_dims=active_idx,
        n_effective=n_eff,
        correlation=corr,
    )

    print(f"Effective dimensions (KL ≥ {args.kl_threshold}): {n_eff} / {len(kl)}")
    print(f"Saved KL plot: {kl_path}")
    if corr_path:
        print(f"Saved correlation heatmap: {corr_path}")
    if trace_path:
        print(f"Saved temporal traces: {trace_path}")
    if compare_path:
        print(f"Saved backbone comparison: {compare_path}")


if __name__ == "__main__":
    run_structure(parse_args())
