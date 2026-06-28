#!/usr/bin/env python3
"""Causal ablation: clamp z[i] to prior mean during imagination and measure position error."""

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
    ensure_output_dir,
    imagine_with_z_hook,
    load_checkpoint,
    load_episode,
    mean_kl_per_dim,
)
from evaluation import position_error_vs_gt
from data import load_dataset
from utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal latent ablation during imagination")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/without_occult/rssm_gru_epoch050.pt",
    )
    p.add_argument("--data", default="data/bouncing_ball.npz")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--context-len", type=int, default=10)
    p.add_argument("--imagine-len", type=int, default=15)
    p.add_argument("--start", type=int, default=20)
    p.add_argument("--output-dir", default="interpretability/outputs/gru")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--n-episodes", type=int, default=20, help="Episodes averaged for ablation ranking")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def mean_position_error(
    model,
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
    imagine_actions: torch.Tensor,
    gt_positions: np.ndarray,
    ablate_dim: int | None = None,
) -> float:
    """Mean ball position error (pixels) over imagination horizon."""
    frames = imagine_with_z_hook(
        model,
        context_obs,
        context_actions,
        imagine_actions,
        ablate_dim=ablate_dim,
    )
    errors = []
    for t in range(frames.size(1)):
        err = position_error_vs_gt(
            frames[0, t].cpu().numpy(),
            gt_positions[t],
        )
        if not np.isnan(err):
            errors.append(err)
    return float(np.mean(errors)) if errors else float("nan")


@torch.no_grad()
def run_ablation(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    out_dir = ensure_output_dir(args.output_dir)
    stem = Path(args.checkpoint).stem

    # KL from first episode slice
    obs0, act0, _ = load_episode(args.data, 0, args.start, args.context_len)
    kl = mean_kl_per_dim(
        model,
        torch.from_numpy(obs0).unsqueeze(0).to(device),
        torch.from_numpy(act0).unsqueeze(0).to(device),
    )
    dims = active_dimensions(kl, top_k=args.top_k)

    obs_all, _, lengths, _ = load_dataset(args.data)
    n_episodes = min(args.n_episodes, len(lengths))

    baseline_errors: list[float] = []
    ablation_delta: dict[int, list[float]] = {int(d): [] for d in dims}

    for ep in range(n_episodes):
        start = args.start + ep * 5
        ctx_len = args.context_len
        img_len = args.imagine_len
        obs, act, pos = load_episode(args.data, ep, start, ctx_len + img_len)

        ctx_obs = torch.from_numpy(obs[:ctx_len]).unsqueeze(0).to(device)
        ctx_act = torch.from_numpy(act[:ctx_len]).unsqueeze(0).to(device)
        img_act = torch.from_numpy(act[ctx_len : ctx_len + img_len]).unsqueeze(0).to(device)
        gt = pos[ctx_len : ctx_len + img_len]

        baseline = mean_position_error(model, ctx_obs, ctx_act, img_act, gt, ablate_dim=None)
        baseline_errors.append(baseline)

        for dim in dims:
            ablated = mean_position_error(model, ctx_obs, ctx_act, img_act, gt, ablate_dim=int(dim))
            ablation_delta[int(dim)].append(ablated - baseline)

    baseline_mean = float(np.nanmean(baseline_errors))
    causal_impact = {d: float(np.nanmean(deltas)) for d, deltas in ablation_delta.items()}

    # Bar charts: causal importance vs KL
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    dim_labels = [f"z[{d}]" for d in dims]
    impacts = [causal_impact[int(d)] for d in dims]
    kls = [kl[d] for d in dims]

    ax1.bar(dim_labels, impacts, color="darkorange")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_ylabel("Δ position error (ablated − baseline)")
    ax1.set_title("Causal importance (higher = more harmful to ablate)")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(dim_labels, kls, color="mediumpurple")
    ax2.set_ylabel("Mean KL per dimension")
    ax2.set_title("KL usage (reference)")
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Ablation vs KL — {stem} (baseline err={baseline_mean:.2f}px)", fontsize=11)
    fig.tight_layout()
    chart_path = out_dir / f"ablation_causal_vs_kl_{stem}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Ranked table
    ranked = sorted(causal_impact.items(), key=lambda x: x[1], reverse=True)
    print(f"Baseline mean position error: {baseline_mean:.3f} px")
    print("Causal ranking (Δ error when z[i] ← prior mean):")
    for dim, delta in ranked:
        print(f"  z[{dim:2d}]: Δ={delta:+.3f} px   KL={kl[dim]:.4f}")

    np.savez(
        out_dir / f"ablation_data_{stem}.npz",
        kl_per_dim=kl,
        active_dims=dims,
        causal_impact=np.array([causal_impact[int(d)] for d in dims]),
        baseline_error=baseline_mean,
    )
    print(f"Saved chart: {chart_path}")


if __name__ == "__main__":
    run_ablation(parse_args())
