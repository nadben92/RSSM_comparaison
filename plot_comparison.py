"""Final comparison plots: position-error curves and KL-per-dimension across backbones."""

from __future__ import annotations

import argparse

import torch

from data import load_dataset
from evaluation import plot_error_curves, position_error_curve, select_episode_indices
from imagine import load_model, mean_kl_per_dim, plot_kl_per_dim_comparison
from utils import ensure_dir, get_device, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot backbone comparison figures")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/rssm_gru_epoch050.pt",
            "checkpoints/rssm_lstm_epoch050.pt",
            "checkpoints/rssm_transformers_epoch050.pt",
        ],
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["GRU", "LSTM", "Transformer"],
    )
    parser.add_argument("--data-path", type=str, default="data/bouncing_ball.npz")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.output_dir)

    if len(args.labels) != len(args.checkpoints):
        raise ValueError("Number of --labels must match --checkpoints")

    observations, actions, lengths = load_dataset(args.data_path)
    episode_indices = select_episode_indices(
        lengths,
        context_len=args.context_len,
        horizon=args.horizon,
        n_episodes=args.n_episodes,
    )

    curves: dict[str, tuple] = {}
    kl_by_label: dict[str, object] = {}

    ep_idx = args.episode_idx
    ep_len = int(lengths[ep_idx])
    kl_steps = args.context_len + min(args.horizon, 10)
    if kl_steps > ep_len:
        raise ValueError(f"Episode {ep_idx} too short for KL diagnostic (need {kl_steps} frames)")

    obs_ep = torch.from_numpy(observations[ep_idx, :kl_steps]).float()
    act_ep = torch.from_numpy(actions[ep_idx, :kl_steps]).float()

    for label, ckpt_path in zip(args.labels, args.checkpoints):
        logger.info("Processing %s (%s)", label, ckpt_path)
        model, config = load_model(ckpt_path, device)

        result = position_error_curve(
            model,
            observations,
            actions,
            lengths,
            episode_indices,
            context_len=args.context_len,
            horizon=args.horizon,
            device=device,
        )
        curves[label] = (result.mean_error, result.std_error)

        kl_by_label[label] = mean_kl_per_dim(
            model,
            obs_ep.unsqueeze(0).to(device),
            act_ep.unsqueeze(0).to(device),
            free_nats=config.free_nats,
        )

    error_path = out_dir / "position_error_all_backbones.png"
    plot_error_curves(
        curves,
        error_path,
        title="Ball position error vs imagination horizon",
        show_std=False,
    )
    logger.info("Saved position-error comparison: %s", error_path)

    kl_path = out_dir / "kl_per_dim_all_backbones.png"
    plot_kl_per_dim_comparison(
        kl_by_label,
        kl_path,
        free_nats=0.0,
    )
    logger.info("Saved KL comparison: %s", kl_path)


if __name__ == "__main__":
    main()
