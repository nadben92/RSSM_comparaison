"""Training loop for the RSSM world model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from config import Config
from data import DatasetStats, build_dataloader, collect_and_save
from models.rssm import RSSM
from utils import current_beta, ensure_dir, get_device, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RSSM world model")
    parser.add_argument(
        "--env-name",
        type=str,
        default="bouncing_ball",
        help="bouncing_ball | bouncing_ball_obstacles | Gymnasium env id",
    )
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=25)
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--drop-last", action="store_true", help="Drop incomplete batches")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument(
        "--anneal-steps",
        type=int,
        default=0,
        help="Fixed beta-anneal steps (0 = auto from --anneal-fraction)",
    )
    parser.add_argument(
        "--anneal-fraction",
        type=float,
        default=0.3,
        help="Fraction of total steps for beta ramp when --anneal-steps=0",
    )
    parser.add_argument("--free-nats", type=float, default=0.0,
                        help="Optional floor on balanced KL (0 = KL balancing only)")
    parser.add_argument(
        "--kl-balance",
        type=float,
        default=0.8,
        help="DreamerV2 KL balancing alpha (prior term weight)",
    )
    parser.add_argument(
        "--lambda-motion",
        type=float,
        default=5.0,
        help="Motion weighting for recon loss (higher = prioritize moving pixels)",
    )
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, default="data/bouncing_ball.npz")
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=1.0,
        help="Center crop for Gym envs only (1.0=no crop); ignored for bouncing_ball",
    )
    parser.add_argument(
        "--ball-radius",
        type=int,
        default=7,
        help="Ball radius for bouncing_ball envs (~15%% at r=7; use r=5 for obstacles variant)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--force-collect", action="store_true", help="Re-collect data")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="rssm-worldmodel")
    parser.add_argument("--wandb-run-name", type=str, default="")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        chunk_stride=args.chunk_stride,
        drop_last=args.drop_last,
        lr=args.lr,
        beta_max=args.beta_max,
        anneal_steps=args.anneal_steps,
        anneal_fraction=args.anneal_fraction,
        free_nats=args.free_nats,
        kl_balance_scale=args.kl_balance,
        lambda_motion=args.lambda_motion,
        grad_clip=args.grad_clip,
        seed=args.seed,
        data_path=args.data_path,
        crop_ratio=args.crop_ratio,
        ball_radius=args.ball_radius,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def resolve_anneal_steps(config: Config, stats: DatasetStats) -> int:
    """Resolve beta-anneal length: fixed if set, else a fraction of total steps."""
    if config.anneal_steps > 0:
        return config.anneal_steps
    return max(1, int(stats.total_steps * config.anneal_fraction))


def log_training_plan(
    config: Config,
    stats: DatasetStats,
    anneal_steps: int,
) -> None:
    """Log dataset and schedule summary before the first optimization step."""
    logger = setup_logging()
    min_recommended = config.recommended_min_episodes()

    logger.info("=== Training plan ===")
    logger.info(
        "Dataset: %d episodes | max_len=%d | %d chunks (seq_len=%d, stride=%d)",
        stats.num_episodes,
        stats.max_episode_len,
        stats.num_chunks,
        config.seq_len,
        config.chunk_stride,
    )
    logger.info(
        "Loader: batch_size=%d | drop_last=%s | steps/epoch=%d | total_steps=%d",
        stats.batch_size,
        stats.drop_last,
        stats.steps_per_epoch,
        stats.total_steps,
    )
    logger.info(
        "Beta: beta_max=%.2f | anneal over %d steps (%.0f%% of training)",
        config.beta_max,
        anneal_steps,
        100.0 * anneal_steps / max(stats.total_steps, 1),
    )

    min_chunks = config.batch_size * 10
    if stats.num_chunks < min_chunks:
        logger.warning(
            "Only %d chunks for batch_size=%d (~%d batches/epoch). "
            "Recommend >= %d chunks: try --num-episodes %d --seq-len 25",
            stats.num_chunks,
            config.batch_size,
            stats.steps_per_epoch,
            min_chunks,
            min_recommended,
        )
    if stats.num_episodes < min_recommended:
        logger.warning(
            "num_episodes=%d is below recommended minimum %d for stable training",
            stats.num_episodes,
            min_recommended,
        )


def save_checkpoint(
    path: Path,
    model: RSSM,
    optimizer: Adam,
    epoch: int,
    global_step: int,
    config: Config,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        },
        path,
    )


def train(config: Config, force_collect: bool = False) -> None:
    """Main training loop."""
    logger = setup_logging()
    set_seed(config.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    collect_and_save(config, force=force_collect)
    dataloader, _, stats = build_dataloader(config)
    anneal_steps = resolve_anneal_steps(config, stats)
    log_training_plan(config, stats, anneal_steps)

    model = RSSM(config).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config={**config.__dict__, "resolved_anneal_steps": anneal_steps},
        )

    ckpt_dir = ensure_dir(config.checkpoint_dir)
    global_step = 0
    final_beta = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_beta = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        for obs_seq, action_seq in pbar:
            obs_seq = obs_seq.to(device)
            action_seq = action_seq.to(device)

            beta = current_beta(global_step, config.beta_max, anneal_steps)

            output = model(obs_seq, action_seq, free_nats=config.free_nats)
            loss = output.recon_loss + beta * output.kl_loss

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_clip
            )

            optimizer.step()
            global_step += 1
            num_batches += 1

            epoch_recon += output.recon_loss.item()
            epoch_kl += output.kl_loss_raw.item()
            epoch_beta += beta

            pbar.set_postfix(
                recon=f"{output.recon_loss.item():.4f}",
                kl=f"{output.kl_loss_raw.item():.2f}",
                beta=f"{beta:.3f}",
            )

            if global_step % config.log_every == 0:
                log_dict = {
                    "train/recon_loss": output.recon_loss.item(),
                    "train/kl_loss_raw": output.kl_loss_raw.item(),
                    "train/kl_loss_free_bits": output.kl_loss.item(),
                    "train/beta": beta,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/global_step": global_step,
                }
                if wandb_run is not None:
                    import wandb

                    wandb.log(log_dict, step=global_step)

        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        avg_beta = epoch_beta / num_batches
        final_beta = avg_beta
        logger.info(
            "Epoch %d | recon=%.4f | kl_raw=%.2f | beta=%.3f | batches=%d | step=%d",
            epoch,
            avg_recon,
            avg_kl,
            avg_beta,
            num_batches,
            global_step,
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            ckpt_path = ckpt_dir / f"rssm_epoch{epoch:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, config)
            logger.info("Saved checkpoint: %s", ckpt_path)

    if final_beta < 0.9 * config.beta_max:
        logger.warning(
            "Beta annealing incomplete: final epoch beta=%.3f < 0.9 * beta_max=%.3f. "
            "Increase --epochs, add data/chunks, or set --anneal-fraction higher.",
            final_beta,
            0.9 * config.beta_max,
        )

    if wandb_run is not None:
        import wandb

        wandb.finish()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    train(config, force_collect=args.force_collect)


if __name__ == "__main__":
    main()
