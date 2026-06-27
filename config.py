"""Hyperparameter configuration for the RSSM world model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """All configurable dimensions and training hyperparameters."""

    # Environment
    env_name: str = "bouncing_ball"
    img_size: int = 32
    img_channels: int = 3

    # Model dimensions (embed_dim derived from CNN bottleneck — see property below)
    latent_dim: int = 32
    hidden_dim: int = 200
    action_dim: int = 1  # bouncing_ball: fixed zero scalar action

    # CNN architecture
    encoder_channels: tuple[int, int, int] = (32, 64, 128)
    decoder_channels: tuple[int, int, int] = (128, 64, 32)

    # Training
    seq_len: int = 25
    batch_size: int = 16
    chunk_stride: int = 1  # sliding-window step between chunks (1 = max overlap)
    drop_last: bool = False
    lr: float = 1e-3
    epochs: int = 50
    num_workers: int = 0

    # Loss
    beta_max: float = 1.0
    anneal_steps: int = 0  # 0 = auto (see anneal_fraction)
    anneal_fraction: float = 0.3  # fraction of total steps for beta ramp when auto
    free_nats: float = 0.0  # 0 = disabled; optional floor on balanced KL scalar
    kl_balance_scale: float = 0.8  # DreamerV2 alpha: prior learns faster than posterior
    lambda_motion: float = 5.0  # lower than 64x64 default: fewer background pixels at 32x32
    grad_clip: float = 100.0

    # Data collection / preprocessing
    num_episodes: int = 200
    max_steps_per_episode: int = 200
    crop_ratio: float = 1.0  # unused for bouncing_ball; 1.0 = no crop for Gym envs
    ball_radius: int = 7  # ~15% disk area on 32x32 (use preview to tune)
    data_path: str = "data/bouncing_ball.npz"
    # Obstacles variant: env_name="bouncing_ball_obstacles",
    # data_path="data/bouncing_ball_obstacles.npz", ball_radius=5

    # Checkpointing & logging
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    log_every: int = 50
    seed: int = 42

    # Imagination
    context_len: int = 30
    imagine_horizon: int = 30
    output_dir: str = "outputs"

    # W&B (optional)
    use_wandb: bool = True
    wandb_project: str = "rssm-worldmodel"
    wandb_run_name: str = ""

    @property
    def embed_dim(self) -> int:
        """Encoder embedding size (= channels × H × W at the CNN bottleneck)."""
        return self.flatten_dim

    @property
    def spatial_size(self) -> int:
        """Spatial resolution after three stride-2 conv layers: img_size // 8."""
        return self.img_size // 8

    @property
    def flatten_dim(self) -> int:
        """Flattened feature dimension before the encoder projection."""
        c = self.encoder_channels[-1]
        s = self.spatial_size
        return c * s * s

    def recommended_min_episodes(self) -> int:
        """Heuristic minimum episodes for ~10 batches/epoch at current settings."""
        target_chunks = self.batch_size * 10
        # Pendulum episodes are fixed-length; each yields (max_steps - seq_len + 1) chunks.
        chunks_per_episode = max(0, self.max_steps_per_episode - self.seq_len + 1)
        if chunks_per_episode == 0:
            return self.batch_size * 10
        return max(200, (target_chunks + chunks_per_episode - 1) // chunks_per_episode)
