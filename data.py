"""Episode collection and PyTorch Dataset for RSSM training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config
from utils import ensure_dir

logger = logging.getLogger("rssm")


def _crop_center(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    """Center-crop a square around the image pivot (Pendulum: geometric center).

    Args:
        frame: ``(H, W, C)`` uint8 RGB from ``env.render()``.
        crop_ratio: Side length as a fraction of ``min(H, W)``. ``1.0`` = no crop.

    Returns:
        Cropped ``(side, side, C)`` array. Verify visually that the arm is never
        clipped at extreme swing angles; increase ``crop_ratio`` if it is.
    """
    if crop_ratio >= 1.0:
        return frame

    h, w = frame.shape[:2]
    side = max(1, int(min(h, w) * crop_ratio))
    cy, cx = h // 2, w // 2  # Pendulum pivot ≈ image center — verify on raw frames
    half = side // 2

    y0 = cy - half
    x0 = cx - half
    y1 = y0 + side
    x1 = x0 + side

    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    if x1 > w:
        x0 -= x1 - w
        x1 = w

    y0 = max(0, y0)
    x0 = max(0, x0)
    return frame[y0:y1, x0:x1]


def _resize_obs(obs: np.ndarray, size: int) -> np.ndarray:
    """Resize RGB observation to ``(C, size, size)`` in ``[0, 1]``."""
    import torch.nn.functional as F

    t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def _process_frame(frame: np.ndarray, img_size: int, crop_ratio: float = 1.0) -> np.ndarray:
    """Center-crop the raw frame, then resize to ``img_size`` (crop before downscale)."""
    cropped = _crop_center(frame, crop_ratio)
    return _resize_obs(cropped, img_size)


def preview_cropped_frames(
    env_name: str = "Pendulum-v1",
    img_size: int = 32,
    crop_ratio: float = 0.5,
    seed: int = 42,
    num_frames: int = 8,
    out_dir: str | Path = "outputs",
) -> Path:
    """Save a grid of cropped+resized frames for visual inspection before long runs.

    Re-run this after changing ``crop_ratio`` to confirm:
    1. the pendulum fills a large fraction of the image, and
    2. the arm is never clipped at extreme swing angles (increase ratio if clipped).
    """
    import matplotlib.pyplot as plt

    env = gym.make(env_name, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    for step in range(num_frames * 3):
        raw = env.render()
        if raw is None:
            raise RuntimeError(f"env.render() returned None for {env_name}")
        if step % 3 == 0:
            proc = _process_frame(raw, img_size, crop_ratio)
            frames.append(proc.transpose(1, 2, 0))
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        if len(frames) >= num_frames:
            break
    env.close()

    out_path = ensure_dir(out_dir) / f"crop_preview_r{crop_ratio:.2f}.png"
    fig, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2))
    if len(frames) == 1:
        axes = [axes]
    for ax, fr in zip(axes, frames):
        ax.imshow(fr.clip(0, 1))
        ax.axis("off")
    fig.suptitle(f"crop_ratio={crop_ratio} → {img_size}x{img_size}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved crop preview: %s (inspect before long training)", out_path)
    return out_path


def collect_episodes(
    env_name: str,
    num_episodes: int,
    max_steps: int,
    action_dim: int,
    img_size: int = 32,
    crop_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect random-agent rollouts with rgb_array rendering.

    Args:
        env_name: Gymnasium environment id (e.g. ``Pendulum-v1``).
        num_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.
        action_dim: Continuous action dimensionality.
        img_size: Side length of square resized observations.
        crop_ratio: Center-crop side as fraction of raw ``min(H,W)`` before resize
            (``1.0`` = no crop). Default ``0.5`` keeps the central half.
        seed: Base random seed.

    Returns:
        ``observations`` ``(N, T, C, H, W)`` float32 in ``[0, 1]``,
        ``actions`` ``(N, T, action_dim)`` float32,
        ``lengths`` ``(N,)`` actual episode lengths.
    """
    env = gym.make(env_name, render_mode="rgb_array")

    all_obs: list[list[np.ndarray]] = []
    all_actions: list[list[np.ndarray]] = []
    lengths: list[int] = []

    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []

        obs, _ = env.reset(seed=seed + ep)
        for step in range(max_steps):
            frame = env.render()
            if frame is None:
                raise RuntimeError(
                    f"env.render() returned None for {env_name}. "
                    "Ensure render_mode='rgb_array'."
                )
            obs_list.append(_process_frame(frame, img_size, crop_ratio))

            # Pendulum torque in [-2, 2]; passed raw (normalization may help later).
            action = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
            if action.shape[0] != action_dim:
                raise ValueError(
                    f"Expected action_dim={action_dim}, got shape {action.shape} "
                    f"from {env_name}"
                )
            act_list.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        all_obs.append(obs_list)
        all_actions.append(act_list)
        lengths.append(len(obs_list))

    env.close()

    max_len = max(lengths)
    n = num_episodes
    c = all_obs[0][0].shape[0]

    obs_array = np.zeros((n, max_len, c, img_size, img_size), dtype=np.float32)
    act_array = np.zeros((n, max_len, action_dim), dtype=np.float32)

    for i, (obs_ep, act_ep, length) in enumerate(zip(all_obs, all_actions, lengths)):
        for t in range(length):
            obs_array[i, t] = obs_ep[t]
            act_array[i, t] = act_ep[t]

    return obs_array, act_array, np.array(lengths, dtype=np.int64)


def count_chunks(
    lengths: np.ndarray,
    seq_len: int,
    chunk_stride: int = 1,
) -> int:
    """Count valid sliding-window chunks across all episodes."""
    stride = max(1, chunk_stride)
    total = 0
    for length in lengths:
        if length >= seq_len:
            total += (int(length) - seq_len) // stride + 1
    return total


def save_dataset(
    path: str | Path,
    observations: np.ndarray,
    actions: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Save collected episodes to a compressed ``.npz`` file."""
    ensure_dir(Path(path).parent)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        lengths=lengths,
    )


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load episodes from a ``.npz`` file."""
    data = np.load(path)
    return data["observations"], data["actions"], data["lengths"]


def cache_meets_requirements(config: Config) -> tuple[bool, str]:
    """Check whether an on-disk dataset satisfies the current training config."""
    path = Path(config.data_path)
    if not path.exists():
        return False, "dataset file not found"

    _, actions, lengths = load_dataset(path)
    num_episodes = len(lengths)
    max_len = int(lengths.max())

    if actions.ndim != 3 or actions.dtype != np.float32:
        return (
            False,
            "cached actions must be float32 with shape (N, T, action_dim); "
            "use --force-collect",
        )
    if actions.shape[-1] != config.action_dim:
        return (
            False,
            f"cached action_dim={actions.shape[-1]} != config.action_dim="
            f"{config.action_dim} (use --force-collect)",
        )

    if num_episodes < config.num_episodes:
        return (
            False,
            f"cached dataset has {num_episodes} episodes, "
            f"need {config.num_episodes} (use --force-collect)",
        )

    if max_len < config.seq_len:
        return (
            False,
            f"max episode length is {max_len}, but seq_len={config.seq_len}. "
            f"Use --force-collect to re-collect, or lower --seq-len "
            f"(e.g. --seq-len {max_len})",
        )

    # Count how many training chunks would be available
    num_chunks = count_chunks(lengths, config.seq_len, config.chunk_stride)
    if num_chunks == 0:
        return False, f"no sequence chunks of length {config.seq_len} in cached data"

    logger.info(
        "Using cached dataset: %s (%d episodes, max_len=%d, %d chunks)",
        path,
        num_episodes,
        max_len,
        num_chunks,
    )
    return True, ""


def collect_and_save(config: Config, force: bool = False) -> str:
    """Collect episodes if needed and return the dataset path."""
    path = Path(config.data_path)

    if not force:
        ok, reason = cache_meets_requirements(config)
        if ok:
            return str(path)
        logger.info("Re-collecting data: %s", reason)
    else:
        logger.info("Force re-collecting data")

    obs, actions, lengths = collect_episodes(
        env_name=config.env_name,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps_per_episode,
        action_dim=config.action_dim,
        img_size=config.img_size,
        crop_ratio=config.crop_ratio,
        seed=config.seed,
    )
    max_len = int(lengths.max())
    if max_len < config.seq_len:
        raise ValueError(
            f"Collected episodes are too short for seq_len={config.seq_len} "
            f"(max episode length={max_len}). Lower --seq-len or increase "
            f"--max-steps / collect more episodes."
        )
    save_dataset(path, obs, actions, lengths)
    logger.info(
        "Saved dataset: %s (%d episodes, max_len=%d)",
        path,
        len(lengths),
        max_len,
    )
    return str(path)


class RSSMDataset(Dataset):
    """Dataset yielding fixed-length sequence chunks from collected episodes."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        lengths: np.ndarray,
        seq_len: int,
        action_dim: int,
        chunk_stride: int = 1,
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.lengths = lengths
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.chunk_stride = max(1, chunk_stride)

        # Sliding-window chunking: every ``chunk_stride`` frames within each episode
        self.indices: list[tuple[int, int]] = []
        for ep_idx, length in enumerate(lengths):
            if length >= seq_len:
                for start in range(0, int(length) - seq_len + 1, self.chunk_stride):
                    self.indices.append((ep_idx, start))

        if not self.indices:
            max_len = int(lengths.max())
            raise ValueError(
                f"No valid chunks of length {seq_len} found "
                f"(max episode length: {max_len}). "
                f"Re-collect with --force-collect or lower --seq-len "
                f"(e.g. --seq-len {max_len})."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, start = self.indices[idx]
        end = start + self.seq_len

        obs = self.observations[ep_idx, start:end]  # (T, C, H, W)
        acts = self.actions[ep_idx, start:end]  # (T, action_dim)

        obs_t = torch.from_numpy(obs)
        acts_t = torch.from_numpy(acts).float()

        return obs_t, acts_t


@dataclass
class DatasetStats:
    """Summary of the training dataset and DataLoader schedule."""

    num_episodes: int
    max_episode_len: int
    num_chunks: int
    batch_size: int
    drop_last: bool
    steps_per_epoch: int
    total_steps: int


def build_dataset(config: Config) -> RSSMDataset:
    """Load episodes from disk and build the chunk dataset."""
    observations, actions, lengths = load_dataset(config.data_path)
    return RSSMDataset(
        observations=observations,
        actions=actions,
        lengths=lengths,
        seq_len=config.seq_len,
        action_dim=config.action_dim,
        chunk_stride=config.chunk_stride,
    )


def dataset_stats(config: Config, dataset: RSSMDataset) -> DatasetStats:
    """Compute dataloader and training-step counts for logging."""
    num_chunks = len(dataset)
    steps_per_epoch = (
        num_chunks // config.batch_size
        if config.drop_last
        else (num_chunks + config.batch_size - 1) // config.batch_size
    )
    steps_per_epoch = max(steps_per_epoch, 1 if num_chunks > 0 else 0)
    return DatasetStats(
        num_episodes=len(dataset.lengths),
        max_episode_len=int(dataset.lengths.max()),
        num_chunks=num_chunks,
        batch_size=config.batch_size,
        drop_last=config.drop_last,
        steps_per_epoch=steps_per_epoch,
        total_steps=steps_per_epoch * config.epochs,
    )


def build_dataloader(
    config: Config,
    dataset: RSSMDataset | None = None,
) -> tuple[torch.utils.data.DataLoader, RSSMDataset, DatasetStats]:
    """Load dataset from disk and wrap in a DataLoader."""
    if dataset is None:
        dataset = build_dataset(config)
    stats = dataset_stats(config, dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset, stats
