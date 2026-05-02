"""Random-window dataset over precomputed bench2drive episode latents.

Layout (anchored at b2d_root, the bench2drive raw-data directory):
    <b2d_root>/splits.json                                 (from b2d_split.py)
    <b2d_root>/latents/{train,valid}/<episode>.pt          (from b2d_encode_latents.py)

Each episode .pt file stores a (T_lat, 16, 60, 104) bf16 tensor. Each __getitem__
samples a contiguous `num_frames`-latent window from a random episode-and-offset.

Output shape is (1, num_frames, 16, 60, 104) under key `ode_latent`, matching the
convention of `ShardingLMDBDataset` (where the leading dim is "denoising step";
the trainer indexes `batch["ode_latent"][:, -1]` to take the clean latent).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class Bench2DriveLatentDataset(Dataset):
    def __init__(
        self,
        b2d_root: str | Path,
        split: str,
        num_frames: int,
        fixed_caption: str,
    ) -> None:
        self.split = split
        self.num_frames = num_frames
        self.caption = fixed_caption
        self.b2d_root = Path(b2d_root)
        self.latent_dir = self.b2d_root / "latents" / split
        splits_path = self.b2d_root / "splits.json"

        with open(splits_path) as f:
            episodes = json.load(f)[split]

        # Filter to episodes that have a latent file. Assume all encoded episodes
        # are at least num_frames long; verify lazily on first access otherwise.
        self.episodes: list[str] = [
            ep for ep in episodes if (self.latent_dir / f"{ep}.pt").exists()
        ]
        if len(self.episodes) == 0:
            raise RuntimeError(
                f"No latent .pt files found for split={split} in {self.latent_dir}. "
                "Run scripts/b2d_encode_latents.py first."
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        latents = torch.load(
            self.latent_dir / f"{ep}.pt", map_location="cpu", weights_only=True
        )  # (T_lat, 16, 60, 104), bf16
        T = latents.shape[0]
        if T < self.num_frames:
            raise RuntimeError(
                f"Episode {ep} has {T} latents (< num_frames={self.num_frames}). "
                "Re-encode with longer source or filter it out."
            )
        s = random.randint(0, T - self.num_frames)
        clip = latents[s : s + self.num_frames].contiguous()  # (num_frames, 16, 60, 104)
        return {
            "prompts": self.caption,
            "ode_latent": clip.unsqueeze(0).float(),  # (1, num_frames, 16, 60, 104)
            "idx": idx,
            "episode": ep,
            "start": s,
        }
