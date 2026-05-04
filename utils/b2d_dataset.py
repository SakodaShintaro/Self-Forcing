"""Random-window dataset over precomputed bench2drive episode latents.

Layout (anchored at b2d_root, the bench2drive raw-data directory):
    <b2d_root>/splits.json                                 (from scripts/split.py)
    <b2d_root>/latents/{train,valid}/<episode>.pt          (from scripts/encode_latents.py)

Each episode .pt file stores a (T_lat, 16, 60, 104) bf16 tensor. Each __getitem__
samples a contiguous `num_frames`-latent window from a random episode-and-offset.

Output shape is (1, num_frames, 16, 60, 104) under key `ode_latent` (the leading
dim is "denoising step"; the trainer indexes `batch["ode_latent"][:, -1]` to take
the clean latent).
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

        # Pre-filter: keep only episodes whose .pt exists AND has at least
        # num_frames latents. We mmap each tensor to read just the shape header,
        # so this is cheap even for hundreds of files.
        self.episodes: list[str] = []
        too_short: list[tuple[str, int]] = []
        for ep in episodes:
            path = self.latent_dir / f"{ep}.pt"
            if not path.exists():
                continue
            latents = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            T = latents.shape[0]
            if T >= num_frames:
                self.episodes.append(ep)
            else:
                too_short.append((ep, T))

        if too_short:
            preview = ", ".join(f"{ep}({T})" for ep, T in too_short[:3])
            more = f" ...(+{len(too_short) - 3} more)" if len(too_short) > 3 else ""
            print(
                f"[Bench2DriveLatentDataset] split={split}: dropped "
                f"{len(too_short)} episode(s) shorter than num_frames={num_frames}: "
                f"{preview}{more}",
                flush=True,
            )

        if len(self.episodes) == 0:
            raise RuntimeError(
                f"No usable latent .pt files (>= {num_frames} frames) for split={split} "
                f"in {self.latent_dir}. Run scripts/encode_latents.py first."
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
