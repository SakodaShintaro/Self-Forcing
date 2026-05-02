"""Encode bench2drive episodes (rgb_front) through Wan VAE -> save per-episode latent .pt files.

Inputs/outputs are anchored under --src:
  splits  : <src>/splits.json   (produced by scripts/b2d_split.py)
  latents : <src>/latents/{train,valid}/<episode>.pt

Run:
  uv run python scripts/b2d_encode_latents.py --src /path/to/bench2drive
Existing .pt files are skipped so it is safe to interrupt and resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.wan_wrapper import WanVAEWrapper  # noqa: E402

TARGET_H, TARGET_W = 480, 832  # Wan T2V-1.3B expected resolution


def encode_episode(
    vae: WanVAEWrapper, episode_dir: Path, device: torch.device
) -> torch.Tensor | None:
    rgb_dir = episode_dir / "camera" / "rgb_front"
    frame_paths = sorted(rgb_dir.glob("*.jpg"))
    if not frame_paths:
        return None

    transform = transforms.Compose(
        [
            transforms.Resize((TARGET_H, TARGET_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
        ]
    )

    frames = [transform(Image.open(p).convert("RGB")) for p in frame_paths]
    pixels = torch.stack(frames, dim=1).unsqueeze(0)  # (1, 3, T, H, W)
    pixels = pixels.to(device=device, dtype=torch.bfloat16)

    # Truncate to 1 + 4*N frames (encode requirement when use_cache=False)
    T = pixels.shape[2]
    T_eff = 1 + 4 * ((T - 1) // 4)
    if T_eff < 1:
        return None
    if T_eff != T:
        pixels = pixels[:, :, :T_eff]

    with torch.no_grad():
        latent = vae.encode_to_latent(pixels)  # (1, T_lat, 16, 60, 104) float32
    return latent.squeeze(0).to(torch.bfloat16).cpu().contiguous()  # (T_lat, 16, 60, 104)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="all",
        help="Which split to encode (default: all).",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    splits_path = args.src / "splits.json"
    out_root = args.src / "latents"
    splits = json.load(open(splits_path))
    if args.split != "all":
        splits = {args.split: splits[args.split]}

    print(f"loading VAE on {device}")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    for split_name, episodes in splits.items():
        out_dir = out_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, ep in enumerate(episodes):
            out_path = out_dir / f"{ep}.pt"
            if out_path.exists():
                continue
            t0 = time.time()
            try:
                latent = encode_episode(vae, args.src / ep, device=device)
            except Exception as e:
                print(f"{ep}: ERROR {type(e).__name__}: {e}")
                continue
            if latent is None:
                print(f"{ep}: SKIP (no frames)")
                continue
            torch.save(latent, out_path)
            print(
                f"{split_name} [{i + 1}/{len(episodes)}] "
                f"{ep}: shape={tuple(latent.shape)} dt={time.time() - t0:.1f}s"
            )

    print("done")


if __name__ == "__main__":
    main()
