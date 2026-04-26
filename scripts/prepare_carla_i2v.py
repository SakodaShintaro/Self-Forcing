"""Prepare CARLA frames into TextImagePairDataset format for I2V inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TARGET_W, TARGET_H = 832, 480  # Wan T2V-1.3B latent grid (104, 60) * 8


def fit_letterbox(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize-and-pad so the full source image is visible (no information lost)."""
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h
    if src_ratio > target_ratio:
        new_w = target_w
        new_h = int(round(target_w / src_ratio))
    else:
        new_h = target_h
        new_w = int(round(target_h * src_ratio))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="CARLA frames directory (PNG sequence)")
    parser.add_argument("--dst", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--caption", type=str, required=True, help="Caption to associate with the clip")
    parser.add_argument("--num_frames", type=int, default=1,
                        help="How many leading frames to include (1 for plain I2V).")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Index into the sorted PNG list to start from.")
    parser.add_argument("--ratio_tag", type=str, default="carla",
                        help="Aspect-ratio tag used as subfolder name and metadata suffix.")
    args = parser.parse_args()

    image_dir = args.dst / args.ratio_tag
    image_dir.mkdir(parents=True, exist_ok=True)

    src_frames = sorted(args.src.glob("*.png"))
    if not src_frames:
        raise FileNotFoundError(f"No PNG frames found in {args.src}")
    selected = src_frames[args.start_index : args.start_index + args.num_frames]
    if len(selected) < args.num_frames:
        raise ValueError(f"Need {args.num_frames} frames from index {args.start_index}, got {len(selected)}")

    file_names = []
    origin_size = None
    for frame_path in selected:
        img = Image.open(frame_path).convert("RGB")
        if origin_size is None:
            origin_size = img.size
        fit_letterbox(img, TARGET_W, TARGET_H).save(image_dir / frame_path.name)
        file_names.append(frame_path.name)
    bbox = [0, 0, origin_size[0], origin_size[1]]

    entry = {
        "caption": args.caption,
        "type": "carla",
        "origin_width": origin_size[0],
        "origin_height": origin_size[1],
        "target_crop": {
            "target_bbox": bbox,
            "target_ratio": f"{TARGET_W}-{TARGET_H}",
        },
    }
    if len(file_names) == 1:
        entry["file_name"] = file_names[0]
    else:
        entry["file_names"] = file_names

    metadata = [entry]
    metadata_path = args.dst / f"target_crop_info_{args.ratio_tag}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(file_names)} frames into 1 sample at {image_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
