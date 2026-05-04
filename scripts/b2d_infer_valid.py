"""Teacher-forced per-block predictions on the bench2drive valid split.

For each valid episode we slide a fixed-size GT context window over the saved
latent stream and ask the model to predict the *next* block from that context
(no autoregressive feedback). Concretely, with `num_context_blocks=K` and
`num_pred_blocks=M`:

    block 0..K-1   : GT
    block K        : predicted from GT blocks 0..K-1
    block K+1      : predicted from GT blocks 1..K
    ...
    block K+M-1    : predicted from GT blocks M-1..K+M-2

Each prediction is an independent CausalInferencePipeline call: KV cache is
reset, the K context blocks are pushed in (Step-2 timestep=0 path), and a
single noise block is denoised. The K context latents come straight from
`<b2d_root>/latents/valid/<episode>.pt`; only the predicted block latents are
kept and concatenated with the GT context for a single VAE decode at the end.

Run:
  uv run python scripts/b2d_infer_valid.py \
    --config_path configs/b2d_finetune.yaml --tag baseline
  uv run python scripts/b2d_infer_valid.py \
    --config_path configs/b2d_finetune.yaml \
    --checkpoint_path logs/b2d_finetune/checkpoint_model_001000/model.pt \
    --tag step_001000
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.causal_inference import CausalInferencePipeline  # noqa: E402
from utils.misc import resolve_checkpoint_path, set_seed  # noqa: E402


class _CachedTextEncoder(torch.nn.Module):
    """Stand-in for WanTextEncoder when the caption is fixed for the whole run."""

    def __init__(self, cached: dict, device: torch.device) -> None:
        super().__init__()
        for k, v in cached.items():
            self.register_buffer(f"_cached_{k}", v.detach().to(device=device))
        self._cached_keys = list(cached.keys())
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, text_prompts):
        return {k: getattr(self, f"_cached_{k}") for k in self._cached_keys}


def _load_state_dict(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("generator_ema", "generator", "model"):
        if k in sd:
            sd = sd[k]
            break
    return {
        k.replace("_fsdp_wrapped_module.", "")
         .replace("_checkpoint_wrapped_module.", "")
         .replace("_orig_mod.", ""): v
        for k, v in sd.items()
    }


def _decode_latent_to_uint8(pipeline, latents_thwc: torch.Tensor) -> torch.Tensor:
    """latents_thwc: (B, T, 16, 60, 104) on GPU. Returns (T_pix, H, W, 3) uint8 on CPU."""
    video = pipeline.vae.decode_to_pixel(latents_thwc, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    pipeline.vae.model.clear_cache()
    return (video[0].permute(0, 2, 3, 1).cpu().float() * 255).to(torch.uint8)


def _load_real_frames(episode_dir: Path, num_pixel_frames: int,
                      target_h: int, target_w: int) -> torch.Tensor:
    """Load and resize the first `num_pixel_frames` raw jpgs from a bench2drive
    episode. Returns (T_pix, H, W, 3) uint8."""
    rgb_dir = episode_dir / "camera" / "rgb_front"
    paths = sorted(rgb_dir.glob("*.jpg"))[:num_pixel_frames]
    if len(paths) < num_pixel_frames:
        raise RuntimeError(
            f"Need {num_pixel_frames} jpgs in {rgb_dir}, got {len(paths)}"
        )
    resize = transforms.Resize((target_h, target_w))
    frames = []
    for p in paths:
        img = resize(Image.open(p).convert("RGB"))
        frames.append(torch.from_numpy(np.asarray(img)))  # (H, W, 3) uint8
    return torch.stack(frames, dim=0)  # (T, H, W, 3)


def _block_frame_index(pixel_idx: int, fpb: int) -> tuple[int, int]:
    """Map a pixel-frame index to (block_idx, frame_within_block).

    The Wan VAE encodes the very first latent from 1 pixel frame and every
    subsequent latent from 4 pixel frames. With `fpb` latents per block, the
    first block thus covers (1 + (fpb-1)*4) pixel frames; later blocks cover
    fpb*4 pixel frames each.
    """
    first_block_pix = 1 + (fpb - 1) * 4   # = 9 when fpb=3
    later_block_pix = fpb * 4             # = 12 when fpb=3
    if pixel_idx < first_block_pix:
        return 0, pixel_idx
    rem = pixel_idx - first_block_pix
    return 1 + rem // later_block_pix, rem % later_block_pix


def _per_frame_psnr(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """pred, ref: (T, H, W, 3) uint8. Returns (T,) float PSNR per frame in dB."""
    p = pred.float()
    r = ref.float()
    mse = ((p - r) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 20.0 * torch.log10(255.0 / torch.sqrt(mse))


def _annotate(video: torch.Tensor, side: str, fpb: int, K: int,
              psnrs: torch.Tensor | None) -> torch.Tensor:
    """Annotate each frame with block.frame index (and PSNR for the pred side).

    side: 'real' or 'pred'. Frames in blocks [0..K-1] on the pred side are
    actually decoded from GT context (not model output); those are tagged 'ctx'
    instead of 'pred' so the viewer can tell predictions apart.
    """
    arr = video.numpy().copy()
    for t in range(arr.shape[0]):
        block, f_in_block = _block_frame_index(t, fpb)
        if side == "real":
            tag = "real"
        else:
            tag = "ctx" if block < K else "pred"
        label = f"{tag} B{block} F{f_in_block}"
        if side == "pred" and psnrs is not None:
            label += f"  PSNR {psnrs[t].item():.1f}dB"
        cv2.putText(arr[t], label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(arr[t], label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return torch.from_numpy(arr)


def _side_by_side(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concat two (T, H, W, 3) uint8 videos along width (T must match)."""
    T = min(left.shape[0], right.shape[0])
    return torch.cat([left[:T], right[:T]], dim=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional LoRA fine-tune ckpt. None = base self_forcing_dmd.pt only.")
    parser.add_argument("--b2d_root", type=str, required=True,
                        help="Bench2Drive root (contains splits.json and latents/valid/).")
    parser.add_argument("--out_root", type=Path, default=None,
                        help="Output root. Required only when --checkpoint_path is omitted; "
                             "with a checkpoint, results are saved next to it.")
    parser.add_argument("--tag", type=str, default="eval",
                        help="Suffix appended to the timestamped output directory name.")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Limit to first N valid episodes (default: all).")
    parser.add_argument("--num_context_blocks", type=int, default=1,
                        help="K: how many GT blocks to feed as context for each prediction.")
    parser.add_argument("--num_pred_blocks", type=int, default=6,
                        help="M: how many sliding predictions to make per episode "
                             "(K+M total decoded blocks; default 1+6 = 7 blocks = 21 latents).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10,
                        help="Mp4 playback fps (bench2drive native rate is 10).")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.checkpoint_path:
        out_dir = Path(args.checkpoint_path).parent / f"{stamp}_{args.tag}"
    else:
        if args.out_root is None:
            parser.error("--out_root is required when --checkpoint_path is not given.")
        out_dir = args.out_root / f"{stamp}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    config.b2d_root = args.b2d_root
    config.denoising_step_list = [1000, 750, 500, 250]
    config.warp_denoising_step = True

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    print(f"output dir: {out_dir}")
    print("building pipeline")
    pipeline = CausalInferencePipeline(config, device=device)

    base_ckpt_path = resolve_checkpoint_path(config.generator_ckpt)
    print(f"loading base ckpt: {base_ckpt_path}")
    pipeline.generator.load_state_dict(_load_state_dict(base_ckpt_path), strict=True)

    lora_cfg = getattr(config, "lora", None)
    if args.checkpoint_path:
        if not (lora_cfg and lora_cfg.get("enabled", False)):
            raise ValueError("--checkpoint_path requires lora.enabled=true in config.")
        from peft import LoraConfig, get_peft_model
        pipeline.generator.model.requires_grad_(False)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.rank),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            target_modules=list(lora_cfg.target_modules),
            bias="none",
        )
        pipeline.generator.model = get_peft_model(pipeline.generator.model, peft_cfg)
        sd = _load_state_dict(resolve_checkpoint_path(args.checkpoint_path))
        missing, unexpected = pipeline.generator.load_state_dict(sd, strict=False)
        print(f"finetune load: {len(missing)} missing, {len(unexpected)} unexpected")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # Run T5 once on the fixed caption, cache the embedding, and free T5 weights.
    fixed_caption = config.b2d_caption
    pipeline.text_encoder.to(device=device)
    cached_text = pipeline.text_encoder([fixed_caption])
    pipeline.text_encoder = _CachedTextEncoder(cached_text, device=device)
    torch.cuda.empty_cache()

    # Discover valid episodes with precomputed latents
    b2d_root = Path(config.b2d_root)
    splits = json.load(open(b2d_root / "splits.json"))
    latent_dir = b2d_root / "latents" / "valid"
    episodes = [ep for ep in splits["valid"] if (latent_dir / f"{ep}.pt").exists()]
    if args.num_episodes is not None:
        episodes = episodes[:args.num_episodes]
    print(f"valid episodes to infer: {len(episodes)}")

    K = args.num_context_blocks
    M = args.num_pred_blocks
    fpb = config.num_frame_per_block  # latents per block (= 3 for self_forcing_dmd)
    K_lat = K * fpb
    pred_lat = fpb  # one block per inference call

    summary = []
    ep_pbar = tqdm(episodes, desc="episodes", unit="ep")
    for i, ep in enumerate(ep_pbar):
        ep_pbar.set_postfix_str(ep, refresh=False)
        latent = torch.load(
            latent_dir / f"{ep}.pt", map_location="cpu", weights_only=True
        )  # (T, 16, 60, 104), bf16
        T_lat = latent.shape[0]
        T_blocks = T_lat // fpb
        if T_blocks < K + 1:
            tqdm.write(f"[{i+1}/{len(episodes)}] {ep}: SKIP (T_blocks={T_blocks} < K+1={K+1})")
            continue
        usable_M = min(M, T_blocks - K)

        # Each prediction: shift the K-block window forward by 1 block, ask the
        # model to denoise the next block from noise conditioned on those K
        # *ground truth* blocks. KV cache is auto-reset per pipeline.inference().
        t0 = time.time()
        pred_block_latents: list[torch.Tensor] = []
        block_pbar = tqdm(range(usable_M), desc=f"  blocks ({ep[:30]})",
                          unit="blk", leave=False)
        for j in block_pbar:
            ctx_start = j * fpb
            ctx_end = ctx_start + K_lat
            initial_latent = latent[ctx_start:ctx_end].unsqueeze(0).to(
                device=device, dtype=torch.bfloat16
            )  # (1, K_lat, 16, 60, 104)
            noise = torch.randn(
                (1, pred_lat, 16, 60, 104), device=device, dtype=torch.bfloat16
            )
            _, all_lat = pipeline.inference(
                noise=noise,
                text_prompts=[fixed_caption],
                return_latents=True,
                initial_latent=initial_latent,
                low_memory=False,
            )
            # all_lat: (1, K_lat + pred_lat, 16, 60, 104). Keep only the predicted block.
            pred_block_latents.append(all_lat[:, -pred_lat:].cpu())
        block_pbar.close()

        # Decode each predicted block with **GT** as the VAE decoder cache prefix
        # (not the previously-predicted block), so that decode-time temporal
        # cache matches the inference-time context. This removes the spurious
        # boundary discontinuity that would arise if we concatenated
        # [GT, pred_1, pred_2, ...] and decoded once.
        ctx_seq = latent[:K_lat].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        pix_chunks = [_decode_latent_to_uint8(pipeline, ctx_seq)]   # K context blocks decoded
        pred_pix_per_block = pred_lat * 4   # last pred_lat latents -> pred_lat*4 pixel frames (each is a "subsequent" latent)
        for j in range(usable_M):
            n_gt_lat = (K + j) * fpb  # GT latents preceding this predicted block
            seq = torch.cat(
                [latent[:n_gt_lat].unsqueeze(0), pred_block_latents[j]], dim=1,
            ).to(device=device, dtype=torch.bfloat16)
            decoded = _decode_latent_to_uint8(pipeline, seq)
            # Keep only the pixel frames corresponding to the predicted block.
            pix_chunks.append(decoded[-pred_pix_per_block:])
        pred_uint8 = torch.cat(pix_chunks, dim=0)

        # Load matching real jpgs (resized to model resolution) for left side.
        num_pix = pred_uint8.shape[0]
        real_uint8 = _load_real_frames(
            episode_dir=b2d_root / ep,
            num_pixel_frames=num_pix,
            target_h=pred_uint8.shape[1],
            target_w=pred_uint8.shape[2],
        )

        # Per-frame PSNR (pred vs real). High in context blocks (just VAE
        # roundtrip), drops where the model has to predict.
        psnrs = _per_frame_psnr(pred_uint8, real_uint8)
        # Mean PSNR over predicted blocks only (skip the K context blocks).
        ctx_pix = 1 + (fpb - 1) * 4 + (K - 1) * fpb * 4   # = pixels covered by K context blocks
        pred_only_psnr = float(psnrs[ctx_pix:].mean().item()) if num_pix > ctx_pix else float("nan")

        # Per-block mean PSNR (helps diagnose where model errors concentrate).
        per_block_buckets: list[list[float]] = []
        for f in range(num_pix):
            block, _ = _block_frame_index(f, fpb)
            while len(per_block_buckets) <= block:
                per_block_buckets.append([])
            per_block_buckets[block].append(psnrs[f].item())
        per_block_psnr = [round(sum(b) / len(b), 2) for b in per_block_buckets]
        per_frame_psnr = [round(psnrs[f].item(), 3) for f in range(num_pix)]

        compare = _side_by_side(
            _annotate(real_uint8, side="real", fpb=fpb, K=K, psnrs=None),
            _annotate(pred_uint8, side="pred", fpb=fpb, K=K, psnrs=psnrs),
        )
        write_video(str(out_dir / f"{ep}_compare.mp4"), compare, fps=args.fps)

        dt = time.time() - t0
        running = [s["mean_psnr_predicted_blocks_db"] for s in summary] + [pred_only_psnr]
        running_mean = sum(running) / len(running)
        ep_pbar.set_postfix(
            ep=ep[:30], psnr=f"{pred_only_psnr:.2f}dB",
            avg=f"{running_mean:.2f}dB", dt=f"{dt:.1f}s",
        )
        tqdm.write(
            f"[{i+1}/{len(episodes)}] {ep}: T_blocks={T_blocks} K={K} M={usable_M} "
            f"pred_PSNR={pred_only_psnr:.2f}dB dt={dt:.1f}s"
        )
        summary.append({
            "episode": ep,
            "T_blocks": T_blocks,
            "K": K,
            "M": usable_M,
            "mean_psnr_predicted_blocks_db": round(pred_only_psnr, 3),
            "per_block_psnr_db": per_block_psnr,
            "per_frame_psnr_db": per_frame_psnr,
            "dt_seconds": round(dt, 2),
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "checkpoint_path": args.checkpoint_path,
                "num_context_blocks": K,
                "num_pred_blocks": M,
                "num_frame_per_block": fpb,
                "fps": args.fps,
                "seed": args.seed,
                "fixed_caption": fixed_caption,
                "episodes": summary,
            },
            f,
            indent=2,
        )
    print(f"done. wrote {len(summary)} videos to {out_dir}")


if __name__ == "__main__":
    main()
