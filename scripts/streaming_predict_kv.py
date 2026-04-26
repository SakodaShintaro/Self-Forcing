"""Streaming +Δt prediction with sensor-frame KV-cache injection.

Each block, the just-generated 3 latents are replaced in the KV cache by the
encoded latents of the actual incoming sensor frames. The model thus carries
long-term context built entirely from real sensor history while continuously
predicting +Δt-ahead frames.

KV cache budget for self_forcing_dmd is 21 latents → at most 6 prediction
blocks (3 initial + 6×3 = 21).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu  # noqa: E402
from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline  # noqa: E402
from utils.misc import set_seed  # noqa: E402

TARGET_W, TARGET_H = 832, 480
SOURCE_FPS = 16
LATENTS_PER_BLOCK = 3
INITIAL_LATENTS = 3
INITIAL_PX_FRAMES = 9               # 1 + 4*(3-1)
FRAMES_PER_BLOCK = 12               # 4 px per non-first latent × 3
KV_CACHE_LATENT_LIMIT = 21          # self_forcing_dmd

def fit_letterbox(img: Image.Image, target_w: int = TARGET_W, target_h: int = TARGET_H) -> Image.Image:
    src_w, src_h = img.size
    if (src_w, src_h) == (target_w, target_h):
        return img
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


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int]) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def resolve_checkpoint(spec: str) -> str:
    if spec.startswith("hf:"):
        _, repo_id, filename = spec.split(":", 2)
        return hf_hub_download(repo_id, filename)
    return spec


def build_pipeline(args, device: torch.device):
    print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    state_dict = torch.load(resolve_checkpoint(args.checkpoint_path), map_location="cpu")
    pipeline.generator.load_state_dict(
        state_dict["generator_ema" if args.use_ema else "generator"]
    )
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)
    return pipeline, low_memory


class StreamingPredictor:
    """Stateful streaming predictor.

    Usage:
        sp = StreamingPredictor(pipeline, conditional_dict, device)
        sp.initialize(initial_9_frames)
        pixel_preds = [sp.predict_and_decode()]
        for new_12_frames in sensor_stream:
            sp.ingest(new_12_frames)
            pixel_preds.append(sp.predict_and_decode())
        sp.finish()

    Both `ingest` and `predict_and_decode` use the VAE's incremental causal cache
    (`cached_decode`), so each call processes only the latents handed to it. The
    decoder cache snapshot is saved/restored around prediction decoding so the
    cache reflects only real sensor history (predictions never pollute it).
    """

    def __init__(self, pipeline, conditional_dict, device: torch.device):
        self.pipeline = pipeline
        self.conditional_dict = conditional_dict
        self.device = device
        self.dtype = torch.bfloat16
        self.batch_size = 1
        self.context_noise = int(getattr(pipeline.args, "context_noise", 0))

        self.current_start_frame = 0
        self.encode_times: list[float] = []
        self.predict_times: list[float] = []
        self.decode_times: list[float] = []
        self.advance_dec_times: list[float] = []

    def initialize(self, initial_pixels: torch.Tensor) -> None:
        """Seed KV cache + VAE encoder/decoder caches from the first 9 px frames."""
        assert initial_pixels.shape[2] == INITIAL_PX_FRAMES, (
            f"initial_pixels expects {INITIAL_PX_FRAMES} frames, got {initial_pixels.shape[2]}"
        )
        self.pipeline._initialize_kv_cache(
            batch_size=self.batch_size, dtype=self.dtype, device=self.device,
        )
        self.pipeline._initialize_crossattn_cache(
            batch_size=self.batch_size, dtype=self.dtype, device=self.device,
        )
        self.pipeline.vae.model.clear_cache()
        self.current_start_frame = 0
        self.encode_times = []
        self.predict_times = []
        self.decode_times = []
        self.advance_dec_times = []

        latents = self._encode(initial_pixels)
        self._inject(latents)
        self.current_start_frame += INITIAL_LATENTS
        self._advance_decoder(latents)

    def ingest(self, new_pixels: torch.Tensor) -> None:
        """A new 12-frame sensor block has arrived: encode, inject into KV, advance decoder."""
        assert new_pixels.shape[2] == FRAMES_PER_BLOCK, (
            f"ingest expects {FRAMES_PER_BLOCK} frames, got {new_pixels.shape[2]}"
        )
        new_latents = self._encode(new_pixels)
        self._inject(new_latents)
        self.current_start_frame += LATENTS_PER_BLOCK
        self._advance_decoder(new_latents)

    def predict_and_decode(self) -> torch.Tensor:
        """Generate the next 0.75s prediction and decode it incrementally.

        Returns (T_pixel, C, H, W) in [0, 1]. T_pixel = 12 (= 3 latents × 4 px each).
        The decoder cache is saved/restored so the prediction does not pollute the
        sensor-only context that subsequent ingests will build on.
        """
        torch.cuda.synchronize()
        t_p = time.perf_counter()
        prediction = self._predict()
        torch.cuda.synchronize()
        self.predict_times.append(time.perf_counter() - t_p)

        saved_feat_map = [
            t.clone() if isinstance(t, torch.Tensor) else t
            for t in self.pipeline.vae.model._feat_map
        ]

        torch.cuda.synchronize()
        t_d = time.perf_counter()
        decoded = self.pipeline.vae.decode_to_pixel(prediction, use_cache=True)
        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
        torch.cuda.synchronize()
        self.decode_times.append(time.perf_counter() - t_d)

        feat_map = self.pipeline.vae.model._feat_map
        for i in range(len(feat_map)):
            feat_map[i] = saved_feat_map[i]

        return decoded[0]

    def finish(self) -> None:
        self.pipeline.vae.model.clear_cache()

    # --- internal helpers -------------------------------------------------

    def _advance_decoder(self, sensor_latents: torch.Tensor) -> None:
        """Push real sensor latents through the decoder so its causal cache stays in
        sync with the actual past. Output pixels are discarded (left panel of the
        side-by-side video uses the original sensor PNGs)."""
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = self.pipeline.vae.decode_to_pixel(sensor_latents, use_cache=True)
        torch.cuda.synchronize()
        self.advance_dec_times.append(time.perf_counter() - t)

    # --- internal helpers -------------------------------------------------

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(dtype=self.dtype)
        torch.cuda.synchronize()
        t = time.perf_counter()
        latents = self.pipeline.vae.encode_to_latent(pixels, use_cache=True).to(dtype=self.dtype)
        torch.cuda.synchronize()
        self.encode_times.append(time.perf_counter() - t)
        return latents

    def _inject(self, latents: torch.Tensor) -> None:
        """Write sensor latents into KV cache at current_start_frame slot, timestep=0."""
        timestep = torch.full(
            [self.batch_size, 1], self.context_noise, dtype=torch.int64, device=self.device,
        )
        self.pipeline.generator(
            noisy_image_or_video=latents,
            conditional_dict=self.conditional_dict,
            timestep=timestep,
            kv_cache=self.pipeline.kv_cache1,
            crossattn_cache=self.pipeline.crossattn_cache,
            current_start=self.current_start_frame * self.pipeline.frame_seq_length,
        )

    def _predict(self) -> torch.Tensor:
        """Run denoising loop at current_start_frame slot. Returns 3 prediction latents."""
        sampled_noise = torch.randn(
            [self.batch_size, LATENTS_PER_BLOCK, 16, 60, 104],
            device=self.device, dtype=self.dtype,
        )
        noisy_input = sampled_noise
        denoising_steps = self.pipeline.denoising_step_list
        for index, current_timestep in enumerate(denoising_steps):
            timestep = torch.full(
                [self.batch_size, LATENTS_PER_BLOCK],
                int(current_timestep.item()),
                device=self.device, dtype=torch.int64,
            )
            _, denoised_pred = self.pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=self.current_start_frame * self.pipeline.frame_seq_length,
            )
            if index < len(denoising_steps) - 1:
                next_t = denoising_steps[index + 1]
                noisy_input = self.pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_t * torch.ones(
                        [self.batch_size * LATENTS_PER_BLOCK], device=self.device, dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
        return denoised_pred.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument(
        "--checkpoint_path",
        default="hf:gdhe17/Self-Forcing:checkpoints/self_forcing_dmd.pt",
    )
    parser.add_argument(
        "--caption",
        default="First-person dashcam view from a car driving on a CARLA simulated road.",
    )
    parser.add_argument("--start_index", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of sensor frames to consume. "
                             "Default: use as many as fit in the KV cache (= 81 frames = 6 blocks).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()

    max_blocks = (KV_CACHE_LATENT_LIMIT - INITIAL_LATENTS) // LATENTS_PER_BLOCK  # 6

    set_seed(args.seed)
    device = torch.device("cuda")
    pipeline, _ = build_pipeline(args, device)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    png_paths = sorted(args.src.glob("*.png"))
    available = len(png_paths) - args.start_index
    cap_by_kv = INITIAL_PX_FRAMES + FRAMES_PER_BLOCK * max_blocks  # 81
    if args.num_frames is None:
        num_frames_total = min(available, cap_by_kv)
    else:
        if args.num_frames > available:
            raise ValueError(
                f"--num_frames={args.num_frames} > available {available} "
                f"(from start_index {args.start_index} in {args.src})"
            )
        num_frames_total = min(args.num_frames, cap_by_kv)
    if num_frames_total < INITIAL_PX_FRAMES + FRAMES_PER_BLOCK:
        raise ValueError(
            f"need at least {INITIAL_PX_FRAMES + FRAMES_PER_BLOCK} frames "
            f"(= seed 9 + 1 block 12), got {num_frames_total}"
        )

    num_pred_blocks = (num_frames_total - INITIAL_PX_FRAMES) // FRAMES_PER_BLOCK
    # Trim to an exact integer number of blocks
    num_frames_total = INITIAL_PX_FRAMES + FRAMES_PER_BLOCK * num_pred_blocks

    print(
        f"Loading {num_frames_total} sensor frames from {args.src.name} "
        f"→ {num_pred_blocks} prediction blocks"
    )
    sensor_pixel_frames = [
        fit_letterbox(Image.open(png_paths[i]).convert("RGB"))
        for i in range(args.start_index, args.start_index + num_frames_total)
    ]
    sensor_tensor = torch.stack(
        [normalize(im) for im in sensor_pixel_frames], dim=1
    ).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    conditional_dict = pipeline.text_encoder(text_prompts=[args.caption])

    print(f"Streaming inference: {num_pred_blocks} blocks")
    predictor = StreamingPredictor(pipeline, conditional_dict, device)

    # Seed: encode first 9 frames, inject into KV, prime decoder cache, then make
    # the first prediction directly on the seed (no new sensor has arrived yet).
    predictor.initialize(sensor_tensor[:, :, :INITIAL_PX_FRAMES])
    decoded_blocks: list[torch.Tensor] = [predictor.predict_and_decode()]

    # Streaming loop: each iteration ingests the next 12 px frames and predicts.
    last_ingest_start = INITIAL_PX_FRAMES + FRAMES_PER_BLOCK * (num_pred_blocks - 1)
    for start_px in range(INITIAL_PX_FRAMES, last_ingest_start, FRAMES_PER_BLOCK):
        predictor.ingest(sensor_tensor[:, :, start_px:start_px + FRAMES_PER_BLOCK])
        decoded_blocks.append(predictor.predict_and_decode())
    predictor.finish()

    encode_times = predictor.encode_times
    predict_times = predictor.predict_times
    decode_times = predictor.decode_times
    advance_times = predictor.advance_dec_times
    block_times = [
        predict_times[k] + decode_times[k]
        + (encode_times[k] + advance_times[k] if k < len(encode_times) and k > 0 else 0)
        for k in range(num_pred_blocks)
    ]
    print(
        f"  encode total {sum(encode_times):.2f}s ({len(encode_times)} calls, "
        f"per-call {sum(encode_times)/len(encode_times):.3f}s)\n"
        f"  predict total {sum(predict_times):.2f}s ({len(predict_times)} calls, "
        f"per-call {sum(predict_times)/len(predict_times):.3f}s)\n"
        f"  decode (pred) total {sum(decode_times):.2f}s "
        f"per-call {sum(decode_times)/len(decode_times):.3f}s\n"
        f"  decode (sensor advance) total {sum(advance_times):.2f}s "
        f"per-call {sum(advance_times)/len(advance_times):.3f}s"
    )

    composed: list[np.ndarray] = []
    for k, pred_block in enumerate(decoded_blocks):
        block_start_local = INITIAL_PX_FRAMES - 1 + FRAMES_PER_BLOCK * k  # 8, 20, ...
        block_start_global = args.start_index + block_start_local
        block_start_time = block_start_global / SOURCE_FPS
        print(
            f"  block {k+1}: predicted from t={block_start_time:.2f}s for next "
            f"{FRAMES_PER_BLOCK} frames | gen {block_times[k]:.2f}s"
        )

        for i in range(FRAMES_PER_BLOCK):
            sensor_local = block_start_local + 1 + i
            sensor_global = args.start_index + sensor_local
            sensor_time = sensor_global / SOURCE_FPS
            lookahead = sensor_time - block_start_time

            sensor_frame = np.array(sensor_pixel_frames[sensor_local])
            pred_arr = pred_block[i].permute(1, 2, 0).cpu().float().numpy()
            pred_frame = (pred_arr * 255.0).clip(0, 255).astype(np.uint8).copy()

            side_by_side = np.concatenate([sensor_frame, pred_frame], axis=1)
            draw_label(side_by_side, f"sensor t={sensor_time:.2f}s", (16, 36))
            draw_label(
                side_by_side,
                f"pred t={sensor_time:.2f}s (made @ {block_start_time:.2f}s, +{lookahead:.2f}s)",
                (TARGET_W + 16, 36),
            )
            draw_label(
                side_by_side,
                f"block {k+1}/{num_pred_blocks} frame {i+1}/{FRAMES_PER_BLOCK} | gen {block_times[k]:.2f}s",
                (16, TARGET_H - 16),
            )
            composed.append(side_by_side)

    out = torch.from_numpy(np.stack(composed, axis=0)).float()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(args.output), out, fps=SOURCE_FPS)

    sensor_dt = FRAMES_PER_BLOCK / SOURCE_FPS  # 0.75s per block
    avg_block = sum(block_times) / len(block_times)
    print(
        f"\nWrote {args.output} ({len(composed)} frames @ {SOURCE_FPS} fps)\n"
        f"Per-block wall {avg_block:.2f}s vs sensor {sensor_dt:.2f}s "
        f"→ {avg_block / sensor_dt:.2f}× real-time ({sensor_dt / avg_block:.2f}× speed)"
    )


if __name__ == "__main__":
    main()
