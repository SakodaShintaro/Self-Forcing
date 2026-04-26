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

DELTA_T_TO_GEN_IDX = {0.25: 3, 0.5: 7, 0.75: 11}


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
        sp.initialize(initial_9_frames)        # seed KV cache with 3 sensor latents
        for new_12_frames in sensor_stream:
            prediction = sp.step(new_12_frames)  # inject sensor, generate one prediction
    """

    def __init__(self, pipeline, conditional_dict, device: torch.device):
        self.pipeline = pipeline
        self.conditional_dict = conditional_dict
        self.device = device
        self.dtype = torch.bfloat16
        self.batch_size = 1
        self.context_noise = int(getattr(pipeline.args, "context_noise", 0))

        self.current_start_frame = 0
        self.accumulated_latents: list[torch.Tensor] = []
        self.encode_times: list[float] = []
        self.step_times: list[float] = []

    def initialize(self, initial_pixels: torch.Tensor) -> None:
        """Seed VAE encoder + KV cache from the first 9 px frames."""
        assert initial_pixels.shape[2] == INITIAL_PX_FRAMES, (
            f"initial_pixels expects {INITIAL_PX_FRAMES} frames, got {initial_pixels.shape[2]}"
        )
        self.pipeline._initialize_kv_cache(batch_size=self.batch_size, dtype=self.dtype, device=self.device)
        self.pipeline._initialize_crossattn_cache(batch_size=self.batch_size, dtype=self.dtype, device=self.device)
        self.pipeline.vae.model.clear_cache()
        self.current_start_frame = 0
        self.accumulated_latents = []
        self.encode_times = []
        self.step_times = []

        latents = self._encode(initial_pixels)
        self.accumulated_latents.append(latents)
        self._inject(latents)
        self.current_start_frame += INITIAL_LATENTS

    def step(self, new_pixels: torch.Tensor) -> torch.Tensor:
        """Streaming tick: encode 12 incoming frames, inject into KV, predict 1 block.

        new_pixels: (1, C, 12, H, W). Returns the predicted 3 latents.
        """
        assert new_pixels.shape[2] == FRAMES_PER_BLOCK, (
            f"step expects {FRAMES_PER_BLOCK} frames per call, got {new_pixels.shape[2]}"
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. Sensor arrives: encode + inject into KV at the slot of the *next* prediction
        new_latents = self._encode(new_pixels)
        self.accumulated_latents.append(new_latents)
        self._inject(new_latents)
        self.current_start_frame += LATENTS_PER_BLOCK

        # 2. Predict the next 0.75s
        prediction = self._predict()

        torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - t0)
        return prediction

    def step_initial(self) -> torch.Tensor:
        """Variant of step() that produces the very first prediction (no sensor injection
        beyond the initial seed). Used because at k=0 we predict before any new sensor has
        arrived past the seed window."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prediction = self._predict()
        torch.cuda.synchronize()
        self.step_times.append(time.perf_counter() - t0)
        return prediction

    @property
    def all_sensor_latents(self) -> torch.Tensor:
        return torch.cat(self.accumulated_latents, dim=1)

    def finish(self) -> None:
        self.pipeline.vae.model.clear_cache()

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


def decode_prediction_block(pipeline, all_sensor_latents: torch.Tensor,
                            prediction: torch.Tensor, block_idx: int) -> torch.Tensor:
    """Decode a 3-latent prediction in the context of all preceding sensor latents.

    Returns (12, C, H, W) in [0, 1].
    """
    end_slot = INITIAL_LATENTS + LATENTS_PER_BLOCK * block_idx
    context = all_sensor_latents[:, :end_slot]
    full_seq = torch.cat([context, prediction], dim=1)
    pipeline.vae.model.clear_cache()
    decoded = pipeline.vae.decode_to_pixel(full_seq, use_cache=False)
    pipeline.vae.model.clear_cache()
    decoded = (decoded * 0.5 + 0.5).clamp(0, 1)  # (1, T_px, C, H, W)
    return decoded[0, -FRAMES_PER_BLOCK:]


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
    parser.add_argument(
        "--delta_t", type=float, default=0.5, choices=[0.25, 0.5, 0.75],
        help="Prediction lookahead in seconds (0.25 / 0.50 / 0.75).",
    )
    parser.add_argument("--start_index", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Number of sensor frames to consume. "
                             "Default: use as many as fit in the KV cache (= 81 frames = 6 blocks).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--output_fps", type=int, default=SOURCE_FPS,
                        help="Side-by-side video fps (12 frames per block; 16 = real-time).")
    args = parser.parse_args()

    max_blocks = (KV_CACHE_LATENT_LIMIT - INITIAL_LATENTS) // LATENTS_PER_BLOCK  # 6
    delta_idx = DELTA_T_TO_GEN_IDX[args.delta_t]

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

    print(f"Streaming inference: {num_pred_blocks} blocks (Δt={args.delta_t}s)")
    predictor = StreamingPredictor(pipeline, conditional_dict, device)

    # Seed: encode first 9 frames and inject into KV cache, then make the first
    # prediction directly on the seed (no new sensor has arrived yet).
    predictor.initialize(sensor_tensor[:, :, :INITIAL_PX_FRAMES])
    predictions: list[torch.Tensor] = [predictor.step_initial()]

    # Streaming loop: each iteration ingests the next 12 px frames and predicts.
    last_ingest_start = INITIAL_PX_FRAMES + FRAMES_PER_BLOCK * (num_pred_blocks - 1)
    for start_px in range(INITIAL_PX_FRAMES, last_ingest_start, FRAMES_PER_BLOCK):
        new_pixels = sensor_tensor[:, :, start_px:start_px + FRAMES_PER_BLOCK]
        predictions.append(predictor.step(new_pixels))
    predictor.finish()

    block_times = predictor.step_times
    encode_times = predictor.encode_times
    all_sensor_latents = predictor.all_sensor_latents

    avg_block = sum(block_times) / len(block_times)
    avg_enc = sum(encode_times) / len(encode_times)
    print(
        f"  blocks total {sum(block_times):.2f}s | per-block {avg_block:.2f}s\n"
        f"  encode total {sum(encode_times):.2f}s ({len(encode_times)} calls, "
        f"per-call {avg_enc:.3f}s)"
    )

    print("Decoding prediction blocks...")
    decoded_blocks: list[torch.Tensor] = []
    decode_start = time.perf_counter()
    for k, pred in enumerate(predictions):
        decoded_blocks.append(decode_prediction_block(pipeline, all_sensor_latents, pred, k))
    decode_time = time.perf_counter() - decode_start
    print(f"  decode {decode_time:.2f}s ({decode_time / len(decoded_blocks):.2f}s/block)")

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

            if i == delta_idx:
                cv2.rectangle(
                    pred_frame, (0, 0),
                    (pred_frame.shape[1] - 1, pred_frame.shape[0] - 1),
                    (255, 255, 0), thickness=6,
                )

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
    write_video(str(args.output), out, fps=args.output_fps)

    sensor_dt = FRAMES_PER_BLOCK / SOURCE_FPS  # 0.75s per block
    print(
        f"\nWrote {args.output} ({len(composed)} frames @ {args.output_fps} fps)\n"
        f"Per-block wall {avg_block:.2f}s vs sensor {sensor_dt:.2f}s "
        f"→ {avg_block / sensor_dt:.2f}× real-time ({sensor_dt / avg_block:.2f}× speed)"
    )


if __name__ == "__main__":
    main()
