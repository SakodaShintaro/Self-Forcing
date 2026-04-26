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


def streaming_kv_inference(pipeline, sensor_tensor: torch.Tensor,
                           conditional_dict, num_pred_blocks: int, device: torch.device):
    """Streaming inference with per-block sensor encoding and KV cache injection.

    sensor_tensor: (1, C, T, H, W). For each block k = 0..num_pred_blocks-1:
        1. Generate the block's prediction (KV cache holds prior sensor latents).
        2. If k < last: take the next 12 px frames, encode them with the VAE in
           streaming mode (use_cache=True), and write them into the KV cache slot
           that the prediction would otherwise have occupied.

    Returns (predictions, block_times, encode_times, all_sensor_latents).
    """
    batch_size = 1
    dtype = torch.bfloat16

    pipeline._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
    pipeline._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
    pipeline.vae.model.clear_cache()  # fresh streaming VAE state

    encode_times: list[float] = []
    accumulated_latents: list[torch.Tensor] = []

    # === Initial 9 frames → 3 latents (seed the VAE feat_cache) ===
    initial_pixels = sensor_tensor[:, :, :INITIAL_PX_FRAMES].to(dtype=dtype)
    torch.cuda.synchronize()
    t_e = time.perf_counter()
    initial_latents = pipeline.vae.encode_to_latent(initial_pixels, use_cache=True).to(dtype=dtype)
    torch.cuda.synchronize()
    encode_times.append(time.perf_counter() - t_e)
    accumulated_latents.append(initial_latents)

    # Cache initial sensor latents into KV cache slot 0..2
    current_start_frame = 0
    init_timestep = torch.zeros([batch_size, 1], dtype=torch.int64, device=device)
    pipeline.generator(
        noisy_image_or_video=initial_latents,
        conditional_dict=conditional_dict,
        timestep=init_timestep,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start_frame * pipeline.frame_seq_length,
    )
    current_start_frame += INITIAL_LATENTS

    predictions: list[torch.Tensor] = []
    block_times: list[float] = []
    context_noise = int(getattr(pipeline.args, "context_noise", 0))

    for k in range(num_pred_blocks):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        # --- Denoising loop on fresh noise (3 latents) ---
        sampled_noise = torch.randn(
            [batch_size, LATENTS_PER_BLOCK, 16, 60, 104],
            device=device, dtype=dtype,
        )
        noisy_input = sampled_noise

        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.full(
                [batch_size, LATENTS_PER_BLOCK],
                int(current_timestep.item()),
                device=device, dtype=torch.int64,
            )
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            if index < len(pipeline.denoising_step_list) - 1:
                next_timestep = pipeline.denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * LATENTS_PER_BLOCK], device=device, dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        predictions.append(denoised_pred.detach().clone())

        # --- Streaming sensor arrival: encode next 12 frames + inject ---
        if k < num_pred_blocks - 1:
            start_px = INITIAL_PX_FRAMES + FRAMES_PER_BLOCK * k
            end_px = start_px + FRAMES_PER_BLOCK
            next_pixels = sensor_tensor[:, :, start_px:end_px].to(dtype=dtype)

            torch.cuda.synchronize()
            t_e = time.perf_counter()
            next_latents = pipeline.vae.encode_to_latent(next_pixels, use_cache=True).to(dtype=dtype)
            torch.cuda.synchronize()
            encode_times.append(time.perf_counter() - t_e)
            accumulated_latents.append(next_latents)

            replace_timestep = torch.full(
                [batch_size, 1], context_noise, dtype=torch.int64, device=device,
            )
            pipeline.generator(
                noisy_image_or_video=next_latents,
                conditional_dict=conditional_dict,
                timestep=replace_timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            current_start_frame += LATENTS_PER_BLOCK

        torch.cuda.synchronize()
        block_times.append(time.perf_counter() - t_start)

    pipeline.vae.model.clear_cache()  # release encoder feat_cache after streaming
    all_sensor_latents = torch.cat(accumulated_latents, dim=1)
    return predictions, block_times, encode_times, all_sensor_latents


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
    parser.add_argument("--num_pred_blocks", type=int, default=6,
                        help="Number of prediction steps. Max 6 with self_forcing_dmd.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--output_fps", type=int, default=SOURCE_FPS,
                        help="Side-by-side video fps (12 frames per block; 16 = real-time).")
    args = parser.parse_args()

    max_blocks = (KV_CACHE_LATENT_LIMIT - INITIAL_LATENTS) // LATENTS_PER_BLOCK  # 6
    if args.num_pred_blocks > max_blocks:
        raise ValueError(f"num_pred_blocks > {max_blocks} exceeds the 21-latent KV cache budget")

    delta_idx = DELTA_T_TO_GEN_IDX[args.delta_t]
    delta_t_frames = int(round(args.delta_t * SOURCE_FPS))

    set_seed(args.seed)
    device = torch.device("cuda")
    pipeline, _ = build_pipeline(args, device)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    png_paths = sorted(args.src.glob("*.png"))
    num_frames_total = INITIAL_PX_FRAMES + FRAMES_PER_BLOCK * args.num_pred_blocks
    if args.start_index + num_frames_total > len(png_paths):
        raise ValueError(
            f"Not enough sensor frames at {args.src}: need {num_frames_total} from index "
            f"{args.start_index}, but only {len(png_paths)} present."
        )

    print(f"Loading {num_frames_total} sensor frames from {args.src.name}...")
    sensor_pixel_frames = [
        fit_letterbox(Image.open(png_paths[i]).convert("RGB"))
        for i in range(args.start_index, args.start_index + num_frames_total)
    ]
    sensor_tensor = torch.stack(
        [normalize(im) for im in sensor_pixel_frames], dim=1
    ).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    conditional_dict = pipeline.text_encoder(text_prompts=[args.caption])

    print(f"Streaming inference: {args.num_pred_blocks} blocks (Δt={args.delta_t}s)")
    predictions, block_times, encode_times, all_sensor_latents = streaming_kv_inference(
        pipeline, sensor_tensor, conditional_dict, args.num_pred_blocks, device,
    )
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
                f"block {k+1}/{args.num_pred_blocks} frame {i+1}/{FRAMES_PER_BLOCK} | gen {block_times[k]:.2f}s",
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
