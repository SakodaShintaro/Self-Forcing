"""Validation metrics for a Self-Forcing checkpoint on bench2drive valid latents.

Two metrics:
  1. Held-out denoising loss (cheap): forward-pass only, average over N valid clips.
     Uses the same flow-matching target as training.
  2. I2V rollout PSNR (optional, slower): for each clip, take the first 3 latents
     as context and predict the rest with CausalInferencePipeline. Decode both
     prediction and ground truth through the VAE and compute per-pixel PSNR.

Run:
  uv run python scripts/b2d_eval.py \
    --config_path configs/b2d_finetune.yaml \
    --checkpoint_path logs/b2d_finetune/checkpoint_model_001000/model.pt \
    --num_clips 64 [--rollout_psnr]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.misc import resolve_checkpoint_path, set_seed  # noqa: E402
from utils.scheduler import FlowMatchScheduler  # noqa: E402
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper  # noqa: E402


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


def load_generator(
    base_ckpt: str | None,
    lora_cfg: dict | None,
    finetune_ckpt: str | None,
    model_kwargs: dict,
    device: torch.device,
) -> WanDiffusionWrapper:
    """Load generator. Order: base ckpt -> apply LoRA wrap -> overlay finetune ckpt."""
    gen = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    if base_ckpt:
        gen.load_state_dict(_load_state_dict(resolve_checkpoint_path(base_ckpt)), strict=True)

    use_lora = bool(lora_cfg and lora_cfg.get("enabled", False))
    if use_lora:
        from peft import LoraConfig, get_peft_model

        gen.model.requires_grad_(False)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("rank", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(
                lora_cfg.get(
                    "target_modules",
                    [
                        "self_attn.q",
                        "self_attn.k",
                        "self_attn.v",
                        "self_attn.o",
                        "cross_attn.q",
                        "cross_attn.k",
                        "cross_attn.v",
                        "cross_attn.o",
                    ],
                )
            ),
            bias="none",
        )
        gen.model = get_peft_model(gen.model, peft_cfg)

    if finetune_ckpt:
        sd = _load_state_dict(resolve_checkpoint_path(finetune_ckpt))
        missing, unexpected = gen.load_state_dict(sd, strict=False)
        # Allow missing base weights (already loaded above) but flag any unexpected keys
        unexpected_non_trivial = [k for k in unexpected if not k.endswith(".base_layer.weight")]
        if unexpected_non_trivial:
            print(
                f"warning: unexpected keys when loading {finetune_ckpt}: {unexpected_non_trivial[:5]}..."
            )

    gen = gen.to(device=device, dtype=torch.bfloat16).eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen


def collect_clips(
    b2d_root: Path, split: str, num_frames: int, num_clips: int, seed: int
) -> list[tuple[str, int, torch.Tensor]]:
    """Pick `num_clips` random (episode, start, latent_clip) tuples from valid split."""
    rng = random.Random(seed)
    splits_path = b2d_root / "splits.json"
    latent_dir = b2d_root / "latents" / split
    with open(splits_path) as f:
        episodes = json.load(f)[split]
    eligible = [ep for ep in episodes if (latent_dir / f"{ep}.pt").exists()]
    if not eligible:
        raise RuntimeError(f"No latent files for split={split} in {latent_dir}")

    out = []
    for _ in range(num_clips):
        ep = rng.choice(eligible)
        latents = torch.load(latent_dir / f"{ep}.pt", map_location="cpu", weights_only=True)
        T = latents.shape[0]
        if T < num_frames:
            continue
        s = rng.randint(0, T - num_frames)
        out.append((ep, s, latents[s : s + num_frames].contiguous()))
    return out


@torch.no_grad()
def denoising_loss(
    generator: WanDiffusionWrapper,
    scheduler: FlowMatchScheduler,
    text_emb: dict,
    clips: list,
    device: torch.device,
) -> dict:
    """Average flow-matching MSE over the given clips at random timesteps."""
    losses = []
    per_t_losses = []
    for ep, s, clip in clips:
        clean = clip.unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # (1, F, 16, 60, 104)
        noise = torch.randn_like(clean)
        F = clean.shape[1]
        # Sample one timestep per latent frame (matches generator_loss without uniformity)
        t_idx = torch.randint(0, scheduler.num_train_timesteps, (1, F), device=device)
        timestep = scheduler.timesteps[t_idx].to(dtype=torch.bfloat16, device=device)
        noisy = scheduler.add_noise(
            clean.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (1, F))
        target = scheduler.training_target(clean, noise, timestep)
        flow_pred, _ = generator(
            noisy_image_or_video=noisy,
            conditional_dict=text_emb,
            timestep=timestep,
        )
        per_clip = torch.nn.functional.mse_loss(
            flow_pred.float(), target.float(), reduction="none"
        ).mean()
        losses.append(per_clip.item())
        per_t_losses.append((timestep.float().mean().item(), per_clip.item()))

    losses = torch.tensor(losses)
    return {
        "n_clips": len(losses),
        "mean_loss": losses.mean().item(),
        "median_loss": losses.median().item(),
        "stdev_loss": losses.std().item() if len(losses) > 1 else 0.0,
    }


@torch.no_grad()
def rollout_psnr(
    generator: WanDiffusionWrapper,
    vae: WanVAEWrapper,
    scheduler: FlowMatchScheduler,
    text_emb: dict,
    clips: list,
    num_input_latents: int,
    denoising_steps: list[int],
    device: torch.device,
    num_frame_per_block: int,
) -> dict:
    """Run causal I2V rollout with `num_input_latents` of context, decode, compute PSNR vs GT."""
    from pipeline.causal_inference import CausalInferencePipeline

    # Wrap into a CausalInferencePipeline lookalike using the existing class
    cfg = OmegaConf.create(
        {
            "denoising_step_list": denoising_steps,
            "warp_denoising_step": True,
            "context_noise": 0,
            "num_frame_per_block": num_frame_per_block,
            "independent_first_frame": False,
            "model_kwargs": {"timestep_shift": scheduler.shift},
            "i2v": True,
        }
    )
    pipeline = CausalInferencePipeline.__new__(CausalInferencePipeline)
    # Initialize the nn.Module dicts (_parameters / _modules / _buffers) without
    # running CausalInferencePipeline.__init__ (which would re-load models).
    torch.nn.Module.__init__(pipeline)
    # Manually set up pipeline state without re-loading models
    pipeline.generator = generator
    pipeline.text_encoder = None  # we use cached text_emb directly
    pipeline.vae = vae
    pipeline.scheduler = generator.get_scheduler()
    pipeline.denoising_step_list = torch.tensor(denoising_steps, dtype=torch.long)
    timesteps_full = torch.cat(
        (pipeline.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    pipeline.denoising_step_list = timesteps_full[1000 - pipeline.denoising_step_list]
    pipeline.num_transformer_blocks = 30
    pipeline.frame_seq_length = 1560
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.args = cfg
    pipeline.num_frame_per_block = num_frame_per_block
    pipeline.independent_first_frame = False
    pipeline.local_attn_size = generator.model.local_attn_size
    if num_frame_per_block > 1:
        pipeline.generator.model.num_frame_per_block = num_frame_per_block

    psnrs = []
    for ep, s, clip in clips:
        clean = clip.unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # (1, F, 16, 60, 104)
        F = clean.shape[1]
        initial_latent = clean[:, :num_input_latents]
        sampled_noise = torch.randn(
            (1, F - num_input_latents, 16, 60, 104), device=device, dtype=torch.bfloat16
        )
        # Patch text_encoder usage by short-circuiting (we hand text_emb directly)
        original_text_encoder = pipeline.text_encoder

        def _enc(text_prompts):
            return text_emb

        class _Stub:
            def __call__(self, text_prompts):
                return text_emb

        pipeline.text_encoder = _Stub()
        try:
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[""],
                initial_latent=initial_latent,
                return_latents=True,
            )
        finally:
            pipeline.text_encoder = original_text_encoder
        # video: (1, T_pix, 3, H, W) in [0,1]; decode GT for comparison.
        # VAE was loaded as bf16, so feed bf16 latents (matches `clean`'s dtype).
        gt_video = vae.decode_to_pixel(clean, use_cache=False)
        gt_video = (gt_video * 0.5 + 0.5).clamp(0, 1)
        # Compare only the predicted region (skip the initial-latent pixel range)
        T_pred_pix = video.shape[1] - (1 + 4 * (num_input_latents - 1))
        pred_pix = video[:, -T_pred_pix:].float()
        gt_pix = gt_video[:, -T_pred_pix:].float()
        mse = (pred_pix - gt_pix).pow(2).mean().item()
        if mse <= 0:
            psnr = float("inf")
        else:
            psnr = 10.0 * math.log10(1.0 / mse)
        psnrs.append(psnr)

    psnrs_t = torch.tensor(psnrs)
    return {
        "n_clips": len(psnrs),
        "mean_psnr": psnrs_t.mean().item(),
        "median_psnr": psnrs_t.median().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional checkpoint to load. If omitted, evaluates the pretrained init.",
    )
    parser.add_argument("--num_clips", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout_psnr",
        action="store_true",
        help="Also run I2V rollout and compute PSNR (slower).",
    )
    parser.add_argument("--num_input_latents", type=int, default=3)
    parser.add_argument("--denoising_steps", type=int, nargs="+", default=[1000, 750, 500, 250])
    parser.add_argument(
        "--b2d_root",
        type=str,
        required=True,
        help="Bench2Drive root (contains splits.json and latents/{train,valid}/). "
        "Overrides config.b2d_root.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.b2d_root = args.b2d_root

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    # Load generator: base ckpt (config.generator_ckpt) + optional LoRA wrap + optional fine-tuned ckpt.
    base_ckpt = getattr(config, "generator_ckpt", None)
    lora_cfg = getattr(config, "lora", None)
    generator = load_generator(
        base_ckpt=base_ckpt,
        lora_cfg=lora_cfg,
        finetune_ckpt=args.checkpoint_path,
        model_kwargs=getattr(config, "model_kwargs", {}),
        device=device,
    )
    # `num_frame_per_block` lives on the underlying CausalWanModel; with peft it is exposed
    # via PeftModel.__getattr__.
    if config.num_frame_per_block > 1:
        base_model = generator.model
        # If wrapped in peft, drill down to the actual CausalWanModel.
        for attr in ("base_model",):
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                if hasattr(base_model, "model"):
                    base_model = base_model.model
                break
        base_model.num_frame_per_block = config.num_frame_per_block
    scheduler = generator.get_scheduler()
    scheduler.set_timesteps(1000, training=True)
    scheduler.timesteps = scheduler.timesteps.to(device=device)
    scheduler.sigmas = scheduler.sigmas.to(device=device)

    # Cache text embedding for the fixed caption
    print("encoding text")
    text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval()
    text_emb = text_encoder([getattr(config, "b2d_caption", "")])
    text_emb = {k: v.detach() for k, v in text_emb.items()}
    del text_encoder
    torch.cuda.empty_cache()

    # Collect valid clips
    print(f"collecting {args.num_clips} valid clips")
    clips = collect_clips(
        b2d_root=Path(config.b2d_root),
        split="valid",
        num_frames=config.image_or_video_shape[1],
        num_clips=args.num_clips,
        seed=args.seed,
    )
    print(f"got {len(clips)} clips")

    # 1) Denoising loss
    t0 = time.time()
    metrics = denoising_loss(generator, scheduler, text_emb, clips, device=device)
    metrics["denoising_loss_seconds"] = time.time() - t0
    print(json.dumps({"denoising": metrics}, indent=2))

    # 2) Rollout PSNR (optional)
    if args.rollout_psnr:
        print("running rollout (slower)")
        vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
        t0 = time.time()
        rollout = rollout_psnr(
            generator,
            vae,
            scheduler,
            text_emb,
            clips,
            num_input_latents=args.num_input_latents,
            denoising_steps=args.denoising_steps,
            device=device,
            num_frame_per_block=config.num_frame_per_block,
        )
        rollout["rollout_seconds"] = time.time() - t0
        print(json.dumps({"rollout_psnr": rollout}, indent=2))


if __name__ == "__main__":
    main()
