import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.training_model import CausalDiffusion  # noqa: E402
from utils.b2d_dataset import Bench2DriveLatentDataset  # noqa: E402
from utils.misc import (  # noqa: E402
    load_generator_state_dict,
    resolve_checkpoint_path,
    set_seed,
    strip_wrap_prefixes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Parent dir for run output. Each invocation creates "
        "<root_dir>/<YYYYmmdd_HHMMSS>_<config_name>/.",
    )
    parser.add_argument(
        "--b2d_root",
        type=str,
        required=True,
        help="Bench2Drive root directory (contains splits.json and latents/{train,valid}/).",
    )
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """Single-GPU EMA shadow over a subset of named_parameters.

    With trainable_only=True only requires_grad params (e.g. LoRA adapters)
    are tracked, keeping the shadow tiny.
    """

    def __init__(self, module: torch.nn.Module, decay: float, trainable_only: bool):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for n, p in module.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for n, p in module.named_parameters():
            if n not in self.shadow:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach().float().cpu(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


class Trainer:
    def __init__(
        self,
        config,
        *,
        run_dir: str,
        config_name: str,
        b2d_root: str,
        no_save: bool,
        disable_wandb: bool,
    ):
        self.config = config
        self.run_dir = run_dir
        self.no_save = no_save
        self.disable_wandb = disable_wandb
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.device("cuda")

        seed = config.seed if config.seed != 0 else int(torch.randint(0, 10_000_000, (1,)).item())
        set_seed(seed)

        if not self.disable_wandb:
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config_name,
                mode="online",
                project="self-forcing",
                dir=run_dir,
            )

        # Step 2: Initialize the model
        self.model = CausalDiffusion(
            device=self.device,
            timestep_shift=config.timestep_shift,
            num_frame_per_block=config.num_frame_per_block,
            mixed_precision=config.mixed_precision,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        # Step 2.1: Load pretrained generator weights BEFORE LoRA wrap.
        # (LoRA wrap renames base linear keys, so the upstream checkpoint must be
        # applied to the bare CausalWanModel first.)
        ckpt_path = resolve_checkpoint_path(config.generator_ckpt)
        print(f"Loading pretrained generator from {ckpt_path}")
        cleaned = load_generator_state_dict(ckpt_path, explicit_key=config.generator_ckpt_key)
        self.model.generator.load_state_dict(cleaned, strict=True)

        # Step 2.2: Attach LoRA adapters.
        if not config.lora.enabled:
            raise ValueError("Trainer requires `lora.enabled: true` in the config.")
        from peft import LoraConfig, get_peft_model

        self.model.generator.model.requires_grad_(False)
        peft_cfg = LoraConfig(
            r=int(config.lora.rank),
            lora_alpha=int(config.lora.alpha),
            lora_dropout=float(config.lora.dropout),
            target_modules=list(config.lora.target_modules),
            bias="none",
        )
        self.model.generator.model = get_peft_model(self.model.generator.model, peft_cfg)
        self.model.generator.model.print_trainable_parameters()

        # Step 2.3: Move generator + text_encoder to GPU at the chosen dtype.
        self.model.generator.to(device=self.device, dtype=self.dtype)
        self.model.text_encoder.to(device=self.device, dtype=self.dtype)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # Step 3: Initialize the dataloader
        dataset = Bench2DriveLatentDataset(
            b2d_root=b2d_root,
            split="train",
            num_frames=config.image_or_video_shape[1],
            fixed_caption=config.b2d_caption,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        print(f"DATASET SIZE {len(dataset)}")
        self.dataloader = cycle(dataloader)

        # Step 3.1: Optional validation loader.
        # Driven by config.valid_iters (0 disables) and config.valid_batches.
        self.valid_iters = int(config.valid_iters)
        self.valid_batches = int(config.valid_batches)
        self.valid_dataloader = None
        if self.valid_iters > 0 and self.valid_batches > 0:
            valid_dataset = Bench2DriveLatentDataset(
                b2d_root=b2d_root,
                split="valid",
                num_frames=config.image_or_video_shape[1],
                fixed_caption=config.b2d_caption,
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=2,
            )
            print(
                f"VALID DATASET SIZE {len(valid_dataset)} "
                f"(valid_batches={self.valid_batches} every {self.valid_iters} steps)"
            )

        # Step 4: trainable-param map for LoRA-only checkpoint filtering.
        # EMA shadow itself is built lazily once `step` reaches `ema_start_step`
        # (see train_one_step) so the first ema_start_step-1 steps don't pay for it.
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue
            self.name_to_trainable_params[strip_wrap_prefixes(n)] = p
        self.generator_ema: EMA | None = None

        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        # LoRA-only save: drop frozen base weights so checkpoints stay tiny.
        # Match by canonical (renamed) name to be robust to gradient-checkpoint /
        # torch.compile wrapper prefixes that PEFT can introduce.
        trainable_renamed = set(self.name_to_trainable_params.keys())
        generator_state_dict = {
            k: v
            for k, v in self.model.generator.state_dict().items()
            if strip_wrap_prefixes(k) in trainable_renamed
        }

        state_dict = {"generator": generator_state_dict}
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        ckpt_dir = os.path.join(self.run_dir, f"checkpoint_model_{self.step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        torch.save(state_dict, ckpt_path)
        print(f"Model saved to {ckpt_path}")

    def train_one_step(self, batch):
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts and precomputed latents
        text_prompts = batch["prompts"]
        clean_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional info
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            clean_latent=clean_latent,
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if not self.disable_wandb:
            wandb.log(wandb_loss_dict, step=self.step)
        now = time.time()
        iter_time = (now - self.previous_time) if self.previous_time is not None else 0.0
        print(
            f"step={self.step} loss={wandb_loss_dict['generator_loss']:.4f} "
            f"grad_norm={wandb_loss_dict['generator_grad_norm']:.3f} "
            f"iter_time={iter_time:.2f}s",
            flush=True,
        )

        if self.step % self.config.gc_interval == 0:
            gc.collect()

        # Step 5: Lazily build the EMA shadow at ema_start_step, then update each iter.
        ema_weight = self.config.ema_weight
        if ema_weight > 0.0 and self.step >= self.config.ema_start_step:
            if self.generator_ema is None:
                print(f"Setting up EMA with weight {ema_weight} at step {self.step}", flush=True)
                self.generator_ema = EMA(
                    self.model.generator, decay=ema_weight, trainable_only=True
                )
            else:
                self.generator_ema.update(self.model.generator)

    @torch.no_grad()
    def validate(self) -> float | None:
        """Run a fixed number of forward-only generator_loss evaluations on the
        valid split and return the mean loss (or None if disabled)."""
        if self.valid_dataloader is None:
            return None

        self.model.generator.eval()
        # Reproducible val sampling: fix RNG so val numbers are comparable across
        # train steps (generator_loss internally samples timesteps and noise).
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        torch.manual_seed(self.config.seed + 1)
        torch.cuda.manual_seed(self.config.seed + 1)

        losses: list[torch.Tensor] = []
        image_or_video_shape = list(self.config.image_or_video_shape)
        try:
            for i, batch in enumerate(self.valid_dataloader):
                if i >= self.valid_batches:
                    break
                text_prompts = batch["prompts"]
                clean_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
                batch_size = len(text_prompts)
                shape = list(image_or_video_shape)
                shape[0] = batch_size

                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

                loss, _ = self.model.generator_loss(
                    image_or_video_shape=shape,
                    conditional_dict=conditional_dict,
                    clean_latent=clean_latent,
                )
                losses.append(loss.detach().float())
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
            self.model.generator.train()

        if not losses:
            return None
        return torch.stack(losses).mean().item()

    def train(self):
        max_steps = self.config.max_steps
        while self.step < max_steps:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            if (not self.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.valid_dataloader is not None and self.step % self.valid_iters == 0:
                torch.cuda.empty_cache()
                val_loss = self.validate()
                torch.cuda.empty_cache()
                if val_loss is not None:
                    print(f"step={self.step} val_loss={val_loss:.4f}", flush=True)
                    if not self.disable_wandb:
                        wandb.log({"val/loss": val_loss}, step=self.step)
                # Don't let validation time pollute the next iteration's iter_time.
                self.previous_time = time.time()

            current_time = time.time()
            if self.previous_time is None:
                self.previous_time = current_time
            else:
                if not self.disable_wandb:
                    wandb.log(
                        {"per iteration time": current_time - self.previous_time},
                        step=self.step,
                    )
                self.previous_time = current_time
        # Always save the final checkpoint at exit.
        if not self.no_save:
            torch.cuda.empty_cache()
            self.save()


def main():
    args = parse_args()

    config_name = os.path.basename(args.config_path).split(".")[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.root_dir, f"{stamp}_{config_name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}", flush=True)

    config = OmegaConf.load(args.config_path)

    trainer = Trainer(
        config,
        run_dir=run_dir,
        config_name=config_name,
        b2d_root=args.b2d_root,
        no_save=args.no_save,
        disable_wandb=args.disable_wandb,
    )
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
