import gc
import logging
import os
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from model import CausalDiffusion
from utils.b2d_dataset import Bench2DriveLatentDataset
from utils.dataset import ShardingLMDBDataset, cycle
from utils.distributed import EMA_FSDP, barrier, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from utils.misc import resolve_checkpoint_path, set_seed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            # Only call wandb.login() when an explicit key is provided; otherwise
            # let wandb fall back to its cached credentials (e.g. ~/.netrc).
            if config.wandb_key:
                wandb.login(host=config.wandb_host or None, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity or None,
                project=config.wandb_project or "self-forcing",
                dir=config.wandb_save_dir or None,
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model
        self.model = CausalDiffusion(config, device=self.device)

        # Step 2.1: Load pretrained generator weights BEFORE LoRA / FSDP wrap.
        # (LoRA wrap renames base linear keys, so the upstream checkpoint must be
        # applied to the bare CausalWanModel first.)
        if getattr(config, "generator_ckpt", False):
            ckpt_path = resolve_checkpoint_path(config.generator_ckpt)
            print(f"Loading pretrained generator from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            ckpt_key = getattr(config, "generator_ckpt_key", None)
            if ckpt_key and ckpt_key in state_dict:
                state_dict = state_dict[ckpt_key]
            elif "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            cleaned = {
                k.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }
            self.model.generator.load_state_dict(cleaned, strict=True)

        # Step 2.2: Attach LoRA adapters (after base load, before FSDP).
        # This trainer is LoRA-only -- config must define a `lora` block.
        lora_cfg = getattr(config, "lora", None)
        if not (lora_cfg and lora_cfg.get("enabled", False)):
            raise ValueError("DiffusionTrainer requires `lora.enabled: true` in the config.")
        from peft import LoraConfig, get_peft_model

        self.model.generator.model.requires_grad_(False)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.rank),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(lora_cfg.target_modules),
            bias="none",
        )
        self.model.generator.model = get_peft_model(self.model.generator.model, peft_cfg)
        if dist.get_rank() == 0:
            self.model.generator.model.print_trainable_parameters()

        # Step 2.3: FSDP wrap (LoRA layers get wrapped together with the base)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
            )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # Step 3: Initialize the dataloader
        dataset_type = getattr(config, "dataset_type", "lmdb")
        if dataset_type == "b2d_latent":
            dataset = Bench2DriveLatentDataset(
                b2d_root=config.b2d_root,
                split=config.b2d_split,
                num_frames=config.image_or_video_shape[1],
                fixed_caption=config.b2d_caption,
            )
        else:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 3.1: Optional validation loader (b2d_latent only).
        # Driven by config.valid_iters (0 disables) and config.valid_batches.
        self.valid_iters = int(getattr(config, "valid_iters", 0) or 0)
        self.valid_batches = int(getattr(config, "valid_batches", 0) or 0)
        self.valid_dataloader = None
        if dataset_type == "b2d_latent" and self.valid_iters > 0 and self.valid_batches > 0:
            valid_dataset = Bench2DriveLatentDataset(
                b2d_root=config.b2d_root,
                split="valid",
                num_frames=config.image_or_video_shape[1],
                fixed_caption=config.b2d_caption,
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset, shuffle=False, drop_last=False
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                sampler=valid_sampler,
                num_workers=2,
            )
            if dist.get_rank() == 0:
                print(
                    f"VALID DATASET SIZE {len(valid_dataset)} "
                    f"(valid_batches={self.valid_batches} every {self.valid_iters} steps)"
                )

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = lambda name: (
            name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(
                self.model.generator, decay=ema_weight, trainable_only=True
            )

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)

        # LoRA-only save: drop frozen base weights so checkpoints stay tiny.
        # Match by canonical (renamed) name to be robust to FSDP / checkpoint /
        # torch.compile wrapper prefixes.
        def _rename(k: str) -> str:
            return (
                k.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("_orig_mod.", "")
            )

        trainable_renamed = set(self.name_to_trainable_params.keys())
        generator_state_dict = {
            k: v for k, v in generator_state_dict.items() if _rename(k) in trainable_renamed
        }

        state_dict = {"generator": generator_state_dict}
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main_process:
            os.makedirs(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True
            )
            torch.save(
                state_dict,
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"),
            )
            print(
                "Model saved to",
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"),
            )

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(frames).to(
                    device=self.device, dtype=self.dtype
                )
        image_latent = clean_latent[
            :,
            0:1,
        ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
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
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

        # Step 5: Lazily build the EMA shadow at ema_start_step, then update each iter.
        ema_weight = self.config.ema_weight
        if ema_weight and ema_weight > 0.0 and self.step >= self.config.ema_start_step:
            if self.generator_ema is None:
                if self.is_main_process:
                    print(
                        f"Setting up EMA with weight {ema_weight} at step {self.step}",
                        flush=True,
                    )
                self.generator_ema = EMA_FSDP(
                    self.model.generator, decay=ema_weight, trainable_only=True
                )
            else:
                self.generator_ema.update(self.model.generator)

    @torch.no_grad()
    def validate(self):
        """Run a fixed number of forward-only generator_loss evaluations on the
        valid split and return the cross-rank mean loss (or None if disabled)."""
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
                image_latent = clean_latent[
                    :,
                    0:1,
                ]
                batch_size = len(text_prompts)
                shape = list(image_or_video_shape)
                shape[0] = batch_size

                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
                unconditional_dict = getattr(self, "unconditional_dict", None)
                if unconditional_dict is None:
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size
                    )
                    unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict

                loss, _ = self.model.generator_loss(
                    image_or_video_shape=shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent,
                )
                losses.append(loss.detach().float())
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
            self.model.generator.train()

        local_count = torch.tensor([float(len(losses))], device=self.device, dtype=torch.float32)
        local_sum = (
            torch.stack(losses).sum()
            if losses
            else torch.zeros((), device=self.device, dtype=torch.float32)
        )
        local_sum = local_sum.float().reshape(1)

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        denom = local_count.clamp(min=1.0)
        return (local_sum / denom).item()

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        sampled_noise = torch.randn([batch_size, 21, 16, 60, 104], device="cuda", dtype=self.dtype)
        video, _ = pipeline.inference(
            noise=sampled_noise, text_prompts=prompts, return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        max_steps = getattr(self.config, "max_steps", None)
        while max_steps is None or self.step < max_steps:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.valid_dataloader is not None and self.step % self.valid_iters == 0:
                torch.cuda.empty_cache()
                val_loss = self.validate()
                torch.cuda.empty_cache()
                if val_loss is not None and self.is_main_process:
                    print(f"step={self.step} val_loss={val_loss:.4f}", flush=True)
                    if not self.disable_wandb:
                        wandb.log({"val/loss": val_loss}, step=self.step)
                # Don't let validation time pollute the next iteration's iter_time.
                self.previous_time = time.time()

            barrier()
            if self.is_main_process:
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
        # Always save the final checkpoint at exit if we ran to a hard step cap.
        if max_steps is not None and not self.config.no_save:
            torch.cuda.empty_cache()
            self.save()
