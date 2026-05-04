from typing import Tuple

import torch
from torch import nn

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusion(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}))
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        """Sample a per-frame timestep tensor of shape [batch_size, num_frame].

        Frames within the same block share the same timestep (block-uniform noise level).
        """
        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.long,
        )
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Add noise at a random per-block timestep and minimise flow-matching MSE.

        Output:
            loss : scalar tensor (training-weighted MSE between predicted flow and target)
            log_dict : intermediate tensors for downstream logging (x0, x0_pred)
        """
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        log_dict = {"x0": clean_latent.detach(), "x0_pred": x0_pred.detach()}
        return loss, log_dict
