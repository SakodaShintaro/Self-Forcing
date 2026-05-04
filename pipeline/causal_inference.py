from typing import List

import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(**args.model_kwargs)
        self.text_encoder = WanTextEncoder()
        self.vae = WanVAEWrapper()

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        # Inference uses sigma-warped denoising steps (training uses raw).
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        timesteps = torch.cat(
            (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
        )
        self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.context_noise = args.context_noise
        self.num_frame_per_block = args.num_frame_per_block
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run KV-cached causal inference seeded with `initial_latent`.

        Inputs:
            noise:          (B, num_output_frames, C, H, W)
            text_prompts:   list of length B
            initial_latent: (B, num_input_frames, C, H, W) — context block(s) to
                            seed the KV cache before sampling continues from `noise`.
        Returns:
            (video, latents) where video is in [0, 1] and latents is the raw
            denoised tensor (B, num_output_frames + num_input_frames, C, H, W).
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1]
        assert num_input_frames % self.num_frame_per_block == 0
        num_input_blocks = num_input_frames // self.num_frame_per_block
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device
                )

        # Step 2: Push initial latent blocks through the generator at t=0 to seed
        # the KV cache. Their decoded output is just the GT context, copied straight
        # into `output`.
        current_start_frame = 0
        zero_timestep = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64)
        for _ in range(num_input_blocks):
            current_ref_latents = initial_latent[
                :, current_start_frame : current_start_frame + self.num_frame_per_block
            ]
            output[:, current_start_frame : current_start_frame + self.num_frame_per_block] = (
                current_ref_latents
            )
            self.generator(
                noisy_image_or_video=current_ref_latents,
                conditional_dict=conditional_dict,
                timestep=zero_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        for current_num_frames in [self.num_frame_per_block] * num_blocks:
            noisy_input = noise[
                :,
                current_start_frame - num_input_frames : current_start_frame
                + current_num_frames
                - num_input_frames,
            ]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.int64
                    )
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame : current_start_frame + current_num_frames] = (
                denoised_pred
            )

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache
