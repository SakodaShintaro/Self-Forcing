import os
from typing import List, Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from utils.scheduler import FlowMatchScheduler
from wan.modules.causal_model import CausalWanModel
from wan.modules.t5 import umt5_xxl
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import _video_vae

WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        t5_path = hf_hub_download(WAN_REPO_ID, "models_t5_umt5-xxl-enc-bf16.pth")
        self.text_encoder.load_state_dict(
            torch.load(t5_path, map_location="cpu", weights_only=False)
        )

        tokenizer_root = snapshot_download(WAN_REPO_ID, allow_patterns="google/umt5-xxl/*")
        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(tokenizer_root, "google/umt5-xxl/"), seq_len=512, clean="whitespace"
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        vae_path = hf_hub_download(WAN_REPO_ID, "Wan2.1_VAE.pth")
        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def _scale(self, device, dtype):
        return [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        # Each batch item gets a fresh encoder cache → equivalent to standalone
        # one-shot encode (no state carried across items).
        scale = self._scale(pixel.device, pixel.dtype)
        output = torch.stack(
            [
                self.model.encode(u.unsqueeze(0), scale, cache=self.model.make_encoder_cache())
                .float()
                .squeeze(0)
                for u in pixel
            ],
            dim=0,
        )
        # [B, C, T, H, W] -> [B, T, C, H, W]
        return output.permute(0, 2, 1, 3, 4)

    def make_encoder_cache(self):
        """Allocate a fresh per-session encoder cache for streaming use.

        Pass the returned list to ``encode_chunk()`` repeatedly to encode an
        episode chunk-by-chunk while preserving causal state across calls.
        """
        return self.model.make_encoder_cache()

    def encode_chunk(self, pixel: torch.Tensor, cache: list) -> torch.Tensor:
        """Encode one chunk of pixels in streaming mode.

        First call (cache freshly allocated) must have shape (B, C, 1+4*N, H, W);
        subsequent calls in the same session must have (B, C, 4*N, H, W). The
        cache is mutated in place. Returns latents in (B, T_lat, C_lat, H_lat, W_lat).
        """
        scale = self._scale(pixel.device, pixel.dtype)
        out = self.model.encode(pixel, scale, cache=cache).float()
        return out.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        zs = latent.permute(0, 2, 1, 3, 4)
        scale = self._scale(latent.device, latent.dtype)
        output = torch.stack(
            [self.model.decode(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0) for u in zs],
            dim=0,
        )
        # [B, C, T, H, W] -> [B, T, C, H, W]
        return output.permute(0, 2, 1, 3, 4)


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(self, timestep_shift):
        super().__init__()

        # All callers in this fork target Wan2.1-T2V-1.3B with global causal attention
        # (no local window, no attention sink), so these are hard-coded.
        model_dir = snapshot_download(WAN_REPO_ID)
        self.model = CausalWanModel.from_pretrained(model_dir, local_attn_size=-1, sink_size=0)
        self.model.eval()

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift,
            sigma_min=0.0,
            sigma_max=1.0,
            num_train_timesteps=1000,
            extra_one_step=True,
        )
        self.seq_len = 32760  # [1, 21, 16, 60, 104]

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # X0 prediction. kv_cache present -> inference path; otherwise -> training forward.
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> FlowMatchScheduler:
        return self.scheduler
