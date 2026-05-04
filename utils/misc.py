import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def resolve_checkpoint_path(spec: str) -> str:
    """Resolve --checkpoint_path / generator_ckpt. Supports a local path or `hf:repo_id:filename`."""
    if spec.startswith("hf:"):
        _, repo_id, filename = spec.split(":", 2)
        return hf_hub_download(repo_id, filename)
    return spec


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
