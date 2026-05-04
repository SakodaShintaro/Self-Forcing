import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download

_WRAP_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")


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


def strip_wrap_prefixes(k: str) -> str:
    """Strip the FSDP / gradient-checkpoint / torch.compile wrapper prefixes that
    sneak into state-dict keys when modules are wrapped at training time."""
    for p in _WRAP_PREFIXES:
        k = k.replace(p, "")
    return k


def load_generator_state_dict(
    path: str,
    explicit_key: str | None = None,
    prefer_keys: tuple[str, ...] = ("generator", "generator_ema", "model"),
) -> dict:
    """Load a generator checkpoint and return a flat state_dict with wrapper
    prefixes stripped.

    `explicit_key` (e.g. config.generator_ckpt_key) wins if it exists in the
    file; otherwise the first hit from `prefer_keys` is used. Caller controls
    the priority order: training prefers raw `generator`, inference prefers
    `generator_ema`.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if explicit_key and explicit_key in sd:
        sd = sd[explicit_key]
    else:
        for k in prefer_keys:
            if k in sd:
                sd = sd[k]
                break
    return {strip_wrap_prefixes(k): v for k, v in sd.items()}
