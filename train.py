import argparse
import os
import time

import wandb
from omegaconf import OmegaConf

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer


def _set_distributed_defaults() -> None:
    """Populate env vars expected by utils.distributed.launch_distributed_job.

    When launched via torchrun these are already set; when launched as
    `python train.py` they default to a single-rank local job.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="If set, create {root_dir}/{stamp}_{config_name}/ and use it for "
                             "checkpoints, wandb files, and a tee'd train.log.")
    parser.add_argument("--logdir", type=str, default="",
                        help="Explicit log/checkpoint dir (advanced; use --root-dir instead).")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--b2d_root", type=str, default=None,
                        help="Bench2Drive root directory (contains splits.json and latents/{train,valid}/). "
                             "Required when dataset_type=b2d_latent; overrides config.b2d_root if both are set.")

    args = parser.parse_args()

    if args.root_dir is not None:
        if args.logdir:
            parser.error("--root-dir cannot be combined with --logdir.")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        config_tag = os.path.basename(args.config_path).split(".")[0]
        run_dir = os.path.join(args.root_dir, f"{stamp}_{config_tag}")
        os.makedirs(run_dir, exist_ok=True)
        args.logdir = run_dir
        print(f"run dir: {run_dir}", flush=True)

    _set_distributed_defaults()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    if args.b2d_root is not None:
        config.b2d_root = args.b2d_root
    if getattr(config, "dataset_type", None) == "b2d_latent" and not getattr(config, "b2d_root", None):
        parser.error(
            "dataset_type=b2d_latent requires --b2d_root (or config.b2d_root) to be set."
        )

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.logdir
    config.disable_wandb = args.disable_wandb

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
