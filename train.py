import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--b2d_root", type=str, default=None,
                        help="Bench2Drive root directory (contains splits.json and latents/{train,valid}/). "
                             "Required when dataset_type=b2d_latent; overrides config.b2d_root if both are set.")

    args = parser.parse_args()

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
    config.wandb_save_dir = args.wandb_save_dir
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
