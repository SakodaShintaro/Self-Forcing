# Self-Forcing (bench2drive LoRA fork)

This repository is a stripped-down fork of <https://github.com/guandeh17/Self-Forcing>, focused on LoRA fine-tuning the Wan 2.1 T2V-1.3B causal generator on [bench2drive](https://github.com/Thinklab-SJTU/Bench2Drive) episode latents and evaluating the adapted checkpoint on the held-out split.

## What changed from upstream

- **Single trainer / model / pipeline path**. Removed GAN, ODE, score-distillation trainers; removed bidirectional / non-causal pipelines; removed CausVid / DMD / SiD / GAN / ODE model variants. Only the causal diffusion + LoRA path remains.
- **bench2drive-specific dataset**: `utils/b2d_dataset.Bench2DriveLatentDataset` consumes precomputed per-episode latent tensors and serves random fixed-length windows.
- **LoRA is mandatory.** Training fails fast unless `lora.enabled: true` is set in the config. Saved checkpoints contain only trainable (LoRA + EMA) parameters, so each `model.pt` is a few MB instead of ~11 GB.
- **In-loop validation.** Every `valid_iters` train steps we run a fixed number of forward-only `generator_loss` evaluations on the valid split (with frozen RNG for comparability) and log `val/loss` to wandb / stdout.
- **`train.py` is self-contained**. The trainer class moved out of `trainer/` into `train.py` and the trainer dispatch was removed.
- **Single config**. `configs/default_config.yaml` was inlined into `configs/b2d_finetune.yaml`.
- **Wan minimal subset**. Dropped `wan/{configs,distributed,utils,image2video.py,text2video.py}` and `wan/modules/{clip,xlm_roberta}.py`; the kept subset is `wan/modules/{attention, causal_model, model, t5, tokenizers, vae}.py`.
- Dropped `inference.py` / `demo.py` / `demo_utils/` (Gradio demo + low-memory swapping helpers used only by the demo).
- `b2d_root` is set per-machine via the `--b2d_root` CLI flag rather than committed to the yaml.
- `--root_dir` creates a fresh timestamped subdir for every run (logs, checkpoints, wandb files).

## Setup

```bash
cd ~/work/Self-Forcing
uv sync
```

Wan 2.1 weights (VAE / T5 / DiT) are pulled from HuggingFace Hub on first run. The `generator_ckpt: hf:gdhe17/Self-Forcing:checkpoints/self_forcing_dmd.pt` entry in the config fetches the pre-distilled DMD checkpoint that we adapt with LoRA.

## End-to-end workflow

The expected layout under `<b2d_root>` (default in this repo: `/home/sakoda/data/bench2drive`):

```text
<b2d_root>/
├── splits.json                              # produced by scripts/b2d_split.py
├── latents/
│   ├── train/<episode>.pt                   # produced by scripts/b2d_encode_latents.py
│   └── valid/<episode>.pt
└── <episode>/                               # raw bench2drive episodes
    └── camera/rgb_front/*.jpg
```

### 1. Build the train/valid split

```bash
uv run python scripts/b2d_split.py --src <b2d_root>
```

Episodes are partitioned deterministically by Route ID hash. Output: `<b2d_root>/splits.json`.

### 2. Pre-encode VAE latents (one-time)

```bash
uv run python scripts/b2d_encode_latents.py --src <b2d_root>
```

Runs the Wan VAE encoder over every episode's `rgb_front/*.jpg` stream and dumps a `(T_lat, 16, 60, 104)` bf16 tensor per episode under `<b2d_root>/latents/{train,valid}/`. Existing files are skipped, so the script is resumable.

### 3. LoRA fine-tune

```bash
uv run python train.py \
    --config_path configs/b2d_finetune.yaml \
    --b2d_root <b2d_root> \
    --root_dir /path/to/results
```

Each run lands at `<root_dir>/<YYYYmmdd_HHMMSS>_b2d_finetune/`. wandb files, logs and `checkpoint_model_<step>/model.pt` are all written there. `local/train.sh` is a thin wrapper with the local paths baked in.

Key knobs in [`configs/b2d_finetune.yaml`](configs/b2d_finetune.yaml):

| key | meaning |
| --- | --- |
| `lora.{rank, alpha, target_modules}` | LoRA shape; `target_modules` defaults to all attention `q/k/v/o` |
| `image_or_video_shape` | `[1, 21, 16, 60, 104]` — 21-latent clip per training step |
| `num_frame_per_block` | `3` — must match the base DMD checkpoint |
| `max_steps` | hard step cap for the training loop |
| `log_iters` | save a checkpoint every N steps |
| `valid_iters` / `valid_batches` | run validation every N steps over M batches |
| `ema_weight` / `ema_start_step` | EMA over LoRA params, lazily started at the threshold step |

### 4. Evaluate a checkpoint

```bash
uv run python scripts/b2d_infer_valid.py \
    --config_path configs/b2d_finetune.yaml \
    --b2d_root <b2d_root> \
    --checkpoint_path <run_dir>/checkpoint_model_<step>/model.pt
```

For each valid episode we slide a `K=num_context_blocks=1` ground-truth context window over the saved latent stream, ask the LoRA-adapted model to denoise the next block (KV cache reset per call), decode pred + GT through the Wan VAE and emit a side-by-side MP4 plus per-frame / per-block PSNR. Output lands at `<ckpt_dir>/<YYYYmmdd_HHMMSS>_eval/`.

Without `--checkpoint_path` the LoRA layers stay zero-initialised, giving you the pretrained baseline as a sanity number. `local/valid.sh` is a thin wrapper that takes a checkpoint path as `$1`.
