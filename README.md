# Self-Forcing

This repository is a fork of <https://github.com/guandeh17/Self-Forcing>.

## 概要

CARLA streaming 推論用に以下の改修を加えています:

- 依存解決を `pyproject.toml` (uv 管理) に移行
- モデル取得を HuggingFace Hub 経由に変更（`wan_models/` のローカル展開不要）
- `inference.py` の `--checkpoint_path` で `hf:<repo>:<filename>` 形式をサポート
- `TextImagePairDataset` を **multi-frame** (`file_names` リスト) 対応に拡張
- `inference.py` の I2V 分岐を `(B, C, T, H, W)` 直渡しに変更（複数フレームを initial latent としてエンコード可能）
- CARLA 画像を I2V 用に letterbox 整形する `scripts/prepare_carla_i2v.py` を追加

## セットアップ

```bash
cd ~/work/Self-Forcing
uv sync
```

`uv sync` で `.venv` 配下に PyTorch 2.7+cu128, flash-attn 2.8.1 等が入ります。
モデルファイル（VAE / T5 / DiT）は初回実行時に HF Hub から自動ダウンロードされます。

## I2V (CARLA データ)

### Step 1: 入力フレームの前処理

```bash
uv run python scripts/prepare_carla_i2v.py \
    --src /path/to/carla/obs/ep_xxxxxxxx/ \
    --dst data/carla_sample \
    --caption "First-person dashcam view ..." \
    --num_frames 9 \
    --start_index 20
```

- 入力 PNG を **letterbox** で 832×480 に整形（情報を失わない、上下/左右に黒帯）
- `data/carla_sample/target_crop_info_carla.json` と `data/carla_sample/carla/*.png` を生成
- `--start_index` で先頭から飛ばす枚数を指定

`--num_frames` の指針（VAE のテンポラル粒度）:

| pixel frame | latent | 用途 |
| --- | --- | --- |
| 1 | 1 | 単フレーム I2V |
| 9 | 3 | 1 block ぶんの context（最小実用単位） |
| 13 | 4 | I2V + 1 latent context |
| 17 | 5 | I2V + 2 latent context |

(VAE の 1 latent = 1 + 4 × N pixel frame で、`N=2` のとき 9 pixel)

### Step 2: 推論

```bash
uv run python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/carla_sample \
    --checkpoint_path hf:gdhe17/Self-Forcing:checkpoints/self_forcing_dmd.pt \
    --data_path data/carla_sample \
    --num_output_frames 21 \
    --i2v \
    --use_ema
```

制約:

- `--num_output_frames` は **総 latent 数**（入力 latent + 生成ノイズ latent）
- `(num_output_frames - num_input_latents)` が `num_frame_per_block` (= 3) の倍数である必要あり

入力 3 latent (= 9 px) のときの取りうる値:

| `--num_output_frames` | input | 生成 latent | 備考 |
| --- | --- | --- | --- |
| **6** | 3 | 3 | 最小（1 block 生成） |
| 9 | 3 | 6 | |
| 12 | 3 | 9 | |
| 15 | 3 | 12 | |
| 18 | 3 | 15 | |
| **21** | 3 | 18 | `self_forcing_dmd.pt` の上限 (KV cache = 21 latent) |
| > 21 | — | — | `self_forcing_10s.pt` (10 秒 ≈ 161 latent 対応) が必要 |

- 出力は **16 fps** (Wan 2.1 の学習レート)。21 latent ≈ 5.25 秒
- `inference.py` の `write_video(..., fps=16)` は mp4 メタデータ上の再生速度。値を変えると再生速度が変わるだけで、生成された動きの速度は 16 fps 固定

## 解像度について

**832×480** (Wan 2.1 T2V-1.3B が 480p で学習されているため)

## オリジナル README

オリジナルの開発・学習手順は [https://github.com/guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) を参照してください。
