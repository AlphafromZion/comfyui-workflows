# HunyuanVideo 1.5 I2V LoRA Training Guide

Training Image-to-Video LoRAs for HunyuanVideo 1.5 using musubi-tuner. This guide covers the full pipeline from dataset to deployment, with ROCm-specific fixes.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Model Downloads](#model-downloads)
- [Cache Pipeline](#cache-pipeline)
- [Training](#training)
- [Deployment](#deployment)
- [ROCm-Specific Issues](#rocm-specific-issues)
- [Troubleshooting](#troubleshooting)

---

## Overview

| Component | Details |
|-----------|---------|
| Training Tool | musubi-tuner v0.2.15 |
| Training Script | `hv_1_5_train_network.py` (NOT `hv_train_network.py`) |
| Network Module | `networks.lora_hv_1_5` (NOT `lora_hv`) |
| GPU | R9700 32GB (ROCm) or any 24GB+ GPU (CUDA) |
| Training Time | ~9 hours on R9700 32GB, ~2-3 hours on Blackwell 96GB |
| Dataset | 50-120 images with natural language captions |

**Critical:** HunyuanVideo 1.5 uses different scripts and modules than HunyuanVideo 1.0. Don't mix them.

---

## Requirements

### Install musubi-tuner

```bash
pip install git+https://github.com/kohya-ss/musubi-tuner.git@v0.2.15
```

Or clone and install:

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout v0.2.15
pip install -e .
```

### Dependencies

```bash
pip install accelerate transformers safetensors
```

For ROCm, install PyTorch ROCm 7.2 first. See the [AMD ROCm Setup Guide](amd-rocm-setup.md).

---

## Dataset Preparation

### Image Requirements

| Requirement | Value |
|-------------|-------|
| Images | 50-120 recommended |
| Resolution | Consistent (e.g., 960×544 or 544×960) |
| Format | JPG or PNG |
| Face crops | 10-15% of dataset |
| Captions | Natural language `.txt` files |

### Captioning with Qwen2.5-VL

Use Qwen2.5-VL-7B for natural language captioning (not booru tags):

```python
# Prompt for captioning
"""Describe this image in one detailed paragraph. Focus on the person's
appearance, clothing, pose, expression, and environment. Be specific
about colors, textures, and lighting. Use natural language."""
```

### Naming

```
image_001.jpg
image_001.txt
image_002.jpg
image_002.txt
...
```

### Dataset Config (TOML)

```toml
[general]
resolution = [960, 544]  # Width, Height
enable_bucket = true

[[datasets]]
batch_size = 1

[[datasets.subsets]]
image_dir = "training/my-dataset"
caption_extension = ".txt"
num_repeats = 8
```

---

## Model Downloads

HunyuanVideo 1.5 needs **~53GB** of models:

| Model | Size | Source |
|-------|------|--------|
| DiT I2V (bf16) | 33.3 GB | [Tencent HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) |
| VAE | 4.7 GB | Same repo |
| Qwen2.5-VL-7B (text encoder) | 16 GB | [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| BYT5 Small | 419 MB | Same as DiT repo |
| SigLIP CLIP Vision | 817 MB | Same as DiT repo |

**Note:** For training, use the Tencent-format models (not Comfy-Org repackaged). musubi-tuner expects the original format.

### Directory Structure

```
models/hunyuan1.5/
├── dit/
│   └── mp_rank_00_model_states.pt       # 33.3 GB
├── vae/
│   └── pytorch_model.pt                  # 4.7 GB
├── text_encoder/
│   └── model.safetensors                 # Qwen2.5-VL 16 GB
├── text_encoder_2/
│   └── model.safetensors                 # BYT5 419 MB
└── clip_vision/
    └── pytorch_model.bin                 # SigLIP 817 MB
```

---

## Cache Pipeline

Training has a 3-step pipeline: cache latents → cache text encoders → train.

### Step 1: Cache Latents

Encodes images through VAE and SigLIP. Creates `.npz` files.

```bash
python hv_1_5_cache_latents.py \
  --vae "models/hunyuan1.5/vae/pytorch_model.pt" \
  --clip_vision "models/hunyuan1.5/clip_vision/pytorch_model.bin" \
  --dataset_config "training/dataset.toml" \
  --batch_size 1 \
  --i2v
```

**`--i2v` flag is required** for Image-to-Video LoRAs. It tells the caching to include SigLIP vision encoding.

### Step 2: Cache Text Encoders

Encodes captions through Qwen2.5-VL and BYT5.

```bash
python hv_1_5_cache_text_encoder_outputs.py \
  --text_encoder1 "models/hunyuan1.5/text_encoder/model.safetensors" \
  --text_encoder2 "models/hunyuan1.5/text_encoder_2/model.safetensors" \
  --dataset_config "training/dataset.toml" \
  --batch_size 1 \
  --device cpu
```

**`--device cpu` is critical.** The Qwen2.5-VL text encoder crashes on GPU even in bf16. CPU caching is slower (~17s per item) but stable. On a 92-image dataset, this takes about 26 minutes.

### Cache Files

Cache files are saved alongside your images:

```
training/my-dataset/
├── image_001.jpg
├── image_001.txt
├── image_001_cache_latent.npz
├── image_001_cache_text.npz
├── ...
```

Cache files are **reusable**. You only need to rebuild them if you change images or captions.

---

## Training

### Proven Config (R9700 32GB)

```bash
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1

accelerate launch hv_1_5_train_network.py \
  --dit "models/hunyuan1.5/dit/mp_rank_00_model_states.pt" \
  --dataset_config "training/dataset.toml" \
  --output_dir "training/output/" \
  --output_name "my-hv15-i2v-lora" \
  --network_module networks.lora_hv_1_5 \
  --network_dim 128 \
  --network_alpha 64 \
  --optimizer_type adafactor \
  --learning_rate 2e-5 \
  --max_train_epochs 32 \
  --mixed_precision bf16 \
  --blocks_to_swap 20 \
  --max_grad_norm 1.0 \
  --lr_warmup_steps 50 \
  --save_every_n_epochs 8 \
  --gradient_checkpointing
```

### Config Breakdown

| Parameter | Value | Why |
|-----------|-------|-----|
| `network_dim` | 128 | Max quality. Use 64 for faster training. |
| `network_alpha` | 64 | Half of dim. Standard practice. |
| `optimizer` | adafactor | More stable than Adam variants for video LoRAs |
| `learning_rate` | 2e-5 | Sweet spot for adafactor on HV1.5 |
| `max_train_epochs` | 32 | Converges around epoch 28-30 |
| `mixed_precision` | bf16 | Required on ROCm. fp8 training untested. |
| `blocks_to_swap` | 20 | Fits 33GB DiT on 32GB GPU |
| `max_grad_norm` | 1.0 | **CRITICAL** — prevents NaN at step 2 |
| `lr_warmup_steps` | 50 | Stabilizes early training with gradient clipping |
| `save_every_n_epochs` | 8 | Checkpoints at epochs 8, 16, 24, 32 |

### Training Metrics

On R9700 32GB (ROCm 7.2), 92-image dataset:
- **Steps:** 6400 (32 epochs × 200 steps/epoch)
- **Speed:** 5.22 seconds/step
- **Total time:** 9 hours 17 minutes
- **Loss:** Steady decrease, final sub-0.001 (best 0.000796 at epoch 31)
- **Output:** 1.4 GB LoRA file

### What to Watch

- **Loss should decrease steadily.** Spikes are normal, trending down is what matters.
- **NaN loss at step 2** = missing `--max_grad_norm 1.0`
- **hipErrorIllegalAddress** = wrong PyTorch version (use ROCm 7.2)
- **OOM** = increase `blocks_to_swap` or reduce `network_dim`

---

## Deployment

### In ComfyUI

1. Copy the LoRA `.safetensors` to `ComfyUI/models/loras/`
2. Use the [HunyuanVideo 1.5 I2V workflow](../workflows/hunyuan-video-15-i2v/)
3. Add a `LoraLoaderModelOnly` node between UNETLoader and the pipeline

### Inference Settings

| Setting | Value |
|---------|-------|
| DiT | fp16 or fp8 (fp8 won't load LoRA) |
| Resolution | 544×960 |
| Frames | 121-193 (32GB max) |
| CFG | 6.0 |
| Shift | 7 |
| Steps | 20-30 |

**Note:** LoRAs require the fp16 DiT, not fp8. The fp8 quantized DiT doesn't support LoRA injection.

---

## ROCm-Specific Issues

### musubi-tuner CLIP Vision Bug

`clip_vision.py` in musubi-tuner expects images in H,W,C format but receives C,H,W from the dataloader. You need to patch this:

```python
# In clip_vision.py, find the image preprocessing
# Add this before the transform:
if image.shape[0] == 3:  # C,H,W format
    image = image.permute(1, 2, 0)  # Convert to H,W,C
```

### PyTorch Version

**ROCm 7.2 official wheels ONLY.** See the [AMD ROCm Setup Guide](amd-rocm-setup.md) for the full story on why nightlies crash.

### Environment Variables

```bash
export HSA_ENABLE_SDMA=0         # Prevents memory corruption
export GPU_MAX_HW_QUEUES=1       # Stability over throughput
```

Set these before training. Without them, you may get random crashes under memory pressure.

---

## Troubleshooting

### Loss goes NaN at step 2

Add `--max_grad_norm 1.0` and `--lr_warmup_steps 50`. bf16 gradients explode without clipping.

### Text encoder caching crashes on GPU

Use `--device cpu`. The Qwen2.5-VL encoder is too large for GPU caching.

### `hipErrorIllegalAddress` during training

Wrong PyTorch version. Install ROCm 7.2 official wheels from `repo.radeon.com`.

### OOM during training

Increase `--blocks_to_swap` (try 24 or 28). Or reduce `--network_dim` to 64.

### LoRA doesn't affect output in ComfyUI

Make sure you're using:
- The fp16 DiT (not fp8 — fp8 doesn't support LoRA)
- `LoraLoaderModelOnly` connected between UNETLoader and the sampling pipeline
- LoRA strength at 0.8-1.0

### Training is very slow

On R9700 32GB with `blocks_to_swap 20`, expect ~5.2s/step. That's normal — the CPU↔GPU block swapping adds overhead. On Blackwell 96GB without block swapping, expect ~1.0s/step.

---

*This guide is based on a successful training run: 9h 17min on R9700 32GB, 6400 steps, final loss 0.000796. Every parameter listed here is proven.*
