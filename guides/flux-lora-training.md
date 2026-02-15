# FLUX LoRA Training Guide

How to train FLUX.1-dev LoRAs for consistent character generation. Covers dataset prep, training config, and deployment.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Training Environment](#training-environment)
- [Training Config](#training-config)
- [Running Training](#running-training)
- [Deployment](#deployment)
- [Tips and Gotchas](#tips-and-gotchas)

---

## Overview

| Component | Choice |
|-----------|--------|
| Base Model | FLUX.1-dev |
| Training Tool | Kohya / Flux Gym |
| GPU | Blackwell 96GB (Vast.ai) recommended |
| Training Time | ~2-3 hours on Blackwell |
| Cost | ~$2-3 on Vast.ai |

You CAN train on consumer GPUs (RTX 3080, R9700), but Vast.ai Blackwell instances at $0.70-0.80/hr are faster and cheaper in total time.

---

## Dataset Preparation

### Image Requirements

| Requirement | Value |
|-------------|-------|
| Minimum images | 15-30 |
| Optimal images | 50-120 |
| Resolution | At least 1024px on shortest side |
| Format | JPG or PNG |
| Face crops | 10-15% of dataset |

### Composition Mix

For a character LoRA, aim for:
- **50%** full body shots
- **30%** upper body / waist up
- **20%** face closeups and side profiles

Side profiles are especially valuable — most datasets lack them, and it hurts pose diversity.

### Naming Convention

```
trigger_001.jpg
trigger_001.txt
trigger_002.jpg
trigger_002.txt
...
```

Sequential numbering. Each image gets a matching `.txt` caption file.

### Captioning

Use a VLM for captioning. Qwen2.5-VL-7B works well:

```
Describe this image in detail for AI training. Focus on the person's appearance,
clothing, pose, expression, and environment. Be specific about colors, textures,
and lighting.
```

**Verify flagged images.** VLMs hallucinate. Check any caption that mentions things not in the image.

**Don't use `shuffle_caption` with `--cache_text_encoder_outputs`.** They're incompatible in Kohya.

### Folder Structure (Kohya Format)

```
dataset/
└── 4_trigger_word/    # 4 repeats, "trigger_word" is the trigger
    ├── trigger_001.jpg
    ├── trigger_001.txt
    ├── trigger_002.jpg
    ├── trigger_002.txt
    └── ...
```

The `4_` prefix means each image is seen 4 times per epoch. Adjust based on dataset size:
- 15-30 images: `4-8` repeats
- 50-120 images: `2-4` repeats

---

## Training Environment

### Vast.ai (Recommended)

**Instance requirements:**
- GPU: Blackwell (B200/B100) 96GB+
- Internet: 1Gbps+ (slow downloads waste hours)
- Storage: 2x what you estimate (models + checkpoints add up)
- Cost: ~$0.70-0.80/hr

**Why Blackwell specifically:** 96GB VRAM fits the full FLUX model + training state without quantization compromises. Cheaper cards take 2-4x longer, costing more total.

**Template:** Flux Gym (has training scripts pre-installed, needs model download)

**Before downloading models:** Accept the FLUX.1-dev license on HuggingFace. The gated model download will fail silently if you haven't.

### Local (R9700 32GB / RTX 4090 24GB)

Possible with lower dim and quantization. See Kohya documentation for low-VRAM configs. Training will be slower.

---

## Training Config

### Proven Config (Blackwell 96GB)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `network_dim` | 64 or 128 | 64 = standard quality, 128 = max quality |
| `network_alpha` | Half of dim | 32 (dim 64) or 64 (dim 128) |
| `optimizer` | adafactor | More stable than adamw8bit for FLUX |
| `learning_rate` | 8e-5 | Sweet spot for adafactor |
| `num_repeats` | 4 | Adjust per dataset size |
| `max_train_epochs` | 16 | Usually converges by epoch 12-14 |
| `resolution` | 1024 | With bucketing (512-1536) |

### Full Command

```bash
accelerate launch train_network.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --dataset_config dataset.toml \
  --output_dir output/ \
  --output_name my-flux-lora \
  --network_module networks.lora_flux \
  --network_dim 128 \
  --network_alpha 64 \
  --optimizer_type adafactor \
  --learning_rate 8e-5 \
  --max_train_epochs 16 \
  --mixed_precision bf16 \
  --save_every_n_epochs 4 \
  --resolution 1024 \
  --enable_bucket \
  --min_bucket_reso 512 \
  --max_bucket_reso 1536 \
  --cache_text_encoder_outputs \
  --gradient_checkpointing
```

---

## Running Training

### 1. Prepare Dataset

```bash
# Create folder structure
mkdir -p dataset/4_character_name/

# Copy images + captions
cp *.jpg *.txt dataset/4_character_name/
```

### 2. Create dataset.toml

```toml
[general]
resolution = 1024
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536

[[datasets]]
batch_size = 1

[[datasets.subsets]]
image_dir = "dataset/4_character_name"
caption_extension = ".txt"
num_repeats = 4
```

### 3. Run Training

```bash
accelerate launch train_network.py \
  --dataset_config dataset.toml \
  --output_dir output/ \
  --output_name my-flux-lora \
  # ... (rest of config above)
```

### 4. Monitor

Watch the loss curve. For FLUX LoRAs:
- Loss should decrease steadily for first 8-10 epochs
- Final loss typically 0.01-0.05 range
- If loss spikes or goes NaN, reduce learning rate

---

## Deployment

### In ComfyUI

1. Copy the `.safetensors` LoRA file to `ComfyUI/models/loras/`
2. Use the [FLUX LoRA Character workflow](../workflows/flux-lora-character/)
3. Set LoRA strength to 0.9-1.0

### Inference Settings

| Setting | Value |
|---------|-------|
| Checkpoint | `flux1-krea-dev_fp8_scaled.safetensors` |
| LoRA Strength | 0.9-1.0 |
| Steps | 20-30 |
| CFG | 1.0 |
| Sampler | euler / simple |
| Resolution | 1088×1920 (portrait) |

---

## Tips and Gotchas

### Dataset Quality > Quantity

30 excellent images beat 200 mediocre ones. Every image should be:
- Well-lit, clear, high resolution
- Showing the character consistently
- Varied in pose, angle, and background

### Face Crops Lock Identity

Include 10-15% face closeups. Without them, the LoRA may struggle with face consistency at different distances.

### Training Data Bias is Real

If your dataset is 80% NSFW, the LoRA will default to NSFW. If it's all outdoor shots, indoor prompts will be weak. Balance matters.

### FLUX.2 Compatibility Warning

FLUX.2 was released November 2025 with a different architecture. **FLUX.1 LoRAs are NOT compatible with FLUX.2.** The ecosystem is still maturing. Stick with FLUX.1-dev for now unless you have a specific reason to migrate.

### Non-US/European Locations Are Weak

FLUX's training data is US/Europe biased. Specific landmarks or skylines outside that bubble give inconsistent results. Use descriptive terms instead: `"modern waterfront cityscape"` beats naming a specific city.

---

*Tested and proven. These configs produced production-quality character LoRAs on first training runs.*
