# AMD ROCm Setup for ComfyUI — The Guide Nobody Else Publishes

> **TL;DR:** Use official AMD ROCm 7.2 wheels. Not PyTorch nightlies. Not ROCm 7.0. ROCm 7.2 official wheels from `repo.radeon.com`. I crashed 7 times on 7.0 before getting first-attempt success on 7.2.

This guide covers running ComfyUI on AMD GPUs with ROCm, specifically tested on the **Radeon AI PRO R9700 (32GB)**. Everything here applies to other RDNA3+ GPUs (7900 XTX, 7900 XT, W7900, etc.) with ROCm support.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [The PyTorch Version Problem](#the-pytorch-version-problem)
- [Installing ROCm + PyTorch (The Right Way)](#installing-rocm--pytorch-the-right-way)
- [fp8 Status on ROCm](#fp8-status-on-rocm)
- [ComfyUI Setup](#comfyui-setup)
- [blocks_to_swap for Large Models](#blocks_to_swap-for-large-models)
- [Training on ROCm](#training-on-rocm)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Ubuntu 22.04 or 24.04 (tested on 24.04)
- AMD GPU with RDNA3 or newer (gfx1100, gfx1101, gfx1102, gfx1200, gfx1201)
- ROCm drivers installed via `amdgpu-install`
- Python 3.10+

### Install ROCm Drivers

```bash
# Add AMD repo
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
sudo dpkg -i amdgpu-install_6.4.60400-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=graphics,rocm

# Add user to groups
sudo usermod -aG render,video $USER

# Reboot
sudo reboot
```

Verify with:
```bash
rocminfo  # Should list your GPU
rocm-smi  # Should show GPU stats
```

**Important:** ROCm drivers ≠ ROCm PyTorch. Having `rocm-smi` work does NOT mean PyTorch can see your GPU. I learned this the hard way — ComfyUI started fine, models loaded to system RAM, and I spent hours confused before realizing PyTorch ROCm was never installed.

---

## The PyTorch Version Problem

This is where most people get stuck, and where I burned 7 attempts before figuring it out.

### What Does NOT Work

**PyTorch nightly ROCm 7.0 builds:**
```bash
# DON'T DO THIS
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

This installs fine. Basic tensor ops work. `torch.cuda.is_available()` returns True. You'll think everything is good.

Then you try to run a model with `blocks_to_swap` (required for fitting large models in VRAM), and at some random step between 33 and 77, you get:

```
hipErrorIllegalAddress: an illegal memory access was encountered
```

The process crashes. No recovery. No useful error message. Just dead.

I hit this 7 times across different configs, different block swap values, different models. The pattern was always the same: works for a while, then `hipErrorIllegalAddress` out of nowhere.

### What WORKS

**Official AMD ROCm 7.2 wheels from repo.radeon.com:**

```bash
pip install torch==2.9.1+rocm7.2.0 torchvision torchaudio \
  --index-url https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/
```

Or if the index-url doesn't work, download the wheels directly:

```bash
# From repo.radeon.com
pip install https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/torch-2.9.1%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
pip install https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/torchvision-0.24.0%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
pip install https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/torchaudio-2.9.1%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
pip install https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/triton-3.5.1%2Brocm7.2.0-cp312-cp312-linux_x86_64.whl
```

First attempt with ROCm 7.2: **700+ steps stable**, zero crashes. Same config that crashed 7 times on 7.0.

### Why This Matters

The nightlies and the official AMD wheels look identical from the outside. Same Python API, same version numbers (roughly), same `torch.cuda.is_available()` output. The difference is in the HIP runtime behavior under memory pressure — exactly the scenario `blocks_to_swap` creates.

**Rule: Always use official AMD wheels from repo.radeon.com. Never PyTorch nightlies for ROCm production work.**

---

## Installing ROCm + PyTorch (The Right Way)

```bash
# Create a venv
python3 -m venv ~/comfyui-venv
source ~/comfyui-venv/bin/activate

# Install PyTorch with ROCm 7.2
pip install torch==2.9.1+rocm7.2.0 torchvision torchaudio \
  --index-url https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/

# Install triton (needed for some optimizations)
pip install triton==3.5.1+rocm7.2.0 \
  --index-url https://repo.radeon.com/rocm/manylinux2_28/rocm-7.2.0/

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
# Test GPU compute
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
print(f'GPU matmul: PASSED')
"
```

Expected output:
```
PyTorch: 2.9.1+rocm7.2.0
CUDA available: True
Device: AMD Radeon AI PRO R9700
VRAM: 31.9 GB
GPU matmul: PASSED
```

If `torch.cuda.is_available()` returns `False`, you need to fix your ROCm drivers first.

---

## fp8 Status on ROCm

This changes with every PyTorch/ROCm version. Here's the current state as of February 2026:

### ROCm 7.0 (PyTorch Nightlies)

**ALL fp8 modes BROKEN.** `hipErrorIllegalAddress`, promotion errors, you name it.

### ROCm 7.2 (Official AMD Wheels)

| Format | Status | Notes |
|--------|--------|-------|
| `e4m3fn` (NVIDIA format) | ✅ WORKS | Used by FLUX fp8 checkpoints, HV1.5 fp8 DiT |
| `e4m3fnuz` (AMD native) | ❌ BROKEN | hipBLAS returns UNSUPPORTED for matmul |

**`e4m3fn` is the one that matters.** It's what FLUX `fp8_scaled` checkpoints use, and what HunyuanVideo 1.5 fp8 DiT models use. The AMD-specific `e4m3fnuz` format is broken, but nothing in the ComfyUI ecosystem actually uses it.

### What This Means in Practice

- **FLUX fp8 checkpoints:** Work on ROCm 7.2. The Krea fp8_scaled model runs fine.
- **HV1.5 DiT fp8:** Works. Reduces 33GB bf16 DiT to ~16.5GB, fits on 32GB without `blocks_to_swap`.
- **Training:** Still bf16. fp8 training is untested on ROCm 7.2, and bf16 is proven stable.

### Verify fp8 on Your System

```python
import torch

# Test fp8 e4m3fn matmul
a = torch.randn(512, 512, device='cuda').to(torch.float8_e4m3fn)
b = torch.randn(512, 512, device='cuda').to(torch.float8_e4m3fn)
scale_a = torch.ones(1, device='cuda', dtype=torch.float32)
scale_b = torch.ones(1, device='cuda', dtype=torch.float32)
result = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
print(f"fp8 e4m3fn matmul: PASSED (shape: {result.shape})")
```

---

## ComfyUI Setup

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Use your ROCm venv
source ~/comfyui-venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start ComfyUI
python main.py --listen 0.0.0.0
```

### Model Directory Structure

```
ComfyUI/models/
├── diffusion_models/     # FLUX UNET, HV1.5 DiT
├── text_encoders/        # CLIP-L, T5-XXL, Qwen2.5-VL, BYT5
├── clip_vision/          # SigCLIP
├── vae/                  # FLUX ae, HV1.5 VAE
└── loras/                # Your LoRA files
```

### Performance Reference (R9700 32GB)

| Workflow | Resolution | Speed |
|----------|-----------|-------|
| FLUX (fp8 checkpoint) | 1088×1920 | ~3.45 it/s |
| FLUX (fp8 + LoRA) | 1088×1920 | ~3.2 it/s |
| HV1.5 I2V (fp16 DiT) | 544×960 | ~5.2 s/step (training), ~100 s/step (inference*) |

*HV1.5 inference speed is being investigated. Training is significantly faster than inference due to different memory access patterns.

---

## blocks_to_swap for Large Models

When a model doesn't fit in VRAM (e.g., HV1.5 DiT at 33GB bf16 on 32GB GPU), `blocks_to_swap` moves some transformer blocks to CPU RAM during forward passes. It's slower but it works.

### When You Need It

- HV1.5 DiT bf16 (33GB) on 32GB GPU: **blocks_to_swap 20**
- HV1.5 DiT fp8 (16.5GB) on 32GB GPU: **not needed**
- FLUX fp8 (6GB) on 10GB GPU: **not needed**

### The Stability Catch

`blocks_to_swap` is where PyTorch version matters most. The constant CPU↔GPU memory transfers stress the HIP runtime in ways that normal inference doesn't.

**ROCm 7.0:** Crashes with `hipErrorIllegalAddress` at random steps (my 7 failed attempts were all with `blocks_to_swap`).

**ROCm 7.2:** Stable through 6400+ steps (9+ hours of continuous training).

---

## Training on ROCm

### Critical Requirements for bf16 Training

1. **`--max_grad_norm 1.0`** — Without gradient clipping, bf16 training goes NaN at step 2. Not step 200. Step 2.

2. **`--lr_warmup_steps 50`** — Stabilizes early training. Combined with gradient clipping, this prevents loss explosions.

3. **Text encoder caching on CPU** — The Qwen2.5-VL-7B text encoder crashes on GPU even in bf16 for HunyuanVideo 1.5 training. Use `--device cpu` when caching text encoder outputs. It's slower (~17s per item) but stable.

### Environment Variables

```bash
export HSA_ENABLE_SDMA=0         # Prevents memory corruption on some configs
export GPU_MAX_HW_QUEUES=1       # Stability — trades throughput for reliability
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # Better memory management
```

### Proven Training Config (HV1.5 I2V LoRA on R9700 32GB)

```bash
accelerate launch hv_1_5_train_network.py \
  --dit "models/hunyuan1.5/dit/mp_rank_00_model_states.pt" \
  --vae "models/hunyuan1.5/vae/pytorch_model.pt" \
  --text_encoder1 "models/hunyuan1.5/text_encoder/model.safetensors" \
  --text_encoder2 "models/hunyuan1.5/text_encoder_2/model.safetensors" \
  --dataset_config "training/dataset.toml" \
  --output_dir "training/output/" \
  --output_name "my-hv15-lora" \
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

**Result:** 6400 steps, 9h 17min, final loss sub-0.001.

---

## Troubleshooting

### `hipErrorIllegalAddress`

You're using the wrong PyTorch build. Switch to official AMD ROCm 7.2 wheels.

### `torch.cuda.is_available()` returns False

1. Check ROCm drivers: `rocminfo`
2. Check user groups: `groups` (need `render` and `video`)
3. Check PyTorch build: `python3 -c "import torch; print(torch.__version__)"` should show `+rocm`

### Models loading to CPU/RAM instead of GPU

PyTorch ROCm is not installed (ROCm drivers alone aren't enough). Install PyTorch with ROCm wheels.

### Out of VRAM during inference

- Use fp8 checkpoints instead of fp16/bf16
- Reduce resolution
- Reduce frame count (for video)
- Add `--lowvram` flag to ComfyUI

### Loss goes NaN immediately during training

Add `--max_grad_norm 1.0` and `--lr_warmup_steps 50` to your training command.

### Text encoder crashes during caching

Use `--device cpu` for text encoder caching. The Qwen2.5-VL-7B encoder is too large for GPU caching even in bf16.

---

## Version History

| Date | PyTorch | ROCm | Status |
|------|---------|------|--------|
| Feb 2026 | 2.9.1+rocm7.2.0 | 7.2 | ✅ Stable — fp8 works, blocks_to_swap works, training works |
| Feb 2026 | nightly+rocm7.0 | 7.0 | ❌ Broken — hipErrorIllegalAddress with blocks_to_swap |

---

*This guide is maintained by Alpha. If you find corrections or updates, open an issue or PR.*
