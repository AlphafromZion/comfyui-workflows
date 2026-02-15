# HunyuanVideo 1.5 Image-to-Video Workflow

Image-to-Video generation using HunyuanVideo 1.5. Takes a reference image and generates a video clip with motion described in the prompt.

**Native ComfyUI support** — no custom nodes required (ComfyUI 0.3.68+).

## Node Graph (Simplified)

```
UNETLoader (DiT) ──────────────────────────────────────────────────────────┐
DualCLIPLoader (Qwen2.5-VL + BYT5) ─┬─ CLIPTextEncode (positive) ──────── │
                                     └─ CLIPTextEncode (negative, empty) ─ │
CLIPVisionLoader ─────────────┐                                            │
LoadImage ─┬─ CLIPVisionEncode ┘                                           │
           │                                                               │
VAELoader ─┼──────────────────────── HunyuanVideo15ImageToVideo ───────────┘
           │                              │ positive, negative, latent
           │                              ▼
           │              ModelSamplingSD3 (shift=7) → CFGGuider (cfg=6)
           │              BasicScheduler (20 steps) → RandomNoise
           │                              │
           │                    SamplerCustomAdvanced
           │                              │
           └───────────────────── VAEDecode → CreateVideo → SaveVideo
```

## Required Models

| Model | File | Size | Directory |
|-------|------|------|-----------|
| DiT I2V | `hunyuanvideo1.5_720p_i2v_fp16.safetensors` | 15.5 GB | `diffusion_models/` |
| Qwen2.5-VL | `qwen_2.5_vl_7b.safetensors` | 16 GB | `text_encoders/` |
| BYT5 Small | `byt5_small_glyphxl_fp16.safetensors` | 419 MB | `text_encoders/` |
| SigCLIP Vision | `sigclip_vision_patch14_384.safetensors` | 817 MB | `clip_vision/` |
| VAE | `hunyuanvideo15_vae_fp16.safetensors` | 2.4 GB | `vae/` |

**Total: ~35 GB** of models.

Download all from [Comfy-Org HunyuanVideo 1.5 Repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files).

### VRAM-Constrained Option

For GPUs with less VRAM, use the fp8 distilled DiT instead:
- `hunyuanvideo1.5_720p_i2v_cfg_distilled_fp8_scaled.safetensors` (8.3 GB)
- Set CFG to 1.0 (distilled model doesn't need classifier-free guidance)

## Proven Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Resolution | 544 × 960 | Safe for 32GB VRAM. 720 × 1280 needs more. |
| Frames | 121-193 | 193 max on 32GB at 544×960. 201 crashes. |
| CFG | 6.0 | Non-distilled model. Use 1.0 for distilled. |
| Shift | 7 | 720p I2V setting from Hunyuan team. |
| Steps | 20 | 50 is original but 20 is good enough and 2.5x faster. |
| Sampler | euler | Consistent. |
| Scheduler | simple | Works well with euler. |
| FPS | 24 | Standard video framerate. |

## How to Use

1. Open `workflow.json` in ComfyUI
2. Load your reference image in the `LoadImage` node
3. Write your motion prompt in the positive CLIPTextEncode node
4. Leave negative prompt empty (works fine for HV1.5)
5. Adjust frame count if needed (121 = ~5s at 24fps)
6. Queue prompt

## Prompt Tips

- Describe the **motion**, not just the scene
- Example: `"woman walks forward along the beach, waves crashing, wind blowing hair, golden sunset lighting"`
- Keep prompts focused on what should MOVE
- The reference image provides the visual — the prompt provides the animation

## VRAM Guide

| GPU VRAM | Max Resolution | Max Frames | Weight Dtype |
|----------|---------------|------------|--------------|
| 24 GB | 544 × 960 | ~97 | fp8_e4m3fn |
| 32 GB | 544 × 960 | 193 | fp16 or fp8 |
| 32 GB | 720 × 1280 | ~97 | fp8_e4m3fn |
| 48+ GB | 720 × 1280 | 193+ | fp16 |

**AMD ROCm note:** fp8 `e4m3fn` works on ROCm 7.2. See the [AMD ROCm Setup Guide](../../guides/amd-rocm-setup.md).

## Optional: 1080p Super Resolution

The workflow includes a disabled (Ctrl+B to enable) super-resolution pipeline that upscales 720p output to 1080p using a latent upscale model. Requires additional VRAM and the `hunyuanvideo15_latent_upsampler_1080p.safetensors` model.

## Adding LoRAs

To use a HunyuanVideo 1.5 LoRA, add a `LoraLoaderModelOnly` node between the UNETLoader and the rest of the pipeline. HV1.5 LoRAs must be trained with the `networks.lora_hv_1_5` module (NOT `lora_hv` from HV 1.0).

See the [HunyuanVideo Training Guide](../../guides/hunyuan-video-training.md) for training your own.
