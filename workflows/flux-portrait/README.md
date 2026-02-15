# FLUX Portrait Workflow (No LoRA)

Clean FLUX.1-dev text-to-image workflow for portrait generation. No LoRA, no custom nodes — just the core FLUX pipeline.

## Node Graph

```
UNETLoader ─────────────────────────────────┐
DualCLIPLoader → CLIPTextEncode ─┬─ positive ┤
                                 └─ ConditioningZeroOut → negative ┤
VAELoader ──────────────────────────────────────────────┐          │
EmptySD3LatentImage ─────────────────── latent ──────── KSampler → VAEDecode → SaveImage
```

## Required Models

| Model | File | Size |
|-------|------|------|
| UNET | `flux1-krea-dev_fp8_scaled.safetensors` | ~6 GB |
| CLIP-L | `clip_l.safetensors` | ~250 MB |
| T5-XXL | `t5xxl_fp16.safetensors` | ~9.5 GB |
| VAE | `ae.safetensors` | ~160 MB |

Place models in their respective ComfyUI directories:
- `models/diffusion_models/` (UNET)
- `models/text_encoders/` (CLIP-L, T5-XXL)  
- `models/vae/` (VAE)

## Proven Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Resolution | 1088 × 1920 | 9:16 portrait. Best for character shots. |
| Steps | 20 | Sweet spot. 30 for extra detail, diminishing returns after. |
| CFG | 1.0 | FLUX uses classifier-free guidance differently — 1.0 is correct. |
| Sampler | euler | Consistent results. |
| Scheduler | simple | Works well with euler. |

## Prompt Tips

- **Front-load important terms.** FLUX gives more weight to the beginning of the prompt.
- `"full body shot, head to toe, feet visible"` at the START for full body.
- Keep prompts under ~50 words. Focused > verbose.
- Specific descriptors beat generic ones: `"crimson silk evening gown"` > `"nice dress"`.

## Resolution Presets

| Aspect | Width | Height | Use Case |
|--------|-------|--------|----------|
| 9:16 | 1088 | 1920 | Portrait / full body |
| 16:9 | 1920 | 1088 | Landscape / environmental |
| 1:1 | 1024 | 1024 | Close-up / square |

## VRAM Requirements

- **~12-14 GB** with fp8 checkpoint
- Works on RTX 3080 10GB with fp8 (tight but doable)
- Comfortable on R9700 32GB or RTX 4090 24GB

## Usage

1. Open `workflow.json` in ComfyUI (drag and drop or File → Open)
2. Edit the prompt in the CLIPTextEncode node
3. Adjust resolution in EmptySD3LatentImage if needed
4. Queue prompt (Ctrl+Enter)
