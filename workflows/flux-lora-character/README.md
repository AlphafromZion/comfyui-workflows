# FLUX LoRA Character Workflow

FLUX.1-dev with LoRA support for consistent character generation. Same clean pipeline as the portrait workflow, with a `LoraLoaderModelOnly` node inserted between the UNET and KSampler.

## Node Graph

```
UNETLoader → LoraLoaderModelOnly ──────────────────────┐
DualCLIPLoader → CLIPTextEncode ────┬─ positive ────────┤
                                    └─ ConditioningZeroOut → negative ┤
VAELoader ──────────────────────────────────────────────────┐        │
EmptySD3LatentImage ──────────────────── latent ─────────── KSampler → VAEDecode → SaveImage
```

## Setup

1. Place your FLUX LoRA `.safetensors` in `ComfyUI/models/loras/`
2. Open `workflow.json` in ComfyUI
3. In the `LoraLoaderModelOnly` node, select your LoRA file
4. Update the prompt with your LoRA's trigger word
5. Queue prompt

## Required Models

Same as the [portrait workflow](../flux-portrait/README.md#required-models), plus your LoRA file.

## Proven Settings

| Setting | Value | Notes |
|---------|-------|-------|
| LoRA Strength | 0.9-1.0 | Start at 1.0, reduce if overfitting |
| Resolution | 1088 × 1920 | 9:16 portrait |
| Steps | 20 | 25-30 for extra detail |
| CFG | 1.0 | Standard for FLUX |
| Sampler | euler / simple | Consistent results |

## LoRA Training Tips

See the full [FLUX LoRA Training Guide](../../guides/flux-lora-training.md) for how to train your own.

**Quick summary:**
- 15-30 images minimum, 50-120 for best quality
- Include 10-15% face crops in your dataset
- Use Kohya/Flux Gym with adafactor optimizer
- network_dim 64 (standard) or 128 (max quality)
- 16 epochs, learning rate 8e-5

## Prompt Engineering for LoRA Characters

- **Always include the trigger word** at the start of your prompt
- Character traits baked into the LoRA don't need repeating (hair color, eye color, etc.)
- **Front-load composition:** `"full body shot, head to toe, your_trigger, wearing..."` 
- Be explicit about clothing — some LoRAs default NSFW depending on training data bias
- LoRA strength 0.7-0.8 gives more prompt flexibility; 1.0 gives strongest character likeness

## VRAM Requirements

- Same as portrait workflow (~12-14 GB with fp8)
- LoRA adds minimal overhead (~5-10 seconds per generation)
