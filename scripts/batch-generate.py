#!/usr/bin/env python3
"""
Batch generation helper for ComfyUI.
Queues multiple prompts from a text file or generates variations of a single prompt.

Usage:
    python batch-generate.py --host 192.168.20.11 prompts.txt
    python batch-generate.py --host 192.168.20.11 --variations "beautiful portrait" --count 5
    python batch-generate.py --host 192.168.20.11 prompts.json
"""

import argparse
import json
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def queue_prompt(host: str, port: int, workflow: dict) -> str:
    """Queue a prompt on ComfyUI and return the prompt ID."""
    url = f"http://{host}:{port}/prompt"
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get("prompt_id", "unknown")
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot connect to ComfyUI at {host}:{port} — {e}")
        return None


def check_status(host: str, port: int) -> bool:
    """Check if ComfyUI is running."""
    url = f"http://{host}:{port}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def poll_completion(host: str, port: int, prompt_id: str, timeout: int = 600) -> bool:
    """Poll ComfyUI until the prompt completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            url = f"http://{host}:{port}/history/{prompt_id}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                history = json.loads(resp.read())
                if prompt_id in history:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def build_flux_workflow(prompt: str, width: int = 1088, height: int = 1920,
                        steps: int = 20, seed: int = None,
                        checkpoint: str = "flux1-krea-dev_fp8_scaled.safetensors",
                        lora: str = None, lora_strength: float = 1.0) -> dict:
    """Build a FLUX workflow dict for the ComfyUI API."""
    if seed is None:
        seed = random.randint(0, 2**53)

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp16.safetensors",
                "type": "flux",
                "device": "default"
            }
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["3", 0]}
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]}
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0] if not lora else ["2", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["4", 0]}
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "batch"}
        }
    }

    if lora:
        workflow["2"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora,
                "strength_model": lora_strength
            }
        }

    return workflow


def load_prompts(filepath: str) -> list:
    """Load prompts from a text or JSON file."""
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            print("ERROR: JSON file must contain an array of prompt objects")
            sys.exit(1)
    else:
        # Text file: one prompt per line
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return [{"prompt": line} for line in lines]


def main():
    parser = argparse.ArgumentParser(description="Batch generate images via ComfyUI API")
    parser.add_argument("file", nargs="?", help="Prompt file (text or JSON)")
    parser.add_argument("--host", default="127.0.0.1", help="ComfyUI host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI port (default: 8188)")
    parser.add_argument("--variations", help="Generate variations of a single prompt")
    parser.add_argument("--count", type=int, default=5, help="Number of variations (default: 5)")
    parser.add_argument("--width", type=int, default=1088, help="Image width (default: 1088)")
    parser.add_argument("--height", type=int, default=1920, help="Image height (default: 1920)")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps (default: 20)")
    parser.add_argument("--lora", help="LoRA filename (e.g., my-lora.safetensors)")
    parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA strength (default: 1.0)")
    parser.add_argument("--checkpoint", default="flux1-krea-dev_fp8_scaled.safetensors",
                        help="FLUX checkpoint filename")
    parser.add_argument("--wait", action="store_true", help="Wait for each generation to complete")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per generation in seconds")

    args = parser.parse_args()

    if not args.file and not args.variations:
        parser.error("Provide a prompt file or --variations")

    # Check ComfyUI
    if not check_status(args.host, args.port):
        print(f"ERROR: ComfyUI not reachable at {args.host}:{args.port}")
        sys.exit(1)

    print(f"Connected to ComfyUI at {args.host}:{args.port}")

    # Build prompt list
    if args.variations:
        prompts = [{"prompt": args.variations} for _ in range(args.count)]
        print(f"Generating {args.count} variations of: {args.variations}")
    else:
        prompts = load_prompts(args.file)
        print(f"Loaded {len(prompts)} prompts from {args.file}")

    # Queue all prompts
    queued = 0
    for i, item in enumerate(prompts):
        prompt_text = item if isinstance(item, str) else item.get("prompt", "")
        p_width = item.get("width", args.width) if isinstance(item, dict) else args.width
        p_height = item.get("height", args.height) if isinstance(item, dict) else args.height
        p_steps = item.get("steps", args.steps) if isinstance(item, dict) else args.steps
        p_seed = item.get("seed", None) if isinstance(item, dict) else None
        p_lora = item.get("lora", args.lora) if isinstance(item, dict) else args.lora

        workflow = build_flux_workflow(
            prompt=prompt_text,
            width=p_width,
            height=p_height,
            steps=p_steps,
            seed=p_seed,
            checkpoint=args.checkpoint,
            lora=p_lora,
            lora_strength=args.lora_strength,
        )

        prompt_id = queue_prompt(args.host, args.port, workflow)
        if prompt_id:
            preview = prompt_text[:60] + "..." if len(prompt_text) > 60 else prompt_text
            print(f"  [{i+1}/{len(prompts)}] Queued: {preview} (id: {prompt_id[:8]})")
            queued += 1

            if args.wait:
                print(f"    Waiting for completion (timeout: {args.timeout}s)...")
                if poll_completion(args.host, args.port, prompt_id, args.timeout):
                    print(f"    Done!")
                else:
                    print(f"    Timed out after {args.timeout}s")
        else:
            print(f"  [{i+1}/{len(prompts)}] FAILED to queue")

        # Small delay between queues
        if i < len(prompts) - 1:
            time.sleep(0.5)

    print(f"\nQueued {queued}/{len(prompts)} prompts.")
    if not args.wait:
        print("Prompts are processing in ComfyUI. Check the web UI for progress.")


if __name__ == "__main__":
    main()
