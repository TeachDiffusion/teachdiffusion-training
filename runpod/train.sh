#!/bin/bash
# Launch TeachDiffusion LoRA training on RunPod

set -e

echo "=== Starting TeachDiffusion LoRA Training ==="

cd teachdiffusion-training

python scripts/train_lora.py \
    --config configs/wan2_lora_480p.yaml

echo "=== Training Complete ==="
