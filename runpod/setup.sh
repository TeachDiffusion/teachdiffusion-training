#!/bin/bash
# One-command RunPod environment setup for TeachDiffusion training

set -e

echo "=== TeachDiffusion Training Environment Setup ==="

# Update system
apt-get update -qq

# Install Python dependencies
pip install --quiet torch torchvision torchaudio
pip install --quiet diffusers accelerate transformers peft
pip install --quiet huggingface-hub tensorboard
pip install --quiet pyyaml tqdm

# Clone repos
git clone https://github.com/TeachDiffusion/TeachDiffusion.git || true
git clone https://github.com/TeachDiffusion/teachdiffusion-training.git || true
git clone https://github.com/TeachDiffusion/teachdiffusion-data.git || true

# Install TeachDiffusion
cd TeachDiffusion && pip install -e . && cd ..

echo "=== Setup Complete ==="
echo "Run: bash runpod/train.sh"
