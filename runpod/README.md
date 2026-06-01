# RunPod Training Guide

## Setup
1. Create a RunPod instance with A100 80GB
2. SSH in and run: `bash runpod/setup.sh`
3. Upload your dataset to `teachdiffusion-data/data/filtered/`
4. Run: `bash runpod/train.sh`

## Estimated Cost
- A100 80GB: ~$2.50/hr
- Training ~5000 steps: ~4-8 hours
- Total: ~$10-20
