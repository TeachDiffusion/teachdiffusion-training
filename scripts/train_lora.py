"""LoRA fine-tuning script for Wan 2.2 on math teaching data.

Usage:
    python train_lora.py --config configs/wan2_lora_480p.yaml
"""

import argparse
import json
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model(config: dict):
    """Load base model and apply LoRA."""
    try:
        import torch
        from diffusers import DiffusionPipeline
        from peft import LoraConfig, get_peft_model

        print(f"Loading base model: {config['model']['pretrained_path']}")
        pipeline = DiffusionPipeline.from_pretrained(
            config["model"]["pretrained_path"],
            torch_dtype=getattr(torch, config["model"]["dtype"]),
        )

        lora_config = LoraConfig(
            r=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
        )

        model = get_peft_model(pipeline.unet, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.2%})")

        return pipeline, model

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch diffusers peft accelerate transformers")
        return None, None


def load_dataset(config: dict):
    """Load training dataset from manifest."""
    manifest_path = Path(config["data"]["dataset_path"]) / config["data"]["manifest_file"]
    if not manifest_path.exists():
        print(f"Dataset manifest not found: {manifest_path}")
        print("Run the teachdiffusion-data pipeline first.")
        return None

    with open(manifest_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples")
    return data


def train(config: dict):
    """Main training loop."""
    print("=" * 60)
    print("TeachDiffusion LoRA Training")
    print("=" * 60)

    pipeline, model = setup_model(config)
    if model is None:
        print("\nDry run complete. Install dependencies and GPU to train.")
        return

    dataset = load_dataset(config)
    if dataset is None:
        return

    training_config = config["training"]
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining config:")
    print(f"  Steps: {training_config['max_train_steps']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  LoRA rank: {config['lora']['rank']}")
    print(f"  Output: {output_dir}")

    # Training loop would go here
    # For now, print instructions
    print("\n[Training loop placeholder]")
    print("Full training implementation requires:")
    print("  1. Video dataloader with frame sampling")
    print("  2. Noise scheduling + diffusion loss")
    print("  3. LoRA gradient computation")
    print("  4. Checkpoint saving + evaluation")
    print("\nSee HuggingFace Diffusers documentation for implementation details.")


def main():
    parser = argparse.ArgumentParser(description="Train TeachDiffusion LoRA")
    parser.add_argument("--config", default="configs/wan2_lora_480p.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
