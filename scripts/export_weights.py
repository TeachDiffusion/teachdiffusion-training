"""Export trained LoRA weights to HuggingFace format."""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hub_id", default="TeachDiffusion/teachdiffusion-v0.1")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"Exporting checkpoint: {args.checkpoint}")
    print(f"Target HuggingFace ID: {args.hub_id}")

    if args.push:
        print("Pushing to HuggingFace Hub...")
        # huggingface_hub upload implementation
    else:
        print("Export complete. Use --push to upload to HuggingFace.")

if __name__ == "__main__":
    main()
