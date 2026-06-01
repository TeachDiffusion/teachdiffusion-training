"""Generate sample teaching videos from a checkpoint for evaluation."""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/samples")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples from {args.checkpoint}")
    print(f"Output directory: {args.output}")
    print("[Requires GPU + trained weights]")

if __name__ == "__main__":
    main()
