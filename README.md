<p align="center">
  <img src="https://raw.githubusercontent.com/TeachDiffusion/.github/main/assets/teachdiffusion_logo.svg" alt="TeachDiffusion" width="280"/>
</p>

<h1 align="center">teachdiffusion-training</h1>

<p align="center">
  LoRA fine-tuning scripts and configs for Wan 2.2.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

> For project mission, the full 8-layer architecture, sibling repositories, and roadmap, see the [TeachDiffusion organization profile](https://github.com/TeachDiffusion).

---

## About this repo

This repo holds **only the training code** — scripts, configs, and cloud-GPU setup. It consumes the manifest produced by [`teachdiffusion-data`](https://github.com/TeachDiffusion/teachdiffusion-data) and emits LoRA weights that the core [`TeachDiffusion`](https://github.com/TeachDiffusion/TeachDiffusion) package loads at inference time.

## Quick start

### RunPod (recommended)

```bash
bash runpod/setup.sh
bash runpod/train.sh
```

See [`runpod/README.md`](runpod/README.md) for the cloud-GPU specifics.

### Local / manual

```bash
pip install torch diffusers accelerate peft transformers
python scripts/train_lora.py --config configs/wan2_lora_480p.yaml
```

## Configs

| Config | Resolution | Approx. VRAM |
|---|---|---|
| [`configs/wan2_lora_480p.yaml`](configs/wan2_lora_480p.yaml) | 480p | ~40 GB |
| [`configs/wan2_lora_720p.yaml`](configs/wan2_lora_720p.yaml) | 720p | ~60 GB |

## Scripts

| Script | Purpose |
|---|---|
| [`scripts/train_lora.py`](scripts/train_lora.py) | Train the LoRA adapter against a config |
| [`scripts/generate_samples.py`](scripts/generate_samples.py) | Render sample videos from a checkpoint |
| [`scripts/evaluate.py`](scripts/evaluate.py) | Evaluate a checkpoint against held-out clips |
| [`scripts/export_weights.py`](scripts/export_weights.py) | Export checkpoint to HuggingFace format |

### Typical post-training flow

```bash
python scripts/evaluate.py         --checkpoint outputs/checkpoint-5000
python scripts/generate_samples.py --checkpoint outputs/checkpoint-5000
python scripts/export_weights.py   --checkpoint outputs/checkpoint-5000 \
                                    --hub_id TeachDiffusion/teachdiffusion-v0.1
```

## Experiments

Notes on training runs and ablations are tracked in [`experiments/`](experiments).

## License

Apache 2.0 — see [LICENSE](LICENSE).
