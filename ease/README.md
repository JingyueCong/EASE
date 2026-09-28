# EASE TOFU implementation

This directory contains the canonical TOFU training and evaluation pipeline
for EASE. The Python package is `ease`; Hydra configurations are under
`configs/`, and runnable entry points are under `scripts/` and
`bashes/tofu/`.

## Setup

```bash
conda env create -f environment.yaml
conda activate ease
pip install -e .
```

## Run

From this directory:

```bash
GPUS=0 SPLIT=forget10 K=80 bash bashes/tofu/ease_pipeline.sh
```

The pipeline selects `R_sub`, trains assistants A1 and A2, and evaluates the
composed EASE model. Override the environment variables documented in the
script to change the split, GPU assignment, checkpoint paths, or output
location.

## Layout

- `ease/`: model, data, trainer, and evaluation utilities
- `configs/model_mode/assistant.yaml`: single-assistant training mode
- `configs/model_mode/ease.yaml`: two-assistant EASE inference mode
- `configs/data_mode/ease_a1.yaml`: A1 data composition
- `configs/data_mode/ease_a2.yaml`: A2 data composition
- `scripts/hf_forget_train.py`: assistant training entry point
- `scripts/eval_tofu.py`: TOFU evaluation entry point
- `bashes/tofu/ease_pipeline.sh`: end-to-end pipeline

## Upstream attribution

This implementation builds on the MIT-licensed framework introduced in
“Reversing the Forget-Retain Objectives: An Efficient LLM Unlearning
Framework from Logit Difference” (Ji et al., 2024). The upstream license is
preserved in [LICENSE](LICENSE).
