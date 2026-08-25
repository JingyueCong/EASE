#!/usr/bin/env bash
# Isolated Llama-3.1-8B replication of the forget05 static 4/4 sweep.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export EASE_ROOT
export MODEL_SIZE_TAG=8b
export MODEL_DISPLAY_NAME=Llama-3.1-8B
export TRAIN_MODEL_CONFIG=llama-3-8b
export EVAL_MODEL_CONFIG=Llama-3.1-8B-Instruct_DualULD
export HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.1-8B-Instruct
export HF_MODEL_NAME=Llama-3.1-8B-Instruct
export TASK_MODEL_NAME=Llama-3.1-8B-Instruct
export MODEL_TAG="${MODEL_TAG:-f2d_v512_dual_depth4_8b_train_seed42}"
export SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth4_8b_coarse12_seed42}"
export RETAIN_REFERENCE_RELATIVE="${RETAIN_REFERENCE_RELATIVE:-tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json}"

exec bash "$EASE_ROOT/scripts/sweep_f2d_v512_dual_depth4_forget05_3b.sh"
