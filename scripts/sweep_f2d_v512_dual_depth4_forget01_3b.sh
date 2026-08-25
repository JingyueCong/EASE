#!/usr/bin/env bash
# Isolated Llama-3.2-3B replication of the forget01 static 4/4 sweep.
# The shared implementation retains the exact forget01 data and audit gates;
# all model, checkpoint, task, log, and sweep paths are 3B-specific.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export MODEL_TAG=3b
export TRAIN_MODEL_CONFIG=llama-3-3b
export EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD
export HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.2-3B-Instruct
export HF_MODEL_NAME=Llama-3.2-3B-Instruct
export TASK_MODEL_NAME=Llama-3.2-3B-Instruct
export SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth4_3b_exact_subset_seed42}"
export TRAIN_BS="${TRAIN_BS:-1}"
export TRAIN_GA="${TRAIN_GA:-16}"
export EVAL_BS="${EVAL_BS:-1}"

exec bash "$EASE_ROOT/scripts/sweep_f2d_v512_dual_depth4_forget01.sh"
