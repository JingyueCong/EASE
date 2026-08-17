#!/usr/bin/env bash
# Four-GPU factorial Difference-in-Differences training sweep for F2D.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
SPLIT="${SPLIT:-forget05}"
SEED="${SEED:-42}"
UNITS="${UNITS:-40}"
SWEEP_NAME="${SWEEP_NAME:-f2d_did${UNITS}_seed${SEED}}"
CIRU_PATH="${CIRU_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2.jsonl}"

if [ ! -s "$CIRU_PATH" ]; then
    echo "Missing audited factorial data: $CIRU_PATH" >&2
    exit 1
fi

# This 2x2 grid tests duration and contrast strength while holding the
# architecture, causal units, optimizer, and inference point fixed.
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
did_e12_u1:2:2:16:16:1e-3:1e-3:12:12:1:1 \
did_e12_u2:2:2:16:16:1e-3:1e-3:12:12:2:2 \
did_e18_u1:2:2:16:16:1e-3:1e-3:18:18:1:1 \
did_e18_u2:2:2:16:16:1e-3:1e-3:18:18:2:2}"

export MODE="${MODE:-full}"
export SPLIT SEED SWEEP_NAME TRAIN_CONFIGS
export VIEWS="${VIEWS:-1}"
export GPUS="${GPUS:-0 1 2 3}"
export TASK_PREFIX="${TASK_PREFIX:-tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2D_DiD${UNITS}_seed${SEED}}"
export CF_PATH="$CIRU_PATH"
export MODELS_SWEEP_ROOT="${MODELS_SWEEP_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2d_did_1b_${SPLIT}_${SWEEP_NAME}}"
export RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
export A1_DATA_MODE=f2d_did_a1
export A2_DATA_MODE=f2d_did_a2
export F2R_VARIANT="${F2R_VARIANT:-F2D-DiD-Balanced}"
# This is the uncalibrated causal-assignment experiment.  Isolate it from
# ALIGNMENT/GATE variables left in the login shell or .env by other ladders.
export ALIGNMENT_ENABLED=false
export GATE_ENABLED=false
export CALIBRATION_PATH=null
export LOAD_DOTENV=0
export WEIGHT_A1="${WEIGHT_A1:--1.3}"
export WEIGHT_A2="${WEIGHT_A2:-0.8}"
export TOP_FILTER="${TOP_FILTER:-0.0002}"

echo "F2D-DiD estimand: (C11-C01) - (C10-C00)"
echo "Audited units: $CIRU_PATH"
echo "Stage-1 inference point: $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "Calibration isolation: alignment=$ALIGNMENT_ENABLED gate=$GATE_ENABLED artifact=$CALIBRATION_PATH"
exec bash "${EASE_ROOT}/scripts/sweep_f2r_training.sh"
