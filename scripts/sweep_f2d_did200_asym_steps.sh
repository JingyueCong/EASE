#!/usr/bin/env bash
# Four-GPU asymmetric A1/A2 step sweep for full-coverage F2D-DiD-200.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
# Preserve explicit launch settings before sourcing a generic experiment .env.
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"
REQUESTED_DRY_RUN="${DRY_RUN:-false}"
REQUESTED_CIRU_PATH="${CIRU_PATH:-}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

SPLIT="${F2D_DID_SPLIT:-forget05}"
SEED="${F2D_DID_SEED:-42}"
UNITS=200
SWEEP_NAME="${F2D_DID_SWEEP_NAME:-f2d_did200_asym_steps_seed${SEED}}"
if [ -n "$REQUESTED_CIRU_PATH" ]; then
    CIRU_PATH="$REQUESTED_CIRU_PATH"
else
    CIRU_PATH="${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_v1.jsonl"
fi

if [ ! -s "$CIRU_PATH" ]; then
    echo "Missing audited full-coverage causal data: $CIRU_PATH" >&2
    exit 1
fi

# Format: tag:layers:ranks:lrs:epochs:uniform_weights:explicit_steps.
# Epochs remain one because positive explicit step budgets are authoritative.
TRAIN_CONFIGS="${F2D_TRAIN_CONFIGS:-\
a72_a60:2:2:16:16:1e-3:1e-3:1:1:1:1:72:60 \
a72_a72:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72 \
a84_a60:2:2:16:16:1e-3:1e-3:1:1:1:1:84:60 \
a84_a72:2:2:16:16:1e-3:1e-3:1:1:1:1:84:72}"

echo "============================================================"
echo "F2D-DiD-200 asymmetric assistant-step sweep"
echo "  causal data      : $CIRU_PATH"
echo "  GPUs             : $REQUESTED_GPUS"
echo "  configurations   : 72/60 72/72 84/60 84/72"
echo "  inference point  : -2.0 / 1.8 / 0.0002"
echo "  target           : lower EM/knowledge-TR while preserving MU"
echo "============================================================"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 \
    CIRU_PATH="$CIRU_PATH" CF_PATH="$CIRU_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1=-2.0 WEIGHT_A2=1.8 TOP_FILTER=0.0002 \
    F2R_VARIANT="F2D-DiD-Balanced-200-AsymSteps" \
    EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
    DRY_RUN="$REQUESTED_DRY_RUN" \
    TARGET_AGG="${TARGET_AGG:-0.58}" TARGET_MARGIN="${TARGET_MARGIN:-0.005}" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
