#!/usr/bin/env bash
# Two-stage frozen-assistant inference sweep for full-coverage F2D-DiD-200.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
# Capture experiment-specific overrides before .env is sourced. Generic .env
# files from CIRU-40 runs often contain UNITS=40 and must not retarget this
# fixed full-coverage sweep.
REQUESTED_UNITS="${F2D_DID_UNITS:-200}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

SPLIT="${SPLIT:-forget05}"
SEED="${SEED:-42}"
UNITS="$REQUESTED_UNITS"
TRAIN_TAG="${TRAIN_TAG:-b200_s48_u1}"
TRAINING_SWEEP_NAME="${TRAINING_SWEEP_NAME:-f2d_did200_full_authorblock_seed42}"
MANIFEST="${MANIFEST:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${TRAINING_SWEEP_NAME}/manifest.csv}"
CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_v1.jsonl}"
SEARCH_NAME="${SEARCH_NAME:-f2d_did200_s48_infer}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-48}"

if [ "$UNITS" -ne 200 ]; then
    echo "F2D-DiD full-coverage inference requires F2D_DID_UNITS=200." >&2
    exit 1
fi

if [ ! -s "$MANIFEST" ]; then
    echo "Missing training manifest: $MANIFEST" >&2
    exit 1
fi
if [ ! -s "$CF_PATH" ]; then
    echo "Missing frozen full-coverage causal data: $CF_PATH" >&2
    exit 1
fi

MODEL_ROOT="${MODEL_ROOT:-$(awk -F, -v tag="$TRAIN_TAG" '$1==tag {print $NF}' "$MANIFEST")}"
if [ -z "$MODEL_ROOT" ] || [ ! -d "$MODEL_ROOT" ]; then
    echo "Could not resolve model root for $TRAIN_TAG from $MANIFEST" >&2
    exit 1
fi

checkpoint_for() {
    find "$MODEL_ROOT/$1" -type d -name "checkpoint-${CHECKPOINT_STEP}" 2>/dev/null \
        | sort | tail -1
}
A1_CKPT="$(checkpoint_for a1)"
A2_CKPT="$(checkpoint_for a2)"
if [ -z "$A1_CKPT" ] || [ -z "$A2_CKPT" ]; then
    echo "Missing paired checkpoint-${CHECKPOINT_STEP} under $MODEL_ROOT" >&2
    exit 1
fi

export MODE=full SPLIT GPUS="${GPUS:-0 1 2 3}"
export MODELS_ROOT="$MODEL_ROOT" CF_PATH
export A1_CHECKPOINT_OVERRIDE="$A1_CKPT"
export A2_CHECKPOINT_OVERRIDE="$A2_CKPT"
export VIEWS=1
export A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2
export A1_TRAIN_STEPS="$CHECKPOINT_STEP" A2_TRAIN_STEPS="$CHECKPOINT_STEP"
export F2R_VARIANT="F2D-DiD-Balanced-${UNITS}-FullAuthorBlock-Inference"
export ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null
export EVAL_BS="${EVAL_BS:-4}" RESUME="${RESUME:-true}"
export LOAD_DOTENV=0

echo "============================================================"
echo "F2D-DiD-200 frozen inference search"
echo "  training tag : $TRAIN_TAG"
echo "  model root   : $MODEL_ROOT"
echo "  A1           : $A1_CKPT"
echo "  A2           : $A2_CKPT"
echo "  causal data  : $CF_PATH"
echo "  GPUs         : $GPUS"
echo "  coarse       : 5 x 4 x 2 = 40"
echo "  fine         : 3 x 3 x 3 = 27"
echo "============================================================"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "Dry run complete; paired frozen checkpoints resolved successfully."
    exit 0
fi

exec env \
    SEARCH_NAME="$SEARCH_NAME" \
    COARSE_A1="${COARSE_A1:--1.2 -1.3 -1.4 -1.5 -1.6}" \
    COARSE_A2="${COARSE_A2:-0.4 0.6 0.8 1.0}" \
    COARSE_FILTERS="${COARSE_FILTERS:-0.0001 0.0002}" \
    TARGET_AGG="${TARGET_AGG:-0.58}" \
    TARGET_MARGIN="${TARGET_MARGIN:-0.005}" \
    bash "$EASE_ROOT/scripts/sweep_f2r_beat_bss.sh"
