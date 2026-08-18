#!/usr/bin/env bash
# Extend the F2D-DiD-200 S60 inference frontier along the A1/A2 diagonal.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"
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
TRAIN_TAG=b200_s60_u1
CHECKPOINT_STEP=60
TRAINING_SWEEP_NAME="f2d_did200_full_authorblock_seed42"
MANIFEST="${MANIFEST:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${TRAINING_SWEEP_NAME}/manifest.csv}"
CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_v1.jsonl}"
SWEEP_NAME="${SWEEP_NAME:-f2d_did200_s60_boundary}"

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

WEIGHT_PAIRS="${WEIGHT_PAIRS:-\
-1.7:1.5 \
-1.8:1.5 -1.8:1.6 \
-1.9:1.6 -1.9:1.7 \
-2.0:1.7 -2.0:1.8 \
-2.1:1.8 -2.1:1.9}"
TOP_FILTERS="${TOP_FILTERS:-0.0001 0.0002 0.0004}"

echo "============================================================"
echo "F2D-DiD-200 S60 boundary inference sweep"
echo "  training tag : $TRAIN_TAG"
echo "  A1           : $A1_CKPT"
echo "  A2           : $A2_CKPT"
echo "  pairs        : $WEIGHT_PAIRS"
echo "  filters      : $TOP_FILTERS"
echo "  GPUs         : $REQUESTED_GPUS"
echo "  configs      : 27"
echo "============================================================"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "Dry run complete; paired frozen checkpoints resolved successfully."
    exit 0
fi

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" VIEWS=1 \
    MODELS_ROOT="$MODEL_ROOT" CF_PATH="$CF_PATH" \
    A1_CHECKPOINT_OVERRIDE="$A1_CKPT" \
    A2_CHECKPOINT_OVERRIDE="$A2_CKPT" \
    A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
    A1_TRAIN_STEPS=60 A2_TRAIN_STEPS=60 \
    WEIGHT_A1_GRID="" WEIGHT_A2_GRID="" \
    WEIGHT_PAIRS="$WEIGHT_PAIRS" TOP_FILTERS="$TOP_FILTERS" \
    SWEEP_NAME="$SWEEP_NAME" \
    F2R_VARIANT="F2D-DiD-Balanced-200-S60-Boundary" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
    TARGET_AGG="${TARGET_AGG:-0.58}" TARGET_MARGIN="${TARGET_MARGIN:-0.005}" \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
