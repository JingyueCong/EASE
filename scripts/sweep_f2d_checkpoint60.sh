#!/usr/bin/env bash
# Evaluation-only F2D checkpoint-60 sweep using the completed ep7/a2u2 run.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT="${SPLIT:-forget05}"
UNITS="${UNITS:-40}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0 1 2 3}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-60}"
SWEEP_NAME="${SWEEP_NAME:-f2d${UNITS}_ep7_a2u2_ckpt${CHECKPOINT_STEP}_infer}"
TRAINING_SWEEP_NAME="${TRAINING_SWEEP_NAME:-f2d${UNITS}_intermediate_seed${SEED}}"
MANIFEST="${MANIFEST:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${TRAINING_SWEEP_NAME}/manifest.csv}"
CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/f2d/${SPLIT}_f2d${UNITS}_seed${SEED}.jsonl}"

if [ ! -s "$MANIFEST" ]; then
    echo "Missing F2D training manifest: $MANIFEST" >&2
    exit 1
fi
MODEL_ROOT="${MODEL_ROOT:-$(awk -F, '$1=="ep7_a2u2"{print $NF}' "$MANIFEST")}"
if [ -z "$MODEL_ROOT" ] || [ ! -d "$MODEL_ROOT" ]; then
    echo "Could not resolve ep7_a2u2 model root from $MANIFEST" >&2
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

# A six-epoch checkpoint is expected to need slightly stronger deletion than
# checkpoint-70. The paired grid follows that hypothesis and concentrates on
# the low-filter plateau found by the completed checkpoint-70 sweep.
WEIGHT_PAIRS="${WEIGHT_PAIRS:-\
-1.20:0.50 -1.20:0.60 -1.30:0.60 -1.30:0.70 \
-1.30:0.80 -1.40:0.70 -1.40:0.80 -1.40:0.90}"
TOP_FILTERS="${TOP_FILTERS:-0.0001 0.0002 0.0005}"

echo "============================================================"
echo "F2D checkpoint-${CHECKPOINT_STEP} frozen-assistant inference sweep"
echo "  model root : $MODEL_ROOT"
echo "  A1         : $A1_CKPT"
echo "  A2         : $A2_CKPT"
echo "  pairs      : $WEIGHT_PAIRS"
echo "  filters    : $TOP_FILTERS"
echo "  GPUs       : $GPUS"
echo "  configs    : 24"
echo "============================================================"

exec env \
    MODE=full SPLIT="$SPLIT" GPUS="$GPUS" VIEWS=1 \
    MODELS_ROOT="$MODEL_ROOT" CF_PATH="$CF_PATH" \
    A1_CHECKPOINT_OVERRIDE="$A1_CKPT" \
    A2_CHECKPOINT_OVERRIDE="$A2_CKPT" \
    A1_NUM_LAYER=2 A2_NUM_LAYER=2 \
    A1_LORA_R=16 A2_LORA_R=16 \
    A1_LORA_ALPHA=32 A2_LORA_ALPHA=32 \
    A1_TRAIN_LR=1e-3 A2_TRAIN_LR=1e-3 \
    A1_TRAIN_EP=7 A2_TRAIN_EP=7 \
    A1_RETAIN_WEIGHT=5 A2_RETAIN_WEIGHT=2 \
    A1_SEED="$SEED" A2_SEED="$SEED" \
    WEIGHT_A1_GRID="" WEIGHT_A2_GRID="" \
    WEIGHT_PAIRS="$WEIGHT_PAIRS" TOP_FILTERS="$TOP_FILTERS" \
    SWEEP_NAME="$SWEEP_NAME" \
    F2R_VARIANT="F2D-C01+Placebo-Checkpoint${CHECKPOINT_STEP}" \
    EVAL_BS="${EVAL_BS:-4}" RESUME="${RESUME:-true}" \
    TARGET_AGG="${TARGET_AGG:-0.58}" TARGET_MARGIN="${TARGET_MARGIN:-0.005}" \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
