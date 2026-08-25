#!/usr/bin/env bash
# Frozen inference-strength scan around the current Llama-3.2-3B forget01
# boundary. Reuses the completed moderate 4/4 assistants and performs no
# retraining, routing, gating, calibration, or retain-data training access.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_SWEEP="${BASE_SWEEP:-f2d_v512_dual_depth4_3b_exact_subset_seed42}"
BASE_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_3b_forget01_${BASE_SWEEP}/moderate"

latest_checkpoint() {
    find "$1" -type d -name 'checkpoint-*' 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

A1_CHECKPOINT="$(latest_checkpoint "$BASE_ROOT/a1_job/a1")"
A2_CHECKPOINT="$(latest_checkpoint "$BASE_ROOT/a2_job/a2")"
for checkpoint in "$A1_CHECKPOINT" "$A2_CHECKPOINT"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing completed forget01 3B moderate checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

REFERENCE_PATH="$(cd "$A1_CHECKPOINT/../fullmodel" 2>/dev/null && pwd || true)"
if [ -z "$REFERENCE_PATH" ] || [ ! -d "$REFERENCE_PATH" ]; then
    echo "Missing depth-4 reference for $A1_CHECKPOINT" >&2
    exit 1
fi

DATA_PATH="${F2D_V512_DATA_PATH:-$EASE_ROOT/ULD/data/ciru/forget01_author_pairbudget40_seed42_v5_12_exact_subset_v1.jsonl}"
RETAIN_REFERENCE="$EASE_ROOT/open-unlearning/saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json"
if [ ! -s "$DATA_PATH" ] || [ ! -s "$RETAIN_REFERENCE" ]; then
    echo "Missing frozen data or retain99 evaluation reference." >&2
    echo "data=$DATA_PATH" >&2
    echo "retain=$RETAIN_REFERENCE" >&2
    exit 1
fi

SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth4_3b_strength12_seed42}"
RESULTS_DIR="${RESULTS_DIR:-$EASE_ROOT/open-unlearning/saves/sweeps/forget01_${SWEEP_NAME}}"

cat <<EOF
============================================================
Forget01 Llama-3.2-3B frozen strength scan
  assistants     : moderate 4/4 (32/24 steps)
  A1 weights     : -2.0 -2.2 -2.4 -2.6
  A2 weights     : 1.5 1.7 1.9
  top filter     : 0.0004
  evaluations    : 12
  retraining     : none
  routing/gating : disabled
  composition    : shared depth-4 reference_delta
  retain training: none
  results        : $RESULTS_DIR
============================================================
EOF

env \
    MODE=full SPLIT=forget01 \
    GPUS="${GPUS:-0 1 2 3}" EVAL_BS="${EVAL_BS:-1}" RESUME="${RESUME:-true}" \
    CF_PATH="$DATA_PATH" \
    A1_CHECKPOINT_OVERRIDE="$A1_CHECKPOINT" \
    A2_CHECKPOINT_OVERRIDE="$A2_CHECKPOINT" \
    A1_NUM_LAYER=4 A2_NUM_LAYER=4 \
    A1_TRAIN_STEPS=32 A2_TRAIN_STEPS=24 \
    A1_TRAIN_LR=5e-4 A2_TRAIN_LR=1e-3 \
    A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
    TRAIN_MODEL_CONFIG=llama-3-3b \
    EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD \
    HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.2-3B-Instruct \
    HF_MODEL_NAME=Llama-3.2-3B-Instruct \
    TASK_MODEL_NAME=Llama-3.2-3B-Instruct \
    RETAIN_LOGS_PATH="$RETAIN_REFERENCE" AUTO_FETCH_RETAIN_LOGS=0 \
    COMPOSITION_MODE=reference_delta REFERENCE_PATH="$REFERENCE_PATH" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false SEQUENCE_ROUTER_ENABLED=false \
    CALIBRATION_PATH=null SELECTION_RETAIN_ACCESS=true \
    WEIGHT_A1_GRID="-2.0 -2.2 -2.4 -2.6" \
    WEIGHT_A2_GRID="1.5 1.7 1.9" \
    TOP_FILTERS=0.0004 \
    SWEEP_NAME="$SWEEP_NAME" RESULTS_DIR="$RESULTS_DIR" \
    TARGET_AGG=0.57 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
