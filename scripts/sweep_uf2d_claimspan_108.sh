#!/usr/bin/env bash
# 108-point, four-GPU inference search for the frozen U-F2D Claim+Span+KL pair.
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

SPLIT="${SPLIT:-forget05}"
SEED="${SEED:-42}"
UNITS="${UNITS:-200}"
SWEEP_NAME="${SWEEP_NAME:-uf2d_claimspan_infer108_seed${SEED}}"
CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_hier_v1.jsonl}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/uf2d_1b_${SPLIT}_uf2d_hierarchy_ladder_seed${SEED}/claim_span_kl_b0p05}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"

# This broad grid is intentionally asymmetric toward a stronger negative A1
# because the fixed-point ladder showed high utility but insufficient removal.
WEIGHT_A1_GRID="${WEIGHT_A1_GRID:--1.8 -2.2 -2.6 -3.0 -3.4 -3.8}"
WEIGHT_A2_GRID="${WEIGHT_A2_GRID:-1.0 1.4 1.8 2.2 2.6 3.0}"
TOP_FILTERS="${TOP_FILTERS:-0.0001 0.0004 0.001}"

if [ ! -s "$CF_PATH" ]; then
    echo "Missing hierarchical 200-unit data: $CF_PATH" >&2
    echo "Run scripts/run_uf2d_tofu_ladder.sh first." >&2
    exit 1
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "Missing frozen Claim+Span+KL model root: $MODELS_ROOT" >&2
    echo "Run scripts/run_uf2d_tofu_ladder.sh first." >&2
    exit 1
fi

read -r -a GPU_LIST <<< "$REQUESTED_GPUS"
if [ "${#GPU_LIST[@]}" -ne 4 ]; then
    echo "Expected exactly four GPU ids, got: $REQUESTED_GPUS" >&2
    exit 1
fi

MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-14000}"
for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [ "$free_mib" -lt "$MIN_FREE_GPU_MIB" ]; then
        echo "GPU $gpu has only ${free_mib} MiB free; need ${MIN_FREE_GPU_MIB} MiB." >&2
        echo "Inspect nvidia-smi/ollama before launching the 108-point sweep." >&2
        exit 1
    fi
done

read -r -a A1_LIST <<< "$WEIGHT_A1_GRID"
read -r -a A2_LIST <<< "$WEIGHT_A2_GRID"
read -r -a FILTER_LIST <<< "$TOP_FILTERS"
EXPECTED=$((${#A1_LIST[@]} * ${#A2_LIST[@]} * ${#FILTER_LIST[@]}))
if [ "$EXPECTED" -lt 100 ]; then
    echo "This runner is intended for >=100 configurations; resolved $EXPECTED." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
cat > "$RESULTS_DIR/SEARCH_PROTOCOL.txt" <<EOF
method=U-F2D-ClaimSpan-KL
stage=forget05-development-inference-only
training_retain_access=false
selection_retain_access=true
seed=$SEED
units=$UNITS
frozen_data=$CF_PATH
frozen_models=$MODELS_ROOT
weight_a1_grid=$WEIGHT_A1_GRID
weight_a2_grid=$WEIGHT_A2_GRID
top_filters=$TOP_FILTERS
configurations=$EXPECTED
confirmation_required=new-seeds-and-splits
EOF

echo "============================================================"
echo "U-F2D Claim+Span+KL large inference search"
echo "  split / units   : $SPLIT / $UNITS"
echo "  frozen models   : $MODELS_ROOT"
echo "  A1 grid         : $WEIGHT_A1_GRID"
echo "  A2 grid         : $WEIGHT_A2_GRID"
echo "  filters         : $TOP_FILTERS"
echo "  configurations  : $EXPECTED"
echo "  GPUs            : $REQUESTED_GPUS"
echo "  result root     : $RESULTS_DIR"
echo "  protocol        : development only; selection_retain_access=true"
echo "============================================================"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
    VIEWS=1 CF_PATH="$CF_PATH" MODELS_ROOT="$MODELS_ROOT" \
    RESULTS_DIR="$RESULTS_DIR" SWEEP_NAME="$SWEEP_NAME" \
    A1_DATA_MODE=uf2d_hier_a1 A2_DATA_MODE=uf2d_hier_a2 \
    TRAIN_LOSS_CONFIG=factorial_hierarchical \
    PRESERVE_KL_WEIGHT=0.05 EVIDENCE_WEIGHT=1.0 \
    F2R_VARIANT=U-F2D-ClaimSpan-KL-Inference108 \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    WEIGHT_A1_GRID="$WEIGHT_A1_GRID" WEIGHT_A2_GRID="$WEIGHT_A2_GRID" \
    TOP_FILTERS="$TOP_FILTERS" EVAL_BS="$REQUESTED_EVAL_BS" \
    RESUME="$REQUESTED_RESUME" TARGET_AGG="${TARGET_AGG:-0.58}" \
    TARGET_MARGIN="${TARGET_MARGIN:-0.005}" \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
