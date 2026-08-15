#!/usr/bin/env bash
# Two-stage, target-aware F2R inference sweep against LLM-Beliefs BS-S.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-full}"
SPLIT="${SPLIT:-forget05}"
GPUS="${GPUS:-0 1 2 3}"
SEARCH_NAME="${SEARCH_NAME:-beat_bss_$(date +%Y%m%d_%H%M%S)}"
SEARCH_ROOT="${SEARCH_ROOT:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SEARCH_NAME}}"
COARSE_A1="${COARSE_A1:--0.4 -0.7 -1.0 -1.3}"
COARSE_A2="${COARSE_A2:-0.3 0.6 0.9 1.2}"
COARSE_FILTERS="${COARSE_FILTERS:-0.005 0.01 0.1}"
RESUME="${RESUME:-true}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"

case "$SPLIT" in
    forget01) TARGET_AGG="${TARGET_AGG:-0.57}" ;;
    forget05) TARGET_AGG="${TARGET_AGG:-0.58}" ;;
    forget10) TARGET_AGG="${TARGET_AGG:-0.61}" ;;
    *) echo "Unsupported SPLIT=$SPLIT" >&2; exit 1 ;;
esac

run_stage() {
    local name="$1" a1_grid="$2" a2_grid="$3" filters="$4" output="$5"
    MODE="$MODE" SPLIT="$SPLIT" GPUS="$GPUS" \
        WEIGHT_A1_GRID="$a1_grid" WEIGHT_A2_GRID="$a2_grid" \
        TOP_FILTERS="$filters" TARGET_AGG="$TARGET_AGG" RESUME="$RESUME" \
        TARGET_MARGIN="$TARGET_MARGIN" \
        SWEEP_NAME="${SEARCH_NAME}_${name}" RESULTS_DIR="$output" \
        bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
}

best_row() {
    local csv="$1"
    sed -n '2p' "$csv"
}

echo "[stage 1/2] Coarse Cartesian search; BS-S target Agg=$TARGET_AGG"
COARSE_DIR="$SEARCH_ROOT/coarse"
run_stage coarse "$COARSE_A1" "$COARSE_A2" "$COARSE_FILTERS" "$COARSE_DIR"

IFS=, read -r coarse_tag coarse_w1 coarse_w2 coarse_filter \
    coarse_target coarse_margin coarse_required coarse_delta coarse_beats coarse_agg _ \
    <<< "$(best_row "$COARSE_DIR/F2R_SWEEP.csv")"
echo "Coarse best: $coarse_tag Agg=$coarse_agg delta=$coarse_delta"
if [ "$coarse_beats" = "True" ] || [ "$coarse_beats" = "true" ]; then
    echo "Target beaten in coarse search. Report: $COARSE_DIR/F2R_SWEEP.md"
    exit 0
fi

FINE_A1="$(awk -v x="$coarse_w1" 'BEGIN {printf "%.3f %.3f %.3f", x-0.1, x, x+0.1}')"
FINE_A2="$(awk -v x="$coarse_w2" 'BEGIN {printf "%.3f %.3f %.3f", x-0.1, x, x+0.1}')"
FINE_FILTERS="$(awk -v x="$coarse_filter" 'BEGIN {printf "%.6g %.6g %.6g", x/2, x, x*2}')"

echo "[stage 2/2] Refining around w1=$coarse_w1 w2=$coarse_w2 filter=$coarse_filter"
FINE_DIR="$SEARCH_ROOT/fine"
run_stage fine "$FINE_A1" "$FINE_A2" "$FINE_FILTERS" "$FINE_DIR"

IFS=, read -r fine_tag fine_w1 fine_w2 fine_filter \
    fine_target fine_margin fine_required fine_delta fine_beats fine_agg _ \
    <<< "$(best_row "$FINE_DIR/F2R_SWEEP.csv")"
echo "Fine best: $fine_tag Agg=$fine_agg delta=$fine_delta"
if [ "$fine_beats" = "True" ] || [ "$fine_beats" = "true" ]; then
    echo "Target beaten. Report: $FINE_DIR/F2R_SWEEP.md"
else
    echo "Inference sweep completed below target. Keep the Pareto frontier and move to the training-hyperparameter sweep (views/layers/LoRA rank/learning rate)."
    echo "Report: $FINE_DIR/F2R_SWEEP.md"
fi
