#!/usr/bin/env bash
# Evaluation-only F2R inference sweep. Reuses frozen counterfactuals and A1/A2.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-full}"
SPLIT="${SPLIT:-forget05}"
GPUS="${GPUS:-${GPU:-0}}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:--0.4:0.4 -0.6:0.6 -0.8:0.8 -1.0:1.0}"
TOP_FILTERS="${TOP_FILTERS:-0.01 0.1}"
SWEEP_NAME="${SWEEP_NAME:-weights_$(date +%Y%m%d_%H%M%S)}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}}"
CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/f2r/${SPLIT}_${MODE}.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
RUNNER="${EASE_ROOT}/scripts/run_f2r_tofu.sh"

latest_checkpoint() {
    find "$1" -name 'checkpoint-*' -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

if [ ! -s "$CF_PATH" ]; then
    echo "Missing frozen counterfactual data: $CF_PATH" >&2
    echo "Run one MODE=$MODE SPLIT=$SPLIT full experiment before sweeping." >&2
    exit 1
fi
A1_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a1")"
A2_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a2")"
if [ -z "$A1_CKPT" ] || [ -z "$A2_CKPT" ]; then
    echo "Missing frozen A1/A2 checkpoints under $MODELS_ROOT" >&2
    echo "Run one MODE=$MODE SPLIT=$SPLIT full experiment before sweeping." >&2
    exit 1
fi

read -r -a GPU_LIST <<< "$GPUS"
read -r -a PAIR_LIST <<< "$WEIGHT_PAIRS"
read -r -a FILTER_LIST <<< "$TOP_FILTERS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU id." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report" > "$MANIFEST"

echo "============================================================"
echo "F2R inference sweep (frozen assistants; no retraining)"
echo "  split/mode    : $SPLIT / $MODE"
echo "  GPUs          : $GPUS"
echo "  weight pairs  : $WEIGHT_PAIRS"
echo "  top filters   : $TOP_FILTERS"
echo "  A1            : $A1_CKPT"
echo "  A2            : $A2_CKPT"
echo "  results       : $RESULTS_DIR"
echo "  protocol      : selection_retain_access=true"
echo "============================================================"

run_one() {
    local gpu="$1" tag="$2" w1="$3" w2="$4" filter="$5" task_name="$6"
    echo "[$(date '+%H:%M:%S')] start $tag on GPU $gpu"
    MODE="$MODE" SPLIT="$SPLIT" GPU="$gpu" \
        CF_PATH="$CF_PATH" MODELS_ROOT="$MODELS_ROOT" \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER="$filter" \
        TASK_NAME="$task_name" HF_PREFLIGHT=0 EVAL_OVERWRITE=true \
        SELECTION_RETAIN_ACCESS=true \
        bash "$RUNNER" > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  $tag on GPU $gpu"
}

pids=()
FAILURES=0
wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            FAILURES=$((FAILURES + 1))
        fi
    done
    pids=()
}

INDEX=0
for pair in "${PAIR_LIST[@]}"; do
    if [[ "$pair" != *:* ]]; then
        echo "Invalid WEIGHT_PAIRS entry '$pair'; expected w1:w2" >&2
        exit 1
    fi
    w1="${pair%%:*}"
    w2="${pair#*:}"
    for filter in "${FILTER_LIST[@]}"; do
        gpu="${GPU_LIST[$((INDEX % ${#GPU_LIST[@]}))]}"
        tag="w1${w1//-/m}_w2${w2//-/m}_f${filter//./p}"
        tag="${tag//./p}"
        task_name="tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_sweep_${SWEEP_NAME}_${tag}"
        report="${EASE_ROOT}/open-unlearning/saves/eval/${task_name}/F2R_REPORT.json"
        echo "$tag,$w1,$w2,$filter,$task_name,$report" >> "$MANIFEST"
        run_one "$gpu" "$tag" "$w1" "$w2" "$filter" "$task_name" &
        pids+=("$!")
        INDEX=$((INDEX + 1))
        if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
            wait_batch
        fi
    done
done
if [ "${#pids[@]}" -gt 0 ]; then
    wait_batch
fi

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" || FAILURES=$((FAILURES + 1))

echo "Sweep table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES sweep process/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
