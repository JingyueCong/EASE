#!/usr/bin/env bash
# Sequential, resumable method ladder:
#   F2R -> F2R+Alignment -> F2R+Gate -> F2R+Alignment+Gate
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
if [ "${LOAD_DOTENV:-1}" = "1" ] && [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODE="${MODE:-full}"
SPLIT="${SPLIT:-forget05}"
GPU="${GPU:-0}"
VIEWS="${VIEWS:-2}"
LADDER_NAME="${LADDER_NAME:-alignment_gate}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
WEIGHT_A1="${WEIGHT_A1:--1.2}"
WEIGHT_A2="${WEIGHT_A2:-0.4}"
TOP_FILTER="${TOP_FILTER:-0.0025}"
TARGET_AGG="${TARGET_AGG:-0.58}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"
EVAL_BS="${EVAL_BS:-4}"
CALIBRATION_BS="${CALIBRATION_BS:-2}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-14000}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-true}"

CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/f2r/${SPLIT}_${MODE}.jsonl}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}_train_stage1_${SPLIT}/lr1e3_e5}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_calibration/${SPLIT}_${LADDER_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${LADDER_NAME}}"
RUNNER="${EASE_ROOT}/scripts/run_f2r_tofu.sh"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"

latest_checkpoint() {
    find "$1" -name 'checkpoint-*' -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

if [ -z "${A1_CKPT:-}" ]; then
    A1_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a1" || true)"
fi
if [ -z "${A2_CKPT:-}" ]; then
    A2_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a2" || true)"
fi
if [ "$DRY_RUN" != "true" ]; then
    for path in "$CF_PATH" "$A1_CKPT" "$A2_CKPT"; do
        if [ ! -e "$path" ]; then
            echo "Missing required input: $path" >&2
            exit 1
        fi
    done
    if [ ! -x "$EVAL_PY" ]; then
        echo "Missing evaluation Python: $EVAL_PY" >&2
        exit 1
    fi
fi

mkdir -p "$CALIBRATION_DIR" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report" > "$MANIFEST"
ALIGNMENT_ARTIFACT="$CALIBRATION_DIR/alignment.npz"
GATE_ARTIFACT="$CALIBRATION_DIR/gate.npz"
ALIGNMENT_GATE_ARTIFACT="$CALIBRATION_DIR/alignment_gate.npz"

wait_for_gpu() {
    if [ "$WAIT_FOR_GPU" != "true" ] || [ "$DRY_RUN" = "true" ]; then return; fi
    while true; do
        free_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        if [ "$free_mib" -ge "$MIN_FREE_GPU_MIB" ]; then
            echo "[$(date '+%H:%M:%S')] GPU $GPU ready: ${free_mib} MiB free"
            return
        fi
        echo "[$(date '+%H:%M:%S')] waiting for GPU $GPU: ${free_mib}/${MIN_FREE_GPU_MIB} MiB free"
        sleep 60
    done
}

report_complete() {
    [ -s "$1" ] && grep -q '"forget_truth_ratio_knowledge"' "$1"
}

run_eval_stage() {
    local tag="$1" alignment="$2" gate="$3" artifact="$4"
    local task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_ladder_${LADDER_NAME}_${tag}"
    local report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report" >> "$MANIFEST"
    if [ "$RESUME" = "true" ] && report_complete "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse stage $tag"
        return
    fi
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] evaluate $tag alignment=$alignment gate=$gate artifact=$artifact"
        return
    fi
    wait_for_gpu
    echo "[$(date '+%H:%M:%S')] start stage $tag"
    MODE="$MODE" SPLIT="$SPLIT" GPU="$GPU" VIEWS="$VIEWS" \
        CF_PATH="$CF_PATH" MODELS_ROOT="$MODELS_ROOT" TASK_NAME="$task" \
        A1_NUM_LAYER=2 A2_NUM_LAYER=2 A1_LORA_R=16 A2_LORA_R=16 \
        A1_LORA_ALPHA=32 A2_LORA_ALPHA=32 \
        A1_TRAIN_LR=1e-3 A2_TRAIN_LR=1e-3 \
        A1_TRAIN_EP=5 A2_TRAIN_EP=5 \
        A1_RETAIN_WEIGHT=5 A2_RETAIN_WEIGHT=5 \
        A1_SEED=42 A2_SEED=42 \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        F2R_VARIANT="$tag" CALIBRATION_PATH="$artifact" \
        ALIGNMENT_ENABLED="$alignment" GATE_ENABLED="$gate" \
        EVAL_BS="$EVAL_BS" EVAL_OVERWRITE=true HF_PREFLIGHT=0 \
        SELECTION_RETAIN_ACCESS=true \
        bash "$RUNNER" > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  stage $tag"
}

train_calibration() {
    local mode="$1" output="$2" alignment_input="${3:-}"
    if [ "$RESUME" = "true" ] && [ -s "$output" ] && [ -s "${output%.npz}.json" ]; then
        echo "[$(date '+%H:%M:%S')] reuse calibration $mode"
        return
    fi
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] train calibration $mode -> $output"
        return
    fi
    wait_for_gpu
    extra=()
    if [ -n "$alignment_input" ]; then extra=(--alignment-input "$alignment_input"); fi
    echo "[$(date '+%H:%M:%S')] start calibration $mode"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$EVAL_PY" "$EASE_ROOT/scripts/train_f2r_calibration.py" \
        --mode "$mode" \
        --counterfactual-path "$CF_PATH" \
        --base-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --tokenizer open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --a1-path "$A1_CKPT" --a2-path "$A2_CKPT" \
        --weight-a1 "$WEIGHT_A1" --weight-a2 "$WEIGHT_A2" \
        --top-filter "$TOP_FILTER" --batch-size "$CALIBRATION_BS" \
        --output "$output" "${extra[@]}" \
        > "$RESULTS_DIR/logs/calibration_${mode}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  calibration $mode"
}

echo "============================================================"
echo "F2R residual-alignment / learned-gate ladder"
echo "  split/mode       : $SPLIT / $MODE"
echo "  GPU              : $GPU (minimum free ${MIN_FREE_GPU_MIB} MiB)"
echo "  operating point  : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  frozen assistants: $MODELS_ROOT"
echo "  frozen controls  : $CF_PATH"
echo "  results          : $RESULTS_DIR"
echo "  retain access    : calibration=false, diagnostic selection=true"
echo "============================================================"

# The order below is intentional and is the paper's method ladder.
run_eval_stage "F2R" false false null
train_calibration alignment "$ALIGNMENT_ARTIFACT"
run_eval_stage "F2R_Alignment" true false "$ALIGNMENT_ARTIFACT"
train_calibration gate "$GATE_ARTIFACT"
run_eval_stage "F2R_Gate" false true "$GATE_ARTIFACT"
train_calibration alignment-gate "$ALIGNMENT_GATE_ARTIFACT" "$ALIGNMENT_ARTIFACT"
run_eval_stage "F2R_AlignmentGate" true true "$ALIGNMENT_GATE_ARTIFACT"

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run complete: $MANIFEST"
    exit 0
fi

"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin "$TARGET_MARGIN" \
    --sweep-kind method-ladder
echo "Method ladder complete: $RESULTS_DIR/F2R_SWEEP.md"
