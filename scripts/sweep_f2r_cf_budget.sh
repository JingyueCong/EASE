#!/usr/bin/env bash
# Four-GPU, resumable sweep over matched-counterfactual supervision budgets.
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
GPUS="${GPUS:-0 1 2 3}"
BUDGETS="${BUDGETS:-40 80 200 400}"
BUDGET_SEED="${BUDGET_SEED:-42}"
SWEEP_NAME="${SWEEP_NAME:-cf_budget_seed${BUDGET_SEED}}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"

SOURCE_CF_PATH="${SOURCE_CF_PATH:-${EASE_ROOT}/ULD/data/f2r/${SPLIT}_${MODE}.jsonl}"
SUBSET_DIR="${SUBSET_DIR:-${EASE_ROOT}/ULD/data/f2r/budgets/${SPLIT}_seed${BUDGET_SEED}}"
BUDGET_MANIFEST="${SUBSET_DIR}/budget_manifest.csv"
MODELS_SWEEP_ROOT="${MODELS_SWEEP_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}_${SWEEP_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
RUNNER="${EASE_ROOT}/scripts/run_f2r_tofu.sh"

# Keep training and inference fixed so the first-stage table isolates CF budget.
A1_NUM_LAYER="${A1_NUM_LAYER:-2}"
A2_NUM_LAYER="${A2_NUM_LAYER:-2}"
A1_LORA_R="${A1_LORA_R:-16}"
A2_LORA_R="${A2_LORA_R:-16}"
A1_TRAIN_LR="${A1_TRAIN_LR:-1e-3}"
A2_TRAIN_LR="${A2_TRAIN_LR:-1e-3}"
A1_TRAIN_EP="${A1_TRAIN_EP:-5}"
A2_TRAIN_EP="${A2_TRAIN_EP:-5}"
A1_RETAIN_WEIGHT="${A1_RETAIN_WEIGHT:-5}"
A2_RETAIN_WEIGHT="${A2_RETAIN_WEIGHT:-5}"
TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_GA="${TRAIN_GA:-4}"
WEIGHT_A1="${WEIGHT_A1:--1.2}"
WEIGHT_A2="${WEIGHT_A2:-0.4}"
TOP_FILTER="${TOP_FILTER:-0.0025}"
EVAL_BS="${EVAL_BS:-4}"
TARGET_AGG="${TARGET_AGG:-0.58}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"

if [ ! -s "$SOURCE_CF_PATH" ]; then
    echo "Missing complete counterfactual file: $SOURCE_CF_PATH" >&2
    exit 1
fi
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing evaluation Python: $EVAL_PY" >&2
    exit 1
fi

mkdir -p "$SUBSET_DIR" "$MODELS_SWEEP_ROOT" "$RESULTS_DIR/logs"
read -r -a BUDGET_LIST <<< "$BUDGETS"
"$EVAL_PY" "$EASE_ROOT/scripts/prepare_f2r_budget.py" \
    --input "$SOURCE_CF_PATH" --output-dir "$SUBSET_DIR" \
    --budgets "${BUDGET_LIST[@]}" --seed "$BUDGET_SEED" \
    --manifest "$BUDGET_MANIFEST"

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU index" >&2
    exit 1
fi

MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,views,counterfactual_budget,counterfactual_sources,counterfactual_path,counterfactual_seed,a1_num_layer,a2_num_layer,a1_lora_r,a2_lora_r,a1_lora_alpha,a2_lora_alpha,a1_train_lr,a2_train_lr,a1_train_ep,a2_train_ep,a1_retain_weight,a2_retain_weight,a1_seed,a2_seed,models_root" > "$MANIFEST"

echo "============================================================"
echo "F2R counterfactual-budget sweep"
echo "  split/mode       : $SPLIT / $MODE"
echo "  GPUs             : $GPUS"
echo "  budgets          : $BUDGETS records (seed=$BUDGET_SEED)"
echo "  source CF        : $SOURCE_CF_PATH"
echo "  operating point  : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  results          : $RESULTS_DIR"
echo "  protocol         : training retain=false; selection retain=true"
echo "============================================================"

run_one() {
    local gpu="$1" budget="$2" records="$3" sources="$4" max_views="$5"
    local cf_path="$6"
    local tag="cf_budget${budget}"
    local models_root="${MODELS_SWEEP_ROOT}/${tag}"
    local task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_${SWEEP_NAME}_${tag}"
    local report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    local a1_alpha=$((2 * A1_LORA_R))
    local a2_alpha=$((2 * A2_LORA_R))

    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report,$max_views,$records,$sources,$cf_path,$BUDGET_SEED,$A1_NUM_LAYER,$A2_NUM_LAYER,$A1_LORA_R,$A2_LORA_R,$a1_alpha,$a2_alpha,$A1_TRAIN_LR,$A2_TRAIN_LR,$A1_TRAIN_EP,$A2_TRAIN_EP,$A1_RETAIN_WEIGHT,$A2_RETAIN_WEIGHT,$BUDGET_SEED,$BUDGET_SEED,$models_root" >> "$MANIFEST"

    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $tag GPU=$gpu records=$records sources=$sources views<=$max_views"
        return
    fi
    if [ "$RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag"
        return
    fi

    echo "[$(date '+%H:%M:%S')] start $tag on GPU $gpu"
    MODE="$MODE" SPLIT="$SPLIT" GPU="$gpu" VIEWS="$max_views" \
        CF_PATH="$cf_path" MODELS_ROOT="$models_root" TASK_NAME="$task" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}" \
        A1_NUM_LAYER="$A1_NUM_LAYER" A2_NUM_LAYER="$A2_NUM_LAYER" \
        A1_LORA_R="$A1_LORA_R" A2_LORA_R="$A2_LORA_R" \
        A1_LORA_ALPHA="$a1_alpha" A2_LORA_ALPHA="$a2_alpha" \
        A1_TRAIN_LR="$A1_TRAIN_LR" A2_TRAIN_LR="$A2_TRAIN_LR" \
        A1_TRAIN_EP="$A1_TRAIN_EP" A2_TRAIN_EP="$A2_TRAIN_EP" \
        A1_RETAIN_WEIGHT="$A1_RETAIN_WEIGHT" A2_RETAIN_WEIGHT="$A2_RETAIN_WEIGHT" \
        A1_TRAIN_BS="$TRAIN_BS" A2_TRAIN_BS="$TRAIN_BS" \
        A1_TRAIN_GA="$TRAIN_GA" A2_TRAIN_GA="$TRAIN_GA" \
        A1_SEED="$BUDGET_SEED" A2_SEED="$BUDGET_SEED" \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        EVAL_BS="$EVAL_BS" HF_PREFLIGHT=0 EVAL_OVERWRITE=true \
        SELECTION_RETAIN_ACCESS=true \
        bash "$RUNNER" > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  $tag on GPU $gpu"
}

pids=()
FAILURES=0
INDEX=0
while IFS=, read -r budget records sources max_views seed input_hash output_hash cf_path; do
    [ "$budget" = "budget" ] && continue
    gpu="${GPU_LIST[$((INDEX % ${#GPU_LIST[@]}))]}"
    run_one "$gpu" "$budget" "$records" "$sources" "$max_views" "$cf_path" &
    pids+=("$!")
    INDEX=$((INDEX + 1))
    if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then FAILURES=$((FAILURES + 1)); fi
        done
        pids=()
    fi
done < "$BUDGET_MANIFEST"
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then FAILURES=$((FAILURES + 1)); fi
done

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run complete: $MANIFEST"
    exit 0
fi

"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin "$TARGET_MARGIN" \
    --sweep-kind counterfactual-budget \
    || FAILURES=$((FAILURES + 1))

echo "Sweep table: $RESULTS_DIR/F2R_SWEEP.md"
echo "Budget manifest: $BUDGET_MANIFEST"
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES budget job/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
