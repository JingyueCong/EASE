#!/usr/bin/env bash
# Four-GPU, resumable sweep over F2R assistant training configurations.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-full}"
SPLIT="${SPLIT:-forget05}"
GPUS="${GPUS:-${GPU:-0}}"
VIEWS="${VIEWS:-2}"
SWEEP_NAME="${SWEEP_NAME:-training_$(date +%Y%m%d_%H%M%S)}"
TASK_PREFIX="${TASK_PREFIX:-tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_train_${SWEEP_NAME}}"
CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/f2r/${SPLIT}_${MODE}.jsonl}"
MODELS_SWEEP_ROOT="${MODELS_SWEEP_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}_${SWEEP_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
RUNNER="${EASE_ROOT}/scripts/run_f2r_tofu.sh"
RESUME="${RESUME:-true}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-false}"
A1_DATA_MODE="${A1_DATA_MODE:-f2r_a1}"
A2_DATA_MODE="${A2_DATA_MODE:-f2r_a2}"
F2R_VARIANT="${F2R_VARIANT:-F2R}"

# Fixed inference operating point from the completed forget05 search. Override
# for another split or after a newer development-only inference sweep.
WEIGHT_A1="${WEIGHT_A1:--0.7}"
WEIGHT_A2="${WEIGHT_A2:-0.2}"
TOP_FILTER="${TOP_FILTER:-0.0025}"

# Format per whitespace-separated entry:
# tag:a1_layers:a2_layers:a1_rank:a2_rank:a1_lr:a2_lr:a1_epochs:a2_epochs:a1_uniform_weight:a2_uniform_weight
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
lr1e4_e5:2:2:16:16:1e-4:1e-4:5:5:5:5 \
lr3e4_e5:2:2:16:16:3e-4:3e-4:5:5:5:5 \
lr5e4_e5:2:2:16:16:5e-4:5e-4:5:5:5:5 \
lr1e3_e5:2:2:16:16:1e-3:1e-3:5:5:5:5 \
lr3e4_e3:2:2:16:16:3e-4:3e-4:3:3:5:5 \
layer1_rank8:1:1:8:8:3e-4:3e-4:5:5:5:5 \
layer1_rank16:1:1:16:16:3e-4:3e-4:5:5:5:5 \
layer2_rank8:2:2:8:8:3e-4:3e-4:5:5:5:5 \
uniform10:2:2:16:16:3e-4:3e-4:5:5:10:10 \
uniform20:2:2:16:16:3e-4:3e-4:5:5:20:20 \
asym_lr:2:2:16:16:5e-4:1e-4:5:5:5:10 \
asym_small_a2:2:1:16:8:5e-4:1e-4:5:3:5:20}"

case "$SPLIT" in
    forget01) DEFAULT_TARGET_AGG="0.57" ;;
    forget05) DEFAULT_TARGET_AGG="0.58" ;;
    forget10) DEFAULT_TARGET_AGG="0.61" ;;
    *) echo "Unsupported SPLIT=$SPLIT" >&2; exit 1 ;;
esac
TARGET_AGG="${TARGET_AGG:-$DEFAULT_TARGET_AGG}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"

if [ ! -s "$CF_PATH" ]; then
    echo "Missing frozen counterfactual data: $CF_PATH" >&2
    echo "Generate it once before launching parallel training; concurrent API generation is disabled." >&2
    exit 1
fi

read -r -a GPU_LIST <<< "$GPUS"
read -r -a CONFIG_LIST <<< "$TRAIN_CONFIGS"
if [ "${#GPU_LIST[@]}" -eq 0 ] || [ "${#CONFIG_LIST[@]}" -eq 0 ]; then
    echo "GPUS and TRAIN_CONFIGS must be non-empty." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR/logs" "$MODELS_SWEEP_ROOT"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,views,a1_num_layer,a2_num_layer,a1_lora_r,a2_lora_r,a1_lora_alpha,a2_lora_alpha,a1_train_lr,a2_train_lr,a1_train_ep,a2_train_ep,a1_retain_weight,a2_retain_weight,a1_seed,a2_seed,a1_data_mode,a2_data_mode,variant,models_root" > "$MANIFEST"

echo "============================================================"
echo "F2R assistant-training sweep"
echo "  split/mode      : $SPLIT / $MODE"
echo "  GPUs            : $GPUS"
echo "  configurations  : ${#CONFIG_LIST[@]}"
echo "  frozen CF       : $CF_PATH (declared views=$VIEWS)"
echo "  assistant data  : A1=$A1_DATA_MODE / A2=$A2_DATA_MODE"
echo "  inference point : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  model roots     : $MODELS_SWEEP_ROOT"
echo "  results         : $RESULTS_DIR"
echo "  required Agg    : $TARGET_AGG + $TARGET_MARGIN"
echo "============================================================"

run_one() {
    local gpu="$1" tag="$2" a1_layers="$3" a2_layers="$4"
    local a1_rank="$5" a2_rank="$6" a1_lr="$7" a2_lr="$8"
    local a1_epochs="$9" a2_epochs="${10}" a1_uniform="${11}" a2_uniform="${12}"
    local models_root="${MODELS_SWEEP_ROOT}/${tag}"
    local task_name="${TASK_PREFIX}_${tag}"
    local report="${EASE_ROOT}/open-unlearning/saves/eval/${task_name}/F2R_REPORT.json"
    local a1_alpha=$((2 * a1_rank))
    local a2_alpha=$((2 * a2_rank))

    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task_name,$report,$VIEWS,$a1_layers,$a2_layers,$a1_rank,$a2_rank,$a1_alpha,$a2_alpha,$a1_lr,$a2_lr,$a1_epochs,$a2_epochs,$a1_uniform,$a2_uniform,$SEED,$SEED,$A1_DATA_MODE,$A2_DATA_MODE,$F2R_VARIANT,$models_root" >> "$MANIFEST"

    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $tag GPU=$gpu A1(layers=$a1_layers,r=$a1_rank,lr=$a1_lr,ep=$a1_epochs,u=$a1_uniform) A2(layers=$a2_layers,r=$a2_rank,lr=$a2_lr,ep=$a2_epochs,u=$a2_uniform)"
        return
    fi

    if [ "$RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag (complete report)"
        return
    fi

    echo "[$(date '+%H:%M:%S')] start $tag on GPU $gpu"
    MODE="$MODE" SPLIT="$SPLIT" GPU="$gpu" VIEWS="$VIEWS" \
        CF_PATH="$CF_PATH" MODELS_ROOT="$models_root" TASK_NAME="$task_name" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}" \
        A1_NUM_LAYER="$a1_layers" A2_NUM_LAYER="$a2_layers" \
        A1_LORA_R="$a1_rank" A2_LORA_R="$a2_rank" \
        A1_LORA_ALPHA="$a1_alpha" A2_LORA_ALPHA="$a2_alpha" \
        A1_TRAIN_LR="$a1_lr" A2_TRAIN_LR="$a2_lr" \
        A1_TRAIN_EP="$a1_epochs" A2_TRAIN_EP="$a2_epochs" \
        A1_RETAIN_WEIGHT="$a1_uniform" A2_RETAIN_WEIGHT="$a2_uniform" \
        A1_SEED="$SEED" A2_SEED="$SEED" \
        A1_DATA_MODE="$A1_DATA_MODE" A2_DATA_MODE="$A2_DATA_MODE" \
        F2R_VARIANT="$F2R_VARIANT" \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        HF_PREFLIGHT=0 EVAL_OVERWRITE=true SELECTION_RETAIN_ACCESS=true \
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
for config in "${CONFIG_LIST[@]}"; do
    IFS=: read -r tag a1_layers a2_layers a1_rank a2_rank a1_lr a2_lr \
        a1_epochs a2_epochs a1_uniform a2_uniform <<< "$config"
    if [ -z "${a2_uniform:-}" ]; then
        echo "Invalid TRAIN_CONFIGS entry: $config" >&2
        exit 1
    fi
    gpu="${GPU_LIST[$((INDEX % ${#GPU_LIST[@]}))]}"
    run_one "$gpu" "$tag" "$a1_layers" "$a2_layers" "$a1_rank" "$a2_rank" \
        "$a1_lr" "$a2_lr" "$a1_epochs" "$a2_epochs" "$a1_uniform" "$a2_uniform" &
    pids+=("$!")
    INDEX=$((INDEX + 1))
    if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
        wait_batch
    fi
done
if [ "${#pids[@]}" -gt 0 ]; then wait_batch; fi

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run complete. Planned configurations: $MANIFEST"
    exit 0
fi

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin "$TARGET_MARGIN" \
    --sweep-kind training \
    || FAILURES=$((FAILURES + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    best = next(csv.DictReader(handle), None)
if best:
    print(
        "Training best: "
        f"{best['tag']} Agg={best['aggregate_score']} "
        f"Mem={best['memorization_score']} Util={best['retain_utility_score']} "
        f"delta={best['delta_to_target']} beats_target={best['beats_target']}"
    )
PY
fi

echo "Training sweep table: $RESULTS_DIR/F2R_SWEEP.md"
echo "Training configurations: $MANIFEST"
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES training/evaluation/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
