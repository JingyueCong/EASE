#!/usr/bin/env bash
# Four-GPU parallel TOFU ladder for unified hierarchical F2D.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_RESUME="${RESUME:-true}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
TRAIN_PY="${TRAIN_PY:-${CONDA_BASE}/envs/${TRAIN_ENV:-ease-f2r-train}/bin/python}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"

SPLIT="${SPLIT:-forget05}"
SEED="${SEED:-42}"
UNITS="${UNITS:-200}"
SOURCE_DATA="${SOURCE_DATA:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_v1.jsonl}"
HIER_DATA="${HIER_DATA:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_hier_v1.jsonl}"
SWEEP_NAME="${SWEEP_NAME:-uf2d_hierarchy_ladder_seed${SEED}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/uf2d_1b_${SPLIT}_${SWEEP_NAME}}"
WEIGHT_A1="${WEIGHT_A1:--1.8}"
WEIGHT_A2="${WEIGHT_A2:-1.8}"
TOP_FILTER="${TOP_FILTER:-0.0004}"

if [ ! -s "$SOURCE_DATA" ]; then
    echo "Missing frozen full-coverage factorial data: $SOURCE_DATA" >&2
    exit 1
fi
if [ ! -s "$HIER_DATA" ]; then
    echo "[1/2] Annotating paired claim/evidence hierarchy"
    "$TRAIN_PY" "$EASE_ROOT/scripts/annotate_f2d_hierarchy.py" \
        --input "$SOURCE_DATA" --output "$HIER_DATA" \
        --expected-units "$UNITS"
else
    echo "[1/2] Reusing hierarchical annotation: $HIER_DATA"
fi

read -r -a GPU_LIST <<< "$REQUESTED_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "This strict parallel ladder requires four GPU ids in GPUS." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR/logs" "$MODELS_ROOT"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,views,a1_num_layer,a2_num_layer,a1_lora_r,a2_lora_r,a1_lora_alpha,a2_lora_alpha,a1_train_lr,a2_train_lr,a1_train_ep,a2_train_ep,a1_train_steps,a2_train_steps,a1_retain_weight,a2_retain_weight,a1_seed,a2_seed,a1_data_mode,a2_data_mode,variant,models_root" > "$MANIFEST"

# tag:data_path:a1_mode:a2_mode:loss:preserve_kl:evidence_weight:variant
CONFIGS=(
  "full_answer:${SOURCE_DATA}:f2d_did_a1:f2d_did_a2:remember+uniform:0.0:0.0:U-F2D-FullAnswer"
  "claim_mask:${HIER_DATA}:uf2d_hier_a1:uf2d_hier_a2:factorial_hierarchical:0.0:0.0:U-F2D-ClaimMask"
  "claim_kl_b0p05:${HIER_DATA}:uf2d_hier_a1:uf2d_hier_a2:factorial_hierarchical:0.05:0.0:U-F2D-ClaimMask-KL"
  "claim_span_kl_b0p05:${HIER_DATA}:uf2d_hier_a1:uf2d_hier_a2:factorial_hierarchical:0.05:1.0:U-F2D-ClaimSpan-KL"
)

echo "============================================================"
echo "U-F2D TOFU strict method ladder"
echo "  split / units   : $SPLIT / $UNITS"
echo "  GPUs            : ${GPU_LIST[*]:0:4}"
echo "  data            : same frozen 2x2 units for all stages"
echo "  training        : A1/A2=72/72 steps, lr=1e-3, LoRA=2x r16"
echo "  inference       : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  stages          : FullAnswer / ClaimMask / +KL / +Span+KL"
echo "  retain access   : train=false, diagnostic selection=true"
echo "============================================================"

run_one() {
    local gpu="$1" config="$2"
    IFS=: read -r tag data_path a1_mode a2_mode loss preserve_kl evidence_weight variant <<< "$config"
    local task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_UF2D_${tag}_seed${SEED}"
    local report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    local model_root="${MODELS_ROOT}/${tag}"
    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report,1,2,2,16,16,32,32,1e-3,1e-3,1,1,72,72,1,1,$SEED,$SEED,$a1_mode,$a2_mode,$variant,$model_root" >> "$MANIFEST"
    if [ "$REQUESTED_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag"
        return
    fi
    echo "[$(date '+%H:%M:%S')] start $tag on GPU $gpu"
    env LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPU="$gpu" VIEWS=1 \
        CF_PATH="$data_path" MODELS_ROOT="$model_root" TASK_NAME="$task" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}" \
        A1_DATA_MODE="$a1_mode" A2_DATA_MODE="$a2_mode" \
        A1_NUM_LAYER=2 A2_NUM_LAYER=2 A1_LORA_R=16 A2_LORA_R=16 \
        A1_LORA_ALPHA=32 A2_LORA_ALPHA=32 \
        A1_TRAIN_LR=1e-3 A2_TRAIN_LR=1e-3 \
        A1_TRAIN_EP=1 A2_TRAIN_EP=1 A1_TRAIN_STEPS=72 A2_TRAIN_STEPS=72 \
        A1_RETAIN_WEIGHT=1 A2_RETAIN_WEIGHT=1 A1_SEED="$SEED" A2_SEED="$SEED" \
        TRAIN_LOSS_CONFIG="$loss" PRESERVE_KL_WEIGHT="$preserve_kl" \
        EVIDENCE_WEIGHT="$evidence_weight" \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        F2R_VARIANT="$variant" ALIGNMENT_ENABLED=false GATE_ENABLED=false \
        CALIBRATION_PATH=null HF_PREFLIGHT=0 EVAL_OVERWRITE=true \
        SELECTION_RETAIN_ACCESS=true EVAL_BS="${EVAL_BS:-4}" \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  $tag on GPU $gpu"
}

echo "[2/2] Running four stages in parallel"
pids=()
for index in "${!CONFIGS[@]}"; do
    run_one "${GPU_LIST[$index]}" "${CONFIGS[$index]}" &
    pids+=("$!")
done

FAILURES=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        FAILURES=$((FAILURES + 1))
    fi
done

"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "${TARGET_AGG:-0.58}" \
    --target-margin "${TARGET_MARGIN:-0.005}" \
    --sweep-kind training || FAILURES=$((FAILURES + 1))

echo "Ladder table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES ladder stage/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
