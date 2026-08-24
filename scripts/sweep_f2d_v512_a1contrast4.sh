#!/usr/bin/env bash
# Four-cell A1 causal-contrast pilot. Trains only A1; FullAnswer A2 is frozen.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_AUDIT="${F2D_V512_AUDIT_SUMMARY:-}"
LAUNCH_FULL_MANIFEST="${FULLANSWER_MANIFEST:-}"
LAUNCH_SWEEP_NAME="${SWEEP_NAME:-}"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

absolute_from_root() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$EASE_ROOT" "$1" ;;
    esac
}

checkpoint_exact() {
    local root="$1" role="$2" step="$3"
    find "$root/$role" -type d -name "checkpoint-${step}" 2>/dev/null \
        | sort | tail -n 1
}

PY="${PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
AUDIT="$(absolute_from_root "${LAUNCH_AUDIT:-audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_FULL_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_STEP="${FULLANSWER_STEP:-72}"
SWEEP_NAME="${LAUNCH_SWEEP_NAME:-f2d_v512_a1contrast4_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_1b_forget05_${SWEEP_NAME}"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"

for required in "$DATA" "$AUDIT" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing A1-contrast prerequisite: $required" >&2
        exit 1
    fi
done

"$PY" - "$DATA" "$AUDIT" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
audit = json.load(open(sys.argv[2]))
errors = []
if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
    errors.append("data must contain 200 unique rows")
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in rows):
    errors.append("data contains a non-V5.12 row")
if audit.get("units_with_deterministic_errors") != 0:
    errors.append("audit contains deterministic errors")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    errors.append("audit coverage is incomplete")
if errors:
    raise SystemExit("A1-contrast preflight failed: " + "; ".join(errors))
print("A1-contrast preflight OK: rows=200 deterministic_errors=0")
PY

FULL_ROOT="$(awk -F, -v tag="$FULL_TAG" '$1==tag {gsub(/\r/, "", $NF); print $NF}' "$FULL_MANIFEST" | tail -n 1)"
if [ -z "$FULL_ROOT" ]; then
    echo "Could not resolve FullAnswer tag $FULL_TAG" >&2
    exit 1
fi
FULL_ROOT="$(absolute_from_root "$FULL_ROOT")"
FULL_A2="$(checkpoint_exact "$FULL_ROOT" a2 "$FULL_STEP")"
if [ -z "$FULL_A2" ] || [ ! -d "$FULL_A2" ]; then
    echo "Missing FullAnswer A2 checkpoint-$FULL_STEP under $FULL_ROOT" >&2
    exit 1
fi

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "A1-contrast pilot requires four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

CONFIGS=(
    "cw0p1_pk0p01:0.1:0.01"
    "cw0p1_pk0p03:0.1:0.03"
    "cw0p3_pk0p01:0.3:0.01"
    "cw0p3_pk0p03:0.3:0.03"
)

mkdir -p "$MODELS_ROOT" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,contrast_weight,contrast_margin,placebo_kl_weight,models_root" > "$MANIFEST"

cat <<EOF
============================================================
V5.12 paired causal-contrast A1 pilot
  causal data       : $DATA
  A1 objective      : CE(C11) + uniform(C01) + paired margin + placebo KL
  paired negative   : C01 question + immutable C11 answer
  contrast weights  : 0.1 0.3
  placebo KL weights: 0.01 0.03
  contrast margin   : 0.5 nat/token
  A1 fixed          : steps=96 uniform=1.5 lr=5e-4 LoRA=2/r16
  A2 frozen         : $FULL_A2
  inference point   : -2.0 / 1.7 / 0.0004
  composition       : reference_delta
  configurations    : 4
  GPUs              : $LAUNCH_GPUS
============================================================
EOF

run_one() {
    local gpu="$1" tag="$2" contrast_weight="$3" placebo_kl="$4"
    local model_root="$MODELS_ROOT/$tag"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_A1CONTRAST4_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/$task/F2R_REPORT.json"
    echo "$tag,-2.0,1.7,0.0004,$task,$report,$contrast_weight,0.5,$placebo_kl,$model_root" >> "$MANIFEST"

    if [ "$LAUNCH_DRY_RUN" = "true" ]; then
        echo "[dry-run] $tag GPU=$gpu contrast=$contrast_weight placebo_kl=$placebo_kl"
        return
    fi
    if [ "$LAUNCH_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "contrast_reuse tag=$tag"
        return
    fi

    echo "contrast_start tag=$tag GPU=$gpu"
    env \
        ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        VIEWS=1 CF_PATH="$DATA" MODELS_ROOT="$model_root" TASK_NAME="$task" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}" \
        TRAIN_LOSS_CONFIG=factorial_contrastive_a1 \
        CONTRAST_WEIGHT="$contrast_weight" CONTRAST_MARGIN=0.5 \
        PLACEBO_KL_WEIGHT="$placebo_kl" \
        A1_DATA_MODE=f2d_contrast_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=2 A1_LORA_R=16 A1_LORA_ALPHA=32 \
        A1_TRAIN_LR=5e-4 A1_TRAIN_EP=1 A1_TRAIN_STEPS=96 \
        A1_RETAIN_WEIGHT=1.5 A1_TRAIN_BS=5 A1_TRAIN_GA=4 A1_SEED=42 \
        A2_CHECKPOINT_OVERRIDE="$FULL_A2" \
        A2_NUM_LAYER=2 A2_LORA_R=16 A2_LORA_ALPHA=32 \
        A2_TRAIN_LR=1e-3 A2_TRAIN_EP=1 A2_TRAIN_STEPS=72 \
        A2_RETAIN_WEIGHT=1.0 A2_SEED=42 \
        WEIGHT_A1=-2.0 WEIGHT_A2=1.7 TOP_FILTER=0.0004 \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        F2R_VARIANT=F2D-V512-A1PairedContrast-FullAnswerA2-Hybrid \
        EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        SELECTION_RETAIN_ACCESS=true HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "contrast_done tag=$tag GPU=$gpu"
}

pids=()
failures=0
index=0
for config in "${CONFIGS[@]}"; do
    IFS=: read -r tag contrast_weight placebo_kl <<< "$config"
    gpu="${GPU_LIST[$index]}"
    run_one "$gpu" "$tag" "$contrast_weight" "$placebo_kl" &
    pids+=("$!")
    index=$((index + 1))
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failures=$((failures + 1))
    fi
done

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run complete: four causal-contrast configurations validated."
    exit 0
fi

"$PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind training \
    || failures=$((failures + 1))

echo "A1-contrast table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$failures" -gt 0 ]; then
    echo "$failures A1-contrast job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
