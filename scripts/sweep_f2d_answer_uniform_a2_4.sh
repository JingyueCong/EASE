#!/usr/bin/env bash
# Isolate whether FullAnswer's ceiling comes from generated controls or from
# the legacy sequence-wide uniform objective. Freeze the strongest causal A1
# and train only four answer-masked A2 variants.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_FULL_DATA="${FULLANSWER_DATA_PATH:-}"
LAUNCH_V512_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_A1_ROOT="${F2D_V512_STRONG_A1_ROOT:-}"

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
FULL_DATA="$(absolute_from_root "${LAUNCH_FULL_DATA:-ULD/data/ciru/forget05_ciru200_seed42_full_authorblock_v1.jsonl}")"
V512_DATA="$(absolute_from_root "${LAUNCH_V512_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
A1_TAG="${F2D_V512_STRONG_A1_TAG:-v512_a1s96_u1p5}"
A1_STEP="${F2D_V512_STRONG_A1_STEP:-96}"
A1_ROOT="$(absolute_from_root "${LAUNCH_A1_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_a1strength4_train_seed42/${A1_TAG}}")"
A1_CHECKPOINT="$(checkpoint_exact "$A1_ROOT" a1 "$A1_STEP")"
SWEEP_NAME="${SWEEP_NAME:-f2d_answer_uniform_a2_4_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_1b_forget05_${SWEEP_NAME}"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"
LOSS_CONFIG="$EASE_ROOT/ULD/configs/unlearn_loss/remember+answer_uniform.yaml"

for required in "$FULL_DATA" "$V512_DATA" "$LOSS_CONFIG"; do
    if [ ! -s "$required" ]; then
        echo "Missing answer-uniform prerequisite: $required" >&2
        exit 1
    fi
done
if [ -z "$A1_CHECKPOINT" ] || [ ! -d "$A1_CHECKPOINT" ]; then
    echo "Missing frozen V5.12 strong A1 checkpoint-$A1_STEP under $A1_ROOT" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "Missing evaluation Python: $PY" >&2
    exit 1
fi

"$PY" - "$FULL_DATA" "$V512_DATA" <<'PY'
import json
import sys

full_path, v512_path = sys.argv[1:]

def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

errors = []
for name, path in (("FullAnswer", full_path), ("V5.12", v512_path)):
    rows = load(path)
    ids = [row.get("source_id") for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200:
        errors.append(f"{name} must contain 200 unique rows")
    for row in rows:
        cells = row.get("cells", {})
        if set(cells) != {"C11", "C01", "C10", "C00"}:
            errors.append(f"{name} {row.get('source_id')}: incomplete cells")
            break
        if cells["C11"].get("question") != row.get("source_question"):
            errors.append(f"{name} {row.get('source_id')}: mutable C11 question")
            break
        if cells["C11"].get("answer") != row.get("source_answer"):
            errors.append(f"{name} {row.get('source_id')}: mutable C11 answer")
            break
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in load(v512_path)):
    errors.append("V5.12 file contains a non-pairbudget row")
if errors:
    raise SystemExit("Answer-uniform preflight failed: " + "; ".join(errors))
print("Answer-uniform data preflight OK: FullAnswer=200 V5.12=200")
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "The four-cell A2 experiment needs four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

CONFIGS=(
    "answeru_full_s60:full:60"
    "answeru_full_s72:full:72"
    "answeru_v512_s60:v512:60"
    "answeru_v512_s72:v512:72"
)

mkdir -p "$MODELS_ROOT" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,views,a1_num_layer,a2_num_layer,a1_lora_r,a2_lora_r,a1_lora_alpha,a2_lora_alpha,a1_train_lr,a2_train_lr,a1_train_ep,a2_train_ep,a1_train_steps,a2_train_steps,a1_retain_weight,a2_retain_weight,a1_seed,a2_seed,a1_data_mode,a2_data_mode,variant,models_root,data_source,training_loss" > "$MANIFEST"

cat <<EOF
============================================================
Answer-masked A2 source/steps experiment
  frozen A1        : $A1_CHECKPOINT
  A1 provenance    : V5.12 strong A1, steps=$A1_STEP, uniform=1.5
  FullAnswer data  : $FULL_DATA
  V5.12 data       : $V512_DATA
  A2 factors       : source={FullAnswer,V5.12} x steps={60,72}
  A2 fixed         : answer-only uniform, lr=1e-3, LoRA=2/r16
  inference        : -2.0 / 1.7 / 0.0004
  composition      : reference_delta
  evaluations      : 4
  GPUs             : $LAUNCH_GPUS
  resume           : $LAUNCH_RESUME
============================================================
EOF

run_one() {
    local gpu="$1" tag="$2" source="$3" steps="$4"
    local data
    case "$source" in
        full) data="$FULL_DATA" ;;
        v512) data="$V512_DATA" ;;
        *) echo "Unknown A2 data source: $source" >&2; return 1 ;;
    esac
    local model_root="$MODELS_ROOT/$tag"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_ANSWERU_A2_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/$task/F2R_REPORT.json"
    echo "$tag,-2.0,1.7,0.0004,$task,$report,1,2,2,16,16,32,32,5e-4,1e-3,1,1,$A1_STEP,$steps,1.5,1.0,42,42,f2d_did_a1,f2d_did_a2,F2D-AnswerMaskedUniform-A2,$model_root,$source,remember+answer_uniform" >> "$MANIFEST"

    if [ "$LAUNCH_DRY_RUN" = "true" ]; then
        echo "[dry-run] $tag GPU=$gpu source=$source A2_steps=$steps"
        return
    fi
    if [ "$LAUNCH_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "answeru_reuse tag=$tag source=$source steps=$steps"
        return
    fi

    echo "answeru_start tag=$tag source=$source steps=$steps GPU=$gpu"
    env \
        ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        VIEWS=1 CF_PATH="$data" MODELS_ROOT="$model_root" TASK_NAME="$task" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}" \
        TRAIN_LOSS_CONFIG=remember+answer_uniform \
        A1_CHECKPOINT_OVERRIDE="$A1_CHECKPOINT" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=2 A1_LORA_R=16 A1_LORA_ALPHA=32 \
        A1_TRAIN_LR=5e-4 A1_TRAIN_EP=1 A1_TRAIN_STEPS="$A1_STEP" \
        A1_RETAIN_WEIGHT=1.5 A1_SEED=42 \
        A2_NUM_LAYER=2 A2_LORA_R=16 A2_LORA_ALPHA=32 \
        A2_TRAIN_LR=1e-3 A2_TRAIN_EP=1 A2_TRAIN_STEPS="$steps" \
        A2_RETAIN_WEIGHT=1.0 A2_SEED=42 \
        WEIGHT_A1=-2.0 WEIGHT_A2=1.7 TOP_FILTER=0.0004 \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        F2R_VARIANT=F2D-AnswerMaskedUniform-A2-Isolation \
        EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        SELECTION_RETAIN_ACCESS=true HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "answeru_done tag=$tag source=$source steps=$steps GPU=$gpu"
}

pids=()
failures=0
index=0
for config in "${CONFIGS[@]}"; do
    IFS=: read -r tag source steps <<< "$config"
    run_one "${GPU_LIST[$index]}" "$tag" "$source" "$steps" &
    pids+=("$!")
    index=$((index + 1))
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failures=$((failures + 1))
    fi
done

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run complete: four A2-only configurations validated."
    exit 0
fi

"$PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind training \
    || failures=$((failures + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

rows = [row for row in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if row.get("aggregate_score")]
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
if rows:
    best = rows[0]
    agg = float(best["aggregate_score"])
    print("===== Answer-masked A2 current best =====")
    print("config =", best["tag"])
    print("Agg    =", f"{agg:.6f}")
    print("Mem    =", best["memorization_score"])
    print("Util   =", best["retain_utility_score"])
    print("relative to legacy hybrid 0.551721:", f"{agg - 0.551721:+.6f}")
    print("distance to target 0.580000:", f"{agg - 0.580000:+.6f}")
PY
fi

echo "Answer-masked A2 table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$failures" -gt 0 ]; then
    echo "$failures answer-masked A2 job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
