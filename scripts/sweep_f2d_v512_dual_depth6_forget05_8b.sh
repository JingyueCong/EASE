#!/usr/bin/env bash
# Retain-free Llama-3.1-8B forget05 static Dual Assistant depth-6 pilot.
# Existing depth-4 checkpoints, logs, manifests, and reports are read-only.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-1}"
LAUNCH_TRAIN_BS="${TRAIN_BS:-1}"
LAUNCH_TRAIN_GA="${TRAIN_GA:-16}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
TARGET_AGG="${TARGET_AGG:-0.60}"

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

latest_checkpoint() {
    find "$1" -type d -name 'checkpoint-*' 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

reference_for() {
    local checkpoint="$1"
    (cd "$checkpoint/../fullmodel" 2>/dev/null && pwd)
}

TRAIN_PY="${TRAIN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
V512_DATA="$(absolute_from_root "${F2D_V512_DATA_PATH:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
FULL_DATA="$(absolute_from_root "${FULLANSWER_DATA_PATH:-ULD/data/ciru/forget05_ciru200_seed42_full_authorblock_v1.jsonl}")"
V512_AUDIT="$(absolute_from_root "${F2D_V512_AUDIT_SUMMARY:-audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}")"
MODEL_TAG="${MODEL_TAG:-f2d_v512_dual_depth6_8b_train_seed42}"
SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth6_8b_pilot_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_8b_forget05_${MODEL_TAG}"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"
RETAIN_REFERENCE="$EASE_ROOT/open-unlearning/saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json"
TRAIN_CONFIG_PATH="$EASE_ROOT/ULD/configs/model/llama-3-8b.yaml"
EVAL_CONFIG_PATH="$EASE_ROOT/open-unlearning/configs/model/Llama-3.1-8B-Instruct_DualULD.yaml"

for required in \
    "$V512_DATA" "$FULL_DATA" "$V512_AUDIT" \
    "$TRAIN_CONFIG_PATH" "$EVAL_CONFIG_PATH"; do
    if [ ! -s "$required" ]; then
        echo "Missing immutable depth-6 prerequisite: $required" >&2
        exit 1
    fi
done
for executable in "$TRAIN_PY" "$EVAL_PY"; do
    if [ ! -x "$executable" ]; then
        echo "Missing Python environment: $executable" >&2
        exit 1
    fi
done

"$TRAIN_PY" - "$V512_DATA" "$FULL_DATA" "$V512_AUDIT" <<'PY'
import json
import sys

v512_path, full_path, audit_path = sys.argv[1:]
expected = [f"forget05_perturbed-{index:05d}" for index in range(200)]
for label, path in (("V5.12", v512_path), ("FullAnswer", full_path)):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ids = [row.get("source_id") for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200 or ids != expected:
        raise SystemExit(f"{label}: expected 200 unique, canonically ordered forget05 rows")
    if any(set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"} for row in rows):
        raise SystemExit(f"{label}: incomplete factorial cells")

audit = json.load(open(audit_path, encoding="utf-8"))
if audit.get("records") != 200 or audit.get("unique_source_ids") != 200:
    raise SystemExit("V5.12 audit coverage gate failed")
if audit.get("units_with_deterministic_errors") != 0:
    raise SystemExit("V5.12 deterministic audit gate failed")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    raise SystemExit("V5.12 source-index audit gate failed")
print("Depth-6 training gate OK: rows=200 blocks=10 deterministic_errors=0")
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "Depth-6 pilot requires four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

# 56/48 preserves approximately the layer-update exposure of the best 4-layer
# 84/72 checkpoint pair.  The 64/54 arm tests a slightly wider budget while
# reducing both learning rates to limit depth-induced overfitting.
TRAININGS=(
    "exposure:56:48:5e-4:1e-3"
    "conservative:64:54:4e-4:8e-4"
)
POINTS=(
    "light:-1.2:1.2:0.0004"
    "conservative:-1.4:1.2:0.0004"
    "balanced:-1.6:1.4:0.0004"
    "depth4_best:-2.0:2.0:0.0004"
)

cat <<EOF
============================================================
Forget05 Llama-3.1-8B static Dual Assistant depth-6 pilot
  causal A1 data    : $V512_DATA
  FullAnswer A2 data: $FULL_DATA
  old artifacts     : read-only; no overwrite
  architecture      : A1=6 layers / A2=6 layers / LoRA rank=16
  training budgets  : exposure 56/48; conservative 64/54
  composition       : shared depth-6 reference_delta
  routing/gating    : disabled
  inference points  : 4 per training budget (8 total)
  target Agg        : $TARGET_AGG
  GPUs              : $LAUNCH_GPUS
  train bs/ga       : $LAUNCH_TRAIN_BS / $LAUNCH_TRAIN_GA
  models            : $MODELS_ROOT
  results           : $RESULTS_DIR
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run OK; frozen data, audit, model configs, and isolated depth-6 paths validated."
    exit 0
fi

mkdir -p "$MODELS_ROOT" "$RESULTS_DIR/logs"

train_role() {
    local gpu="$1" training="$2" role="$3"
    local a1_steps="$4" a2_steps="$5" a1_lr="$6" a2_lr="$7"
    local role_root="$MODELS_ROOT/$training/${role}_job/$role"
    local data="$V512_DATA"
    if [ "$role" = "a2" ]; then data="$FULL_DATA"; fi
    if [ -n "$(latest_checkpoint "$role_root")" ]; then
        echo "depth6_train_reuse training=$training role=$role"
        return
    fi
    echo "depth6_train_start training=$training role=$role GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 \
        GPU="$gpu" CF_PATH="$data" \
        MODELS_ROOT="$MODELS_ROOT/$training/${role}_job" \
        TRAIN_RUN_TAG="${MODEL_TAG}_${training}_${role}" \
        TRAIN_ONLY=true TRAIN_ROLE="$role" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=6 A1_LORA_R=16 A1_LORA_ALPHA=32 \
        A1_TRAIN_LR="$a1_lr" A1_TRAIN_EP=1 A1_TRAIN_STEPS="$a1_steps" \
        A1_RETAIN_WEIGHT=1.5 A1_TRAIN_BS="$LAUNCH_TRAIN_BS" A1_TRAIN_GA="$LAUNCH_TRAIN_GA" A1_SEED=42 \
        A2_NUM_LAYER=6 A2_LORA_R=16 A2_LORA_ALPHA=32 \
        A2_TRAIN_LR="$a2_lr" A2_TRAIN_EP=1 A2_TRAIN_STEPS="$a2_steps" \
        A2_RETAIN_WEIGHT=1.0 A2_TRAIN_BS="$LAUNCH_TRAIN_BS" A2_TRAIN_GA="$LAUNCH_TRAIN_GA" A2_SEED=42 \
        TRAIN_MODEL_CONFIG=llama-3-8b \
        EVAL_MODEL_CONFIG=Llama-3.1-8B-Instruct_DualULD \
        HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.1-8B-Instruct \
        HF_MODEL_NAME=Llama-3.1-8B-Instruct \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        F2R_VARIANT="F2D-Forget05-V512-DualDepth6-8B-${training}-${role}" \
        HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/train_${training}_${role}.log" 2>&1
    echo "depth6_train_done training=$training role=$role GPU=$gpu"
}

echo "[1/4] Train two preregistered depth-6 budgets (four role checkpoints)"
pids=()
index=0
for spec in "${TRAININGS[@]}"; do
    IFS=: read -r training a1_steps a2_steps a1_lr a2_lr <<< "$spec"
    for role in a1 a2; do
        train_role "${GPU_LIST[$index]}" "$training" "$role" \
            "$a1_steps" "$a2_steps" "$a1_lr" "$a2_lr" &
        pids+=("$!")
        index=$((index + 1))
    done
done
failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
if [ "$failures" -gt 0 ]; then
    echo "$failures depth-6 training job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi

echo "[2/4] Resolve depth-6 checkpoints and shared references"
declare -A TRAIN_A1 TRAIN_A2 TRAIN_REFERENCE TRAIN_A1_STEPS TRAIN_A2_STEPS
declare -A TRAIN_A1_LR TRAIN_A2_LR
for spec in "${TRAININGS[@]}"; do
    IFS=: read -r training a1_steps a2_steps a1_lr a2_lr <<< "$spec"
    a1="$(latest_checkpoint "$MODELS_ROOT/$training/a1_job/a1")"
    a2="$(latest_checkpoint "$MODELS_ROOT/$training/a2_job/a2")"
    for checkpoint in "$a1" "$a2"; do
        if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
            echo "Missing depth-6 checkpoint: ${checkpoint:-unresolved}" >&2
            exit 1
        fi
    done
    reference="$(reference_for "$a1")"
    if [ -z "$reference" ] || [ ! -d "$reference" ]; then
        echo "Missing shared depth-6 reference for $training" >&2
        exit 1
    fi
    TRAIN_A1["$training"]="$a1"
    TRAIN_A2["$training"]="$a2"
    TRAIN_REFERENCE["$training"]="$reference"
    TRAIN_A1_STEPS["$training"]="$a1_steps"
    TRAIN_A2_STEPS["$training"]="$a2_steps"
    TRAIN_A1_LR["$training"]="$a1_lr"
    TRAIN_A2_LR["$training"]="$a2_lr"
done

echo "[2.5/4] Materialize the frozen retain95 reference before parallel evaluation"
if [ ! -s "$RETAIN_REFERENCE" ]; then
    "$EVAL_PY" - "$EASE_ROOT/open-unlearning/saves/eval" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="open-unlearning/eval",
    repo_type="dataset",
    allow_patterns=["tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json"],
    local_dir=sys.argv[1],
)
PY
fi
if [ ! -s "$RETAIN_REFERENCE" ]; then
    echo "Missing frozen retain95 reference after serial materialization: $RETAIN_REFERENCE" >&2
    exit 1
fi

MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,training,a1_steps,a2_steps,a1_lr,a2_lr,a1_checkpoint,a2_checkpoint,reference_a1_path,reference_a2_path,composition_mode" > "$MANIFEST"

run_eval() {
    local gpu="$1" training="$2" point="$3" w1="$4" w2="$5" filter="$6"
    local a1="${TRAIN_A1[$training]}" a2="${TRAIN_A2[$training]}"
    local reference="${TRAIN_REFERENCE[$training]}"
    local a1_steps="${TRAIN_A1_STEPS[$training]}" a2_steps="${TRAIN_A2_STEPS[$training]}"
    local a1_lr="${TRAIN_A1_LR[$training]}" a2_lr="${TRAIN_A2_LR[$training]}"
    local tag="${training}_${point}"
    local task="tofu_Llama-3.1-8B-Instruct_forget05_F2R_V512_DUAL_DEPTH6_8B_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/$task/F2R_REPORT.json"
    echo "$tag,$w1,$w2,$filter,$task,$report,$training,$a1_steps,$a2_steps,$a1_lr,$a2_lr,$a1,$a2,$reference,$reference,reference_delta" >> "$MANIFEST"
    if [ "$LAUNCH_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "depth6_eval_reuse training=$training point=$point"
        return
    fi
    echo "depth6_eval_start training=$training point=$point GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        CF_PATH="$V512_DATA" MODELS_ROOT="$RESULTS_DIR/frozen_${training}" \
        A1_CHECKPOINT_OVERRIDE="$a1" A2_CHECKPOINT_OVERRIDE="$a2" \
        A1_REFERENCE_PATH="$reference" A2_REFERENCE_PATH="$reference" \
        REFERENCE_PATH="$reference" COMPOSITION_MODE=reference_delta \
        A1_NUM_LAYER=6 A2_NUM_LAYER=6 \
        A1_TRAIN_STEPS="$a1_steps" A2_TRAIN_STEPS="$a2_steps" \
        A1_TRAIN_LR="$a1_lr" A2_TRAIN_LR="$a2_lr" \
        A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
        TRAIN_MODEL_CONFIG=llama-3-8b \
        EVAL_MODEL_CONFIG=Llama-3.1-8B-Instruct_DualULD \
        HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.1-8B-Instruct \
        HF_MODEL_NAME=Llama-3.1-8B-Instruct \
        TASK_MODEL_NAME=Llama-3.1-8B-Instruct \
        RETAIN_LOGS_PATH="$RETAIN_REFERENCE" AUTO_FETCH_RETAIN_LOGS=0 \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER="$filter" \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false SEQUENCE_ROUTER_ENABLED=false \
        CALIBRATION_PATH=null SELECTION_RETAIN_ACCESS=true HF_PREFLIGHT=0 \
        F2R_VARIANT="F2D-Forget05-V512-DualDepth6-8B-Static-NoRouter" \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "depth6_eval_done training=$training point=$point GPU=$gpu"
}

echo "[3/4] Evaluate eight static depth-6 configurations"
pids=()
index=0
wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failures=$((failures + 1))
    done
    pids=()
}
for spec in "${TRAININGS[@]}"; do
    IFS=: read -r training _ _ _ _ <<< "$spec"
    for point_spec in "${POINTS[@]}"; do
        IFS=: read -r point w1 w2 filter <<< "$point_spec"
        gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
        run_eval "$gpu" "$training" "$point" "$w1" "$w2" "$filter" &
        pids+=("$!")
        index=$((index + 1))
        if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then wait_batch; fi
    done
done
if [ "${#pids[@]}" -gt 0 ]; then wait_batch; fi

echo "[4/4] Summarize static depth-6 results"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin 0.005 --sweep-kind training \
    || failures=$((failures + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

rows = [row for row in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if row.get("aggregate_score")]
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
if rows:
    best = rows[0]
    print("===== Forget05 Llama-3.1-8B static 6/6 best =====")
    print("config =", best.get("config"))
    print("Agg    =", best["aggregate_score"])
    print("Mem    =", best["memorization_score"])
    print("Util   =", best["retain_utility_score"])
    print("relative to completed 4/4 best 0.602100:", f"{float(best['aggregate_score']) - 0.602100:+.6f}")
    print("distance to 0.58:", f"{float(best['aggregate_score']) - 0.58:+.6f}")
PY
fi

echo "Depth-6 table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$failures" -gt 0 ]; then
    echo "$failures depth-6 evaluation/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi

