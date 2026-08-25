#!/usr/bin/env bash
# Retain-free Llama-3.2-3B forget05 static Dual Assistant experiment.
# Train one isolated 4/4 assistant pair on the frozen 200-row V5.12 and
# FullAnswer artifacts, then perform a preregistered 12-point strength scan.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
MODEL_SIZE_TAG="${MODEL_SIZE_TAG:-3b}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-Llama-3.2-3B}"
TRAIN_MODEL_CONFIG="${TRAIN_MODEL_CONFIG:-llama-3-3b}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-Llama-3.2-3B-Instruct_DualULD}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-3B-Instruct}"
HF_MODEL_NAME="${HF_MODEL_NAME:-Llama-3.2-3B-Instruct}"
TASK_MODEL_NAME="${TASK_MODEL_NAME:-Llama-3.2-3B-Instruct}"

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
MODEL_TAG="${MODEL_TAG:-f2d_v512_dual_depth4_${MODEL_SIZE_TAG}_train_seed42}"
SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth4_${MODEL_SIZE_TAG}_coarse12_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_${MODEL_SIZE_TAG}_forget05_${MODEL_TAG}"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"
TRAIN_LOG_DIR="$RESULTS_DIR/training_logs"
RETAIN_REFERENCE_RELATIVE="${RETAIN_REFERENCE_RELATIVE:-tofu_${HF_MODEL_NAME}_retain95/TOFU_EVAL.json}"
RETAIN_REFERENCE="$EASE_ROOT/open-unlearning/saves/eval/$RETAIN_REFERENCE_RELATIVE"
TRAIN_CONFIG_PATH="$EASE_ROOT/ULD/configs/model/${TRAIN_MODEL_CONFIG}.yaml"
EVAL_CONFIG_PATH="$EASE_ROOT/open-unlearning/configs/model/${EVAL_MODEL_CONFIG}.yaml"

for required in \
    "$V512_DATA" "$FULL_DATA" "$V512_AUDIT" \
    "$TRAIN_CONFIG_PATH" "$EVAL_CONFIG_PATH"; do
    if [ ! -s "$required" ]; then
        echo "Missing immutable forget05 prerequisite: $required" >&2
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
for label, path in (("V5.12", v512_path), ("FullAnswer", full_path)):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ids = [row.get("source_id") for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200:
        raise SystemExit(f"{label}: expected 200 unique forget05 rows")
    if ids != [f"forget05_perturbed-{index:05d}" for index in range(200)]:
        raise SystemExit(f"{label}: non-canonical forget05 ordering or coverage")
    if any(set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"} for row in rows):
        raise SystemExit(f"{label}: incomplete factorial cells")

audit = json.load(open(audit_path, encoding="utf-8"))
if audit.get("records") != 200 or audit.get("unique_source_ids") != 200:
    raise SystemExit("V5.12 audit coverage gate failed")
if audit.get("units_with_deterministic_errors") != 0:
    raise SystemExit("V5.12 deterministic audit gate failed")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    raise SystemExit("V5.12 source-index audit gate failed")
print("Forget05 training gate OK: rows=200 blocks=10 deterministic_errors=0")
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "Forget05 $MODEL_SIZE_TAG experiment requires four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

cat <<EOF
============================================================
Forget05 $MODEL_DISPLAY_NAME static Dual Assistant 4/4 experiment
  causal A1 data    : $V512_DATA
  FullAnswer A2 data: $FULL_DATA
  architecture      : A1=4 layers / A2=4 layers / LoRA rank=16
  A1 training       : 96 steps, lr=5e-4, uniform=1.5
  A2 training       : 72 steps, lr=1e-3, uniform=1.0
  train bs/ga       : 1 / 16
  composition       : shared depth-4 reference_delta
  routing/gating    : disabled
  weight grid       : A1=-1.7,-2.0,-2.4,-2.7; A2=1.4,1.7,2.0
  top filter        : 0.0004
  evaluations       : 12
  retain training   : none
  selection access  : retain95 evaluation only
  models            : $MODELS_ROOT
  results           : $RESULTS_DIR
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run OK; frozen data, audit, model isolation, and sweep grid validated."
    exit 0
fi

mkdir -p "$MODELS_ROOT" "$TRAIN_LOG_DIR"

train_role() {
    local gpu="$1" role="$2"
    local data="$V512_DATA"
    if [ "$role" = "a2" ]; then data="$FULL_DATA"; fi
    if [ -n "$(latest_checkpoint "$MODELS_ROOT/${role}_job/$role")" ]; then
        echo "forget05_${MODEL_SIZE_TAG}_train_reuse role=$role"
        return
    fi
    echo "forget05_${MODEL_SIZE_TAG}_train_start role=$role GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 \
        GPU="$gpu" CF_PATH="$data" \
        MODELS_ROOT="$MODELS_ROOT/${role}_job" \
        TRAIN_RUN_TAG="${MODEL_TAG}_${role}" TRAIN_ONLY=true TRAIN_ROLE="$role" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=4 A1_LORA_R=16 A1_LORA_ALPHA=32 \
        A1_TRAIN_LR=5e-4 A1_TRAIN_EP=1 A1_TRAIN_STEPS=96 \
        A1_RETAIN_WEIGHT=1.5 A1_TRAIN_BS=1 A1_TRAIN_GA=16 A1_SEED=42 \
        A2_NUM_LAYER=4 A2_LORA_R=16 A2_LORA_ALPHA=32 \
        A2_TRAIN_LR=1e-3 A2_TRAIN_EP=1 A2_TRAIN_STEPS=72 \
        A2_RETAIN_WEIGHT=1.0 A2_TRAIN_BS=1 A2_TRAIN_GA=16 A2_SEED=42 \
        TRAIN_MODEL_CONFIG="$TRAIN_MODEL_CONFIG" \
        EVAL_MODEL_CONFIG="$EVAL_MODEL_CONFIG" \
        HF_BASE_PREFIX="$HF_BASE_PREFIX" \
        HF_MODEL_NAME="$HF_MODEL_NAME" \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        F2R_VARIANT="F2D-Forget05-V512-DualDepth4-${MODEL_SIZE_TAG^^}-${role}" \
        HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$TRAIN_LOG_DIR/train_${role}.log" 2>&1
    echo "forget05_${MODEL_SIZE_TAG}_train_done role=$role GPU=$gpu"
}

echo "[1/3] Train isolated forget05 $MODEL_SIZE_TAG 4/4 assistants"
train_role "${GPU_LIST[0]}" a1 & a1_pid=$!
train_role "${GPU_LIST[1]}" a2 & a2_pid=$!
failures=0
wait "$a1_pid" || failures=$((failures + 1))
wait "$a2_pid" || failures=$((failures + 1))
if [ "$failures" -gt 0 ]; then
    echo "$failures forget05 $MODEL_SIZE_TAG training job(s) failed; inspect $TRAIN_LOG_DIR" >&2
    exit 1
fi

A1_CHECKPOINT="$(latest_checkpoint "$MODELS_ROOT/a1_job/a1")"
A2_CHECKPOINT="$(latest_checkpoint "$MODELS_ROOT/a2_job/a2")"
for checkpoint in "$A1_CHECKPOINT" "$A2_CHECKPOINT"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing completed forget05 $MODEL_SIZE_TAG checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done
REFERENCE_PATH="$(reference_for "$A1_CHECKPOINT")"
if [ -z "$REFERENCE_PATH" ] || [ ! -d "$REFERENCE_PATH" ]; then
    echo "Missing shared depth-4 reference for $A1_CHECKPOINT" >&2
    exit 1
fi

echo "[2/3] Materialize the frozen $MODEL_SIZE_TAG retain95 evaluation reference"
if [ ! -s "$RETAIN_REFERENCE" ]; then
    "$EVAL_PY" - "$RETAIN_REFERENCE_RELATIVE" \
        "$EASE_ROOT/open-unlearning/saves/eval" <<'PY'
import sys
from huggingface_hub import snapshot_download

relative_path, output_dir = sys.argv[1:]
snapshot_download(
    repo_id="open-unlearning/eval",
    repo_type="dataset",
    allow_patterns=[relative_path],
    local_dir=output_dir,
)
PY
fi
if [ ! -s "$RETAIN_REFERENCE" ]; then
    echo "Missing frozen retain95 reference after serial materialization: $RETAIN_REFERENCE" >&2
    exit 1
fi

echo "[3/3] Evaluate the 12-point static strength grid"
env \
    MODE=full SPLIT=forget05 \
    GPUS="$LAUNCH_GPUS" EVAL_BS="${EVAL_BS:-1}" RESUME="$LAUNCH_RESUME" \
    CF_PATH="$V512_DATA" \
    A1_CHECKPOINT_OVERRIDE="$A1_CHECKPOINT" \
    A2_CHECKPOINT_OVERRIDE="$A2_CHECKPOINT" \
    A1_NUM_LAYER=4 A2_NUM_LAYER=4 \
    A1_TRAIN_STEPS=96 A2_TRAIN_STEPS=72 \
    A1_TRAIN_LR=5e-4 A2_TRAIN_LR=1e-3 \
    A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
    TRAIN_MODEL_CONFIG="$TRAIN_MODEL_CONFIG" \
    EVAL_MODEL_CONFIG="$EVAL_MODEL_CONFIG" \
    HF_BASE_PREFIX="$HF_BASE_PREFIX" \
    HF_MODEL_NAME="$HF_MODEL_NAME" \
    TASK_MODEL_NAME="$TASK_MODEL_NAME" \
    RETAIN_LOGS_PATH="$RETAIN_REFERENCE" AUTO_FETCH_RETAIN_LOGS=0 \
    COMPOSITION_MODE=reference_delta REFERENCE_PATH="$REFERENCE_PATH" \
    A1_REFERENCE_PATH="$REFERENCE_PATH" A2_REFERENCE_PATH="$REFERENCE_PATH" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false SEQUENCE_ROUTER_ENABLED=false \
    CALIBRATION_PATH=null SELECTION_RETAIN_ACCESS=true \
    WEIGHT_A1_GRID="-1.7 -2.0 -2.4 -2.7" \
    WEIGHT_A2_GRID="1.4 1.7 2.0" \
    TOP_FILTERS=0.0004 \
    SWEEP_NAME="$SWEEP_NAME" RESULTS_DIR="$RESULTS_DIR" \
    TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"

echo "Forget05 $MODEL_SIZE_TAG table: $RESULTS_DIR/F2R_SWEEP.md"
