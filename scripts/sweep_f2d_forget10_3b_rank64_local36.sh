#!/usr/bin/env bash
# Evaluation-only local refinement around the best Forget10 3B rank-64 point.
# Reuses the frozen rank-64 A1/A2 checkpoints; no assistant is retrained.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
GPUS="${GPUS:-0 1 2 3}"
EVAL_BS="${EVAL_BS:-1}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
TARGET_AGG="${TARGET_AGG:-0.64}"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

DATA="${F2D_FORGET10_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget10_author_hybrid400_seed42_v512reused_v511new_incremental_manualfix_v1_full.jsonl}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-${EASE_ROOT}/open-unlearning/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json}"
MODEL_ROOT="${MODEL_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2d_forget10_3b_rank3_seed42/rank64_seed42}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/forget10_f2d_3b_rank64_local36_seed42}"
SWEEPER="$EASE_ROOT/scripts/sweep_f2r_weights.sh"
TRAIN_PY="${TRAIN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"

checkpoint_at() {
    find "$1" -type d -name "checkpoint-$2" 2>/dev/null | sort | tail -n 1
}

reference_for() {
    (cd "$1/../fullmodel" 2>/dev/null && pwd)
}

for required in "$DATA" "$RETAIN_LOGS_PATH" "$SWEEPER"; do
    [ -s "$required" ] || { echo "Missing rank-64 local-sweep prerequisite: $required" >&2; exit 1; }
done
[ -x "$TRAIN_PY" ] || { echo "Missing training Python: $TRAIN_PY" >&2; exit 1; }

A1_CHECKPOINT="$(checkpoint_at "$MODEL_ROOT/a1" 192)"
A2_CHECKPOINT="$(checkpoint_at "$MODEL_ROOT/a2" 144)"
for checkpoint in "$A1_CHECKPOINT" "$A2_CHECKPOINT"; do
    [ -n "$checkpoint" ] && [ -d "$checkpoint" ] \
        || { echo "Missing completed rank-64 checkpoint" >&2; exit 1; }
done
A1_REFERENCE="$(reference_for "$A1_CHECKPOINT")"
A2_REFERENCE="$(reference_for "$A2_CHECKPOINT")"
for reference in "$A1_REFERENCE" "$A2_REFERENCE"; do
    [ -d "$reference" ] \
        || { echo "Missing depth-matched reference: $reference" >&2; exit 1; }
done

"$TRAIN_PY" - "$DATA" "$A1_CHECKPOINT" "$A2_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

data_path, *checkpoints = sys.argv[1:]
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
expected = [f"forget10_perturbed-{index:05d}" for index in range(400)]
if len(rows) != 400 or [row.get("source_id") for row in rows] != expected:
    raise SystemExit("Forget10 coverage/order gate failed")
if any(set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"} for row in rows):
    raise SystemExit("Forget10 factorial-cell gate failed")
for checkpoint in checkpoints:
    config = json.load(open(Path(checkpoint) / "adapter_config.json", encoding="utf-8"))
    if config.get("r") != 64 or config.get("lora_alpha") != 128:
        raise SystemExit(f"rank/alpha gate failed for {checkpoint}")
    modules = set(config.get("target_modules", []))
    required = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if modules != required:
        raise SystemExit(f"all-linear target-module gate failed for {checkpoint}: {sorted(modules)}")
print("rank64_local_hard_gate rows=400 rank=64 alpha=128 target_modules=all_linear retraining=false")
PY

read -r -a GPU_LIST <<< "$GPUS"
[ "${#GPU_LIST[@]}" -gt 0 ] || { echo "GPUS must not be empty" >&2; exit 1; }

cat <<EOF
============================================================
Forget10 Llama-3.2-3B rank-64 local inference refinement
  frozen A1      : $A1_CHECKPOINT
  frozen A2      : $A2_CHECKPOINT
  w1 grid        : -1.55 -1.60 -1.65 -1.70
  w2 grid        : 1.75 1.85 1.95
  filter grid    : 0.00012 0.00017 0.00022
  evaluations    : 36
  retraining     : none
  composition    : role-specific reference_delta
  routing/gating : disabled
  target Agg     : $TARGET_AGG
  results        : $RESULTS_DIR
============================================================
EOF

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run OK; frozen data, checkpoints, references, and 36-point grid passed."
    exit 0
fi

echo "[1/2] Evaluate 36 frozen rank-64 reference-delta points"
env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget10 \
    GPUS="$GPUS" EVAL_BS="$EVAL_BS" RESUME="$RESUME" \
    CF_PATH="$DATA" MODELS_ROOT="$RESULTS_DIR/frozen_unused" \
    A1_CHECKPOINT_OVERRIDE="$A1_CHECKPOINT" \
    A2_CHECKPOINT_OVERRIDE="$A2_CHECKPOINT" \
    A1_REFERENCE_PATH="$A1_REFERENCE" A2_REFERENCE_PATH="$A2_REFERENCE" \
    REFERENCE_PATH=auto COMPOSITION_MODE=reference_delta \
    A1_NUM_LAYER=4 A2_NUM_LAYER=4 A1_LORA_R=64 A2_LORA_R=64 \
    A1_LORA_ALPHA=128 A2_LORA_ALPHA=128 \
    A1_TRAIN_STEPS=192 A2_TRAIN_STEPS=144 \
    A1_TRAIN_LR=5e-4 A2_TRAIN_LR=7.5e-4 \
    A1_RETAIN_WEIGHT=0.4 A2_RETAIN_WEIGHT=0.3 \
    TRAIN_LOSS_CONFIG=factorial_reference_preserving \
    TRAIN_MODEL_CONFIG=llama-3-3b \
    EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD \
    HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.2-3B-Instruct \
    HF_MODEL_NAME=Llama-3.2-3B-Instruct \
    TASK_MODEL_NAME=Llama-3.2-3B-Instruct \
    RETAIN_LOGS_PATH="$RETAIN_LOGS_PATH" AUTO_FETCH_RETAIN_LOGS=0 \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false SEQUENCE_ROUTER_ENABLED=false \
    CALIBRATION_PATH=null SELECTION_RETAIN_ACCESS=true \
    WEIGHT_A1_GRID="-1.55 -1.60 -1.65 -1.70" \
    WEIGHT_A2_GRID="1.75 1.85 1.95" \
    TOP_FILTERS="0.00012 0.00017 0.00022" \
    SWEEP_NAME=f2d_forget10_3b_rank64_local36_seed42 \
    RESULTS_DIR="$RESULTS_DIR" TARGET_AGG="$TARGET_AGG" TARGET_MARGIN=0 \
    bash "$SWEEPER"

echo "[2/2] Forget10 3B rank-64 local refinement complete"
echo "Sweep table: $RESULTS_DIR/F2R_SWEEP.md"
