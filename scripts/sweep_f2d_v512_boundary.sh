#!/usr/bin/env bash
# Frozen-assistant boundary refinement around the completed V5.12 pilot best.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PY="${PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="${F2D_V512_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}"
MODELS_ROOT="${F2D_V512_MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_pilot_train_seed42/v512_p_s48_lr6e4}"
SWEEP_NAME="${F2D_V512_BOUNDARY_NAME:-f2d_v512_boundary_s48_lr6e4}"
RESULTS_DIR="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"

if [ ! -s "$DATA" ]; then
    echo "Missing frozen V5.12 data: $DATA" >&2
    exit 1
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "Missing frozen V5.12 s48/lr6e-4 assistants: $MODELS_ROOT" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "Missing evaluation Python: $PY" >&2
    exit 1
fi

cat <<EOF
============================================================
F2D V5.12 frozen-assistant boundary refinement
  training config : v512_p_s48_lr6e4 (no retraining)
  w1 grid         : -1.5 -1.4 -1.3
  w2 grid         : 1.3 1.4 1.5
  filter grid     : 0.00015 0.0002 0.0003
  evaluations     : 27
  GPUs            : $REQUESTED_GPUS
  results         : $RESULTS_DIR
  resume          : $REQUESTED_RESUME
  protocol        : selection_retain_access=true
============================================================
EOF

env \
    LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPUS="$REQUESTED_GPUS" \
    EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
    CF_PATH="$DATA" MODELS_ROOT="$MODELS_ROOT" \
    SWEEP_NAME="$SWEEP_NAME" RESULTS_DIR="$RESULTS_DIR" \
    WEIGHT_A1_GRID="-1.5 -1.4 -1.3" \
    WEIGHT_A2_GRID="1.3 1.4 1.5" \
    TOP_FILTERS="0.00015 0.0002 0.0003" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"

"$PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("aggregate_score")]
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
if not rows:
    raise SystemExit("No valid V5.12 boundary results")

best = rows[0]
agg = float(best["aggregate_score"])
print("\n===== V5.12 boundary best =====")
print("config =", best["tag"])
print(f"Agg    = {agg:.6f}")
print(f"Mem    = {float(best['memorization_score']):.6f}")
print(f"Util   = {float(best['retain_utility_score']):.6f}")
print(f"MU     = {float(best['model_utility']):.6f}")
print(
    "w1/w2/filter =",
    best["weight_a1"], best["weight_a2"], best["top_filter"],
)
print(f"relative to V5.12 pilot 0.506728: {agg - 0.506728:+.6f}")
print(f"relative to V5.11 0.489164:       {agg - 0.489164:+.6f}")
print(f"relative to FullAnswer 0.553712:  {agg - 0.553712:+.6f}")
print(f"distance to BS-S 0.580000:       {agg - 0.580000:+.6f}")
print("report =", best["report"])
PY

echo "Boundary table: $RESULTS_DIR/F2R_SWEEP.md"
