#!/usr/bin/env bash
# Frozen-inference refinement for the V5.12 asymmetric 84/60 assistant pair.
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

absolute_from_root() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$EASE_ROOT" "$1" ;;
    esac
}

PY="${PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="$(absolute_from_root "${F2D_V512_DATA_PATH:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
MODELS_ROOT="$(absolute_from_root "${F2D_V512_MODELS_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_asym6_train_seed42/v512_asym_a84_a60}")"
SWEEP_NAME="${F2D_V512_FINE_NAME:-f2d_v512_asym_a84a60_fine}"
RESULTS_DIR="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"

if [ ! -s "$DATA" ]; then
    echo "Missing frozen V5.12 data: $DATA" >&2
    exit 1
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "Missing frozen V5.12 asymmetric 84/60 assistants: $MODELS_ROOT" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "Missing evaluation Python: $PY" >&2
    exit 1
fi

cat <<EOF
============================================================
F2D V5.12 asymmetric 84/60 frozen-inference refinement
  training config : A1=84 / A2=60, LR=5e-4 / 5e-4
  w1 grid         : -1.4 -1.3 -1.2
  w2 grid         : 1.2 1.3 1.4 1.5
  filter grid     : 0.0002 0.0003 0.0004
  evaluations     : 36
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
    WEIGHT_A1_GRID="-1.4 -1.3 -1.2" \
    WEIGHT_A2_GRID="1.2 1.3 1.4 1.5" \
    TOP_FILTERS="0.0002 0.0003 0.0004" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"

"$PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("aggregate_score")]
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
if len(rows) != 36:
    raise SystemExit(f"Expected 36 valid V5.12 84/60 fine results; got {len(rows)}")

best = rows[0]
agg = float(best["aggregate_score"])
print("\n===== V5.12 asymmetric 84/60 fine best =====")
print("config =", best["tag"])
print(f"Agg    = {agg:.6f}")
print(f"Mem    = {float(best['memorization_score']):.6f}")
print(f"Util   = {float(best['retain_utility_score']):.6f}")
print(f"MU     = {float(best['model_utility']):.6f}")
print(
    "w1/w2/filter =",
    best["weight_a1"], best["weight_a2"], best["top_filter"],
)
print(f"relative to asymmetric screen 0.516224: {agg - 0.516224:+.6f}")
print(f"relative to V5.12 boundary 0.511242:  {agg - 0.511242:+.6f}")
print(f"relative to FullAnswer 0.553712:      {agg - 0.553712:+.6f}")
print(f"distance to BS-S 0.580000:           {agg - 0.580000:+.6f}")
print("report =", best["report"])
PY

echo "Fine table: $RESULTS_DIR/F2R_SWEEP.md"
