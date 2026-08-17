#!/usr/bin/env bash
# Generate a nested CIRU-80 design, then run a compute-controlled F2D-DiD sweep.
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

SPLIT="${SPLIT:-forget05}"
SEED="${SEED:-42}"
UNITS="${UNITS:-80}"
GPUS="${GPUS:-0 1 2 3}"
PARENT_UNITS="${PARENT_UNITS:-40}"
PARENT_PATH="${PARENT_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${PARENT_UNITS}_seed${SEED}_strict_v2.jsonl}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_nested${PARENT_UNITS}_strict_v1.jsonl}"
SWEEP_NAME="${SWEEP_NAME:-f2d_did${UNITS}_nested${PARENT_UNITS}_seed${SEED}}"

if [ ! -s "$PARENT_PATH" ]; then
    echo "Missing audited parent design: $PARENT_PATH" >&2
    exit 1
fi
if [ "$UNITS" -le "$PARENT_UNITS" ]; then
    echo "UNITS must exceed PARENT_UNITS for a nested budget experiment." >&2
    exit 1
fi

echo "============================================================"
echo "Nested F2D-DiD causal-budget experiment"
echo "  parent/new units : $PARENT_UNITS -> $UNITS"
echo "  parent            : $PARENT_PATH"
echo "  nested data       : $DATA_PATH"
echo "  GPUs              : $GPUS"
echo "  exact steps       : 36 / 48 / 60"
echo "============================================================"

if [ ! -s "$DATA_PATH" ]; then
    LOAD_DOTENV=0 STOP_AFTER_GENERATION=true \
        SPLIT="$SPLIT" UNITS="$UNITS" SEED="$SEED" \
        DATA_PATH="$DATA_PATH" INCLUDE_SOURCE_IDS_FROM="$PARENT_PATH" \
        bash "$EASE_ROOT/scripts/run_ciru40_tofu.sh"
fi

"${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}" - \
    "$PARENT_PATH" "$DATA_PATH" "$PARENT_UNITS" "$UNITS" <<'PY'
import json
import sys

parent_path, nested_path, expected_parent, expected_nested = sys.argv[1:]
def records(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["source_id"]: row for row in rows}
parent = records(parent_path)
nested = records(nested_path)
if len(parent) != int(expected_parent) or len(nested) != int(expected_nested):
    raise SystemExit(
        f"Unexpected source counts: parent={len(parent)}, nested={len(nested)}"
    )
missing = sorted(parent.keys() - nested.keys())
if missing:
    raise SystemExit(f"Nested design dropped parent sources: {missing[:3]}")
changed = sorted(source_id for source_id, row in parent.items() if nested[source_id] != row)
if changed:
    raise SystemExit(f"Nested design changed parent records: {changed[:3]}")
print(
    f"Nested audit OK: {len(parent)}/{len(nested)} complete parent records preserved"
)
PY

# The primary b80_s36_u1 condition exactly matches the optimizer-step and
# uniform-loss budget of the best CIRU-40 assistant pair. Other cells expose
# whether the larger design benefits from more compute or stronger uniformity.
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
b${UNITS}_s36_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:36:36 \
b${UNITS}_s36_u2:2:2:16:16:1e-3:1e-3:1:1:2:2:36:36 \
b${UNITS}_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
b${UNITS}_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.5}" WEIGHT_A2="${WEIGHT_A2:-0.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0002}" \
    F2R_VARIANT="F2D-DiD-Balanced-${UNITS}-Nested${PARENT_UNITS}" \
    EVAL_BS="${EVAL_BS:-4}" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
