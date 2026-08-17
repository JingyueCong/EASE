#!/usr/bin/env bash
# Full-coverage F2D-DiD: one four-cell causal unit for every forget05 QA.
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
UNITS="${UNITS:-200}"
BLOCK_SIZE="${BLOCK_SIZE:-20}"
GPUS="${GPUS:-0 1 2 3}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_full_authorblock_v1.jsonl}"
SWEEP_NAME="${SWEEP_NAME:-f2d_did${UNITS}_full_authorblock_seed${SEED}}"

if [ "$UNITS" -ne 200 ]; then
    echo "This full-coverage forget05 protocol requires UNITS=200." >&2
    exit 1
fi

echo "============================================================"
echo "Full-coverage F2D-DiD experiment"
echo "  split / units      : $SPLIT / $UNITS (no source sampling)"
echo "  factorial data     : $DATA_PATH"
echo "  author blocks      : $((UNITS / BLOCK_SIZE)) x $BLOCK_SIZE"
echo "  shared replacement : one identity per author block"
echo "  GPUs               : $GPUS"
echo "  exact steps        : 36 / 48 / 60"
echo "============================================================"

if [ ! -s "$DATA_PATH" ]; then
    LOAD_DOTENV=0 STOP_AFTER_GENERATION=true \
        SPLIT="$SPLIT" UNITS="$UNITS" SEED="$SEED" \
        DATA_PATH="$DATA_PATH" \
        SHARED_REPLACEMENT_PER_BLOCK=true \
        CF_CONCURRENCY="${CF_CONCURRENCY:-4}" \
        bash "$EASE_ROOT/scripts/run_ciru40_tofu.sh"
fi

"${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}" - \
    "$DATA_PATH" "$UNITS" "$BLOCK_SIZE" <<'PY'
import json
import sys
from collections import Counter, defaultdict

path, expected_units, block_size = sys.argv[1:]
expected_units = int(expected_units)
block_size = int(block_size)
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

ids = [row["source_id"] for row in rows]
if len(rows) != expected_units or len(set(ids)) != expected_units:
    raise SystemExit(
        f"Expected {expected_units} unique units, got rows={len(rows)} ids={len(set(ids))}"
    )
indices = sorted(int(source_id.rsplit("-", 1)[1]) for source_id in ids)
if indices != list(range(expected_units)):
    raise SystemExit("Full design does not cover every forget05 source index 0..199")

block_counts = Counter(index // block_size for index in indices)
replacements = defaultdict(set)
for row in rows:
    index = int(row["source_id"].rsplit("-", 1)[1])
    replacements[index // block_size].add(row["replacement_entity"].casefold())
if sorted(block_counts.values()) != [block_size] * (expected_units // block_size):
    raise SystemExit(f"Unexpected block counts: {dict(block_counts)}")
bad = {block: values for block, values in replacements.items() if len(values) != 1}
if bad:
    raise SystemExit(f"Author blocks do not share one replacement identity: {bad}")
print(
    f"Full-design audit OK: units={len(rows)}, cells={4 * len(rows)}, "
    f"blocks={len(block_counts)}, replacements={len(replacements)}"
)
PY

# b200_s36_u1 is the compute-matched primary comparison against the best
# 40-unit F2D-DiD run. The remaining cells test uniformity and compute scaling.
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
b200_s36_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:36:36 \
b200_s36_u2:2:2:16:16:1e-3:1e-3:1:1:2:2:36:36 \
b200_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
b200_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.5}" WEIGHT_A2="${WEIGHT_A2:-0.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0002}" \
    F2R_VARIANT="F2D-DiD-Balanced-${UNITS}-FullAuthorBlock" \
    EVAL_BS="${EVAL_BS:-4}" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
