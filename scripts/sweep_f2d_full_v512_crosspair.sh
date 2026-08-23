#!/usr/bin/env bash
# Evaluation-only cross-pair screen between the strongest FullAnswer and
# V5.12 asymmetric assistants.  No assistant is retrained.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
# Preserve launch-specific settings before sourcing a generic project .env.
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_FULL_MANIFEST="${FULLANSWER_MANIFEST:-}"
LAUNCH_V512_ROOT="${F2D_V512_ASYM_ROOT:-}"
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

EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_FULL_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_STEP="${FULLANSWER_STEP:-72}"
V512_ROOT="$(absolute_from_root "${LAUNCH_V512_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_asym6_train_seed42/v512_asym_a84_a60}")"
V512_A1_STEP="${F2D_V512_A1_STEP:-84}"
V512_A2_STEP="${F2D_V512_A2_STEP:-60}"
REQUESTED_GPUS="$LAUNCH_GPUS"
REQUESTED_EVAL_BS="$LAUNCH_EVAL_BS"
REQUESTED_RESUME="$LAUNCH_RESUME"
SUMMARY_ROOT="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_f2d_full_v512_crosspair"

for required in "$DATA" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing cross-pair prerequisite: $required" >&2
        exit 1
    fi
done
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing evaluation Python: $EVAL_PY" >&2
    exit 1
fi

FULL_ROOT="$(awk -F, -v tag="$FULL_TAG" '$1==tag {gsub(/\r/, "", $NF); print $NF}' "$FULL_MANIFEST" | tail -n 1)"
if [ -z "$FULL_ROOT" ]; then
    echo "Could not resolve FullAnswer tag $FULL_TAG from $FULL_MANIFEST" >&2
    exit 1
fi
FULL_ROOT="$(absolute_from_root "$FULL_ROOT")"

checkpoint_exact() {
    local root="$1" role="$2" step="$3"
    find "$root/$role" -type d -name "checkpoint-${step}" 2>/dev/null \
        | sort | tail -n 1
}

FULL_A1="$(checkpoint_exact "$FULL_ROOT" a1 "$FULL_STEP")"
FULL_A2="$(checkpoint_exact "$FULL_ROOT" a2 "$FULL_STEP")"
V512_A1="$(checkpoint_exact "$V512_ROOT" a1 "$V512_A1_STEP")"
V512_A2="$(checkpoint_exact "$V512_ROOT" a2 "$V512_A2_STEP")"

for checkpoint in "$FULL_A1" "$FULL_A2" "$V512_A1" "$V512_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing required frozen checkpoint: ${checkpoint:-unresolved}" >&2
        echo "FullAnswer root: $FULL_ROOT" >&2
        echo "V5.12 root:      $V512_ROOT" >&2
        exit 1
    fi
done

cat <<EOF
============================================================
FullAnswer/V5.12 frozen cross-pair screen
  FullAnswer A1 : $FULL_A1
  FullAnswer A2 : $FULL_A2
  V5.12 A1      : $V512_A1
  V5.12 A2      : $V512_A2
  pair 1        : FullAnswer A1 + V5.12 A2
  pair 2        : V5.12 A1 + FullAnswer A2
  evaluations   : 18 + 18 = 36
  retraining    : none
  GPUs          : $REQUESTED_GPUS
============================================================
EOF

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "Dry run complete; all four frozen checkpoints resolved."
    exit 0
fi

mkdir -p "$SUMMARY_ROOT"

run_combo() {
    local label="$1" a1="$2" a2="$3" a1_grid="$4" a2_grid="$5"
    local sweep_name="f2d_full_v512_crosspair_${label}"
    local result_dir="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${sweep_name}"

    echo "crosspair_start combo=$label"
    env LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPUS="$REQUESTED_GPUS" \
        EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
        CF_PATH="$DATA" MODELS_ROOT="$SUMMARY_ROOT/frozen_${label}" \
        A1_CHECKPOINT_OVERRIDE="$a1" A2_CHECKPOINT_OVERRIDE="$a2" \
        SWEEP_NAME="$sweep_name" RESULTS_DIR="$result_dir" \
        WEIGHT_A1_GRID="$a1_grid" WEIGHT_A2_GRID="$a2_grid" \
        TOP_FILTERS="0.0002 0.0003" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
        bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
    echo "crosspair_done combo=$label"
}

echo "[1/3] FullAnswer A1 + V5.12 A2"
run_combo \
    fulla1_v512a2 "$FULL_A1" "$V512_A2" \
    "-1.4 -1.6 -1.8" "1.2 1.4 1.6"

echo "[2/3] V5.12 A1 + FullAnswer A2"
run_combo \
    v512a1_fulla2 "$V512_A1" "$FULL_A2" \
    "-1.2 -1.4 -1.6" "1.4 1.6 1.8"

echo "[3/3] Consolidating 36 cross-pair reports"
"$EVAL_PY" - "$EASE_ROOT" "$SUMMARY_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
sources = {
    "fulla1_v512a2": root / "open-unlearning/saves/sweeps/forget05_f2d_full_v512_crosspair_fulla1_v512a2/F2R_SWEEP.csv",
    "v512a1_fulla2": root / "open-unlearning/saves/sweeps/forget05_f2d_full_v512_crosspair_v512a1_fulla2/F2R_SWEEP.csv",
}
rows = []
for combo, path in sources.items():
    with path.open(encoding="utf-8", newline="") as handle:
        current = [row for row in csv.DictReader(handle) if row.get("aggregate_score")]
    if len(current) != 18:
        raise SystemExit(f"Expected 18 valid results for {combo}; got {len(current)}")
    for row in current:
        item = dict(row)
        item["cross_pair"] = combo
        rows.append(item)

rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
fields = list(dict.fromkeys(["cross_pair"] + [key for row in rows for key in row]))
output.mkdir(parents=True, exist_ok=True)
all_csv = output / "F2R_CROSSPAIR_ALL.csv"
with all_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print("\n #       Agg       Mem      Util        MU     w1     w2    filter  cross pair")
for index, row in enumerate(rows[:12], 1):
    print(
        f"{index:2d} "
        f"{float(row['aggregate_score']):9.6f} "
        f"{float(row['memorization_score']):9.6f} "
        f"{float(row['retain_utility_score']):9.6f} "
        f"{float(row['model_utility']):9.6f} "
        f"{float(row['weight_a1']):6.1f} "
        f"{float(row['weight_a2']):6.1f} "
        f"{float(row['top_filter']):9.4g}  "
        f"{row['cross_pair']}"
    )

best = rows[0]
agg = float(best["aggregate_score"])
print("\n===== FullAnswer/V5.12 cross-pair best =====")
print("pair   =", best["cross_pair"])
print("config =", best["tag"])
print(f"Agg    = {agg:.6f}")
print(f"Mem    = {float(best['memorization_score']):.6f}")
print(f"Util   = {float(best['retain_utility_score']):.6f}")
print(f"MU     = {float(best['model_utility']):.6f}")
print(
    "w1/w2/filter =",
    best["weight_a1"], best["weight_a2"], best["top_filter"],
)
print(f"relative to V5.12 0.526883:   {agg - 0.526883:+.6f}")
print(f"relative to FullAnswer 0.553712: {agg - 0.553712:+.6f}")
print(f"distance to BS-S 0.580000:    {agg - 0.580000:+.6f}")
print("report =", best["report"])
print("all results =", all_csv)
PY

echo "Cross-pair results: $SUMMARY_ROOT/F2R_CROSSPAIR_ALL.csv"
