#!/usr/bin/env bash
# Resumable V5.12 pilot: 12 training pairs, then two 75-point fine sweeps.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PY="${PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
SPLIT=forget05
SEED=42
DATA="${F2D_V512_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}"
PROFILES="${F2D_V512_PROFILES_PATH:-${DATA}.profiles.json}"
AUDIT_SUMMARY="${F2D_V512_AUDIT_SUMMARY:-${EASE_ROOT}/audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}"
TRAIN_SWEEP="${F2D_V512_TRAIN_SWEEP:-f2d_v512_pilot_train_seed42}"
TRAIN_RESULTS="${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${TRAIN_SWEEP}"
MODELS_ROOT="${EASE_ROOT}/ULD/outputs_trained_models/f2d_did_1b_${SPLIT}_${TRAIN_SWEEP}"
SUMMARY_ROOT="${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_f2d_v512_pilot_search_seed42"

for required in "$DATA" "$PROFILES" "$AUDIT_SUMMARY"; do
    if [ ! -s "$required" ]; then
        echo "Missing frozen V5.12 prerequisite: $required" >&2
        exit 1
    fi
done
if [ ! -x "$PY" ]; then
    echo "Missing evaluation Python: $PY" >&2
    exit 1
fi

"$PY" - "$DATA" "$PROFILES" "$AUDIT_SUMMARY" <<'PY'
import json
import sys

data_path, profiles_path, audit_path = sys.argv[1:]
rows = [json.loads(line) for line in open(data_path) if line.strip()]
profiles = json.load(open(profiles_path))
audit = json.load(open(audit_path))
errors = []
if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
    errors.append("data must contain 200 unique rows")
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in rows):
    errors.append("data contains a non-V5.12 design")
if profiles.get("design_version") != "tofu-author-pairbudget-v5.12":
    errors.append("profile design is not V5.12")
if audit.get("records") != 200 or audit.get("unique_source_ids") != 200:
    errors.append("audit does not cover 200 unique rows")
if audit.get("units_with_deterministic_errors") != 0:
    errors.append("audit contains deterministic errors")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    errors.append("audit source-index coverage is incomplete")
if errors:
    raise SystemExit("V5.12 pilot preflight failed: " + "; ".join(errors))
print("V5.12 pilot preflight OK: rows=200 deterministic_errors=0")
PY

configs=()
for steps in 24 32 40 48; do
    for spec in 6e4:6e-4 8e4:8e-4 1e3:1e-3; do
        lr_tag="${spec%%:*}"
        lr="${spec#*:}"
        configs+=(
            "v512_p_s${steps}_lr${lr_tag}:2:2:16:16:${lr}:${lr}:1:1:1:1:${steps}:${steps}"
        )
    done
done
TRAIN_CONFIGS="${configs[*]}"

mkdir -p "$SUMMARY_ROOT"

cat <<EOF
============================================================
F2D V5.12 pilot parameter search
  frozen data       : $DATA
  GPUs              : $REQUESTED_GPUS
  training configs  : ${#configs[@]}
  screen point      : -1.8 / 1.8 / 0.0001
  fine finalists    : 2 x 75 evaluations
  total evaluations : $((${#configs[@]} + 150))
  resume            : $REQUESTED_RESUME
============================================================
EOF

echo "[1/4] Train and screen 12 V5.12 assistant pairs"
env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
    UNITS=200 SEED="$SEED" VIEWS=1 EVAL_BS="$REQUESTED_EVAL_BS" \
    CIRU_PATH="$DATA" CF_PATH="$DATA" \
    SWEEP_NAME="$TRAIN_SWEEP" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    MODELS_SWEEP_ROOT="$MODELS_ROOT" RESULTS_DIR="$TRAIN_RESULTS" \
    TASK_PREFIX="tofu_Llama-3.2-1B-Instruct_forget05_F2D_V512_PILOT" \
    WEIGHT_A1=-1.8 WEIGHT_A2=1.8 TOP_FILTER=0.0001 \
    F2R_VARIANT="F2D-PairBudget-v5.12-Pilot" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    RESUME="$REQUESTED_RESUME" TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"

echo "[2/4] Select the two best training configurations"
SELECTION="$SUMMARY_ROOT/screen_top2.tsv"
"$PY" - "$TRAIN_RESULTS/F2R_SWEEP.csv" "$SELECTION" <<'PY'
import csv
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
rows = [
    row for row in csv.DictReader(source.open(encoding="utf-8"))
    if row.get("aggregate_score")
]
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
if len(rows) < 2:
    raise SystemExit(f"Need two valid V5.12 screen results; got {len(rows)}")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for row in rows[:2]:
        handle.write(f"{row['tag']}\t{row['aggregate_score']}\n")
        print(
            f"screen_finalist training={row['tag']} "
            f"Agg={row['aggregate_score']} Mem={row['memorization_score']} "
            f"Util={row['retain_utility_score']}"
        )
PY

echo "[3/4] Run two local 5 x 5 x 3 frozen-inference sweeps"
while IFS=$'\t' read -r tag screen_agg; do
    fine_name="f2d_v512_pilot_fine_${tag}"
    echo "fine_start training_tag=$tag screen_agg=$screen_agg"
    env \
        LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
        EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
        CF_PATH="$DATA" MODELS_ROOT="${MODELS_ROOT}/${tag}" \
        SWEEP_NAME="$fine_name" \
        RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/${SPLIT}_${fine_name}" \
        WEIGHT_A1_GRID="-2.2 -2.0 -1.8 -1.6 -1.4" \
        WEIGHT_A2_GRID="1.4 1.6 1.8 2.0 2.2" \
        TOP_FILTERS="0.00005 0.0001 0.0002" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
        bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
    echo "fine_done training_tag=$tag"
done < "$SELECTION"

echo "[4/4] Consolidate screen and fine results"
"$PY" - "$TRAIN_RESULTS/F2R_SWEEP.csv" "$EASE_ROOT" "$SUMMARY_ROOT" <<'PY'
import csv
import glob
import sys
from pathlib import Path

screen_path = Path(sys.argv[1])
root = Path(sys.argv[2])
output = Path(sys.argv[3])
rows = []

for row in csv.DictReader(screen_path.open(encoding="utf-8")):
    if not row.get("aggregate_score"):
        continue
    item = dict(row)
    item["search_stage"] = "screen"
    item["training_tag"] = row["tag"]
    rows.append(item)

pattern = str(
    root / "open-unlearning/saves/sweeps/forget05_f2d_v512_pilot_fine_*"
    / "F2R_SWEEP.csv"
)
prefix = "forget05_f2d_v512_pilot_fine_"
for name in glob.glob(pattern):
    training_tag = Path(name).parent.name[len(prefix):]
    for row in csv.DictReader(open(name, encoding="utf-8")):
        if not row.get("aggregate_score"):
            continue
        item = dict(row)
        item["search_stage"] = "fine"
        item["training_tag"] = training_tag
        rows.append(item)

if not rows:
    raise SystemExit("No V5.12 pilot results found")
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
fields = list(dict.fromkeys(
    ["search_stage", "training_tag"]
    + [key for row in rows for key in row]
))
output.mkdir(parents=True, exist_ok=True)
with (output / "F2R_V512_PILOT_ALL.csv").open(
    "w", encoding="utf-8", newline=""
) as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
with (output / "F2R_V512_PILOT_BEST.txt").open(
    "w", encoding="utf-8"
) as handle:
    for key in (
        "search_stage", "training_tag", "tag", "aggregate_score",
        "memorization_score", "retain_utility_score", "model_utility",
        "weight_a1", "weight_a2", "top_filter", "report",
    ):
        handle.write(f"{key}={best.get(key, '')}\n")
print(
    "V5.12 pilot best: "
    f"training={best['training_tag']} stage={best['search_stage']} "
    f"Agg={best['aggregate_score']} Mem={best['memorization_score']} "
    f"Util={best['retain_utility_score']} w1={best['weight_a1']} "
    f"w2={best['weight_a2']} filter={best['top_filter']}"
)
PY

echo "V5.12 pilot best: $SUMMARY_ROOT/F2R_V512_PILOT_BEST.txt"
echo "All results      : $SUMMARY_ROOT/F2R_V512_PILOT_ALL.csv"
