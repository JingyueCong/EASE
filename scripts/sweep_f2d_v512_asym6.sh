#!/usr/bin/env bash
# Resumable V5.12 asymmetric assistant-training screen.
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
PROFILES="${F2D_V512_PROFILES_PATH:-${DATA}.profiles.json}"
AUDIT_SUMMARY="${F2D_V512_AUDIT_SUMMARY:-${EASE_ROOT}/audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}"
TRAIN_SWEEP="${F2D_V512_ASYM_TRAIN_SWEEP:-f2d_v512_asym6_train_seed42}"
TRAIN_RESULTS="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${TRAIN_SWEEP}"
MODELS_ROOT="${EASE_ROOT}/ULD/outputs_trained_models/f2d_did_1b_forget05_${TRAIN_SWEEP}"
SUMMARY_ROOT="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_f2d_v512_asym6_search_seed42"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"

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
    raise SystemExit("V5.12 asymmetric preflight failed: " + "; ".join(errors))
print("V5.12 asymmetric preflight OK: rows=200 deterministic_errors=0")
PY

# tag:layers:layers:rank:rank:lr:lr:epochs:epochs:uniform:uniform:a1_steps:a2_steps
TRAIN_CONFIGS="\
v512_asym_a60_a48:2:2:16:16:5e-4:5e-4:1:1:1:1:60:48 \
v512_asym_a60_a60:2:2:16:16:5e-4:5e-4:1:1:1:1:60:60 \
v512_asym_a72_a48:2:2:16:16:5e-4:5e-4:1:1:1:1:72:48 \
v512_asym_a72_a60:2:2:16:16:5e-4:5e-4:1:1:1:1:72:60 \
v512_asym_a84_a48:2:2:16:16:5e-4:5e-4:1:1:1:1:84:48 \
v512_asym_a84_a60:2:2:16:16:5e-4:5e-4:1:1:1:1:84:60"

mkdir -p "$SUMMARY_ROOT/logs"

cat <<EOF
============================================================
F2D V5.12 asymmetric training screen
  training configs : 6
  A1/A2 steps      : 60/48 60/60 72/48 72/60 84/48 84/60
  A1/A2 LR         : 5e-4 / 5e-4
  points/config    : 3 frozen inference points
  total evaluations: 18 (6 training reports + 12 extra screens)
  GPUs             : $REQUESTED_GPUS
  resume           : $REQUESTED_RESUME
  protocol         : selection_retain_access=true
============================================================
EOF

echo "[1/4] Train six asymmetric V5.12 assistant pairs"
env \
    LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPUS="$REQUESTED_GPUS" \
    UNITS=200 SEED=42 VIEWS=1 EVAL_BS="$REQUESTED_EVAL_BS" \
    CIRU_PATH="$DATA" CF_PATH="$DATA" \
    SWEEP_NAME="$TRAIN_SWEEP" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    MODELS_SWEEP_ROOT="$MODELS_ROOT" RESULTS_DIR="$TRAIN_RESULTS" \
    TASK_PREFIX="tofu_Llama-3.2-1B-Instruct_forget05_F2D_V512_ASYM6" \
    WEIGHT_A1=-1.4 WEIGHT_A2=1.5 TOP_FILTER=0.0003 \
    F2R_VARIANT="F2D-PairBudget-v5.12-Asym6" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    RESUME="$REQUESTED_RESUME" TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"

echo "[2/4] Evaluate two additional frozen inference points per pair"
JOBS="$SUMMARY_ROOT/screen_jobs.tsv"
: > "$JOBS"
tags=(
    v512_asym_a60_a48 v512_asym_a60_a60
    v512_asym_a72_a48 v512_asym_a72_a60
    v512_asym_a84_a48 v512_asym_a84_a60
)
for tag in "${tags[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" mem -1.5 1.5 0.0003 >> "$JOBS"
    printf '%s\t%s\t%s\t%s\t%s\n' "$tag" filter -1.4 1.6 0.0004 >> "$JOBS"
done

IFS=' ' read -r -a gpu_list <<< "$REQUESTED_GPUS"
if [ "${#gpu_list[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU id" >&2
    exit 1
fi

run_screen() {
    local gpu="$1" tag="$2" label="$3" w1="$4" w2="$5" filter="$6"
    local name="f2d_v512_asym6_${tag}_${label}"
    local result_dir="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${name}"
    echo "screen_start training=$tag point=$label GPU=$gpu"
    env \
        LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPUS="$gpu" \
        EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
        CF_PATH="$DATA" MODELS_ROOT="${MODELS_ROOT}/${tag}" \
        SWEEP_NAME="$name" RESULTS_DIR="$result_dir" \
        WEIGHT_PAIRS="${w1}:${w2}" TOP_FILTERS="$filter" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
        bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh" \
        > "$SUMMARY_ROOT/logs/${tag}_${label}.log" 2>&1
    echo "screen_done training=$tag point=$label GPU=$gpu"
}

pids=()
failures=0
wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done
    pids=()
}

index=0
while IFS=$'\t' read -r tag label w1 w2 filter; do
    gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
    run_screen "$gpu" "$tag" "$label" "$w1" "$w2" "$filter" &
    pids+=("$!")
    index=$((index + 1))
    if [ "${#pids[@]}" -eq "${#gpu_list[@]}" ]; then
        wait_batch
    fi
done < "$JOBS"
if [ "${#pids[@]}" -gt 0 ]; then
    wait_batch
fi
if [ "$failures" -gt 0 ]; then
    echo "$failures asymmetric screen job(s) failed; inspect $SUMMARY_ROOT/logs" >&2
    exit 1
fi

echo "[3/4] Consolidate the 18 comparable results"
"$PY" - "$TRAIN_RESULTS/F2R_SWEEP.csv" "$JOBS" "$EASE_ROOT" "$SUMMARY_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

training_csv, jobs_path, root, output = map(Path, sys.argv[1:])
rows = []

for row in csv.DictReader(training_csv.open(encoding="utf-8")):
    if not row.get("aggregate_score"):
        continue
    item = dict(row)
    item["training_tag"] = row["tag"]
    item["screen_point"] = "balanced"
    rows.append(item)

for line in jobs_path.read_text(encoding="utf-8").splitlines():
    training_tag, label, _w1, _w2, _filter = line.split("\t")
    name = f"f2d_v512_asym6_{training_tag}_{label}"
    path = root / f"open-unlearning/saves/sweeps/forget05_{name}/F2R_SWEEP.csv"
    current = [
        row for row in csv.DictReader(path.open(encoding="utf-8"))
        if row.get("aggregate_score")
    ]
    if len(current) != 1:
        raise SystemExit(f"Expected one valid result in {path}; got {len(current)}")
    item = dict(current[0])
    item["training_tag"] = training_tag
    item["screen_point"] = label
    rows.append(item)

if len(rows) != 18:
    raise SystemExit(f"Expected 18 V5.12 asymmetric results; got {len(rows)}")
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
fields = list(dict.fromkeys(
    ["training_tag", "screen_point"] + [key for row in rows for key in row]
))
output.mkdir(parents=True, exist_ok=True)
all_csv = output / "F2R_V512_ASYM6_ALL.csv"
with all_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
best_path = output / "F2R_V512_ASYM6_BEST.txt"
with best_path.open("w", encoding="utf-8") as handle:
    for key in (
        "training_tag", "screen_point", "tag", "aggregate_score",
        "memorization_score", "retain_utility_score", "model_utility",
        "weight_a1", "weight_a2", "top_filter", "report",
    ):
        handle.write(f"{key}={best.get(key, '')}\n")

agg = float(best["aggregate_score"])
print("\n===== V5.12 asymmetric current best =====")
print("training =", best["training_tag"])
print("point    =", best["screen_point"])
print("config   =", best["tag"])
print(f"Agg      = {agg:.6f}")
print(f"Mem      = {float(best['memorization_score']):.6f}")
print(f"Util     = {float(best['retain_utility_score']):.6f}")
print(f"MU       = {float(best['model_utility']):.6f}")
print(
    "w1/w2/filter =",
    best["weight_a1"], best["weight_a2"], best["top_filter"],
)
print(f"relative to V5.12 boundary 0.511242: {agg - 0.511242:+.6f}")
print(f"relative to FullAnswer 0.553712:     {agg - 0.553712:+.6f}")
print(f"distance to BS-S 0.580000:          {agg - 0.580000:+.6f}")
print("report =", best["report"])
print("all results =", all_csv)
PY

echo "[4/4] V5.12 asymmetric training screen complete"
echo "Best: $SUMMARY_ROOT/F2R_V512_ASYM6_BEST.txt"
