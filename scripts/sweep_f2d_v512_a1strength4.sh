#!/usr/bin/env bash
# Pure V5.12 representation-strength experiment after alignment failed.
# Factorial training design: A1 steps {96,108} x A1 uniform {1.25,1.5};
# A2 and every other training choice remain fixed.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"

# Preserve explicit launch settings before loading the project environment.
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_PROFILES="${F2D_V512_PROFILES_PATH:-}"
LAUNCH_AUDIT="${F2D_V512_AUDIT_SUMMARY:-}"

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
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
PROFILES="$(absolute_from_root "${LAUNCH_PROFILES:-${DATA}.profiles.json}")"
AUDIT_SUMMARY="$(absolute_from_root "${LAUNCH_AUDIT:-audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}")"
TRAIN_SWEEP="${F2D_V512_A1_STRENGTH_SWEEP:-f2d_v512_a1strength4_train_seed42}"
TRAIN_RESULTS="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${TRAIN_SWEEP}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_1b_forget05_${TRAIN_SWEEP}"
SUMMARY_ROOT="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_f2d_v512_a1strength4_seed42"

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
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
profiles = json.load(open(profiles_path, encoding="utf-8"))
audit = json.load(open(audit_path, encoding="utf-8"))
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
    raise SystemExit("V5.12 A1-strength preflight failed: " + "; ".join(errors))
print("V5.12 A1-strength preflight OK: rows=200 deterministic_errors=0")
PY

# tag:layers:layers:rank:rank:lr:lr:epochs:epochs:a1_uniform:a2_uniform:a1_steps:a2_steps
TRAIN_CONFIGS="\
v512_a1s96_u1p25:2:2:16:16:5e-4:5e-4:1:1:1.25:1.0:96:60 \
v512_a1s96_u1p5:2:2:16:16:5e-4:5e-4:1:1:1.5:1.0:96:60 \
v512_a1s108_u1p25:2:2:16:16:5e-4:5e-4:1:1:1.25:1.0:108:60 \
v512_a1s108_u1p5:2:2:16:16:5e-4:5e-4:1:1:1.5:1.0:108:60"

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "This four-cell training experiment needs four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

mkdir -p "$SUMMARY_ROOT/logs"

cat <<EOF
============================================================
Pure V5.12 A1-strength 2x2 representation experiment
  causal data       : $DATA
  A1 factors        : steps={96,108} x uniform={1.25,1.5}
  A2 fixed          : steps=60, uniform=1.0
  shared LR/LoRA    : 5e-4, layers=2, rank=16
  composition       : reference_delta
  fixed point       : -1.2 / 1.2 / 0.0003
  extra points      : (-1.3,1.2,0.0003), (-1.3,1.3,0.0003)
  total evaluations : 12 (4 training reports + 8 frozen screens)
  gate/alignment    : disabled
  GPUs              : $LAUNCH_GPUS
  resume            : $LAUNCH_RESUME
  selection protocol: retain metrics used only for report/ranking
============================================================
EOF

echo "[1/4] Train/evaluate four V5.12 A1-strength configurations"
env \
    ENV_FILE=/dev/null LOAD_DOTENV=0 \
    MODE=full SPLIT=forget05 GPUS="$LAUNCH_GPUS" \
    UNITS=200 SEED=42 VIEWS=1 EVAL_BS="$LAUNCH_EVAL_BS" \
    CIRU_PATH="$DATA" CF_PATH="$DATA" \
    SWEEP_NAME="$TRAIN_SWEEP" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    MODELS_SWEEP_ROOT="$MODELS_ROOT" RESULTS_DIR="$TRAIN_RESULTS" \
    TASK_PREFIX="tofu_Llama-3.2-1B-Instruct_forget05_F2D_V512_A1STRENGTH4" \
    WEIGHT_A1=-1.2 WEIGHT_A2=1.2 TOP_FILTER=0.0003 \
    COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
    F2R_VARIANT="F2D-PairBudget-v5.12-A1Strength4" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    RESUME="$LAUNCH_RESUME" DRY_RUN="$LAUNCH_DRY_RUN" \
    TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run complete; four training cells were validated."
    exit 0
fi

echo "[2/4] Evaluate two pre-registered frozen points per assistant pair"
JOBS="$SUMMARY_ROOT/screen_jobs.tsv"
: > "$JOBS"
tags=(
    v512_a1s96_u1p25 v512_a1s96_u1p5
    v512_a1s108_u1p25 v512_a1s108_u1p5
)
for tag in "${tags[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$tag" a1strong -1.3 1.2 0.0003 >> "$JOBS"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$tag" symmetric -1.3 1.3 0.0003 >> "$JOBS"
done

run_screen() {
    local gpu="$1" tag="$2" label="$3" w1="$4" w2="$5" filter="$6"
    local name="f2d_v512_a1strength4_${tag}_${label}"
    local result_dir="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${name}"
    local a1_steps a1_uniform
    case "$tag" in
        *a1s96*) a1_steps=96 ;;
        *a1s108*) a1_steps=108 ;;
        *) echo "Unknown A1-strength step tag: $tag" >&2; return 1 ;;
    esac
    case "$tag" in
        *u1p25) a1_uniform=1.25 ;;
        *u1p5) a1_uniform=1.5 ;;
        *) echo "Unknown A1-strength uniform tag: $tag" >&2; return 1 ;;
    esac
    echo "screen_start training=$tag point=$label GPU=$gpu"
    env \
        ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPUS="$gpu" \
        EVAL_BS="$LAUNCH_EVAL_BS" RESUME="$LAUNCH_RESUME" VIEWS=1 \
        CF_PATH="$DATA" MODELS_ROOT="$MODELS_ROOT/$tag" \
        SWEEP_NAME="$name" RESULTS_DIR="$result_dir" \
        WEIGHT_PAIRS="${w1}:${w2}" TOP_FILTERS="$filter" \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_TRAIN_LR=5e-4 A2_TRAIN_LR=5e-4 \
        A1_TRAIN_EP=1 A2_TRAIN_EP=1 \
        A1_TRAIN_STEPS="$a1_steps" A2_TRAIN_STEPS=60 \
        A1_RETAIN_WEIGHT="$a1_uniform" A2_RETAIN_WEIGHT=1.0 \
        F2R_VARIANT="F2D-PairBudget-v5.12-A1Strength4-FrozenScreen" \
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
    gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
    run_screen "$gpu" "$tag" "$label" "$w1" "$w2" "$filter" &
    pids+=("$!")
    index=$((index + 1))
    if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
        wait_batch
    fi
done < "$JOBS"
if [ "${#pids[@]}" -gt 0 ]; then
    wait_batch
fi
if [ "$failures" -gt 0 ]; then
    echo "$failures A1-strength screen job(s) failed; inspect $SUMMARY_ROOT/logs." >&2
    exit 1
fi

echo "[3/4] Consolidate the 12 comparable results"
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
    item["screen_point"] = "fixed"
    rows.append(item)

for line in jobs_path.read_text(encoding="utf-8").splitlines():
    training_tag, label, _w1, _w2, _filter = line.split("\t")
    name = f"f2d_v512_a1strength4_{training_tag}_{label}"
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

if len(rows) != 12:
    raise SystemExit(f"Expected 12 V5.12 A1-strength results; got {len(rows)}")
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
fields = list(dict.fromkeys(
    ["training_tag", "screen_point"] + [key for row in rows for key in row]
))
output.mkdir(parents=True, exist_ok=True)
all_csv = output / "F2R_V512_A1STRENGTH4_ALL.csv"
with all_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
best_path = output / "F2R_V512_A1STRENGTH4_BEST.txt"
with best_path.open("w", encoding="utf-8") as handle:
    for key in (
        "training_tag", "screen_point", "tag", "aggregate_score",
        "memorization_score", "retain_utility_score", "model_utility",
        "weight_a1", "weight_a2", "top_filter", "a1_train_steps",
        "a2_train_steps", "a1_retain_weight", "a2_retain_weight", "report",
    ):
        handle.write(f"{key}={best.get(key, '')}\n")

agg = float(best["aggregate_score"])
print("\n===== V5.12 A1-strength best =====")
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
print(f"relative to V5.12 refdelta baseline 0.527140: {agg - 0.527140:+.6f}")
print(f"relative to FullAnswer 0.553712:              {agg - 0.553712:+.6f}")
print(f"distance to BS-S 0.580000:                   {agg - 0.580000:+.6f}")
print("report =", best["report"])
print("all results =", all_csv)
PY

echo "[4/4] Pure V5.12 A1-strength experiment complete"
echo "Best: $SUMMARY_ROOT/F2R_V512_A1STRENGTH4_BEST.txt"
