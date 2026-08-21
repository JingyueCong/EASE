#!/usr/bin/env bash
# Deep, resumable V5.11 assistant-training + frozen-inference sweep.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_EVAL_BS="${EVAL_BS:-4}"
REQUESTED_RESUME="${RESUME:-true}"
REQUESTED_DRY_RUN="${DRY_RUN:-false}"
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
DATA="${F2D_V511_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget05_author_premisefix200_seed42_v5_11_manualfix_v1_full.jsonl}"
TRAIN_SWEEP="${F2D_V511_TRAIN_SWEEP:-f2d_v511_deep_train_seed42}"
TRAIN_RESULTS="${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${TRAIN_SWEEP}"
MODELS_ROOT="${EASE_ROOT}/ULD/outputs_trained_models/f2d_did_1b_${SPLIT}_${TRAIN_SWEEP}"
SUMMARY_ROOT="${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_f2d_v511_deep_search_seed42"

if [ ! -s "$DATA" ]; then
    echo "Missing frozen V5.11 data: $DATA" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "Missing evaluation Python: $PY" >&2
    exit 1
fi

configs=()
for steps in 24 32 40 48 56 64; do
    for spec in 4e4:4e-4 6e4:6e-4 8e4:8e-4 1e3:1e-3; do
        lr_tag="${spec%%:*}"
        lr="${spec#*:}"
        configs+=(
            "v511_d_s${steps}_lr${lr_tag}:2:2:16:16:${lr}:${lr}:1:1:1:1:${steps}:${steps}"
        )
    done
done

for pair in 32:40 40:32 40:48 48:40; do
    a1_steps="${pair%%:*}"
    a2_steps="${pair#*:}"
    for spec in 6e4:6e-4 8e4:8e-4; do
        lr_tag="${spec%%:*}"
        lr="${spec#*:}"
        configs+=(
            "v511_d_a${a1_steps}_${a2_steps}_lr${lr_tag}:2:2:16:16:${lr}:${lr}:1:1:1:1:${a1_steps}:${a2_steps}"
        )
    done
done

for steps in 40 48; do
    configs+=(
        "v511_d_s${steps}_lr8e4_u0p75:2:2:16:16:8e-4:8e-4:1:1:0.75:0.75:${steps}:${steps}"
        "v511_d_s${steps}_lr8e4_u1p25:2:2:16:16:8e-4:8e-4:1:1:1.25:1.25:${steps}:${steps}"
    )
done

TRAIN_CONFIGS="${configs[*]}"
COARSE_A1="${F2D_V511_COARSE_A1:--0.8 -1.3 -1.8}"
COARSE_A2="${F2D_V511_COARSE_A2:-0.8 1.3 1.8}"
COARSE_FILTERS="${F2D_V511_COARSE_FILTERS:-0.0001 0.0004}"

mkdir -p "$SUMMARY_ROOT"

cat <<EOF
============================================================
F2D V5.11 deep parameter search
  frozen data       : $DATA
  GPUs              : $REQUESTED_GPUS
  training configs  : ${#configs[@]}
  coarse/config     : 3 x 3 x 2 = 18 evaluations
  coarse total      : $((${#configs[@]} * 18)) evaluations
  fine finalists    : 2 x 75 evaluations
  total evaluations : $((${#configs[@]} * 18 + 150))
  resume            : $REQUESTED_RESUME
============================================================
EOF

if [ "$REQUESTED_DRY_RUN" = "true" ]; then
    printf '%s\n' "${configs[@]}"
    exit 0
fi

echo "[1/4] Train and evaluate 36 V5.11 assistant pairs"
env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
    UNITS=200 SEED="$SEED" VIEWS=1 EVAL_BS="$REQUESTED_EVAL_BS" \
    CIRU_PATH="$DATA" CF_PATH="$DATA" \
    SWEEP_NAME="$TRAIN_SWEEP" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    MODELS_SWEEP_ROOT="$MODELS_ROOT" RESULTS_DIR="$TRAIN_RESULTS" \
    TASK_PREFIX="tofu_Llama-3.2-1B-Instruct_forget05_F2D_V511_DEEP" \
    WEIGHT_A1=-1.3 WEIGHT_A2=1.3 TOP_FILTER=0.0002 \
    F2R_VARIANT="F2D-PremiseFix-v5.11-DeepTrain" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    RESUME="$REQUESTED_RESUME" TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"

echo "[2/4] Coarse frozen-inference sweep for every trained pair"
for config in "${configs[@]}"; do
    tag="${config%%:*}"
    model_root="${MODELS_ROOT}/${tag}"
    infer_name="f2d_v511_deep_coarse_${tag}"
    echo "coarse_start training_tag=$tag"
    env \
        LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
        EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
        CF_PATH="$DATA" MODELS_ROOT="$model_root" \
        SWEEP_NAME="$infer_name" \
        RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/${SPLIT}_${infer_name}" \
        WEIGHT_A1_GRID="$COARSE_A1" WEIGHT_A2_GRID="$COARSE_A2" \
        TOP_FILTERS="$COARSE_FILTERS" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
        bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
    echo "coarse_done training_tag=$tag"
done

echo "[3/4] Select two finalists and run local 5 x 5 x 3 inference grids"
SELECTION="$SUMMARY_ROOT/coarse_top2.tsv"
"$PY" - "$EASE_ROOT" "$SELECTION" <<'PY'
import csv
import glob
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
pattern = str(
    root / "open-unlearning/saves/sweeps/forget05_f2d_v511_deep_coarse_*"
    / "F2R_SWEEP.csv"
)
best_by_training = []
prefix = "forget05_f2d_v511_deep_coarse_"
for name in sorted(glob.glob(pattern)):
    rows = [
        row for row in csv.DictReader(open(name, encoding="utf-8"))
        if row.get("aggregate_score")
    ]
    if not rows:
        continue
    best = max(rows, key=lambda row: float(row["aggregate_score"]))
    directory = Path(name).parent.name
    tag = directory[len(prefix):]
    best_by_training.append((float(best["aggregate_score"]), tag, best))

best_by_training.sort(reverse=True)
if len(best_by_training) < 2:
    raise SystemExit(
        f"Need two complete coarse training configurations; got {len(best_by_training)}"
    )
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for agg, tag, row in best_by_training[:2]:
        handle.write(
            "\t".join((
                tag,
                row["weight_a1"],
                row["weight_a2"],
                row["top_filter"],
                f"{agg:.12g}",
            )) + "\n"
        )
PY

while IFS=$'\t' read -r tag best_w1 best_w2 best_filter coarse_agg; do
    mapfile -t fine_grid < <("$PY" - "$best_w1" "$best_w2" "$best_filter" <<'PY'
from decimal import Decimal
import sys

def fmt(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text

w1, w2, filt = map(Decimal, sys.argv[1:])
print(" ".join(fmt(w1 + Decimal(i) / 10) for i in (-2, -1, 0, 1, 2)))
print(" ".join(fmt(w2 + Decimal(i) / 10) for i in (-2, -1, 0, 1, 2)))
print(" ".join(fmt(filt * factor) for factor in (
    Decimal("0.5"), Decimal("1"), Decimal("2")
)))
PY
    )
    fine_a1="${fine_grid[0]}"
    fine_a2="${fine_grid[1]}"
    fine_filters="${fine_grid[2]}"
    fine_name="f2d_v511_deep_fine_${tag}"
    echo "fine_start training_tag=$tag coarse_agg=$coarse_agg"
    env \
        LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$REQUESTED_GPUS" \
        EVAL_BS="$REQUESTED_EVAL_BS" RESUME="$REQUESTED_RESUME" \
        CF_PATH="$DATA" MODELS_ROOT="${MODELS_ROOT}/${tag}" \
        SWEEP_NAME="$fine_name" \
        RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/${SPLIT}_${fine_name}" \
        WEIGHT_A1_GRID="$fine_a1" WEIGHT_A2_GRID="$fine_a2" \
        TOP_FILTERS="$fine_filters" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
        bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
    echo "fine_done training_tag=$tag"
done < "$SELECTION"

echo "[4/4] Consolidate all coarse and fine results"
"$PY" - "$EASE_ROOT" "$SUMMARY_ROOT" <<'PY'
import csv
import glob
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
patterns = [
    root / "open-unlearning/saves/sweeps/forget05_f2d_v511_deep_coarse_*" / "F2R_SWEEP.csv",
    root / "open-unlearning/saves/sweeps/forget05_f2d_v511_deep_fine_*" / "F2R_SWEEP.csv",
]
rows = []
for pattern in patterns:
    for name in glob.glob(str(pattern)):
        stage_dir = Path(name).parent.name
        stage = "fine" if "_fine_" in stage_dir else "coarse"
        marker = "_fine_" if stage == "fine" else "_coarse_"
        training_tag = stage_dir.split(marker, 1)[1]
        for row in csv.DictReader(open(name, encoding="utf-8")):
            if not row.get("aggregate_score"):
                continue
            item = dict(row)
            item["search_stage"] = stage
            item["training_tag"] = training_tag
            rows.append(item)

if not rows:
    raise SystemExit("No deep-search inference results found")
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
fields = ["search_stage", "training_tag"] + list(rows[0])
fields = list(dict.fromkeys(fields))
output.mkdir(parents=True, exist_ok=True)
with (output / "F2R_DEEP_ALL.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
with (output / "F2R_DEEP_BEST.txt").open("w", encoding="utf-8") as handle:
    for key in (
        "search_stage", "training_tag", "tag", "aggregate_score",
        "memorization_score", "retain_utility_score", "model_utility",
        "weight_a1", "weight_a2", "top_filter", "report",
    ):
        handle.write(f"{key}={best.get(key, '')}\n")
print(
    "V5.11 deep best: "
    f"training={best['training_tag']} stage={best['search_stage']} "
    f"Agg={best['aggregate_score']} Mem={best['memorization_score']} "
    f"Util={best['retain_utility_score']} w1={best['weight_a1']} "
    f"w2={best['weight_a2']} filter={best['top_filter']}"
)
PY

echo "Deep-search best: $SUMMARY_ROOT/F2R_DEEP_BEST.txt"
echo "All results    : $SUMMARY_ROOT/F2R_DEEP_ALL.csv"
