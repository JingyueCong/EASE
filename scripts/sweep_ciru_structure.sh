#!/usr/bin/env bash
# Four-GPU CIRU layer/rank localization sweep with frozen factorial data.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
if [ "${LOAD_DOTENV:-1}" = "1" ] && [ -f "$ENV_FILE" ]; then
    echo "Loading environment: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

SPLIT="${SPLIT:-forget05}"
UNITS="${UNITS:-40}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0 1 2 3}"
ALPHA="${ALPHA:-1.5}"
EVAL_BS="${EVAL_BS:-4}"
RESUME="${RESUME:-true}"
TARGET_AGG="${TARGET_AGG:-0.58}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"
ALPHA_TAG="${ALPHA//-/m}"
ALPHA_TAG="${ALPHA_TAG//./p}"
SWEEP_NAME="${SWEEP_NAME:-ciru${UNITS}_structure_alpha${ALPHA_TAG}_seed${SEED}}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2.jsonl}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/ciru/${SPLIT}_${SWEEP_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
LOG_DIR="${RESULTS_DIR}/logs"
MANIFEST="${RESULTS_DIR}/manifest.csv"

# tag:comma-separated-layers:rank
STRUCTURE_CONFIGS="${STRUCTURE_CONFIGS:-layer8_r8:8:8 layer12_r8:12:8 layer15_r8:15:8 multi_r4:8,12,15:4}"

case "$RESUME" in true|false) ;; *) echo "RESUME must be true or false" >&2; exit 1 ;; esac
if [ ! -s "$DATA_PATH" ]; then
    echo "Missing audited strict-v2 data: $DATA_PATH" >&2
    exit 1
fi

read -r -a gpu_list <<< "$GPUS"
read -r -a config_list <<< "$STRUCTURE_CONFIGS"
if [ "${#config_list[@]}" -gt "${#gpu_list[@]}" ]; then
    echo "This isolated sweep requires at least one GPU per structure." >&2
    echo "structures=${#config_list[@]}, GPUs=${#gpu_list[@]}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$ARTIFACT_ROOT"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,causal_units,causal_layers,causal_rank,intervention_alpha,method_artifact" > "$MANIFEST"

echo "============================================================"
echo "CIRU layer/rank localization sweep"
echo "  split / units : $SPLIT / $UNITS"
echo "  GPUs          : $GPUS"
echo "  alpha / gate  : $ALPHA / false"
echo "  configurations: $STRUCTURE_CONFIGS"
echo "  frozen data   : $DATA_PATH"
echo "  artifacts     : $ARTIFACT_ROOT"
echo "  results       : $RESULTS_DIR"
echo "  protocol      : train=false, selection=false, final-eval=true"
echo "============================================================"

pids=()
labels=()
for index in "${!config_list[@]}"; do
    config="${config_list[$index]}"
    IFS=: read -r tag layer_csv rank <<< "$config"
    if [ -z "${rank:-}" ]; then
        echo "Invalid STRUCTURE_CONFIGS entry: $config" >&2
        exit 1
    fi
    gpu="${gpu_list[$index]}"
    layers="${layer_csv//,/ }"
    artifact="${ARTIFACT_ROOT}/${tag}/subspace.npz"
    task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_CIRU${UNITS}_${SWEEP_NAME}_${tag}"
    report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    log="${LOG_DIR}/${tag}.log"
    echo "$tag,,,,${task},${report},${UNITS},${layers},${rank},${ALPHA},${artifact}" >> "$MANIFEST"

    if [ "$RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag (complete report)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] start $tag layers=[$layers] rank=$rank on GPU $gpu"
    (
        env DATA_PATH="$DATA_PATH" ARTIFACT_PATH="$artifact" \
            SPLIT="$SPLIT" UNITS="$UNITS" SEED="$SEED" GPU="$gpu" \
            LAYERS="$layers" RANK="$rank" ALPHA="$ALPHA" \
            GATE_ENABLED=false STOP_AFTER_GENERATION=false \
            EVAL_BS="$EVAL_BS" EVAL_OVERWRITE=true TASK_NAME="$task" \
            bash "$EASE_ROOT/scripts/run_ciru40_tofu.sh"
        if [ ! -s "$report" ] \
            || ! grep -q '"forget_truth_ratio_knowledge"' "$report"; then
            echo "$tag finished without a complete F2R report: $report" >&2
            exit 1
        fi
        echo "[$(date '+%H:%M:%S')] done $tag on GPU $gpu"
    ) > "$log" 2>&1 &
    pids+=("$!")
    labels+=("$tag")
done

failures=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "${labels[$index]} failed; inspect $LOG_DIR" >&2
        failures=$((failures + 1))
    fi
done

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin "$TARGET_MARGIN" \
    --sweep-kind ciru-structure \
    || failures=$((failures + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    best = next(csv.DictReader(handle), None)
if best and best.get("aggregate_score"):
    print(
        "CIRU structure best: "
        f"{best['tag']} layers={best.get('causal_layers')} "
        f"rank={best.get('causal_rank')} Agg={best['aggregate_score']} "
        f"Mem={best['memorization_score']} Util={best['retain_utility_score']}"
    )
PY
fi

if [ "$failures" -gt 0 ]; then
    echo "$failures structure job/summary failure(s)" >&2
    exit 1
fi
echo "Sweep complete: $RESULTS_DIR/F2R_SWEEP.md"
