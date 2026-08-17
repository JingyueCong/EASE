#!/usr/bin/env bash
# Four-GPU frozen-artifact sweep of CIRU intervention strength without a gate.
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
ALPHAS="${ALPHAS:-0.5 1.0 1.5 2.0}"
GPUS="${GPUS:-0 1 2 3}"
LAYERS="${LAYERS:-8 12 15}"
RANK="${RANK:-8}"
EVAL_BS="${EVAL_BS:-4}"
TARGET_AGG="${TARGET_AGG:-0.58}"
RESUME="${RESUME:-true}"
SWEEP_NAME="${SWEEP_NAME:-ciru40_strict_v2_nogate_alpha}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2.jsonl}"
ARTIFACT_PATH="${ARTIFACT_PATH:-${EASE_ROOT}/ULD/outputs_trained_models/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2/subspace.npz}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
LOG_DIR="${RESULTS_DIR}/logs"
MANIFEST="${RESULTS_DIR}/manifest.csv"

case "$RESUME" in true|false) ;; *) echo "RESUME must be true or false" >&2; exit 1 ;; esac
if [ ! -s "$DATA_PATH" ]; then
    echo "Missing audited data: $DATA_PATH" >&2
    exit 1
fi
if [ ! -s "$ARTIFACT_PATH" ]; then
    echo "Missing frozen CIRU artifact: $ARTIFACT_PATH" >&2
    exit 1
fi

read -r -a alpha_list <<< "$ALPHAS"
read -r -a gpu_list <<< "$GPUS"
if [ "${#alpha_list[@]}" -gt "${#gpu_list[@]}" ]; then
    echo "This isolated sweep requires at least one GPU per alpha." >&2
    echo "alphas=${#alpha_list[@]}, GPUs=${#gpu_list[@]}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "alpha,gpu,task,report,log" > "$MANIFEST"

echo "============================================================"
echo "CIRU frozen-subspace no-gate alpha sweep"
echo "  split/units : $SPLIT / $UNITS"
echo "  alphas      : $ALPHAS"
echo "  GPUs        : $GPUS"
echo "  data        : $DATA_PATH"
echo "  artifact    : $ARTIFACT_PATH"
echo "  results     : $RESULTS_DIR"
echo "  target Agg  : $TARGET_AGG"
echo "============================================================"

pids=()
labels=()
for index in "${!alpha_list[@]}"; do
    alpha="${alpha_list[$index]}"
    gpu="${gpu_list[$index]}"
    tag="${alpha//-/m}"
    tag="${tag//./p}"
    task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_CIRU${UNITS}_seed${SEED}_strict_v2_nogate_alpha${tag}"
    report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    log="${LOG_DIR}/alpha${tag}.log"
    echo "$alpha,$gpu,$task,$report,$log" >> "$MANIFEST"

    if [ "$RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"aggregate_score"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse alpha=$alpha"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] start alpha=$alpha on GPU $gpu"
    (
        env DATA_PATH="$DATA_PATH" ARTIFACT_PATH="$ARTIFACT_PATH" \
            SPLIT="$SPLIT" UNITS="$UNITS" SEED="$SEED" GPU="$gpu" \
            LAYERS="$LAYERS" RANK="$RANK" ALPHA="$alpha" \
            GATE_ENABLED=false EVAL_BS="$EVAL_BS" EVAL_OVERWRITE=true \
            TASK_NAME="$task" \
            bash "$EASE_ROOT/scripts/run_ciru40_tofu.sh"
        echo "[$(date '+%H:%M:%S')] done alpha=$alpha on GPU $gpu"
    ) > "$log" 2>&1 &
    pids+=("$!")
    labels+=("$alpha")
done

failures=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "alpha=${labels[$index]} failed; inspect $LOG_DIR" >&2
        failures=$((failures + 1))
    fi
done

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_ciru_alpha_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG"

if [ "$failures" -gt 0 ]; then
    echo "$failures alpha job(s) failed" >&2
    exit 1
fi
echo "Sweep complete: $RESULTS_DIR/CIRU_ALPHA_SWEEP.md"
