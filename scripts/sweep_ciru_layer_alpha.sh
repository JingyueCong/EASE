#!/usr/bin/env bash
# Four-GPU layer-specific CIRU intervention-strength sweep on one frozen artifact.
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
LAYERS="${LAYERS:-8 12 15}"
RANK="${RANK:-8}"
GLOBAL_ALPHA="${GLOBAL_ALPHA:-1.45}"
EVAL_BS="${EVAL_BS:-4}"
RESUME="${RESUME:-true}"
TARGET_AGG="${TARGET_AGG:-0.58}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"
SWEEP_NAME="${SWEEP_NAME:-ciru40_strict_v2_layer_alpha_seed42}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2.jsonl}"
ARTIFACT_PATH="${ARTIFACT_PATH:-${EASE_ROOT}/ULD/outputs_trained_models/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2/subspace.npz}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
LOG_DIR="${RESULTS_DIR}/logs"
MANIFEST="${RESULTS_DIR}/manifest.csv"

# tag:alpha8:alpha12:alpha15.  The four defaults test progressively stronger
# shallow/middle interventions while holding the strongest layer fixed.
LAYER_ALPHA_CONFIGS="${LAYER_ALPHA_CONFIGS:-a8_0p50_a12_1p00_a15_1p50:0.50:1.00:1.50 a8_0p75_a12_1p00_a15_1p50:0.75:1.00:1.50 a8_0p75_a12_1p25_a15_1p50:0.75:1.25:1.50 a8_1p00_a12_1p25_a15_1p50:1.00:1.25:1.50}"

case "$RESUME" in true|false) ;; *) echo "RESUME must be true or false" >&2; exit 1 ;; esac
if [ ! -s "$DATA_PATH" ]; then
    echo "Missing audited strict-v2 data: $DATA_PATH" >&2
    exit 1
fi
if [ ! -s "$ARTIFACT_PATH" ]; then
    echo "Missing frozen CIRU artifact: $ARTIFACT_PATH" >&2
    exit 1
fi
if [ "$LAYERS" != "8 12 15" ]; then
    echo "This sweep currently maps alpha columns to LAYERS='8 12 15'; got '$LAYERS'." >&2
    exit 1
fi

read -r -a gpu_list <<< "$GPUS"
read -r -a config_list <<< "$LAYER_ALPHA_CONFIGS"
if [ "${#config_list[@]}" -gt "${#gpu_list[@]}" ]; then
    echo "This isolated sweep requires one GPU per layer-alpha configuration." >&2
    echo "configs=${#config_list[@]}, GPUs=${#gpu_list[@]}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,causal_units,causal_layers,causal_rank,intervention_alpha,intervention_layer_alphas,method_artifact" > "$MANIFEST"

echo "============================================================"
echo "CIRU layer-specific alpha sweep (frozen DiD artifact)"
echo "  split / units : $SPLIT / $UNITS"
echo "  GPUs          : $GPUS"
echo "  layers / rank : $LAYERS / $RANK"
echo "  fallback alpha: $GLOBAL_ALPHA"
echo "  configurations: $LAYER_ALPHA_CONFIGS"
echo "  frozen data   : $DATA_PATH"
echo "  frozen artifact: $ARTIFACT_PATH"
echo "  results       : $RESULTS_DIR"
echo "  protocol      : train=false, diagnostic selection=true"
echo "============================================================"

pids=()
labels=()
for index in "${!config_list[@]}"; do
    config="${config_list[$index]}"
    IFS=: read -r tag alpha8 alpha12 alpha15 extra <<< "$config"
    if [ -z "${alpha15:-}" ] || [ -n "${extra:-}" ]; then
        echo "Invalid LAYER_ALPHA_CONFIGS entry: $config" >&2
        exit 1
    fi
    gpu="${gpu_list[$index]}"
    layer_alphas="8:${alpha8}/12:${alpha12}/15:${alpha15}"
    task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_CIRU${UNITS}_${SWEEP_NAME}_${tag}"
    report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    log="${LOG_DIR}/${tag}.log"
    echo "$tag,,,,${task},${report},${UNITS},${LAYERS},${RANK},${GLOBAL_ALPHA},${layer_alphas},${ARTIFACT_PATH}" >> "$MANIFEST"

    if [ "$RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag (complete report)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] start $tag layer_alphas=$layer_alphas on GPU $gpu"
    (
        env DATA_PATH="$DATA_PATH" ARTIFACT_PATH="$ARTIFACT_PATH" \
            SPLIT="$SPLIT" UNITS="$UNITS" SEED="$SEED" GPU="$gpu" \
            LAYERS="$LAYERS" RANK="$RANK" ALPHA="$GLOBAL_ALPHA" \
            LAYER_ALPHAS="$layer_alphas" GATE_ENABLED=false \
            STOP_AFTER_GENERATION=false EVAL_BS="$EVAL_BS" \
            EVAL_OVERWRITE=true TASK_NAME="$task" \
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
    --sweep-kind ciru-layer-alpha \
    || failures=$((failures + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    best = next(csv.DictReader(handle), None)
if best and best.get("aggregate_score"):
    print(
        "CIRU layer-alpha best: "
        f"{best['tag']} layer_alphas={best.get('intervention_layer_alphas')} "
        f"Agg={best['aggregate_score']} Mem={best['memorization_score']} "
        f"Util={best['retain_utility_score']}"
    )
PY
fi

if [ "$failures" -gt 0 ]; then
    echo "$failures layer-alpha job/summary failure(s)" >&2
    exit 1
fi
echo "Sweep complete: $RESULTS_DIR/F2R_SWEEP.md"
