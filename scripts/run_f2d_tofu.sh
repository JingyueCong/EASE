#!/usr/bin/env bash
# F2D: feed CIRU C01 pseudo-retain and C10/C00 placebos to F2R dual assistants.
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
GPU="${GPU:-0}"
SEED="${SEED:-42}"
UNITS="${UNITS:-40}"
MODE="${MODE:-full}"
CIRU_PATH="${CIRU_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2.jsonl}"
F2D_PATH="${F2D_PATH:-${EASE_ROOT}/ULD/data/f2d/${SPLIT}_f2d${UNITS}_seed${SEED}.jsonl}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2d_1b_${SPLIT}_${MODE}_ciru${UNITS}_seed${SEED}}"
TASK_NAME="${TASK_NAME:-tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2D${UNITS}_seed${SEED}}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
TRAIN_PY="${TRAIN_PY:-${CONDA_BASE}/envs/${TRAIN_ENV:-ease-f2r-train}/bin/python}"
if [ ! -x "$TRAIN_PY" ]; then
    echo "Missing training Python: $TRAIN_PY" >&2
    exit 1
fi
if [ ! -s "$CIRU_PATH" ]; then
    echo "Missing audited CIRU dataset: $CIRU_PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$F2D_PATH")" "$MODELS_ROOT"
echo "[F2D 0/4] Validating and mapping CIRU cells to dual-assistant roles"
"$TRAIN_PY" "$EASE_ROOT/scripts/prepare_f2d_from_ciru.py" \
    --input "$CIRU_PATH" --output "$F2D_PATH" --expected-units "$UNITS"

echo "============================================================"
echo "F2D: factorial counterfactuals with dual assistants"
echo "  CIRU units        : $CIRU_PATH"
echo "  mapped supervision: $F2D_PATH"
echo "  roles             : C11=forget, C01=pseudo-retain, C10/C00=uniform"
echo "  training counts   : full ${SPLIT} forget + ${UNITS} pseudo-retain; $((2 * UNITS)) controls"
echo "  assistants        : layers=${NUM_LAYER:-2}, LoRA-r=${LORA_R:-16}"
echo "  optimization      : lr=${TRAIN_LR:-1e-3}, epochs=${TRAIN_EP:-5}"
echo "  weights/filter    : ${WEIGHT_A1:--1.2} / ${WEIGHT_A2:-0.4} / ${TOP_FILTER:-0.0025}"
echo "  protocol          : training_retain_access=false, selection_retain_access=${SELECTION_RETAIN_ACCESS:-true}"
echo "  causal claim      : causal construction ablation; dual-assistant estimator (not DiD)"
echo "============================================================"

exec env \
    MODE="$MODE" SPLIT="$SPLIT" GPU="$GPU" VIEWS=1 \
    CF_PATH="$F2D_PATH" MODELS_ROOT="$MODELS_ROOT" TASK_NAME="$TASK_NAME" \
    F2R_VARIANT="F2D-C01+Placebo" \
    NUM_LAYER="${NUM_LAYER:-2}" LORA_R="${LORA_R:-16}" \
    TRAIN_LR="${TRAIN_LR:-1e-3}" TRAIN_EP="${TRAIN_EP:-5}" \
    RETAIN_WEIGHT="${RETAIN_WEIGHT:-5}" \
    WEIGHT_A1="${WEIGHT_A1:--1.2}" WEIGHT_A2="${WEIGHT_A2:-0.4}" \
    TOP_FILTER="${TOP_FILTER:-0.0025}" \
    SELECTION_RETAIN_ACCESS="${SELECTION_RETAIN_ACCESS:-true}" \
    bash "$EASE_ROOT/scripts/run_f2r_tofu.sh"
