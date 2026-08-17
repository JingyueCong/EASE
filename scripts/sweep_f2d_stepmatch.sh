#!/usr/bin/env bash
# Four-GPU equal-step/A2-balance training sweep for audited F2D-40 data.
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

MODE="${MODE:-full}"
SPLIT="${SPLIT:-forget05}"
GPUS="${GPUS:-0 1 2 3}"
UNITS="${UNITS:-40}"
SEED="${SEED:-42}"
SWEEP_NAME="${SWEEP_NAME:-f2d${UNITS}_stepmatch_seed${SEED}}"
CIRU_PATH="${CIRU_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}_strict_v2.jsonl}"
F2D_PATH="${F2D_PATH:-${EASE_ROOT}/ULD/data/f2d/${SPLIT}_f2d${UNITS}_seed${SEED}.jsonl}"
MODELS_SWEEP_ROOT="${MODELS_SWEEP_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2d_1b_${SPLIT}_${MODE}_${SWEEP_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
RESUME="${RESUME:-true}"
EVAL_BS="${EVAL_BS:-4}"

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
    echo "Missing audited CIRU data: $CIRU_PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$F2D_PATH")" "$RESULTS_DIR"
echo "[0/1] Validating strict-v2 units and preparing one frozen F2D mapping"
"$TRAIN_PY" "$EASE_ROOT/scripts/prepare_f2d_from_ciru.py" \
    --input "$CIRU_PATH" --output "$F2D_PATH" --expected-units "$UNITS"

# tag:a1_layers:a2_layers:a1_rank:a2_rank:a1_lr:a2_lr:a1_epochs:a2_epochs:a1_uniform:a2_uniform
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
ep15_a2u5:2:2:16:16:1e-3:1e-3:15:15:5:5 \
ep15_a2u2:2:2:16:16:1e-3:1e-3:15:15:5:2 \
ep15_a2u1:2:2:16:16:1e-3:1e-3:15:15:5:1 \
ep15_a2u0p5:2:2:16:16:1e-3:1e-3:15:15:5:0.5}"

echo "============================================================"
echo "F2D-40 equal-step / A2-balance sweep"
echo "  split / units    : $SPLIT / $UNITS"
echo "  GPUs             : $GPUS"
echo "  frozen F2D data  : $F2D_PATH"
echo "  training         : 15 epochs, A1 uniform=5"
echo "  A2 uniform grid  : 5 / 2 / 1 / 0.5"
echo "  inference point  : -1.2 / 0.4 / 0.0025"
echo "  results          : $RESULTS_DIR"
echo "  protocol         : training_retain_access=false; diagnostic selection=true"
echo "============================================================"

exec env \
    MODE="$MODE" SPLIT="$SPLIT" GPUS="$GPUS" VIEWS=1 \
    SWEEP_NAME="$SWEEP_NAME" TASK_PREFIX="tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2D${UNITS}_train_${SWEEP_NAME}" \
    CF_PATH="$F2D_PATH" MODELS_SWEEP_ROOT="$MODELS_SWEEP_ROOT" \
    RESULTS_DIR="$RESULTS_DIR" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1=-1.2 WEIGHT_A2=0.4 TOP_FILTER=0.0025 \
    F2R_VARIANT="F2D-C01+Placebo-StepMatch" \
    TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    EVAL_BS="$EVAL_BS" SEED="$SEED" RESUME="$RESUME" \
    bash "$EASE_ROOT/scripts/sweep_f2r_training.sh"
