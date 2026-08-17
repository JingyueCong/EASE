#!/usr/bin/env bash
# One-command CIRU-40: generate fixed 2x2 units, estimate DiD subspace, evaluate.
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
UNITS="${UNITS:-40}"
SEED="${SEED:-42}"
LAYERS="${LAYERS:-8 12 15}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-1.0}"
GATE_ENABLED="${GATE_ENABLED:-true}"
EVAL_BS="${EVAL_BS:-4}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-true}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${CIRU_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"
GEN_PY="${GEN_PY:-${CONDA_BASE}/envs/${TRAIN_ENV:-ease-f2r-train}/bin/python}"
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing CIRU/eval Python: $EVAL_PY" >&2
    exit 1
fi
if [ ! -x "$GEN_PY" ]; then
    echo "Missing generator Python: $GEN_PY" >&2
    exit 1
fi

HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
BASE_MODEL="${BASE_MODEL:-${HF_BASE_PREFIX}_full}"
TOKENIZER="${TOKENIZER:-${HF_BASE_PREFIX}_full}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}.jsonl}"
ARTIFACT_PATH="${ARTIFACT_PATH:-${EASE_ROOT}/ULD/outputs_trained_models/ciru/${SPLIT}_ciru${UNITS}_seed${SEED}/subspace.npz}"
TASK_NAME="${TASK_NAME:-tofu_Llama-3.2-1B-Instruct_${SPLIT}_CIRU${UNITS}_seed${SEED}_a${ALPHA}_g${GATE_ENABLED}}"

CF_PROVIDER="${CF_PROVIDER:-auto}"
if [ "$CF_PROVIDER" = "auto" ]; then
    if [ -n "${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}" ] \
        && [ -n "${OPENAI_API_KEY:-}" ] \
        && [ -n "${GENERATION_MODEL:-${DEFAULT_MODEL:-}}" ]; then
        CF_PROVIDER="azure"
    else
        CF_PROVIDER="deepseek"
    fi
fi
case "$CF_PROVIDER" in
    azure)
        CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-}}}"
        CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
        CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
        CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
        ;;
    deepseek)
        CF_MODEL="${CF_MODEL:-deepseek-v4-flash}"
        CF_BASE_URL="${CF_BASE_URL:-https://api.deepseek.com}"
        CF_API_KEY_ENV="${CF_API_KEY_ENV:-DEEPSEEK_API_KEY}"
        CF_TEMPERATURE="${CF_TEMPERATURE:-0.8}"
        ;;
    *) echo "CF_PROVIDER must be auto, azure, or deepseek" >&2; exit 1 ;;
esac
CF_JSON_MODE="${CF_JSON_MODE:-auto}"

case "$GATE_ENABLED" in true|false) ;; *) echo "GATE_ENABLED must be true/false" >&2; exit 1;; esac
mkdir -p "$(dirname "$DATA_PATH")" "$(dirname "$ARTIFACT_PATH")"

echo "============================================================"
echo "CIRU-40 causal intervention experiment"
echo "  split / units     : $SPLIT / $UNITS"
echo "  GPU               : $GPU"
echo "  factorial data    : $DATA_PATH"
echo "  layers/rank       : $LAYERS / $RANK"
echo "  alpha/gate        : $ALPHA / $GATE_ENABLED"
echo "  generator         : $CF_PROVIDER / $CF_MODEL"
echo "  artifact          : $ARTIFACT_PATH"
echo "  task              : $TASK_NAME"
echo "  retain access     : train=false, selection=false, final-eval=true"
echo "============================================================"

if [ ! -s "$DATA_PATH" ]; then
    if [ -z "${!CF_API_KEY_ENV:-}" ]; then
        echo "$CF_API_KEY_ENV is required to generate CIRU units" >&2
        exit 1
    fi
    echo "[1/3] Generating $UNITS joint 2x2 causal units"
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_ciru40.py" \
        --split "${SPLIT}_perturbed" --output "$DATA_PATH" \
        --units "$UNITS" --block-size 20 --seed "$SEED" \
        --model "$CF_MODEL" --base-url "$CF_BASE_URL" \
        --api-key-env "$CF_API_KEY_ENV" --temperature "$CF_TEMPERATURE" \
        --json-mode "$CF_JSON_MODE" --concurrency 4 --retries 5
else
    echo "[1/3] Reusing audited causal units: $DATA_PATH"
fi

if [ ! -s "$ARTIFACT_PATH" ]; then
    echo "[2/3] Estimating four-cell DiD subspace and retain-free gate"
    read -r -a LAYER_LIST <<< "$LAYERS"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$EVAL_PY" "$EASE_ROOT/scripts/train_ciru_subspace.py" \
        --data "$DATA_PATH" --model "$BASE_MODEL" --tokenizer "$TOKENIZER" \
        --output "$ARTIFACT_PATH" --layers "${LAYER_LIST[@]}" \
        --rank "$RANK" --batch-size 4 --max-length 350
else
    echo "[2/3] Reusing CIRU artifact: $ARTIFACT_PATH"
fi

forget_percent=$((10#${SPLIT#forget}))
RETAIN_SPLIT="retain$(printf '%02d' "$((100 - forget_percent))")"
HF_MODEL_NAME="${HF_MODEL_NAME:-${HF_BASE_PREFIX#open-unlearning/tofu_}}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-${EASE_ROOT}/open-unlearning/saves/eval/tofu_${HF_MODEL_NAME}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
if [ ! -s "$RETAIN_LOGS_PATH" ]; then
    echo "Missing frozen retain reference: $RETAIN_LOGS_PATH" >&2
    echo "Run the existing F2R setup/evaluation once or set RETAIN_LOGS_PATH." >&2
    exit 1
fi

echo "[3/3] Running complete Open-Unlearning TOFU evaluation"
(cd "$EASE_ROOT/open-unlearning" && \
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$EVAL_PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct_CIRU \
        model.model_args.pretrained_model_name_or_path="$BASE_MODEL" \
        model.model_args.ciru_artifact_path="$ARTIFACT_PATH" \
        model.model_args.ciru_alpha="$ALPHA" \
        model.model_args.ciru_gate_enabled="$GATE_ENABLED" \
        model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER" \
        forget_split="$SPLIT" holdout_split="holdout${SPLIT#forget}" \
        eval.tofu.batch_size="$EVAL_BS" eval.tofu.overwrite="$EVAL_OVERWRITE" \
        retain_logs_path="$RETAIN_LOGS_PATH" task_name="$TASK_NAME")

EVAL_DIR="$EASE_ROOT/open-unlearning/saves/eval/$TASK_NAME"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_tofu.py" \
    --eval-json "$EVAL_DIR/TOFU_EVAL.json" \
    --summary-json "$EVAL_DIR/TOFU_SUMMARY.json" \
    --output-dir "$EVAL_DIR" --mode full --split "$SPLIT" \
    --base-model "$BASE_MODEL" --a1-checkpoint not-applicable \
    --a2-checkpoint not-applicable --retain-reference "$RETAIN_LOGS_PATH" \
    --variant "CIRU-${UNITS}" --calibration-path "$ARTIFACT_PATH" \
    --gate-enabled "$GATE_ENABLED" --views 1 \
    --method-artifact "$ARTIFACT_PATH" --causal-units "$UNITS" \
    --causal-layers "$LAYERS" --causal-rank "$RANK" \
    --intervention-alpha "$ALPHA" \
    --selection-retain-access false

echo "Done: $EVAL_DIR/F2R_REPORT.md"
