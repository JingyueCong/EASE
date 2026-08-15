#!/usr/bin/env bash
# One-command F2R experiment: generate forget-conditioned counterfactuals,
# train A1/A2, then evaluate the frozen DualULD model with TOFU.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
TRAIN_ENV="${TRAIN_ENV:-ease-f2r-train}"
EVAL_ENV="${EVAL_ENV:-ease-f2r-eval}"
TRAIN_PY="${TRAIN_PY:-${CONDA_BASE}/envs/${TRAIN_ENV}/bin/python}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV}/bin/python}"

MODE="${MODE:-smoke}"                 # smoke | full
SPLIT="${SPLIT:-forget05}"            # forget01 | forget05 | forget10
GPU="${GPU:-0}"
VIEWS="${VIEWS:-2}"
CF_MODEL="${CF_MODEL:-deepseek-v4-flash}"
CF_BASE_URL="${CF_BASE_URL:-https://api.deepseek.com}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CF_ROOT="${CF_ROOT:-${EASE_ROOT}/ULD/data/f2r}"
CF_PATH="${CF_PATH:-${CF_ROOT}/${SPLIT}_${MODE}.jsonl}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}}"
TASK_NAME="${TASK_NAME:-tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_${MODE}}"

HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-${HF_BASE_PREFIX}_full}"
NUM_LAYER="${NUM_LAYER:-2}"
LORA_R="${LORA_R:-16}"
WEIGHT_A1="${WEIGHT_A1:--1.0}"
WEIGHT_A2="${WEIGHT_A2:-1.0}"
TOP_FILTER="${TOP_FILTER:-0.01}"
TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_OPTIM="${TRAIN_OPTIM:-adamw_torch}"
EVAL_BS="${EVAL_BS:-4}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-true}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-auto}"
AUTO_FETCH_RETAIN_LOGS="${AUTO_FETCH_RETAIN_LOGS:-1}"
HF_ENDPOINT_SETTING="${HF_ENDPOINT:-auto}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
HF_PREFLIGHT="${HF_PREFLIGHT:-1}"

case "$MODE" in
    smoke)
        CF_LIMIT="${CF_LIMIT:-8}"
        TRAIN_EP="${TRAIN_EP:-1}"
        ;;
    full)
        CF_LIMIT="${CF_LIMIT:-0}"
        TRAIN_EP="${TRAIN_EP:-10}"
        ;;
    *) echo "MODE must be smoke or full (got: $MODE)" >&2; exit 1 ;;
esac

for executable in "$TRAIN_PY" "$EVAL_PY"; do
    if [ ! -x "$executable" ]; then
        echo "Missing Python environment: $executable" >&2
        echo "Run: bash scripts/setup_f2r_server.sh" >&2
        exit 1
    fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found: this experiment requires a CUDA server." >&2
    exit 1
fi

mkdir -p "$CF_ROOT" "$MODELS_ROOT"
export EASE_ROOT
export PYTHONPATH="${EASE_ROOT}/ULD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

hf_preflight() {
    local endpoint="$1"
    echo "      Checking $endpoint"
    HF_ENDPOINT="$endpoint" "$TRAIN_PY" - \
        "${SPLIT}_perturbed" "${HF_BASE_PREFIX}_full" "$HF_TOKENIZER" <<'PY'
import sys

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

config, model_id, tokenizer_id = sys.argv[1:]
try:
    dataset = load_dataset("locuslab/TOFU", config)["train"]
    hf_hub_download(repo_id=model_id, filename="config.json")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
except Exception as exc:
    print(f"      {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(
    f"      OK: locuslab/TOFU/{config} ({len(dataset)} rows), "
    f"{model_id}, tokenizer={tokenizer.__class__.__name__}"
)
PY
}

select_hf_endpoint() {
    if [ "$HF_PREFLIGHT" = "0" ]; then
        if [ "$HF_ENDPOINT_SETTING" = "auto" ]; then
            export HF_ENDPOINT="https://huggingface.co"
        else
            export HF_ENDPOINT="$HF_ENDPOINT_SETTING"
        fi
        echo "[0/4] Hugging Face preflight skipped (HF_ENDPOINT=$HF_ENDPOINT)"
        return
    fi

    echo "[0/4] Checking Hugging Face dataset/model access"
    if [ "$HF_ENDPOINT_SETTING" != "auto" ]; then
        if hf_preflight "$HF_ENDPOINT_SETTING"; then
            export HF_ENDPOINT="$HF_ENDPOINT_SETTING"
            return
        fi
        echo "Hugging Face preflight failed for HF_ENDPOINT=$HF_ENDPOINT_SETTING" >&2
        echo "Check the endpoint, proxy, HF_TOKEN, and tokenizer versions; use HF_PREFLIGHT=0 only when all artifacts are cached and parse correctly." >&2
        exit 1
    fi

    local endpoint
    for endpoint in "https://huggingface.co" "$HF_MIRROR_ENDPOINT"; do
        if hf_preflight "$endpoint"; then
            export HF_ENDPOINT="$endpoint"
            echo "      Selected HF_ENDPOINT=$HF_ENDPOINT"
            return
        fi
    done
    echo "Could not load TOFU, the base-model config, and tokenizer through either Hugging Face endpoint." >&2
    echo "Set a working proxy/endpoint, e.g. HF_ENDPOINT=https://hf-mirror.com, and rerun." >&2
    echo "Use HF_PREFLIGHT=0 only if both the dataset and model are already cached locally." >&2
    exit 1
}

select_hf_endpoint

echo "============================================================"
echo "F2R TOFU experiment"
echo "  mode/split       : $MODE / $SPLIT"
echo "  GPU              : $GPU"
echo "  counterfactuals  : $CF_PATH (views=$VIEWS, limit=$CF_LIMIT)"
echo "  CF API/JSON mode : $CF_MODEL / $CF_JSON_MODE (key=$CF_API_KEY_ENV)"
echo "  assistants       : layers=$NUM_LAYER, LoRA-r=$LORA_R"
echo "  weights/filter   : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  optimizer        : $TRAIN_OPTIM"
echo "  eval overwrite   : $EVAL_OVERWRITE"
echo "  Hugging Face     : $HF_ENDPOINT"
echo "============================================================"

if [ ! -s "$CF_PATH" ]; then
    if [ -z "${!CF_API_KEY_ENV:-}" ]; then
        echo "$CF_API_KEY_ENV is required to generate $CF_PATH" >&2
        exit 1
    fi
    echo "[1/4] Generating matched counterfactual supervision"
    limit_args=()
    if [ "$CF_LIMIT" -gt 0 ]; then limit_args=(--limit "$CF_LIMIT"); fi
    "$TRAIN_PY" "$EASE_ROOT/ULD/scripts/generate_f2r_pairs.py" \
        --split "${SPLIT}_perturbed" \
        --output "$CF_PATH" \
        --backend openai \
        --model "$CF_MODEL" \
        --base-url "$CF_BASE_URL" \
        --api-key-env "$CF_API_KEY_ENV" \
        --json-mode "$CF_JSON_MODE" \
        --views "$VIEWS" \
        "${limit_args[@]}"
else
    echo "[1/4] Reusing counterfactuals: $CF_PATH"
fi

train_role() {
    local role="$1"
    local output_root="${MODELS_ROOT}/${role}"
    if find "$output_root" -name 'checkpoint-*' -type d 2>/dev/null | grep -q .; then
        echo "      Reusing existing $role checkpoint under $output_root"
        return
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$TRAIN_PY" "$EASE_ROOT/ULD/scripts/hf_forget_train.py" \
        project="f2r_${role}_${SPLIT}" \
        data=tofu_chat3 \
        data.dataset.split="${SPLIT}_perturbed" \
        data_mode="f2r_${role}" \
        data_mode.counterfactual_path="$CF_PATH" \
        model=llama-3-1b \
        model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="$HF_TOKENIZER" \
        model_mode=uld \
        model_mode.num_layer="$NUM_LAYER" \
        model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" \
        trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" \
        trainer.optim="$TRAIN_OPTIM" \
        trainer.max_epochs="$TRAIN_EP" \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$output_root" \
        postfix="$role" \
        "hydra.run.dir=outputs/tune_log/f2r_${role}_${SPLIT}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo "[2/4] Training A1"
(cd "$EASE_ROOT/ULD" && train_role a1)
echo "[3/4] Training A2"
(cd "$EASE_ROOT/ULD" && train_role a2)

latest_checkpoint() {
    find "$1" -name 'checkpoint-*' -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-
}
A1_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a1")"
A2_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a2")"
if [ -z "$A1_CKPT" ] || [ -z "$A2_CKPT" ]; then
    echo "Could not resolve both assistant checkpoints." >&2
    exit 1
fi

forget_percent=$((10#${SPLIT#forget}))
RETAIN_SPLIT="retain$(printf '%02d' "$((100 - forget_percent))")"
HF_MODEL_NAME="${HF_MODEL_NAME:-${HF_BASE_PREFIX#open-unlearning/tofu_}}"
REFERENCE_REL="tofu_${HF_MODEL_NAME}_${RETAIN_SPLIT}/TOFU_EVAL.json"
REFERENCE_DEFAULT="$EASE_ROOT/open-unlearning/saves/eval/$REFERENCE_REL"
if [ "$RETAIN_LOGS_PATH" = "auto" ]; then
    RETAIN_LOGS_PATH="$REFERENCE_DEFAULT"
    if [ ! -f "$RETAIN_LOGS_PATH" ] && [ "$AUTO_FETCH_RETAIN_LOGS" = "1" ]; then
        echo "      Downloading frozen retain reference: $REFERENCE_REL"
        "$EVAL_PY" - "$REFERENCE_REL" "$EASE_ROOT/open-unlearning/saves/eval" <<'PY'
import sys
from huggingface_hub import snapshot_download

relative_path, output_dir = sys.argv[1:]
snapshot_download(
    repo_id="open-unlearning/eval",
    repo_type="dataset",
    allow_patterns=[relative_path],
    local_dir=output_dir,
)
PY
    fi
fi

echo "[4/4] Evaluating frozen F2R model"
retain_arg="retain_logs_path=null"
if [ "$RETAIN_LOGS_PATH" != "null" ]; then
    if [ ! -f "$RETAIN_LOGS_PATH" ]; then
        echo "Missing retain reference log: $RETAIN_LOGS_PATH" >&2
        echo "Set RETAIN_LOGS_PATH=null only for an explicitly incomplete diagnostic evaluation." >&2
        exit 1
    fi
    "$EVAL_PY" - "$RETAIN_LOGS_PATH" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    logs = json.load(handle)

required = ("forget_truth_ratio", "mia_min_k")
bad = []
for name in required:
    metric = logs.get(name)
    value = metric.get("agg_value") if isinstance(metric, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        bad.append(name)
if bad:
    raise SystemExit(
        f"Retain reference {path} is incomplete; missing/invalid: {', '.join(bad)}"
    )
print(f"      Valid retain reference: {path}")
PY
    retain_arg="retain_logs_path=$RETAIN_LOGS_PATH"
fi
(cd "$EASE_ROOT/open-unlearning" && \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$EVAL_PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$A1_CKPT" \
        model.model_args.a2_path="$A2_CKPT" \
        model.model_args.weight_a1="$WEIGHT_A1" \
        model.model_args.weight_a2="$WEIGHT_A2" \
        model.model_args.top_logit_filter="$TOP_FILTER" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$SPLIT" \
        holdout_split="holdout${SPLIT#forget}" \
        eval.tofu.batch_size="$EVAL_BS" \
        eval.tofu.overwrite="$EVAL_OVERWRITE" \
        "$retain_arg" \
        task_name="$TASK_NAME")

SUMMARY="$EASE_ROOT/open-unlearning/saves/eval/$TASK_NAME/TOFU_SUMMARY.json"
EVAL_DIR="$EASE_ROOT/open-unlearning/saves/eval/$TASK_NAME"
EVAL_JSON="$EVAL_DIR/TOFU_EVAL.json"
if [ ! -f "$SUMMARY" ] || [ ! -f "$EVAL_JSON" ]; then
    echo "Evaluation finished without both expected TOFU output files." >&2
    exit 1
fi
summary_args=()
if [ "$RETAIN_LOGS_PATH" = "null" ]; then
    summary_args=(--allow-incomplete)
fi
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_tofu.py" \
    --eval-json "$EVAL_JSON" \
    --summary-json "$SUMMARY" \
    --output-dir "$EVAL_DIR" \
    --mode "$MODE" \
    --split "$SPLIT" \
    --base-model "${HF_BASE_PREFIX}_full" \
    --a1-checkpoint "$A1_CKPT" \
    --a2-checkpoint "$A2_CKPT" \
    --retain-reference "$RETAIN_LOGS_PATH" \
    "${summary_args[@]}"
echo "Done. Full report: $EVAL_DIR/F2R_REPORT.md"
echo "      EASE table: $EVAL_DIR/F2R_EASE_TABLE.md"
