#!/usr/bin/env bash
# One-command F2R experiment: generate forget-conditioned counterfactuals,
# train A1/A2, then evaluate the frozen DualULD model with TOFU.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
if [ "${LOAD_DOTENV:-1}" = "1" ] && [ -f "$ENV_FILE" ]; then
    echo "Loading environment: $ENV_FILE"
    set -a
    # .env is a trusted, shell-compatible local file and is ignored by Git.
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

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
    *) echo "CF_PROVIDER must be auto, azure, or deepseek (got: $CF_PROVIDER)" >&2; exit 1 ;;
esac

if [ -z "$CF_MODEL" ] || [ -z "$CF_BASE_URL" ]; then
    echo "Incomplete $CF_PROVIDER counterfactual API configuration." >&2
    echo "Set CF_MODEL/CF_BASE_URL or the provider-specific .env variables." >&2
    exit 1
fi
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_ROOT="${CF_ROOT:-${EASE_ROOT}/ULD/data/f2r}"
CF_PATH="${CF_PATH:-${CF_ROOT}/${SPLIT}_${MODE}.jsonl}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}}"
TASK_NAME="${TASK_NAME:-tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_${MODE}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-$(basename "$MODELS_ROOT")}"

HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-${HF_BASE_PREFIX}_full}"
NUM_LAYER="${NUM_LAYER:-2}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WEIGHT_A1="${WEIGHT_A1:--1.0}"
WEIGHT_A2="${WEIGHT_A2:-1.0}"
TOP_FILTER="${TOP_FILTER:-0.01}"
TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_OPTIM="${TRAIN_OPTIM:-adamw_torch}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-5.0}"
SEED="${SEED:-42}"
EVAL_BS="${EVAL_BS:-4}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-true}"
SELECTION_RETAIN_ACCESS="${SELECTION_RETAIN_ACCESS:-false}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-auto}"
AUTO_FETCH_RETAIN_LOGS="${AUTO_FETCH_RETAIN_LOGS:-1}"
HF_ENDPOINT_SETTING="${HF_ENDPOINT:-auto}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
HF_PREFLIGHT="${HF_PREFLIGHT:-1}"

case "$SELECTION_RETAIN_ACCESS" in
    true|false) ;;
    *) echo "SELECTION_RETAIN_ACCESS must be true or false (got: $SELECTION_RETAIN_ACCESS)" >&2; exit 1 ;;
esac

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

# Every shared training parameter can be overridden per assistant. Keeping
# these explicit prevents accidental coupling in asymmetric-control studies.
A1_NUM_LAYER="${A1_NUM_LAYER:-$NUM_LAYER}"
A2_NUM_LAYER="${A2_NUM_LAYER:-$NUM_LAYER}"
A1_LORA_R="${A1_LORA_R:-$LORA_R}"
A2_LORA_R="${A2_LORA_R:-$LORA_R}"
A1_LORA_ALPHA="${A1_LORA_ALPHA:-${LORA_ALPHA:-$((2 * A1_LORA_R))}}"
A2_LORA_ALPHA="${A2_LORA_ALPHA:-${LORA_ALPHA:-$((2 * A2_LORA_R))}}"
A1_LORA_DROPOUT="${A1_LORA_DROPOUT:-$LORA_DROPOUT}"
A2_LORA_DROPOUT="${A2_LORA_DROPOUT:-$LORA_DROPOUT}"
A1_TRAIN_LR="${A1_TRAIN_LR:-$TRAIN_LR}"
A2_TRAIN_LR="${A2_TRAIN_LR:-$TRAIN_LR}"
A1_TRAIN_EP="${A1_TRAIN_EP:-$TRAIN_EP}"
A2_TRAIN_EP="${A2_TRAIN_EP:-$TRAIN_EP}"
A1_RETAIN_WEIGHT="${A1_RETAIN_WEIGHT:-$RETAIN_WEIGHT}"
A2_RETAIN_WEIGHT="${A2_RETAIN_WEIGHT:-$RETAIN_WEIGHT}"
A1_TRAIN_BS="${A1_TRAIN_BS:-$TRAIN_BS}"
A2_TRAIN_BS="${A2_TRAIN_BS:-$TRAIN_BS}"
A1_TRAIN_GA="${A1_TRAIN_GA:-$TRAIN_GA}"
A2_TRAIN_GA="${A2_TRAIN_GA:-$TRAIN_GA}"
A1_SEED="${A1_SEED:-$SEED}"
A2_SEED="${A2_SEED:-$SEED}"

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
echo "  CF provider      : $CF_PROVIDER"
echo "  CF API/JSON mode : $CF_MODEL / $CF_JSON_MODE / temp=$CF_TEMPERATURE (key=$CF_API_KEY_ENV)"
echo "  A1 assistant     : layers=$A1_NUM_LAYER, LoRA=$A1_LORA_R/$A1_LORA_ALPHA, dropout=$A1_LORA_DROPOUT"
echo "  A2 assistant     : layers=$A2_NUM_LAYER, LoRA=$A2_LORA_R/$A2_LORA_ALPHA, dropout=$A2_LORA_DROPOUT"
echo "  A1 optimization  : lr=$A1_TRAIN_LR, epochs=$A1_TRAIN_EP, uniform-weight=$A1_RETAIN_WEIGHT, bs/ga=$A1_TRAIN_BS/$A1_TRAIN_GA, seed=$A1_SEED"
echo "  A2 optimization  : lr=$A2_TRAIN_LR, epochs=$A2_TRAIN_EP, uniform-weight=$A2_RETAIN_WEIGHT, bs/ga=$A2_TRAIN_BS/$A2_TRAIN_GA, seed=$A2_SEED"
echo "  weights/filter   : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  optimizer        : $TRAIN_OPTIM"
echo "  eval overwrite   : $EVAL_OVERWRITE"
echo "  selection access : $SELECTION_RETAIN_ACCESS"
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
        --temperature "$CF_TEMPERATURE" \
        --views "$VIEWS" \
        "${limit_args[@]}"
else
    echo "[1/4] Reusing counterfactuals: $CF_PATH"
fi

train_role() {
    local role="$1"
    local output_root="${MODELS_ROOT}/${role}"
    local role_upper="${role^^}"
    local num_layer_var="${role_upper}_NUM_LAYER"
    local lora_r_var="${role_upper}_LORA_R"
    local lora_alpha_var="${role_upper}_LORA_ALPHA"
    local lora_dropout_var="${role_upper}_LORA_DROPOUT"
    local train_lr_var="${role_upper}_TRAIN_LR"
    local train_ep_var="${role_upper}_TRAIN_EP"
    local retain_weight_var="${role_upper}_RETAIN_WEIGHT"
    local train_bs_var="${role_upper}_TRAIN_BS"
    local train_ga_var="${role_upper}_TRAIN_GA"
    local seed_var="${role_upper}_SEED"
    local role_num_layer="${!num_layer_var}"
    local role_lora_r="${!lora_r_var}"
    local role_lora_alpha="${!lora_alpha_var}"
    local role_lora_dropout="${!lora_dropout_var}"
    local role_train_lr="${!train_lr_var}"
    local role_train_ep="${!train_ep_var}"
    local role_retain_weight="${!retain_weight_var}"
    local role_train_bs="${!train_bs_var}"
    local role_train_ga="${!train_ga_var}"
    local role_seed="${!seed_var}"
    local signature
    signature="role=$role|cf=$CF_PATH|layers=$role_num_layer|lora_r=$role_lora_r|lora_alpha=$role_lora_alpha|lora_dropout=$role_lora_dropout|lr=$role_train_lr|epochs=$role_train_ep|retain_weight=$role_retain_weight|bs=$role_train_bs|ga=$role_train_ga|optim=$TRAIN_OPTIM|seed=$role_seed"
    local signature_file="${output_root}/F2R_TRAIN_SIGNATURE.txt"
    if find "$output_root" -name 'checkpoint-*' -type d 2>/dev/null | grep -q .; then
        if [ -f "$signature_file" ] && [ "$(<"$signature_file")" != "$signature" ]; then
            echo "Checkpoint configuration mismatch under $output_root" >&2
            echo "stored:  $(<"$signature_file")" >&2
            echo "current: $signature" >&2
            echo "Use a unique MODELS_ROOT for every training configuration." >&2
            exit 1
        fi
        if [ ! -f "$signature_file" ]; then
            echo "      WARNING: reusing legacy unsigned $role checkpoint under $output_root" >&2
        fi
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
        model_mode.num_layer="$role_num_layer" \
        model_mode.Lora.r="$role_lora_r" \
        model_mode.Lora.alpha="$role_lora_alpha" \
        model_mode.Lora.dropout="$role_lora_dropout" \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight="$role_retain_weight" \
        trainer.batch_size="$role_train_bs" \
        trainer.gradient_accumulation_steps="$role_train_ga" \
        trainer.learning_rate="$role_train_lr" \
        trainer.optim="$TRAIN_OPTIM" \
        trainer.max_epochs="$role_train_ep" \
        trainer.seed="$role_seed" \
        seed="$role_seed" \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$output_root" \
        postfix="$role" \
        "hydra.run.dir=outputs/tune_log/f2r_${role}_${SPLIT}_${TRAIN_RUN_TAG}/\${now:%Y-%m-%d_%H-%M-%S}"
    mkdir -p "$output_root"
    printf '%s\n' "$signature" > "$signature_file"
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
    --weight-a1 "$WEIGHT_A1" \
    --weight-a2 "$WEIGHT_A2" \
    --top-filter "$TOP_FILTER" \
    --views "$VIEWS" \
    --a1-num-layer "$A1_NUM_LAYER" \
    --a2-num-layer "$A2_NUM_LAYER" \
    --a1-lora-r "$A1_LORA_R" \
    --a2-lora-r "$A2_LORA_R" \
    --a1-lora-alpha "$A1_LORA_ALPHA" \
    --a2-lora-alpha "$A2_LORA_ALPHA" \
    --a1-train-lr "$A1_TRAIN_LR" \
    --a2-train-lr "$A2_TRAIN_LR" \
    --a1-train-ep "$A1_TRAIN_EP" \
    --a2-train-ep "$A2_TRAIN_EP" \
    --a1-retain-weight "$A1_RETAIN_WEIGHT" \
    --a2-retain-weight "$A2_RETAIN_WEIGHT" \
    --a1-seed "$A1_SEED" \
    --a2-seed "$A2_SEED" \
    --selection-retain-access "$SELECTION_RETAIN_ACCESS" \
    "${summary_args[@]}"
echo "Done. Full report: $EVAL_DIR/F2R_REPORT.md"
echo "      EASE table: $EVAL_DIR/F2R_EASE_TABLE.md"
