#!/usr/bin/env bash
# One-command F2R experiment: generate forget-conditioned counterfactuals,
# train A1/A2, then evaluate the frozen DualULD model with TOFU.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
TRAIN_ENV="${TRAIN_ENV:-ease-f2r-train}"
EVAL_ENV="${EVAL_ENV:-ease-f2r-eval}"
TRAIN_PY="${TRAIN_PY:-${CONDA_BASE}/envs/${TRAIN_ENV}/bin/python}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV}/bin/python}"

MODE="${MODE:-smoke}"                 # smoke | full
SPLIT="${SPLIT:-forget05}"            # forget01 | forget05 | forget10
GPU="${GPU:-0}"
VIEWS="${VIEWS:-2}"
CF_MODEL="${CF_MODEL:-deepseek-chat}"
CF_BASE_URL="${CF_BASE_URL:-https://api.deepseek.com}"
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
EVAL_BS="${EVAL_BS:-4}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-}"

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

echo "============================================================"
echo "F2R TOFU experiment"
echo "  mode/split       : $MODE / $SPLIT"
echo "  GPU              : $GPU"
echo "  counterfactuals  : $CF_PATH (views=$VIEWS, limit=$CF_LIMIT)"
echo "  assistants       : layers=$NUM_LAYER, LoRA-r=$LORA_R"
echo "  weights/filter   : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "============================================================"

if [ ! -s "$CF_PATH" ]; then
    if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
        echo "DEEPSEEK_API_KEY is required to generate $CF_PATH" >&2
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

echo "[4/4] Evaluating frozen F2R model"
retain_arg="retain_logs_path=null"
if [ -n "$RETAIN_LOGS_PATH" ]; then
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
        eval.tofu.overwrite=true \
        "$retain_arg" \
        task_name="$TASK_NAME")

SUMMARY="$EASE_ROOT/open-unlearning/saves/eval/$TASK_NAME/TOFU_SUMMARY.json"
echo "Done. Summary: $SUMMARY"
if [ -f "$SUMMARY" ]; then cat "$SUMMARY"; fi
