#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2"

# A1 ckpt parent dir
A1_PARENT=$(find "${MODELS_ROOT}/a1_forget10" -name "checkpoint-375" -type d | head -1 | xargs dirname)
[ -d "$A1_PARENT" ] || { echo "ERROR: A1 parent dir missing"; exit 1; }
echo "A1 parent: $A1_PARENT"

# A2 final ckpt
A2_CKPT=$(find "${MODELS_ROOT}/a2_forget10" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A2 ckpt: $A2_CKPT"

# Existing per-epoch ckpts (75=ep1, 150=ep2, 225=ep3, 300=ep4, 375=ep5 already=v2 baseline)
# Step 375 already has eval as v2 baseline; skip
for STEP in 75 150 225 300; do
    a1_ckpt="${A1_PARENT}/checkpoint-${STEP}"
    [ -d "$a1_ckpt" ] || { echo "  skip step=$STEP (no ckpt)"; continue; }
    task="tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep$((STEP/75))"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP step=$STEP : exists"; continue; }
    echo "  → Eval forget10 v2 A1@step=$STEP (ep=$((STEP/75))) → $eval_json"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1_ckpt" \
        model.model_args.a2_path="$A2_CKPT" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2=0.8 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json" \
        task_name="$task"
done
echo "DONE"
