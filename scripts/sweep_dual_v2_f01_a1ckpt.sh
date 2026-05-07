#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A1_PARENT=$(find "${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a1_forget01" -name "checkpoint-30" -type d | head -1 | xargs dirname)
A2_CKPT=$(find "${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget01" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 parent: $A1_PARENT"
echo "A2 ckpt: $A2_CKPT"

# Eval each per-epoch ckpt (30/60/90/120, ep=5 already eval'd as v2 baseline w1=-0.7)
for STEP in 30 60 90 120; do
    a1_ckpt="${A1_PARENT}/checkpoint-${STEP}"
    [ -d "$a1_ckpt" ] || { echo "  skip step=$STEP"; continue; }
    task="tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step${STEP}"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP step=$STEP : exists"; continue; }
    echo "  → Eval forget01 v2 A1@step=$STEP → $eval_json"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1_ckpt" model.model_args.a2_path="$A2_CKPT" \
        model.model_args.weight_a1=-0.7 model.model_args.weight_a2=0.7 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split=forget01 holdout_split=holdout01 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json" \
        task_name="$task"
done
echo "DONE"
