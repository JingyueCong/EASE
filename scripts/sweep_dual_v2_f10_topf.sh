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

# Wait for f10 w2 sweep to finish
until ! pgrep -f "[s]weep_dual_v2_f10_w2.sh" > /dev/null; do sleep 15; done
echo "f10 w2 sweep finished, starting topF sweep at w1=-0.8 w2=0.8"

a1=$(find "${MODELS_ROOT}/a1_forget10" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
a2=$(find "${MODELS_ROOT}/a2_forget10" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

for TOPF in 0.005 0.02 0.05; do
    ts=$(echo "$TOPF" | sed -e 's/\./p/g')
    task="tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_topf${ts}"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP topF=$TOPF : exists"; continue; }
    echo "  → Eval forget10 v2 w1=-0.8 w2=0.8 topF=$TOPF → $eval_json"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2=0.8 \
        model.model_args.top_logit_filter="$TOPF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json" \
        task_name="$task"
done
echo "DONE"
