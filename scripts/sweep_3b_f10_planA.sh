#!/usr/bin/env bash
# 3B f10 Plan A: stronger w1 (-1.0, -1.2) and A1 mid-ckpt scan, eval-only.
# Reuses fixed-pipeline A1/A2 ckpts.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

A1_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_fixed/a1_forget10 \
    -name "checkpoint-*" -type d | head -1 | xargs dirname)
A2_FINAL=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_fixed/a2_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

A1_400="${A1_PARENT}/checkpoint-400"
A1_300="${A1_PARENT}/checkpoint-300"
A1_225="${A1_PARENT}/checkpoint-225"

echo "A1 step=225: $A1_225"
echo "A1 step=300: $A1_300"
echo "A1 step=400: $A1_400"
echo "A2 final:    $A2_FINAL"

run_dual() {
    local a1="$1" w1="$2" w2="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $task"; return; }
    echo "  Eval $task (w1=$w1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_FINAL" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
}

echo "=== Stronger w1 with current A1 ckpt-400 ==="
run_dual "$A1_400" -1.0 0.5 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2fix_a1s400_w1m1p0_w20p5
run_dual "$A1_400" -1.2 0.5 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2fix_a1s400_w1m1p2_w20p5

echo "=== A1 mid-ckpt scan with w1=-1.0 ==="
run_dual "$A1_300" -1.0 0.5 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2fix_a1s300_w1m1p0_w20p5
run_dual "$A1_225" -1.0 0.5 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2fix_a1s225_w1m1p0_w20p5

echo "DONE"
