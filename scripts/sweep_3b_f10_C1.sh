#!/usr/bin/env bash
# 3B f10 Plan C1: top_logit_filter / w1 sweep on Plan B's A1+A2 ckpts.
# Hypothesis: 3B base has sharper logits => topF=0.01 mask too narrow.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

A1_FINAL=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_planB/a1_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_FINAL=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_planB/a2_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "A1 final: $A1_FINAL"
echo "A2 final: $A2_FINAL"

run_dual() {
    local w1="$1" w2="$2" tF="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 1000 ] && { echo "  SKIP $task"; return; }
    rm -f "$eval_json"
    echo "  Eval $task (w1=$w1 w2=$w2 topF=$tF)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$A1_FINAL" model.model_args.a2_path="$A2_FINAL" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter="$tF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
}

echo "=== Plan C1: widen top_logit_filter and/or stronger w1 ==="
# 1) widen mask, keep canonical w1
run_dual -0.8 0.5 0.1   tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_tF0p1_w1m0p8_w20p5
run_dual -0.8 0.5 0.3   tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_tF0p3_w1m0p8_w20p5
# 2) widen mask + stronger w1
run_dual -1.5 0.5 0.1   tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_tF0p1_w1m1p5_w20p5
# 3) widen mask + extreme w1 (only if 1-3 still bad)
run_dual -2.0 0.5 0.3   tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_tF0p3_w1m2p0_w20p5

echo "DONE"
