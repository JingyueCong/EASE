#!/usr/bin/env bash
# forget05 balanced search: a1ep8 step=200/225/275 + w2=0.5/0.7 / topF variants
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A2=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget05 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

EP8_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget05 \
    -name "checkpoint-*" -type d | head -1 | xargs dirname)
A1_200="${EP8_PARENT}/checkpoint-200"
A1_225="${EP8_PARENT}/checkpoint-225"
A1_250="${EP8_PARENT}/checkpoint-250"
A1_275="${EP8_PARENT}/checkpoint-275"

echo "A2: $A2"
echo "A1 200/225/250/275 verified:"
for p in "$A1_200" "$A1_225" "$A1_250" "$A1_275"; do [ -d "$p" ] && echo "  OK $p" || echo "  MISS $p"; done

run_dual() {
    local a1="$1" w2="$2" topf="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $task"; return; }
    echo "  Eval $task (w2=$w2 topF=$topf)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter="$topf" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget05 holdout_split=holdout05 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json" \
        task_name="$task"
}

echo "=== Step 200/225/275 with w2=0.5/0.7 ==="
run_dual "$A1_200" 0.5 0.01 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s200_w20p5
run_dual "$A1_200" 0.7 0.01 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s200_w20p7
run_dual "$A1_225" 0.5 0.01 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s225_w20p5
run_dual "$A1_225" 0.7 0.01 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s225_w20p7
run_dual "$A1_275" 0.5 0.01 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s275_w20p5

echo "=== a1ep8 s250: stronger w2 + relaxed topF ==="
run_dual "$A1_250" 0.7 0.01 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s250_w20p7
run_dual "$A1_250" 0.5 0.02 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s250_w20p5_topf0p02
echo "DONE"
