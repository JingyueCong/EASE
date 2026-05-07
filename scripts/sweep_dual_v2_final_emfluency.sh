#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

# Best-Pareto configs per split; also include retain99/95/90 references (they need fluency too)
# Format: split holdout retain a1_path a2_path w1 w2 task
A2_F01=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget01 -name "checkpoint-*" -type d | tail -1)
A2_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_F10=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F01=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget01 -name "checkpoint-140" -type d | head -1)
A1_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a1_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F10=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget10 -name "checkpoint-400" -type d | head -1)

echo "A1 forget01 step=140: $A1_F01"
echo "A1 forget05 ep=5: $A1_F05"
echo "A1 forget10 step=400: $A1_F10"

run_one() {
    local split="$1" holdout="$2" retain="$3" a1="$4" a2="$5" w1="$6" w2="$7" task="$8"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP $task"; return; }
    echo "  → Eval $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split="$split" holdout_split="$holdout" \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json" \
        task_name="$task"
}

run_one forget01 holdout01 retain99 "$A1_F01" "$A2_F01" -0.7 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step140
run_one forget05 holdout05 retain95 "$A1_F05" "$A2_F05" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2
run_one forget10 holdout10 retain90 "$A1_F10" "$A2_F10" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1step400
echo "DONE"
