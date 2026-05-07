#!/usr/bin/env bash
# After dual_v2 pipeline finishes (forget10 v2 eval done), sweep forget10 w1.
# Also re-eval forget01 v2 at w1=-0.7 to recover that split's optimum.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2"

# Wait until v2 pipeline has GPU free
until ! pgrep -f "[r]un_dual_uld_1b_v2.sh" > /dev/null; do sleep 15; done
echo "v2 pipeline finished, starting sweep"

run_one() {
    local split="$1" holdout="$2" retain="$3" w1="$4" w2="$5"
    local a1=$(find "${MODELS_ROOT}/a1_${split}" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    local a2=$(find "${MODELS_ROOT}/a2_${split}" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    local w1s=$(echo "$w1" | sed -e 's/-/m/g' -e 's/\./p/g')
    local w2s=$(echo "$w2" | sed -e 's/-/m/g' -e 's/\./p/g')
    local task="tofu_Llama-3.2-1B-Instruct_${split}_DualULD_v2_w1${w1s}_w2${w2s}"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP $split w1=$w1 w2=$w2 : exists"; return; }
    echo "  → Eval $split v2 w1=$w1 w2=$w2 → $eval_json"
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

# 1) forget01 at w=-0.7 (recover forget01 win)
run_one forget01 holdout01 retain99 -0.7 0.7

# 2) forget10 sweep over w1, w2=0.8 fixed
for W1 in -0.5 -0.6 -0.7 -0.9; do
    run_one forget10 holdout10 retain90 "$W1" 0.8
done

echo "DONE"
