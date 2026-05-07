#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"; OU_REPO="${ROOT}/open-unlearning"; PY=/usr/bin/python; GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

# Wait until prior round 3 sweep is done (no eval.py running)
until ! pgrep -f "[s]weep_dual_v2_f10_w2.sh" > /dev/null; do sleep 30; done
echo "prior sweep done; starting f05 fine eval"

A1_FINE_F05_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget05 -name "checkpoint-*" -type d | head -1 | xargs dirname)
A2_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

run_dual() {
    local a1="$1" w2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $task"; return; }
    echo "  Eval $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_F05" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget05 holdout_split=holdout05 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json" \
        task_name="$task"
}

# Try mid-training A1 ckpts (75/125/175 between epoch boundaries, plus 225 = past final)
for STEP in 125 175 225; do
    a1="${A1_FINE_F05_PARENT}/checkpoint-${STEP}"
    [ -d "$a1" ] || { echo "  skip step=$STEP"; continue; }
    run_dual "$a1" 0.5 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1fine_step${STEP}_w20p5
    run_dual "$a1" 0.7 tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1fine_step${STEP}_w20p7
done
echo "DONE"
