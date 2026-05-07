#!/usr/bin/env bash
# (a) Re-eval retain99/95/90 with EM + Fluency
# (b) Re-eval Pareto candidates on forget05/10 with EM + Fluency
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

# ============================================
# (a) RETAIN references — use plain Llama-3.2-1B-Instruct model (not dual)
# ============================================
echo "=== (a) Re-eval retain references with EM + Fluency ==="
for sp in "retain99 forget01 holdout01" "retain95 forget05 holdout05" "retain90 forget10 holdout10"; do
    retain=$(echo "$sp" | awk '{print $1}')
    forget=$(echo "$sp" | awk '{print $2}')
    holdout=$(echo "$sp" | awk '{print $3}')
    task="tofu_Llama-3.2-1B-Instruct_${retain}_v2"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $retain"; continue; }
    echo "  Eval $retain (model=${HF_BASE_PREFIX}_${retain})"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_${retain}" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$forget" holdout_split="$holdout" \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json" \
        task_name="$task"
done

# ============================================
# (b) Pareto candidates on f05/f10 with dual-ULD v2 assistants
# ============================================
echo "=== (b) Pareto candidates ==="
A2_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_F10=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a1_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F10_400=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget10 -name "checkpoint-400" -type d | head -1)
A1_F10_EP6=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget10 -name "checkpoint-450" -type d | head -1)
A1_F10_375=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a1_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

run_dual() {
    local split="$1" holdout="$2" retain="$3" a1="$4" a2="$5" w1="$6" w2="$7" task="$8"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $task"; return; }
    echo "  Eval $task (w1=$w1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$split" holdout_split="$holdout" \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json" \
        task_name="$task"
}

# forget05: try different w2 at w1=-0.8 (only had baseline w2=0.8 here)
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05" -0.8 0.5 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_w1m0p8_w20p5_emflu
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05" -0.8 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_w1m0p8_w20p7_emflu

# forget10: re-eval interesting Pareto points with EM/Fluency
run_dual forget10 holdout10 retain90 "$A1_F10_EP6" "$A2_F10" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep6_emflu
run_dual forget10 holdout10 retain90 "$A1_F10_400" "$A2_F10" -0.8 0.5 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_w1m0p8_w20p5_emflu
run_dual forget10 holdout10 retain90 "$A1_F10_400" "$A2_F10" -0.8 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_w1m0p8_w20p7_emflu
echo "DONE"
