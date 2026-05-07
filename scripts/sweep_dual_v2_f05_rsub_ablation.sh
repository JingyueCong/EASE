#!/usr/bin/env bash
# Dual-ULD R_sub ablation on Llama-3.2-1B forget05.
#
# Trains A1+A2 for two new R_sub selection criteria and evaluates each at the
# baseline-best inference weights. The third arm (forget_only) is the existing
# v2_a1ep8s225_w20p7 result (forget_quality=0.9973) and is reused.
#
# Inputs:
#   data/rsub/forget05_k40_retainonly.json
#   data/rsub/forget05_k40_both.json
# Outputs:
#   ULD/outputs_trained_models/llama3_1b_dual_v2_ep8_<variant>/a1_forget05/...
#   ULD/outputs_trained_models/llama3_1b_dual_v2_<variant>/a2_forget05/...
#   open-unlearning/saves/eval/tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_rsubabl_<variant>/
#
# Recipe is fixed to match the baseline that produced fq=0.9973:
#   A1: max_epochs=8, save_steps=25  -> use checkpoint-225
#   A2: max_epochs=3, save_steps=50  -> use checkpoint-150 (final)
#   eval: weight_a1=-0.8, weight_a2=0.7, top_logit_filter=0.01

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

VARIANTS=("retainonly" "both")

train_a1() {
    local variant="$1"
    local rsub_json="${ULD_REPO}/data/rsub/forget05_k40_${variant}.json"
    local out_dir="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8_${variant}/a1_forget05"
    if find "$out_dir" -name "checkpoint-225" -type d 2>/dev/null | grep -q .; then
        echo "  [A1/${variant}] checkpoint-225 exists, skip"
        return
    fi
    [ -f "$rsub_json" ] || { echo "MISSING $rsub_json"; exit 1; }
    echo "  [A1/${variant}] training: max_epochs=8 save_steps=25"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_ep8_${variant}_a1_forget05" \
        data=tofu_chat3 data.dataset.split="forget05_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="$rsub_json" \
        data_mode.retain_num=200 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=8 \
        +trainer.save_steps=25 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out_dir" postfix="ep8_${variant}" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_ep8_${variant}_a1_forget05/\${now:%Y-%m-%d_%H-%M-%S}"
}

train_a2() {
    local variant="$1"
    local rsub_json="${ULD_REPO}/data/rsub/forget05_k40_${variant}.json"
    local out_dir="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_${variant}/a2_forget05"
    if find "$out_dir" -name "checkpoint-150" -type d 2>/dev/null | grep -q .; then
        echo "  [A2/${variant}] checkpoint-150 exists, skip"
        return
    fi
    echo "  [A2/${variant}] training: max_epochs=3 save_steps=50"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_${variant}_a2_forget05" \
        data=tofu_chat3 data.dataset.split="forget05_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="$rsub_json" \
        data_mode.retain_num=200 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=3 \
        +trainer.save_steps=50 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out_dir" postfix="a2_${variant}" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_${variant}_a2_forget05/\${now:%Y-%m-%d_%H-%M-%S}"
}

eval_dual() {
    local variant="$1"
    local a1_root="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8_${variant}/a1_forget05"
    local a2_root="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_${variant}/a2_forget05"
    local a1_parent
    a1_parent=$(find "$a1_root" -name "checkpoint-225" -type d | head -1 | xargs dirname)
    local a2_parent
    a2_parent=$(find "$a2_root" -name "checkpoint-150" -type d | head -1 | xargs dirname)
    local a1="${a1_parent}/checkpoint-225"
    local a2="${a2_parent}/checkpoint-150"
    [ -d "$a1" ] || { echo "MISSING A1 ${a1}"; exit 1; }
    [ -d "$a2" ] || { echo "MISSING A2 ${a2}"; exit 1; }
    local task="tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_rsubabl_${variant}"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    if [ -f "$eval_json" ]; then
        echo "  [eval/${variant}] $task exists, skip"
        return
    fi
    echo "  [eval/${variant}] $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2=0.7 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget05 holdout_split=holdout05 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json" \
        task_name="$task"
}

for variant in "${VARIANTS[@]}"; do
    echo "=== variant: ${variant} ==="
    train_a1 "$variant"
    train_a2 "$variant"
    eval_dual "$variant"
done

echo ""
echo "=== summary ==="
for variant in forget_only "${VARIANTS[@]}"; do
    if [ "$variant" = "forget_only" ]; then
        s="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8s225_w20p7/TOFU_SUMMARY.json"
    else
        s="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_rsubabl_${variant}/TOFU_SUMMARY.json"
    fi
    if [ -f "$s" ]; then
        fq=$($PY -c "import json;d=json.load(open('$s'));print(f\"{d['forget_quality']:.4f}\")")
        mu=$($PY -c "import json;d=json.load(open('$s'));print(f\"{d['model_utility']:.4f}\")")
        echo "  ${variant}: forget_quality=${fq}  model_utility=${mu}"
    else
        echo "  ${variant}: MISSING ($s)"
    fi
done
echo "DONE"
