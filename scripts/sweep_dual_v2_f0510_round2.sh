#!/usr/bin/env bash
# forget05: train A2 ep=1; eval w2 sweep
# forget10: train A2 ep=2; eval a1ep6 + a1step400 with a2ep2
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A1_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a1_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F10_400=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget10 -name "checkpoint-400" -type d | head -1)
A1_F10_EP6=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget10 -name "checkpoint-450" -type d | head -1)

# (1) Train A2 forget05 ep=1
A2_EP1_OUT_F05="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_a2ep1/a2_forget05"
if ! find "$A2_EP1_OUT_F05" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[1] Train A2 forget05 ep=1"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_a2ep1_forget05" \
        data=tofu_chat3 data.dataset.split="forget05_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget05_k40.json" \
        data_mode.retain_num=200 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=1 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A2_EP1_OUT_F05" postfix=a2ep1 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_a2ep1_forget05/\${now:%Y-%m-%d_%H-%M-%S}"
fi

# (2) Train A2 forget10 ep=2
A2_EP2_OUT_F10="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_a2ep2/a2_forget10"
if ! find "$A2_EP2_OUT_F10" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[2] Train A2 forget10 ep=2"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_a2ep2_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=2 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A2_EP2_OUT_F10" postfix=a2ep2 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_a2ep2_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi

A2_F05_EP1=$(find "$A2_EP1_OUT_F05" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_F10_EP2=$(find "$A2_EP2_OUT_F10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

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

# forget05: A2 ep=1, w2 sweep
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05_EP1" -0.8 0.5 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a2ep1_w20p5
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05_EP1" -0.8 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a2ep1_w20p7
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05_EP1" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a2ep1_w20p8

# forget10: A2 ep=2, multiple A1 ckpts, w2 sweep
# a1step400 (max FQ) + a2ep2
run_dual forget10 holdout10 retain90 "$A1_F10_400" "$A2_F10_EP2" -0.8 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1step400_a2ep2_w20p7
run_dual forget10 holdout10 retain90 "$A1_F10_400" "$A2_F10_EP2" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1step400_a2ep2_w20p8
# a1ep6 (max Agg) + a2ep2
run_dual forget10 holdout10 retain90 "$A1_F10_EP6" "$A2_F10_EP2" -0.8 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep6_a2ep2_w20p7
run_dual forget10 holdout10 retain90 "$A1_F10_EP6" "$A2_F10_EP2" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep6_a2ep2_w20p8
echo "DONE"
