#!/usr/bin/env bash
# Train A2 forget01 ep=2 + eval with different A1 ckpts (step=140 / 150)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A1_F01_FINE_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget01 -name "checkpoint-*" -type d | head -1 | xargs dirname)

# Train A2 forget01 ep=2
A2_EP2_OUT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_a2ep2/a2_forget01"
if ! find "$A2_EP2_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[1] Train A2 forget01 ep=2"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_a2ep2_forget01" \
        data=tofu_chat3 data.dataset.split="forget01_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget01_k8.json" \
        data_mode.retain_num=40 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=2 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A2_EP2_OUT" postfix=a2ep2 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_a2ep2_forget01/\${now:%Y-%m-%d_%H-%M-%S}"
fi

A2_F01_EP2=$(find "$A2_EP2_OUT" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A2 ep=2: $A2_F01_EP2"

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
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_F01_EP2" \
        model.model_args.weight_a1=-0.7 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget01 holdout_split=holdout01 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json" \
        task_name="$task"
}

# (b) Combinations: A1 step=140/150 with A2 ep=2 at w2=0.5/0.7
A1_140="${A1_F01_FINE_PARENT}/checkpoint-140"
A1_150="${A1_F01_FINE_PARENT}/checkpoint-150"
run_dual "$A1_140" 0.7 tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step140_a2ep2_w20p7
run_dual "$A1_140" 0.5 tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step140_a2ep2_w20p5
run_dual "$A1_150" 0.7 tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step150_a2ep2_w20p7
run_dual "$A1_150" 0.5 tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step150_a2ep2_w20p5
echo "DONE"
