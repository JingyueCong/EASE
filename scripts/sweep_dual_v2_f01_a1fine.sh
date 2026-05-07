#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

OUTPUT_DIR="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget01"
A2_DIR="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget01"
RSUB="${ULD_REPO}/data/rsub/forget01_k8.json"

# 1) Train A1 forget01 max_ep=5 save_steps=5
if ! find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[1/2] Train A1 forget01 max_ep=5 save_steps=5 → $OUTPUT_DIR"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_fine_a1_forget01" \
        data=tofu_chat3 data.dataset.split="forget01_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="$RSUB" \
        data_mode.retain_num=40 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=5 \
        +trainer.save_steps=5 \
        trainer.strategy=gpu OUTPUTMODELDIR="$OUTPUT_DIR" postfix=a1fine \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_fine_a1_forget01/\${now:%Y-%m-%d_%H-%M-%S}"
fi

# 2) Eval intermediate ckpts (skip 30/60/90/120/150 already eval'd)
A1_PARENT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | head -1 | xargs dirname)
A2_CKPT=$(find "$A2_DIR" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 parent: $A1_PARENT"
echo "A2 ckpt: $A2_CKPT"

# Strategic intermediate steps to bracket peak (between ep=3 fq=0.77 and ep=4 fq=0.92)
for STEP in 95 100 105 110 115 125 130 135 140 145; do
    a1_ckpt="${A1_PARENT}/checkpoint-${STEP}"
    [ -d "$a1_ckpt" ] || { echo "  skip step=$STEP"; continue; }
    task="tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step${STEP}"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP step=$STEP"; continue; }
    echo "[2/2] Eval forget01 v2 A1@step=$STEP → $eval_json"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1_ckpt" model.model_args.a2_path="$A2_CKPT" \
        model.model_args.weight_a1=-0.7 model.model_args.weight_a2=0.7 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split=forget01 holdout_split=holdout01 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json" \
        task_name="$task"
done
echo "DONE"
