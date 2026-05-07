#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

OUTPUT_DIR="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget10"
A2_DIR="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget10"
RSUB="${ULD_REPO}/data/rsub/forget10_k80.json"

# 1) Train A1 max_ep=8 (will get per-epoch ckpts; final at step ~600)
if ! find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[1/2] Train A1 forget10 with max_epochs=8 → $OUTPUT_DIR"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_ep8_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="$RSUB" \
        data_mode.retain_num=400 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=8 \
        trainer.strategy=gpu OUTPUTMODELDIR="$OUTPUT_DIR" postfix=a1ep8 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_ep8_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
else
    echo "[1/2] SKIP train (checkpoints exist)"
fi

# 2) Eval ep=6, 7, 8 with same A2
A1_PARENT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | head -1 | xargs dirname)
A2_CKPT=$(find "$A2_DIR" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 parent: $A1_PARENT"
echo "A2 ckpt: $A2_CKPT"

# Per-epoch step is 75 (forget10 with retain_num=400, batch=4 GA=4 → 75 steps/epoch)
for EP in 6 7 8; do
    STEP=$((EP * 75))
    a1_ckpt="${A1_PARENT}/checkpoint-${STEP}"
    [ -d "$a1_ckpt" ] || { echo "  skip ep=$EP step=$STEP (no ckpt)"; continue; }
    task="tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep${EP}"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP ep=$EP : exists"; continue; }
    echo "[2/2] Eval forget10 v2 A1@ep=$EP (step=$STEP) → $eval_json"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1_ckpt" model.model_args.a2_path="$A2_CKPT" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2=0.8 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json" \
        task_name="$task"
done
echo "DONE"
