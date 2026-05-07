#!/usr/bin/env bash
# 3B f10 Plan B: lower LR (5e-4), longer max_epochs (A1=20, A2=10),
# fine save_steps=25. Goal: A1 train_loss should converge to single digits
# instead of 37 (which made logits too weak to subtract).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

NUM_LAYER=7; LORA_R=16
A1_EP=20; A2_EP=10
TRAIN_BS=2; TRAIN_GA=8
TRAIN_LR=5e-4
SAVE_STEPS=25

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_planB"
mkdir -p "$MODELS_ROOT"

echo "============================================================"
echo "3B Plan B: lr=$TRAIN_LR, A1_ep=$A1_EP, A2_ep=$A2_EP, save_steps=$SAVE_STEPS"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_one() {
    local role="$1" eps="$2"
    local out="${MODELS_ROOT}/${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role : exists"; return
    fi
    echo "  → Train ${role} forget10 (ep=$eps, lr=$TRAIN_LR, save_steps=$SAVE_STEPS) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_planB_${role}_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$eps" \
        +trainer.save_steps="$SAVE_STEPS" trainer.strategy=gpu \
        OUTPUTMODELDIR="$out" postfix="planB${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_planB_${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo "[Phase B] Train A1 + A2"
train_one a1 "$A1_EP"
train_one a2 "$A2_EP"

# Resolve final ckpts
A1_FINAL=$(find "${MODELS_ROOT}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_FINAL=$(find "${MODELS_ROOT}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_MID=$(find "${MODELS_ROOT}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | awk 'NR==int((NR+1)/2){print $0}' | tail -1 | cut -d' ' -f2-)
echo "A1 final: $A1_FINAL"
echo "A2 final: $A2_FINAL"

run_dual() {
    local a1="$1" w1="$2" w2="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ -s "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && { echo "  SKIP $task"; return; }
    rm -f "$eval_json"
    echo "  → Eval $task (w1=$w1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_FINAL" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
}

echo "[Phase C] Eval Plan B"
run_dual "$A1_FINAL" -0.8 0.5 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_w20p5
run_dual "$A1_FINAL" -0.8 0.8 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_w20p8

echo "DONE"
