#!/usr/bin/env bash
# Scope-A: Dual-ULD on Llama-3.2-3B-Instruct, forget10 only.
# Recipe scaled from 1B v2:
#   num_layer  = 7   (= 28 × 25%)
#   A1 epochs  = 5
#   A2 epochs  = 3
#   weight_a1  = -0.8
#   weight_a2  = +0.5  (1B FQ-preserving sweet spot)
#   top_filter = 0.01
#   R_sub      = forget10_k80.json (≈20% of retain_num)
#   bs / ga    = 2 / 8 (effective 16, fits 40GB for 3B)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-3B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.2-3B-Instruct_full}"

NUM_LAYER=7; LORA_R=16
A1_EP=5; A2_EP=3
WEIGHT_A1=-0.8; WEIGHT_A2=0.5; TOP_FILTER=0.01
TRAIN_BS=2; TRAIN_GA=8
TRAIN_LR=1e-3; EVAL_BS=2

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2"
mkdir -p "$MODELS_ROOT"

forget=forget10
holdout=holdout10
retain=retain90
rsub="${ULD_REPO}/data/rsub/forget10_k80.json"
rnum=400

[ -f "$rsub" ] || { echo "  ✗ missing R_sub: $rsub"; exit 1; }

echo "============================================================"
echo "Dual-ULD 3B f10 (Scope A)"
echo "  num_layer=$NUM_LAYER  A1_ep=$A1_EP  A2_ep=$A2_EP"
echo "  W1=$WEIGHT_A1  W2=$WEIGHT_A2  topF=$TOP_FILTER"
echo "============================================================"

# ============================================================
# Phase A — re-eval retain90 (3B) with EM+Fluency for scale-consistent comparison
# ============================================================
RETAIN_TASK="tofu_Llama-3.2-3B-Instruct_retain90_v2"
RETAIN_JSON="${OU_REPO}/saves/eval/${RETAIN_TASK}/TOFU_EVAL.json"
if [ ! -f "$RETAIN_JSON" ]; then
    echo
    echo "[Phase A] Re-eval retain90 (3B) with EM+Fluency"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-3B-Instruct \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_${retain}" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$forget" holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_${retain}/TOFU_EVAL.json" \
        task_name="$RETAIN_TASK"
fi

# ============================================================
# Phase B — Train A1 + A2 (LoRA on 3B truncated to num_layer=7)
# ============================================================
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_one() {
    local role="$1" eps="$2"
    local out="${MODELS_ROOT}/${role}_${forget}"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role: exists at $out"; return
    fi
    echo "  → Train $role (ep=$eps) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_${role}_${forget}" \
        data=tofu_chat3 data.dataset.split="${forget}_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="$rsub" \
        data_mode.retain_num="$rnum" \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$eps" \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_${role}_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo
echo "[Phase B] Train A1 (ep=$A1_EP) + A2 (ep=$A2_EP)"
train_one a1 "$A1_EP"
train_one a2 "$A2_EP"

# ============================================================
# Phase C — Eval Dual-ULD with w2=0.5 (FQ-preserving) and w2=0.8 (default)
# ============================================================
A1=$(find "${MODELS_ROOT}/a1_${forget}" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2=$(find "${MODELS_ROOT}/a2_${forget}" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 final ckpt: $A1"
echo "A2 final ckpt: $A2"

run_dual() {
    local w2="$1" task="$2"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP $task: exists"; return; }
    echo "  → Eval $task (w1=$WEIGHT_A1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$A1" model.model_args.a2_path="$A2" \
        model.model_args.weight_a1="$WEIGHT_A1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter="$TOP_FILTER" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$forget" holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        retain_logs_path="$RETAIN_JSON" task_name="$task"
}

cd "$OU_REPO"
echo
echo "[Phase C] Eval Dual-ULD"
run_dual 0.5 "tofu_Llama-3.2-3B-Instruct_${forget}_DualULD_v2_w20p5"
run_dual 0.8 "tofu_Llama-3.2-3B-Instruct_${forget}_DualULD_v2_w20p8"

echo "============================================================"
echo "DONE — see saves/eval/tofu_Llama-3.2-3B-Instruct_${forget}_DualULD_v2_*"
echo "============================================================"
