#!/usr/bin/env bash
# 3B forget10 FIX: train A1 ep=12 (target step≈444 ≈ 1B's step=450 best),
# A2 ep=6 (target step≈222 ≈ 1B's ep=3=step=225). save_steps=25 for fine scan.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

NUM_LAYER=7; LORA_R=16
WEIGHT_A1=-0.8; TOP_FILTER=0.01
TRAIN_BS=2; TRAIN_GA=8; TRAIN_LR=1e-3; EVAL_BS=2

A1_EP=12; A2_EP=6
SAVE_STEPS=25

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_fixed"
mkdir -p "$MODELS_ROOT"

forget=forget10
holdout=holdout10
retain=retain90
rsub="${ULD_REPO}/data/rsub/forget10_k80.json"
rnum=400

echo "============================================================"
echo "3B Dual-ULD FIX (forget10)"
echo "  num_layer=$NUM_LAYER  A1_ep=$A1_EP  A2_ep=$A2_EP  save_steps=$SAVE_STEPS"
echo "  W1=$WEIGHT_A1  topF=$TOP_FILTER"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_one() {
    local role="$1" eps="$2"
    local out="${MODELS_ROOT}/${role}_${forget}"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role: exists at $out"; return
    fi
    echo "  → Train $role (ep=$eps, save_steps=$SAVE_STEPS) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_fixed_${role}_${forget}" \
        data=tofu_chat3 data.dataset.split="${forget}_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="$rsub" \
        data_mode.retain_num="$rnum" \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$eps" \
        +trainer.save_steps="$SAVE_STEPS" \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="${role}fix" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_fixed_${role}_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo
echo "[Phase B] Train A1 (ep=$A1_EP) + A2 (ep=$A2_EP)"
train_one a1 "$A1_EP"
train_one a2 "$A2_EP"

# Pick A1 ckpt closest to 1B's best step=450 (3B step≈444); A2 final (ep=6 ≈ step=222)
A1_BEST=$(find "${MODELS_ROOT}/a1_${forget}" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '$NF>=400 && $NF<=475 {print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -z "$A1_BEST" ] && A1_BEST=$(find "${MODELS_ROOT}/a1_${forget}" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_FINAL=$(find "${MODELS_ROOT}/a1_${forget}" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2=$(find "${MODELS_ROOT}/a2_${forget}" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 best (~step 450): $A1_BEST"
echo "A1 final:            $A1_FINAL"
echo "A2 final (ep=6):     $A2"

run_dual() {
    local a1="$1" w2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP $task"; return; }
    echo "  → Eval $task (w1=$WEIGHT_A1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2" \
        model.model_args.weight_a1="$WEIGHT_A1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter="$TOP_FILTER" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$forget" holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
}

cd "$OU_REPO"
echo
echo "[Phase C] Eval A1 best (~step 450) + A2 ep=6, w2=0.5/0.8"
run_dual "$A1_BEST" 0.5 "tofu_Llama-3.2-3B-Instruct_${forget}_DualULD_v2fix_w20p5"
run_dual "$A1_BEST" 0.8 "tofu_Llama-3.2-3B-Instruct_${forget}_DualULD_v2fix_w20p8"
echo "DONE"
