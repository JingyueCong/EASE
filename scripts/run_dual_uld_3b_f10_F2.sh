#!/usr/bin/env bash
# 3B f10 Plan F2: extend A1 training (ep=20, save_steps=25) since FQ
# monotonically rises through Plan F's last ckpt (step=600). Reuse Plan F A2.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

NUM_LAYER=4; LORA_R=16
A1_EP=20; A2_EP=3   # extend A1 only; reuse Plan F A2
TRAIN_BS=4; TRAIN_GA=4; TRAIN_LR=1e-3; SAVE_STEPS=25

MODELS_F2="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F2"
A2_PATH=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a2_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "============================================================"
echo "3B Plan F2: extend A1 to ep=20, save_steps=25"
echo "  Reuse Plan F A2: $A2_PATH"
echo "============================================================"

mkdir -p "$MODELS_F2"
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

# Train extended A1
A1_OUT="${MODELS_F2}/a1_forget10"
if ! find "$A1_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[B] Train A1 (ep=$A1_EP, save_steps=$SAVE_STEPS)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_F2_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$A1_EP" \
        +trainer.save_steps="$SAVE_STEPS" trainer.strategy=gpu \
        OUTPUTMODELDIR="$A1_OUT" postfix="F2a1" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_F2_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi

run_dual() {
    local a1="$1" w2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 1000 ] && { echo "  SKIP $task"; return; }
    rm -f "$eval_json"
    echo "  → Eval $task (w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_PATH" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
    fq=$($PY -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null || echo "-1")
    echo "    FQ=$fq"
}

A1_PARENT=$(find "$A1_OUT" -name "checkpoint-*" -type d | head -1 | xargs dirname)
echo "A1 parent: $A1_PARENT"

echo "[C] Eval extended A1 ckpts × w2=0.5"
for s in 700 800 900 1000 1100 1200; do
    a1="${A1_PARENT}/checkpoint-${s}"
    [ -d "$a1" ] || { echo "  miss step=$s"; continue; }
    run_dual "$a1" 0.5 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2F2_a1s${s}_w20p5"
done

# w2 sweep on best (we'll pick best from logs after)
# For now also try a1s600 with w2=0.7 (using Plan F's existing A1, since that ckpt is in F not F2)
A1_F600="$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a1_forget10 -name "checkpoint-600" -type d | head -1)"
[ -d "$A1_F600" ] && run_dual "$A1_F600" 0.7 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2F_a1s600_w20p7"

echo "============================================================"
echo "Plan F2 DONE"
echo "============================================================"
