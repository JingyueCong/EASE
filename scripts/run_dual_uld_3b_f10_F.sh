#!/usr/bin/env bash
# 3B f10 Plan F: STRICTLY replicate 1B v2 recipe (no rescaling).
# num_layer=4 (NOT 25% of 28; absolute = 1B's value), bs=4/ga=4, lr=1e-3.
# Plus: re-eval retain90 with the same eval pipeline used for the dual.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

# === EXACT 1B v2 ep8 recipe ===
NUM_LAYER=4         # ← 1B absolute, not 25%
LORA_R=16
A1_EP=8             # 1B's ep8 training (best ckpt was step=450 = ep6)
A2_EP=3             # 1B used ep=3
TRAIN_BS=4          # 1B used bs=4
TRAIN_GA=4          # 1B used ga=4
TRAIN_LR=1e-3       # 1B working lr
SAVE_STEPS=75       # 1B fine training save_steps
W1=-0.8; W2=0.5; TOPF=0.01

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F"
mkdir -p "$MODELS_ROOT"

echo "============================================================"
echo "3B Plan F: strict 1B replication"
echo "  num_layer=$NUM_LAYER (absolute, not 25%)"
echo "  A1_ep=$A1_EP A2_ep=$A2_EP bs=$TRAIN_BS ga=$TRAIN_GA lr=$TRAIN_LR save_steps=$SAVE_STEPS"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_F() {
    local role="$1" eps="$2"
    local out="${MODELS_ROOT}/${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role: exists"; return
    fi
    echo "  → Train ${role} (num_layer=$NUM_LAYER, ep=$eps, bs=$TRAIN_BS, ga=$TRAIN_GA, lr=$TRAIN_LR) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_F_${role}_forget10" \
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
        OUTPUTMODELDIR="$out" postfix="F${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_F_${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo "[Phase B] Train A1 + A2 (1B exact recipe)"
train_F a1 "$A1_EP"
train_F a2 "$A2_EP"

# Resolve final + a "step ~450 analog" for 3B
A1_FINAL=$(find "${MODELS_ROOT}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_FINAL=$(find "${MODELS_ROOT}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 final: $A1_FINAL"
echo "A2 final: $A2_FINAL"

# Try multiple A1 ckpts: 1B's best was ep6 of ep8 = step=450 (with save_steps=75)
# For 3B with same recipe: same step structure expected (since equal_sampler counts depend on dataset, not model size)
# Pick step≈300, 450, 600 from A1's ckpts (or whatever's closest)
A1_CKPTS=()
for s in 300 450 600; do
    near=$(find "${MODELS_ROOT}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | awk -v t="$s" 'function abs(v){return v<0?-v:v} {d=abs($1-t); if(d<min || NR==1){min=d; line=$0}} END{print line}' | cut -d' ' -f2-)
    [ -n "$near" ] && A1_CKPTS+=("$near")
done

run_dual() {
    local a1="$1" w2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 1000 ] && { echo "  SKIP $task"; return; }
    rm -f "$eval_json"
    echo "  → Eval $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_FINAL" \
        model.model_args.weight_a1="$W1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter="$TOPF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
    fq=$($PY -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null || echo "-1")
    echo "    FQ=$fq"
}

echo "[Phase C] Eval Plan F (multiple A1 ckpts × w2)"
for a1 in "${A1_CKPTS[@]}"; do
    step=$(echo "$a1" | awk -F'checkpoint-' '{print $NF}')
    run_dual "$a1" 0.5 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2F_a1s${step}_w20p5"
done
# Also try final ckpt with w2=0.8
run_dual "$A1_FINAL" 0.8 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2F_final_w20p8"

echo "============================================================"
echo "Plan F DONE — see saves/eval/*_DualULD_v2F_*"
echo "============================================================"
