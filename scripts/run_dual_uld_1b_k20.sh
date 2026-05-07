#!/usr/bin/env bash
#
# Dual-ULD on Llama-3.2-1B-Instruct, R_sub k=20 variant (forget05 + forget10).
# Goals: smaller R_sub → A2 less prone to overfit forget-set; A1 has less
# R_sub contamination so its forget objective is purer.
#
# Outputs to a separate MODELS_ROOT and uses task_name suffix _k20.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"

NUM_LAYER="${NUM_LAYER:-2}"; LORA_R="${LORA_R:-16}"
WEIGHT_A1="${WEIGHT_A1:--0.7}"; WEIGHT_A2="${WEIGHT_A2:-0.7}"; TOP_FILTER="${TOP_FILTER:-0.01}"
TRAIN_BS="${TRAIN_BS:-4}"; TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"; TRAIN_EP="${TRAIN_EP:-10}"; EVAL_BS="${EVAL_BS:-4}"

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_k20"
mkdir -p "$MODELS_ROOT"

# Per-split: forget holdout retain rsub_path retain_num
SPLITS=(
    "forget05 holdout05 retain95 ${ULD_REPO}/data/rsub/forget05_k20.json 200"
    "forget10 holdout10 retain90 ${ULD_REPO}/data/rsub/forget10_k20.json 400"
)

echo "============================================================"
echo "Dual-ULD k=20 retrain — Llama-3.2-1B-Instruct"
echo "  W1/W2/topF : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "============================================================"

# 0) Generate forget10_k20 R_sub if missing
if [ ! -f "${ULD_REPO}/data/rsub/forget10_k20.json" ]; then
    echo "[Phase 0] Generating forget10_k20 R_sub indices"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/select_rsub.py \
        --forget_split forget10_perturbed --retain_split retain90 \
        --retain_num 400 --k 20 \
        --out data/rsub/forget10_k20.json
fi

# 1) Train A1+A2 for each split
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_one() {
    local role="$1" forget="$2" rsub="$3" rnum="$4"
    local out="${MODELS_ROOT}/${role}_${forget}"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role/$forget : exists at $out"; return
    fi
    echo "  → Train $role/$forget → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_k20_${role}_${forget}" \
        data=tofu_chat3 data.dataset.split="${forget}_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="$rsub" \
        data_mode.retain_num="$rnum" \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$TRAIN_EP" \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_k20_${role}_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo
echo "[Phase B] Train A1 + A2 (k=20)"
for sp in "${SPLITS[@]}"; do
    forget=$(echo "$sp" | awk '{print $1}')
    rsub=$(echo "$sp" | awk '{print $4}')
    rnum=$(echo "$sp" | awk '{print $5}')
    [ -f "$rsub" ] || { echo "  ✗ missing R_sub: $rsub"; exit 1; }
    train_one a1 "$forget" "$rsub" "$rnum"
    train_one a2 "$forget" "$rsub" "$rnum"
done

# 2) Eval
cd "$OU_REPO"
echo
echo "[Phase C] Eval Dual-ULD k=20 (w1=$WEIGHT_A1, w2=$WEIGHT_A2)"
for sp in "${SPLITS[@]}"; do
    forget=$(echo "$sp" | awk '{print $1}')
    holdout=$(echo "$sp" | awk '{print $2}')
    retain=$(echo "$sp" | awk '{print $3}')
    a1=$(find "${MODELS_ROOT}/a1_${forget}" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    a2=$(find "${MODELS_ROOT}/a2_${forget}" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    task="tofu_Llama-3.2-1B-Instruct_${forget}_DualULD_k20_w1m0p7_w20p7_topf0p01"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP $forget : $eval_json exists"; continue; }
    retain_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json"
    echo "  → Eval k=20 dual on $forget → $eval_json"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$WEIGHT_A1" model.model_args.weight_a2="$WEIGHT_A2" \
        model.model_args.top_logit_filter="$TOP_FILTER" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split="$forget" holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        retain_logs_path="$retain_json" task_name="$task"
done

echo
echo "============================================================"
echo "DONE — k=20 dual evals saved with task_name *_DualULD_k20_*"
echo "============================================================"
