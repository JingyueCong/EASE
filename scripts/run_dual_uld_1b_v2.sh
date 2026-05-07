#!/usr/bin/env bash
#
# Dual-ULD on Llama-3.2-1B-Instruct, replicating the working Llama-2-7B recipe
# from ULD/bashes/tofu/dual_uld_pipeline.sh + forget05_grid.sh:
#
#   num_layer  = 4   (= 16 × 25%, matches 7B's 8/32 ratio)
#   A1 epochs  = 5
#   A2 epochs  = 3   (LESS — A2 overfits R_sub on too many epochs)
#   weight_a1  = -0.8
#   weight_a2  = +0.8
#   top_filter = 0.01
#   R_sub      = forget01_k8, forget05_k40, forget10_k80 (≈20% of retain_num)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"

NUM_LAYER="${NUM_LAYER:-4}"; LORA_R="${LORA_R:-16}"
A1_EP="${A1_EP:-5}"; A2_EP="${A2_EP:-3}"
WEIGHT_A1="${WEIGHT_A1:--0.8}"; WEIGHT_A2="${WEIGHT_A2:-0.8}"; TOP_FILTER="${TOP_FILTER:-0.01}"
TRAIN_BS="${TRAIN_BS:-4}"; TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"; EVAL_BS="${EVAL_BS:-4}"

MODELS_ROOT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2"
mkdir -p "$MODELS_ROOT"

# Per-split: forget holdout retain rsub_path retain_num
SPLITS=(
    "forget01 holdout01 retain99 ${ULD_REPO}/data/rsub/forget01_k8.json   40"
    "forget05 holdout05 retain95 ${ULD_REPO}/data/rsub/forget05_k40.json  200"
    "forget10 holdout10 retain90 ${ULD_REPO}/data/rsub/forget10_k80.json  400"
)
if [ -n "${ONLY:-}" ]; then
    new=()
    for sp in "${SPLITS[@]}"; do case "$sp" in "$ONLY"*) new+=("$sp");; esac; done
    SPLITS=("${new[@]}")
fi

echo "============================================================"
echo "Dual-ULD v2 (Llama-2-7B recipe ported to 1B)"
echo "  num_layer=$NUM_LAYER  A1_ep=$A1_EP  A2_ep=$A2_EP"
echo "  W1=$WEIGHT_A1  W2=$WEIGHT_A2  topF=$TOP_FILTER"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_one() {
    local role="$1" forget="$2" rsub="$3" rnum="$4" eps="$5"
    local out="${MODELS_ROOT}/${role}_${forget}"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role/$forget : exists"; return
    fi
    echo "  → Train $role/$forget (ep=$eps) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_${role}_${forget}" \
        data=tofu_chat3 data.dataset.split="${forget}_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="$rsub" \
        data_mode.retain_num="$rnum" \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$eps" \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_${role}_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo
echo "[Phase B] Train A1 (ep=$A1_EP) + A2 (ep=$A2_EP), num_layer=$NUM_LAYER"
for sp in "${SPLITS[@]}"; do
    forget=$(echo "$sp" | awk '{print $1}')
    rsub=$(echo "$sp" | awk '{print $4}')
    rnum=$(echo "$sp" | awk '{print $5}')
    [ -f "$rsub" ] || { echo "  ✗ missing R_sub: $rsub"; exit 1; }
    train_one a1 "$forget" "$rsub" "$rnum" "$A1_EP"
    train_one a2 "$forget" "$rsub" "$rnum" "$A2_EP"
done

cd "$OU_REPO"
echo
echo "[Phase C] Eval Dual-ULD v2"
for sp in "${SPLITS[@]}"; do
    forget=$(echo "$sp" | awk '{print $1}')
    holdout=$(echo "$sp" | awk '{print $2}')
    retain=$(echo "$sp" | awk '{print $3}')
    a1=$(find "${MODELS_ROOT}/a1_${forget}" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    a2=$(find "${MODELS_ROOT}/a2_${forget}" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    task="tofu_Llama-3.2-1B-Instruct_${forget}_DualULD_v2"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  → SKIP $forget: exists"; continue; }
    retain_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json"
    echo "  → Eval Dual v2 on $forget → $eval_json"
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

echo "============================================================"
echo "DONE — see saves/eval/*_DualULD_v2/"
echo "============================================================"
