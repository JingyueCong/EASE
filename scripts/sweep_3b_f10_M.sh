#!/usr/bin/env bash
# Plan M: layer-truncation position ablation (the architectural fix)
# Currently: take FIRST num_layer layers. Test other positions in 3B's 28 layers.
#   M1: layers [0,1,2,3]      = first 4 (= Plan F baseline, FQ=0.416)
#   M2: layers [24,25,26,27]  = last 4
#   M3: layers [12,13,14,15]  = middle 4
#   M4: layers [18,19,20,21]  = late-middle 4
#   M5: layers [8,9,10,11]    = early-middle 4
#   M6: layers [0,9,18,27]    = spaced (full-coverage probe)
#   M7: layers [20..27]       = last 8 (bigger end-of-model assistant)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

MODELS_M="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_M"
mkdir -p "$MODELS_M"

echo "============================================================"
echo "Plan M: layer-truncation position ablation (3B, 28 layers)"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a1_layers() {
    local label="$1" layer_indices="$2"
    local out="${MODELS_M}/${label}/a1_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A1/$label : exists"; return
    fi
    echo "  → Train A1/$label (layer_indices=$layer_indices)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_M_${label}_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 \
        data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=0 model_mode.Lora.r=16 \
        +model_mode.layer_indices="$layer_indices" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=8 \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="M${label}a1" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_M_${label}_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

train_a2_layers() {
    local label="$1" layer_indices="$2"
    local out="${MODELS_M}/${label}/a2_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A2/$label : exists"; return
    fi
    echo "  → Train A2/$label (layer_indices=$layer_indices)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_M_${label}_a2_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a2 \
        data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=0 model_mode.Lora.r=16 \
        +model_mode.layer_indices="$layer_indices" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=10 \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="M${label}a2" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_M_${label}_a2_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        echo "  SKIP $task (exists)"
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "    FQ=$fq"
        echo "$task $fq" >> /tmp/3b_M_fq_screen.txt
        return
    }
    rm -f "$eval_json" 2>/dev/null
    echo "  FQ-screen $task"
    cd "$OU_REPO"
    local fqlog="/tmp/fq_eval_${task}.log"
    rm -f "$fqlog" 2>/dev/null
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2=0.5 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task" > "$fqlog" 2>&1 &
    local pid=$!
    local start=$SECONDS
    while kill -0 $pid 2>/dev/null; do
        if grep -q "Result for metric forget_quality:" "$fqlog" 2>/dev/null; then
            sleep 2
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
            break
        fi
        sleep 5
    done
    local fq=$(grep "Result for metric forget_quality:" "$fqlog" 2>/dev/null | tail -1 | sed -E 's/.*forget_quality:[[:space:]]+//' | tr -d ' ')
    [ -z "$fq" ] && fq="-1"
    echo "    FQ=$fq  (elapsed $((SECONDS-start))s)"
    echo "$task $fq" >> /tmp/3b_M_fq_screen.txt
    cd "$ULD_REPO"
}

> /tmp/3b_M_fq_screen.txt

# Configurations to test (skip M1 = first 4, already have FQ=0.416 from Plan F+I)
declare -A CONFIGS=(
    ["last4"]="[24,25,26,27]"
    ["mid4"]="[12,13,14,15]"
    ["latemid4"]="[18,19,20,21]"
    ["earlymid4"]="[8,9,10,11]"
    ["spaced4"]="[0,9,18,27]"
    ["last8"]="[20,21,22,23,24,25,26,27]"
)

# Add baseline reference (already-known FQ from previous experiments)
echo "=== M1: first4 (BASELINE, known FQ=0.416 from Plan I a2ep10) ==="

for label in last4 mid4 latemid4 earlymid4 spaced4 last8; do
    layers="${CONFIGS[$label]}"
    echo ""
    echo "============================================================"
    echo "=== M_${label}: layers ${layers} ==="
    echo "============================================================"
    train_a1_layers "$label" "$layers"
    train_a2_layers "$label" "$layers"
    A1=$(find "${MODELS_M}/${label}/a1_forget10" -name "checkpoint-600" -type d | head -1)
    [ -z "$A1" ] && A1=$(find "${MODELS_M}/${label}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    A2=$(find "${MODELS_M}/${label}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    fq_only_eval "$A1" "$A2" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2M_${label}_FQONLY"
done

echo ""
echo "============================================================"
echo "=== Plan M FQ summary (sorted) ==="
echo "============================================================"
echo "(reference) v2I_a2ep10_full FQ=0.4158 (first4)"
sort -k2 -t' ' -gr /tmp/3b_M_fq_screen.txt
echo "============================================================"

# Best
best_fq=$(awk '{print $2}' /tmp/3b_M_fq_screen.txt | sort -gr | head -1)
echo "Best Plan M FQ: $best_fq"
echo "Current overall best: 0.4158"
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.4158)}'; then
    echo "*** Plan M beat 0.416, full eval recommended ***"
fi
echo "DONE"
