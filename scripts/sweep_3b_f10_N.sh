#!/usr/bin/env bash
# Plan N: 8-layer truncation at various positions (follow-up to Plan M's 4-layer ablation)
#   N1: layers [0..7]    = first 8 (= Plan H4 baseline, FQ=0.281)
#   N2: layers [10..17]  = middle 8
#   N3: layers [14..21]  = late-middle 8 (Geva et al. factual layer range for 28-layer model)
#   N4: layers [6..13]   = early-middle 8
#   N5: layers [20..27]  = last 8 (= Plan M M7, dedup if exists)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

MODELS_N="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_N"
mkdir -p "$MODELS_N"

echo "============================================================"
echo "Plan N: 8-layer truncation position ablation (3B, 28 layers)"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a1_layers() {
    local label="$1" layer_indices="$2"
    local out="${MODELS_N}/${label}/a1_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A1/$label : exists"; return
    fi
    echo "  → Train A1/$label (layer_indices=$layer_indices)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_N_${label}_a1_forget10" \
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
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="N${label}a1" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_N_${label}_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

train_a2_layers() {
    local label="$1" layer_indices="$2"
    local out="${MODELS_N}/${label}/a2_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A2/$label : exists"; return
    fi
    echo "  → Train A2/$label (layer_indices=$layer_indices)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_N_${label}_a2_forget10" \
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
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="N${label}a2" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_N_${label}_a2_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        echo "  SKIP $task (exists)"
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "    FQ=$fq"
        echo "$task $fq" >> /tmp/3b_N_fq_screen.txt
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
    echo "$task $fq" >> /tmp/3b_N_fq_screen.txt
    cd "$ULD_REPO"
}

> /tmp/3b_N_fq_screen.txt

# 8-layer positions (skip first 8 = Plan H4 known FQ=0.281, skip last 8 = Plan M M7)
declare -A CONFIGS=(
    ["mid8"]="[10,11,12,13,14,15,16,17]"
    ["latemid8"]="[14,15,16,17,18,19,20,21]"
    ["earlymid8"]="[6,7,8,9,10,11,12,13]"
    ["postmid8"]="[16,17,18,19,20,21,22,23]"
)

for label in latemid8 mid8 postmid8 earlymid8; do
    layers="${CONFIGS[$label]}"
    echo ""
    echo "============================================================"
    echo "=== N_${label}: layers ${layers} ==="
    echo "============================================================"
    train_a1_layers "$label" "$layers"
    train_a2_layers "$label" "$layers"
    A1=$(find "${MODELS_N}/${label}/a1_forget10" -name "checkpoint-600" -type d | head -1)
    [ -z "$A1" ] && A1=$(find "${MODELS_N}/${label}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    A2=$(find "${MODELS_N}/${label}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    fq_only_eval "$A1" "$A2" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2N_${label}_FQONLY"
done

echo ""
echo "============================================================"
echo "=== Plan N FQ summary (sorted) ==="
echo "============================================================"
echo "(reference) Plan H4 first8 FQ=0.281"
sort -k2 -t' ' -gr /tmp/3b_N_fq_screen.txt
echo "============================================================"

best_fq=$(awk '{print $2}' /tmp/3b_N_fq_screen.txt | sort -gr | head -1)
echo "Best Plan N FQ: $best_fq"
echo "Current overall best: 0.4158"
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.4158)}'; then
    echo "*** Plan N beat 0.416, full eval recommended ***"
fi
echo "DONE"
