#!/usr/bin/env bash
# Plan L: 5 untested dimensions
# L1: A1 lr=2e-3 (vs current best 1e-3)
# L2: A1 retain_weight=2 (vs default 5)
# L3: A2 num_layer=8 (decoupled, A1 stays num_layer=4)
# L4: A2 LoRA r=32 (decoupled, A1 stays r=16)
# L5: R_sub k=40 (vs current k=80, need regenerate)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

MODELS_L="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_L"
mkdir -p "$MODELS_L"

# Find best A1 from Plan F (a1s600) — used by L3, L4
A1_F600=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a1_forget10 -name "checkpoint-600" -type d | head -1)
# Best A2 from Plan H/I (a2ep10)
A2_EP10=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "Plan F A1 (step 600): $A1_F600"
echo "Plan H A2 ep=10:      $A2_EP10"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

# ============================================================
# Helpers: train and FQ-only eval
# ============================================================
train_a1() {
    local label="$1" lr="$2" rw="$3" lora_r="${4:-16}"
    local out="${MODELS_L}/${label}/a1_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A1/$label : exists"; return
    fi
    echo "  → Train A1/$label (lr=$lr, rw=$rw, lora_r=$lora_r)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_L_${label}_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 \
        data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r="$lora_r" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight="$rw" \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate="$lr" trainer.max_epochs=8 \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="L${label}a1" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_L_${label}_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

train_a2() {
    local label="$1" num_layer="$2" lora_r="$3" rsub_path="$4"
    local out="${MODELS_L}/${label}/a2_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A2/$label : exists"; return
    fi
    echo "  → Train A2/$label (num_layer=$num_layer, lora_r=$lora_r, rsub=$rsub_path)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_L_${label}_a2_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a2 \
        data_mode.r_sub_indices_path="$rsub_path" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$num_layer" model_mode.Lora.r="$lora_r" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=10 \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="L${label}a2" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_L_${label}_a2_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        echo "  SKIP $task (exists)"
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "    FQ=$fq"
        echo "$task $fq" >> /tmp/3b_L_fq_screen.txt
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
    echo "$task $fq" >> /tmp/3b_L_fq_screen.txt
    cd "$ULD_REPO"
}

> /tmp/3b_L_fq_screen.txt

# ============================================================
echo "=== L1: A1 lr=2e-3 ==="
train_a1 L1_lr2e3 2e-3 5 16
A1_L1=$(find "${MODELS_L}/L1_lr2e3/a1_forget10" -name "checkpoint-600" -type d | head -1)
[ -z "$A1_L1" ] && A1_L1=$(find "${MODELS_L}/L1_lr2e3/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1_L1" "$A2_EP10" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2L_lr2e3_FQONLY

# ============================================================
echo "=== L2: A1 retain_weight=2 ==="
train_a1 L2_rw2 1e-3 2 16
A1_L2=$(find "${MODELS_L}/L2_rw2/a1_forget10" -name "checkpoint-600" -type d | head -1)
[ -z "$A1_L2" ] && A1_L2=$(find "${MODELS_L}/L2_rw2/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1_L2" "$A2_EP10" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2L_rw2_FQONLY

# ============================================================
echo "=== L3: A2 num_layer=8 (A1 = Plan F a1s600) ==="
train_a2 L3_a2nl8 8 16 "${ULD_REPO}/data/rsub/forget10_k80.json"
A2_L3=$(find "${MODELS_L}/L3_a2nl8/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1_F600" "$A2_L3" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2L_a2nl8_FQONLY

# ============================================================
echo "=== L4: A2 LoRA r=32 ==="
train_a2 L4_a2r32 4 32 "${ULD_REPO}/data/rsub/forget10_k80.json"
A2_L4=$(find "${MODELS_L}/L4_a2r32/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1_F600" "$A2_L4" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2L_a2r32_FQONLY

# ============================================================
echo "=== L5: R_sub k=40 ==="
RSUB_K40="${ULD_REPO}/data/rsub/forget10_k40.json"
if [ ! -f "$RSUB_K40" ]; then
    echo "  → Generating R_sub k=40"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/select_rsub.py \
        --forget_split forget10_perturbed --retain_split retain90 \
        --retain_num 400 --k 40 \
        --out "$RSUB_K40"
fi
train_a1 L5_rsubk40 1e-3 5 16  # uses k=80 by default — need explicit override
# Custom A1 train for L5 with k=40
A1_L5_OUT="${MODELS_L}/L5_rsubk40_proper/a1_forget10"
if ! find "$A1_L5_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "  → Train A1/L5 with k=40 (overrides train_a1 path)"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_L_L5_rsubk40_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="$RSUB_K40" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=8 \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A1_L5_OUT" postfix=LL5a1 \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_L_L5_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi
A2_L5_OUT="${MODELS_L}/L5_rsubk40_proper/a2_forget10"
if ! find "$A2_L5_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "  → Train A2/L5 with k=40"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_L_L5_rsubk40_a2_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="$RSUB_K40" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=10 \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A2_L5_OUT" postfix=LL5a2 \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_L_L5_a2_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi
A1_L5=$(find "$A1_L5_OUT" -name "checkpoint-600" -type d | head -1)
[ -z "$A1_L5" ] && A1_L5=$(find "$A1_L5_OUT" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_L5=$(find "$A2_L5_OUT" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1_L5" "$A2_L5" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2L_rsubk40_FQONLY

# ============================================================
echo "=== Plan L FQ summary (sorted) ==="
sort -k2 -t' ' -gr /tmp/3b_L_fq_screen.txt
echo "============================================================"

# Check best
best_fq=$(awk '{print $2}' /tmp/3b_L_fq_screen.txt | sort -gr | head -1)
echo "Best Plan L FQ: $best_fq"
echo "Current overall best: 0.416 (a2ep10)"
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.416)}'; then
    echo "*** Plan L beat 0.416, full eval needed ***"
fi
echo "DONE"
