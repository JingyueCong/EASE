#!/usr/bin/env bash
# Plan O: stronger-parameter retest of non-first layer truncations
# Hypothesis: with longer training + larger LoRA, A1 in non-first positions can adapt
#   O1: last8 [20-27] + ep=20 + LoRA r=64
#   O2: latemid4 [18-21] + ep=20 + LoRA r=64
#   O3: last8 + ep=8 + lr=5e-4 + LoRA r=64 (slow stable lr)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

MODELS_O="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_O"
mkdir -p "$MODELS_O"

echo "============================================================"
echo "Plan O: stronger params on non-first layer truncations"
echo "============================================================"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a() {
    local role="$1" label="$2" layer_indices="$3" ep="$4" lr="$5" lora_r="$6"
    local out="${MODELS_O}/${label}/a${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A${role}/$label : exists"; return
    fi
    echo "  → Train A${role}/$label (layers=$layer_indices, ep=$ep, lr=$lr, r=$lora_r)"
    local data_role="dual_a${role}"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_O_${label}_a${role}_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode="$data_role" \
        data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=0 model_mode.Lora.r="$lora_r" \
        +model_mode.layer_indices="$layer_indices" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate="$lr" trainer.max_epochs="$ep" \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="O${label}a${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_O_${label}_a${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        echo "  SKIP $task (exists)"
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "    FQ=$fq"
        echo "$task $fq" >> /tmp/3b_O_fq_screen.txt
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
    echo "$task $fq" >> /tmp/3b_O_fq_screen.txt
    cd "$ULD_REPO"
}

> /tmp/3b_O_fq_screen.txt

# ============================================================
# O1: last8 + ep=20 + LoRA r=64 + lr=1e-3
# ============================================================
echo ""
echo "============================================================"
echo "=== O1: last8 [20-27] + ep=20 + LoRA r=64 ==="
echo "============================================================"
train_a 1 O1_last8_ep20_r64 "[20,21,22,23,24,25,26,27]" 20 1e-3 64
train_a 2 O1_last8_ep20_r64 "[20,21,22,23,24,25,26,27]" 10 1e-3 64
A1=$(find "${MODELS_O}/O1_last8_ep20_r64/a1_forget10" -name "checkpoint-600" -type d | head -1)
[ -z "$A1" ] && A1=$(find "${MODELS_O}/O1_last8_ep20_r64/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2=$(find "${MODELS_O}/O1_last8_ep20_r64/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1" "$A2" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2O_last8_ep20_r64_FQONLY

# ============================================================
# O2: latemid4 + ep=20 + LoRA r=64
# ============================================================
echo ""
echo "============================================================"
echo "=== O2: latemid4 [18-21] + ep=20 + LoRA r=64 ==="
echo "============================================================"
train_a 1 O2_latemid4_ep20_r64 "[18,19,20,21]" 20 1e-3 64
train_a 2 O2_latemid4_ep20_r64 "[18,19,20,21]" 10 1e-3 64
A1=$(find "${MODELS_O}/O2_latemid4_ep20_r64/a1_forget10" -name "checkpoint-600" -type d | head -1)
[ -z "$A1" ] && A1=$(find "${MODELS_O}/O2_latemid4_ep20_r64/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2=$(find "${MODELS_O}/O2_latemid4_ep20_r64/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1" "$A2" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2O_latemid4_ep20_r64_FQONLY

# ============================================================
# O3: last8 + ep=8 + lr=5e-4 + LoRA r=64 (stable lr)
# ============================================================
echo ""
echo "============================================================"
echo "=== O3: last8 + ep=8 + lr=5e-4 + LoRA r=64 ==="
echo "============================================================"
train_a 1 O3_last8_lr5e4_r64 "[20,21,22,23,24,25,26,27]" 8 5e-4 64
train_a 2 O3_last8_lr5e4_r64 "[20,21,22,23,24,25,26,27]" 10 5e-4 64
A1=$(find "${MODELS_O}/O3_last8_lr5e4_r64/a1_forget10" -name "checkpoint-600" -type d | head -1)
[ -z "$A1" ] && A1=$(find "${MODELS_O}/O3_last8_lr5e4_r64/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2=$(find "${MODELS_O}/O3_last8_lr5e4_r64/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fq_only_eval "$A1" "$A2" tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2O_last8_lr5e4_r64_FQONLY

# ============================================================
echo ""
echo "============================================================"
echo "=== Plan O FQ summary (sorted) ==="
echo "============================================================"
echo "(reference) Plan M last8 default FQ=3.1e-7"
sort -k2 -t' ' -gr /tmp/3b_O_fq_screen.txt
echo "============================================================"

best_fq=$(awk '{print $2}' /tmp/3b_O_fq_screen.txt | sort -gr | head -1)
echo "Best Plan O FQ: $best_fq"
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.4158)}'; then
    echo "*** Plan O beat 0.416 ***"
fi
echo "DONE"
