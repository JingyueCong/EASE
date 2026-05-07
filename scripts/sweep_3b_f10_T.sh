#!/usr/bin/env bash
# Plan T: last4 [24-27] + strong params (ep=20, r=64) — fair test
# Mirror of Plan O O1 (last8 ep=20 r=64) but with 4 layers from end
set -uo pipefail
ROOT="${EASE_ROOT}"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU=0
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
MODELS_T="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_T"
mkdir -p "$MODELS_T"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a() {
    local role="$1" label="$2" layer_indices="$3" ep="$4" lr="$5" lora_r="$6"
    local out="${MODELS_T}/${label}/a${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A${role}/$label : exists"; return
    fi
    echo "  → Train A${role}/$label (layers=$layer_indices, ep=$ep, lr=$lr, r=$lora_r)"
    local data_role="dual_a${role}"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_T_${label}_a${role}_forget10" \
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
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="T${label}a${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_T_${label}_a${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" w2="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "  SKIP $task FQ=$fq"
        echo "$task $fq" >> /tmp/3b_T_fq.txt
        return
    }
    rm -f "$eval_json" 2>/dev/null
    echo "  FQ-screen $task (w2=$w2)"
    cd "$OU_REPO"
    local fqlog="/tmp/fq_eval_${task}.log"
    rm -f "$fqlog" 2>/dev/null
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task" > "$fqlog" 2>&1 &
    local pid=$!
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
    echo "    FQ=$fq"
    echo "$task $fq" >> /tmp/3b_T_fq.txt
    cd "$ULD_REPO"
}

> /tmp/3b_T_fq.txt

echo "============================================================"
echo "Plan T: last4 [24-27] + ep=20 + r=64 (mirror of O1)"
echo "============================================================"

# Train A1 (ep=20, save_steps=75) and A2 (ep=10)
train_a 1 last4_strong "[24,25,26,27]" 20 1e-3 64
train_a 2 last4_strong "[24,25,26,27]" 10 1e-3 64

A1_PARENT=$(find "${MODELS_T}/last4_strong/a1_forget10" -name "checkpoint-75" -type d | head -1 | xargs dirname)
A2=$(find "${MODELS_T}/last4_strong/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 parent: $A1_PARENT"
echo "A2 final:  $A2"

# Phase 1: A1 ckpt scan with w2=0.5
echo ""
echo "=== Phase 1: A1 ckpt scan (steps 600/750/900/1050/1200/1500) w2=0.5 ==="
for step in 600 750 900 1050 1200 1500; do
    A1="${A1_PARENT}/checkpoint-${step}"
    fq_only_eval "$A1" "$A2" 0.5 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2T_last4_a1s${step}_FQONLY"
done

# Phase 2: w2 sweep at the best ckpt found in Phase 1
echo ""
echo "=== Phase 2: w2 sweep at best A1 ckpt ==="
BEST_TASK=$(sort -k2 -t' ' -gr /tmp/3b_T_fq.txt | head -1 | awk '{print $1}')
BEST_STEP=$(echo "$BEST_TASK" | grep -oE 'a1s[0-9]+' | grep -oE '[0-9]+')
BEST_A1="${A1_PARENT}/checkpoint-${BEST_STEP}"
echo "Best step from phase 1: $BEST_STEP"
for w2 in 0.3 0.7 0.8 1.0; do
    w2_str=$(echo "$w2" | tr '.' 'p')
    fq_only_eval "$BEST_A1" "$A2" "$w2" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2T_last4_a1s${BEST_STEP}_w20${w2_str}_FQONLY"
done

echo "============================================================"
echo "Plan T FQ summary (sorted):"
sort -k2 -t' ' -gr /tmp/3b_T_fq.txt
echo "============================================================"
echo "Reference: first4 (Plan F+I a2ep10) FQ=0.4158, last8 strong (Plan O+Q) FQ=0.3222"
best_fq=$(awk '{print $2}' /tmp/3b_T_fq.txt | sort -gr | head -1)
echo "Best Plan T FQ: $best_fq"
echo "DONE"
