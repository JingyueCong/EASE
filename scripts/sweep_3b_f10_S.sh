#!/usr/bin/env bash
# Plan S: fair num_layer grid scan with Plan F recipe + A2 ep=10
# All same: ep=8 (A1) / ep=10 (A2), lr=1e-3, bs=4/ga=4, r=16, save_steps=75, w2=0.5
# Vary: num_layer ∈ {3, 5, 6, 7, 8 (revisit)}
set -uo pipefail
ROOT="${EASE_ROOT}"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU=0
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
MODELS_S="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_S"
mkdir -p "$MODELS_S"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a() {
    local role="$1" nl="$2" ep="$3"
    local out="${MODELS_S}/nl${nl}/a${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A${role}/nl${nl} : exists"; return
    fi
    echo "  → Train A${role}/nl${nl} (num_layer=$nl, ep=$ep, lr=1e-3, r=16)"
    local data_role="dual_a${role}"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_S_nl${nl}_a${role}_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode="$data_role" \
        data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$nl" model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs="$ep" \
        +trainer.save_steps=75 \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="Snl${nl}a${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_S_nl${nl}_a${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" task="$3"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "  SKIP $task FQ=$fq"
        echo "$task $fq" >> /tmp/3b_S_fq.txt
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
    echo "$task $fq" >> /tmp/3b_S_fq.txt
    cd "$ULD_REPO"
}

> /tmp/3b_S_fq.txt

echo "============================================================"
echo "Plan S: fair num_layer scan {3,5,6,7,8} with A2 ep=10"
echo "============================================================"

for nl in 3 5 6 7 8; do
    echo ""
    echo "============================================================"
    echo "=== num_layer=${nl} ==="
    echo "============================================================"
    train_a 1 "$nl" 8
    train_a 2 "$nl" 10
    A1=$(find "${MODELS_S}/nl${nl}/a1_forget10" -name "checkpoint-600" -type d | head -1)
    [ -z "$A1" ] && A1=$(find "${MODELS_S}/nl${nl}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    A2=$(find "${MODELS_S}/nl${nl}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    fq_only_eval "$A1" "$A2" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2S_nl${nl}_FQONLY"
done

echo ""
echo "============================================================"
echo "Plan S FQ summary (sorted):"
sort -k2 -t' ' -gr /tmp/3b_S_fq.txt
echo "============================================================"
echo "Reference: nl=4 (Plan F+I a2ep10) FQ=0.4158"
best_fq=$(awk '{print $2}' /tmp/3b_S_fq.txt | sort -gr | head -1)
echo "Best Plan S FQ: $best_fq"
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.4158)}'; then
    echo "*** Plan S beat 0.416 ***"
fi
echo "DONE"
