#!/usr/bin/env bash
# Plan Q: fine ckpt scan around last8 peak (a1s900 = 0.21)
# Test step=675, 750, 825, 975, 1050 + retest a1s900 with diff w2
set -uo pipefail
ROOT="${EASE_ROOT}"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU=0
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
MODELS_O="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_O"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"

fq_only_eval() {
    local a1="$1" a2="$2" w2="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ] && {
        local fq=$(/usr/bin/python -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "  SKIP $task FQ=$fq"
        echo "$task $fq" >> /tmp/3b_Q_fq.txt
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
    echo "$task $fq" >> /tmp/3b_Q_fq.txt
    cd "$ULD_REPO"
}

> /tmp/3b_Q_fq.txt

# A2 ckpt = O1 last8 A2 final
A2_O1=$(find "${MODELS_O}/O1_last8_ep20_r64/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_O1_PARENT=$(find "${MODELS_O}/O1_last8_ep20_r64/a1_forget10" -name "checkpoint-75" -type d | head -1 | xargs dirname)
echo "A1 parent: $A1_O1_PARENT"
echo "A2 final:  $A2_O1"
echo "============================================================"

# Phase 1: fine-step scan around peak (a1s900 = 0.21)
echo ""
echo "=== Phase 1: fine-step scan (steps 675/750/825/975/1050) ==="
for step in 675 750 825 975 1050; do
    A1="${A1_O1_PARENT}/checkpoint-${step}"
    fq_only_eval "$A1" "$A2_O1" 0.5 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2Q_last8_a1s${step}_FQONLY"
done

# Phase 2: w2 sweep on a1s900 (current peak)
echo ""
echo "=== Phase 2: w2 sweep on a1s900 ==="
A1_900="${A1_O1_PARENT}/checkpoint-900"
for w2 in 0.3 0.7 0.8; do
    w2_str=$(echo "$w2" | tr '.' 'p')
    fq_only_eval "$A1_900" "$A2_O1" "$w2" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2Q_last8_a1s900_w20${w2_str}_FQONLY"
done

echo "============================================================"
echo "Plan Q FQ summary (sorted):"
sort -k2 -t' ' -gr /tmp/3b_Q_fq.txt
echo "============================================================"
best_fq=$(awk '{print $2}' /tmp/3b_Q_fq.txt | sort -gr | head -1)
echo "Best Plan Q FQ: $best_fq"
echo "DONE"
