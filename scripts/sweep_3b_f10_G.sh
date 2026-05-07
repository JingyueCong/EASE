#!/usr/bin/env bash
# 3B f10 Plan G: w1 ablation on Plan F a1s600 (best A1) with FQ-only screening.
# Hypothesis: 3B's sharper logits need stronger |w1| than 1B's -0.8.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
FQ_THRESHOLD=0.6

# Reuse Plan F's best A1 and final A2
A1_F600=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a1_forget10 -name "checkpoint-600" -type d | head -1)
A2_PATH=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "============================================================"
echo "3B Plan G: w1 ablation, FQ-only screen"
echo "  A1: $A1_F600"
echo "  A2: $A2_PATH"
echo "============================================================"

fq_only_eval() {
    local w1="$1" task="$2"
    local outdir="${OU_REPO}/saves/eval/${task}"
    local fqlog="/tmp/fq_eval_${task}.log"
    rm -rf "$outdir"; rm -f "$fqlog"

    echo "  FQ-screen $task (w1=$w1)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$A1_F600" model.model_args.a2_path="$A2_PATH" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2=0.5 \
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
            kill $pid 2>/dev/null; wait $pid 2>/dev/null
            break
        fi
        if (( SECONDS - start > 1800 )); then
            echo "  TIMEOUT $task"
            kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
            return
        fi
        sleep 5
    done

    local fq=$(grep "Result for metric forget_quality:" "$fqlog" | tail -1 | sed -E 's/.*forget_quality:[[:space:]]*//; s/\x1b\[[0-9;]*m//g')
    echo "    FQ=$fq  (elapsed ${SECONDS}s)"
    echo "$task w1=$w1 $fq" >> /tmp/3b_G_fq_screen.txt
}

full_eval() {
    local w1="$1" task="$2"
    local outdir="${OU_REPO}/saves/eval/${task}"
    rm -rf "$outdir"
    echo "  FULL-eval $task (w1=$w1)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$A1_F600" model.model_args.a2_path="$A2_PATH" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2=0.5 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
}

> /tmp/3b_G_fq_screen.txt

echo "[Phase 1] FQ-only screen w1 ∈ {-0.5, -0.7, -1.0, -1.2, -1.5}"
for w1 in -0.5 -0.7 -1.0 -1.2 -1.5; do
    tag=$(echo "$w1" | sed 's/-/m/; s/\./p/')
    fq_only_eval "$w1" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2G_w1${tag}_w20p5_FQONLY"
done

echo
echo "=== FQ screen results (sorted) ==="
sort -k3 -gr /tmp/3b_G_fq_screen.txt | head -10

echo
echo "[Phase 2] Full eval on candidates with FQ > $FQ_THRESHOLD"
while read task w1str fq; do
    if awk -v f="$fq" -v t="$FQ_THRESHOLD" 'BEGIN{exit !(f+0 > t)}'; then
        w1=$(echo "$w1str" | sed 's/w1=//')
        new_task=$(echo "$task" | sed 's/_FQONLY//')
        full_eval "$w1" "$new_task"
    fi
done < /tmp/3b_G_fq_screen.txt

echo "============================================================"
echo "Plan G DONE — see /tmp/3b_G_fq_screen.txt"
echo "============================================================"
