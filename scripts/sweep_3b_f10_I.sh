#!/usr/bin/env bash
# 3B f10 Plan I: Full eval on best (H3 A2 ep=10) + push A2 to ep=15, ep=20.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

A1_F600=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a1_forget10 -name "checkpoint-600" -type d | head -1)
A2_EP10=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "A1: $A1_F600"
echo "A2 ep=10: $A2_EP10"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a2() {
    local eps="$1" tag="$2"
    local out="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_I_${tag}/a2_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP a2/$tag: exists"; return
    fi
    echo "  → Train a2/$tag (ep=$eps)"
    mkdir -p "$out"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_I_${tag}_a2_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs="$eps" \
        +trainer.save_steps=75 trainer.strategy=gpu \
        OUTPUTMODELDIR="$out" postfix="I${tag}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_I_${tag}_a2_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" a2="$2" task="$3"
    local outdir="${OU_REPO}/saves/eval/${task}"
    local fqlog="/tmp/fq_eval_${task}.log"
    rm -rf "$outdir"; rm -f "$fqlog"
    echo "  FQ-screen $task"
    cd "$OU_REPO"
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
            sleep 2; kill $pid 2>/dev/null; wait $pid 2>/dev/null
            break
        fi
        if (( SECONDS - start > 1800 )); then
            kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; return
        fi
        sleep 5
    done
    local fq=$(grep "Result for metric forget_quality:" "$fqlog" | tail -1 | sed -E 's/.*forget_quality:[[:space:]]*//; s/\x1b\[[0-9;]*m//g')
    echo "    FQ=$fq"
    echo "$task $fq" >> /tmp/3b_I_fq_screen.txt
}

full_eval() {
    local a1="$1" a2="$2" task="$3"
    local outdir="${OU_REPO}/saves/eval/${task}"
    local eval_json="${outdir}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100000 ] && { echo "  SKIP $task (full eval exists)"; return; }
    rm -rf "$outdir"
    echo "  FULL-eval $task"
    cd "$OU_REPO"
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
        task_name="$task"
}

> /tmp/3b_I_fq_screen.txt

# === I1: Full eval current best (H3 A2 ep=10) — get Mem/Util/Agg ===
echo "=== I1: Full eval H3 (A2 ep=10) ==="
full_eval "$A1_F600" "$A2_EP10" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2I_a2ep10_full"

# === I2: A2 ep=15 ===
echo "=== I2: A2 ep=15 ==="
train_a2 15 "a2ep15"
A2_EP15=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_I_a2ep15/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A2_EP15" ] && fq_only_eval "$A1_F600" "$A2_EP15" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2I_a2ep15_FQONLY"

# === I3: A2 ep=20 ===
echo "=== I3: A2 ep=20 ==="
train_a2 20 "a2ep20"
A2_EP20=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_I_a2ep20/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A2_EP20" ] && fq_only_eval "$A1_F600" "$A2_EP20" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2I_a2ep20_FQONLY"

# === I4: If ep=15 or ep=20 beats ep=10, full eval the best ===
echo
echo "=== FQ summary so far ==="
sort -k2 -gr /tmp/3b_I_fq_screen.txt
best_fq=$(sort -k2 -gr /tmp/3b_I_fq_screen.txt | head -1 | awk '{print $2}')
best_task=$(sort -k2 -gr /tmp/3b_I_fq_screen.txt | head -1 | awk '{print $1}')
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.416)}'; then
    echo "Best $best_task FQ=$best_fq beats ep=10's 0.416, running full eval"
    new_task=$(echo "$best_task" | sed 's/_FQONLY//')
    a2_dir=$(echo "$best_task" | grep -oE "a2ep[0-9]+")
    a2=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_I_${a2_dir}/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    full_eval "$A1_F600" "$a2" "$new_task"
fi

echo "============================================================"
echo "Plan I DONE"
echo "============================================================"
