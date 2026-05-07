#!/usr/bin/env bash
# 3B f10 Plan K: A1 ep=10/ep=12/LoRA r=32 ablations + FQ-only screen with best A2.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
FQ_THRESHOLD=0.416   # current best, only beat-this matters

# Reuse best A2 (ep=10 from Plan H)
A2_BEST=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "Using A2 (ep=10): $A2_BEST"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a1() {
    local tag="$1" eps="$2" lora_r="$3"
    local out="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_K_${tag}/a1_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A1/$tag: exists"; return
    fi
    echo "  → Train A1/$tag (ep=$eps, LoRA r=$lora_r)"
    mkdir -p "$out"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_K_${tag}_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r="$lora_r" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs="$eps" \
        +trainer.save_steps=25 trainer.strategy=gpu \
        OUTPUTMODELDIR="$out" postfix="K${tag}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_K_${tag}_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}

fq_only_eval() {
    local a1="$1" task="$2"
    local outdir="${OU_REPO}/saves/eval/${task}"
    local fqlog="/tmp/fq_eval_${task}.log"
    rm -rf "$outdir"; rm -f "$fqlog"
    echo "  FQ-screen $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_BEST" \
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
    echo "$task $fq" >> /tmp/3b_K_fq_screen.txt
}

full_eval() {
    local a1="$1" task="$2"
    local outdir="${OU_REPO}/saves/eval/${task}"
    rm -rf "$outdir"
    echo "  FULL-eval $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_BEST" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2=0.5 \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
}

> /tmp/3b_K_fq_screen.txt

# === K1: A1 ep=10 ===
echo "=== K1: A1 ep=10 (ep=10, save_steps=25, lora_r=16) ==="
train_a1 "ep10" 10 16
A1_K1_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_K_ep10/a1_forget10 -name "checkpoint-*" -type d | head -1 | xargs dirname)
for s in 600 675 750; do
    a1="${A1_K1_PARENT}/checkpoint-${s}"
    [ -d "$a1" ] || { echo "  miss step=$s"; continue; }
    fq_only_eval "$a1" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2K_ep10_a1s${s}_FQONLY"
done

# === K2: A1 ep=12 ===
echo "=== K2: A1 ep=12 ==="
train_a1 "ep12" 12 16
A1_K2_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_K_ep12/a1_forget10 -name "checkpoint-*" -type d | head -1 | xargs dirname)
for s in 750 825 900; do
    a1="${A1_K2_PARENT}/checkpoint-${s}"
    [ -d "$a1" ] || { echo "  miss step=$s"; continue; }
    fq_only_eval "$a1" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2K_ep12_a1s${s}_FQONLY"
done

# === K3: A1 LoRA r=32, ep=8 ===
echo "=== K3: A1 LoRA r=32, ep=8 ==="
train_a1 "lora32" 8 32
A1_K3_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_K_lora32/a1_forget10 -name "checkpoint-*" -type d | head -1 | xargs dirname)
for s in 525 575 600; do
    a1="${A1_K3_PARENT}/checkpoint-${s}"
    [ -d "$a1" ] || { echo "  miss step=$s"; continue; }
    fq_only_eval "$a1" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2K_lora32_a1s${s}_FQONLY"
done

echo
echo "=== Plan K FQ summary (sorted) ==="
sort -k2 -gr /tmp/3b_K_fq_screen.txt | head

# Full eval the winner if FQ > 0.416
best_fq=$(sort -k2 -gr /tmp/3b_K_fq_screen.txt | head -1 | awk '{print $2}')
best_task=$(sort -k2 -gr /tmp/3b_K_fq_screen.txt | head -1 | awk '{print $1}')
if awk -v f="$best_fq" -v t="$FQ_THRESHOLD" 'BEGIN{exit !(f+0 > t)}'; then
    echo "*** Best $best_task FQ=$best_fq beats $FQ_THRESHOLD, running full eval ***"
    # Recover ckpt path
    if [[ "$best_task" == *"_ep10_a1s"* ]]; then PAR="$A1_K1_PARENT"
    elif [[ "$best_task" == *"_ep12_a1s"* ]]; then PAR="$A1_K2_PARENT"
    elif [[ "$best_task" == *"_lora32_a1s"* ]]; then PAR="$A1_K3_PARENT"
    fi
    step=$(echo "$best_task" | sed -E 's/.*a1s([0-9]+).*/\1/')
    new_task=$(echo "$best_task" | sed 's/_FQONLY//')
    full_eval "${PAR}/checkpoint-${step}" "$new_task"
fi

echo "============================================================"
echo "Plan K DONE"
echo "============================================================"
