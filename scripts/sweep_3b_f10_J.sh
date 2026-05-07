#!/usr/bin/env bash
# 3B f10 Plan J: train A1 with finer save_steps=25 + sweep mid-ckpts with best A2.
# Best A2 picked from Plan H/I (ep=10 or higher if I2/I3 wins).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
FQ_THRESHOLD=0.5

# Auto-pick BEST A2 from H + I screens
echo "=== Picking best A2 from Plan H/I FQ screens ==="
BEST_A2_TASK=$(cat /tmp/3b_H_fq_screen.txt /tmp/3b_I_fq_screen.txt 2>/dev/null \
    | grep -E "a2ep" | sort -k2 -gr | head -1 | awk '{print $1}')
echo "Best A2 task: $BEST_A2_TASK"

# Map task name to A2 ckpt path
if [[ "$BEST_A2_TASK" == *"v2H_a2ep10"* ]]; then
    A2_BEST=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
elif [[ "$BEST_A2_TASK" == *"v2I_a2ep15"* ]]; then
    A2_BEST=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_I_a2ep15/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
elif [[ "$BEST_A2_TASK" == *"v2I_a2ep20"* ]]; then
    A2_BEST=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_I_a2ep20/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
else
    # Fallback to ep=10
    A2_BEST=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
fi
echo "A2 ckpt:  $A2_BEST"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

# === Train A1 with save_steps=25 (24 ckpts: 25/50/.../600) ===
A1_OUT="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_J/a1_forget10"
if ! find "$A1_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[J train] A1 ep=8, save_steps=25"
    mkdir -p "$A1_OUT"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_J_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=8 \
        +trainer.save_steps=25 trainer.strategy=gpu \
        OUTPUTMODELDIR="$A1_OUT" postfix="Ja1" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_J_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi

A1_PARENT=$(find "$A1_OUT" -name "checkpoint-*" -type d | head -1 | xargs dirname)
echo "A1 parent: $A1_PARENT"

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
    echo "$task $fq" >> /tmp/3b_J_fq_screen.txt
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

> /tmp/3b_J_fq_screen.txt

echo "[J FQ-only] Sweep A1 step={475,500,525,550,575,600} with best A2"
for s in 475 500 525 550 575 600; do
    a1="${A1_PARENT}/checkpoint-${s}"
    [ -d "$a1" ] || { echo "  miss step=$s"; continue; }
    fq_only_eval "$a1" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2J_a1s${s}_FQONLY"
done

echo
echo "=== FQ screen results ==="
sort -k2 -gr /tmp/3b_J_fq_screen.txt | head

# Run full eval for ckpt with FQ > current best (0.416)
best_fq=$(sort -k2 -gr /tmp/3b_J_fq_screen.txt | head -1 | awk '{print $2}')
best_task=$(sort -k2 -gr /tmp/3b_J_fq_screen.txt | head -1 | awk '{print $1}')
if awk -v f="$best_fq" 'BEGIN{exit !(f+0 > 0.416)}'; then
    echo "Best $best_task FQ=$best_fq beats 0.416, running full eval"
    step=$(echo "$best_task" | sed -E 's/.*a1s([0-9]+).*/\1/')
    a1="${A1_PARENT}/checkpoint-${step}"
    new_task=$(echo "$best_task" | sed 's/_FQONLY//')
    full_eval "$a1" "$new_task"
fi

echo "============================================================"
echo "Plan J DONE"
echo "============================================================"
