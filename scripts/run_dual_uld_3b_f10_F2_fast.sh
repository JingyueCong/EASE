#!/usr/bin/env bash
# 3B f10 Plan F2 (fast): FQ-only screening, then full eval on FQ>0.6 candidates.
# Kills python eval after "Result for metric forget_quality" line appears.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

NUM_LAYER=4; LORA_R=16
A1_EP=20; TRAIN_BS=4; TRAIN_GA=4; TRAIN_LR=1e-3; SAVE_STEPS=25
FQ_THRESHOLD=0.6

MODELS_F2="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F2"
A2_PATH=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a2_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

echo "============================================================"
echo "3B Plan F2-fast: extend A1 ep=$A1_EP, FQ-only screen, full eval if FQ>$FQ_THRESHOLD"
echo "  Reuse Plan F A2: $A2_PATH"
echo "============================================================"

mkdir -p "$MODELS_F2"
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

# === Train A1 ===
A1_OUT="${MODELS_F2}/a1_forget10"
if ! find "$A1_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[B] Train A1 ep=$A1_EP save_steps=$SAVE_STEPS"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_F2_a1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$NUM_LAYER" model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" trainer.max_epochs="$A1_EP" \
        +trainer.save_steps="$SAVE_STEPS" trainer.strategy=gpu \
        OUTPUTMODELDIR="$A1_OUT" postfix="F2a1" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_F2_a1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi

A1_PARENT=$(find "$A1_OUT" -name "checkpoint-*" -type d | head -1 | xargs dirname)

# === FQ-only eval helper ===
fq_only_eval() {
    local a1="$1" w2="$2" task="$3"
    local outdir="${OU_REPO}/saves/eval/${task}"
    local fqlog="/tmp/fq_eval_${task}.log"
    rm -rf "$outdir"
    rm -f "$fqlog"

    echo "  FQ-screen $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_PATH" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task" > "$fqlog" 2>&1 &
    local pid=$!

    # Watch the log for FQ; kill once seen (timeout 600s safety)
    local start=$SECONDS
    while kill -0 $pid 2>/dev/null; do
        if grep -q "Result for metric forget_quality:" "$fqlog" 2>/dev/null; then
            sleep 2  # let it flush
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
            break
        fi
        if (( SECONDS - start > 600 )); then
            echo "  TIMEOUT for $task"
            kill -9 $pid 2>/dev/null
            wait $pid 2>/dev/null
            return
        fi
        sleep 5
    done

    local fq=$(grep "Result for metric forget_quality:" "$fqlog" | tail -1 | sed -E 's/.*forget_quality:[[:space:]]*//')
    echo "    FQ=$fq  (elapsed ${SECONDS}s)"
    echo "$task $fq" >> /tmp/3b_F2_fq_screen.txt
}

# === Full eval (writes complete JSON) ===
full_eval() {
    local a1="$1" w2="$2" task="$3"
    local outdir="${OU_REPO}/saves/eval/${task}"
    local eval_json="${outdir}/TOFU_EVAL.json"
    rm -rf "$outdir"

    echo "  FULL-eval $task (w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$A2_PATH" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"
    fq=$($PY -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null || echo "-1")
    echo "    [FULL] FQ=$fq"
}

# === Phase 1: FQ-only screening ===
> /tmp/3b_F2_fq_screen.txt
echo "[Phase 1] FQ-only screen: A1 step=600..1200 step=100, w2=0.5"
for s in 600 700 800 900 1000 1100 1200; do
    a1="${A1_PARENT}/checkpoint-${s}"
    [ -d "$a1" ] || { echo "  miss step=$s"; continue; }
    fq_only_eval "$a1" 0.5 "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2F2_a1s${s}_w20p5_FQONLY"
done

echo
echo "=== FQ screen results ==="
sort -k2 -gr /tmp/3b_F2_fq_screen.txt | head -10
echo

# === Phase 2: full eval on FQ > threshold ===
echo "[Phase 2] Full eval on candidates with FQ > $FQ_THRESHOLD"
while read task fq; do
    if awk -v f="$fq" -v t="$FQ_THRESHOLD" 'BEGIN{exit !(f+0 > t)}'; then
        # Extract a1 step from task name
        step=$(echo "$task" | sed -E 's/.*a1s([0-9]+).*/\1/')
        a1="${A1_PARENT}/checkpoint-${step}"
        new_task=$(echo "$task" | sed 's/_FQONLY//')
        full_eval "$a1" 0.5 "$new_task"
    fi
done < /tmp/3b_F2_fq_screen.txt

echo "============================================================"
echo "Plan F2-fast DONE; FQ screen at /tmp/3b_F2_fq_screen.txt"
echo "============================================================"
