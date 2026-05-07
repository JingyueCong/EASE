#!/usr/bin/env bash
# 3B f10 Plan H: untouched dimensions — A2 ep, num_layer=8 (controlled),
# retain_weight, F2 a1s1500 (lr→0 at end of ep=20). FQ-only screen.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
FQ_THRESHOLD=0.5

A1_F600=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a1_forget10 -name "checkpoint-600" -type d | head -1)
A2_F=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F2_1500=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_F2/a1_forget10 -name "checkpoint-1500" -type d | tail -1)

echo "Plan F a1s600: $A1_F600"
echo "Plan F A2:     $A2_F"
echo "Plan F2 a1s1500: $A1_F2_1500"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a() {
    local role="$1" eps="$2" nl="$3" rw="$4" tag="$5"
    local out="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_${tag}/${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role/$tag: exists"; return
    fi
    echo "  → Train ${role}/${tag} (ep=$eps, num_layer=$nl, retain_weight=$rw)"
    mkdir -p "$out"
    cd "$ULD_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_H_${tag}_${role}_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer="$nl" model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight="$rw" \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs="$eps" \
        +trainer.save_steps=75 trainer.strategy=gpu \
        OUTPUTMODELDIR="$out" postfix="H${tag}${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_H_${tag}_${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
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
    echo "$task $fq" >> /tmp/3b_H_fq_screen.txt
}

> /tmp/3b_H_fq_screen.txt

# === Quick test 1: F2 a1s1500 (ep=20 final, lr→0) ===
echo "=== H1: F2 a1s1500 (lr→0 at ep=20 end) ==="
fq_only_eval "$A1_F2_1500" "$A2_F" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2H_F2a1s1500_FQONLY"

# === H2: A2 retrained ep=6 with num_layer=4 (= A1's setting) ===
echo "=== H2: A2 ep=6 ==="
train_a a2 6 4 5 "a2ep6"
A2_EP6=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep6/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A2_EP6" ] && fq_only_eval "$A1_F600" "$A2_EP6" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2H_a2ep6_FQONLY"

# === H3: A2 retrained ep=10 ===
echo "=== H3: A2 ep=10 ==="
train_a a2 10 4 5 "a2ep10"
A2_EP10=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_a2ep10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A2_EP10" ] && fq_only_eval "$A1_F600" "$A2_EP10" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2H_a2ep10_FQONLY"

# === H4: num_layer=8 controlled, ep=8 (same as Plan F) ===
echo "=== H4: num_layer=8 + Plan F recipe ==="
train_a a1 8 8 5 "nl8"
train_a a2 3 8 5 "nl8"
A1_NL8=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_nl8/a1_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_NL8=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_nl8/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A1_NL8" ] && [ -n "$A2_NL8" ] && fq_only_eval "$A1_NL8" "$A2_NL8" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2H_nl8_FQONLY"

# === H5: num_layer=2 (smaller) ===
echo "=== H5: num_layer=2 ==="
train_a a1 8 2 5 "nl2"
train_a a2 3 2 5 "nl2"
A1_NL2=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_nl2/a1_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_NL2=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_nl2/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A1_NL2" ] && [ -n "$A2_NL2" ] && fq_only_eval "$A1_NL2" "$A2_NL2" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2H_nl2_FQONLY"

# === H6: retain_weight=10 ===
echo "=== H6: retain_weight=10 ==="
train_a a1 8 4 10 "rw10"
train_a a2 3 4 10 "rw10"
A1_RW10=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_rw10/a1_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_RW10=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_H_rw10/a2_forget10 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$A1_RW10" ] && [ -n "$A2_RW10" ] && fq_only_eval "$A1_RW10" "$A2_RW10" "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2H_rw10_FQONLY"

echo
echo "============================================================"
echo "Plan H FQ screen results (sorted by FQ desc):"
sort -k2 -gr /tmp/3b_H_fq_screen.txt
echo "============================================================"
echo "DONE"
