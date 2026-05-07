#!/usr/bin/env bash
# Chain: D → C2 → E. Each stops only if FQ > 0.9 (retain-equivalent).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
SUCCESS_SENTINEL=/tmp/3b_f10_success

A2_PB=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_planB/a2_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_PB_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_planB/a1_forget10 \
    -name "checkpoint-*" -type d | head -1 | xargs dirname)

echo "Plan B A1 parent: $A1_PB_PARENT"
echo "Plan B A2 final:  $A2_PB"

run_dual() {
    local a1="$1" a2="$2" w1="$3" w2="$4" tF="$5" task="$6"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 1000 ] && { echo "  SKIP $task"; return; }
    rm -f "$eval_json"
    echo "  Eval $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter="$tF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=2 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain90_v2/TOFU_EVAL.json" \
        task_name="$task"

    # Check FQ from the just-created JSON; abort chain only if FQ > 0.9
    local fq
    fq=$($PY -c "import json,sys; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null || echo "-1")
    echo "    FQ=$fq"
    awk -v f="$fq" 'BEGIN{exit !(f+0 > 0.9)}' && {
        echo "    *** FQ > 0.9 (retain-equivalent) — success sentinel created, will stop chain ***"
        echo "$task FQ=$fq" > "$SUCCESS_SENTINEL"
    }
}

check_done() { [ -f "$SUCCESS_SENTINEL" ] && { echo "[STOP] FQ goal achieved at $(cat $SUCCESS_SENTINEL); aborting chain"; exit 0; }; }

rm -f "$SUCCESS_SENTINEL"

# ============================================
# Plan D — A1 mid-ckpt scan (eval-only)
# ============================================
echo "============================================================"
echo "[Plan D] A1 mid-ckpt scan: step=100/200/400/550, vanilla settings"
echo "============================================================"
for step in 100 200 400 550; do
    a1="${A1_PB_PARENT}/checkpoint-${step}"
    [ -d "$a1" ] || { echo "  miss $a1"; continue; }
    run_dual "$a1" "$A2_PB" -0.8 0.5 0.01 \
        "tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2pB_a1s${step}_w20p5"
    check_done
done

# ============================================
# Plan C2 — num_layer 7 → 14 retrain + eval
# ============================================
echo "============================================================"
echo "[Plan C2] num_layer=14 retrain"
echo "============================================================"
MODELS_C2="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_C2"
mkdir -p "$MODELS_C2"
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_C2() {
    local role="$1" eps="$2"
    local out="${MODELS_C2}/${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role: exists"; return
    fi
    echo "  → Train $role (num_layer=14, ep=$eps, lr=5e-4) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_C2_${role}_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=14 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=2 trainer.gradient_accumulation_steps=8 \
        trainer.learning_rate=5e-4 trainer.max_epochs="$eps" \
        +trainer.save_steps=25 trainer.strategy=gpu \
        OUTPUTMODELDIR="$out" postfix="C2${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_C2_${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}
train_C2 a1 12
check_done
train_C2 a2 6
check_done

A1_C2=$(find "${MODELS_C2}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_C2=$(find "${MODELS_C2}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
run_dual "$A1_C2" "$A2_C2" -0.8 0.5 0.01 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2C2_w20p5
check_done

# ============================================
# Plan E — lr 5e-4 → 1e-4 retrain + eval
# ============================================
echo "============================================================"
echo "[Plan E] lr=1e-4 retrain (deeper convergence)"
echo "============================================================"
MODELS_E="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_E"
mkdir -p "$MODELS_E"

train_E() {
    local role="$1" eps="$2"
    local out="${MODELS_E}/${role}_forget10"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role: exists"; return
    fi
    echo "  → Train $role (lr=1e-4, ep=$eps) → $out"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_E_${role}_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode="dual_${role}" data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=7 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=2 trainer.gradient_accumulation_steps=8 \
        trainer.learning_rate=1e-4 trainer.max_epochs="$eps" \
        +trainer.save_steps=25 trainer.strategy=gpu \
        OUTPUTMODELDIR="$out" postfix="E${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_E_${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
}
train_E a1 30
check_done
train_E a2 15
check_done

A1_E=$(find "${MODELS_E}/a1_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_E=$(find "${MODELS_E}/a2_forget10" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
run_dual "$A1_E" "$A2_E" -0.8 0.5 0.01 tofu_Llama-3.2-3B-Instruct_forget10_DualULD_v2E_w20p5

echo "============================================================"
echo "CHAIN DONE"
[ -f "$SUCCESS_SENTINEL" ] && echo "Best: $(cat $SUCCESS_SENTINEL)" || echo "No config hit FQ > 0.9 — pick best from logs"
