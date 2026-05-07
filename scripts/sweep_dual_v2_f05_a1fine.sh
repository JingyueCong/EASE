#!/usr/bin/env bash
# (a) Retrain A2 forget05 with ep=2 + eval at w1=-0.8 w2=0.5/0.7
# (b) Re-eval forget01 dual v2 at a1step135/w2=0.5/w2=1.0 (existing ckpts)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A1_F05=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a1_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_F01=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget01 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A1_F01_FINE_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget01 -name "checkpoint-*" -type d | head -1 | xargs dirname)

# (a) Retrain A2 forget05 with ep=2
A2_EP2_OUT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_a2ep2/a2_forget05"
if ! find "$A2_EP2_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[a-1] Train A2 forget05 ep=2 → $A2_EP2_OUT"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_a2ep2_forget05" \
        data=tofu_chat3 data.dataset.split="forget05_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget05_k40.json" \
        data_mode.retain_num=200 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=2 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A2_EP2_OUT" postfix=a2ep2 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_a2ep2_forget05/\${now:%Y-%m-%d_%H-%M-%S}"
fi

A2_F05_EP2=$(find "$A2_EP2_OUT" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A2 ep=2: $A2_F05_EP2"

run_dual() {
    local split="$1" holdout="$2" retain="$3" a1="$4" a2="$5" w1="$6" w2="$7" task="$8"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $task"; return; }
    echo "  Eval $task (w1=$w1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$split" holdout_split="$holdout" \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json" \
        task_name="$task"
}

# (a) Eval forget05 with A2-ep=2 at w2=0.5/0.7
echo "[a-2] Eval forget05 with A2 ep=2 at multiple w2"
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05_EP2" -0.8 0.5 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a2ep2_w20p5
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05_EP2" -0.8 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a2ep2_w20p7
run_dual forget05 holdout05 retain95 "$A1_F05" "$A2_F05_EP2" -0.8 0.8 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a2ep2_w20p8

# (b) forget01: re-eval candidates with EM+Fluency
echo "[b] Eval forget01 alternative configs"
A1_F01_135="${A1_F01_FINE_PARENT}/checkpoint-135"
A1_F01_120="${A1_F01_FINE_PARENT}/checkpoint-120"
A1_F01_145="${A1_F01_FINE_PARENT}/checkpoint-145"
A1_F01_150="${A1_F01_FINE_PARENT}/checkpoint-150"
run_dual forget01 holdout01 retain99 "$A1_F01_135" "$A2_F01" -0.7 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step135_emflu
run_dual forget01 holdout01 retain99 "$A1_F01_120" "$A2_F01" -0.7 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step120_emflu
run_dual forget01 holdout01 retain99 "$A1_F01_145" "$A2_F01" -0.7 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step145_emflu
run_dual forget01 holdout01 retain99 "$A1_F01_150" "$A2_F01" -0.7 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget01_DualULD_v2_a1step150_emflu
echo "DONE"
