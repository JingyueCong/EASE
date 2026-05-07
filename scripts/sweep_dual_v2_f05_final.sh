#!/usr/bin/env bash
# Final forget05 push: 
# (1) Combine A1 fine step=225 + a2ep2 (cheap: 2 evals)
# (2) Train A1 max_ep=8 with save_steps=25, eval mid-training steps 250/300/350
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"; OU_REPO="${ROOT}/open-unlearning"; PY=/usr/bin/python; GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A1_FINE_F05_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget05 -name "checkpoint-*" -type d | head -1 | xargs dirname)
A2_F05_EP3=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
A2_F05_EP2=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_a2ep2/a2_forget05 -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

run_dual() {
    local a1="$1" a2="$2" w2="$3" task="$4"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    [ -f "$eval_json" ] && { echo "  SKIP $task"; return; }
    echo "  Eval $task"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1=-0.8 model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget05 holdout_split=holdout05 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json" \
        task_name="$task"
}

# (1) Combine: A1 fine step=225 + a2ep2 + various w2
echo "[1] A1 fine step=225 + A2 ep=2 combos"
A1_225="${A1_FINE_F05_PARENT}/checkpoint-225"
run_dual "$A1_225" "$A2_F05_EP2" 0.5 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1step225_a2ep2_w20p5
run_dual "$A1_225" "$A2_F05_EP2" 0.7 \
    tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1step225_a2ep2_w20p7

# (2) Train A1 forget05 max_ep=8, save_steps=25
A1_EP8_OUT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget05"
if ! find "$A1_EP8_OUT" -name "checkpoint-300" -type d 2>/dev/null | grep -q .; then
    echo "[2] Train A1 forget05 max_ep=8 save_steps=25"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_ep8_a1_forget05" \
        data=tofu_chat3 data.dataset.split="forget05_perturbed" \
        data_mode=dual_a1 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget05_k40.json" \
        data_mode.retain_num=200 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=8 \
        +trainer.save_steps=25 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A1_EP8_OUT" postfix=ep8 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_ep8_a1_forget05/\${now:%Y-%m-%d_%H-%M-%S}"
fi

A1_EP8_PARENT=$(find "$A1_EP8_OUT" -name "checkpoint-*" -type d | head -1 | xargs dirname)

# (3) Eval mid-training A1 steps with a2ep3 (max FQ direction)
echo "[3] f05 A1 ep=8 mid-training scan with a2ep3 w2=0.5"
for STEP in 250 300 350; do
    a1="${A1_EP8_PARENT}/checkpoint-${STEP}"
    [ -d "$a1" ] || { echo "  skip step=$STEP"; continue; }
    run_dual "$a1" "$A2_F05_EP3" 0.5 \
        tofu_Llama-3.2-1B-Instruct_forget05_DualULD_v2_a1ep8_step${STEP}_w20p5
done
echo "DONE"
