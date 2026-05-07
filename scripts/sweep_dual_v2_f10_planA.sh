#!/usr/bin/env bash
# forget10 Plan A: untried A1 ckpts (ep8 step=375/525/600, fine step=525)
# + a2ep1 forget10 retraining and 3 w2 combos.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-1B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A2_F10_EP3=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2/a2_forget10 \
    -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

EP8_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget10 \
    -name "checkpoint-*" -type d | head -1 | xargs dirname)
A1_EP8_375="${EP8_PARENT}/checkpoint-375"
A1_EP8_525="${EP8_PARENT}/checkpoint-525"
A1_EP8_600="${EP8_PARENT}/checkpoint-600"

FINE_PARENT=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_fine/a1_forget10 \
    -name "checkpoint-*" -type d | head -1 | xargs dirname)
A1_FINE_525="${FINE_PARENT}/checkpoint-525"

A1_EP6=$(find ${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_ep8/a1_forget10 \
    -name "checkpoint-450" -type d | head -1)

echo "A2 ep=3:        $A2_F10_EP3"
echo "A1 ep8 s375:    $A1_EP8_375"
echo "A1 ep8 s525:    $A1_EP8_525"
echo "A1 ep8 s600:    $A1_EP8_600"
echo "A1 fine s525:   $A1_FINE_525"
echo "A1 ep6 (=s450): $A1_EP6"

# ----- Train A2 forget10 ep=1 -----
A2_EP1_OUT="${ULD_REPO}/outputs_trained_models/llama3_1b_dual_v2_a2ep1/a2_forget10"
if ! find "$A2_EP1_OUT" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
    echo "[1] Train A2 forget10 ep=1"
    cd "$ULD_REPO"
    export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
    export USE_TF=0; export TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_v2_a2ep1_forget10" \
        data=tofu_chat3 data.dataset.split="forget10_perturbed" \
        data_mode=dual_a2 data_mode.r_sub_indices_path="${ULD_REPO}/data/rsub/forget10_k80.json" \
        data_mode.retain_num=400 \
        model=llama-3-1b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs=1 \
        trainer.strategy=gpu OUTPUTMODELDIR="$A2_EP1_OUT" postfix=a2ep1 \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_v2_a2ep1_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
fi
A2_F10_EP1=$(find "$A2_EP1_OUT" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A2 ep=1: $A2_F10_EP1"

run_dual() {
    local a1="$1" a2="$2" w1="$3" w2="$4" task="$5"
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
        forget_split=forget10 holdout_split=holdout10 \
        eval.tofu.batch_size=4 \
        retain_logs_path="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json" \
        task_name="$task"
}

echo "=== A1 over-training scan with a2ep3, w1=-0.8 ==="
run_dual "$A1_EP8_375" "$A2_F10_EP3" -0.8 0.8 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep8s375_w20p8
run_dual "$A1_EP8_375" "$A2_F10_EP3" -0.8 0.5 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep8s375_w20p5
run_dual "$A1_EP8_525" "$A2_F10_EP3" -0.8 0.8 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep8s525_w20p8
run_dual "$A1_EP8_525" "$A2_F10_EP3" -0.8 0.5 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep8s525_w20p5
run_dual "$A1_EP8_600" "$A2_F10_EP3" -0.8 0.8 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep8s600_w20p8
run_dual "$A1_EP8_600" "$A2_F10_EP3" -0.8 0.5 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep8s600_w20p5

echo "=== Fine A1 step=525 with a2ep3 ==="
run_dual "$A1_FINE_525" "$A2_F10_EP3" -0.8 0.8 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1fine525_w20p8

echo "=== a2ep1 forget10 + a1ep6 sweep ==="
run_dual "$A1_EP6" "$A2_F10_EP1" -0.8 0.5 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep6_a2ep1_w20p5
run_dual "$A1_EP6" "$A2_F10_EP1" -0.8 0.7 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep6_a2ep1_w20p7
run_dual "$A1_EP6" "$A2_F10_EP1" -0.8 0.8 tofu_Llama-3.2-1B-Instruct_forget10_DualULD_v2_a1ep6_a2ep1_w20p8

echo "DONE"
