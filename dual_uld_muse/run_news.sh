#!/usr/bin/env bash
# Train News A1, A2 then eval. Sequential (single GPU).
set -euo pipefail
cd ${EASE_ROOT}/dual_uld_muse

LOG=logs/news_pipeline.log
echo "== News A1 training ==" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 /usr/bin/python train_assistant.py \
    --split News --role a1 --epochs 3 --batch_size 1 --grad_accum 4 \
    >> $LOG 2>&1
echo "== News A2 training ==" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 /usr/bin/python train_assistant.py \
    --split News --role a2 --epochs 3 --batch_size 1 --grad_accum 4 \
    >> $LOG 2>&1
echo "== News eval ==" | tee -a $LOG
cd ${EASE_ROOT}/open-unlearning
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python src/eval.py \
    experiment=eval/muse/default \
    data_split=News \
    model=Llama-2-7b-hf_DualULD \
    model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target \
    model.model_args.a1_path=${EASE_ROOT}/dual_uld_muse/models/News_a1/checkpoint-final \
    model.model_args.a2_path=${EASE_ROOT}/dual_uld_muse/models/News_a2/checkpoint-final \
    model.model_args.weight_a1=-0.8 \
    model.model_args.weight_a2=0.8 \
    model.model_args.top_logit_filter=0.01 \
    model.model_args.attn_implementation=sdpa \
    model.tokenizer_args.pretrained_model_name_or_path=meta-llama/Llama-2-7b-hf \
    retain_logs_path=${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json \
    task_name=muse_Llama-2-7b-hf_News_DualULD \
    >> ${EASE_ROOT}/dual_uld_muse/$LOG 2>&1
echo "== News pipeline done ==" | tee -a ${EASE_ROOT}/dual_uld_muse/$LOG
