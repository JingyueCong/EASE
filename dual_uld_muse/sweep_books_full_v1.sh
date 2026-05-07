#!/usr/bin/env bash
# Books full eval (with verbmem + extraction) sweep across 6 weight points.
# Reuses existing Books_a1/a2 checkpoints; only weights vary.
set -uo pipefail
ROOT=${EASE_ROOT}/dual_uld_muse
cd ${EASE_ROOT}/open-unlearning

EXP="eval/muse/default"
SPLIT="Books"
TOPF="0.01"
LOG_ALL=$ROOT/logs/sweep_books_full_v1.log
: > "$LOG_ALL"

run_one() {
    local LABEL="$1" W1="$2" W2="$3"
    local TASK="muse_Llama-2-7b-hf_${SPLIT}_DualULD_${LABEL}_full"
    local OUT_JSON="saves/eval/${TASK}/MUSE_SUMMARY.json"
    if [ -f "$OUT_JSON" ]; then
        echo "[skip] $TASK already exists: $(cat $OUT_JSON | tr -d '\n ')" | tee -a "$LOG_ALL"
        return 0
    fi
    echo "====================" | tee -a "$LOG_ALL"
    echo "[$(date +%H:%M:%S)] [eval] $LABEL  w1=$W1 w2=$W2 topF=$TOPF" | tee -a "$LOG_ALL"
    echo "====================" | tee -a "$LOG_ALL"
    local LOG="${ROOT}/logs/${TASK}.log"
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /usr/bin/python src/eval.py \
        experiment=${EXP} \
        data_split=${SPLIT} \
        model=Llama-2-7b-hf_DualULD \
        model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${SPLIT}_target \
        model.model_args.a1_path=${ROOT}/models/${SPLIT}_a1/checkpoint-final \
        model.model_args.a2_path=${ROOT}/models/${SPLIT}_a2/checkpoint-final \
        model.model_args.weight_a1=${W1} \
        model.model_args.weight_a2=${W2} \
        model.model_args.top_logit_filter=${TOPF} \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path=meta-llama/Llama-2-7b-hf \
        retain_logs_path=saves/eval/muse_Llama-2-7b-hf_${SPLIT}_retrain/MUSE_EVAL.json \
        task_name=${TASK} \
        > "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$OUT_JSON" ]; then
        local v
        v=$(/usr/bin/python -c "import json; d=json.load(open('$OUT_JSON')); print(f\"f_know={d.get('forget_knowmem_ROUGE',0)*100:5.1f} f_verb={d.get('forget_verbmem_ROUGE',0)*100:5.1f} privleak={d.get('privleak',0):7.2f} r_know={d.get('retain_knowmem_ROUGE',0)*100:5.1f} extr={d.get('extraction_strength',0)*100:5.2f}")
        echo "[$(date +%H:%M:%S)] [done] $LABEL  $v" | tee -a "$LOG_ALL"
    else
        echo "[$(date +%H:%M:%S)] [FAIL rc=$rc] $LABEL  see $LOG" | tee -a "$LOG_ALL"
    fi
    echo "" | tee -a "$LOG_ALL"
}

echo "[$(date +%H:%M:%S)] === sweep_books_full_v1 START ===" | tee -a "$LOG_ALL"

run_one w1m1p0_w21p0  -1.0  1.0
run_one w1m1p5_w21p5  -1.5  1.5
run_one w1m2p0_w22p0  -2.0  2.0
run_one w1m1p5_w20p3  -1.5  0.3
run_one w1m1p5_w20p5  -1.5  0.5
run_one w1m2p0_w20p5  -2.0  0.5

echo "[$(date +%H:%M:%S)] === sweep_books_full_v1 DONE ===" | tee -a "$LOG_ALL"
