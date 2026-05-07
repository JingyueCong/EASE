#!/usr/bin/env bash
# Custom-config sweep: each line is "label w1 w2 topF".
# Usage: bash sweep_eval2.sh <split> <config_file>
set -euo pipefail
cd ${EASE_ROOT}/open-unlearning

SPLIT="${1:?split (Books|News)}"
CFG_RAW="${2:?config file with rows: label w1 w2 topF}"
# Resolve to absolute before cd
if [ -f "$CFG_RAW" ]; then
    CFG="$(realpath "$CFG_RAW")"
elif [ -f "${EASE_ROOT}/dual_uld_muse/$CFG_RAW" ]; then
    CFG="${EASE_ROOT}/dual_uld_muse/$CFG_RAW"
else
    echo "Cannot find config: $CFG_RAW"; exit 1
fi
EXP="${EXP:-eval/muse/fast}"
ROOT=${EASE_ROOT}/dual_uld_muse

while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    read -r LABEL W1 W2 TOPF <<< "$line"
    TASK="muse_Llama-2-7b-hf_${SPLIT}_DualULD_${LABEL}_fast"
    OUT_JSON="saves/eval/${TASK}/MUSE_SUMMARY.json"
    if [ -f "$OUT_JSON" ]; then
        echo "[skip] $TASK exists: $(cat $OUT_JSON | tr '\n' ' ')"
        continue
    fi
    echo "===================="
    echo "[eval] $LABEL  split=$SPLIT  w1=$W1 w2=$W2 topF=$TOPF"
    echo "===================="
    LOG="${ROOT}/logs/${TASK}.log"
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
    echo "[done] $TASK"
    cat "$OUT_JSON" 2>/dev/null || echo "MISSING"
    echo
done < "$CFG"
echo "Sweep done"
