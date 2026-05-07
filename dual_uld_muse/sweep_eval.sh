#!/usr/bin/env bash
# Sweep DualULD weights on MUSE eval. Same trained A1/A2 reused; only weights vary.
#
# Usage:
#   bash sweep_eval.sh Books "-0.3 -0.5 -0.6"
#   bash sweep_eval.sh News  "-0.3 -0.5 -0.6"
set -euo pipefail
cd ${EASE_ROOT}/open-unlearning

SPLIT="${1:?split (Books|News)}"
WS="${2:?space-separated w1 list, e.g. \"-0.3 -0.5\"}"
TOPF="${TOPF:-0.01}"
EXP="${EXP:-eval/muse/fast}"   # fast = skip verbmem+extraction; default for sweep
ROOT=${EASE_ROOT}/dual_uld_muse
SUFFIX="${SUFFIX:-_fast}"        # rename task to avoid clobbering full evals

for W1 in $WS; do
    W2=$(/usr/bin/python -c "print(abs(float('$W1')))")
    W1S=$(echo "$W1" | sed -e 's/-/m/g' -e 's/\./p/g')
    W2S=$(echo "$W2" | sed -e 's/-/m/g' -e 's/\./p/g')
    TASK="muse_Llama-2-7b-hf_${SPLIT}_DualULD_w1${W1S}_w2${W2S}${SUFFIX}"
    OUT_JSON="saves/eval/${TASK}/MUSE_SUMMARY.json"
    if [ -f "$OUT_JSON" ]; then
        echo "[skip] $TASK exists"
        continue
    fi
    echo "===================="
    echo "[eval] split=$SPLIT w1=$W1 w2=$W2  topF=$TOPF  task=$TASK"
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
    echo "[done] cat $OUT_JSON:"
    cat "$OUT_JSON" 2>/dev/null || echo "MISSING"
    echo
done
echo "Sweep done for $SPLIT"
