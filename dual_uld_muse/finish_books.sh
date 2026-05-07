#!/usr/bin/env bash
# After Books A2 (kfrac=0.40) finishes: fast sweep → pick best → full eval.
set -uo pipefail
ROOT=${EASE_ROOT}/dual_uld_muse
LOG=$ROOT/logs/finish_books.log
STATUS=$ROOT/logs/finish_books_status.txt
: > "$LOG"; : > "$STATUS"

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" "$STATUS"; }

# Wait for Books A2 retrain to finish
step "Waiting for Books A2 retrain..."
until [ -f $ROOT/models/Books_a2/checkpoint-final/adapter_model.safetensors ] && \
      ! pgrep -f "train_assistant.py.*Books.*a2" > /dev/null 2>&1; do
    sleep 30
done
step "Books A2 ready."

# Clean stale Books fast evals
rm -rf ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_Books_DualULD_w*_fast 2>/dev/null

# Fast sweep on the new Books assistants (kfrac=0.40)
step "Phase D — Books fast sweep (sym + asym)"
cd $ROOT
bash sweep_eval.sh Books "-0.5 -0.7 -0.9 -1.1" > $ROOT/logs/finish_books_sweep_sym.log 2>&1

cat > $ROOT/sweep_books_v4_asym.txt <<EOF
w1m0p9_w20p3       -0.9   0.3   0.01
w1m0p9_w20p5       -0.9   0.5   0.01
w1m1p1_w20p3       -1.1   0.3   0.01
w1m1p1_w20p5       -1.1   0.5   0.01
EOF
bash sweep_eval2.sh Books $ROOT/sweep_books_v4_asym.txt >> $ROOT/logs/finish_books_sweep_sym.log 2>&1

step "Books sweep results:"
for f in ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_Books_DualULD_w*_fast/MUSE_SUMMARY.json; do
    label=$(basename $(dirname $f) | sed 's/.*_DualULD_//;s/_fast//')
    v=$(/usr/bin/python -c "import json; d=json.load(open('$f')); print(f\"f={d.get('forget_knowmem_ROUGE',0)*100:5.1f} l={d.get('privleak',0):7.2f} r={d.get('retain_knowmem_ROUGE',0)*100:5.1f}\")")
    printf "  %-30s %s\n" "$label" "$v" | tee -a "$LOG" "$STATUS"
done

step "DONE — pick best weight, then run full eval manually."
