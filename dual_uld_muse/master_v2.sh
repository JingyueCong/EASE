#!/usr/bin/env bash
# Master v2: News A1 r=64 retrain → News A2 retrain → Books R_sub 0.40 → Books A1/A2 retrain
#            → fast sweeps → full evals → write best results.
#
# Status file: logs/master_v2_status.txt
# Logs: logs/master_v2.log
set -uo pipefail
ROOT=${EASE_ROOT}/dual_uld_muse
LOG=$ROOT/logs/master_v2.log
STATUS=$ROOT/logs/master_v2_status.txt
mkdir -p $(dirname $LOG)
: > "$LOG"
: > "$STATUS"

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" "$STATUS"; }
sub()  { echo "  $*" | tee -a "$LOG"; }

#####################################################################
# Phase A — News A1 r=64, lr=5e-4, num_layer=16, 10 epochs
#####################################################################
step "Phase A — News A1 retrain (r=64, lr=5e-4, num_layer=16, 10ep)"
mv $ROOT/models/News_a1 $ROOT/models/News_a1_archive_$(date +%H%M%S) 2>/dev/null || true
cd $ROOT
/usr/bin/python train_assistant.py \
    --split News --role a1 \
    --epochs 10 --batch_size 1 --grad_accum 4 \
    --num_layer 16 --lora_r 64 --lora_alpha 128 \
    --lr 5e-4 \
    --paraphrase_path $ROOT/aug/News_paraphrases.jsonl \
    > $ROOT/logs/A_news_a1.log 2>&1
step "Phase A done (final CE/loss tail):"
tail -3 $ROOT/logs/A_news_a1.log | tee -a "$LOG" "$STATUS"

#####################################################################
# Phase B — News A2 r=64, num_layer=16, 5 epochs
#####################################################################
step "Phase B — News A2 retrain (r=64, num_layer=16, 5ep)"
mv $ROOT/models/News_a2 $ROOT/models/News_a2_archive_$(date +%H%M%S) 2>/dev/null || true
/usr/bin/python train_assistant.py \
    --split News --role a2 \
    --epochs 5 --batch_size 1 --grad_accum 4 \
    --num_layer 16 --lora_r 64 --lora_alpha 128 \
    --lr 5e-4 \
    > $ROOT/logs/B_news_a2.log 2>&1
step "Phase B done (final tail):"
tail -3 $ROOT/logs/B_news_a2.log | tee -a "$LOG" "$STATUS"

#####################################################################
# Phase C — Books R_sub 0.40 + A1/A2 retrain (8L same as before)
#####################################################################
step "Phase C — Books R_sub k_frac=0.40 + A1/A2 retrain"
/usr/bin/python build_rsub.py --split Books --k_frac 0.40 \
    > $ROOT/logs/C_rsub_books.log 2>&1
sub "  R_sub Books generated:"
grep "R_sub size" $ROOT/logs/C_rsub_books.log | tee -a "$LOG" "$STATUS"

mv $ROOT/models/Books_a1 $ROOT/models/Books_a1_kfrac025 2>/dev/null || true
mv $ROOT/models/Books_a2 $ROOT/models/Books_a2_kfrac025 2>/dev/null || true

step "Phase C.1 — Books A1 retrain (kfrac=0.40, 8L+aug, 5ep)"
/usr/bin/python train_assistant.py \
    --split Books --role a1 \
    --epochs 5 --batch_size 1 --grad_accum 4 --num_layer 8 \
    --paraphrase_path $ROOT/aug/Books_paraphrases.jsonl \
    > $ROOT/logs/C_books_a1.log 2>&1
sub "C.1 final tail:"
tail -3 $ROOT/logs/C_books_a1.log | tee -a "$LOG" "$STATUS"

step "Phase C.2 — Books A2 retrain (kfrac=0.40, 8L, 3ep)"
/usr/bin/python train_assistant.py \
    --split Books --role a2 \
    --epochs 3 --batch_size 1 --grad_accum 4 --num_layer 8 \
    > $ROOT/logs/C_books_a2.log 2>&1
sub "C.2 final tail:"
tail -3 $ROOT/logs/C_books_a2.log | tee -a "$LOG" "$STATUS"

#####################################################################
# Phase D — Fast sweeps on both new assistants
#####################################################################
step "Phase D — clean stale fast evals + sweep"
rm -rf ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_News_DualULD_w*_fast 2>/dev/null
rm -rf ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_Books_DualULD_w*_fast 2>/dev/null

step "Phase D.1 — News fast sweep"
bash sweep_eval.sh News "-0.5 -0.9 -1.3" > $ROOT/logs/D_news_sweep.log 2>&1
sub "News sweep results:"
for f in ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_News_DualULD_w*_fast/MUSE_SUMMARY.json; do
    label=$(basename $(dirname $f) | sed 's/.*_DualULD_//;s/_fast//')
    v=$(/usr/bin/python -c "import json; d=json.load(open('$f')); print(f\"f={d.get('forget_knowmem_ROUGE',0)*100:5.1f} l={d.get('privleak',0):7.2f} r={d.get('retain_knowmem_ROUGE',0)*100:5.1f}\")")
    printf "    %-30s %s\n" "$label" "$v" | tee -a "$LOG" "$STATUS"
done

step "Phase D.2 — Books fast sweep"
bash sweep_eval.sh Books "-0.5 -0.7 -0.9 -1.1" > $ROOT/logs/D_books_sweep.log 2>&1
# Also run asymmetric (-0.9, +0.3) and (-0.9, +0.5)
cat > $ROOT/sweep_books_v3_asym.txt <<EOF
w1m0p9_w20p3       -0.9   0.3   0.01
w1m0p9_w20p5       -0.9   0.5   0.01
w1m1p1_w20p3       -1.1   0.3   0.01
EOF
bash sweep_eval2.sh Books $ROOT/sweep_books_v3_asym.txt >> $ROOT/logs/D_books_sweep.log 2>&1
sub "Books sweep results:"
for f in ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_Books_DualULD_w*_fast/MUSE_SUMMARY.json; do
    label=$(basename $(dirname $f) | sed 's/.*_DualULD_//;s/_fast//')
    v=$(/usr/bin/python -c "import json; d=json.load(open('$f')); print(f\"f={d.get('forget_knowmem_ROUGE',0)*100:5.1f} l={d.get('privleak',0):7.2f} r={d.get('retain_knowmem_ROUGE',0)*100:5.1f}\")")
    printf "    %-30s %s\n" "$label" "$v" | tee -a "$LOG" "$STATUS"
done

step "ALL DONE — sweep results above. Pick weight + run full eval manually."
