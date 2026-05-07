#!/usr/bin/env bash
# Master v3: perturb done → retrain Books A1+A2 + News A1+A2 with paraphrase+perturb,
# then sweep + full eval.
set -uo pipefail
ROOT=${EASE_ROOT}/dual_uld_muse
LOG=$ROOT/logs/master_v3.log
STATUS=$ROOT/logs/master_v3_status.txt
: > "$LOG"; : > "$STATUS"

step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" "$STATUS"; }

#####################################################################
# Wait for perturb generation to finish
#####################################################################
step "Wait for Books + News perturbations..."
while true; do
    bn=$(wc -l < $ROOT/aug/Books_perturbations.jsonl 2>/dev/null || echo 0)
    nn=$(wc -l < $ROOT/aug/News_perturbations.jsonl 2>/dev/null || echo 0)
    bp=$(pgrep -f "perturb_forget.*Books" > /dev/null && echo "RUN" || echo "DONE")
    np=$(pgrep -f "perturb_forget.*News"  > /dev/null && echo "RUN" || echo "DONE")
    [[ "$bp" == "DONE" && "$np" == "DONE" ]] && break
    echo "  [$(date +%H:%M:%S)] Books=$bn ($bp)  News=$nn ($np)"
    sleep 120
done
step "Perturb done. Books=$(wc -l < $ROOT/aug/Books_perturbations.jsonl)  News=$(wc -l < $ROOT/aug/News_perturbations.jsonl)"

#####################################################################
# Phase A — Books A1 8L + paraphrase + perturb (k_frac=0.40 already)
#####################################################################
step "A.1 — Books A1 retrain (8L + paraphrase + perturb)"
mv $ROOT/models/Books_a1 $ROOT/models/Books_a1_archive_v3 2>/dev/null || true
cd $ROOT
/usr/bin/python train_assistant.py \
    --split Books --role a1 \
    --epochs 5 --batch_size 1 --grad_accum 4 --num_layer 8 \
    --paraphrase_path $ROOT/aug/Books_paraphrases.jsonl \
    --perturb_path $ROOT/aug/Books_perturbations.jsonl \
    > $ROOT/logs/v3_books_a1.log 2>&1
sub_tail=$(tail -3 $ROOT/logs/v3_books_a1.log | grep loss=); step "  $sub_tail"

step "A.2 — Books A2 retrain (8L + perturb)"
mv $ROOT/models/Books_a2 $ROOT/models/Books_a2_archive_v3 2>/dev/null || true
/usr/bin/python train_assistant.py \
    --split Books --role a2 \
    --epochs 3 --batch_size 1 --grad_accum 4 --num_layer 8 \
    --perturb_path $ROOT/aug/Books_perturbations.jsonl \
    > $ROOT/logs/v3_books_a2.log 2>&1

#####################################################################
# Phase B — News A1 16L + r=64 + paraphrase + perturb (lr=5e-4, 10ep)
#####################################################################
step "B.1 — News A1 retrain (16L + r=64 + paraphrase + perturb + lr=5e-4 + 10ep)"
mv $ROOT/models/News_a1 $ROOT/models/News_a1_archive_v3 2>/dev/null || true
/usr/bin/python train_assistant.py \
    --split News --role a1 \
    --epochs 10 --batch_size 1 --grad_accum 4 \
    --num_layer 16 --lora_r 64 --lora_alpha 128 \
    --lr 5e-4 \
    --paraphrase_path $ROOT/aug/News_paraphrases.jsonl \
    --perturb_path $ROOT/aug/News_perturbations.jsonl \
    > $ROOT/logs/v3_news_a1.log 2>&1
sub_tail=$(tail -3 $ROOT/logs/v3_news_a1.log | grep loss=); step "  $sub_tail"

step "B.2 — News A2 retrain (16L + r=64 + perturb + 5ep)"
mv $ROOT/models/News_a2 $ROOT/models/News_a2_archive_v3 2>/dev/null || true
/usr/bin/python train_assistant.py \
    --split News --role a2 \
    --epochs 5 --batch_size 1 --grad_accum 4 \
    --num_layer 16 --lora_r 64 --lora_alpha 128 \
    --lr 5e-4 \
    --perturb_path $ROOT/aug/News_perturbations.jsonl \
    > $ROOT/logs/v3_news_a2.log 2>&1

#####################################################################
# Phase C — Fast sweeps
#####################################################################
step "C — clean stale fast evals + sweep"
rm -rf ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_News_DualULD_w*_fast 2>/dev/null
rm -rf ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_Books_DualULD_w*_fast 2>/dev/null

step "C.1 — News fast sweep"
bash sweep_eval.sh News "-0.5 -0.7 -0.9 -1.1 -1.3" > $ROOT/logs/v3_news_sweep.log 2>&1
step "  News sweep results:"
for f in ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_News_DualULD_w*_fast/MUSE_SUMMARY.json; do
    label=$(basename $(dirname $f) | sed 's/.*_DualULD_//;s/_fast//')
    v=$(/usr/bin/python -c "import json; d=json.load(open('$f')); print(f\"f={d.get('forget_knowmem_ROUGE',0)*100:5.1f} l={d.get('privleak',0):7.2f} r={d.get('retain_knowmem_ROUGE',0)*100:5.1f}\")")
    printf "    %-30s %s\n" "$label" "$v" | tee -a "$LOG" "$STATUS"
done

step "C.2 — Books fast sweep (sym + asym)"
bash sweep_eval.sh Books "-0.3 -0.5 -0.7 -0.9 -1.1" > $ROOT/logs/v3_books_sweep.log 2>&1
cat > $ROOT/sweep_books_v5_asym.txt <<EOF
w1m0p7_w20p3       -0.7   0.3   0.01
w1m0p7_w20p5       -0.7   0.5   0.01
w1m0p9_w20p3       -0.9   0.3   0.01
w1m0p9_w20p5       -0.9   0.5   0.01
EOF
bash sweep_eval2.sh Books $ROOT/sweep_books_v5_asym.txt >> $ROOT/logs/v3_books_sweep.log 2>&1
step "  Books sweep results:"
for f in ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_Books_DualULD_w*_fast/MUSE_SUMMARY.json; do
    label=$(basename $(dirname $f) | sed 's/.*_DualULD_//;s/_fast//')
    v=$(/usr/bin/python -c "import json; d=json.load(open('$f')); print(f\"f={d.get('forget_knowmem_ROUGE',0)*100:5.1f} l={d.get('privleak',0):7.2f} r={d.get('retain_knowmem_ROUGE',0)*100:5.1f}\")")
    printf "    %-30s %s\n" "$label" "$v" | tee -a "$LOG" "$STATUS"
done

step "ALL DONE — pick best weight + run full eval manually."
