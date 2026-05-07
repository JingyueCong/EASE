#!/usr/bin/env bash
# Master sweep: waits for News A1 15ep to finish, then sweeps both splits over
# symmetric + asymmetric weights, then runs full eval at each split's best weight.
#
# Output summary: ${EASE_ROOT}/dual_uld_muse/logs/master_sweep_summary.txt

set -uo pipefail
ROOT=${EASE_ROOT}/dual_uld_muse
LOG=$ROOT/logs/master_sweep.log
SUMMARY=$ROOT/logs/master_sweep_summary.txt
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

#############################
# Phase A — wait for News A1 retrain
#############################
log "Phase A — wait for News A1 15-epoch retrain"
until [ -f "$ROOT/models/News_a1/checkpoint-final/adapter_model.safetensors" ]; do
    sleep 60
done
log "News A1 ready"

#############################
# Phase B — fast sweeps
#############################
# Books: 5 symmetric already done, queue asymmetric
BOOKS_CFG=$ROOT/sweep_books_asym.txt
cat > "$BOOKS_CFG" <<EOF
# label              w1    w2   topF
w1m0p7_w20p3       -0.7   0.3   0.01
w1m0p7_w20p5       -0.7   0.5   0.01
w1m0p9_w20p3       -0.9   0.3   0.01
w1m0p9_w20p5       -0.9   0.5   0.01
w1m0p9_w20p7       -0.9   0.7   0.01
w1m1p1_w20p3       -1.1   0.3   0.01
w1m1p1_w20p5       -1.1   0.5   0.01
EOF

# News: full sweep with new A1 (sym + asym)
NEWS_CFG=$ROOT/sweep_news_all.txt
cat > "$NEWS_CFG" <<EOF
# label              w1    w2   topF
w1m0p5_w20p5       -0.5   0.5   0.01
w1m0p7_w20p7       -0.7   0.7   0.01
w1m0p9_w20p9       -0.9   0.9   0.01
w1m1p1_w21p1       -1.1   1.1   0.01
w1m1p3_w21p3       -1.3   1.3   0.01
w1m0p9_w20p3       -0.9   0.3   0.01
w1m0p9_w20p5       -0.9   0.5   0.01
w1m1p1_w20p3       -1.1   0.3   0.01
w1m1p1_w20p5       -1.1   0.5   0.01
EOF

# Clean any stale News fast eval results since A1 changed
log "Clearing stale News fast eval results"
rm -rf ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_News_DualULD_w*_fast 2>/dev/null

log "Phase B.1 — Books asymmetric sweep"
cd "$ROOT"
bash sweep_eval2.sh Books "$BOOKS_CFG" 2>&1 | tee -a "$LOG"

log "Phase B.2 — News all-config sweep"
bash sweep_eval2.sh News "$NEWS_CFG" 2>&1 | tee -a "$LOG"

#############################
# Phase C — collect summaries
#############################
collect() {
    local split="$1"
    echo "=== $split ===" >> "$SUMMARY"
    for f in ${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_${split}_DualULD_*_fast/MUSE_SUMMARY.json; do
        [ -f "$f" ] || continue
        local label=$(basename $(dirname "$f") | sed "s/.*_DualULD_//;s/_fast//")
        local v=$(/usr/bin/python -c "
import json
try:
    d = json.load(open('$f'))
    forget = d.get('forget_knowmem_ROUGE', 0) * 100
    leak = d.get('privleak', 0)
    retain = d.get('retain_knowmem_ROUGE', 0) * 100
    print(f'forget={forget:5.1f}  leak={leak:7.2f}  retain={retain:5.1f}')
except Exception as e:
    print(f'ERR {e}')
")
        printf "  %-30s %s\n" "$label" "$v" >> "$SUMMARY"
    done
}
echo "MUSE Dual-ULD master sweep results — $(date)" > "$SUMMARY"
collect Books
echo "" >> "$SUMMARY"
collect News
echo "" >> "$SUMMARY"
log "Sweep summary:"
cat "$SUMMARY" | tee -a "$LOG"
log "DONE — pick best weight per split, then run full eval"
