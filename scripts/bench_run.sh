#!/usr/bin/env bash
# bench_run.sh — Wrap a training (or training+eval) command and log
# wall-clock time + peak GPU memory + TOFU quality metrics to a single CSV.
#
# Usage:
#   GPU=0 ./bench_run.sh <method> <model_size> <forget_pct> <task_name> -- <cmd...>
#
# Args:
#   method       e.g. GradDiff, NPO, RMU, WGA, SimNPO, UNDIAL, DualULD
#   model_size   e.g. 1B, 3B, 8B  (free-form; just a CSV label)
#   forget_pct   e.g. forget01, forget05, forget10
#   task_name    must equal the Hydra task_name= passed to src/train.py or
#                src/eval.py — used to locate TOFU_SUMMARY.json afterwards.
#                Pass "-" to skip quality lookup (timing-only run).
#   --           separator before the actual command to wrap.
#
# Output (appended to ./efficiency_results.csv):
#   method,model_size,forget_pct,task_name,train_time_s,peak_mem_mb,
#   forget_quality,model_utility,forget_Q_A_ROUGE,timestamp
#
# Memory tracking: nvidia-smi polled at 1Hz on $GPU. Run on an otherwise idle
# GPU for clean readings; the value is total used-VRAM on that device, not
# just this process. For per-process accuracy, prefer the in-Python helper at
# the bottom of this file (commented snippet).

set -uo pipefail

if [ "$#" -lt 6 ] || [ "$5" != "--" ]; then
    sed -n '2,30p' "$0" >&2
    exit 2
fi

method="$1"; model_size="$2"; forget_pct="$3"; task_name="$4"
shift 5  # drop method, model_size, forget_pct, task_name, --

GPU="${GPU:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${BENCH_CSV:-${ROOT}/efficiency_results.csv}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"

TMP_MEM=$(mktemp)
cleanup() { [ -n "${POLL_PID:-}" ] && kill "$POLL_PID" 2>/dev/null; rm -f "$TMP_MEM"; }
trap cleanup EXIT

# Header (idempotent)
if [ ! -f "$CSV" ]; then
    echo "method,model_size,forget_pct,task_name,train_time_s,peak_mem_mb,forget_quality,model_utility,forget_Q_A_ROUGE,timestamp" > "$CSV"
fi

# Background memory poller @ 1Hz
(
    while :; do
        nvidia-smi --id="$GPU" --query-gpu=memory.used \
                   --format=csv,noheader,nounits 2>/dev/null >> "$TMP_MEM" || true
        sleep 1
    done
) &
POLL_PID=$!

# Run + time
echo "[bench_run] start: ${method} ${model_size} ${forget_pct} task=${task_name} gpu=${GPU}"
t0=$(date +%s.%N)
"$@"
status=$?
t1=$(date +%s.%N)

kill "$POLL_PID" 2>/dev/null; POLL_PID=""
train_time=$(awk -v t0="$t0" -v t1="$t1" 'BEGIN{printf "%.2f", t1-t0}')
peak_mem=$(awk 'BEGIN{m=0} {v=$1+0; if (v>m) m=v} END{print m}' "$TMP_MEM")

# Quality lookup (skip if task_name == "-")
fq="NA"; mu="NA"; rouge="NA"
if [ "$task_name" != "-" ]; then
    SUMMARY="${OU_REPO}/saves/eval/${task_name}/TOFU_SUMMARY.json"
    [ -f "$SUMMARY" ] || SUMMARY="${OU_REPO}/saves/unlearn/${task_name}/TOFU_SUMMARY.json"
    if [ -f "$SUMMARY" ]; then
        read fq mu rouge < <(python3 -c "
import json
d = json.load(open('${SUMMARY}'))
def g(k): v = d.get(k); return 'NA' if v is None else f'{v:.6f}'
print(g('forget_quality'), g('model_utility'), g('forget_Q_A_ROUGE'))
")
    else
        echo "[bench_run] WARN: no TOFU_SUMMARY.json found for task '${task_name}' (looked in saves/eval and saves/unlearn)" >&2
    fi
fi

ts=$(date -Iseconds)
echo "${method},${model_size},${forget_pct},${task_name},${train_time},${peak_mem},${fq},${mu},${rouge},${ts}" >> "$CSV"

echo "[bench_run] done:  time=${train_time}s peak_mem=${peak_mem}MB fq=${fq} mu=${mu} rouge=${rouge} (status=${status})"
exit "$status"

# -----------------------------------------------------------------------------
# Optional: drop-in Python helper for per-process memory (more accurate than
# nvidia-smi polling). Insert into your training script around the train loop:
#
#   import time, json, os, torch
#   torch.cuda.reset_peak_memory_stats()
#   t0 = time.time()
#   trainer.train()
#   stats = {
#       "train_time_s": round(time.time() - t0, 2),
#       "peak_mem_mb":  torch.cuda.max_memory_allocated() // (1024**2),
#   }
#   os.makedirs(cfg.paths.output_dir, exist_ok=True)
#   with open(os.path.join(cfg.paths.output_dir, "bench.json"), "w") as f:
#       json.dump(stats, f)
#
# Then bench_run.sh can read bench.json instead of polling nvidia-smi.
# -----------------------------------------------------------------------------
