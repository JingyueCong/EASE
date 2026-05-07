#!/usr/bin/env bash
#
# Hyperparam sweep for Dual-ULD on Llama-3.2-1B-Instruct, single forget split.
# Reuses A1/A2 trained by run_dual_uld_1b.sh; only re-runs Phase C eval for
# each weight_a1 value (weight_a2 fixed) with a unique task_name per combo.
#
# Usage:
#   SPLIT=forget01 bash sweep_dual_1b_forget.sh
#   SPLIT=forget01 W1S="-0.4 -0.5 -0.6 -0.7" bash sweep_dual_1b_forget.sh
#   SPLIT=forget05 W1S="-0.5" W2="0.5" bash sweep_dual_1b_forget.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"

GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
EVAL_BS="${EVAL_BS:-4}"

W1S="${W1S:--0.4 -0.5 -0.6 -0.7 -0.8}"
W2="${W2:-1.0}"
TOPF="${TOPF:-0.01}"

SPLIT="${SPLIT:-forget01}"
case "$SPLIT" in
    forget01) holdout=holdout01; retain=retain99 ;;
    forget05) holdout=holdout05; retain=retain95 ;;
    forget10) holdout=holdout10; retain=retain90 ;;
    *) echo "ERROR: SPLIT must be forget01/05/10"; exit 1 ;;
esac

REPORT_PATH="${REPORT_PATH:-${ROOT}/uld_llama3_1b_${SPLIT}_dual_sweep.md}"
MODELS_ROOT="${MODELS_ROOT:-${ULD_REPO}/outputs_trained_models/llama3_1b_dual}"

retain_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json"
[ -f "$retain_json" ] || { echo "ERROR: retain JSON missing: $retain_json"; exit 1; }

a1_dir="${MODELS_ROOT}/a1_${SPLIT}"
a2_dir="${MODELS_ROOT}/a2_${SPLIT}"
a1_ck=$(find "$a1_dir" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
a2_ck=$(find "$a2_dir" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$a1_ck" ] || { echo "ERROR: no A1 ckpt in $a1_dir"; exit 1; }
[ -n "$a2_ck" ] || { echo "ERROR: no A2 ckpt in $a2_dir"; exit 1; }

sanitize() { echo "$1" | sed -e 's/-/m/g' -e 's/\./p/g'; }

echo "============================================================"
echo "Dual-ULD ${SPLIT} sweep — Llama-3.2-1B-Instruct"
echo "  A1 ckpt           : $a1_ck"
echo "  A2 ckpt           : $a2_ck"
echo "  W1 grid (W2=$W2) : $W1S"
echo "  topF              : $TOPF"
echo "  report            : $REPORT_PATH"
echo "============================================================"
cd "$OU_REPO"

declare -a runs

for w1 in $W1S; do
    w1s=$(sanitize "$w1")
    w2s=$(sanitize "$W2")
    ts=$(sanitize "$TOPF")
    task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_DualULD_w1${w1s}_w2${w2s}_topf${ts}"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    runs+=("${w1}|${W2}|${TOPF}|${task}|${eval_json}")
    if [ -f "$eval_json" ]; then
        echo "  → SKIP w1=$w1 : exists"; continue
    fi
    echo "  → Eval w1=$w1 w2=$W2 topF=$TOPF → $eval_json"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1_ck" \
        model.model_args.a2_path="$a2_ck" \
        model.model_args.weight_a1="$w1" \
        model.model_args.weight_a2="$W2" \
        model.model_args.top_logit_filter="$TOPF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split="$SPLIT" \
        holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        retain_logs_path="$retain_json" \
        task_name="$task"
done

# Include baseline (default config) under the standard name as well
base_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${SPLIT}_DualULD/TOFU_EVAL.json"
if [ -f "$base_json" ]; then
    runs+=("-1.0|1.0|$TOPF|tofu_Llama-3.2-1B-Instruct_${SPLIT}_DualULD|${base_json}")
fi

echo
echo "[Aggregate] writing $REPORT_PATH"
"$PY" - "$REPORT_PATH" "${runs[@]}" <<'PYEOF'
import json, os, sys, datetime, pathlib
from statistics import harmonic_mean

report_path = sys.argv[1]
rows = []
for r in sys.argv[2:]:
    w1, w2, t, task, jp = r.split("|", 4)
    rows.append((float(w1), float(w2), float(t), task, jp))

METRICS = ['forget_quality','model_utility','forget_truth_ratio','forget_Q_A_Prob','forget_Q_A_ROUGE','retain_Q_A_ROUGE','privleak','extraction_strength']

def get(d, k):
    v = d.get(k); return v.get('agg_value', v) if isinstance(v, dict) else v

def derive(jp):
    if not os.path.isfile(jp): return None
    try: d = json.load(open(jp))
    except Exception: return None
    out = {m: get(d, m) for m in METRICS}
    fp, frl, ft, util = out['forget_Q_A_Prob'], out['forget_Q_A_ROUGE'], out['forget_truth_ratio'], out['model_utility']
    if None not in (fp, frl, ft, util):
        out['Mem'] = harmonic_mean([max(1-fp,1e-6), max(1-frl,1e-6), max(ft,1e-6)])
        out['Util'] = util
        out['Agg'] = harmonic_mean([max(out['Mem'],1e-6), max(out['Util'],1e-6)])
    return out

def fmt(v):
    if v is None: return '—'
    if isinstance(v, str): return v
    if abs(v) < 0.01 and v != 0: return f'{v:.2e}'
    return f'{v:.4f}'

cols = ['Agg','Mem','Util','forget_quality','forget_Q_A_ROUGE','model_utility','retain_Q_A_ROUGE',
        'forget_truth_ratio','forget_Q_A_Prob','privleak','extraction_strength']

processed = []
for w1, w2, t, task, jp in rows:
    r = derive(jp)
    processed.append((w1, w2, t, task, r))
processed.sort(key=lambda x: -x[0])

lines = []
lines.append(f'# Dual-ULD weight_a1 sweep — Llama-3.2-1B-Instruct ({rows[0][3].split("_DualULD")[0].split("_")[-1] if rows else ""})')
lines.append('')
lines.append(f'_Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_')
lines.append('')
header = '| weight_a1 | weight_a2 | topF | ' + ' | '.join(cols) + ' |'
sep    = '|' + '---|' * (len(cols) + 3)
lines.append(header); lines.append(sep)
for w1, w2, t, task, r in processed:
    if r is None:
        lines.append(f'| {w1} | {w2} | {t} | ' + ' | '.join(['MISSING']*len(cols)) + ' |')
    else:
        lines.append(f'| {w1} | {w2} | {t} | ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')
lines.append('')
ok = [p for p in processed if p[4] is not None and p[4].get('forget_quality') is not None]
if ok:
    best = max(ok, key=lambda p: p[4]['forget_quality'])
    r = best[4]
    lines.append(f'**Best forget_quality**: w1={best[0]}, w2={best[1]}, topF={best[2]} → fq={fmt(r["forget_quality"])}, util={fmt(r["model_utility"])}, mem={fmt(r.get("Mem"))}, agg={fmt(r.get("Agg"))}')
    lines.append('')
pathlib.Path(report_path).write_text('\n'.join(lines))
print(f'  → wrote {report_path}')
PYEOF

echo
echo "============================================================"
echo "DONE — see $REPORT_PATH"
echo "============================================================"
