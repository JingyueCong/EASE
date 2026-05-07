#!/usr/bin/env bash
# Phase 2 redo for both 3B forget01 and forget05 sweeps.
# Cleans buggy task-name entries from FQ logs (caused by grep matching the
# stray "1" in "a1s..."), re-extracts BEST_STEP, runs the w2 sweep with
# correct names, then re-aggregates both reports.
set -uo pipefail
ROOT="${EASE_ROOT}"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

eval_dual_3b() {
    local split="$1" a1="$2" a2="$3" w1="$4" w2="$5" task="$6" retain_ref="$7" fq_log="$8"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    if [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ]; then
        local fq=$("$PY" -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "  SKIP $task FQ=$fq"
        echo "$task $fq" >> "$fq_log"
        return
    fi
    rm -f "$eval_json" 2>/dev/null
    echo "  Eval $split $task (w1=$w1 w2=$w2)"
    cd "$OU_REPO"
    local holdout="${split/forget/holdout}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split="$split" holdout_split="$holdout" \
        eval.tofu.batch_size=2 \
        retain_logs_path="$retain_ref" \
        task_name="$task"
    local fq=$("$PY" -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
    [ -z "$fq" ] && fq="-1"
    echo "    FQ=$fq"
    echo "$task $fq" >> "$fq_log"
    cd "$ULD_REPO"
}

# Drop lines that don't have the canonical 2-field "task fq" format with proper task pattern.
filter_fq() {
    local in="$1" out="$2"
    awk 'NF==2 && $1 ~ /^tofu_.*_a1s[0-9]+_w1m0p8_w/ && $2 != "" {print}' "$in" > "$out"
}

run_split() {
    local split="$1" version="$2" retain_ref_split="$3"
    local fq_log="/tmp/3b_${version}_fq.txt"
    local fq_bak="/tmp/3b_${version}_fq.bak"
    local models_dir="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_${version}"
    local report="${ROOT}/uld_llama3_3b_${split}_dual_sweep.md"
    local retain_ref="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_${retain_ref_split}/TOFU_EVAL.json"

    echo "============================================================"
    echo "Phase 2 redo: $split (version $version)"
    echo "============================================================"

    [ -f "$fq_log" ] || { echo "  ERROR: $fq_log missing — skipping"; return; }
    cp "$fq_log" "$fq_bak"
    filter_fq "$fq_bak" "$fq_log"
    echo "  Cleaned FQ log: $(wc -l < "$fq_bak") → $(wc -l < "$fq_log") lines"

    local best_task best_step a1_parent a1 a2
    best_task=$(sort -k2 -t' ' -gr "$fq_log" | head -1 | awk '{print $1}')
    local a1s_token=$(echo "$best_task" | grep -oE 'a1s[0-9]+' | head -1)
    best_step=${a1s_token#a1s}
    a1_parent=$(find "${models_dir}/a1_${split}" -name "checkpoint-*" -type d | head -1 | xargs dirname)
    a1="${a1_parent}/checkpoint-${best_step}"
    a2=$(find "${models_dir}/a2_${split}" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

    [ -d "$a1" ] || { echo "  ERROR: a1 ckpt not found: $a1"; return; }
    [ -d "$a2" ] || { echo "  ERROR: a2 ckpt not found"; return; }

    echo "  BEST_STEP=$best_step"
    echo "  A1=$a1"
    echo "  A2=$a2"

    for w2 in 0.3 0.7 0.8 1.0; do
        local w2_str=$(echo "$w2" | tr '.' 'p')
        eval_dual_3b "$split" "$a1" "$a2" -0.8 "$w2" \
            "tofu_Llama-3.2-3B-Instruct_${split}_DualULD_v2${version}_a1s${best_step}_w1m0p8_w2${w2_str}" \
            "$retain_ref" "$fq_log"
    done

    echo ""
    echo "  Re-aggregating $report"
    "$PY" - "$report" "$OU_REPO" "$split" "$fq_log" <<'PYEOF'
import json, os, sys, datetime, pathlib, re
report_path, ou_repo, split, fq_log = sys.argv[1:5]
METRICS = ["forget_quality", "model_utility",
           "forget_truth_ratio", "forget_Q_A_Prob", "forget_Q_A_ROUGE",
           "privleak", "extraction_strength"]
def read_metric(d, key):
    v = d.get(key, None)
    if isinstance(v, dict) and "agg_value" in v:
        v = v["agg_value"]
    return v if isinstance(v, (int, float)) else None
def parse_task(task):
    a1step = re.search(r'a1s(\d+)', task)
    w1 = re.search(r'w1(m?\d+p\d+)', task)
    w2 = re.search(r'_w2(\d+p\d+)$', task) or re.search(r'_w20(p\d+)$', task)
    def unsan(s):
        return s.replace('m', '-').replace('p', '.') if s else None
    return (int(a1step.group(1)) if a1step else None,
            unsan(w1.group(1)) if w1 else None,
            unsan(w2.group(1)) if w2 else None)
tasks = []
with open(fq_log) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 2: continue
        tasks.append(parts[0])
seen = set(); uniq_tasks = []
for t in tasks:
    if t in seen: continue
    seen.add(t); uniq_tasks.append(t)
rows = []
for task in uniq_tasks:
    jp = os.path.join(ou_repo, "saves/eval", task, "TOFU_EVAL.json")
    a1step, w1, w2 = parse_task(task)
    if not os.path.isfile(jp):
        rows.append((a1step, w1, w2, task, {m: None for m in METRICS}, "MISSING")); continue
    try:
        d = json.load(open(jp))
        rows.append((a1step, w1, w2, task, {m: read_metric(d, m) for m in METRICS}, "ok"))
    except Exception as e:
        rows.append((a1step, w1, w2, task, {m: None for m in METRICS}, f"ERR:{e}"))
rows.sort(key=lambda r: (-(r[4].get("forget_quality") or -1)))
def fmt(v):
    if v is None: return "—"
    if isinstance(v, str): return v
    if abs(v) < 0.01 and v != 0: return f"{v:.2e}"
    return f"{v:.4f}"
recipe_a1ep = "20" if split == "forget01" else "10"
retain_ref_name = "retain99" if split == "forget01" else "retain95"
rsub_name = "forget01_k8" if split == "forget01" else "forget05_k40"
retain_num = "40" if split == "forget01" else "200"
lines = []
lines.append(f"# Dual-ULD {split} sweep — Llama-3.2-3B-Instruct")
lines.append("")
lines.append(f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
lines.append("")
lines.append("- Base: `open-unlearning/tofu_Llama-3.2-3B-Instruct_full`")
lines.append(f"- Recipe: Plan F/T mirror (num_layer=4, A1 ep={recipe_a1ep} / A2 ep=10, r=16, lr=1e-3, retain_weight=5.0)")
lines.append(f"- Retain reference: `tofu_Llama-3.2-3B-Instruct_{retain_ref_name}/TOFU_EVAL.json`")
lines.append(f"- rsub: `{rsub_name}.json`, retain_num={retain_num}")
lines.append("")
lines.append("Higher `forget_quality` is better (KS p-value vs retain).")
lines.append("")
header = "| A1 step | w1 | w2 | " + " | ".join(METRICS) + " | task |"
sep = "|" + "---|" * (len(METRICS) + 4)
lines.append(header); lines.append(sep)
for a1step, w1, w2, task, vals, status in rows:
    if status != "ok":
        lines.append(f"| {a1step} | {w1} | {w2} | " + " | ".join(["—"]*len(METRICS)) + f" | `{task}` ({status}) |"); continue
    cells = " | ".join(fmt(vals[m]) for m in METRICS)
    lines.append(f"| {a1step} | {w1} | {w2} | {cells} | `{task}` |")
lines.append("")
ok_rows = [r for r in rows if r[5] == "ok"]
if ok_rows:
    best = max(ok_rows, key=lambda r: (r[4]["forget_quality"] or -1))
    lines.append(f"**Best forget_quality**: A1 step={best[0]}, w1={best[1]}, w2={best[2]} → "
                 f"forget_quality={fmt(best[4]['forget_quality'])}, "
                 f"model_utility={fmt(best[4]['model_utility'])}")
    lines.append("")
pathlib.Path(report_path).write_text("\n".join(lines))
print(f"  → wrote {report_path} ({len(lines)} lines, {len(ok_rows)} ok rows)")
PYEOF
}

run_split forget01 f01 retain99
run_split forget05 f05 retain95

echo ""
echo "============================================================"
echo "Phase 2 redo DONE"
echo "  f01 report: ${ROOT}/uld_llama3_3b_forget01_dual_sweep.md"
echo "  f05 report: ${ROOT}/uld_llama3_3b_forget05_dual_sweep.md"
echo "============================================================"
