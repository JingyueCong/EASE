#!/usr/bin/env bash
# Plan f01: 3B forget01 dual-ULD sweep, mirrored from Plan F/T (forget10) recipe
#   - num_layer=4, ep=20 (A1) / ep=10 (A2), lr=1e-3, r=16, bs=4 ga=4, w2=0.5
#   - retain_num=40 (1/10 of f10), rsub=forget01_k8.json
#   - retain ref: tofu_Llama-3.2-3B-Instruct_retain99
# Phase 1: scan A1 ckpts at default w2/weight_a1
# Phase 2: w2 sweep at best A1 ckpt
# Phase 3: aggregate all TOFU_EVAL.json files into a markdown report
set -uo pipefail
ROOT="${EASE_ROOT}"
ULD_REPO="${ROOT}/ULD"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_BASE_PREFIX="open-unlearning/tofu_Llama-3.2-3B-Instruct"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
MODELS_DIR="${ULD_REPO}/outputs_trained_models/llama3_3b_dual_v2_f01"
RSUB="${ULD_REPO}/data/rsub/forget01_k8.json"
RETAIN_REF="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json"
REPORT_PATH="${ROOT}/uld_llama3_3b_forget01_dual_sweep.md"
FQ_LOG="/tmp/3b_f01_fq.txt"

[ -f "$RETAIN_REF" ] || { echo "ERROR: retain99 ref missing: $RETAIN_REF"; exit 1; }
[ -f "$RSUB" ] || { echo "ERROR: rsub indices missing: $RSUB"; exit 1; }
mkdir -p "$MODELS_DIR"

cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0; export TOKENIZERS_PARALLELISM=false

train_a() {
    local role="$1" ep="$2" save_steps="$3"
    local out="${MODELS_DIR}/a${role}_forget01"
    if find "$out" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP A${role} : checkpoints exist in $out"; return
    fi
    echo "  → Train A${role} (ep=$ep, save_steps=$save_steps, num_layer=4, r=16, lr=1e-3)"
    local data_role="dual_a${role}"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_3b_dual_v2_f01_a${role}_forget01" \
        data=tofu_chat3 data.dataset.split="forget01_perturbed" \
        data_mode="$data_role" \
        data_mode.r_sub_indices_path="$RSUB" \
        data_mode.retain_num=40 \
        model=llama-3-3b model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld model_mode.num_layer=4 model_mode.Lora.r=16 \
        unlearn_loss=remember+uniform unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=4 trainer.gradient_accumulation_steps=4 \
        trainer.learning_rate=1e-3 trainer.max_epochs="$ep" \
        +trainer.save_steps="$save_steps" \
        trainer.strategy=gpu OUTPUTMODELDIR="$out" postfix="f01a${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_3b_dual_v2_f01_a${role}_forget01/\${now:%Y-%m-%d_%H-%M-%S}"
}

eval_dual() {
    local a1="$1" a2="$2" w1="$3" w2="$4" task="$5"
    local eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    if [ -f "$eval_json" ] && [ "$(stat -c%s "$eval_json")" -gt 100 ]; then
        local fq=$("$PY" -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
        echo "  SKIP $task FQ=$fq"
        echo "$task $fq" >> "$FQ_LOG"
        return
    fi
    rm -f "$eval_json" 2>/dev/null
    echo "  Eval $task (w1=$w1 w2=$w2)"
    cd "$OU_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default model=Llama-3.2-3B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1" model.model_args.a2_path="$a2" \
        model.model_args.weight_a1="$w1" model.model_args.weight_a2="$w2" \
        model.model_args.top_logit_filter=0.01 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
        forget_split=forget01 holdout_split=holdout01 \
        eval.tofu.batch_size=2 \
        retain_logs_path="$RETAIN_REF" \
        task_name="$task"
    local fq=$("$PY" -c "import json; d=json.load(open('$eval_json')); print(d.get('forget_quality',{}).get('agg_value',-1))" 2>/dev/null)
    [ -z "$fq" ] && fq="-1"
    echo "    FQ=$fq"
    echo "$task $fq" >> "$FQ_LOG"
    cd "$ULD_REPO"
}

> "$FQ_LOG"

echo "============================================================"
echo "Plan f01: 3B dual-ULD forget01 sweep"
echo "  models  : $MODELS_DIR"
echo "  rsub    : $RSUB"
echo "  retain  : $RETAIN_REF"
echo "  report  : $REPORT_PATH"
echo "============================================================"

# Train A1 (ep=20, save_steps=5 → ckpts at 5,10,...,~100) and A2 (ep=10)
# forget01: 40 retain + ~40 perturbed → ~5 steps/epoch (bs=4 ga=4 effective bs=16)
echo ""
echo "=== Training ==="
train_a 1 20 5
train_a 2 10 5

A1_PARENT=$(find "${MODELS_DIR}/a1_forget01" -name "checkpoint-*" -type d | head -1 | xargs dirname)
A2=$(find "${MODELS_DIR}/a2_forget01" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "A1 parent: $A1_PARENT"
echo "A2 final:  $A2"

# Phase 1: A1 ckpt scan with default w1=-0.8, w2=0.5
echo ""
echo "=== Phase 1: A1 ckpt scan (w1=-0.8 w2=0.5) ==="
declare -a A1_STEPS
mapfile -t A1_STEPS < <(find "$A1_PARENT" -maxdepth 1 -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF}' | sort -n)
# subsample if too many
if [ "${#A1_STEPS[@]}" -gt 8 ]; then
    n=${#A1_STEPS[@]}
    SCAN_STEPS=()
    for i in 0 $((n/6)) $((n/3)) $((n/2)) $((2*n/3)) $((5*n/6)) $((n-1)); do
        SCAN_STEPS+=("${A1_STEPS[$i]}")
    done
else
    SCAN_STEPS=("${A1_STEPS[@]}")
fi
echo "  Scanning A1 steps: ${SCAN_STEPS[*]}"
for step in "${SCAN_STEPS[@]}"; do
    A1="${A1_PARENT}/checkpoint-${step}"
    [ -d "$A1" ] || continue
    eval_dual "$A1" "$A2" -0.8 0.5 "tofu_Llama-3.2-3B-Instruct_forget01_DualULD_v2f01_a1s${step}_w1m0p8_w20p5"
done

# Phase 2: w2 sweep at the best A1 ckpt found in Phase 1
echo ""
echo "=== Phase 2: w2 sweep at best A1 ckpt (w1=-0.8) ==="
BEST_TASK=$(sort -k2 -t' ' -gr "$FQ_LOG" | head -1 | awk '{print $1}')
A1S_TOKEN=$(echo "$BEST_TASK" | grep -oE 'a1s[0-9]+' | head -1)
BEST_STEP=${A1S_TOKEN#a1s}
BEST_A1="${A1_PARENT}/checkpoint-${BEST_STEP}"
echo "Best A1 step from phase 1: $BEST_STEP"
for w2 in 0.3 0.7 0.8 1.0; do
    w2_str=$(echo "$w2" | tr '.' 'p')
    eval_dual "$BEST_A1" "$A2" -0.8 "$w2" "tofu_Llama-3.2-3B-Instruct_forget01_DualULD_v2f01_a1s${BEST_STEP}_w1m0p8_w2${w2_str}"
done

echo ""
echo "============================================================"
echo "FQ summary (sorted):"
sort -k2 -t' ' -gr "$FQ_LOG"
echo "============================================================"
best_fq=$(awk '{print $2}' "$FQ_LOG" | sort -gr | head -1)
echo "Best forget01 FQ: $best_fq"

# Phase 3: aggregate full metrics from all TOFU_EVAL.json files
echo ""
echo "=== Phase 3: aggregating to $REPORT_PATH ==="
"$PY" - "$REPORT_PATH" "$OU_REPO" forget01 "$FQ_LOG" <<'PYEOF'
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
    w2 = re.search(r'w20(p\d+)', task)
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
        tasks.append(parts[0])

rows = []
for task in tasks:
    jp = os.path.join(ou_repo, "saves/eval", task, "TOFU_EVAL.json")
    a1step, w1, w2 = parse_task(task)
    if not os.path.isfile(jp):
        rows.append((a1step, w1, w2, task, {m: None for m in METRICS}, "MISSING"))
        continue
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

lines = []
lines.append(f"# Dual-ULD {split} sweep — Llama-3.2-3B-Instruct")
lines.append("")
lines.append(f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
lines.append("")
lines.append("- Base: `open-unlearning/tofu_Llama-3.2-3B-Instruct_full`")
lines.append("- Recipe: Plan F/T mirror (num_layer=4, A1 ep=20 / A2 ep=10, r=16, lr=1e-3, retain_weight=5.0)")
lines.append("- Retain reference: `tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json`")
lines.append("- rsub: `forget01_k8.json`, retain_num=40")
lines.append("")
lines.append("Higher `forget_quality` is better (KS p-value vs retain).")
lines.append("")
header = "| A1 step | w1 | w2 | " + " | ".join(METRICS) + " | task |"
sep = "|" + "---|" * (len(METRICS) + 4)
lines.append(header); lines.append(sep)
for a1step, w1, w2, task, vals, status in rows:
    if status != "ok":
        lines.append(f"| {a1step} | {w1} | {w2} | " + " | ".join(["—"]*len(METRICS)) + f" | `{task}` ({status}) |")
        continue
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
print(f"  → wrote {report_path} ({len(lines)} lines)")
PYEOF

echo "DONE — see $REPORT_PATH"
