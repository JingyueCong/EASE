#!/usr/bin/env bash
#
# Dual-ULD on Llama-3.2-1B-Instruct, evaluated with open-unlearning TOFU.
# Mirror of run_uld_open_unlearning_1b.sh but trains TWO assistants per split:
#   A1: data_mode=dual_a1  (forget ∪ R_sub  → GD ; R_far + perturb → uniform)
#   A2: data_mode=dual_a2  (R_sub only      → GD ; forget + R_far + perturb → uniform)
# At inference: final_logits = base + w1·A1 + w2·A2.
#
# Phases:
#   A. Reuse retain TOFU_EVAL.json on disk (same as ULD pipeline).
#   B1. Train A1 for each split.
#   B2. Train A2 for each split.
#   C.  Eval with DualULDForCausalLM.
#   D.  Aggregate metrics into $REPORT_PATH.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"

GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"

NUM_LAYER="${NUM_LAYER:-2}"
LORA_R="${LORA_R:-16}"
WEIGHT_A1="${WEIGHT_A1:--1.0}"
WEIGHT_A2="${WEIGHT_A2:-1.0}"
TOP_FILTER="${TOP_FILTER:-0.01}"   # match ULD-on-1B finding

TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_EP="${TRAIN_EP:-10}"
EVAL_BS="${EVAL_BS:-4}"

MODELS_ROOT="${MODELS_ROOT:-${ULD_REPO}/outputs_trained_models/llama3_1b_dual}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/uld_llama3_1b_dual_results.md}"

# Per-split: forget holdout retain rsub_path retain_num
SPLITS=(
    "forget01 holdout01 retain99 ${ULD_REPO}/data/rsub/forget01_k8.json   40"
    "forget05 holdout05 retain95 ${ULD_REPO}/data/rsub/forget05_k80.json  200"
    "forget10 holdout10 retain90 ${ULD_REPO}/data/rsub/forget10_k80.json  400"
)
if [ -n "${ONLY:-}" ]; then
    new=()
    for sp in "${SPLITS[@]}"; do
        case "$sp" in "$ONLY"*) new+=("$sp");; esac
    done
    SPLITS=("${new[@]}")
fi

mkdir -p "$MODELS_ROOT"
echo "============================================================"
echo "Dual-ULD × open-unlearning, Llama-3.2-1B-Instruct"
echo "  GPU                : $GPU"
echo "  num_layer / lora_r : $NUM_LAYER / $LORA_R"
echo "  weight_a1 / weight_a2 / topF : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  splits             :"
for sp in "${SPLITS[@]}"; do echo "    $sp"; done
echo "  models root        : $MODELS_ROOT"
echo "  report             : $REPORT_PATH"
echo "============================================================"

#####################################
# Phase B — train A1 and A2 per split
#####################################
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false

train_assistant() {
    local role="$1"            # a1 or a2
    local forget="$2"
    local rsub_path="$3"
    local retain_num="$4"
    local out_dir="${MODELS_ROOT}/${role}_${forget}"

    if find "$out_dir" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $role/$forget : checkpoint exists at $out_dir"
        return
    fi
    echo "  → Train $role/$forget → $out_dir"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_dual_${role}_${forget}" \
        data=tofu_chat3 \
        data.dataset.split="${forget}_perturbed" \
        data_mode="dual_${role}" \
        data_mode.r_sub_indices_path="$rsub_path" \
        data_mode.retain_num="$retain_num" \
        model=llama-3-1b \
        model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld \
        model_mode.num_layer="$NUM_LAYER" \
        model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" \
        trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" \
        trainer.max_epochs="$TRAIN_EP" \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$out_dir" \
        postfix="${role}" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_dual_${role}_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
}

echo
echo "[Phase B] Train A1 + A2 assistants"
for sp in "${SPLITS[@]}"; do
    forget=$(echo "$sp" | awk '{print $1}')
    rsub=$(echo "$sp" | awk '{print $4}')
    rnum=$(echo "$sp" | awk '{print $5}')
    [ -f "$rsub" ] || { echo "  ✗ missing R_sub: $rsub"; exit 1; }
    train_assistant a1 "$forget" "$rsub" "$rnum"
    train_assistant a2 "$forget" "$rsub" "$rnum"
done

#####################################
# Phase C — eval Dual-ULD
#####################################
cd "$OU_REPO"
echo
echo "[Phase C] Eval Dual-ULD with open-unlearning TOFU metrics"
for sp in "${SPLITS[@]}"; do
    forget=$(echo "$sp" | awk '{print $1}')
    holdout=$(echo "$sp" | awk '{print $2}')
    retain=$(echo "$sp" | awk '{print $3}')

    a1_dir="${MODELS_ROOT}/a1_${forget}"
    a2_dir="${MODELS_ROOT}/a2_${forget}"
    a1_ck=$(find "$a1_dir" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    a2_ck=$(find "$a2_dir" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$a1_ck" ] || { echo "  ✗ no A1 ckpt for $forget"; exit 1; }
    [ -n "$a2_ck" ] || { echo "  ✗ no A2 ckpt for $forget"; exit 1; }

    task="tofu_Llama-3.2-1B-Instruct_${forget}_DualULD"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
    retain_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json"
    if [ -f "$eval_json" ]; then
        echo "  → SKIP $forget: $eval_json exists"; continue
    fi
    [ -f "$retain_json" ] || { echo "  ! retain JSON missing, forget_quality skipped"; }

    retain_arg="retain_logs_path=$retain_json"
    [ -f "$retain_json" ] || retain_arg="retain_logs_path=null"

    echo "  → Eval Dual-ULD on $forget (a1=$a1_ck, a2=$a2_ck) → $eval_json"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct_DualULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.a1_path="$a1_ck" \
        model.model_args.a2_path="$a2_ck" \
        model.model_args.weight_a1="$WEIGHT_A1" \
        model.model_args.weight_a2="$WEIGHT_A2" \
        model.model_args.top_logit_filter="$TOP_FILTER" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split="$forget" \
        holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        $retain_arg \
        task_name="$task"
done

#####################################
# Phase D — aggregate
#####################################
echo
echo "[Phase D] Aggregating → $REPORT_PATH"
"$PY" - "$REPORT_PATH" "$OU_REPO" "$WEIGHT_A1" "$WEIGHT_A2" "$TOP_FILTER" "$NUM_LAYER" "$LORA_R" "${SPLITS[@]}" <<'PYEOF'
import json, os, sys, datetime, pathlib
from statistics import harmonic_mean

report_path = sys.argv[1]
ou_repo     = sys.argv[2]
w1, w2, topf = sys.argv[3], sys.argv[4], sys.argv[5]
nl, lr      = sys.argv[6], sys.argv[7]
splits_args = sys.argv[8:]

splits = []
for sp in splits_args:
    parts = sp.split()
    splits.append((parts[0], parts[2]))   # (forget, retain)

METRICS = ['forget_quality','model_utility','forget_truth_ratio','forget_Q_A_Prob',
           'forget_Q_A_ROUGE','retain_Q_A_ROUGE','privleak','extraction_strength']

def read(jp):
    if not os.path.isfile(jp): return None
    try:
        d = json.load(open(jp))
    except Exception:
        return None
    return {m: (d.get(m,{}).get('agg_value') if isinstance(d.get(m), dict) else d.get(m)) for m in METRICS + ['forget_Q_A_PARA_Prob','forget_Q_A_PERT_Prob']}

def derived(r):
    if r is None: return None
    fp, frl, ft = r.get('forget_Q_A_Prob'), r.get('forget_Q_A_ROUGE'), r.get('forget_truth_ratio')
    util = r.get('model_utility')
    if None in (fp, frl, ft, util): return r
    mem = harmonic_mean([max(1-fp,1e-6), max(1-frl,1e-6), max(ft,1e-6)])
    agg = harmonic_mean([max(mem,1e-6), max(util,1e-6)])
    r['Mem'] = mem; r['Util'] = util; r['Agg'] = agg
    return r

rows_dual = {}
rows_retain = {}
for forget, retain in splits:
    j = f"{ou_repo}/saves/eval/tofu_Llama-3.2-1B-Instruct_{forget}_DualULD/TOFU_EVAL.json"
    rows_dual[forget] = derived(read(j))
    j = f"{ou_repo}/saves/eval/tofu_Llama-3.2-1B-Instruct_{retain}/TOFU_EVAL.json"
    rows_retain[retain] = derived(read(j))

def fmt(v):
    if v is None: return '—'
    if isinstance(v, str): return v
    if abs(v) < 0.01 and v != 0: return f'{v:.2e}'
    return f'{v:.4f}'

cols = ['Agg','Mem','Util','forget_quality','forget_Q_A_ROUGE','model_utility','retain_Q_A_ROUGE',
        'forget_truth_ratio','forget_Q_A_Prob','privleak','extraction_strength']

lines = []
lines.append('# Dual-ULD on Llama-3.2-1B-Instruct — open-unlearning TOFU evaluation')
lines.append('')
lines.append(f'_Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_')
lines.append('')
lines.append('## Configuration')
lines.append(f'- Base: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`')
lines.append(f'- Two assistants A1, A2: each {nl}-layer LoRA r={lr}, 10 epochs, `remember+uniform` loss')
lines.append(f'- Inference: `weight_a1={w1}`, `weight_a2={w2}`, `top_logit_filter={topf}`')
lines.append('')
lines.append('## Dual-ULD — TOFU metrics')
lines.append('')
header = '| split | ' + ' | '.join(cols) + ' |'
sep    = '|' + '---|' * (len(cols) + 1)
lines.append(header); lines.append(sep)
for forget,_ in splits:
    r = rows_dual[forget]
    if r is None:
        lines.append(f'| {forget} | ' + ' | '.join(['MISSING']*len(cols)) + ' |')
    else:
        lines.append(f'| {forget} | ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')
lines.append('')
lines.append('## Retain reference')
lines.append('')
lines.append(header.replace('split','retain')); lines.append(sep)
for _,retain in splits:
    r = rows_retain[retain]
    if r is None:
        lines.append(f'| {retain} | ' + ' | '.join(['MISSING']*len(cols)) + ' |')
    else:
        lines.append(f'| {retain} | ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')
lines.append('')
lines.append('## LaTeX-ready Ours row')
lines.append('')
def fmt2(v): return f'{v:.2f}' if isinstance(v,(int,float)) else '--'
seg = []
for forget,_ in splits:
    r = rows_dual[forget]
    if r is None:
        seg.append('-- '*7); continue
    s = ['Agg','Mem','Util','forget_quality','forget_Q_A_ROUGE','model_utility','retain_Q_A_ROUGE']
    seg.append(' & '.join(fmt2(r.get(k)) for k in s))
lines.append('```latex')
lines.append('\\textbf{Ours}')
lines.append('& ' + seg[0] if len(seg)>=1 else '& ' + '-- & '*6 + '--')
lines.append('& ' + seg[1] if len(seg)>=2 else '& ' + '-- & '*6 + '--')
lines.append('& ' + (seg[2] if len(seg)>=3 else '-- & '*6 + '--') + ' \\\\')
lines.append('```')
lines.append('')

pathlib.Path(report_path).write_text('\n'.join(lines))
print(f'  → wrote {report_path}')
PYEOF
echo
echo "============================================================"
echo "DONE — see $REPORT_PATH"
echo "============================================================"
