#!/usr/bin/env bash
#
# One-click ULD on Llama-3.2-1B-Instruct, evaluated with the open-unlearning
# TOFU benchmark. Mirror of run_uld_open_unlearning_8b.sh but for the 1B model.
#
# Phases (idempotent — each step skips if its output already exists):
#   A. Reuse open-unlearning retain-model TOFU_EVAL.json (already on disk for
#      Llama-3.2-1B-Instruct_retain{99,95,90}). Skipped automatically if found.
#   B. Train a ULD assistant for each forget split, against the
#      open-unlearning Llama-3.2-1B-Instruct full-finetune.
#   C. Eval each ULD model (base + assistant) using the open-unlearning
#      TOFU evaluator and write per-split TOFU_EVAL.json.
#   D. Aggregate the 7 TOFU metrics from all splits into a single Markdown
#      report at $REPORT_PATH.
#
# Usage:
#   bash run_uld_open_unlearning_1b.sh
#   GPU=0 ONLY=forget01 bash run_uld_open_unlearning_1b.sh   # smoke test
#
# Hardware: a single A100-40GB is overkill for 1B; runs much faster than 8B.

set -euo pipefail

#####################################
# Config
#####################################
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"

GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.2-1B-Instruct}"
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"

# Llama-3.2-1B has 16 hidden layers (vs 32 for 8B). Default to 2 layers for the
# assistant — same ~12.5% ratio used in the 8B run (4/32). Override via env.
NUM_LAYER="${NUM_LAYER:-2}"
LORA_R="${LORA_R:-16}"
ULD_WEIGHT="${ULD_WEIGHT:--0.8}"
ULD_TOPF="${ULD_TOPF:-0.01}"

TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_EP="${TRAIN_EP:-10}"

# ULD eval has KV cache disabled (base/assistant layer counts differ), so
# generation re-runs the full forward at every decode step. Default
# eval.tofu.batch_size=32 OOMs at ROUGE/extraction; 4 is safe on a 40GB A100.
EVAL_BS="${EVAL_BS:-4}"

MODELS_ROOT="${MODELS_ROOT:-${ULD_REPO}/outputs_trained_models/llama3_1b_uld}"
REPORT_PATH="${REPORT_PATH:-${ROOT}/uld_llama3_1b_results.md}"

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

if [ -n "${ONLY:-}" ]; then
    new_splits=()
    for sp in "${splits[@]}"; do
        case "$sp" in "$ONLY"*) new_splits+=("$sp");; esac
    done
    splits=("${new_splits[@]}")
fi

mkdir -p "$MODELS_ROOT"

echo "============================================================"
echo "ULD × open-unlearning, Llama-3.2-1B-Instruct"
echo "  ULD repo            : $ULD_REPO"
echo "  open-unlearning repo: $OU_REPO"
echo "  python              : $PY"
echo "  GPU                 : $GPU"
echo "  num_layer / lora_r  : $NUM_LAYER / $LORA_R"
echo "  ULD weight / topF   : $ULD_WEIGHT / $ULD_TOPF"
echo "  splits              : ${splits[*]}"
echo "  report              : $REPORT_PATH"
echo "============================================================"

#####################################
# Phase A — retain-model eval (reference for forget_quality)
#####################################
echo
if [ "${SKIP_PHASE_A:-0}" = "1" ]; then
    echo "[Phase A] SKIPPED (SKIP_PHASE_A=1) — forget_quality may be omitted in eval"
else
    echo "[Phase A] Reference TOFU_EVAL.json for each retain model"
    cd "$OU_REPO"
    need_download=0
    for sp in "${splits[@]}"; do
        retain=$(echo "$sp" | cut -d' ' -f3)
        out_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json"
        [ -f "$out_json" ] || need_download=1
    done
    if [ "$need_download" = "1" ] && [ "${FORCE_PHASE_A_RECOMPUTE:-0}" != "1" ]; then
        echo "  → Downloading precomputed retain eval logs (open-unlearning/eval)"
        "$PY" setup_data.py --eval_logs
    fi
    for sp in "${splits[@]}"; do
        forget=$(echo "$sp" | cut -d' ' -f1)
        holdout=$(echo "$sp" | cut -d' ' -f2)
        retain=$(echo "$sp" | cut -d' ' -f3)
        task="tofu_Llama-3.2-1B-Instruct_${retain}"
        out_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
        if [ -f "$out_json" ]; then
            echo "  → OK $retain: $out_json"
            continue
        fi
        echo "  → Recomputing $retain → $out_json"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/eval.py \
            experiment=eval/tofu/default \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_${retain}" \
            model.model_args.attn_implementation=sdpa \
            model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
            forget_split="$forget" \
            holdout_split="$holdout" \
            task_name="$task"
    done
fi

#####################################
# Phase B — train ULD assistants
#####################################
echo
echo "[Phase B] Train ULD assistants"
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
for sp in "${splits[@]}"; do
    forget=$(echo "$sp" | cut -d' ' -f1)
    out_dir="${MODELS_ROOT}/uld_assistant_${forget}"
    if find "$out_dir" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $forget: assistant already trained at $out_dir"
        continue
    fi
    echo "  → Train ULD assistant for $forget → $out_dir"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_1b_uld_${forget}" \
        data=tofu_chat3 \
        data.dataset.split="${forget}_perturbed" \
        data_mode=forget_more_retain_perturb \
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
        postfix=uld \
        "hydra.run.dir=outputs/tune_log/llama3_1b_uld_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
done

#####################################
# Phase C — eval ULD with open-unlearning metrics
#####################################
echo
echo "[Phase C] Eval ULD with open-unlearning TOFU metrics"
cd "$OU_REPO"
for sp in "${splits[@]}"; do
    forget=$(echo "$sp" | cut -d' ' -f1)
    holdout=$(echo "$sp" | cut -d' ' -f2)
    retain=$(echo "$sp" | cut -d' ' -f3)
    out_dir="${MODELS_ROOT}/uld_assistant_${forget}"

    ck=$(find "$out_dir" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$ck" ]; then
        echo "  ✗ ERROR: no checkpoint found in $out_dir"; exit 1
    fi

    task="tofu_Llama-3.2-1B-Instruct_${forget}_ULD"
    retain_json="${OU_REPO}/saves/eval/tofu_Llama-3.2-1B-Instruct_${retain}/TOFU_EVAL.json"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"

    if [ -f "$eval_json" ]; then
        echo "  → SKIP $forget: $eval_json exists"
        continue
    fi

    if [ -f "$retain_json" ]; then
        retain_arg="retain_logs_path=$retain_json"
    else
        retain_arg="retain_logs_path=null"
        echo "  ! retain_logs_path=null (forget_quality will be omitted)"
    fi

    echo "  → Eval ULD on $forget (assistant=$ck) → $eval_json"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.2-1B-Instruct_ULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.assistant_path="$ck" \
        model.model_args.weight="$ULD_WEIGHT" \
        model.model_args.top_logit_filter="$ULD_TOPF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split="$forget" \
        holdout_split="$holdout" \
        eval.tofu.batch_size="$EVAL_BS" \
        $retain_arg \
        task_name="$task"
done

#####################################
# Phase D — aggregate results into a single Markdown report
#####################################
echo
echo "[Phase D] Aggregating results → $REPORT_PATH"

"$PY" - "$REPORT_PATH" "$OU_REPO" "$ULD_WEIGHT" "$ULD_TOPF" "$NUM_LAYER" "$LORA_R" "${splits[@]}" <<'PYEOF'
import json, os, sys, datetime, pathlib

report_path = sys.argv[1]
ou_repo     = sys.argv[2]
weight      = sys.argv[3]
topf        = sys.argv[4]
num_layer   = sys.argv[5]
lora_r      = sys.argv[6]
splits_args = sys.argv[7:]

splits = []
for sp in splits_args:
    parts = sp.split()
    splits.append((parts[0], parts[2]))   # (forget, retain)

METRICS = [
    "forget_quality", "model_utility",
    "forget_truth_ratio", "forget_Q_A_Prob", "forget_Q_A_ROUGE",
    "privleak", "extraction_strength",
]

def read_metric(path, key):
    if not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except Exception as e:
        return f"ERR:{e}"
    v = d.get(key, None)
    if isinstance(v, dict) and "agg_value" in v:
        v = v["agg_value"]
    if isinstance(v, (int, float)):
        return v
    return None

rows_uld = {}
rows_retain = {}
for forget, retain in splits:
    uld_json    = f"{ou_repo}/saves/eval/tofu_Llama-3.2-1B-Instruct_{forget}_ULD/TOFU_EVAL.json"
    retain_json = f"{ou_repo}/saves/eval/tofu_Llama-3.2-1B-Instruct_{retain}/TOFU_EVAL.json"
    rows_uld[forget]    = {m: read_metric(uld_json, m) for m in METRICS}
    rows_retain[retain] = {m: read_metric(retain_json, m) for m in METRICS}

def fmt(v):
    if v is None: return "—"
    if isinstance(v, str): return v
    return f"{v:.4f}"

lines = []
lines.append(f"# ULD on Llama-3.2-1B-Instruct — open-unlearning TOFU evaluation")
lines.append("")
lines.append(f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
lines.append("")
lines.append("## Configuration")
lines.append("")
lines.append(f"- Base model: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`")
lines.append(f"- Assistant: {num_layer}-layer LoRA (r={lora_r}), trained 10 epochs, `remember+uniform` loss")
lines.append(f"- ULD logit-difference: `weight={weight}`, `top_logit_filter={topf}`")
lines.append(f"- Splits evaluated: {', '.join(f for f,_ in splits)}")
lines.append("")
lines.append("## ULD-unlearned model — TOFU metrics")
lines.append("")
header = "| split | " + " | ".join(METRICS) + " |"
sep    = "|" + "---|" * (len(METRICS) + 1)
lines.append(header)
lines.append(sep)
for forget, _ in splits:
    row = rows_uld[forget]
    lines.append("| " + forget + " | " + " | ".join(fmt(row[m]) for m in METRICS) + " |")
lines.append("")
lines.append("## Retain model (reference upper bound) — TOFU metrics")
lines.append("")
lines.append(header.replace("split", "retain"))
lines.append(sep)
for _, retain in splits:
    row = rows_retain[retain]
    lines.append("| " + retain + " | " + " | ".join(fmt(row[m]) for m in METRICS) + " |")
lines.append("")
lines.append("## How to read the metrics")
lines.append("")
lines.append("- **forget_quality**: KS-test p-value comparing forget-set truth-ratio dist. of unlearned model vs retain model. Higher (→1.0) = more retain-like = better forget.")
lines.append("- **model_utility**: harmonic mean across utility sub-metrics on retain/world-facts/real-authors subsets. Higher = better preserved utility.")
lines.append("- **forget_truth_ratio**: per-question ratio of perturbed-answer prob to true-answer prob on forget set. Closer to 1.0 = model treats truth and lies equally (good forget).")
lines.append("- **forget_Q_A_Prob / ROUGE**: probability / ROUGE of the true forget-set answer. Lower = stronger forget.")
lines.append("- **privleak**: MIA-based privacy-leak score. Negative magnitudes correspond to retain-like behavior; closer to retain-model value = better.")
lines.append("- **extraction_strength**: how easily the model regurgitates forget-set answers from prefixes. Lower = better.")
lines.append("")
lines.append("## Source files")
lines.append("")
for forget, retain in splits:
    lines.append(f"- forget `{forget}`: `saves/eval/tofu_Llama-3.2-1B-Instruct_{forget}_ULD/TOFU_EVAL.json`")
    lines.append(f"- retain `{retain}`: `saves/eval/tofu_Llama-3.2-1B-Instruct_{retain}/TOFU_EVAL.json`")
lines.append("")

pathlib.Path(report_path).write_text("\n".join(lines))
print(f"  → wrote {report_path} ({len(lines)} lines)")
PYEOF

echo
echo "============================================================"
echo "DONE — see $REPORT_PATH"
echo "============================================================"
