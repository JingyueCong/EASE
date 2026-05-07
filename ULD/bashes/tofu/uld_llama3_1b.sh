#!/usr/bin/env bash
#
# ULD baseline (original single-assistant) on Llama-3.2-1B-Instruct.
#
# Phases (all are idempotent — skip if output exists):
#   0. Fine-tune Llama-3 1B on full TOFU          → base model the ULD paper assumes
#   1. Fine-tune Llama-3 1B on retain-only subset → retrain-LLM for FQ baseline
#   1b. Run eval on retain-only model             → creates the retain_result JSON
#   2. Train ULD assistant (4-layer, LoRA)
#   3. Evaluate the unlearned model
#
# Prerequisites:
#   - huggingface-cli login  (to access meta-llama/Llama-3.2-1B-Instruct, gated)
#   - ULD env ready: `pip install -e .` inside this repo
#   - ≥ 20 GB free disk for checkpoints
#
# Usage:
#   bash bashes/tofu/uld_llama3_1b.sh
#   GPU=0 SPLIT=forget10 bash bashes/tofu/uld_llama3_1b.sh
#
# Total cost on single A100 40GB: ~3-4 hours per split.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export USE_TF=0   # avoid Keras-3 incompat with transformers >= 4.45
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SPLIT="${SPLIT:-forget10}"                                   # forget01 | forget05 | forget10
BASE_HF_MODEL="${BASE_HF_MODEL:-meta-llama/Llama-3.2-1B}"
NUM_LAYER="${NUM_LAYER:-4}"                                  # assistant = 4 layers out of Llama-3 1B's 16
LORA_R="${LORA_R:-16}"
MODELS_ROOT="${MODELS_ROOT:-outputs_trained_models/llama3_1b}"

# derived: retain split name matching the forget split
RAW_FORGET_PCT=${SPLIT#forget}
RETAIN_SPLIT="retain$((100 - RAW_FORGET_PCT))"
RETAIN_SPLIT=$(printf "retain%02d" $((100 - RAW_FORGET_PCT)))

BASE_FULL_TUNED="${MODELS_ROOT}/tofu_fullft_base"
RETAIN_ONLY_TUNED="${MODELS_ROOT}/retain_only_${RETAIN_SPLIT}"
RETAIN_EVAL_JSON="${MODELS_ROOT}/retain_eval_${SPLIT}/eval_log_aggregated.json"
ULD_OUT="${MODELS_ROOT}/uld_assistant_${SPLIT}"

echo "============================================================"
echo "Llama-3 1B ULD baseline on $SPLIT"
echo "  BASE_HF_MODEL : $BASE_HF_MODEL"
echo "  RETAIN_SPLIT  : $RETAIN_SPLIT"
echo "  NUM_LAYER     : $NUM_LAYER"
echo "============================================================"

# helper: full FT (no LoRA) a Llama-3 model on a TOFU subset
finetune_base() {
    local TARGET_SPLIT=$1         # e.g. "full" (all 4000) or "retain90"
    local OUT_DIR=$2
    local DATA_SPLIT=$3           # what to pass to data.dataset.split

    if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "  → SKIP ${TARGET_SPLIT} fine-tune: already exists at $OUT_DIR"
        return
    fi

    echo "  → Training base on $TARGET_SPLIT …"
    # For full-TOFU base we extend forget split with all retain (3600); for
    # retain-only we just use the retain split as forget-role data.
    local DM_OVERRIDES=""
    if [ "$TARGET_SPLIT" = "full" ]; then
        DM_OVERRIDES="data_mode=forget_retain data_mode.retain_num=3600 +data_mode.retain_num_no_clamp=true"
    else
        DM_OVERRIDES="data_mode=forget"
    fi

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
        project="llama3_1b_ft_${TARGET_SPLIT}" \
        data=tofu \
        data.dataset.split=${DATA_SPLIT} \
        $DM_OVERRIDES \
        model=llama-3-1b \
        model_mode=base \
        unlearn_loss=gd+gd \
        trainer.batch_size=8 \
        trainer.gradient_accumulation_steps=2 \
        trainer.learning_rate=5e-5 \
        trainer.max_epochs=20 \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$OUT_DIR" \
        postfix="ft" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_ft_${TARGET_SPLIT}/\${now:%Y-%m-%d_%H-%M-%S}"

    # `model_mode=base` with Lora.r=0 produces a full-model ckpt, not LoRA. The last
    # checkpoint directory is the thing we point to next.
    local LAST_CKPT
    LAST_CKPT=$(find "$OUT_DIR" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    echo "    final ckpt: $LAST_CKPT"
}

############################
# Phase 0 – full-TOFU base
############################
echo; echo "[Phase 0/3] Full-TOFU fine-tune as base"
# Use full dataset: data.dataset.split=full trains on ALL 4000 (forget+retain).
# Our ToFU_DataModule doesn't natively expose "full", so we piggyback by using
# the forget10 split with with_retain=True retain_num=3600 (everything).
finetune_base full "$BASE_FULL_TUNED" "forget10_perturbed"
BASE_CKPT=$(find "$BASE_FULL_TUNED" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -z "$BASE_CKPT" ] && { echo "ERROR: base full-FT checkpoint not produced"; exit 1; }

############################
# Phase 1 – retain-only fine-tune for FQ baseline
############################
echo; echo "[Phase 1/3] Retain-only fine-tune ($RETAIN_SPLIT)"
finetune_base "$RETAIN_SPLIT" "$RETAIN_ONLY_TUNED" "$RETAIN_SPLIT"
RETAIN_CKPT=$(find "$RETAIN_ONLY_TUNED" -name "checkpoint-*" -type d \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)

# Phase 1b: evaluate retain-only LLM on the forget set (produces the JSON ULD eval wants)
if [ ! -f "$RETAIN_EVAL_JSON" ]; then
    echo "  → Running retain-only eval to produce baseline JSON …"
    mkdir -p "$(dirname "$RETAIN_EVAL_JSON")"
    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=${SPLIT}_perturbed \
        data.dataset.eval.batch_size=4 \
        model=llama-3-1b \
        model_mode=base \
        ckpt_path="$RETAIN_CKPT" \
        OUTDIRNAME="outputs_eval/llama3_1b_retain_eval_${SPLIT}" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_retain_eval_${SPLIT}/\${now:%Y-%m-%d_%H-%M-%S}"

    # eval_tofu.py emits eval_log_forget.json + retain_perturbed.json + ... but no
    # ULD-style eval_log_aggregated.json. Aggregate them ourselves.
    EVAL_DIR=$(ls -td outputs/tune_log/llama3_1b_retain_eval_${SPLIT}/* 2>/dev/null | head -1)
    if [ -n "$EVAL_DIR" ]; then
        python3 - <<EOF
import json, os
src = "$EVAL_DIR"
out = {}
mapping = {
    "eval_log_forget.json":          "eval_log_forget.json",
    "retain_perturbed.json":         "eval_log.json",
    "real_authors_perturbed.json":   "eval_real_author_wo_options.json",
    "world_facts_perturbed.json":    "eval_real_world_wo_options.json",
}
for fname, key in mapping.items():
    fp = os.path.join(src, fname)
    if os.path.isfile(fp):
        out[key] = json.load(open(fp))
target = "$RETAIN_EVAL_JSON"
os.makedirs(os.path.dirname(target), exist_ok=True)
json.dump(out, open(target, "w"))
print("aggregated to", target, "keys:", list(out.keys()))
EOF
    fi
fi
[ ! -f "$RETAIN_EVAL_JSON" ] && { echo "ERROR: retain eval JSON missing: $RETAIN_EVAL_JSON"; exit 1; }

############################
# Phase 2 – ULD assistant (original, single-assistant)
############################
echo; echo "[Phase 2/3] Train ULD assistant (num_layer=$NUM_LAYER)"
if [ ! -d "$ULD_OUT" ] || [ -z "$(find "$ULD_OUT" -name 'checkpoint-*' -type d 2>/dev/null)" ]; then
    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/hf_forget_train.py \
        project="llama3_1b_uld_${SPLIT}" \
        data=tofu \
        data.dataset.split=${SPLIT}_perturbed \
        data_mode=forget_more_retain_perturb \
        model=llama-3-1b \
        model.model_path="$BASE_CKPT" \
        model.tokenizer_path="$BASE_HF_MODEL" \
        model_mode=uld \
        model_mode.num_layer=$NUM_LAYER \
        model_mode.Lora.r=$LORA_R \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight=5.0 \
        trainer.batch_size=8 \
        trainer.gradient_accumulation_steps=2 \
        trainer.learning_rate=1e-3 \
        trainer.max_epochs=10 \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$ULD_OUT" \
        postfix="uld" \
        "hydra.run.dir=outputs/tune_log/llama3_1b_uld_${SPLIT}/\${now:%Y-%m-%d_%H-%M-%S}"
else
    echo "  → SKIP ULD train: already exists"
fi

A1_CKPT=$(find "$ULD_OUT" -name "checkpoint-*" -type d | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -z "$A1_CKPT" ] && { echo "ERROR: ULD assistant not produced"; exit 1; }
echo "  → ULD assistant: $A1_CKPT"

############################
# Phase 3 – final evaluation
############################
echo; echo "[Phase 3/3] Evaluate Llama-3 1B ULD on $SPLIT"
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
    data=tofu \
    data.dataset.split=${SPLIT}_perturbed \
    data.dataset.eval.retain_result="$RETAIN_EVAL_JSON" \
    data.dataset.eval.batch_size=4 \
    model=llama-3-1b \
    model.model_path="$BASE_CKPT" \
    model.tokenizer_path="$BASE_HF_MODEL" \
    model_mode=uld \
    model_mode.num_layer=$NUM_LAYER \
    model_mode.weight=-0.8 \
    model_mode.top_logit_filter=1e-2 \
    ckpt_path="$A1_CKPT" \
    OUTDIRNAME="outputs_eval/llama3_1b_uld_${SPLIT}" \
    "hydra.run.dir=outputs/tune_log/llama3_1b_uld_eval_${SPLIT}/\${now:%Y-%m-%d_%H-%M-%S}"

CSV=$(find "outputs/tune_log/llama3_1b_uld_eval_${SPLIT}" -name "aggregate_stat.csv" | head -1)
if [ -n "$CSV" ]; then
    echo
    echo "============================================================"
    echo "FINAL  —  Llama-3 1B ULD on $SPLIT"
    echo "============================================================"
    awk -F',' 'NR==2 {
        printf "FQ         = %.4g\n", $14
        printf "MU         = %.4f\n", $13
        printf "Forget R-L = %.4f\n", $10
        printf "Retain R-L = %.4f\n", $1
        printf "Real_Authors R-L = %.4f\n", $4
        printf "Real_World   R-L = %.4f\n", $7
        printf "Forget P   = %.4f\n", $11
        printf "Retain P   = %.4f\n", $2
    }' "$CSV"
    echo "CSV:  $CSV"
fi
