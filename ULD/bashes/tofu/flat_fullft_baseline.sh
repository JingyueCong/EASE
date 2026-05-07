#!/usr/bin/env bash
#
# FLAT baseline with FULL fine-tuning (matching paper setup).
# Uses DeepSpeed ZeRO-3 + CPU offload to fit 7B full FT on 40GB A100.
# Requires ~100GB RAM free for offload.
#
# Compared to flat_baseline.sh (LoRA):
#   - model_mode=base (no LoRA)
#   - trainer.strategy=deepspeed
#   - DEEPSPEED_CONFIG=configs/ds_config_offload.json (CPU offload enabled)
#
# Total cost: ~3 splits × (~25 min train + ~20 min eval) ≈ 2.3 hours
# (deepspeed CPU offload is ~3× slower than LoRA per step)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SPLITS="${SPLITS:-forget01 forget05 forget10}"
OUTPUTMODELDIR="${OUTPUTMODELDIR:-outputs_trained_models/tofu_flat_fullft}"
RESULTS_FILE="/tmp/flat_fullft_results.csv"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_config_offload.json}"

echo "split,FQ,MU,Forget_ROUGE,Retain_ROUGE,Real_Authors_ROUGE,Real_World_ROUGE,Forget_Proba,Retain_Proba" > "$RESULTS_FILE"

run_split() {
    local SPLIT=$1
    local TAG="flatfft_${SPLIT}"
    local TRAIN_OUT="${OUTPUTMODELDIR}/${TAG}"
    local BASELINE_PATH="data/${SPLIT}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

    if [ ! -f "$BASELINE_PATH" ]; then
        echo "MISSING retain baseline at $BASELINE_PATH — skip $SPLIT"
        echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"
        return
    fi

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] $SPLIT - Phase 1: training FLAT full-FT (deepspeed ZeRO-3 + CPU offload)"
    echo "============================================================"

    local EXISTING
    EXISTING=$(find "$TRAIN_OUT" -name "checkpoint-*" -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$EXISTING" ]; then
        # For deepspeed ZeRO-3, launch via torchrun (even single GPU)
        CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled \
        torchrun --nproc_per_node=1 --master_port=29518 \
            scripts/hf_forget_train.py \
            project="$TAG" \
            data=tofu \
            data.dataset.split=${SPLIT}_perturbed \
            data_mode=dpo \
            model=tofu-llama-2 \
            model_mode=base \
            unlearn_loss=flat \
            trainer.batch_size=2 \
            trainer.gradient_accumulation_steps=8 \
            trainer.learning_rate=1e-5 \
            trainer.max_epochs=5 \
            trainer.strategy=deepspeed \
            OUTPUTMODELDIR="$TRAIN_OUT" \
            postfix="flatfft" \
            "hydra.run.dir=outputs/tune_log/${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

        EXISTING=$(find "$TRAIN_OUT" -name "checkpoint-*" -type d \
            | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    [ -z "$EXISTING" ] && { echo "ERROR: no ckpt for $SPLIT"; echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"; return; }
    echo "  ckpt: $EXISTING"

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] $SPLIT - Phase 2: evaluating"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=${SPLIT}_perturbed \
        data.dataset.eval.retain_result="$BASELINE_PATH" \
        data.dataset.eval.batch_size=4 \
        model=tofu-llama-2 \
        model_mode=base \
        ckpt_path="$EXISTING" \
        OUTDIRNAME="outputs_eval/flat_fullft_baseline/$SPLIT" \
        "hydra.run.dir=outputs/tune_log/eval_${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

    local CSV
    CSV=$(find "outputs/tune_log/eval_${TAG}" -name "aggregate_stat.csv" | head -1)
    if [ -n "$CSV" ]; then
        awk -F',' -v S="$SPLIT" 'NR==2 {
            printf "%s,%.4g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   S, $14, $13, $10, $1, $4, $7, $11, $2
        }' "$CSV" >> "$RESULTS_FILE"
        awk -F',' 'NR==2 { printf "  → FQ=%.4g  MU=%.4f  Forget-RL=%.4f  Retain-RL=%.4f\n", $14, $13, $10, $1 }' "$CSV"
    else
        echo "$SPLIT,,,,,,,," >> "$RESULTS_FILE"
        echo "  → (no CSV)"
    fi
}

for SPLIT in $SPLITS; do
    run_split "$SPLIT"
done

echo
echo "============================================================"
echo "FLAT Full-FT baseline — final table"
echo "============================================================"
column -s, -t "$RESULTS_FILE"
