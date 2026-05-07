#!/usr/bin/env bash
#
# Re-eval best Dual-ULD ckpts with GENTLER inference weights to boost
# Retain-RL / Forget-RL (closer to retrain-only).
#
# Keeps existing A1/A2 checkpoints — only changes inference weight.
#
# Tests 3 weight configs per split × 2 splits = 6 evals.
# Total cost ≈ 6 × ~18 min = ~1.8 hours.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

GPU="${GPU:-0}"

# forget10 ckpts (from FQ=0.6536 run)
F10_A1="outputs_trained_models/tofu_dual_y/dual_uld_a1_y/2026-04-20_03-04-28/logs/dual_uld_a1_y/dataset:tofu|loss:remember+uniform|model:tofu-llama-2|datamode:dual_a1/2026-04-20T03-04-28a1y/checkpoint-1350"
F10_A2="outputs_trained_models/tofu_dual/dual_uld_a2/2026-04-20_01-00-00/logs/dual_uld_a2/dataset:tofu|loss:remember+uniform|model:tofu-llama-2|datamode:dual_a2/2026-04-20T01-00-01a2/checkpoint-900"

# forget05 ckpts (from step=700 FQ=0.7126 run)
F05_A1="outputs_trained_models/tofu_dual_f05_a1longrun/dual_f05_a1longrun/2026-04-21_04-31-04/logs/dual_f05_a1longrun/dataset:tofu|loss:remember+uniform|model:tofu-llama-2|datamode:dual_a1/2026-04-21T04-31-04a1lr/checkpoint-700"
F05_A2="outputs_trained_models/tofu_dual_f05a2ep3/dual_f05a2ep3/2026-04-20_14-28-05/logs/dual_f05a2ep3/dataset:tofu|loss:remember+uniform|model:tofu-llama-2|datamode:dual_a2/2026-04-20T14-28-05a2/checkpoint-300"

RESULTS_FILE="/tmp/weight_softer_results.csv"
echo "split,w1,w2,FQ,MU,Forget_RL,Retain_RL,Real_Authors_RL,Real_World_RL,Forget_Proba,Retain_Proba" > "$RESULTS_FILE"

run_eval() {
    local SPLIT=$1 A1=$2 A2=$3 W1=$4 W2=$5
    local TAG="softer_${SPLIT}_w1${W1//./p}_w2${W2//./p}"
    TAG="${TAG//-/m}"

    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] eval $SPLIT w1=$W1 w2=$W2"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled python scripts/eval_tofu.py \
        data=tofu \
        data.dataset.split=${SPLIT}_perturbed \
        data.dataset.eval.retain_result="data/${SPLIT}_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json" \
        data.dataset.eval.batch_size=4 \
        model=tofu-llama-2 \
        model_mode=dual_uld \
        model_mode.weight_a1=$W1 \
        model_mode.weight_a2=$W2 \
        model_mode.top_logit_filter=1e-2 \
        model_mode.a1_ckpt_path="$A1" \
        model_mode.a2_ckpt_path="$A2" \
        ckpt_path="$A1" \
        OUTDIRNAME="outputs_eval/softer/$TAG" \
        "hydra.run.dir=outputs/tune_log/${TAG}/\${now:%Y-%m-%d_%H-%M-%S}"

    local CSV
    CSV=$(find "outputs/tune_log/${TAG}" -name "aggregate_stat.csv" | head -1)
    if [ -n "$CSV" ]; then
        awk -F',' -v S="$SPLIT" -v W1="$W1" -v W2="$W2" 'NR==2 {
            printf "%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   S, W1, W2, $14, $13, $10, $1, $4, $7, $11, $2
        }' "$CSV" >> "$RESULTS_FILE"
        awk -F',' 'NR==2 { printf "  → FQ=%.4f  MU=%.4f  Forget-RL=%.4f  Retain-RL=%.4f\n", $14, $13, $10, $1 }' "$CSV"
    else
        echo "$SPLIT,$W1,$W2,,,,,,,," >> "$RESULTS_FILE"
        echo "  → (no CSV)"
    fi
}

# --- forget10 sweep ---
run_eval forget10 "$F10_A1" "$F10_A2" -0.6 0.6
run_eval forget10 "$F10_A1" "$F10_A2" -0.7 0.7
run_eval forget10 "$F10_A1" "$F10_A2" -0.6 0.8

# --- forget05 sweep ---
run_eval forget05 "$F05_A1" "$F05_A2" -0.6 0.6
run_eval forget05 "$F05_A1" "$F05_A2" -0.7 0.7
run_eval forget05 "$F05_A1" "$F05_A2" -0.6 0.8

echo
echo "============================================================"
echo "All done. Results:"
echo "============================================================"
column -s, -t "$RESULTS_FILE"
