#!/usr/bin/env bash
# Retain-free 2x2 pilot: train and evaluate one merged causal checkpoint.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$EASE_ROOT/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
TRAIN_PY="${TRAIN_PY:-$CONDA_BASE/envs/ease-f2r-train/bin/python}"
EVAL_PY="${EVAL_PY:-$CONDA_BASE/envs/ease-f2r-eval/bin/python}"
HF_BASE="${HF_BASE:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
HF_TOKENIZER="${HF_TOKENIZER:-$HF_BASE}"
DATA="${F2D_V512_DATA_PATH:-$EASE_ROOT/ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}"
AUDIT="${F2D_V512_AUDIT_SUMMARY:-$EASE_ROOT/audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}"
RETAIN_REFERENCE="${RETAIN_LOGS_PATH:-$EASE_ROOT/open-unlearning/saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json}"
SWEEP_NAME="${SWEEP_NAME:-f2d_v512_single_causal_npo4_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/$SWEEP_NAME"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_$SWEEP_NAME"
TRAIN_STEPS="${TRAIN_STEPS:-60}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
LOCALITY_WEIGHT="${LOCALITY_WEIGHT:-0.1}"
LOCALITY_MARGIN="${LOCALITY_MARGIN:-0.05}"

for executable in "$TRAIN_PY" "$EVAL_PY"; do
    if [ ! -x "$executable" ]; then
        echo "Missing Python environment: $executable" >&2
        exit 1
    fi
done
for required in "$DATA" "$AUDIT" "$RETAIN_REFERENCE"; do
    if [ ! -s "$required" ]; then
        echo "Missing single-causal prerequisite: $required" >&2
        exit 1
    fi
done

# This is an experiment-level hard gate, not a naming convention.  The only
# training rows are the four V5.12 cells; retain95 is report-only after every
# candidate and all hyperparameters have already been frozen.
if [ "${WITH_RETAIN:-false}" != "false" ] \
    || [ "${SELECTION_RETAIN_ACCESS:-false}" != "false" ]; then
    echo "Retain-free hard gate failed: WITH_RETAIN and SELECTION_RETAIN_ACCESS must be false" >&2
    exit 1
fi

"$EVAL_PY" - "$DATA" "$AUDIT" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
audit = json.load(open(sys.argv[2]))
errors = []
if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
    errors.append("V5.12 must contain 200 unique rows")
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in rows):
    errors.append("unexpected design_version")
if audit.get("units_with_deterministic_errors") != 0:
    errors.append("deterministic audit errors are nonzero")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    errors.append("audit source coverage is incomplete")
if errors:
    raise SystemExit("Single-causal preflight failed: " + "; ".join(errors))
print("Single-causal retain-free hard gate OK: rows=200 cells=800 retain_access=false")
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "Single-causal 2x2 requires four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

CONFIGS=(
    "b0p1_k0p5:0.1:0.5"
    "b0p1_k1p0:0.1:1.0"
    "b0p2_k0p5:0.2:0.5"
    "b0p2_k1p0:0.2:1.0"
)

mkdir -p "$MODELS_ROOT" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,npo_beta,control_kl_weight,locality_weight,locality_margin,models_root" > "$MANIFEST"

cat <<EOF
============================================================
V5.12 retain-free single-checkpoint causal NPO pilot
  data              : $DATA
  target             : NPO(C11)
  controls           : base-KL(C01,C10,C00)
  locality           : C11 drift > matched-control drift
  beta               : 0.1 0.2
  control KL weight  : 0.5 1.0
  locality weight    : $LOCALITY_WEIGHT
  locality margin    : $LOCALITY_MARGIN
  optimization       : steps=$TRAIN_STEPS lr=$TRAIN_LR LoRA=full-model/r16
  final artifact     : one merged Llama checkpoint per configuration
  retain train data  : none
  retain selection   : none (four configurations are preregistered)
  retain evaluation  : report-only after checkpoint freeze
  GPUs               : $LAUNCH_GPUS
============================================================
EOF

latest_checkpoint() {
    find "$1" -name 'checkpoint-*' -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_one() {
    local gpu="$1" tag="$2" beta="$3" control_kl="$4"
    local train_root="$MODELS_ROOT/$tag/train"
    local merged="$MODELS_ROOT/$tag/merged-checkpoint"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_SINGLE_CAUSAL_${tag}"
    local eval_dir="$EASE_ROOT/open-unlearning/saves/eval/$task"
    local report="$eval_dir/F2R_REPORT.json"
    echo "$tag,,,,${task},${report},${beta},${control_kl},${LOCALITY_WEIGHT},${LOCALITY_MARGIN},${MODELS_ROOT}/$tag" >> "$MANIFEST"

    if [ "$LAUNCH_DRY_RUN" = "true" ]; then
        echo "[dry-run] $tag GPU=$gpu beta=$beta control_kl=$control_kl"
        return
    fi
    if [ "$LAUNCH_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"aggregate_score"' "$report"; then
        echo "single_causal_reuse tag=$tag"
        return
    fi

    local adapter
    adapter="$(latest_checkpoint "$train_root")"
    if [ -z "$adapter" ]; then
        echo "single_causal_train_start tag=$tag GPU=$gpu beta=$beta control_kl=$control_kl"
        (
            cd "$EASE_ROOT/ULD"
            CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled \
            PYTHONPATH="$EASE_ROOT/ULD:${PYTHONPATH:-}" \
            "$TRAIN_PY" scripts/hf_forget_train.py \
                project=f2d_single_causal_forget05 \
                data=tofu_chat3 \
                data.dataset.split=forget05_perturbed \
                data_mode=f2d_single_causal \
                data_mode.counterfactual_path="$DATA" \
                model=llama-3-1b \
                model.model_path="$HF_BASE" \
                model.tokenizer_path="$HF_TOKENIZER" \
                model_mode=uld \
                model_mode.num_layer=0 \
                model_mode.Lora.r=16 \
                model_mode.Lora.alpha=32 \
                model_mode.Lora.dropout=0.05 \
                unlearn_loss=factorial_causal_npo \
                unlearn_loss.beta="$beta" \
                unlearn_loss.control_kl_weight="$control_kl" \
                unlearn_loss.locality_weight="$LOCALITY_WEIGHT" \
                unlearn_loss.locality_margin="$LOCALITY_MARGIN" \
                trainer.batch_size=4 \
                trainer.gradient_accumulation_steps=4 \
                trainer.learning_rate="$TRAIN_LR" \
                trainer.optim=adamw_torch \
                trainer.max_epochs=1 \
                trainer.max_steps="$TRAIN_STEPS" \
                +trainer.save_steps="$TRAIN_STEPS" \
                +trainer.eval_steps="$TRAIN_STEPS" \
                trainer.seed=42 seed=42 trainer.strategy=gpu \
                OUTPUTMODELDIR="$train_root" postfix=single \
                "hydra.run.dir=outputs/tune_log/f2d_single_causal_${tag}/\${now:%Y-%m-%d_%H-%M-%S}"
        )
        adapter="$(latest_checkpoint "$train_root")"
        if [ -z "$adapter" ]; then
            echo "No adapter checkpoint produced for $tag" >&2
            return 1
        fi
    else
        echo "single_causal_train_reuse tag=$tag adapter=$adapter"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN_PY" \
        "$EASE_ROOT/scripts/merge_f2d_single_checkpoint.py" \
        --base-model "$HF_BASE" --adapter "$adapter" \
        --output "$merged" --tokenizer "$HF_TOKENIZER"

    echo "single_causal_eval_start tag=$tag GPU=$gpu"
    (
        cd "$EASE_ROOT/open-unlearning"
        CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$EVAL_PY" src/eval.py \
            experiment=eval/tofu/default \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path="$merged" \
            model.model_args.attn_implementation=sdpa \
            model.tokenizer_args.pretrained_model_name_or_path="$merged" \
            forget_split=forget05 holdout_split=holdout05 \
            eval.tofu.batch_size="$LAUNCH_EVAL_BS" eval.tofu.overwrite=true \
            retain_logs_path="$RETAIN_REFERENCE" task_name="$task"
    )
    "$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_tofu.py" \
        --eval-json "$eval_dir/TOFU_EVAL.json" \
        --summary-json "$eval_dir/TOFU_SUMMARY.json" \
        --output-dir "$eval_dir" --mode full --split forget05 \
        --base-model "$HF_BASE" --a1-checkpoint "$merged" \
        --a2-checkpoint not-applicable --a1-data-mode f2d_single_causal \
        --a2-data-mode not-applicable --retain-reference "$RETAIN_REFERENCE" \
        --variant V5.12-SingleCausalNPO --views 1 \
        --a1-num-layer 0 --a2-num-layer 0 --a1-lora-r 16 --a2-lora-r 0 \
        --a1-lora-alpha 32 --a2-lora-alpha 0 \
        --a1-train-lr "$TRAIN_LR" --a1-train-ep 1 \
        --a1-train-steps "$TRAIN_STEPS" --a1-seed 42 \
        --training-loss factorial_causal_npo \
        --npo-beta "$beta" --control-kl-weight "$control_kl" \
        --locality-weight "$LOCALITY_WEIGHT" \
        --locality-margin "$LOCALITY_MARGIN" \
        --selection-retain-access false
    echo "single_causal_done tag=$tag GPU=$gpu"
}

pids=()
failures=0
for index in 0 1 2 3; do
    IFS=: read -r tag beta control_kl <<< "${CONFIGS[$index]}"
    run_one "${GPU_LIST[$index]}" "$tag" "$beta" "$control_kl" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failures=$((failures + 1))
    fi
done

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run complete: four retain-free single-model configurations validated."
    exit 0
fi

"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.0 --sweep-kind training \
    || failures=$((failures + 1))

echo "Single-causal table: $RESULTS_DIR/F2R_SWEEP.md"
echo "IMPORTANT: retain metrics above are report-only; no follow-up point may be chosen from them under the strict retain-free claim."
if [ "$failures" -gt 0 ]; then
    echo "$failures single-causal job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
