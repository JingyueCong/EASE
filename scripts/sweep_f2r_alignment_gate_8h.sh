#!/usr/bin/env bash
# Four-GPU, resumable F2R-AG calibration sweep with an approximately eight-hour
# launch budget. One independent configuration is assigned to each GPU.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
if [ "${LOAD_DOTENV:-1}" = "1" ] && [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODE="${MODE:-full}"
SPLIT="${SPLIT:-forget05}"
GPUS="${GPUS:-0 1 2 3}"
SWEEP_NAME="${SWEEP_NAME:-alignment_gate_8h}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
MAX_HOURS="${MAX_HOURS:-8}"
MIN_JOB_MINUTES="${MIN_JOB_MINUTES:-45}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-14000}"
WEIGHT_A1="${WEIGHT_A1:--1.2}"
WEIGHT_A2="${WEIGHT_A2:-0.4}"
TOP_FILTER="${TOP_FILTER:-0.0025}"
TARGET_AGG="${TARGET_AGG:-0.58}"
TARGET_MARGIN="${TARGET_MARGIN:-0.005}"
EVAL_BS="${EVAL_BS:-4}"
CALIBRATION_BS="${CALIBRATION_BS:-2}"
VIEWS="${VIEWS:-2}"

CF_PATH="${CF_PATH:-${EASE_ROOT}/ULD/data/f2r/${SPLIT}_${MODE}.jsonl}"
MODELS_ROOT="${MODELS_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_1b_${SPLIT}_${MODE}_train_stage1_${SPLIT}/lr1e3_e5}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2r_calibration/${SPLIT}_${SWEEP_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/${SPLIT}_${SWEEP_NAME}}"
RUNNER="${EASE_ROOT}/scripts/run_f2r_tofu.sh"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] && [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
CONDA_BASE="${CONDA_BASE:-$(${CONDA_BIN:-false} info --base 2>/dev/null || true)}"
EVAL_PY="${EVAL_PY:-${CONDA_BASE}/envs/${EVAL_ENV:-ease-f2r-eval}/bin/python}"

latest_checkpoint() {
    find "$1" -name 'checkpoint-*' -type d 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

if [ -z "${A1_CKPT:-}" ]; then
    A1_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a1" || true)"
fi
if [ -z "${A2_CKPT:-}" ]; then
    A2_CKPT="$(latest_checkpoint "${MODELS_ROOT}/a2" || true)"
fi

if [ "$DRY_RUN" != "true" ]; then
    for path in "$CF_PATH" "$A1_CKPT" "$A2_CKPT"; do
        if [ ! -e "$path" ]; then
            echo "Missing required input: $path" >&2
            exit 1
        fi
    done
    if [ ! -x "$EVAL_PY" ]; then
        echo "Missing evaluation Python: $EVAL_PY" >&2
        exit 1
    fi
fi

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU index" >&2
    exit 1
fi

mkdir -p "$CALIBRATION_ROOT" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,calibration_kind,alignment_ridge,alignment_scale_max,alignment_min_observations,gate_l2,gate_steps,gate_learning_rate,calibration_path,alignment_input" > "$MANIFEST"

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH=$((START_EPOCH + MAX_HOURS * 3600))
BUDGET_EXHAUSTED=false
FAILURES=0

budget_allows_job() {
    if [ "$MAX_HOURS" -le 0 ]; then return 0; fi
    now="$(date +%s)"
    remaining=$((DEADLINE_EPOCH - now))
    [ "$remaining" -ge $((MIN_JOB_MINUTES * 60)) ]
}

wait_for_gpu() {
    local gpu="$1"
    if [ "$DRY_RUN" = "true" ]; then return; fi
    while true; do
        free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        if [ "$free_mib" -ge "$MIN_FREE_GPU_MIB" ]; then return; fi
        echo "[$(date '+%H:%M:%S')] waiting GPU $gpu: ${free_mib}/${MIN_FREE_GPU_MIB} MiB free"
        sleep 60
    done
}

report_complete() {
    [ -s "$1" ] && grep -q '"forget_truth_ratio_knowledge"' "$1"
}

run_configuration() {
    local gpu="$1" kind="$2" tag="$3" ridge="$4" scale_max="$5"
    local min_obs="$6" gate_l2="$7" gate_steps="$8" gate_lr="$9"
    local alignment_input="${10}" artifact="${11}" task="${12}" report="${13}"

    if [ "$RESUME" = "true" ] && report_complete "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag"
        return
    fi
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] GPU=$gpu kind=$kind tag=$tag ridge=$ridge scale=$scale_max obs=$min_obs l2=$gate_l2 steps=$gate_steps lr=$gate_lr input=$alignment_input"
        return
    fi

    wait_for_gpu "$gpu"
    echo "[$(date '+%H:%M:%S')] start calibration $tag on GPU $gpu"
    if [ ! -s "$artifact" ] || [ ! -s "${artifact%.npz}.json" ] || [ "$RESUME" != "true" ]; then
        calibration_mode="$kind"
        train_args=(
            --mode "$calibration_mode"
            --counterfactual-path "$CF_PATH"
            --base-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full
            --tokenizer open-unlearning/tofu_Llama-3.2-1B-Instruct_full
            --a1-path "$A1_CKPT" --a2-path "$A2_CKPT"
            --weight-a1 "$WEIGHT_A1" --weight-a2 "$WEIGHT_A2"
            --top-filter "$TOP_FILTER" --batch-size "$CALIBRATION_BS"
            --output "$artifact"
        )
        if [ "$kind" = "alignment" ]; then
            train_args+=(
                --alignment-ridge "$ridge"
                --alignment-scale-max "$scale_max"
                --alignment-min-observations "$min_obs"
            )
        else
            train_args+=(
                --gate-l2 "$gate_l2"
                --gate-steps "$gate_steps"
                --gate-learning-rate "$gate_lr"
            )
        fi
        if [ "$kind" = "alignment-gate" ]; then
            train_args+=(--alignment-input "$alignment_input")
        fi
        CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$EVAL_PY" "$EASE_ROOT/scripts/train_f2r_calibration.py" "${train_args[@]}"
    else
        echo "[$(date '+%H:%M:%S')] reuse calibration artifact $artifact"
    fi

    alignment=false
    gate=false
    if [ "$kind" = "alignment" ] || [ "$kind" = "alignment-gate" ]; then alignment=true; fi
    if [ "$kind" = "gate" ] || [ "$kind" = "alignment-gate" ]; then gate=true; fi

    wait_for_gpu "$gpu"
    echo "[$(date '+%H:%M:%S')] start evaluation $tag on GPU $gpu"
    MODE="$MODE" SPLIT="$SPLIT" GPU="$gpu" VIEWS="$VIEWS" \
        CF_PATH="$CF_PATH" MODELS_ROOT="$MODELS_ROOT" TASK_NAME="$task" \
        A1_NUM_LAYER=2 A2_NUM_LAYER=2 A1_LORA_R=16 A2_LORA_R=16 \
        A1_LORA_ALPHA=32 A2_LORA_ALPHA=32 \
        A1_TRAIN_LR=1e-3 A2_TRAIN_LR=1e-3 A1_TRAIN_EP=5 A2_TRAIN_EP=5 \
        A1_RETAIN_WEIGHT=5 A2_RETAIN_WEIGHT=5 A1_SEED=42 A2_SEED=42 \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        F2R_VARIANT="$tag" CALIBRATION_PATH="$artifact" \
        ALIGNMENT_ENABLED="$alignment" GATE_ENABLED="$gate" \
        EVAL_BS="$EVAL_BS" EVAL_OVERWRITE=true HF_PREFLIGHT=0 \
        SELECTION_RETAIN_ACCESS=true \
        bash "$RUNNER"
    echo "[$(date '+%H:%M:%S')] done $tag on GPU $gpu"
}

pids=()
pid_tags=()
wait_batch() {
    local index pid tag
    for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        tag="${pid_tags[$index]}"
        if wait "$pid"; then
            echo "[$(date '+%H:%M:%S')] completed $tag"
        else
            echo "[$(date '+%H:%M:%S')] FAILED $tag (see $RESULTS_DIR/logs/$tag.log)" >&2
            FAILURES=$((FAILURES + 1))
        fi
    done
    pids=()
    pid_tags=()
}

INDEX=0
launch_configuration() {
    local kind="$1" tag="$2" ridge="$3" scale_max="$4" min_obs="$5"
    local gate_l2="$6" gate_steps="$7" gate_lr="$8" alignment_input="$9"
    if ! budget_allows_job; then
        echo "[$(date '+%H:%M:%S')] launch budget exhausted; no new jobs will start"
        BUDGET_EXHAUSTED=true
        return
    fi
    gpu="${GPU_LIST[$((INDEX % ${#GPU_LIST[@]}))]}"
    artifact="$CALIBRATION_ROOT/${tag}.npz"
    task="tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_calibration_${SWEEP_NAME}_${tag}"
    report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report,$kind,$ridge,$scale_max,$min_obs,$gate_l2,$gate_steps,$gate_lr,$artifact,$alignment_input" >> "$MANIFEST"
    echo "[$(date '+%H:%M:%S')] queue $tag on GPU $gpu"
    run_configuration "$gpu" "$kind" "$tag" "$ridge" "$scale_max" "$min_obs" \
        "$gate_l2" "$gate_steps" "$gate_lr" "$alignment_input" \
        "$artifact" "$task" "$report" > "$RESULTS_DIR/logs/${tag}.log" 2>&1 &
    pids+=("$!")
    pid_tags+=("$tag")
    INDEX=$((INDEX + 1))
    if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then wait_batch; fi
}

finish_phase() {
    if [ "${#pids[@]}" -gt 0 ]; then wait_batch; fi
}

# Measured on the target 4x4090 server, one calibration+evaluation takes
# roughly 8--11 minutes.  This 192-candidate pool therefore fills an eight-hour
# four-GPU window; MAX_HOURS remains the hard launch budget if throughput varies.
ALIGNMENT_CONFIGS=()
for ridge_spec in "r0p1:0.1" "r1:1" "r10:10" "r100:100"; do
    IFS=: read -r ridge_tag ridge <<< "$ridge_spec"
    for scale_max in 2 3 4; do
        for min_obs in 4 8; do
            ALIGNMENT_CONFIGS+=(
                "align_${ridge_tag}_s${scale_max}_o${min_obs}:${ridge}:${scale_max}:${min_obs}"
            )
        done
    done
done

GATE_CONFIGS=()
for l2_spec in "l1em5:1e-5" "l1em4:1e-4" "l1em3:1e-3" "l1em2:1e-2"; do
    IFS=: read -r l2_tag gate_l2 <<< "$l2_spec"
    for gate_steps in 400 800 1200; do
        for lr_spec in "lr1em2:0.01" "lr3em2:0.03"; do
            IFS=: read -r lr_tag gate_lr <<< "$lr_spec"
            GATE_CONFIGS+=(
                "gate_${l2_tag}_n${gate_steps}_${lr_tag}:${gate_l2}:${gate_steps}:${gate_lr}"
            )
        done
    done
done

COMBINATION_ALIGNMENTS=()
for ridge_tag in r1 r10 r100; do
    for scale_max in 2 4; do
        for min_obs in 4 8; do
            COMBINATION_ALIGNMENTS+=(
                "align_${ridge_tag}_s${scale_max}_o${min_obs}"
            )
        done
    done
done

COMBINATION_GATES=()
for l2_spec in "l1em4:1e-4" "l1em3:1e-3" "l1em2:1e-2"; do
    IFS=: read -r l2_tag gate_l2 <<< "$l2_spec"
    for gate_steps in 400 800; do
        for lr_spec in "lr1em2:0.01" "lr3em2:0.03"; do
            IFS=: read -r lr_tag gate_lr <<< "$lr_spec"
            COMBINATION_GATES+=(
                "gate_${l2_tag}_n${gate_steps}_${lr_tag}:${gate_l2}:${gate_steps}:${gate_lr}"
            )
        done
    done
done

COMBINATION_COUNT=$((${#COMBINATION_ALIGNMENTS[@]} * ${#COMBINATION_GATES[@]}))
PLANNED_COUNT=$((${#ALIGNMENT_CONFIGS[@]} + ${#GATE_CONFIGS[@]} + COMBINATION_COUNT))

# Include the completed fixed-parameter anchor when the preceding method ladder
# has already produced it. It costs no additional GPU time.
BASELINE_TASK="tofu_Llama-3.2-1B-Instruct_${SPLIT}_F2R_ladder_alignment_gate_F2R"
BASELINE_REPORT="${EASE_ROOT}/open-unlearning/saves/eval/${BASELINE_TASK}/F2R_REPORT.json"
if report_complete "$BASELINE_REPORT"; then
    echo "F2R_baseline,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$BASELINE_TASK,$BASELINE_REPORT,baseline,,,,,,,," >> "$MANIFEST"
fi

echo "============================================================"
echo "F2R-AG four-GPU calibration sweep"
echo "  split/mode       : $SPLIT / $MODE"
echo "  GPUs             : $GPUS"
echo "  launch budget    : $MAX_HOURS hours (no new job in final $MIN_JOB_MINUTES min)"
echo "  planned configs  : ${#ALIGNMENT_CONFIGS[@]} alignment + ${#GATE_CONFIGS[@]} gate + $COMBINATION_COUNT combined = $PLANNED_COUNT"
echo "  operating point  : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER"
echo "  frozen assistants: $MODELS_ROOT"
echo "  results          : $RESULTS_DIR"
echo "  protocol         : calibration retain=false; selection retain=true"
echo "============================================================"

echo "[$(date '+%H:%M:%S')] phase 1/3: alignment"
for config in "${ALIGNMENT_CONFIGS[@]}"; do
    IFS=: read -r tag ridge scale_max min_obs <<< "$config"
    launch_configuration alignment "$tag" "$ridge" "$scale_max" "$min_obs" "" "" "" ""
    if [ "$BUDGET_EXHAUSTED" = "true" ]; then break; fi
done
finish_phase

if [ "$BUDGET_EXHAUSTED" = "false" ]; then
    echo "[$(date '+%H:%M:%S')] phase 2/3: gate"
    for config in "${GATE_CONFIGS[@]}"; do
        IFS=: read -r tag gate_l2 gate_steps gate_lr <<< "$config"
        launch_configuration gate "$tag" "" "" "" "$gate_l2" "$gate_steps" "$gate_lr" ""
        if [ "$BUDGET_EXHAUSTED" = "true" ]; then break; fi
    done
    finish_phase
fi

if [ "$BUDGET_EXHAUSTED" = "false" ]; then
    echo "[$(date '+%H:%M:%S')] phase 3/3: alignment + gate"
    for alignment_tag in "${COMBINATION_ALIGNMENTS[@]}"; do
        alignment_input="$CALIBRATION_ROOT/${alignment_tag}.npz"
        for gate_config in "${COMBINATION_GATES[@]}"; do
            IFS=: read -r gate_tag gate_l2 gate_steps gate_lr <<< "$gate_config"
            tag="combo_${alignment_tag#align_}_${gate_tag#gate_}"
            launch_configuration alignment-gate "$tag" "" "" "" \
                "$gate_l2" "$gate_steps" "$gate_lr" "$alignment_input"
            if [ "$BUDGET_EXHAUSTED" = "true" ]; then break 2; fi
        done
    done
    finish_phase
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run complete. Manifest: $MANIFEST"
    exit 0
fi

"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin "$TARGET_MARGIN" \
    --sweep-kind calibration || FAILURES=$((FAILURES + 1))

echo "Sweep table: $RESULTS_DIR/F2R_SWEEP.md"
echo "Elapsed minutes: $((($(date +%s) - START_EPOCH) / 60))"
echo "Budget exhausted: $BUDGET_EXHAUSTED"
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES job/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
