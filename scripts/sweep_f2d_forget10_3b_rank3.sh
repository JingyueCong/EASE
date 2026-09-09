#!/usr/bin/env bash
# Retain-free Llama-3.2-3B Forget10 capacity experiment.
# Train independent rank-16/32/64 4-layer Dual Assistants on the frozen
# causal/FullAnswer data, then evaluate four static reference-delta points.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
GPUS="${GPUS:-0 1 2 3}"
EVAL_BS="${EVAL_BS:-1}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
TARGET_AGG="${TARGET_AGG:-0.63}"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CAUSAL_DATA="${F2D_FORGET10_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget10_author_hybrid400_seed42_v512reused_v511new_incremental_manualfix_v1_full.jsonl}"
FULL_DATA="${FULLANSWER_FORGET10_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget10_ciru400_seed42_full_authorblock_incremental_v1.jsonl}"
CAUSAL_AUDIT="${F2D_FORGET10_AUDIT_SUMMARY:-${EASE_ROOT}/audits/forget10_author_hybrid400_seed42_v512reused_v511new_incremental_manualfix_v1_full/SUMMARY.json}"
HUMAN_APPROVAL="${F2D_FORGET10_HUMAN_APPROVAL:-${EASE_ROOT}/audits/forget10_author_hybrid400_seed42_v512reused_v511new_incremental_manualfix_v1_full/HUMAN_REVIEW_APPROVAL.md}"
RETAIN_RELATIVE="tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-${EASE_ROOT}/open-unlearning/saves/eval/${RETAIN_RELATIVE}}"
MODEL_ROOT="${MODEL_ROOT:-${EASE_ROOT}/ULD/outputs_trained_models/f2d_forget10_3b_rank3_seed42}"
RESULTS_DIR="${RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/forget10_f2d_3b_rank3_seed42}"
TRAIN_LOGS="$RESULTS_DIR/training_logs"
EVAL_LOGS="$RESULTS_DIR/eval_logs"
RUNNER="$EASE_ROOT/scripts/run_f2r_tofu.sh"
SUMMARIZER="$EASE_ROOT/scripts/summarize_f2r_sweep.py"
TRAIN_CONFIG="$EASE_ROOT/ULD/configs/model/llama-3-3b.yaml"
EVAL_CONFIG="$EASE_ROOT/open-unlearning/configs/model/Llama-3.2-3B-Instruct_DualULD.yaml"
TRAIN_PY="${TRAIN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"

checkpoint_at() {
    find "$1" -type d -name "checkpoint-$2" 2>/dev/null | sort | tail -n 1
}

reference_for() {
    (cd "$1/../fullmodel" 2>/dev/null && pwd)
}

for required in "$CAUSAL_DATA" "$FULL_DATA" "$CAUSAL_AUDIT" \
    "$HUMAN_APPROVAL" "$RUNNER" "$SUMMARIZER" "$TRAIN_CONFIG" \
    "$EVAL_CONFIG"; do
    [ -s "$required" ] || { echo "Missing Forget10 3B prerequisite: $required" >&2; exit 1; }
done
for executable in "$TRAIN_PY" "$EVAL_PY"; do
    [ -x "$executable" ] || { echo "Missing Python environment: $executable" >&2; exit 1; }
done

"$TRAIN_PY" - "$CAUSAL_DATA" "$FULL_DATA" "$CAUSAL_AUDIT" \
    "$HUMAN_APPROVAL" <<'PY'
import json
import sys

causal_path, full_path, audit_path, approval_path = sys.argv[1:]
expected = [f"forget10_perturbed-{index:05d}" for index in range(400)]
for label, path in (("causal", causal_path), ("fullanswer", full_path)):
    rows = [
        json.loads(line)
        for line in open(path, encoding="utf-8")
        if line.strip()
    ]
    ids = [row.get("source_id") for row in rows]
    if len(rows) != 400 or ids != expected or len(set(ids)) != 400:
        raise SystemExit(f"{label} coverage/order gate failed")
    if any(
        set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"}
        for row in rows
    ):
        raise SystemExit(f"{label} factorial-cell gate failed")

audit = json.load(open(audit_path, encoding="utf-8"))
if audit.get("records") != 400 or audit.get("unique_source_ids") != 400:
    raise SystemExit("Forget10 audit coverage gate failed")
if audit.get("units_with_deterministic_errors") != 0:
    raise SystemExit("Forget10 deterministic audit gate failed")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    raise SystemExit("Forget10 source-index audit gate failed")
if "APPROVED for downstream training" not in open(
    approval_path, encoding="utf-8"
).read():
    raise SystemExit("Forget10 human approval gate failed")
print(
    "forget10_3b_rank3_hard_gate rows=400 deterministic_errors=0 "
    "human_review=approved target_modules=all_linear"
)
PY

read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "Forget10 3B rank sweep requires four GPU ids; got: $GPUS" >&2
    exit 1
fi

cat <<EOF
============================================================
Forget10 Llama-3.2-3B static rank-capacity experiment
  immutable data : causal A1=400 / FullAnswer A2=400
  architecture   : A1=4 layers / A2=4 layers
  rank variants  : 16, 32, 64; alpha/r=2; all linear layers
  A1 training    : KL=.4, 192 steps, lr=5e-4
  A2 training    : KL=.3, 144 steps, lr=7.5e-4
  evaluation     : 4 static points per rank (12 total)
  composition    : role-specific reference_delta
  routing/gating : disabled
  retain policy  : absent from training; retain90 is selection-only
  target Agg     : $TARGET_AGG
  old artifacts  : read-only; all outputs are isolated
  model root     : $MODEL_ROOT
  result root    : $RESULTS_DIR
============================================================
EOF

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run OK; data, audit, approval, model configs, and rank grid passed."
    exit 0
fi

mkdir -p "$MODEL_ROOT" "$TRAIN_LOGS" "$EVAL_LOGS"

train_role() {
    local gpu="$1" rank="$2" role="$3"
    local data="$CAUSAL_DATA" steps=192 lr=5e-4 lambda=0.4
    if [ "$role" = "a2" ]; then
        data="$FULL_DATA"; steps=144; lr=7.5e-4; lambda=0.3
    fi
    local variant="rank${rank}_seed42"
    local root="$MODEL_ROOT/$variant"
    if [ -n "$(checkpoint_at "$root/$role" "$steps")" ]; then
        echo "rank3_train_reuse variant=$variant role=$role rank=$rank"
        return
    fi
    echo "rank3_train_start variant=$variant role=$role rank=$rank GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget10 \
        GPU="$gpu" CF_PATH="$data" MODELS_ROOT="$root" \
        TRAIN_RUN_TAG="forget10_3b_${variant}_${role}" \
        TRAIN_ONLY=true TRAIN_ROLE="$role" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=4 A1_LORA_R="$rank" A1_LORA_ALPHA="$((2 * rank))" \
        A1_TRAIN_LR=5e-4 A1_TRAIN_EP=1 A1_TRAIN_STEPS=192 \
        A1_RETAIN_WEIGHT=0.4 A1_TRAIN_BS=1 A1_TRAIN_GA=16 A1_SEED=42 \
        A2_NUM_LAYER=4 A2_LORA_R="$rank" A2_LORA_ALPHA="$((2 * rank))" \
        A2_TRAIN_LR=7.5e-4 A2_TRAIN_EP=1 A2_TRAIN_STEPS=144 \
        A2_RETAIN_WEIGHT=0.3 A2_TRAIN_BS=1 A2_TRAIN_GA=16 A2_SEED=42 \
        TRAIN_MODEL_CONFIG=llama-3-3b \
        EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD \
        HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.2-3B-Instruct \
        HF_MODEL_NAME=Llama-3.2-3B-Instruct \
        TRAIN_LOSS_CONFIG=factorial_reference_preserving \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false \
        SEQUENCE_ROUTER_ENABLED=false CALIBRATION_PATH=null HF_PREFLIGHT=0 \
        F2R_VARIANT=F2D-Forget10-3B-Rank3-NoRouter \
        bash "$RUNNER" > "$TRAIN_LOGS/${variant}_${role}.log" 2>&1
    echo "rank3_train_done variant=$variant role=$role rank=$rank GPU=$gpu"
}

wait_jobs() {
    local failures=0 pid
    for pid in "$@"; do
        wait "$pid" || failures=$((failures + 1))
    done
    if [ "$failures" -gt 0 ]; then
        echo "$failures training/evaluation job(s) failed; inspect $TRAIN_LOGS and $EVAL_LOGS" >&2
        exit 1
    fi
}

echo "[1/4] Train rank-16 and rank-32 assistant pairs"
train_role "${GPU_LIST[0]}" 16 a1 & p1=$!
train_role "${GPU_LIST[1]}" 16 a2 & p2=$!
train_role "${GPU_LIST[2]}" 32 a1 & p3=$!
train_role "${GPU_LIST[3]}" 32 a2 & p4=$!
wait_jobs "$p1" "$p2" "$p3" "$p4"

echo "[2/4] Train rank-64 assistant pair"
train_role "${GPU_LIST[0]}" 64 a1 & p1=$!
train_role "${GPU_LIST[1]}" 64 a2 & p2=$!
wait_jobs "$p1" "$p2"

echo "[3/4] Materialize retain90 reference and evaluate 12 static points"
if [ ! -s "$RETAIN_LOGS_PATH" ]; then
    "$EVAL_PY" - "$RETAIN_RELATIVE" \
        "$EASE_ROOT/open-unlearning/saves/eval" <<'PY'
import sys
from huggingface_hub import snapshot_download

relative_path, output_dir = sys.argv[1:]
snapshot_download(
    repo_id="open-unlearning/eval",
    repo_type="dataset",
    allow_patterns=[relative_path],
    local_dir=output_dir,
)
PY
fi
[ -s "$RETAIN_LOGS_PATH" ] \
    || { echo "Missing frozen retain90 reference: $RETAIN_LOGS_PATH" >&2; exit 1; }

declare -A A1_PATH A2_PATH A1_REF A2_REF
for rank in 16 32 64; do
    variant="rank${rank}_seed42"
    A1_PATH[$rank]="$(checkpoint_at "$MODEL_ROOT/$variant/a1" 192)"
    A2_PATH[$rank]="$(checkpoint_at "$MODEL_ROOT/$variant/a2" 144)"
    for checkpoint in "${A1_PATH[$rank]}" "${A2_PATH[$rank]}"; do
        [ -n "$checkpoint" ] && [ -d "$checkpoint" ] \
            || { echo "Missing completed rank-$rank checkpoint" >&2; exit 1; }
    done
    A1_REF[$rank]="$(reference_for "${A1_PATH[$rank]}")"
    A2_REF[$rank]="$(reference_for "${A2_PATH[$rank]}")"
    for reference in "${A1_REF[$rank]}" "${A2_REF[$rank]}"; do
        [ -d "$reference" ] \
            || { echo "Missing rank-$rank reference: $reference" >&2; exit 1; }
    done
done

MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,variant,lora_rank,a1_checkpoint,a2_checkpoint,composition_mode,selection_retain_access" > "$MANIFEST"
POINTS=(
    "light:-1.6:1.8"
    "balanced:-1.9:2.0"
    "utility:-2.1:2.2"
    "memory:-2.3:2.3"
)

run_eval() {
    local gpu="$1" rank="$2" point="$3" w1="$4" w2="$5" task="$6" report="$7"
    if [ "$RESUME" = true ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "rank3_eval_reuse rank=$rank point=$point"
        return
    fi
    echo "rank3_eval_start rank=$rank point=$point GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget10 \
        GPU="$gpu" CF_PATH="$CAUSAL_DATA" \
        MODELS_ROOT="$RESULTS_DIR/eval_only_unused" \
        A1_CHECKPOINT_OVERRIDE="${A1_PATH[$rank]}" \
        A2_CHECKPOINT_OVERRIDE="${A2_PATH[$rank]}" \
        A1_REFERENCE_PATH="${A1_REF[$rank]}" \
        A2_REFERENCE_PATH="${A2_REF[$rank]}" \
        REFERENCE_PATH=auto COMPOSITION_MODE=reference_delta \
        A1_NUM_LAYER=4 A2_NUM_LAYER=4 \
        A1_LORA_R="$rank" A2_LORA_R="$rank" \
        A1_TRAIN_STEPS=192 A2_TRAIN_STEPS=144 \
        A1_TRAIN_LR=5e-4 A2_TRAIN_LR=7.5e-4 \
        A1_RETAIN_WEIGHT=0.4 A2_RETAIN_WEIGHT=0.3 \
        TRAIN_LOSS_CONFIG=factorial_reference_preserving \
        TRAIN_MODEL_CONFIG=llama-3-3b \
        EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD \
        HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.2-3B-Instruct \
        HF_MODEL_NAME=Llama-3.2-3B-Instruct \
        TASK_MODEL_NAME=Llama-3.2-3B-Instruct \
        RETAIN_LOGS_PATH="$RETAIN_LOGS_PATH" AUTO_FETCH_RETAIN_LOGS=0 \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER=0.00017 \
        TASK_NAME="$task" EVAL_BS="$EVAL_BS" EVAL_OVERWRITE=true \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false \
        SEQUENCE_ROUTER_ENABLED=false CALIBRATION_PATH=null \
        SELECTION_RETAIN_ACCESS=true HF_PREFLIGHT=0 \
        F2R_VARIANT=F2D-Forget10-3B-Rank3-NoRouter \
        bash "$RUNNER" > "$EVAL_LOGS/rank${rank}_${point}.log" 2>&1
    echo "rank3_eval_done rank=$rank point=$point GPU=$gpu"
}

pids=(); failures=0; index=0
for rank in 16 32 64; do
    for point_spec in "${POINTS[@]}"; do
        IFS=: read -r point w1 w2 <<< "$point_spec"
        tag="rank${rank}_${point}"
        task="tofu_Llama-3.2-3B-Instruct_forget10_F2R_RANK3_${tag}"
        report="$EASE_ROOT/open-unlearning/saves/eval/$task/F2R_REPORT.json"
        echo "$tag,$w1,$w2,0.00017,$task,$report,rank_capacity,$rank,${A1_PATH[$rank]},${A2_PATH[$rank]},reference_delta,true" >> "$MANIFEST"
        gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
        run_eval "$gpu" "$rank" "$point" "$w1" "$w2" "$task" "$report" &
        pids+=("$!")
        index=$((index + 1))
        if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
            for pid in "${pids[@]}"; do
                wait "$pid" || failures=$((failures + 1))
            done
            pids=()
        fi
    done
done
if [ "${#pids[@]}" -gt 0 ]; then
    for pid in "${pids[@]}"; do
        wait "$pid" || failures=$((failures + 1))
    done
fi
if [ "$failures" -gt 0 ]; then
    echo "$failures rank evaluation job(s) failed; inspect $EVAL_LOGS" >&2
    exit 1
fi

echo "[4/4] Summarize Forget10 3B rank-capacity results"
"$EVAL_PY" "$SUMMARIZER" --manifest "$MANIFEST" \
    --output-dir "$RESULTS_DIR" --target-agg "$TARGET_AGG" \
    --target-margin 0 --sweep-kind training

"$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" "$TARGET_AGG" <<'PY'
import csv
import sys

rows = [
    row
    for row in csv.DictReader(open(sys.argv[1], encoding="utf-8"))
    if row.get("aggregate_score")
]
if len(rows) != 12:
    raise SystemExit(f"expected 12 reports, found {len(rows)}")
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
best = rows[0]
agg = float(best["aggregate_score"])
target = float(sys.argv[2])
print("===== Forget10 Llama-3.2-3B rank-capacity best =====")
for key in (
    "tag", "aggregate_score", "memorization_score",
    "retain_utility_score", "model_utility", "weight_a1",
    "weight_a2", "top_filter", "report",
):
    print(f"{key} = {best.get(key)}")
print(f"distance to {target:.2f}:", f"{agg - target:+.6f}")
PY
echo "Forget10 3B rank experiment complete: $RESULTS_DIR/F2R_SWEEP.md"
