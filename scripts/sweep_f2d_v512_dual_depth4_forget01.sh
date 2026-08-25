#!/usr/bin/env bash
# Retain-free forget01 replication of the successful static 4-layer/4-layer
# Dual Assistant.  Every artifact uses a forget01-specific versioned path;
# no forget05 data, checkpoint, log, or sweep is modified.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-2}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

absolute_from_root() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$EASE_ROOT" "$1" ;;
    esac
}

latest_checkpoint() {
    find "$1" -type d -name 'checkpoint-*' 2>/dev/null \
        | awk -F'checkpoint-' '{print $NF, $0}' \
        | sort -n | tail -n 1 | cut -d' ' -f2-
}

reference_for() {
    local checkpoint="$1"
    (cd "$checkpoint/../fullmodel" 2>/dev/null && pwd)
}

TRAIN_PY="${TRAIN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
EVAL_PY="${PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
SOURCE_V512="$(absolute_from_root "${F2D_V512_FORGET05_SOURCE:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
SOURCE_FULL="$(absolute_from_root "${FULLANSWER_FORGET05_SOURCE:-ULD/data/ciru/forget05_ciru200_seed42_full_authorblock_v1.jsonl}")"
V512_DATA="$(absolute_from_root "${F2D_V512_DATA_PATH:-ULD/data/ciru/forget01_author_pairbudget40_seed42_v5_12_exact_subset_v1.jsonl}")"
FULL_DATA="$(absolute_from_root "${FULLANSWER_DATA_PATH:-ULD/data/ciru/forget01_ciru40_seed42_full_authorblock_exact_subset_v1.jsonl}")"
DERIVATION_MANIFEST="$(absolute_from_root "${F2D_FORGET01_DERIVATION_MANIFEST:-ULD/data/ciru/forget01_v512_full_exact_subset_v1.derivation.json}")"
V512_AUDIT="$(absolute_from_root "${F2D_V512_AUDIT_DIR:-audits/forget01_author_pairbudget40_seed42_v5_12_exact_subset_v1}")"
FULL_AUDIT="$(absolute_from_root "${FULLANSWER_AUDIT_DIR:-audits/forget01_fullanswer40_seed42_exact_subset_v1}")"
SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth4_exact_subset_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_1b_forget01_${SWEEP_NAME}"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget01_${SWEEP_NAME}"
TARGET_AGG="${TARGET_AGG:-0.57}"

for required in "$SOURCE_V512" "$SOURCE_FULL"; do
    if [ ! -s "$required" ]; then
        echo "Missing immutable forget05 source artifact: $required" >&2
        exit 1
    fi
done
for executable in "$TRAIN_PY" "$EVAL_PY"; do
    if [ ! -x "$executable" ]; then
        echo "Missing Python environment: $executable" >&2
        exit 1
    fi
done

reference_args=()
if [ -n "${FORGET01_REFERENCE_JSONL:-}" ]; then
    reference_args=(--reference-jsonl "$(absolute_from_root "$FORGET01_REFERENCE_JSONL")")
fi

echo "[0/4] Verify exact C11 subset and derive immutable forget01 copies"
"$TRAIN_PY" "$EASE_ROOT/scripts/derive_f2d_forget01_from_forget05.py" \
    --v512-source "$SOURCE_V512" \
    --fullanswer-source "$SOURCE_FULL" \
    --v512-output "$V512_DATA" \
    --fullanswer-output "$FULL_DATA" \
    --manifest-output "$DERIVATION_MANIFEST" \
    "${reference_args[@]}"

mkdir -p "$V512_AUDIT" "$FULL_AUDIT"
"$TRAIN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$V512_DATA" --output-dir "$V512_AUDIT" \
    --expected-units 40 --block-size 20 --sample-count 20 --seed 42 \
    --fail-on-deterministic-errors
"$TRAIN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$FULL_DATA" --output-dir "$FULL_AUDIT" \
    --expected-units 40 --block-size 20 --sample-count 20 --seed 42 \
    --fail-on-deterministic-errors

"$TRAIN_PY" - "$V512_DATA" "$FULL_DATA" "$DERIVATION_MANIFEST" <<'PY'
import json
import sys

for label, path in (("V5.12", sys.argv[1]), ("FullAnswer", sys.argv[2])):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ids = [row.get("source_id") for row in rows]
    if len(rows) != 40 or len(set(ids)) != 40:
        raise SystemExit(f"{label}: expected 40 unique forget01 rows")
    if ids != [f"forget01_perturbed-{index:05d}" for index in range(40)]:
        raise SystemExit(f"{label}: non-canonical forget01 ordering or coverage")
    if any(set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"} for row in rows):
        raise SystemExit(f"{label}: incomplete factorial cells")
manifest = json.load(open(sys.argv[3], encoding="utf-8"))
if not manifest.get("c11_exact_match") or manifest.get("content_cells_changed") is not False:
    raise SystemExit("forget01 derivation provenance gate failed")
print("Forget01 training gate OK: rows=40 blocks=2 C11=exact inherited_audit=approved")
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "Forget01 depth-4 sweep requires four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

cat <<EOF
============================================================
Forget01 static Dual Assistant 4/4 replication
  causal A1 data    : $V512_DATA
  FullAnswer A2 data: $FULL_DATA
  source policy     : exact uniquely matched subset of frozen forget05 artifacts
  old artifacts     : read-only; no overwrite
  architecture      : A1=4 layers / A2=4 layers / LoRA rank=16
  training pairs    : exposure-matched 20/14; moderate 32/24 steps
  composition       : shared depth-4 reference_delta
  routing/gating    : disabled
  inference points  : 4 per training pair (8 total)
  target Agg        : $TARGET_AGG (forget01 BS-S target)
  GPUs              : $LAUNCH_GPUS
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run OK; derivation, coverage, audit, and unique output paths validated."
    exit 0
fi

mkdir -p "$MODELS_ROOT" "$RESULTS_DIR/logs"

# Scaling 96/72 steps by 40/200 gives approximately 20/14 updates at the
# same per-example exposure.  The moderate pair tests a deliberately wider
# small-data budget without reusing either checkpoint.
TRAININGS=(
    "exposure:20:14"
    "moderate:32:24"
)

train_role() {
    local gpu="$1" tag="$2" role="$3" a1_steps="$4" a2_steps="$5"
    local data="$V512_DATA"
    if [ "$role" = "a2" ]; then data="$FULL_DATA"; fi
    echo "forget01_train_start training=$tag role=$role GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget01 \
        GPU="$gpu" CF_PATH="$data" \
        MODELS_ROOT="$MODELS_ROOT/$tag/${role}_job" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}_${role}" \
        TRAIN_ONLY=true TRAIN_ROLE="$role" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=4 A1_LORA_R=16 A1_LORA_ALPHA=32 \
        A1_TRAIN_LR=5e-4 A1_TRAIN_EP=1 A1_TRAIN_STEPS="$a1_steps" \
        A1_RETAIN_WEIGHT=1.5 A1_TRAIN_BS=2 A1_TRAIN_GA=8 A1_SEED=42 \
        A2_NUM_LAYER=4 A2_LORA_R=16 A2_LORA_ALPHA=32 \
        A2_TRAIN_LR=1e-3 A2_TRAIN_EP=1 A2_TRAIN_STEPS="$a2_steps" \
        A2_RETAIN_WEIGHT=1.0 A2_TRAIN_BS=2 A2_TRAIN_GA=8 A2_SEED=42 \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        F2R_VARIANT="F2D-Forget01-V512-DualDepth4-${role}" \
        HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/train_${tag}_${role}.log" 2>&1
    echo "forget01_train_done training=$tag role=$role GPU=$gpu"
}

echo "[1/4] Train two preregistered 4/4 budgets (four role checkpoints)"
pids=()
index=0
for spec in "${TRAININGS[@]}"; do
    IFS=: read -r tag a1_steps a2_steps <<< "$spec"
    for role in a1 a2; do
        gpu="${GPU_LIST[$index]}"
        train_role "$gpu" "$tag" "$role" "$a1_steps" "$a2_steps" &
        pids+=("$!")
        index=$((index + 1))
    done
done
failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
if [ "$failures" -gt 0 ]; then
    echo "$failures forget01 training job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi

MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,training,a1_steps,a2_steps,a1_checkpoint,a2_checkpoint,reference_a1_path,reference_a2_path,composition_mode" > "$MANIFEST"
POINTS=(
    "light:-1.1:1.0:0.0004"
    "conservative:-1.4:1.2:0.0004"
    "intermediate:-1.7:1.4:0.0004"
    "legacy_best:-2.0:1.7:0.0004"
)

run_eval() {
    local gpu="$1" training="$2" a1_steps="$3" a2_steps="$4"
    local a1="$5" a2="$6" reference="$7" point="$8"
    local w1="$9" w2="${10}" filter="${11}"
    local tag="${training}_${point}"
    local task="tofu_Llama-3.2-1B-Instruct_forget01_F2R_V512_DUAL_DEPTH4_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/$task/F2R_REPORT.json"
    echo "$tag,$w1,$w2,$filter,$task,$report,$training,$a1_steps,$a2_steps,$a1,$a2,$reference,$reference,reference_delta" >> "$MANIFEST"
    if [ "$LAUNCH_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "forget01_eval_reuse training=$training point=$point"
        return
    fi
    echo "forget01_eval_start training=$training point=$point GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget01 GPU="$gpu" \
        CF_PATH="$V512_DATA" MODELS_ROOT="$RESULTS_DIR/frozen_${training}" \
        A1_CHECKPOINT_OVERRIDE="$a1" A2_CHECKPOINT_OVERRIDE="$a2" \
        A1_REFERENCE_PATH="$reference" A2_REFERENCE_PATH="$reference" \
        REFERENCE_PATH="$reference" COMPOSITION_MODE=reference_delta \
        A1_NUM_LAYER=4 A2_NUM_LAYER=4 A1_TRAIN_STEPS="$a1_steps" A2_TRAIN_STEPS="$a2_steps" \
        A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER="$filter" \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false SEQUENCE_ROUTER_ENABLED=false \
        CALIBRATION_PATH=null F2R_VARIANT=F2D-Forget01-V512-DualDepth4-Static-NoRouter \
        SELECTION_RETAIN_ACCESS=true HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "forget01_eval_done training=$training point=$point GPU=$gpu"
}

echo "[2/4] Resolve depth-4 checkpoints and references"
declare -A TRAIN_A1 TRAIN_A2 TRAIN_REFERENCE TRAIN_A1_STEPS TRAIN_A2_STEPS
for spec in "${TRAININGS[@]}"; do
    IFS=: read -r training a1_steps a2_steps <<< "$spec"
    a1="$(latest_checkpoint "$MODELS_ROOT/$training/a1_job/a1")"
    a2="$(latest_checkpoint "$MODELS_ROOT/$training/a2_job/a2")"
    for checkpoint in "$a1" "$a2"; do
        if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
            echo "Missing forget01 depth-4 checkpoint: ${checkpoint:-unresolved}" >&2
            exit 1
        fi
    done
    reference="$(reference_for "$a1")"
    if [ -z "$reference" ] || [ ! -d "$reference" ]; then
        echo "Missing forget01 depth-4 base reference for $training" >&2
        exit 1
    fi
    TRAIN_A1["$training"]="$a1"
    TRAIN_A2["$training"]="$a2"
    TRAIN_REFERENCE["$training"]="$reference"
    TRAIN_A1_STEPS["$training"]="$a1_steps"
    TRAIN_A2_STEPS["$training"]="$a2_steps"
done

echo "[3/4] Evaluate eight static 4/4 configurations"
pids=()
index=0
wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failures=$((failures + 1))
    done
    pids=()
}
for spec in "${TRAININGS[@]}"; do
    IFS=: read -r training _ _ <<< "$spec"
    # Keep Hydra checkpoint paths in associative arrays.  Those paths contain
    # literal ':' and '|' characters and must never be delimiter-serialized.
    a1_steps="${TRAIN_A1_STEPS[$training]}"
    a2_steps="${TRAIN_A2_STEPS[$training]}"
    a1="${TRAIN_A1[$training]}"
    a2="${TRAIN_A2[$training]}"
    reference="${TRAIN_REFERENCE[$training]}"
    for point_spec in "${POINTS[@]}"; do
        IFS=: read -r point w1 w2 filter <<< "$point_spec"
        gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
        run_eval "$gpu" "$training" "$a1_steps" "$a2_steps" \
            "$a1" "$a2" "$reference" "$point" "$w1" "$w2" "$filter" &
        pids+=("$!")
        index=$((index + 1))
        if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then wait_batch; fi
    done
done
if [ "${#pids[@]}" -gt 0 ]; then wait_batch; fi

echo "[4/4] Summarize forget01 static depth-4 results"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg "$TARGET_AGG" --target-margin 0.005 --sweep-kind training \
    || failures=$((failures + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

rows = [row for row in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if row.get("aggregate_score")]
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
if rows:
    best = rows[0]
    agg = float(best["aggregate_score"])
    print("===== Forget01 static 4/4 best =====")
    print("config =", best["tag"])
    print("Agg    =", f"{agg:.6f}")
    print("Mem    =", best["memorization_score"])
    print("Util   =", best["retain_utility_score"])
    print("relative to forget01 BS-S 0.570000:", f"{agg - 0.57:+.6f}")
    print("relative to global 0.580000 target:", f"{agg - 0.58:+.6f}")
    print("report =", best["report"])
PY
fi
echo "Forget01 table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$failures" -gt 0 ]; then
    echo "$failures forget01 job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
