#!/usr/bin/env bash
# Static Dual Assistant capacity ablation: train one 4-layer V5.12 A1 and one
# 4-layer FullAnswer A2, then compare 4/2, 2/4, and 4/4 without routing.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-2}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_V512_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_FULL_DATA="${FULLANSWER_DATA_PATH:-}"
LAUNCH_FULL_MANIFEST="${FULLANSWER_MANIFEST:-}"
LAUNCH_V512_ROOT="${F2D_V512_STRONG_A1_ROOT:-}"

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

checkpoint_exact() {
    local root="$1" role="$2" step="$3"
    find "$root/$role" -type d -name "checkpoint-${step}" 2>/dev/null \
        | sort | tail -n 1
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

PY="${PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
V512_DATA="$(absolute_from_root "${LAUNCH_V512_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
FULL_DATA="$(absolute_from_root "${LAUNCH_FULL_DATA:-ULD/data/ciru/forget05_ciru200_seed42_full_authorblock_v1.jsonl}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_FULL_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
V512_TAG="${F2D_V512_STRONG_A1_TAG:-v512_a1s96_u1p5}"
V512_A1_STEP="${F2D_V512_STRONG_A1_STEP:-96}"
V512_ROOT="$(absolute_from_root "${LAUNCH_V512_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_a1strength4_train_seed42/${V512_TAG}}")"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_A2_STEP="${FULLANSWER_STEP:-72}"
SWEEP_NAME="${SWEEP_NAME:-f2d_v512_dual_depth4_seed42}"
MODELS_ROOT="$EASE_ROOT/ULD/outputs_trained_models/f2d_did_1b_forget05_${SWEEP_NAME}"
RESULTS_DIR="$EASE_ROOT/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"

for required in "$V512_DATA" "$FULL_DATA" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing depth-4 prerequisite: $required" >&2
        exit 1
    fi
done
if [ ! -x "$PY" ]; then
    echo "Missing evaluation Python: $PY" >&2
    exit 1
fi

"$PY" - "$V512_DATA" "$FULL_DATA" <<'PY'
import json
import sys

for name, path in (("V5.12", sys.argv[1]), ("FullAnswer", sys.argv[2])):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ids = [row.get("source_id") for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200:
        raise SystemExit(f"{name} depth-4 preflight requires 200 unique rows")
    if any(set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"} for row in rows):
        raise SystemExit(f"{name} depth-4 preflight found incomplete factorial cells")
print("Dual depth-4 data preflight OK: V5.12=200 FullAnswer=200")
PY

FULL_ROOT="$(awk -F, -v tag="$FULL_TAG" '$1==tag {gsub(/\r/, "", $NF); print $NF}' "$FULL_MANIFEST" | tail -n 1)"
if [ -z "$FULL_ROOT" ]; then
    echo "Could not resolve FullAnswer tag $FULL_TAG from $FULL_MANIFEST" >&2
    exit 1
fi
FULL_ROOT="$(absolute_from_root "$FULL_ROOT")"
OLD_A1="$(checkpoint_exact "$V512_ROOT" a1 "$V512_A1_STEP")"
OLD_A2="$(checkpoint_exact "$FULL_ROOT" a2 "$FULL_A2_STEP")"
for checkpoint in "$OLD_A1" "$OLD_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing frozen 2-layer baseline checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "Depth-4 capacity experiment requires four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

cat <<EOF
============================================================
Static Dual Assistant depth-4 capacity ablation
  causal A1 data    : $V512_DATA
  FullAnswer A2 data: $FULL_DATA
  old baseline      : A1=2 layers / A2=2 layers, Agg=0.551721
  new training      : A1=4 layers, A2=4 layers, LoRA rank=16
  A1 settings       : steps=96, uniform=1.5, lr=5e-4
  A2 settings       : steps=72, uniform=1.0, lr=1e-3
  evaluated pairs   : 4/2, 2/4, 4/4
  composition       : role-specific reference_delta
  routing/gating    : disabled
  inference points  : 4 per pair (12 total)
  GPUs              : $LAUNCH_GPUS
  eval batch size   : $LAUNCH_EVAL_BS
  resume            : $LAUNCH_RESUME
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run OK; data, frozen checkpoints, and depth-4 design were validated."
    exit 0
fi

mkdir -p "$MODELS_ROOT" "$RESULTS_DIR/logs"

train_a1() {
    echo "depth4_train_start role=A1 GPU=${GPU_LIST[0]}"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 \
        GPU="${GPU_LIST[0]}" CF_PATH="$V512_DATA" \
        MODELS_ROOT="$MODELS_ROOT/a1_depth4" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_a1_depth4" TRAIN_ONLY=true \
        A2_CHECKPOINT_OVERRIDE="$OLD_A2" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=4 A1_LORA_R=16 A1_LORA_ALPHA=32 \
        A1_TRAIN_LR=5e-4 A1_TRAIN_EP=1 A1_TRAIN_STEPS=96 \
        A1_RETAIN_WEIGHT=1.5 A1_TRAIN_BS=2 A1_TRAIN_GA=8 A1_SEED=42 \
        A2_NUM_LAYER=2 A2_TRAIN_STEPS=72 A2_RETAIN_WEIGHT=1.0 \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        F2R_VARIANT=F2D-V512-DualDepth4-A1 \
        HF_PREFLIGHT=0 bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/train_a1_depth4.log" 2>&1
    echo "depth4_train_done role=A1 GPU=${GPU_LIST[0]}"
}

train_a2() {
    echo "depth4_train_start role=A2 GPU=${GPU_LIST[1]}"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 \
        GPU="${GPU_LIST[1]}" CF_PATH="$FULL_DATA" \
        MODELS_ROOT="$MODELS_ROOT/a2_depth4" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_a2_depth4" TRAIN_ONLY=true \
        A1_CHECKPOINT_OVERRIDE="$OLD_A1" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_NUM_LAYER=2 A1_TRAIN_STEPS=96 A1_RETAIN_WEIGHT=1.5 \
        A2_NUM_LAYER=4 A2_LORA_R=16 A2_LORA_ALPHA=32 \
        A2_TRAIN_LR=1e-3 A2_TRAIN_EP=1 A2_TRAIN_STEPS=72 \
        A2_RETAIN_WEIGHT=1.0 A2_TRAIN_BS=2 A2_TRAIN_GA=8 A2_SEED=42 \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH=auto \
        F2R_VARIANT=F2D-FullAnswer-DualDepth4-A2 \
        HF_PREFLIGHT=0 bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/train_a2_depth4.log" 2>&1
    echo "depth4_train_done role=A2 GPU=${GPU_LIST[1]}"
}

echo "[1/3] Train 4-layer A1 and A2 concurrently"
train_a1 & a1_pid=$!
train_a2 & a2_pid=$!
failures=0
wait "$a1_pid" || failures=$((failures + 1))
wait "$a2_pid" || failures=$((failures + 1))
if [ "$failures" -gt 0 ]; then
    echo "$failures depth-4 training job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi

NEW_A1="$(latest_checkpoint "$MODELS_ROOT/a1_depth4/a1")"
NEW_A2="$(latest_checkpoint "$MODELS_ROOT/a2_depth4/a2")"
for checkpoint in "$NEW_A1" "$NEW_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Depth-4 training completed without expected checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

OLD_A1_REF="$(reference_for "$OLD_A1")"
OLD_A2_REF="$(reference_for "$OLD_A2")"
NEW_A1_REF="$(reference_for "$NEW_A1")"
NEW_A2_REF="$(reference_for "$NEW_A2")"

MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,pair,a1_layers,a2_layers,a1_checkpoint,a2_checkpoint,reference_a1_path,reference_a2_path,composition_mode" > "$MANIFEST"

POINTS=(
    "conservative:-1.4:1.2:0.0004"
    "intermediate:-1.7:1.4:0.0004"
    "legacy_best:-2.0:1.7:0.0004"
    "strong:-2.2:1.9:0.0004"
)
PAIRS=(d4_2 d2_4 d4_4)

run_eval() {
    local gpu="$1" pair="$2" a1_layers="$3" a2_layers="$4"
    local a1="$5" a2="$6" a1_ref="$7" a2_ref="$8"
    local point="$9" w1="${10}" w2="${11}" filter="${12}"
    local tag="${pair}_${point}"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_DUAL_DEPTH4_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/$task/F2R_REPORT.json"
    echo "$tag,$w1,$w2,$filter,$task,$report,$pair,$a1_layers,$a2_layers,$a1,$a2,$a1_ref,$a2_ref,reference_delta" >> "$MANIFEST"
    if [ "$LAUNCH_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "depth4_eval_reuse pair=$pair point=$point"
        return
    fi
    echo "depth4_eval_start pair=$pair point=$point GPU=$gpu"
    env ENV_FILE=/dev/null LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        CF_PATH="$V512_DATA" MODELS_ROOT="$RESULTS_DIR/frozen_${pair}" \
        A1_CHECKPOINT_OVERRIDE="$a1" A2_CHECKPOINT_OVERRIDE="$a2" \
        A1_REFERENCE_PATH="$a1_ref" A2_REFERENCE_PATH="$a2_ref" \
        REFERENCE_PATH=null COMPOSITION_MODE=reference_delta \
        A1_NUM_LAYER="$a1_layers" A2_NUM_LAYER="$a2_layers" \
        A1_TRAIN_STEPS=96 A2_TRAIN_STEPS=72 \
        A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER="$filter" \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false \
        SEQUENCE_ROUTER_ENABLED=false CALIBRATION_PATH=null \
        F2R_VARIANT=F2D-V512-DualDepth4-Static-NoRouter \
        SELECTION_RETAIN_ACCESS=true HF_PREFLIGHT=0 \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "depth4_eval_done pair=$pair point=$point GPU=$gpu"
}

echo "[2/3] Evaluate 4/2, 2/4, and 4/4 with depth-matched references"
pids=()
failures=0
index=0
wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failures=$((failures + 1))
    done
    pids=()
}
for pair in "${PAIRS[@]}"; do
    # Checkpoint directories contain literal `|` characters in Hydra's run
    # name.  Assign fields explicitly instead of serializing them with a
    # delimiter, otherwise valid paths are truncated at `|loss:...`.
    case "$pair" in
        d4_2)
            a1_layers=4; a2_layers=2
            a1="$NEW_A1"; a2="$OLD_A2"
            a1_ref="$NEW_A1_REF"; a2_ref="$OLD_A2_REF"
            ;;
        d2_4)
            a1_layers=2; a2_layers=4
            a1="$OLD_A1"; a2="$NEW_A2"
            a1_ref="$OLD_A1_REF"; a2_ref="$NEW_A2_REF"
            ;;
        d4_4)
            a1_layers=4; a2_layers=4
            a1="$NEW_A1"; a2="$NEW_A2"
            # Both fullmodels are deterministic four-layer slices of the same
            # base, so one shared reference avoids redundant GPU memory.
            a1_ref="$NEW_A1_REF"; a2_ref="$NEW_A1_REF"
            ;;
        *)
            echo "Unknown depth pair: $pair" >&2
            exit 1
            ;;
    esac
    for point_spec in "${POINTS[@]}"; do
        IFS=: read -r point w1 w2 filter <<< "$point_spec"
        gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
        run_eval "$gpu" "$pair" "$a1_layers" "$a2_layers" \
            "$a1" "$a2" "$a1_ref" "$a2_ref" "$point" "$w1" "$w2" "$filter" &
        pids+=("$!")
        index=$((index + 1))
        if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
            wait_batch
        fi
    done
done
if [ "${#pids[@]}" -gt 0 ]; then
    wait_batch
fi

echo "[3/3] Summarize static depth-capacity results"
"$PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind training \
    || failures=$((failures + 1))

if [ -s "$RESULTS_DIR/F2R_SWEEP.csv" ]; then
    "$PY" - "$RESULTS_DIR/F2R_SWEEP.csv" <<'PY'
import csv
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r.get("aggregate_score")]
rows.sort(key=lambda r: float(r["aggregate_score"]), reverse=True)
if rows:
    best = rows[0]
    agg = float(best["aggregate_score"])
    print("===== Static Dual Assistant depth-4 best =====")
    print("config =", best["tag"])
    print("Agg    =", f"{agg:.6f}")
    print("Mem    =", best["memorization_score"])
    print("Util   =", best["retain_utility_score"])
    print("relative to 2/2 best 0.551721:", f"{agg - 0.551721:+.6f}")
    print("distance to target 0.580000:", f"{agg - 0.580000:+.6f}")
    print("report =", best["report"])
PY
fi
echo "Depth-4 table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$failures" -gt 0 ]; then
    echo "$failures depth-4 job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
