#!/usr/bin/env bash
# Pure-causal V5.12 inference-only pilot:
# reference-delta baseline vs scalar RMS normalization vs vocabulary alignment.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"

# Preserve caller choices before sourcing the project environment.
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_CALIBRATION_BS="${CALIBRATION_BS:-2}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_MODELS="${F2D_V512_MODELS_ROOT:-}"
LAUNCH_RESULTS="${RESULTS_DIR:-}"
LAUNCH_CALIBRATION_DIR="${CALIBRATION_DIR:-}"
LAUNCH_WEIGHT_A1="${WEIGHT_A1:-}"
LAUNCH_WEIGHT_A2="${WEIGHT_A2:-}"
LAUNCH_TOP_FILTER="${TOP_FILTER:-}"

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

report_complete() {
    [ -s "$1" ] && grep -q '"forget_truth_ratio_knowledge"' "$1"
}

EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
MODELS_ROOT="$(absolute_from_root "${LAUNCH_MODELS:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_asym6_train_seed42/v512_asym_a84_a60}")"
A1_STEP="${F2D_V512_A1_STEP:-84}"
A2_STEP="${F2D_V512_A2_STEP:-60}"
RESULTS_DIR="$(absolute_from_root "${LAUNCH_RESULTS:-open-unlearning/saves/sweeps/forget05_f2d_v512_refdelta_alignment_pilot}")"
CALIBRATION_DIR="$(absolute_from_root "${LAUNCH_CALIBRATION_DIR:-ULD/outputs_trained_models/f2r_calibration/forget05_v512_refdelta_alignment_pilot}")"

# Best completed pure-V5.12 frozen operating point.
WEIGHT_A1="${LAUNCH_WEIGHT_A1:--1.2}"
WEIGHT_A2="${LAUNCH_WEIGHT_A2:-1.2}"
TOP_FILTER="${LAUNCH_TOP_FILTER:-0.0003}"

if [ ! -s "$DATA" ]; then
    echo "Missing frozen V5.12 data: $DATA" >&2
    exit 1
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "Missing frozen V5.12 a84/a60 model root: $MODELS_ROOT" >&2
    exit 1
fi
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing evaluation Python: $EVAL_PY" >&2
    exit 1
fi

V512_A1="$(checkpoint_exact "$MODELS_ROOT" a1 "$A1_STEP")"
V512_A2="$(checkpoint_exact "$MODELS_ROOT" a2 "$A2_STEP")"
for checkpoint in "$V512_A1" "$V512_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing frozen V5.12 checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

A1_REFERENCE="$(cd "$V512_A1/../fullmodel" 2>/dev/null && pwd)"
A2_REFERENCE="$(cd "$V512_A2/../fullmodel" 2>/dev/null && pwd)"

"$EVAL_PY" - "$DATA" "$A1_REFERENCE" "$A2_REFERENCE" \
    "$WEIGHT_A1" "$WEIGHT_A2" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

data_path, a1_reference, a2_reference, weight_a1, weight_a2 = sys.argv[1:]
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
errors = []
if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
    errors.append("data must contain 200 unique rows")
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in rows):
    errors.append("data contains a non-V5.12 design")
if not math.isclose(float(weight_a1) + float(weight_a2), 0.0, abs_tol=1e-12):
    errors.append("baseline weights must be symmetric so raw/reference-delta are equivalent")
if errors:
    raise SystemExit("V5.12 reference-delta preflight failed: " + "; ".join(errors))

def fingerprint(value):
    path = Path(value)
    config = json.load(open(path / "config.json", encoding="utf-8"))
    architecture = tuple(
        config.get(key)
        for key in ("model_type", "vocab_size", "hidden_size", "num_hidden_layers")
    )
    files = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    if not files:
        raise SystemExit(f"No frozen reference weights under {path}")
    digest = hashlib.sha256()
    for file in files:
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return architecture, digest.hexdigest()

left, right = fingerprint(a1_reference), fingerprint(a2_reference)
if left != right:
    raise SystemExit(f"V5.12 A1/A2 reference mismatch: A1={left} A2={right}")
print(
    "V5.12 reference-delta preflight OK: "
    f"rows=200 reference_sha256={left[1]} symmetric_weights=true"
)
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 3 ]; then
    echo "This pilot needs at least three GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

mkdir -p "$CALIBRATION_DIR" "$RESULTS_DIR/logs"
RMS_ARTIFACT="$CALIBRATION_DIR/rms_scalar.npz"
VOCAB_ARTIFACT="$CALIBRATION_DIR/vocab_diagonal.npz"
MANIFEST="$RESULTS_DIR/manifest.csv"

cat <<EOF
============================================================
Pure V5.12 reference-delta residual-alignment pilot
  causal data      : $DATA
  A1 / A2          : V5.12 checkpoint-$A1_STEP / checkpoint-$A2_STEP
  composition      : reference_delta
  reference        : $A1_REFERENCE
  operating point  : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER
  methods          : baseline / rms_scalar / vocab_diagonal
  assistant train  : none
  gate             : disabled
  retain calibration access: false
  GPUs             : ${GPU_LIST[0]} ${GPU_LIST[1]} ${GPU_LIST[2]}
  calibration      : $CALIBRATION_DIR
  results          : $RESULTS_DIR
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run OK; pure V5.12 data, checkpoints, reference, and GPUs were validated."
    exit 0
fi

train_alignment() {
    local gpu="$1" kind="$2" output="$3"
    if [ "$LAUNCH_RESUME" = "true" ] \
        && [ -s "$output" ] && [ -s "${output%.npz}.json" ]; then
        echo "[$(date '+%H:%M:%S')] reuse alignment $kind"
        return
    fi
    echo "[$(date '+%H:%M:%S')] start alignment $kind on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$EVAL_PY" "$EASE_ROOT/scripts/train_f2r_calibration.py" \
        --mode alignment --alignment-kind "$kind" \
        --counterfactual-path "$DATA" \
        --base-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --tokenizer open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --a1-path "$V512_A1" --a2-path "$V512_A2" \
        --composition-mode reference_delta --reference-path "$A1_REFERENCE" \
        --weight-a1 "$WEIGHT_A1" --weight-a2 "$WEIGHT_A2" \
        --top-filter "$TOP_FILTER" --batch-size "$LAUNCH_CALIBRATION_BS" \
        --alignment-ridge "${ALIGNMENT_RIDGE:-10.0}" \
        --alignment-min-observations "${ALIGNMENT_MIN_OBSERVATIONS:-8}" \
        --alignment-scale-min "${ALIGNMENT_SCALE_MIN:-0.25}" \
        --alignment-scale-max "${ALIGNMENT_SCALE_MAX:-4.0}" \
        --output "$output" \
        > "$RESULTS_DIR/logs/alignment_${kind}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  alignment $kind on GPU $gpu"
}

echo "[1/3] Fit scalar RMS and vocabulary-diagonal alignments"
train_alignment "${GPU_LIST[0]}" rms_scalar "$RMS_ARTIFACT" &
P_RMS=$!
train_alignment "${GPU_LIST[1]}" vocab_diagonal "$VOCAB_ARTIFACT" &
P_VOCAB=$!
ALIGNMENT_FAILURES=0
wait "$P_RMS" || ALIGNMENT_FAILURES=$((ALIGNMENT_FAILURES + 1))
wait "$P_VOCAB" || ALIGNMENT_FAILURES=$((ALIGNMENT_FAILURES + 1))
if [ "$ALIGNMENT_FAILURES" -ne 0 ]; then
    echo "$ALIGNMENT_FAILURES alignment job(s) failed; inspect $RESULTS_DIR/logs." >&2
    exit 1
fi

echo "tag,weight_a1,weight_a2,top_filter,task_name,report,composition_mode,reference_path,calibration_kind,calibration_path" > "$MANIFEST"

run_eval() {
    local gpu="$1" tag="$2" enabled="$3" artifact="$4"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512RefDeltaAlign_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report,reference_delta,$A1_REFERENCE,$tag,$artifact" >> "$MANIFEST"
    if [ "$LAUNCH_RESUME" = "true" ] && report_complete "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse evaluation $tag"
        return
    fi
    echo "[$(date '+%H:%M:%S')] start evaluation $tag on GPU $gpu"
    env LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        CF_PATH="$DATA" MODELS_ROOT="$RESULTS_DIR/frozen_models" \
        A1_CHECKPOINT_OVERRIDE="$V512_A1" A2_CHECKPOINT_OVERRIDE="$V512_A2" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_TRAIN_STEPS="$A1_STEP" A2_TRAIN_STEPS="$A2_STEP" \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH="$A1_REFERENCE" \
        ALIGNMENT_ENABLED="$enabled" GATE_ENABLED=false CALIBRATION_PATH="$artifact" \
        F2R_VARIANT="F2D-V512-ReferenceDelta-${tag}" \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        HF_PREFLIGHT=0 SELECTION_RETAIN_ACCESS=true \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  evaluation $tag on GPU $gpu"
}

echo "[2/3] Evaluate the three causal-preserving methods"
run_eval "${GPU_LIST[0]}" baseline false null &
P0=$!
run_eval "${GPU_LIST[1]}" rms_scalar true "$RMS_ARTIFACT" &
P1=$!
run_eval "${GPU_LIST[2]}" vocab_diagonal true "$VOCAB_ARTIFACT" &
P2=$!

EVAL_FAILURES=0
for pid in "$P0" "$P1" "$P2"; do
    wait "$pid" || EVAL_FAILURES=$((EVAL_FAILURES + 1))
done

echo "[3/3] Summarize the fixed-point method comparison"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind method-ladder \
    || EVAL_FAILURES=$((EVAL_FAILURES + 1))

echo "Pilot table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$EVAL_FAILURES" -ne 0 ]; then
    echo "$EVAL_FAILURES evaluation/summary job(s) failed; inspect $RESULTS_DIR/logs." >&2
    exit 1
fi
echo "Pure V5.12 reference-delta alignment pilot complete."
