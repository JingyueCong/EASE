#!/usr/bin/env bash
# Ceiling diagnostic only; not the causal-preserving V5.12 main experiment.
# Inference-only FullAnswer reference-delta calibration pilot:
# baseline vs vocabulary alignment vs token gate vs alignment+gate.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"

# Preserve explicit launch controls before sourcing the project environment.
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_CALIBRATION_BS="${CALIBRATION_BS:-2}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_DATA="${FULLANSWER_DATA_PATH:-}"
LAUNCH_MANIFEST="${FULLANSWER_MANIFEST:-}"
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
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_ciru200_seed42_full_authorblock_v1.jsonl}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_STEP="${FULLANSWER_STEP:-72}"
RESULTS_DIR="$(absolute_from_root "${LAUNCH_RESULTS:-open-unlearning/saves/sweeps/forget05_f2d_fullanswer_refdelta_calibration_pilot}")"
CALIBRATION_DIR="$(absolute_from_root "${LAUNCH_CALIBRATION_DIR:-ULD/outputs_trained_models/f2r_calibration/forget05_fullanswer_refdelta_pilot}")"

# This is the best documented operating point of the frozen FullAnswer pair.
# The weights are symmetric, so raw and reference-delta composition are exactly
# equivalent before alignment/gating: -w*R + w*R cancels.
WEIGHT_A1="${LAUNCH_WEIGHT_A1:--1.8}"
WEIGHT_A2="${LAUNCH_WEIGHT_A2:-1.8}"
TOP_FILTER="${LAUNCH_TOP_FILTER:-0.0004}"

for required in "$DATA" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing calibration-pilot prerequisite: $required" >&2
        exit 1
    fi
done
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing evaluation Python: $EVAL_PY" >&2
    exit 1
fi

FULL_ROOT="$(awk -F, -v tag="$FULL_TAG" '$1==tag {gsub(/\r/, "", $NF); print $NF}' "$FULL_MANIFEST" | tail -n 1)"
if [ -z "$FULL_ROOT" ]; then
    echo "Could not resolve FullAnswer tag $FULL_TAG from $FULL_MANIFEST" >&2
    exit 1
fi
FULL_ROOT="$(absolute_from_root "$FULL_ROOT")"
FULL_A1="$(checkpoint_exact "$FULL_ROOT" a1 "$FULL_STEP")"
FULL_A2="$(checkpoint_exact "$FULL_ROOT" a2 "$FULL_STEP")"
for checkpoint in "$FULL_A1" "$FULL_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing frozen FullAnswer checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

A1_REFERENCE="$(cd "$FULL_A1/../fullmodel" 2>/dev/null && pwd)"
A2_REFERENCE="$(cd "$FULL_A2/../fullmodel" 2>/dev/null && pwd)"

"$EVAL_PY" - "$A1_REFERENCE" "$A2_REFERENCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

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

left, right = map(fingerprint, sys.argv[1:])
if left != right:
    raise SystemExit(f"FullAnswer A1/A2 reference mismatch: A1={left} A2={right}")
print(f"Reference compatibility OK: sha256={left[1]}")
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "This pilot needs four GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

mkdir -p "$CALIBRATION_DIR" "$RESULTS_DIR/logs"
ALIGNMENT_ARTIFACT="$CALIBRATION_DIR/alignment.npz"
GATE_ARTIFACT="$CALIBRATION_DIR/gate.npz"
ALIGNMENT_GATE_ARTIFACT="$CALIBRATION_DIR/alignment_gate.npz"
MANIFEST="$RESULTS_DIR/manifest.csv"

cat <<EOF
============================================================
FullAnswer reference-delta alignment/gate ceiling diagnostic
  causal status   : ablation only; not the V5.12 main method
  data            : $DATA
  A1 / A2         : checkpoint-$FULL_STEP / checkpoint-$FULL_STEP
  composition     : reference_delta
  reference       : $A1_REFERENCE
  operating point : $WEIGHT_A1 / $WEIGHT_A2 / $TOP_FILTER
  methods         : baseline / alignment / gate / alignment+gate
  retraining      : none
  GPUs            : ${GPU_LIST[0]} ${GPU_LIST[1]} ${GPU_LIST[2]} ${GPU_LIST[3]}
  calibration     : $CALIBRATION_DIR
  results         : $RESULTS_DIR
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    echo "Dry run OK; checkpoints, reference, data, and four GPUs were validated."
    exit 0
fi

train_calibration() {
    local gpu="$1" mode="$2" output="$3" alignment_input="${4:-}"
    if [ "$LAUNCH_RESUME" = "true" ] \
        && [ -s "$output" ] && [ -s "${output%.npz}.json" ]; then
        echo "[$(date '+%H:%M:%S')] reuse calibration $mode"
        return
    fi
    local extra=()
    if [ -n "$alignment_input" ]; then
        extra=(--alignment-input "$alignment_input")
    fi
    echo "[$(date '+%H:%M:%S')] start calibration $mode on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$EVAL_PY" "$EASE_ROOT/scripts/train_f2r_calibration.py" \
        --mode "$mode" \
        --counterfactual-path "$DATA" \
        --base-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --tokenizer open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --a1-path "$FULL_A1" --a2-path "$FULL_A2" \
        --composition-mode reference_delta --reference-path "$A1_REFERENCE" \
        --weight-a1 "$WEIGHT_A1" --weight-a2 "$WEIGHT_A2" \
        --top-filter "$TOP_FILTER" --batch-size "$LAUNCH_CALIBRATION_BS" \
        --alignment-ridge "${ALIGNMENT_RIDGE:-10.0}" \
        --alignment-min-observations "${ALIGNMENT_MIN_OBSERVATIONS:-8}" \
        --alignment-scale-min "${ALIGNMENT_SCALE_MIN:-0.25}" \
        --alignment-scale-max "${ALIGNMENT_SCALE_MAX:-4.0}" \
        --gate-steps "${GATE_STEPS:-800}" \
        --gate-learning-rate "${GATE_LEARNING_RATE:-0.03}" \
        --gate-l2 "${GATE_L2:-0.001}" \
        --output "$output" "${extra[@]}" \
        > "$RESULTS_DIR/logs/calibration_${mode}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  calibration $mode on GPU $gpu"
}

echo "[1/3] Fit alignment and gate independently"
train_calibration "${GPU_LIST[0]}" alignment "$ALIGNMENT_ARTIFACT" &
PID_ALIGNMENT=$!
train_calibration "${GPU_LIST[1]}" gate "$GATE_ARTIFACT" &
PID_GATE=$!
CALIBRATION_FAILURES=0
wait "$PID_ALIGNMENT" || CALIBRATION_FAILURES=$((CALIBRATION_FAILURES + 1))
wait "$PID_GATE" || CALIBRATION_FAILURES=$((CALIBRATION_FAILURES + 1))
if [ "$CALIBRATION_FAILURES" -ne 0 ]; then
    echo "$CALIBRATION_FAILURES independent calibration job(s) failed." >&2
    exit 1
fi

echo "[2/3] Fit gate on aligned residuals"
train_calibration "${GPU_LIST[2]}" alignment-gate \
    "$ALIGNMENT_GATE_ARTIFACT" "$ALIGNMENT_ARTIFACT"

echo "tag,weight_a1,weight_a2,top_filter,task_name,report,composition_mode,reference_path,calibration_kind,calibration_path" > "$MANIFEST"

run_eval() {
    local gpu="$1" tag="$2" alignment="$3" gate="$4" artifact="$5"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_FullAnswerRefDeltaCal_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report,reference_delta,$A1_REFERENCE,$tag,$artifact" >> "$MANIFEST"
    if [ "$LAUNCH_RESUME" = "true" ] && report_complete "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse evaluation $tag"
        return
    fi
    echo "[$(date '+%H:%M:%S')] start evaluation $tag on GPU $gpu"
    env LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        CF_PATH="$DATA" MODELS_ROOT="$RESULTS_DIR/frozen_models" \
        A1_CHECKPOINT_OVERRIDE="$FULL_A1" A2_CHECKPOINT_OVERRIDE="$FULL_A2" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_TRAIN_STEPS="$FULL_STEP" A2_TRAIN_STEPS="$FULL_STEP" \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH="$A1_REFERENCE" \
        ALIGNMENT_ENABLED="$alignment" GATE_ENABLED="$gate" CALIBRATION_PATH="$artifact" \
        F2R_VARIANT="F2D-FullAnswer-ReferenceDelta-${tag}" \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        HF_PREFLIGHT=0 SELECTION_RETAIN_ACCESS=true \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  evaluation $tag on GPU $gpu"
}

echo "[3/3] Evaluate the four-method ladder"
run_eval "${GPU_LIST[0]}" baseline false false null &
P0=$!
run_eval "${GPU_LIST[1]}" alignment true false "$ALIGNMENT_ARTIFACT" &
P1=$!
run_eval "${GPU_LIST[2]}" gate false true "$GATE_ARTIFACT" &
P2=$!
run_eval "${GPU_LIST[3]}" alignment_gate true true "$ALIGNMENT_GATE_ARTIFACT" &
P3=$!

EVAL_FAILURES=0
for pid in "$P0" "$P1" "$P2" "$P3"; do
    wait "$pid" || EVAL_FAILURES=$((EVAL_FAILURES + 1))
done

"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind method-ladder \
    || EVAL_FAILURES=$((EVAL_FAILURES + 1))

echo "Pilot table: $RESULTS_DIR/F2R_SWEEP.md"
if [ "$EVAL_FAILURES" -ne 0 ]; then
    echo "$EVAL_FAILURES evaluation/summary job(s) failed; inspect $RESULTS_DIR/logs." >&2
    exit 1
fi
echo "FullAnswer reference-delta calibration pilot complete."
