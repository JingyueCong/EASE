#!/usr/bin/env bash
# Final preregistered static-DualULD diagnostic: fit one causal context gate
# for each of three frozen strong-A1 / FullAnswer-A2 operating points.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"

# Preserve caller controls before loading a generic project environment.
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_CALIBRATION_BS="${CALIBRATION_BS:-2}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_AUDIT="${F2D_V512_AUDIT_SUMMARY:-}"
LAUNCH_V512_ROOT="${F2D_V512_STRONG_A1_ROOT:-}"
LAUNCH_FULL_MANIFEST="${FULLANSWER_MANIFEST:-}"
LAUNCH_RESULTS="${RESULTS_DIR:-}"
LAUNCH_CALIBRATION_DIR="${CALIBRATION_DIR:-}"

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

reference_for() {
    local checkpoint="$1"
    (cd "$checkpoint/../fullmodel" 2>/dev/null && pwd)
}

report_complete() {
    [ -s "$1" ] && grep -q '"forget_truth_ratio_knowledge"' "$1"
}

EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
AUDIT="$(absolute_from_root "${LAUNCH_AUDIT:-audits/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full/SUMMARY.json}")"
V512_ROOT="$(absolute_from_root "${LAUNCH_V512_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_a1strength4_train_seed42/v512_a1s96_u1p5}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_FULL_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
RESULTS_DIR="$(absolute_from_root "${LAUNCH_RESULTS:-open-unlearning/saves/sweeps/forget05_f2d_v512_context_gate3}")"
CALIBRATION_DIR="$(absolute_from_root "${LAUNCH_CALIBRATION_DIR:-ULD/outputs_trained_models/f2r_calibration/forget05_v512_context_gate3}")"

V512_A1_STEP="${F2D_V512_STRONG_A1_STEP:-96}"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_STEP="${FULLANSWER_STEP:-72}"

# Preregistered frontier points: utility side, current boundary winner, and
# memorization side. Do not extend this list after observing the gate results.
POINTS=(
    "gate_util:-1.9:1.6:0.0004"
    "gate_boundary:-2.0:1.7:0.0004"
    "gate_memory:-2.1:1.8:0.0004"
)

for required in "$DATA" "$AUDIT" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing context-gate prerequisite: $required" >&2
        exit 1
    fi
done
if [ ! -d "$V512_ROOT" ]; then
    echo "Missing strong V5.12 A1 root: $V512_ROOT" >&2
    exit 1
fi
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing evaluation Python: $EVAL_PY" >&2
    exit 1
fi

V512_A1="$(checkpoint_exact "$V512_ROOT" a1 "$V512_A1_STEP")"
FULL_ROOT="$(awk -F, -v tag="$FULL_TAG" '$1==tag {gsub(/\r/, "", $NF); print $NF}' "$FULL_MANIFEST" | tail -n 1)"
if [ -z "$FULL_ROOT" ]; then
    echo "Could not resolve FullAnswer tag $FULL_TAG from $FULL_MANIFEST" >&2
    exit 1
fi
FULL_ROOT="$(absolute_from_root "$FULL_ROOT")"
FULL_A2="$(checkpoint_exact "$FULL_ROOT" a2 "$FULL_STEP")"
for checkpoint in "$V512_A1" "$FULL_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing context-gate checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

V512_REFERENCE="$(reference_for "$V512_A1")"
FULL_REFERENCE="$(reference_for "$FULL_A2")"

"$EVAL_PY" - "$DATA" "$AUDIT" "$V512_REFERENCE" "$FULL_REFERENCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_path, audit_path, v512_reference, full_reference = sys.argv[1:]
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
audit = json.load(open(audit_path, encoding="utf-8"))
errors = []
if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
    errors.append("V5.12 data must contain 200 unique rows")
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in rows):
    errors.append("V5.12 data contains a foreign design")
if audit.get("units_with_deterministic_errors") != 0:
    errors.append("V5.12 audit contains deterministic errors")
if audit.get("missing_source_indices") or audit.get("unexpected_source_indices"):
    errors.append("V5.12 audit coverage is incomplete")
if errors:
    raise SystemExit("Context-gate data preflight failed: " + "; ".join(errors))

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

left, right = fingerprint(v512_reference), fingerprint(full_reference)
if left != right:
    raise SystemExit(f"Reference mismatch: V5.12={left} FullAnswer={right}")
print(
    "Context-gate preflight OK: "
    f"rows=200 deterministic_errors=0 reference_sha256={left[1]}"
)
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 3 ]; then
    echo "Context-gate pilot requires at least three GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

mkdir -p "$CALIBRATION_DIR" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,composition_mode,reference_path,calibration_kind,calibration_path,gate_training_access" > "$MANIFEST"

cat <<EOF
============================================================
V5.12 causal context-gate frontier diagnostic
  causal A1        : $V512_A1
  frozen A2        : $FULL_A2
  composition      : reference_delta
  reference        : $V512_REFERENCE
  gate labels      : C11=on; C01/C10/C00=off
  points           : -1.9/1.6, -2.0/1.7, -2.1/1.8
  filter           : 0.0004
  assistant train  : none
  retain gate train: false
  success rule     : Agg>0.551721 and Mem>=0.53 and Util>=0.60
  stop rule        : no success -> stop static DualULD gate/weight variants
  GPUs             : ${GPU_LIST[0]} ${GPU_LIST[1]} ${GPU_LIST[2]}
  calibration      : $CALIBRATION_DIR
  results          : $RESULTS_DIR
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    for point in "${POINTS[@]}"; do
        IFS=: read -r tag w1 w2 filter <<< "$point"
        echo "[dry-run] $tag weights=$w1/$w2 filter=$filter"
    done
    echo "Dry run OK; data, audit, checkpoints, references, and three gate points validated."
    exit 0
fi

train_gate() {
    local gpu="$1" tag="$2" w1="$3" w2="$4" filter="$5"
    local artifact="$CALIBRATION_DIR/${tag}.npz"
    if [ "$LAUNCH_RESUME" = "true" ] \
        && [ -s "$artifact" ] && [ -s "${artifact%.npz}.json" ]; then
        echo "gate_calibration_reuse tag=$tag"
        return
    fi
    echo "gate_calibration_start tag=$tag GPU=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$EVAL_PY" "$EASE_ROOT/scripts/train_f2r_calibration.py" \
        --mode gate \
        --counterfactual-path "$DATA" \
        --base-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --tokenizer open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --a1-path "$V512_A1" --a2-path "$FULL_A2" \
        --composition-mode reference_delta --reference-path "$V512_REFERENCE" \
        --weight-a1 "$w1" --weight-a2 "$w2" --top-filter "$filter" \
        --batch-size "$LAUNCH_CALIBRATION_BS" \
        --gate-steps 800 --gate-learning-rate 0.03 --gate-l2 0.001 \
        --seed 42 --output "$artifact" \
        > "$RESULTS_DIR/logs/${tag}_calibration.log" 2>&1
    echo "gate_calibration_done tag=$tag GPU=$gpu"
}

echo "[1/3] Fit three point-specific causal context gates"
pids=()
index=0
for point in "${POINTS[@]}"; do
    IFS=: read -r tag w1 w2 filter <<< "$point"
    train_gate "${GPU_LIST[$index]}" "$tag" "$w1" "$w2" "$filter" &
    pids+=("$!")
    index=$((index + 1))
done
failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
if [ "$failures" -ne 0 ]; then
    echo "$failures gate calibration job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi

run_eval() {
    local gpu="$1" tag="$2" w1="$3" w2="$4" filter="$5"
    local artifact="$CALIBRATION_DIR/${tag}.npz"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_CONTEXT_GATE3_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    if [ "$LAUNCH_RESUME" = "true" ] && report_complete "$report"; then
        echo "gate_evaluation_reuse tag=$tag"
        return
    fi
    echo "gate_evaluation_start tag=$tag GPU=$gpu"
    env LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        CF_PATH="$DATA" MODELS_ROOT="$RESULTS_DIR/frozen_models" \
        A1_CHECKPOINT_OVERRIDE="$V512_A1" A2_CHECKPOINT_OVERRIDE="$FULL_A2" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_TRAIN_STEPS="$V512_A1_STEP" A2_TRAIN_STEPS="$FULL_STEP" \
        A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER="$filter" \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH="$V512_REFERENCE" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=true CALIBRATION_PATH="$artifact" \
        F2R_VARIANT=F2D-V512-CausalContextGate-FullAnswerA2-Hybrid \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        HF_PREFLIGHT=0 SELECTION_RETAIN_ACCESS=true \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}_evaluation.log" 2>&1
    echo "gate_evaluation_done tag=$tag GPU=$gpu"
}

echo "[2/3] Evaluate the three gated frontier points"
pids=()
index=0
for point in "${POINTS[@]}"; do
    IFS=: read -r tag w1 w2 filter <<< "$point"
    artifact="$CALIBRATION_DIR/${tag}.npz"
    task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_CONTEXT_GATE3_${tag}"
    report="$EASE_ROOT/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    echo "$tag,$w1,$w2,$filter,$task,$report,reference_delta,$V512_REFERENCE,causal_context_gate,$artifact,forget_causal_cells_only" >> "$MANIFEST"
    run_eval "${GPU_LIST[$index]}" "$tag" "$w1" "$w2" "$filter" &
    pids+=("$!")
    index=$((index + 1))
done
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done

echo "[3/3] Summarize and apply the preregistered stop rule"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind method-ladder \
    || failures=$((failures + 1))

"$EVAL_PY" - "$EASE_ROOT" "$MANIFEST" "$RESULTS_DIR/CONTEXT_GATE_DECISION.json" <<'PY'
import json
import sys
from pathlib import Path

root, manifest, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "scripts"))
from summarize_f2r_sweep import load_rows

rows = [row for row in load_rows(manifest) if row.get("aggregate_score") is not None]
rows.sort(key=lambda row: row["aggregate_score"], reverse=True)
successful = [
    row for row in rows
    if row["aggregate_score"] > 0.551721
    and row["memorization_score"] >= 0.53
    and row["retain_utility_score"] >= 0.60
]
decision = {
    "completed_reports": len(rows),
    "expected_reports": 3,
    "prior_boundary_agg": 0.551721,
    "success_requirements": {
        "aggregate_score_strictly_greater_than": 0.551721,
        "memorization_score_at_least": 0.53,
        "retain_utility_score_at_least": 0.60,
    },
    "decision": (
        "advance_context_selective_dual_composition"
        if successful else "stop_static_dual_gate_and_weight_variants"
    ),
    "successful_tags": [row["tag"] for row in successful],
    "best": rows[0] if rows else None,
}
output.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
print(json.dumps(decision, indent=2, default=str))
if len(rows) != 3:
    raise SystemExit(f"Expected 3 complete gate reports; got {len(rows)}")
PY

echo "Context-gate table: $RESULTS_DIR/F2R_SWEEP.md"
echo "Decision artifact : $RESULTS_DIR/CONTEXT_GATE_DECISION.json"
if [ "$failures" -ne 0 ]; then
    echo "$failures gate evaluation/summary job(s) failed; inspect $RESULTS_DIR/logs" >&2
    exit 1
fi
echo "V5.12 causal context-gate diagnostic complete."
