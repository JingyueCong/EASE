#!/usr/bin/env bash
# Request-scoped causal routing: enable the frozen DualULD residual only when
# the user prompt contains one of the ten author entities in the forget request.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"

LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DRY_RUN="${DRY_RUN:-false}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_AUDIT="${F2D_V512_AUDIT_SUMMARY:-}"
LAUNCH_V512_ROOT="${F2D_V512_STRONG_A1_ROOT:-}"
LAUNCH_FULL_MANIFEST="${FULLANSWER_MANIFEST:-}"
LAUNCH_RESULTS="${RESULTS_DIR:-}"
LAUNCH_ROUTER="${SEQUENCE_ROUTER_PATH:-}"
LAUNCH_PHASE="${ENTITY_ROUTER_PHASE:-pilot}"

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
AUTHOR_MANIFEST="$(absolute_from_root "${FORGET_AUTHOR_MANIFEST:-ULD/configs/data/tofu_forget05_author_blocks.json}")"
V512_ROOT="$(absolute_from_root "${LAUNCH_V512_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_a1strength4_train_seed42/v512_a1s96_u1p5}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_FULL_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
ROUTER_PATH="$(absolute_from_root "${LAUNCH_ROUTER:-ULD/outputs_trained_models/f2r_calibration/forget05_v512_entity_router3.json}")"
ROUTER_TOKENIZER="${ROUTER_TOKENIZER:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"

V512_A1_STEP="${F2D_V512_STRONG_A1_STEP:-96}"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_STEP="${FULLANSWER_STEP:-72}"

# ``refine`` is the one refinement permitted by the pilot decision. It is a
# one-dimensional interpolation, not another two-dimensional parameter grid.
case "$LAUNCH_PHASE" in
    pilot)
        DEFAULT_RESULTS="open-unlearning/saves/sweeps/forget05_f2d_v512_entity_router3"
        TASK_NAMESPACE="V512_ENTITY_ROUTER3"
        POINTS=(
            "entity_memory:-2.1:1.6:0.0004"
            "entity_boundary:-2.0:1.7:0.0004"
            "entity_utility:-1.9:1.6:0.0004"
        )
        ;;
    refine)
        DEFAULT_RESULTS="open-unlearning/saves/sweeps/forget05_f2d_v512_entity_router_refine3"
        TASK_NAMESPACE="V512_ENTITY_ROUTER_REFINE3"
        POINTS=(
            "entity_mid35:-2.035:1.665:0.0004"
            "entity_mid50:-2.050:1.650:0.0004"
            "entity_mid65:-2.065:1.635:0.0004"
        )
        ;;
    *)
        echo "ENTITY_ROUTER_PHASE must be pilot or refine (got: $LAUNCH_PHASE)" >&2
        exit 1
        ;;
esac
RESULTS_DIR="$(absolute_from_root "${LAUNCH_RESULTS:-$DEFAULT_RESULTS}")"

for required in "$DATA" "$AUDIT" "$AUTHOR_MANIFEST" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing entity-router prerequisite: $required" >&2
        exit 1
    fi
done
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
        echo "Missing entity-router checkpoint: ${checkpoint:-unresolved}" >&2
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
if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
    raise SystemExit("Entity-router preflight requires 200 unique V5.12 rows")
if any(row.get("design_version") != "tofu-author-pairbudget-v5.12" for row in rows):
    raise SystemExit("Entity-router preflight found a foreign data design")
if audit.get("units_with_deterministic_errors") != 0:
    raise SystemExit("Entity-router preflight found deterministic audit errors")

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
print(f"Entity-router preflight OK: rows=200 reference_sha256={left[1]}")
PY

mkdir -p "$(dirname "$ROUTER_PATH")" "$RESULTS_DIR/logs"
(cd "$EASE_ROOT/open-unlearning" && \
    "$EVAL_PY" "$EASE_ROOT/scripts/build_f2r_sequence_router.py" \
        --manifest "$AUTHOR_MANIFEST" \
        --tokenizer "$ROUTER_TOKENIZER" \
        --match-scale 1.0 --nonmatch-scale 0.0 \
        --output "$ROUTER_PATH")

"$EVAL_PY" - "$ROUTER_PATH" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["kind"] == "forget_entity_subsequence_v1"
assert len(data["entities"]) == 10
assert len(data["patterns"]) >= 10
assert data["match_scale"] == 1.0 and data["nonmatch_scale"] == 0.0
assert data["retain_examples_accessed"] is False
print(
    "Entity-router hard gate OK: "
    f"entities={len(data['entities'])} patterns={len(data['patterns'])} "
    "answer_leakage=blocked retain_access=false"
)
PY

read -r -a GPU_LIST <<< "$LAUNCH_GPUS"
if [ "${#GPU_LIST[@]}" -lt 3 ]; then
    echo "Entity-router experiment requires at least three GPU ids; got: $LAUNCH_GPUS" >&2
    exit 1
fi

MANIFEST="$RESULTS_DIR/manifest.csv"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,composition_mode,reference_path,router_kind,router_path,router_match_scale,router_nonmatch_scale,router_access" > "$MANIFEST"

cat <<EOF
============================================================
V5.12 request-scoped forget-entity router
  phase            : $LAUNCH_PHASE
  causal A1        : $V512_A1
  frozen A2        : $FULL_A2
  composition      : reference_delta
  routing unit     : full input sequence, prompt only
  match            : exact token subsequence of 10 forget authors -> residual x1
  non-match        : residual x0 (base model)
  answer leakage   : blocked by labels / assistant-header boundary
  retain access    : false for router construction
  assistant train  : none
  evaluations      : 3
  target           : Agg>=0.58, Mem>=0.54, Util>=0.63
  advance rule     : Agg>0.551721, Mem>=0.53, Util>=0.60
  router           : $ROUTER_PATH
  results          : $RESULTS_DIR
============================================================
EOF

if [ "$LAUNCH_DRY_RUN" = "true" ]; then
    for point in "${POINTS[@]}"; do
        IFS=: read -r tag w1 w2 filter <<< "$point"
        echo "[dry-run] $tag weights=$w1/$w2 filter=$filter"
    done
    echo "Dry run OK; router, data, checkpoints, references, and three points validated."
    exit 0
fi

run_eval() {
    local gpu="$1" tag="$2" w1="$3" w2="$4" filter="$5"
    local task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_${TASK_NAMESPACE}_${tag}"
    local report="$EASE_ROOT/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    if [ "$LAUNCH_RESUME" = "true" ] && report_complete "$report"; then
        echo "entity_router_reuse tag=$tag"
        return
    fi
    echo "entity_router_start tag=$tag GPU=$gpu"
    env LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" \
        CF_PATH="$DATA" MODELS_ROOT="$RESULTS_DIR/frozen_models" \
        A1_CHECKPOINT_OVERRIDE="$V512_A1" A2_CHECKPOINT_OVERRIDE="$FULL_A2" \
        A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
        A1_TRAIN_STEPS="$V512_A1_STEP" A2_TRAIN_STEPS="$FULL_STEP" \
        A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0 \
        WEIGHT_A1="$w1" WEIGHT_A2="$w2" TOP_FILTER="$filter" \
        COMPOSITION_MODE=reference_delta REFERENCE_PATH="$V512_REFERENCE" \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        SEQUENCE_ROUTER_ENABLED=true SEQUENCE_ROUTER_PATH="$ROUTER_PATH" \
        F2R_VARIANT=F2D-V512-ForgetEntityRouter-FullAnswerA2-Hybrid \
        TASK_NAME="$task" EVAL_BS="$LAUNCH_EVAL_BS" EVAL_OVERWRITE=true \
        HF_PREFLIGHT=0 SELECTION_RETAIN_ACCESS=true \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "entity_router_done tag=$tag GPU=$gpu"
}

echo "[1/2] Evaluate three request-routed causal frontier points"
pids=()
index=0
for point in "${POINTS[@]}"; do
    IFS=: read -r tag w1 w2 filter <<< "$point"
    task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_${TASK_NAMESPACE}_${tag}"
    report="$EASE_ROOT/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    echo "$tag,$w1,$w2,$filter,$task,$report,reference_delta,$V512_REFERENCE,forget_entity_subsequence_v1,$ROUTER_PATH,1.0,0.0,forget_request_entities_only" >> "$MANIFEST"
    run_eval "${GPU_LIST[$index]}" "$tag" "$w1" "$w2" "$filter" &
    pids+=("$!")
    index=$((index + 1))
done
failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done

echo "[2/2] Summarize and apply preregistered decision rules"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.0 --sweep-kind method-ladder \
    || failures=$((failures + 1))

"$EVAL_PY" - "$EASE_ROOT" "$MANIFEST" "$RESULTS_DIR/ENTITY_ROUTER_DECISION.json" "$LAUNCH_PHASE" <<'PY'
import json
import sys
from pathlib import Path

root, manifest, output = map(Path, sys.argv[1:4])
phase = sys.argv[4]
sys.path.insert(0, str(root / "scripts"))
from summarize_f2r_sweep import load_rows

rows = [row for row in load_rows(manifest) if row.get("aggregate_score") is not None]
rows.sort(key=lambda row: row["aggregate_score"], reverse=True)
target = [
    row for row in rows
    if row["aggregate_score"] >= 0.58
    and row["memorization_score"] >= 0.54
    and row["retain_utility_score"] >= 0.63
]
advance = [
    row for row in rows
    if row["aggregate_score"] > 0.551721
    and row["memorization_score"] >= 0.53
    and row["retain_utility_score"] >= 0.60
]
if target:
    verdict = "target_0p58_reached"
elif advance and phase == "pilot":
    verdict = "advance_request_scoped_router_once"
else:
    verdict = "stop_request_scoped_router"
decision = {
    "completed_reports": len(rows),
    "expected_reports": 3,
    "phase": phase,
    "routing_scope": "forget_request_entities_only",
    "answer_leakage_blocked": True,
    "retain_examples_used_to_build_router": False,
    "target_requirements": {"aggregate_score": 0.58, "memorization_score": 0.54, "retain_utility_score": 0.63},
    "advance_requirements": {"aggregate_score_strictly_greater_than": 0.551721, "memorization_score_at_least": 0.53, "retain_utility_score_at_least": 0.60},
    "decision": verdict,
    "best": rows[0] if rows else None,
}
output.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
print(json.dumps(decision, indent=2, default=str))
if len(rows) != 3:
    raise SystemExit(f"Expected 3 complete entity-router reports; got {len(rows)}")
PY

echo "Entity-router table   : $RESULTS_DIR/F2R_SWEEP.md"
echo "Decision artifact     : $RESULTS_DIR/ENTITY_ROUTER_DECISION.json"
if [ "$failures" -ne 0 ]; then
    echo "$failures entity-router evaluation/summary job(s) failed" >&2
    exit 1
fi
echo "V5.12 request-scoped entity-router experiment complete."
