#!/usr/bin/env bash
# Wait for the current V5.12 84/60 inference sweep, then run four exact
# paired-difference-span assistant-training experiments on the final best point.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"
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

TRAIN_PY="${TRAIN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
BASE_DATA="$(absolute_from_root "${F2D_V512_DATA_PATH:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
DIFFSPAN_DATA="$(absolute_from_root "${F2D_V512_DIFFSPAN_PATH:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_diffspan_v1.jsonl}")"
WAIT_PID_FILE="$(absolute_from_root "${WAIT_PID_FILE:-logs/f2d_v512_asym_a84a60_fine.pid}")"
WAIT_RESULTS="$(absolute_from_root "${WAIT_RESULTS:-open-unlearning/saves/sweeps/forget05_f2d_v512_asym_a84a60_fine/F2R_SWEEP.csv}")"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-30}"
REQUESTED_GPUS="${GPUS:-0 1 2 3}"
REQUESTED_RESUME="${RESUME:-true}"
SWEEP_NAME="${SWEEP_NAME:-f2d_v512_diffspan4_seed42}"
RESULTS_DIR="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}"
MODELS_ROOT="${EASE_ROOT}/ULD/outputs_trained_models/f2d_v512_diffspan4_seed42"
MANIFEST="${RESULTS_DIR}/manifest.csv"

for executable in "$TRAIN_PY" "$EVAL_PY"; do
    if [ ! -x "$executable" ]; then
        echo "Missing Python environment: $executable" >&2
        exit 1
    fi
done
if [ ! -s "$BASE_DATA" ]; then
    echo "Missing frozen V5.12 data: $BASE_DATA" >&2
    exit 1
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "QUEUE_POLL_SECONDS must be a positive integer" >&2
    exit 1
fi

echo "============================================================"
echo "Queued V5.12 exact-difference-span experiment"
echo "  wait PID file   : $WAIT_PID_FILE"
echo "  wait results    : $WAIT_RESULTS"
echo "  source data     : $BASE_DATA"
echo "  diff-span data  : $DIFFSPAN_DATA"
echo "  configurations  : 48/48 and 60/48 x preserve-KL 0.02/0.05"
echo "  GPUs            : $REQUESTED_GPUS"
echo "  polling         : ${POLL_SECONDS}s"
echo "============================================================"

if [ -s "$WAIT_PID_FILE" ]; then
    WAIT_PID="$(tr -d '[:space:]' < "$WAIT_PID_FILE")"
    if [[ "$WAIT_PID" =~ ^[0-9]+$ ]]; then
        while kill -0 "$WAIT_PID" 2>/dev/null; do
            WAIT_STAT="$(ps -p "$WAIT_PID" -o stat= 2>/dev/null | tr -d '[:space:]')"
            if [ -z "$WAIT_STAT" ] || [[ "$WAIT_STAT" == Z* ]]; then
                break
            fi
            echo "[$(date '+%F %T')] waiting for prior 84/60 fine sweep PID=$WAIT_PID stat=$WAIT_STAT"
            sleep "$POLL_SECONDS"
        done
        echo "[$(date '+%F %T')] prior PID=$WAIT_PID has finished"
    else
        echo "Invalid prior PID file: $WAIT_PID_FILE" >&2
        exit 1
    fi
else
    echo "No prior PID file found; validating the completed 36-point table directly"
fi

read -r WEIGHT_A1 WEIGHT_A2 TOP_FILTER PRIOR_AGG <<< "$($EVAL_PY - "$WAIT_RESULTS" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("aggregate_score")]
if len(rows) != 36:
    raise SystemExit(
        f"Refusing to start diff-span training: expected 36 valid prior results, got {len(rows)}"
    )
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
best = rows[0]
print(
    best["weight_a1"], best["weight_a2"], best["top_filter"],
    best["aggregate_score"],
)
PY
)"
echo "Prior final best: Agg=$PRIOR_AGG w1=$WEIGHT_A1 w2=$WEIGHT_A2 filter=$TOP_FILTER"

if [ ! -s "$DIFFSPAN_DATA" ]; then
    echo "[1/3] Deriving exact paired-difference spans (no API and no retain data)"
    "$TRAIN_PY" "$EASE_ROOT/scripts/annotate_f2d_hierarchy.py" \
        --input "$BASE_DATA" --output "$DIFFSPAN_DATA" \
        --expected-units 200 --claim-mode evidence
else
    echo "[1/3] Reusing diff-span data: $DIFFSPAN_DATA"
fi

"$TRAIN_PY" - "$BASE_DATA" "$DIFFSPAN_DATA" <<'PY'
import copy
import json
import statistics
import sys

base_path, annotated_path = sys.argv[1:]

def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

base = load(base_path)
annotated = load(annotated_path)
errors = []
if len(base) != 200 or len(annotated) != 200:
    errors.append(f"row coverage mismatch: base={len(base)} annotated={len(annotated)}")
if [row.get("source_id") for row in base] != [row.get("source_id") for row in annotated]:
    errors.append("source order/identity changed")

coverages = []
for original, current in zip(base, annotated):
    stripped = copy.deepcopy(current)
    for name in ("C11", "C01", "C10", "C00"):
        cell = current.get("cells", {}).get(name, {})
        supervision = cell.get("supervision", {})
        claims = supervision.get("claim_spans", [])
        evidence = supervision.get("evidence_spans", [])
        if supervision.get("version") != "paired-diffspan-v1":
            errors.append(f"{current.get('source_id')}.{name}: wrong supervision version")
        if not claims or claims != evidence:
            errors.append(f"{current.get('source_id')}.{name}: claim/evidence mismatch")
        answer = cell.get("answer", "")
        for start, end in claims:
            if not (0 <= int(start) < int(end) <= len(answer)):
                errors.append(f"{current.get('source_id')}.{name}: invalid span {start}:{end}")
        selected = sum(int(end) - int(start) for start, end in claims)
        coverages.append(selected / max(len(answer), 1))
        stripped["cells"][name].pop("supervision", None)
    if stripped != original:
        errors.append(f"{current.get('source_id')}: content changed beyond supervision")

if errors:
    raise SystemExit("Diff-span hard gate failed: " + " | ".join(errors[:20]))
ordered = sorted(coverages)
p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
print(
    "Diff-span hard gate OK: rows=200 cells=800 "
    f"mean_coverage={statistics.mean(coverages):.3f} p95={p95:.3f} "
    "content_changes=0 retain_access=false"
)
PY

IFS=' ' read -r -a GPU_LIST <<< "$REQUESTED_GPUS"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "This four-configuration experiment requires four GPU ids in GPUS" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR/logs" "$MODELS_ROOT"
echo "tag,weight_a1,weight_a2,top_filter,task_name,report,views,a1_num_layer,a2_num_layer,a1_lora_r,a2_lora_r,a1_lora_alpha,a2_lora_alpha,a1_train_lr,a2_train_lr,a1_train_ep,a2_train_ep,a1_train_steps,a2_train_steps,a1_retain_weight,a2_retain_weight,preserve_kl_weight,evidence_weight,claim_mode,a1_seed,a2_seed,a1_data_mode,a2_data_mode,variant,models_root" > "$MANIFEST"

# tag:a1_steps:a2_steps:preserve_kl
CONFIGS=(
    "diffspan_s48_kl0p02:48:48:0.02"
    "diffspan_s48_kl0p05:48:48:0.05"
    "diffspan_a60_a48_kl0p02:60:48:0.02"
    "diffspan_a60_a48_kl0p05:60:48:0.05"
)

for config in "${CONFIGS[@]}"; do
    IFS=: read -r tag a1_steps a2_steps preserve_kl <<< "$config"
    task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_DIFFSPAN4_${tag}"
    report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    model_root="${MODELS_ROOT}/${tag}"
    echo "$tag,$WEIGHT_A1,$WEIGHT_A2,$TOP_FILTER,$task,$report,1,2,2,16,16,32,32,5e-4,5e-4,1,1,$a1_steps,$a2_steps,1,1,$preserve_kl,0.0,evidence,42,42,uf2d_hier_a1,uf2d_hier_a2,F2D-PairBudget-v5.12-DiffSpan-v1,$model_root" >> "$MANIFEST"
done

run_one() {
    local gpu="$1" config="$2"
    local tag a1_steps a2_steps preserve_kl task report model_root
    IFS=: read -r tag a1_steps a2_steps preserve_kl <<< "$config"
    task="tofu_Llama-3.2-1B-Instruct_forget05_F2R_V512_DIFFSPAN4_${tag}"
    report="${EASE_ROOT}/open-unlearning/saves/eval/${task}/F2R_REPORT.json"
    model_root="${MODELS_ROOT}/${tag}"

    if [ "$REQUESTED_RESUME" = "true" ] && [ -s "$report" ] \
        && grep -q '"forget_truth_ratio_knowledge"' "$report"; then
        echo "[$(date '+%H:%M:%S')] reuse $tag"
        return
    fi

    echo "[$(date '+%H:%M:%S')] start $tag on GPU $gpu"
    env LOAD_DOTENV=0 MODE=full SPLIT=forget05 GPU="$gpu" VIEWS=1 \
        CF_PATH="$DIFFSPAN_DATA" MODELS_ROOT="$model_root" TASK_NAME="$task" \
        TRAIN_RUN_TAG="${SWEEP_NAME}_${tag}" \
        A1_DATA_MODE=uf2d_hier_a1 A2_DATA_MODE=uf2d_hier_a2 \
        A1_NUM_LAYER=2 A2_NUM_LAYER=2 A1_LORA_R=16 A2_LORA_R=16 \
        A1_LORA_ALPHA=32 A2_LORA_ALPHA=32 \
        A1_TRAIN_LR=5e-4 A2_TRAIN_LR=5e-4 \
        A1_TRAIN_EP=1 A2_TRAIN_EP=1 \
        A1_TRAIN_STEPS="$a1_steps" A2_TRAIN_STEPS="$a2_steps" \
        A1_RETAIN_WEIGHT=1 A2_RETAIN_WEIGHT=1 A1_SEED=42 A2_SEED=42 \
        TRAIN_LOSS_CONFIG=factorial_hierarchical \
        PRESERVE_KL_WEIGHT="$preserve_kl" EVIDENCE_WEIGHT=0.0 \
        WEIGHT_A1="$WEIGHT_A1" WEIGHT_A2="$WEIGHT_A2" TOP_FILTER="$TOP_FILTER" \
        F2R_VARIANT=F2D-PairBudget-v5.12-DiffSpan-v1 \
        ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
        HF_PREFLIGHT=0 EVAL_OVERWRITE=true SELECTION_RETAIN_ACCESS=true \
        EVAL_BS="${EVAL_BS:-4}" \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        > "$RESULTS_DIR/logs/${tag}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] done  $tag on GPU $gpu"
}

echo "[2/3] Training/evaluating four exact-difference-span configurations"
pids=()
for index in "${!CONFIGS[@]}"; do
    run_one "${GPU_LIST[$index]}" "${CONFIGS[$index]}" &
    pids+=("$!")
done

FAILURES=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        FAILURES=$((FAILURES + 1))
    fi
done

echo "[3/3] Summarizing exact-difference-span results"
"$EVAL_PY" "$EASE_ROOT/scripts/summarize_f2r_sweep.py" \
    --manifest "$MANIFEST" --output-dir "$RESULTS_DIR" \
    --target-agg 0.58 --target-margin 0.005 --sweep-kind training \
    || FAILURES=$((FAILURES + 1))

if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES diff-span job/summary failure(s); inspect $RESULTS_DIR/logs" >&2
    exit 1
fi

"$EVAL_PY" - "$RESULTS_DIR/F2R_SWEEP.csv" "$PRIOR_AGG" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("aggregate_score")]
if len(rows) != 4:
    raise SystemExit(f"Expected four valid diff-span results; got {len(rows)}")
rows.sort(key=lambda row: float(row["aggregate_score"]), reverse=True)
best = rows[0]
agg = float(best["aggregate_score"])
prior = float(sys.argv[2])
print("\n===== V5.12 exact-difference-span best =====")
print("config =", best["tag"])
print(f"Agg    = {agg:.6f}")
print(f"Mem    = {float(best['memorization_score']):.6f}")
print(f"Util   = {float(best['retain_utility_score']):.6f}")
print(f"MU     = {float(best['model_utility']):.6f}")
print("KL     =", best.get("preserve_kl_weight"))
print("steps  =", best.get("a1_train_steps"), "/", best.get("a2_train_steps"))
print(f"relative to completed prior best {prior:.6f}: {agg - prior:+.6f}")
print(f"relative to FullAnswer 0.553712: {agg - 0.553712:+.6f}")
print(f"distance to BS-S 0.580000:      {agg - 0.580000:+.6f}")
print("report =", best["report"])
PY

echo "Diff-span table: $RESULTS_DIR/F2R_SWEEP.md"
