#!/usr/bin/env bash
# V5.2: block profile -> row-local anchor mapping -> deterministic merge/judge.
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

SPLIT="${F2D_V52_SPLIT:-forget05}"
DATASET_SPLIT="${F2D_V52_DATASET_SPLIT:-${SPLIT}_perturbed}"
SEED="${F2D_V52_SEED:-42}"
BLOCK_IDS="${F2D_V52_BLOCK_IDS:-1,4}"
GPUS="${F2D_V52_GPUS:-0 1 2 3}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${F2D_V52_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"

if [ "$BLOCK_IDS" = "all" ] || [ "$BLOCK_IDS" = "*" ]; then
    BLOCK_COUNT=10
    RUN_TAG="full"
else
    IFS=',' read -r -a BLOCK_ARRAY <<< "$BLOCK_IDS"
    BLOCK_COUNT="${#BLOCK_ARRAY[@]}"
    RUN_TAG="blocks${BLOCK_IDS//,/_}"
fi
UNITS=$((BLOCK_COUNT * 20))
DATA_PATH="${F2D_V52_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_rowlocal${UNITS}_seed${SEED}_v5_2_${RUN_TAG}.jsonl}"
PROFILES_PATH="${F2D_V52_PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${F2D_V52_STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${F2D_V52_AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_rowlocal${UNITS}_seed${SEED}_v5_2_${RUN_TAG}}"
SWEEP_NAME="${F2D_V52_SWEEP_NAME:-f2d_author_rowlocal${UNITS}_v5_2_seed${SEED}}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_BLOCK_CONCURRENCY="${CF_BLOCK_CONCURRENCY:-1}"
CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_PROFILE_RETRIES="${CF_PROFILE_RETRIES:-4}"
CF_ROW_RETRIES="${CF_ROW_RETRIES:-4}"
CF_JUDGE_ROUNDS="${CF_JUDGE_ROUNDS:-3}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-0.8}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-8000}"

if [ "$SPLIT" != "forget05" ]; then
    echo "V5.2 currently requires F2D_V52_SPLIT=forget05." >&2
    exit 1
fi
if [ -z "$CF_BASE_URL" ]; then
    echo "Set OPENAI_BASE_URL (or CF_BASE_URL) in $ENV_FILE." >&2
    exit 1
fi
if [ -z "${!CF_API_KEY_ENV:-}" ] || [ -z "${!JUDGE_API_KEY_ENV:-}" ]; then
    echo "Set $CF_API_KEY_ENV and $JUDGE_API_KEY_ENV in $ENV_FILE." >&2
    exit 1
fi

echo "============================================================"
echo "F2D author row-local V5.2"
echo "  split / units      : $SPLIT / $UNITS"
echo "  selected blocks    : $BLOCK_IDS"
echo "  architecture       : profile -> row map -> reconcile -> block judge"
echo "  row workers/retries: $CF_ROW_CONCURRENCY / $CF_ROW_RETRIES"
echo "  judge repair rounds: $CF_JUDGE_ROUNDS"
echo "  factorial JSONL    : $DATA_PATH"
echo "  resumable state    : $STATE_DIR"
echo "  legacy preservation: FullAnswer and V1-V5.1 untouched"
echo "============================================================"

echo "[0/3] Offline frozen-anchor and row-local preflight"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_anchor_v5_full_preflight.py"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_anchor_v5_2_rowlocal.py"

echo "[1/3] Generate row-local author-level causal units"
if [ ! -s "$DATA_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_rowlocal_v5_2.py" \
        --split "$DATASET_SPLIT" \
        --manifest "$MANIFEST" \
        --output "$DATA_PATH" \
        --profiles-output "$PROFILES_PATH" \
        --state-dir "$STATE_DIR" \
        --block-ids "$BLOCK_IDS" \
        --seed "$SEED" \
        --model "$CF_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --base-url "$CF_BASE_URL" \
        --judge-base-url "$JUDGE_BASE_URL" \
        --api-key-env "$CF_API_KEY_ENV" \
        --judge-api-key-env "$JUDGE_API_KEY_ENV" \
        --temperature "$CF_TEMPERATURE" \
        --judge-temperature "$JUDGE_TEMPERATURE" \
        --max-completion-tokens "$CF_MAX_COMPLETION_TOKENS" \
        --block-concurrency "$CF_BLOCK_CONCURRENCY" \
        --row-concurrency "$CF_ROW_CONCURRENCY" \
        --request-retries "$CF_REQUEST_RETRIES" \
        --profile-retries "$CF_PROFILE_RETRIES" \
        --row-retries "$CF_ROW_RETRIES" \
        --judge-rounds "$CF_JUDGE_ROUNDS" \
        --json-mode "$CF_JSON_MODE"
else
    echo "Reusing frozen V5.2 data: $DATA_PATH"
fi

echo "[2/3] Deterministic schema and causal-design audit"
"$GEN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$DATA_PATH" \
    --output-dir "$AUDIT_DIR" \
    --expected-units "$UNITS" \
    --block-size 20 \
    --sample-count 20 \
    --seed "$SEED" \
    --fail-on-deterministic-errors

"$GEN_PY" - "$DATA_PATH" "$PROFILES_PATH" "$UNITS" "$BLOCK_COUNT" <<'PY'
import json
import sys
from collections import Counter

data_path, profiles_path, expected_rows, expected_blocks = sys.argv[1:]
expected_rows, expected_blocks = int(expected_rows), int(expected_blocks)
with open(data_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
with open(profiles_path, encoding="utf-8") as handle:
    ledger = json.load(handle)
design = "tofu-author-rowlocal-v5.2"
judge_fields = (
    "target_relation_match", "target_fact_changed",
    "profile_consistent", "natural_surface",
)
if len(rows) != expected_rows or len({row["source_id"] for row in rows}) != expected_rows:
    raise SystemExit("V5.2 row coverage is incomplete or duplicated")
blocks = Counter(row.get("block_id") for row in rows)
if len(blocks) != expected_blocks or set(blocks.values()) != {20}:
    raise SystemExit(f"V5.2 block coverage mismatch: {blocks}")
for row in rows:
    if row.get("design_version") != design:
        raise SystemExit(f"Wrong design in {row.get('source_id')}")
    generation = row.get("generation", {})
    if generation.get("surface_renderer") != "deterministic-rowlocal-anchor-v5.2":
        raise SystemExit(f"Wrong renderer in {row.get('source_id')}")
    if generation.get("mapping_scope") != "single-row-local-anchors":
        raise SystemExit(f"Wrong mapping scope in {row.get('source_id')}")
    if any(row.get("semantic_judge", {}).get(field) is not True for field in judge_fields):
        raise SystemExit(f"Unapproved semantic row {row.get('source_id')}")
if ledger.get("design_version") != design:
    raise SystemExit("Profile ledger is not row-local V5.2")
if len(ledger.get("profiles", [])) != expected_blocks:
    raise SystemExit("Profile ledger block count mismatch")
print(
    f"Row-local V5.2 hard gate OK: rows={len(rows)} "
    f"blocks={len(blocks)} judges=all-pass"
)
PY

echo "[3/3] V5.2 generation and audit complete"
echo "Human review: $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "Risk review : $AUDIT_DIR/HUMAN_AUDIT_RISK.md"

if [ "${STOP_AFTER_AUDIT:-true}" = "true" ]; then
    echo "STOP_AFTER_AUDIT=true; no assistant was trained."
    exit 0
fi
if [ "$BLOCK_COUNT" -ne 10 ]; then
    echo "Refusing training on a block smoke subset; rerun with F2D_V52_BLOCK_IDS=all." >&2
    exit 2
fi
if [ "${AUDIT_APPROVED:-false}" != "true" ]; then
    echo "Refusing training until AUDIT_APPROVED=true after human review." >&2
    exit 2
fi

TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
rowlocal_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
rowlocal_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60 \
rowlocal_s72_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.8}" WEIGHT_A2="${WEIGHT_A2:-1.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0004}" EVAL_BS="${EVAL_BS:-4}" \
    F2R_VARIANT="F2D-AuthorRowLocal200-v5.2" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
