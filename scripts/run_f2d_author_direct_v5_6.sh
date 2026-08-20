#!/usr/bin/env bash
# V5.6: direct GPT core-fact contrast -> complete C01 -> independent GPT judge.
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

SPLIT="${F2D_V56_SPLIT:-forget05}"
DATASET_SPLIT="${F2D_V56_DATASET_SPLIT:-${SPLIT}_perturbed}"
SEED="${F2D_V56_SEED:-42}"
BLOCK_IDS="${F2D_V56_BLOCK_IDS:-1,4}"
GPUS="${F2D_V56_GPUS:-0 1 2 3}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${F2D_V56_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"

if [ "$BLOCK_IDS" = "all" ] || [ "$BLOCK_IDS" = "*" ]; then
    BLOCK_COUNT=10
    RUN_TAG="full"
else
    IFS=',' read -r -a BLOCK_ARRAY <<< "$BLOCK_IDS"
    BLOCK_COUNT="${#BLOCK_ARRAY[@]}"
    RUN_TAG="blocks${BLOCK_IDS//,/_}"
fi
UNITS=$((BLOCK_COUNT * 20))
DATA_PATH="${F2D_V56_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_direct${UNITS}_seed${SEED}_v5_6_${RUN_TAG}.jsonl}"
PROFILES_PATH="${F2D_V56_PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${F2D_V56_STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${F2D_V56_AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_direct${UNITS}_seed${SEED}_v5_6_${RUN_TAG}}"
SWEEP_NAME="${F2D_V56_SWEEP_NAME:-f2d_author_direct${UNITS}_v5_6_seed${SEED}}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_BLOCK_CONCURRENCY="${CF_BLOCK_CONCURRENCY:-1}"
CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_PROFILE_RETRIES="${CF_PROFILE_RETRIES:-6}"
CF_ROW_RETRIES="${CF_ROW_RETRIES:-4}"
CF_JUDGE_ROUNDS="${CF_JUDGE_ROUNDS:-4}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-18000}"

if [ "$SPLIT" != "forget05" ]; then
    echo "V5.6 currently requires F2D_V56_SPLIT=forget05." >&2
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
echo "F2D author direct-contrast V5.6"
echo "  split / units      : $SPLIT / $UNITS"
echo "  selected blocks    : $BLOCK_IDS"
echo "  architecture       : GPT core contrast -> full C01 -> GPT judge"
echo "  generator / judge  : $CF_MODEL / $JUDGE_MODEL"
echo "  row workers/retries: $CF_ROW_CONCURRENCY / $CF_ROW_RETRIES"
echo "  profile/judge retry: $CF_PROFILE_RETRIES / $CF_JUDGE_ROUNDS"
echo "  factorial JSONL    : $DATA_PATH"
echo "  resumable state    : $STATE_DIR"
echo "  legacy preservation: FullAnswer and V1-V5.5 untouched"
echo "============================================================"

echo "[0/3] Offline legacy and direct-contrast preflight"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_anchor_v5_full_preflight.py"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_anchor_v5_2_rowlocal.py"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_author_ledger_v5_3.py"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_author_slots_v5_4.py"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_author_answers_v5_5.py"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_author_direct_v5_6.py"

echo "[1/3] Generate direct GPT core-fact interventions"
if [ ! -s "$DATA_PATH" ] || [ ! -s "$PROFILES_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_direct_v5_6.py" \
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
    echo "Reusing frozen V5.6 data: $DATA_PATH"
fi

echo "[2/3] Deterministic schema and causal-design audit"
AUDIT_EXPECTED=()
if [ "$BLOCK_COUNT" -eq 10 ]; then
    AUDIT_EXPECTED=(--expected-units 200)
fi
"$GEN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$DATA_PATH" \
    --output-dir "$AUDIT_DIR" \
    "${AUDIT_EXPECTED[@]}" \
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
    profiles = json.load(handle)

design = "tofu-author-direct-contrast-v5.6"
renderer = "direct-contrast-complete-answer-v5.6"
mapping = "gpt-core-fact-contrast-conditioned-complete-c01-answer"
schema = "direct-core-fact-v1"
contract_policy = "semantic-compatible-v5.5.1"
judge_fields = (
    "target_relation_match", "target_fact_changed",
    "profile_consistent", "natural_surface",
)
if len(rows) != expected_rows or len({row["source_id"] for row in rows}) != expected_rows:
    raise SystemExit("V5.6 row coverage is incomplete or duplicated")
blocks = Counter(row.get("block_id") for row in rows)
if len(blocks) != expected_blocks or set(blocks.values()) != {20}:
    raise SystemExit(f"V5.6 block coverage mismatch: {blocks}")
for row in rows:
    generation = row.get("generation", {})
    entry = row.get("fact_ledger_entry", {})
    if row.get("design_version") != design:
        raise SystemExit(f"Wrong design in {row.get('source_id')}")
    if row.get("contrast_schema_version") != schema:
        raise SystemExit(f"Missing direct contrast in {row.get('source_id')}")
    if generation.get("surface_renderer") != renderer:
        raise SystemExit(f"Wrong renderer in {row.get('source_id')}")
    if generation.get("mapping_scope") != mapping:
        raise SystemExit(f"Wrong mapping scope in {row.get('source_id')}")
    if generation.get("response_contract_policy") != contract_policy:
        raise SystemExit(f"Wrong response policy in {row.get('source_id')}")
    if generation.get("contrast_schema_version") != schema:
        raise SystemExit(f"Missing contrast provenance in {row.get('source_id')}")
    if entry.get("source_id") != row.get("source_id"):
        raise SystemExit(f"Missing ledger entry in {row.get('source_id')}")
    if entry.get("contrast_status") == "abstain":
        raise SystemExit(f"ABSTAIN row reached final JSONL: {row.get('source_id')}")
    if any(row.get("semantic_judge", {}).get(field) is not True for field in judge_fields):
        raise SystemExit(f"Unapproved semantic row {row.get('source_id')}")

if profiles.get("design_version") != design:
    raise SystemExit("Profile artifact is not direct-contrast V5.6")
if profiles.get("contrast_schema_version") != schema:
    raise SystemExit("Profile artifact has the wrong contrast schema")
if len(profiles.get("profiles", [])) != expected_blocks:
    raise SystemExit("V5.6 profile block count mismatch")
for profile in profiles["profiles"]:
    if len(profile.get("fact_ledger", [])) != 20:
        raise SystemExit("V5.6 ledger does not cover 20 rows")
    verdict = profile.get("profile_semantic_judge", {})
    if verdict.get("coherent") is not True or verdict.get("contrastive") is not True:
        raise SystemExit("V5.6 profile lacks independent contrast approval")
    if verdict.get("abstain_source_ids"):
        raise SystemExit("V5.6 approved profile contains ABSTAIN rows")
print(
    f"Author direct-contrast V5.6 hard gate OK: rows={len(rows)} "
    f"blocks={len(blocks)} direct_core_facts=approved judges=independent"
)
PY

echo "[3/3] V5.6 generation and audit complete"
echo "Human review: $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "Risk review : $AUDIT_DIR/HUMAN_AUDIT_RISK.md"

if [ "${STOP_AFTER_AUDIT:-true}" = "true" ]; then
    echo "STOP_AFTER_AUDIT=true; no assistant was trained."
    exit 0
fi
if [ "$BLOCK_COUNT" -ne 10 ]; then
    echo "Refusing training on a smoke subset; rerun with F2D_V56_BLOCK_IDS=all." >&2
    exit 2
fi
if [ "${AUDIT_APPROVED:-false}" != "true" ]; then
    echo "Refusing training until AUDIT_APPROVED=true after human review." >&2
    exit 2
fi

TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
direct_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
direct_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60 \
direct_s72_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.8}" WEIGHT_A2="${WEIGHT_A2:-1.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0004}" EVAL_BS="${EVAL_BS:-4}" \
    F2R_VARIANT="F2D-AuthorDirectContrast200-v5.6" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
