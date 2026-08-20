#!/usr/bin/env bash
# V5.9: explicit context -> semantic plan -> Q+A -> independent critic -> repair.
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

SPLIT="${F2D_V59_SPLIT:-forget05}"
DATASET_SPLIT="${F2D_V59_DATASET_SPLIT:-${SPLIT}_perturbed}"
SEED="${F2D_V59_SEED:-42}"
BLOCK_IDS="${F2D_V59_BLOCK_IDS:-1,4}"
GPUS="${F2D_V59_GPUS:-0 1 2 3}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${F2D_V59_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"

if [ "$BLOCK_IDS" = "all" ] || [ "$BLOCK_IDS" = "*" ]; then
    BLOCK_COUNT=10
    RUN_TAG="full"
else
    IFS=',' read -r -a BLOCK_ARRAY <<< "$BLOCK_IDS"
    BLOCK_COUNT="${#BLOCK_ARRAY[@]}"
    RUN_TAG="blocks${BLOCK_IDS//,/_}"
fi
UNITS=$((BLOCK_COUNT * 20))
DATA_PATH="${F2D_V59_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_semantic_agent${UNITS}_seed${SEED}_v5_9_${RUN_TAG}.jsonl}"
PROFILES_PATH="${F2D_V59_PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${F2D_V59_STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${F2D_V59_AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_semantic_agent${UNITS}_seed${SEED}_v5_9_${RUN_TAG}}"
SWEEP_NAME="${F2D_V59_SWEEP_NAME:-f2d_author_semantic_agent${UNITS}_v5_9_seed${SEED}}"
PREFLIGHT_LOG="${F2D_V59_PREFLIGHT_LOG:-${STATE_DIR}.preflight.log}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_BLOCK_CONCURRENCY="${CF_BLOCK_CONCURRENCY:-2}"
CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_PROFILE_RETRIES="${CF_PROFILE_RETRIES:-6}"
CF_ROW_RETRIES="${CF_ROW_RETRIES:-4}"
CF_JUDGE_ROUNDS="${CF_JUDGE_ROUNDS:-4}"
CF_JUDGE_BATCH_SIZE="${CF_JUDGE_BATCH_SIZE:-5}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-18000}"

if [ "$SPLIT" != "forget05" ]; then
    echo "V5.9 currently requires F2D_V59_SPLIT=forget05." >&2
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
echo "F2D author semantic Agent V5.9"
echo "  split / units      : $SPLIT / $UNITS"
echo "  selected blocks    : $BLOCK_IDS"
echo "  architecture       : context -> plan -> Q+A -> critic -> repair"
echo "  deterministic role : coverage / frozen C11 / leakage / provenance"
echo "  semantic role      : row planner + independent critic + block judge"
echo "  generator / critic : $CF_MODEL / $JUDGE_MODEL"
echo "  block/row workers  : $CF_BLOCK_CONCURRENCY / $CF_ROW_CONCURRENCY"
echo "  profile/row retries: $CF_PROFILE_RETRIES / $CF_ROW_RETRIES"
echo "  judge rounds/batch : $CF_JUDGE_ROUNDS / $CF_JUDGE_BATCH_SIZE rows"
echo "  factorial JSONL    : $DATA_PATH"
echo "  resumable state    : $STATE_DIR"
echo "  preflight details  : $PREFLIGHT_LOG"
echo "  legacy preservation: FullAnswer and V1-V5.8 untouched"
echo "============================================================"

echo "[0/3] Offline legacy and semantic-agent preflight"
mkdir -p "$(dirname "$PREFLIGHT_LOG")"
: > "$PREFLIGHT_LOG"
run_preflight() {
    local test_path="$1"
    echo "===== $test_path =====" >> "$PREFLIGHT_LOG"
    if "$GEN_PY" "$EASE_ROOT/$test_path" >> "$PREFLIGHT_LOG" 2>&1; then
        echo "preflight_ok test=$(basename "$test_path")"
    else
        echo "Preflight failed: $test_path" >&2
        tail -n 100 "$PREFLIGHT_LOG" >&2
        exit 1
    fi
}
run_preflight tests/test_tofu_anchor_v5_full_preflight.py
run_preflight tests/test_tofu_anchor_v5_2_rowlocal.py
run_preflight tests/test_tofu_author_ledger_v5_3.py
run_preflight tests/test_tofu_author_slots_v5_4.py
run_preflight tests/test_tofu_author_answers_v5_5.py
run_preflight tests/test_tofu_author_direct_v5_6.py
run_preflight tests/test_tofu_author_joint_v5_7.py
run_preflight tests/test_tofu_author_semantic_v5_8.py
run_preflight tests/test_tofu_author_semantic_agent_v5_9.py

echo "[1/3] Run context-preserving semantic agents"
if [ ! -s "$DATA_PATH" ] || [ ! -s "$PROFILES_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_semantic_agent_v5_9.py" \
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
        --judge-batch-size "$CF_JUDGE_BATCH_SIZE" \
        --json-mode "$CF_JSON_MODE"
else
    echo "Reusing frozen V5.9 data: $DATA_PATH"
fi

echo "[2/3] Deterministic schema, trace, and semantic hard gate"
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
import re
import sys
from collections import Counter

data_path, profiles_path, expected_rows, expected_blocks = sys.argv[1:]
expected_rows, expected_blocks = int(expected_rows), int(expected_blocks)
with open(data_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
with open(profiles_path, encoding="utf-8") as handle:
    profiles = json.load(handle)

design = "tofu-author-semantic-agent-v5.9"
renderer = "semantic-agent-complete-c01-question-answer-v5.9"
mapping = "row-local-context-packet-agent-with-independent-critic"
schema = "semantic-agent-question-answer-v1"
policy = "semantic-agent-critic-v5.9"
agent = "context-plan-generate-critic-repair-v1"
brief_schema = "tofu-row-semantic-brief-v1"
critic_fields = (
    "same_target_relation", "question_premises_updated",
    "answer_addresses_question", "replacement_fact_expressed",
    "source_fact_removed", "response_mode_compatible", "natural_surface",
)
block_judge_fields = (
    "target_relation_match", "question_premise_profile_consistent",
    "answer_satisfies_question", "target_fact_changed",
    "profile_consistent", "natural_surface",
)
if len(rows) != expected_rows or len({row["source_id"] for row in rows}) != expected_rows:
    raise SystemExit("V5.9 row coverage is incomplete or duplicated")
blocks = Counter(row.get("block_id") for row in rows)
if len(blocks) != expected_blocks or set(blocks.values()) != {20}:
    raise SystemExit(f"V5.9 block coverage mismatch: {blocks}")

for row in rows:
    source_id = row["source_id"]
    generation = row.get("generation", {})
    trace = row.get("semantic_agent_trace", {})
    critic = trace.get("critic_verdict", {})
    if row.get("design_version") != design:
        raise SystemExit(f"Wrong design in {source_id}")
    if row.get("contrast_schema_version") != schema:
        raise SystemExit(f"Wrong contrast schema in {source_id}")
    if generation.get("surface_renderer") != renderer:
        raise SystemExit(f"Wrong renderer in {source_id}")
    if generation.get("mapping_scope") != mapping:
        raise SystemExit(f"Wrong mapping scope in {source_id}")
    if generation.get("response_contract_policy") != policy:
        raise SystemExit(f"Wrong response policy in {source_id}")
    if generation.get("agent_protocol_version") != agent:
        raise SystemExit(f"Missing Agent protocol in {source_id}")
    if generation.get("semantic_brief_schema_version") != brief_schema:
        raise SystemExit(f"Wrong semantic brief schema in {source_id}")
    if generation.get("independent_row_critic") is not True:
        raise SystemExit(f"Independent critic missing in {source_id}")
    if trace.get("agent_protocol_version") != agent:
        raise SystemExit(f"Missing Agent trace in {source_id}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(trace.get("context_digest", ""))):
        raise SystemExit(f"Invalid context digest in {source_id}")
    if trace.get("critic_independent_call") is not True:
        raise SystemExit(f"Critic provenance missing in {source_id}")
    if critic.get("accepted") is not True:
        raise SystemExit(f"Unapproved Agent row {source_id}")
    if any(critic.get(field) is not True for field in critic_fields):
        raise SystemExit(f"Incomplete critic checks in {source_id}")
    if any(row.get("semantic_judge", {}).get(field) is not True for field in block_judge_fields):
        raise SystemExit(f"Unapproved final block-judge row {source_id}")
    provenance = row.get("question_rewrite_provenance", {})
    if provenance.get("validation_role") != "audit_only":
        raise SystemExit(f"Question diff became a semantic gate in {source_id}")

if profiles.get("design_version") != design:
    raise SystemExit("Profile artifact is not semantic Agent V5.9")
if profiles.get("contrast_schema_version") != schema:
    raise SystemExit("Profile artifact has wrong V5.9 schema")
if profiles.get("agent_protocol_version") != agent:
    raise SystemExit("Profile artifact lacks Agent protocol")
if profiles.get("semantic_brief_schema_version") != brief_schema:
    raise SystemExit("Profile artifact has wrong semantic brief schema")
if profiles.get("critic_fields") != list(critic_fields):
    raise SystemExit("Profile artifact has wrong critic contract")
if len(profiles.get("profiles", [])) != expected_blocks:
    raise SystemExit("V5.9 profile block count mismatch")

print(
    f"Author semantic Agent V5.9 hard gate OK: rows={len(rows)} "
    f"blocks={len(blocks)} per_row_critic=approved final_judge=approved"
)
PY

echo "[3/3] V5.9 generation and audit complete"
echo "Human review: $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "Risk review : $AUDIT_DIR/HUMAN_AUDIT_RISK.md"

if [ "${STOP_AFTER_AUDIT:-true}" = "true" ]; then
    echo "STOP_AFTER_AUDIT=true; no assistant was trained."
    exit 0
fi
if [ "$BLOCK_COUNT" -ne 10 ]; then
    echo "Refusing training on a smoke subset; rerun with F2D_V59_BLOCK_IDS=all." >&2
    exit 2
fi
if [ "${AUDIT_APPROVED:-false}" != "true" ]; then
    echo "Refusing training until AUDIT_APPROVED=true after human review." >&2
    exit 2
fi

TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
agent_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
agent_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60 \
agent_s72_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.8}" WEIGHT_A2="${WEIGHT_A2:-1.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0004}" EVAL_BS="${EVAL_BS:-4}" \
    F2R_VARIANT="F2D-AuthorSemanticAgent200-v5.9" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
