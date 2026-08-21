#!/usr/bin/env bash
# V5.10: re-audit frozen V5.9 C01 rows and selectively repair only failures.
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

SPLIT="${F2D_V510_SPLIT:-forget05}"
DATASET_SPLIT="${F2D_V510_DATASET_SPLIT:-${SPLIT}_perturbed}"
SEED="${F2D_V510_SEED:-42}"
BLOCK_IDS="${F2D_V510_BLOCK_IDS:-1,4}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${F2D_V510_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"

if [ "$BLOCK_IDS" = "all" ] || [ "$BLOCK_IDS" = "*" ]; then
    BLOCK_COUNT=10
    RUN_TAG="full"
else
    IFS=',' read -r -a BLOCK_ARRAY <<< "$BLOCK_IDS"
    BLOCK_COUNT="${#BLOCK_ARRAY[@]}"
    RUN_TAG="blocks${BLOCK_IDS//,/_}"
fi
UNITS=$((BLOCK_COUNT * 20))

BASE_DATA="${F2D_V510_BASE_DATA:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_semantic_agent${UNITS}_seed${SEED}_v5_9_${RUN_TAG}.jsonl}"
BASE_PROFILES="${F2D_V510_BASE_PROFILES:-${BASE_DATA}.profiles.json}"
BASE_STATE_DIR="${F2D_V510_BASE_STATE_DIR:-${BASE_DATA}.blocks}"
REPAIR_MANIFEST="${F2D_V510_REPAIR_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_v5_9_pairrepair_audit.json}"
DATA_PATH="${F2D_V510_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_pairrepair${UNITS}_seed${SEED}_v5_10_${RUN_TAG}.jsonl}"
PROFILES_PATH="${F2D_V510_PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${F2D_V510_STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${F2D_V510_AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_pairrepair${UNITS}_seed${SEED}_v5_10_${RUN_TAG}}"
PREFLIGHT_LOG="${F2D_V510_PREFLIGHT_LOG:-${STATE_DIR}.preflight.log}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_BLOCK_CONCURRENCY="${CF_BLOCK_CONCURRENCY:-2}"
CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_PROFILE_RETRIES="${CF_PROFILE_RETRIES:-1}"
CF_ROW_RETRIES="${CF_ROW_RETRIES:-4}"
CF_JUDGE_ROUNDS="${CF_JUDGE_ROUNDS:-4}"
CF_JUDGE_BATCH_SIZE="${CF_JUDGE_BATCH_SIZE:-5}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-18000}"

for required in "$BASE_DATA" "$BASE_PROFILES" "$BASE_STATE_DIR" "$REPAIR_MANIFEST"; do
    if [ ! -e "$required" ]; then
        echo "Missing frozen V5.9 artifact: $required" >&2
        exit 1
    fi
done
if [ -z "$CF_BASE_URL" ]; then
    echo "Set OPENAI_BASE_URL (or CF_BASE_URL) in $ENV_FILE." >&2
    exit 1
fi
if [ -z "${!CF_API_KEY_ENV:-}" ] || [ -z "${!JUDGE_API_KEY_ENV:-}" ]; then
    echo "Set $CF_API_KEY_ENV and $JUDGE_API_KEY_ENV in $ENV_FILE." >&2
    exit 1
fi

echo "============================================================"
echo "F2D author selective pair repair V5.10"
echo "  split / units      : $SPLIT / $UNITS"
echo "  selected blocks    : $BLOCK_IDS"
echo "  base V5.9 data     : $BASE_DATA"
echo "  frozen reuse       : profile + ledger + C11 + C10 + C00"
echo "  C01 policy         : re-audit all; regenerate rejected rows only"
echo "  pair critic        : relation/scope/evidence/granularity/no-source-comparison"
echo "  human repair audit : $REPAIR_MANIFEST"
echo "  generator / critic : $CF_MODEL / $JUDGE_MODEL"
echo "  output             : $DATA_PATH"
echo "  state              : $STATE_DIR"
echo "  legacy preservation: V5.9 and every earlier design untouched"
echo "============================================================"

echo "[0/3] Offline V5.9 and V5.10 preflight"
mkdir -p "$(dirname "$PREFLIGHT_LOG")"
: > "$PREFLIGHT_LOG"
for test_path in \
    tests/test_tofu_anchor_v5_full_preflight.py \
    tests/test_tofu_author_semantic_agent_v5_9.py \
    tests/test_tofu_author_pairrepair_v5_10.py
do
    echo "===== $test_path =====" >> "$PREFLIGHT_LOG"
    if "$GEN_PY" "$EASE_ROOT/$test_path" >> "$PREFLIGHT_LOG" 2>&1; then
        echo "preflight_ok test=$(basename "$test_path")"
    else
        echo "Preflight failed: $test_path" >&2
        tail -n 120 "$PREFLIGHT_LOG" >&2
        exit 1
    fi
done

echo "[1/3] Re-audit inherited C01 and selectively repair rejected rows"
if [ ! -s "$DATA_PATH" ] || [ ! -s "$PROFILES_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_pairrepair_v5_10.py" \
        --base-data "$BASE_DATA" \
        --base-profiles "$BASE_PROFILES" \
        --base-state-dir "$BASE_STATE_DIR" \
        --repair-manifest "$REPAIR_MANIFEST" \
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
    echo "Reusing frozen V5.10 data: $DATA_PATH"
fi

echo "[2/3] Deterministic audit and selective-repair hard gate"
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

"$GEN_PY" - "$DATA_PATH" "$PROFILES_PATH" "$BASE_DATA" "$REPAIR_MANIFEST" "$UNITS" <<'PY'
import json
import sys

data_path, profiles_path, base_path, repair_path, expected = sys.argv[1:]
expected = int(expected)
rows = [json.loads(line) for line in open(data_path) if line.strip()]
base = {
    row["source_id"]: row
    for row in (json.loads(line) for line in open(base_path) if line.strip())
}
profiles = json.load(open(profiles_path))
required_repairs = set(json.load(open(repair_path))["repair_rows"])
fields = (
    "same_target_relation", "question_scope_matched",
    "question_premises_updated", "answer_addresses_question",
    "replacement_fact_expressed", "source_fact_removed",
    "source_comparison_absent", "evidence_status_matched",
    "information_granularity_matched", "response_mode_compatible",
    "natural_surface",
)
if len(rows) != expected or len({row["source_id"] for row in rows}) != expected:
    raise SystemExit("V5.10 row coverage is incomplete or duplicated")
reused = repaired = 0
for row in rows:
    source_id = row["source_id"]
    if row.get("design_version") != "tofu-author-pairrepair-v5.10":
        raise SystemExit(f"wrong V5.10 design: {source_id}")
    for cell in ("C11", "C10", "C00"):
        if row["cells"][cell] != base[source_id]["cells"][cell]:
            raise SystemExit(f"frozen cell changed: {source_id}.{cell}")
    generation = row.get("generation", {})
    if generation.get("selective_c01_repair") is not True:
        raise SystemExit(f"selective repair provenance missing: {source_id}")
    if generation.get("frozen_cells_reused") != ["C11", "C10", "C00"]:
        raise SystemExit(f"frozen-cell provenance missing: {source_id}")
    critic = row.get("semantic_agent_trace", {}).get("critic_verdict", {})
    if critic.get("accepted") is not True or any(
        critic.get(field) is not True for field in fields
    ):
        raise SystemExit(f"pair critic incomplete: {source_id}")
    if row["cells"]["C01"] == base[source_id]["cells"]["C01"]:
        if source_id in required_repairs:
            raise SystemExit(f"human-required C01 was not repaired: {source_id}")
        reused += 1
    else:
        repaired += 1
if profiles.get("design_version") != "tofu-author-pairrepair-v5.10":
    raise SystemExit("V5.10 profile artifact has wrong design")
if profiles.get("critic_fields") != list(fields):
    raise SystemExit("V5.10 profile artifact has wrong pair-critic fields")
if set(profiles.get("human_repair_rows", [])) != required_repairs:
    raise SystemExit("V5.10 profile artifact lost human repair provenance")
print(
    f"V5.10 hard gate OK: rows={len(rows)} inherited_c01={reused} "
    f"repaired_c01={repaired} frozen_cells=3/4"
)
PY

echo "[3/3] V5.10 generation and audit complete"
echo "Human review: $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "Risk review : $AUDIT_DIR/HUMAN_AUDIT_RISK.md"
echo "No assistant was trained; V5.10 requires explicit human approval."
