#!/usr/bin/env bash
# Re-audit the missing 160 rows under V5.11, then merge with approved V5.10
# blocks 1/4 into a frozen full-200 artifact.  Training remains disabled.
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

SPLIT="${SPLIT:-forget05}"
SEED="${SEED:-42}"
BLOCK_IDS="${F2D_V511_BLOCK_IDS:-0,2,3,5,6,7,8,9}"
RUN_TAG="blocks${BLOCK_IDS//,/_}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${F2D_V511_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"
HUMAN_REPAIR_MANIFEST="${F2D_V511_HUMAN_REPAIR_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_v5_11_block2_human_repairs.json}"

"$GEN_PY" - "$BLOCK_IDS" <<'PY'
import sys

actual = {int(value) for value in sys.argv[1].split(",") if value.strip()}
expected = {0, 2, 3, 5, 6, 7, 8, 9}
if actual != expected:
    raise SystemExit(
        "F2D_V511_BLOCK_IDS must be exactly the complement of approved "
        f"blocks 1,4; got={sorted(actual)} expected={sorted(expected)}"
    )
PY

BASE_V59_STATE="${BASE_V59_STATE:-${EASE_ROOT}/ULD/data/ciru/forget05_author_semantic_agent160_seed${SEED}_v5_9_${RUN_TAG}.jsonl.blocks}"
APPROVED40_DATA="${APPROVED40_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairrepair40_seed${SEED}_v5_10_manualfix_v1_blocks1_4.jsonl}"
APPROVED40_PROFILES="${APPROVED40_PROFILES:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairrepair40_seed${SEED}_v5_10_blocks1_4.jsonl.profiles.json}"
APPROVED40_HASH_FILE="${APPROVED40_HASH_FILE:-${APPROVED40_DATA}.sha256}"

REMAINDER_DATA="${F2D_V511_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget05_author_premisefix160_seed${SEED}_v5_11_${RUN_TAG}.jsonl}"
REMAINDER_PROFILES="${F2D_V511_PROFILES_PATH:-${REMAINDER_DATA}.profiles.json}"
REMAINDER_STATE="${F2D_V511_STATE_DIR:-${REMAINDER_DATA}.blocks}"
REMAINDER_AUDIT="${F2D_V511_AUDIT_DIR:-${EASE_ROOT}/audits/forget05_author_premisefix160_seed${SEED}_v5_11_${RUN_TAG}}"

FULL_DATA="${FULL_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_premisefix200_seed${SEED}_v5_11_manualfix_v1_full.jsonl}"
FULL_PROFILES="${FULL_PROFILES:-${FULL_DATA}.profiles.json}"
FULL_AUDIT="${FULL_AUDIT:-${EASE_ROOT}/audits/forget05_author_premisefix200_seed${SEED}_v5_11_manualfix_v1_full}"
FULL_HASH_FILE="${FULL_HASH_FILE:-${FULL_DATA}.sha256}"
PREFLIGHT_LOG="${F2D_V511_PREFLIGHT_LOG:-${REMAINDER_STATE}.preflight.log}"

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

for required in "$BASE_V59_STATE" "$APPROVED40_DATA" \
    "$APPROVED40_PROFILES" "$APPROVED40_HASH_FILE" "$MANIFEST" \
    "$HUMAN_REPAIR_MANIFEST"
do
    if [ ! -e "$required" ]; then
        echo "Missing frozen prerequisite: $required" >&2
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

EXPECTED_APPROVED_HASH="$(awk 'NR==1 {print $1}' "$APPROVED40_HASH_FILE")"
ACTUAL_APPROVED_HASH="$(sha256sum "$APPROVED40_DATA" | awk '{print $1}')"
if [ -z "$EXPECTED_APPROVED_HASH" ] \
    || [ "$ACTUAL_APPROVED_HASH" != "$EXPECTED_APPROVED_HASH" ]; then
    echo "Approved 40-row artifact hash mismatch." >&2
    echo "expected=$EXPECTED_APPROVED_HASH" >&2
    echo "actual=$ACTUAL_APPROVED_HASH" >&2
    exit 1
fi

echo "============================================================"
echo "F2D premise-map V5.11 complete-200 continuation"
echo "  approved/frozen   : V5.10 blocks 1,4 (40 rows)"
echo "  re-audit now      : V5.9-state blocks $BLOCK_IDS (160 rows)"
echo "  premise policy    : preserve relation; replace author-specific facts"
echo "  selective repair  : generate only missing/rejected C01 rows"
echo "  explicit repairs  : $HUMAN_REPAIR_MANIFEST"
echo "  base state        : $BASE_V59_STATE"
echo "  full output       : $FULL_DATA"
echo "  training          : disabled pending full human audit"
echo "============================================================"

echo "[0/4] Offline V5.9-V5.11 and merge preflight"
mkdir -p "$(dirname "$PREFLIGHT_LOG")"
: > "$PREFLIGHT_LOG"
for test_path in \
    tests/test_tofu_anchor_v5_full_preflight.py \
    tests/test_tofu_author_semantic_agent_v5_9.py \
    tests/test_tofu_author_pairrepair_v5_10.py \
    tests/test_tofu_author_premisefix_v5_11.py \
    tests/test_merge_tofu_premisefix_v5_11_full.py
do
    echo "===== $test_path =====" >> "$PREFLIGHT_LOG"
    if "$GEN_PY" "$EASE_ROOT/$test_path" >> "$PREFLIGHT_LOG" 2>&1; then
        echo "preflight_ok test=$(basename "$test_path")"
    else
        echo "Preflight failed: $test_path" >&2
        tail -n 140 "$PREFLIGHT_LOG" >&2
        exit 1
    fi
done

echo "[1/4] Re-audit inherited V5.9 rows under premise-map V5.11"
if [ ! -s "$REMAINDER_DATA" ] || [ ! -s "$REMAINDER_PROFILES" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_premisefix_v5_11.py" \
        --base-state-dir "$BASE_V59_STATE" \
        --human-repair-manifest "$HUMAN_REPAIR_MANIFEST" \
        --split "${SPLIT}_perturbed" \
        --manifest "$MANIFEST" \
        --output "$REMAINDER_DATA" \
        --profiles-output "$REMAINDER_PROFILES" \
        --state-dir "$REMAINDER_STATE" \
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
    echo "Reusing frozen V5.11 remainder: $REMAINDER_DATA"
fi

"$GEN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$REMAINDER_DATA" \
    --output-dir "$REMAINDER_AUDIT" \
    --block-size 20 \
    --sample-count 20 \
    --seed "$SEED" \
    --fail-on-deterministic-errors

"$GEN_PY" - "$REMAINDER_DATA" "$REMAINDER_PROFILES" \
    "$HUMAN_REPAIR_MANIFEST" <<'PY'
import json
import hashlib
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
profiles = json.load(open(sys.argv[2]))
repair_manifest = json.load(open(sys.argv[3]))
expected_repairs = set(repair_manifest["repair_rows"])
expected_repair_digest = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
expected_blocks = {0, 2, 3, 5, 6, 7, 8, 9}
critic_fields = (
    "same_target_relation", "question_scope_matched",
    "question_premises_updated", "answer_addresses_question",
    "replacement_fact_expressed", "source_fact_removed",
    "source_comparison_absent", "evidence_status_matched",
    "information_granularity_matched", "response_mode_compatible",
    "natural_surface",
)
judge_fields = (
    "target_relation_match", "question_premise_profile_consistent",
    "answer_satisfies_question", "target_fact_changed",
    "profile_consistent", "natural_surface",
)
errors = []
if len(rows) != 160 or len({row["source_id"] for row in rows}) != 160:
    errors.append("row coverage")
if {int(row["block_id"]) for row in rows} != expected_blocks:
    errors.append("block coverage")
for row in rows:
    source_id = row["source_id"]
    if row.get("design_version") != "tofu-author-premisefix-v5.11":
        errors.append(f"wrong design {source_id}")
        continue
    generation = row.get("generation", {})
    trace = row.get("semantic_agent_trace", {})
    critic = trace.get("critic_verdict", {})
    judge = row.get("semantic_judge", {})
    if generation.get("premise_policy_version") != "source-to-replacement-premise-map-v1":
        errors.append(f"missing premise policy {source_id}")
    if trace.get("premise_policy_version") != "source-to-replacement-premise-map-v1":
        errors.append(f"missing trace policy {source_id}")
    if critic.get("accepted") is not True or any(
        critic.get(field) is not True for field in critic_fields
    ):
        errors.append(f"pair critic incomplete {source_id}")
    if any(judge.get(field) is not True for field in judge_fields):
        errors.append(f"block judge incomplete {source_id}")
    manual = trace.get("human_manual_repair")
    if source_id in expected_repairs:
        if trace.get("generator_model") != "explicit-human-repair":
            errors.append(f"manual generator provenance {source_id}")
        if not isinstance(manual, dict):
            errors.append(f"manual repair provenance {source_id}")
        elif manual.get("manifest_digest") != expected_repair_digest:
            errors.append(f"manual repair digest {source_id}")
    elif manual is not None:
        errors.append(f"unexpected manual repair provenance {source_id}")
if profiles.get("design_version") != "tofu-author-premisefix-v5.11":
    errors.append("profile design")
if profiles.get("selected_block_ids") != sorted(expected_blocks):
    errors.append("profile block coverage")
if set(profiles.get("human_repair_rows", [])) != expected_repairs:
    errors.append("profile human repair coverage")
if profiles.get("human_repair_manifest_digest") != expected_repair_digest:
    errors.append("profile human repair digest")
if errors:
    raise SystemExit("V5.11 remainder hard gate failed: " + "; ".join(errors[:20]))
print(
    "V5.11 remainder hard gate OK: rows=160 blocks=8 "
    f"premise_policy=approved human_repairs={len(expected_repairs)}"
)
PY

echo "[2/4] Merge approved V5.10 40 + V5.11 160 without editing cells"
"$GEN_PY" "$EASE_ROOT/ULD/scripts/merge_tofu_premisefix_v5_11_full.py" \
    --approved-data "$APPROVED40_DATA" \
    --approved-profiles "$APPROVED40_PROFILES" \
    --remainder-data "$REMAINDER_DATA" \
    --remainder-profiles "$REMAINDER_PROFILES" \
    --output "$FULL_DATA" \
    --profiles-output "$FULL_PROFILES" \
    --approved-blocks 1,4 \
    --expected-units 200 \
    --block-size 20

echo "[3/4] Full 200-row deterministic audit and freeze"
"$GEN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$FULL_DATA" \
    --output-dir "$FULL_AUDIT" \
    --expected-units 200 \
    --block-size 20 \
    --sample-count 20 \
    --seed "$SEED" \
    --fail-on-deterministic-errors

"$GEN_PY" - "$FULL_AUDIT/SUMMARY.json" "$FULL_DATA" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
errors = []
if summary.get("records") != 200:
    errors.append(f"records={summary.get('records')}")
if summary.get("unique_source_ids") != 200:
    errors.append(f"unique={summary.get('unique_source_ids')}")
if summary.get("duplicate_source_ids") != 0:
    errors.append(f"duplicates={summary.get('duplicate_source_ids')}")
if summary.get("missing_source_indices"):
    errors.append("missing source indices")
if summary.get("unexpected_source_indices"):
    errors.append("unexpected source indices")
if summary.get("units_with_deterministic_errors") != 0:
    errors.append(
        f"deterministic_errors={summary.get('units_with_deterministic_errors')}"
    )
if {int(row["block_id"]) for row in rows} != set(range(10)):
    errors.append("block coverage")
if any(
    row.get("generation", {}).get("full200_assembly", {}).get(
        "causal_cells_edited_by_merge"
    ) is not False
    for row in rows
):
    errors.append("merge cell-edit provenance")
if errors:
    raise SystemExit("Full-200 V5.11 hard gate failed: " + "; ".join(errors))
print("Full-200 V5.11 hard gate OK: rows=200 unique=200 blocks=10")
PY

sha256sum "$FULL_DATA" | tee "$FULL_HASH_FILE"
echo "Full data   : $FULL_DATA"
echo "Full audit  : $FULL_AUDIT/SUMMARY.md"
echo "Human random: $FULL_AUDIT/HUMAN_AUDIT_RANDOM.md"
echo "Human risk  : $FULL_AUDIT/HUMAN_AUDIT_RISK.md"
echo "No assistant was trained; approve the full-200 human audit first."
