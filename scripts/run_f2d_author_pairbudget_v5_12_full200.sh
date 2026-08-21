#!/usr/bin/env bash
# Re-audit the frozen V5.11 full-200 artifact under a C11-derived semantic
# information budget, selectively repairing C01 only. Training stays disabled.
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

SEED="${SEED:-42}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"

BASE_DATA="${F2D_V512_BASE_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_premisefix200_seed${SEED}_v5_11_manualfix_v1_full.jsonl}"
BASE_PROFILES="${F2D_V512_BASE_PROFILES:-${BASE_DATA}.profiles.json}"
BASE_HASH_FILE="${F2D_V512_BASE_HASH_FILE:-${BASE_DATA}.sha256}"

OUTPUT_DATA="${F2D_V512_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairbudget200_seed${SEED}_v5_12_full.jsonl}"
OUTPUT_PROFILES="${F2D_V512_PROFILES_PATH:-${OUTPUT_DATA}.profiles.json}"
STATE_DIR="${F2D_V512_STATE_DIR:-${OUTPUT_DATA}.blocks}"
AUDIT_DIR="${F2D_V512_AUDIT_DIR:-${EASE_ROOT}/audits/forget05_author_pairbudget200_seed${SEED}_v5_12_full}"
OUTPUT_HASH_FILE="${F2D_V512_HASH_FILE:-${OUTPUT_DATA}.sha256}"
PREFLIGHT_LOG="${F2D_V512_PREFLIGHT_LOG:-${STATE_DIR}.preflight.log}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}"
CF_AUDIT_CONCURRENCY="${CF_AUDIT_CONCURRENCY:-4}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_BUDGET_RETRIES="${CF_BUDGET_RETRIES:-3}"
CF_ROW_RETRIES="${CF_ROW_RETRIES:-4}"
CF_JUDGE_ROUNDS="${CF_JUDGE_ROUNDS:-3}"
CF_JUDGE_BATCH_SIZE="${CF_JUDGE_BATCH_SIZE:-5}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-12000}"

for required in "$BASE_DATA" "$BASE_PROFILES" "$BASE_HASH_FILE"; do
    if [ ! -s "$required" ]; then
        echo "Missing frozen V5.11 prerequisite: $required" >&2
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

EXPECTED_BASE_HASH="$(awk 'NR==1 {print $1}' "$BASE_HASH_FILE")"
ACTUAL_BASE_HASH="$(sha256sum "$BASE_DATA" | awk '{print $1}')"
if [ -z "$EXPECTED_BASE_HASH" ] \
    || [ "$ACTUAL_BASE_HASH" != "$EXPECTED_BASE_HASH" ]; then
    echo "Frozen V5.11 base hash mismatch." >&2
    echo "expected=$EXPECTED_BASE_HASH" >&2
    echo "actual=$ACTUAL_BASE_HASH" >&2
    exit 1
fi

echo "============================================================"
echo "F2D pair-budget V5.12 full-200 selective C01 repair"
echo "  immutable base     : $BASE_DATA"
echo "  C11 budget         : relation + scope + polarity + information units"
echo "  ledger role        : support ceiling, not required transcript"
echo "  frozen artifacts   : C11, C10, C00, profiles, fact ledgers"
echo "  mutable artifact   : rejected C01 question/answer only"
echo "  row audit          : independent pair-budget critic"
echo "  final audit        : second independent batched critic"
echo "  output             : $OUTPUT_DATA"
echo "  training           : disabled pending human approval"
echo "============================================================"

echo "[0/3] Offline V5.11/V5.12 preflight"
mkdir -p "$(dirname "$PREFLIGHT_LOG")"
: > "$PREFLIGHT_LOG"
for test_path in \
    tests/test_tofu_anchor_v5_full_preflight.py \
    tests/test_tofu_author_premisefix_v5_11.py \
    tests/test_merge_tofu_premisefix_v5_11_full.py \
    tests/test_tofu_author_pairbudget_v5_12.py
do
    echo "===== $test_path =====" >> "$PREFLIGHT_LOG"
    if "$GEN_PY" "$EASE_ROOT/$test_path" >> "$PREFLIGHT_LOG" 2>&1; then
        echo "preflight_ok test=$(basename "$test_path")"
    else
        echo "Preflight failed: $test_path" >&2
        tail -n 160 "$PREFLIGHT_LOG" >&2
        exit 1
    fi
done

echo "[1/3] Derive C11 budgets, audit inherited C01, repair rejects"
if [ ! -s "$OUTPUT_DATA" ] || [ ! -s "$OUTPUT_PROFILES" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_pairbudget_v5_12.py" \
        --base-data "$BASE_DATA" \
        --base-profiles "$BASE_PROFILES" \
        --output "$OUTPUT_DATA" \
        --profiles-output "$OUTPUT_PROFILES" \
        --state-dir "$STATE_DIR" \
        --model "$CF_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --base-url "$CF_BASE_URL" \
        --judge-base-url "$JUDGE_BASE_URL" \
        --api-key-env "$CF_API_KEY_ENV" \
        --judge-api-key-env "$JUDGE_API_KEY_ENV" \
        --temperature "$CF_TEMPERATURE" \
        --judge-temperature "$JUDGE_TEMPERATURE" \
        --max-completion-tokens "$CF_MAX_COMPLETION_TOKENS" \
        --request-retries "$CF_REQUEST_RETRIES" \
        --budget-retries "$CF_BUDGET_RETRIES" \
        --row-retries "$CF_ROW_RETRIES" \
        --judge-rounds "$CF_JUDGE_ROUNDS" \
        --row-concurrency "$CF_ROW_CONCURRENCY" \
        --audit-concurrency "$CF_AUDIT_CONCURRENCY" \
        --judge-batch-size "$CF_JUDGE_BATCH_SIZE" \
        --json-mode "$CF_JSON_MODE"
else
    echo "Reusing frozen V5.12 output: $OUTPUT_DATA"
fi

echo "[2/3] Full deterministic audit and C01-only hard gate"
"$GEN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$OUTPUT_DATA" \
    --output-dir "$AUDIT_DIR" \
    --expected-units 200 \
    --block-size 20 \
    --sample-count 40 \
    --seed "$SEED" \
    --fail-on-deterministic-errors

"$GEN_PY" - "$BASE_DATA" "$BASE_PROFILES" "$OUTPUT_DATA" \
    "$OUTPUT_PROFILES" "$AUDIT_DIR/SUMMARY.json" <<'PY'
import hashlib
import json
import sys


def load_jsonl(path):
    return {
        row["source_id"]: row
        for row in (
            json.loads(line) for line in open(path) if line.strip()
        )
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


base_path, base_profiles_path, output_path, output_profiles_path, summary_path = (
    sys.argv[1:]
)
base = load_jsonl(base_path)
output = load_jsonl(output_path)
base_profiles = json.load(open(base_profiles_path))
profiles = json.load(open(output_profiles_path))
summary = json.load(open(summary_path))
audit_fields = (
    "pair_budget_faithful_to_c11", "same_target_relation",
    "question_scope_matched",
    "answer_addresses_question", "answerability_matched",
    "polarity_matched", "information_budget_matched",
    "replacement_answer_sufficient", "replacement_fact_supported",
    "source_fact_removed", "no_extraneous_profile_dump", "natural_surface",
)
errors = []
if len(base) != 200 or len(output) != 200 or set(base) != set(output):
    errors.append("row coverage")
if summary.get("records") != 200 or summary.get("unique_source_ids") != 200:
    errors.append("audit row coverage")
if summary.get("duplicate_source_ids") != 0:
    errors.append("duplicate source IDs")
if summary.get("missing_source_indices") or summary.get("unexpected_source_indices"):
    errors.append("source index coverage")
if summary.get("units_with_deterministic_errors") != 0:
    errors.append("deterministic audit errors")

changed = []
for source_id in sorted(base):
    old, new = base[source_id], output[source_id]
    for cell in ("C11", "C10", "C00"):
        if old["cells"][cell] != new["cells"][cell]:
            errors.append(f"frozen cell changed {source_id}.{cell}")
    if old["cells"]["C01"] != new["cells"]["C01"]:
        changed.append(source_id)
    if new.get("design_version") != "tofu-author-pairbudget-v5.12":
        errors.append(f"wrong design {source_id}")
        continue
    generation = new.get("generation", {})
    trace = new.get("pair_budget_trace", {})
    if generation.get("c01_only_revision") is not True:
        errors.append(f"missing C01-only provenance {source_id}")
    if generation.get("profile_and_ledger_frozen") is not True:
        errors.append(f"profile not frozen {source_id}")
    if trace.get("frozen_cells") != ["C11", "C10", "C00"]:
        errors.append(f"frozen-cell trace {source_id}")
    if trace.get("pair_budget_version") != "c11-semantic-information-budget-v2":
        errors.append(f"stale pair budget {source_id}")
    if trace.get("audit_version") != "independent-pair-budget-audit-v2":
        errors.append(f"stale pair audit {source_id}")
    if trace.get("pair_policy_version") != "replacement-identity-ledger-authority-v2":
        errors.append(f"stale pair policy {source_id}")
    for stage in ("row_pair_audit", "final_pair_audit"):
        verdict = trace.get(stage, {})
        if verdict.get("accepted") is not True or any(
            verdict.get(field) is not True for field in audit_fields
        ):
            errors.append(f"incomplete {stage} {source_id}")

revision = profiles.get("pairbudget_revision", {})
if profiles.get("design_version") != "tofu-author-pairbudget-v5.12":
    errors.append("profile design")
if profiles.get("pair_budget_version") != "c11-semantic-information-budget-v2":
    errors.append("profile pair-budget version")
if profiles.get("audit_version") != "independent-pair-budget-audit-v2":
    errors.append("profile audit version")
if profiles.get("pair_policy_version") != "replacement-identity-ledger-authority-v2":
    errors.append("profile pair-policy version")
if profiles.get("profiles") != base_profiles.get("profiles"):
    errors.append("replacement profiles or ledgers changed")
if revision.get("base_data_sha256") != sha256(base_path):
    errors.append("base data digest provenance")
if revision.get("base_profiles_sha256") != sha256(base_profiles_path):
    errors.append("base profiles digest provenance")
if revision.get("output_data_sha256") != sha256(output_path):
    errors.append("output digest provenance")
if revision.get("repaired_c01") != len(changed):
    errors.append("repaired C01 count provenance")
if revision.get("inherited_c01") != 200 - len(changed):
    errors.append("inherited C01 count provenance")
if revision.get("human_review_status") != "pending":
    errors.append("human review status")

if errors:
    raise SystemExit(
        "V5.12 full-200 hard gate failed: " + "; ".join(errors[:30])
    )
print(
    "V5.12 full-200 hard gate OK: rows=200 frozen=3/4 "
    f"inherited_c01={200-len(changed)} repaired_c01={len(changed)}"
)
PY

sha256sum "$OUTPUT_DATA" | tee "$OUTPUT_HASH_FILE"
echo "V5.12 data : $OUTPUT_DATA"
echo "Audit       : $AUDIT_DIR/SUMMARY.md"
echo "Human random: $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "Human risk  : $AUDIT_DIR/HUMAN_AUDIT_RISK.md"
echo "No assistant was trained; approve the V5.12 full-200 human audit first."
