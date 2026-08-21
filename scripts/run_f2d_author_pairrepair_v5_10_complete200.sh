#!/usr/bin/env bash
# Generate only the eight missing TOFU author blocks, then merge with an
# explicitly approved V5.10 manualfix 40-row partition into a full 200 rows.
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

GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
SEED="${SEED:-42}"
REMAINING_BLOCK_IDS="${REMAINING_BLOCK_IDS:-0,2,3,5,6,7,8,9}"
REMAINING_TAG="blocks${REMAINING_BLOCK_IDS//,/_}"

"${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}" - \
    "$REMAINING_BLOCK_IDS" <<'PY'
import sys

actual = {int(value) for value in sys.argv[1].split(",") if value.strip()}
expected = {0, 2, 3, 5, 6, 7, 8, 9}
if actual != expected:
    raise SystemExit(
        "REMAINING_BLOCK_IDS must be exactly the complement of approved "
        f"blocks 1,4; got={sorted(actual)} expected={sorted(expected)}"
    )
PY

APPROVED40_DATA="${APPROVED40_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairrepair40_seed${SEED}_v5_10_manualfix_v1_blocks1_4.jsonl}"
APPROVED40_PROFILES="${APPROVED40_PROFILES:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairrepair40_seed${SEED}_v5_10_blocks1_4.jsonl.profiles.json}"
APPROVED40_HASH_FILE="${APPROVED40_HASH_FILE:-${APPROVED40_DATA}.sha256}"

V59_DATA="${V59_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_semantic_agent160_seed${SEED}_v5_9_${REMAINING_TAG}.jsonl}"
V59_PROFILES="${V59_PROFILES:-${V59_DATA}.profiles.json}"
V59_STATE="${V59_STATE:-${V59_DATA}.blocks}"
V59_AUDIT="${V59_AUDIT:-${EASE_ROOT}/audits/forget05_author_semantic_agent160_seed${SEED}_v5_9_${REMAINING_TAG}}"

V510_DATA="${V510_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairrepair160_seed${SEED}_v5_10_${REMAINING_TAG}.jsonl}"
V510_PROFILES="${V510_PROFILES:-${V510_DATA}.profiles.json}"
V510_STATE="${V510_STATE:-${V510_DATA}.blocks}"
V510_AUDIT="${V510_AUDIT:-${EASE_ROOT}/audits/forget05_author_pairrepair160_seed${SEED}_v5_10_${REMAINING_TAG}}"
REMAINING_REPAIR_MANIFEST="${REMAINING_REPAIR_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_v5_10_remaining160_repair_audit.json}"

FULL_DATA="${FULL_DATA:-${EASE_ROOT}/ULD/data/ciru/forget05_author_pairrepair200_seed${SEED}_v5_10_manualfix_v1_full.jsonl}"
FULL_PROFILES="${FULL_PROFILES:-${FULL_DATA}.profiles.json}"
FULL_AUDIT="${FULL_AUDIT:-${EASE_ROOT}/audits/forget05_author_pairrepair200_seed${SEED}_v5_10_manualfix_v1_full}"
FULL_HASH_FILE="${FULL_HASH_FILE:-${FULL_DATA}.sha256}"

for required in "$APPROVED40_DATA" "$APPROVED40_PROFILES" \
    "$APPROVED40_HASH_FILE" "$REMAINING_REPAIR_MANIFEST"
do
    if [ ! -s "$required" ]; then
        echo "Missing required frozen artifact: $required" >&2
        exit 1
    fi
done

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
echo "F2D V5.10 complete-200 continuation"
echo "  approved/frozen   : blocks 1,4 (40 rows)"
echo "  generate now      : blocks $REMAINING_BLOCK_IDS (160 rows)"
echo "  approved sha256   : $ACTUAL_APPROVED_HASH"
echo "  full output       : $FULL_DATA"
echo "  training          : disabled pending full human audit"
echo "============================================================"

echo "[1/4] Generate/reuse V5.9 semantic-agent data for missing 160 rows"
env \
    ENV_FILE="$ENV_FILE" \
    F2D_V59_BLOCK_IDS="$REMAINING_BLOCK_IDS" \
    F2D_V59_DATA_PATH="$V59_DATA" \
    F2D_V59_PROFILES_PATH="$V59_PROFILES" \
    F2D_V59_STATE_DIR="$V59_STATE" \
    F2D_V59_AUDIT_DIR="$V59_AUDIT" \
    CF_BLOCK_CONCURRENCY="${CF_BLOCK_CONCURRENCY:-2}" \
    CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}" \
    STOP_AFTER_AUDIT=true \
    bash "$EASE_ROOT/scripts/run_f2d_author_semantic_agent_v5_9.sh"

echo "[2/4] Pair-audit/repair only the missing 160 rows with V5.10"
env \
    ENV_FILE="$ENV_FILE" \
    F2D_V510_BLOCK_IDS="$REMAINING_BLOCK_IDS" \
    F2D_V510_BASE_DATA="$V59_DATA" \
    F2D_V510_BASE_PROFILES="$V59_PROFILES" \
    F2D_V510_BASE_STATE_DIR="$V59_STATE" \
    F2D_V510_REPAIR_MANIFEST="$REMAINING_REPAIR_MANIFEST" \
    F2D_V510_DATA_PATH="$V510_DATA" \
    F2D_V510_PROFILES_PATH="$V510_PROFILES" \
    F2D_V510_STATE_DIR="$V510_STATE" \
    F2D_V510_AUDIT_DIR="$V510_AUDIT" \
    CF_BLOCK_CONCURRENCY="${CF_BLOCK_CONCURRENCY:-2}" \
    CF_ROW_CONCURRENCY="${CF_ROW_CONCURRENCY:-5}" \
    bash "$EASE_ROOT/scripts/run_f2d_author_pairrepair_v5_10.sh"

echo "[3/4] Merge approved 40 + generated 160 without editing causal cells"
"$GEN_PY" "$EASE_ROOT/ULD/scripts/merge_tofu_pairrepair_v5_10_full.py" \
    --approved-data "$APPROVED40_DATA" \
    --approved-profiles "$APPROVED40_PROFILES" \
    --remaining-data "$V510_DATA" \
    --remaining-profiles "$V510_PROFILES" \
    --output "$FULL_DATA" \
    --profiles-output "$FULL_PROFILES" \
    --approved-blocks 1,4 \
    --expected-units 200 \
    --block-size 20

echo "[4/4] Full 200-row deterministic audit and freeze"
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
blocks = {int(row["block_id"]) for row in rows}
if blocks != set(range(10)):
    errors.append(f"blocks={sorted(blocks)}")
if errors:
    raise SystemExit("Full-200 hard gate failed: " + "; ".join(errors))
print("Full-200 hard gate OK: rows=200 unique=200 blocks=10 deterministic_errors=0")
PY

sha256sum "$FULL_DATA" | tee "$FULL_HASH_FILE"
echo "Full data   : $FULL_DATA"
echo "Full audit  : $FULL_AUDIT/SUMMARY.md"
echo "Human random: $FULL_AUDIT/HUMAN_AUDIT_RANDOM.md"
echo "Human risk  : $FULL_AUDIT/HUMAN_AUDIT_RISK.md"
echo "No assistant was trained; approve the full-200 human audit first."
