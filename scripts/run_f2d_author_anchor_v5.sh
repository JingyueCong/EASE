#!/usr/bin/env bash
# V5.1: frozen anchor-id planning -> deterministic rendering -> audit -> optional training.
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

SPLIT="${F2D_V5_SPLIT:-forget05}"
DATASET_SPLIT="${F2D_V5_DATASET_SPLIT:-${SPLIT}_perturbed}"
SEED="${F2D_V5_SEED:-42}"
UNITS=200
GPUS="${F2D_V5_GPUS:-0 1 2 3}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${F2D_V5_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"
DATA_PATH="${F2D_V5_DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_anchor${UNITS}_seed${SEED}_v5_1.jsonl}"
PROFILES_PATH="${F2D_V5_PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${F2D_V5_STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${F2D_V5_AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_anchor${UNITS}_seed${SEED}_v5_1}"
SWEEP_NAME="${F2D_V5_SWEEP_NAME:-f2d_author_anchor${UNITS}_v5_1_seed${SEED}}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_CONCURRENCY="${CF_CONCURRENCY:-2}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_STAGE_RETRIES="${CF_STAGE_RETRIES:-8}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-16000}"

if [ "$SPLIT" != "forget05" ]; then
    echo "Anchor V5 currently requires F2D_V5_SPLIT=forget05." >&2
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
echo "F2D author anchor V5.1 (typed anchors; explicit row policy)"
echo "  split / units      : $SPLIT / $UNITS"
echo "  immutable C11      : 10 authors x 20 TOFU QA"
echo "  planner output     : typed group values + row target_group_ids"
echo "  C01 renderer       : frozen occurrence offsets + identity binding"
echo "  C10/C00 renderer   : frozen professional placebo library"
echo "  factorial JSONL    : $DATA_PATH"
echo "  resumable state    : $STATE_DIR"
echo "  legacy preservation: FullAnswer and V1-V5 untouched"
echo "============================================================"

echo "[0/3] Offline 200-row frozen-anchor preflight"
"$GEN_PY" "$EASE_ROOT/tests/test_tofu_anchor_v5_full_preflight.py"

echo "[1/3] Generate anchor-planned author-level causal units"
if [ ! -s "$DATA_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_anchor_v5.py" \
        --split "$DATASET_SPLIT" \
        --manifest "$MANIFEST" \
        --output "$DATA_PATH" \
        --profiles-output "$PROFILES_PATH" \
        --state-dir "$STATE_DIR" \
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
        --concurrency "$CF_CONCURRENCY" \
        --request-retries "$CF_REQUEST_RETRIES" \
        --stage-retries "$CF_STAGE_RETRIES" \
        --json-mode "$CF_JSON_MODE"
else
    echo "Reusing frozen anchor V5.1 data: $DATA_PATH"
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

"$GEN_PY" - "$DATA_PATH" "$PROFILES_PATH" "$MANIFEST" <<'PY'
import json
import re
import sys
from collections import Counter, defaultdict

data_path, profiles_path, manifest_path = sys.argv[1:]
with open(data_path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
with open(profiles_path, encoding="utf-8") as handle:
    profiles = json.load(handle)
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
expected = {item["block_id"]: item["canonical_name"] for item in manifest["authors"]}
judge_fields = (
    "target_relation_match", "target_fact_changed",
    "profile_consistent", "natural_surface",
)
by_block = defaultdict(list)
for row in rows:
    if row.get("design_version") != "tofu-author-anchor-v5.1":
        raise SystemExit(f"Unexpected design in {row.get('source_id')}")
    if row.get("generation", {}).get("surface_renderer") != "deterministic-anchor-id-v5.1":
        raise SystemExit(f"Non-anchor renderer in {row.get('source_id')}")
    if any(row.get("semantic_judge", {}).get(field) is not True for field in judge_fields):
        raise SystemExit(f"Unapproved semantic row {row.get('source_id')}")
    required = row.get("fact_change_required")
    groups = row.get("target_group_ids")
    policy = row.get("intervention_policy")
    if not isinstance(required, bool) or not isinstance(groups, list):
        raise SystemExit(f"Missing V5.1 row policy in {row.get('source_id')}")
    if required and (not groups or policy != "factual_anchor_change"):
        raise SystemExit(f"Invalid factual policy in {row.get('source_id')}")
    if not required and (groups or policy not in {
        "identity_binding", "identity_binding_with_unavailability_preserved",
    }):
        raise SystemExit(f"Invalid exempt policy in {row.get('source_id')}")
    if not required and set(row.get("semantic_judge", {}).get(
        "deterministic_overrides", []
    )) != {"target_fact_changed"}:
        raise SystemExit(f"Unaudited deterministic override in {row.get('source_id')}")
    by_block[row["block_id"]].append(row)
if len(rows) != 200 or set(by_block) != set(expected):
    raise SystemExit("Anchor V5 coverage does not equal the frozen 200-row manifest")
for block_id, block_rows in sorted(by_block.items()):
    if len(block_rows) != 20:
        raise SystemExit(f"Block {block_id} has {len(block_rows)} rows")
    if len({row["replacement_entity"].casefold() for row in block_rows}) != 1:
        raise SystemExit(f"Block {block_id} has multiple replacement authors")
    counts = Counter(row["placebo_relation"] for row in block_rows)
    if len(counts) != 20 or set(counts.values()) != {1}:
        raise SystemExit(f"Block {block_id} placebo library is not one-to-one")
if profiles.get("design_version") != "tofu-author-anchor-v5.1":
    raise SystemExit("Profile ledger is not anchor V5.1")
profile_rows = profiles.get("profiles", [])
if len(profile_rows) != 10:
    raise SystemExit(f"Profile ledger has {len(profile_rows)} blocks, expected 10")
for profile in profile_rows:
    if not re.fullmatch(r"[0-9a-f]{64}", profile.get("anchor_catalog_digest", "")):
        raise SystemExit(f"Invalid anchor digest in block {profile.get('block_id')}")
print("Anchor V5.1 hard gate OK: rows=200 blocks=10 judges=all-pass anchors=typed")
PY

echo "[3/3] V5.1 generation and audit complete"
echo "Human review: $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "Risk review : $AUDIT_DIR/HUMAN_AUDIT_RISK.md"

if [ "${STOP_AFTER_AUDIT:-true}" = "true" ]; then
    echo "STOP_AFTER_AUDIT=true; no assistant was trained."
    exit 0
fi
if [ "${AUDIT_APPROVED:-false}" != "true" ]; then
    echo "Refusing training until AUDIT_APPROVED=true after human review." >&2
    exit 2
fi

TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
anchor_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
anchor_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60 \
anchor_s72_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72 \
anchor_s72_u0p5:2:2:16:16:1e-3:1e-3:1:1:0.5:0.5:72:72}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.8}" WEIGHT_A2="${WEIGHT_A2:-1.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0004}" EVAL_BS="${EVAL_BS:-4}" \
    F2R_VARIANT="F2D-AuthorAnchor200-v5.1" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
