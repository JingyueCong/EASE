#!/usr/bin/env bash
# Author-profile v2 generation, audit, and optional FullAnswer F2D training.
# Legacy row-wise data and scripts/run_f2d_did200_full.sh are intentionally untouched.
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
DATASET_SPLIT="${DATASET_SPLIT:-${SPLIT}_perturbed}"
SEED="${SEED:-42}"
UNITS="${UNITS:-200}"
BLOCK_SIZE="${BLOCK_SIZE:-20}"
GPUS="${GPUS:-0 1 2 3}"
GEN_PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
MANIFEST="${AUTHOR_MANIFEST:-${EASE_ROOT}/ULD/configs/data/tofu_forget05_author_blocks.json}"
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_twin${UNITS}_seed${SEED}_v2.jsonl}"
PROFILES_PATH="${PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_twin${UNITS}_seed${SEED}_v2}"
SWEEP_NAME="${SWEEP_NAME:-f2d_author_twin${UNITS}_fullanswer_seed${SEED}}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
CF_CONCURRENCY="${CF_CONCURRENCY:-2}"
CF_RETRIES="${CF_RETRIES:-4}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-16000}"

if [ "$SPLIT" != "forget05" ] || [ "$UNITS" -ne 200 ] || [ "$BLOCK_SIZE" -ne 20 ]; then
    echo "Author-profile v2 currently has a frozen manifest only for forget05: 200 rows, blocks of 20." >&2
    exit 1
fi
if [ -z "$CF_BASE_URL" ]; then
    echo "Set OPENAI_BASE_URL (or CF_BASE_URL) in $ENV_FILE." >&2
    exit 1
fi
if [ -z "${!CF_API_KEY_ENV:-}" ]; then
    echo "Set $CF_API_KEY_ENV in $ENV_FILE." >&2
    exit 1
fi

echo "============================================================"
echo "F2D author-profile v2 (legacy FullAnswer preserved)"
echo "  split / units      : $SPLIT / $UNITS"
echo "  experimental units : 10 author blocks x 20 QA"
echo "  canonical manifest : $MANIFEST"
echo "  factorial JSONL    : $DATA_PATH"
echo "  profile ledger     : $PROFILES_PATH"
echo "  audit reports      : $AUDIT_DIR"
echo "  generator          : $CF_MODEL ($CF_CONCURRENCY author blocks concurrently)"
echo "  default action     : generate + audit, then stop"
echo "============================================================"

if [ ! -s "$DATA_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_factorial.py" \
        --split "$DATASET_SPLIT" \
        --manifest "$MANIFEST" \
        --output "$DATA_PATH" \
        --profiles-output "$PROFILES_PATH" \
        --state-dir "$STATE_DIR" \
        --seed "$SEED" \
        --model "$CF_MODEL" \
        --base-url "$CF_BASE_URL" \
        --api-key-env "$CF_API_KEY_ENV" \
        --temperature "$CF_TEMPERATURE" \
        --max-completion-tokens "$CF_MAX_COMPLETION_TOKENS" \
        --concurrency "$CF_CONCURRENCY" \
        --retries "$CF_RETRIES" \
        --json-mode "$CF_JSON_MODE"
else
    echo "Reusing frozen author-profile data: $DATA_PATH"
fi

"$GEN_PY" "$EASE_ROOT/scripts/audit_tofu_factorial.py" \
    --input "$DATA_PATH" \
    --output-dir "$AUDIT_DIR" \
    --expected-units "$UNITS" \
    --block-size "$BLOCK_SIZE" \
    --sample-count 20 \
    --seed "$SEED" \
    --fail-on-deterministic-errors

"$GEN_PY" - "$DATA_PATH" "$PROFILES_PATH" "$MANIFEST" <<'PY'
import json
import sys
from collections import defaultdict

data_path, profiles_path, manifest_path = sys.argv[1:]
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
profiles = json.load(open(profiles_path, encoding="utf-8"))
manifest = json.load(open(manifest_path, encoding="utf-8"))
expected_authors = {
    item["block_id"]: item["canonical_name"] for item in manifest["authors"]
}

by_block = defaultdict(list)
for row in rows:
    if row.get("design_version") != "tofu-author-profile-v2":
        raise SystemExit(f"Unexpected design_version in {row.get('source_id')}")
    by_block[row["block_id"]].append(row)

if set(by_block) != set(expected_authors):
    raise SystemExit("Author block coverage does not match the frozen manifest")
for block, block_rows in sorted(by_block.items()):
    if len(block_rows) != manifest["block_size"]:
        raise SystemExit(f"Block {block} has {len(block_rows)} rows")
    targets = {row["target_entity"].casefold() for row in block_rows}
    twins = {row["replacement_entity"].casefold() for row in block_rows}
    profile_ids = {row["profile_id"] for row in block_rows}
    if targets != {expected_authors[block].casefold()}:
        raise SystemExit(f"Block {block} target mismatch: {targets}")
    if len(twins) != 1 or len(profile_ids) != 1:
        raise SystemExit(f"Block {block} does not use one coherent twin profile")
if len(profiles.get("profiles", [])) != len(expected_authors):
    raise SystemExit("Profile ledger does not contain exactly one entry per author")
print(f"Author-profile gate OK: rows={len(rows)} blocks={len(by_block)} profiles={len(profiles['profiles'])}")
PY

echo
echo "Generation and deterministic audit completed."
echo "Review both files before training:"
echo "  $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "  $AUDIT_DIR/HUMAN_AUDIT_RISK.md"

if [ "${STOP_AFTER_AUDIT:-true}" = "true" ]; then
    echo "STOP_AFTER_AUDIT=true; no assistant was trained."
    echo "The legacy row-wise FullAnswer data and code remain unchanged."
    exit 0
fi
if [ "${AUDIT_APPROVED:-false}" != "true" ]; then
    echo "Refusing to train unreviewed generated data. Set AUDIT_APPROVED=true after human audit." >&2
    exit 2
fi

# FullAnswer is intentionally the existing f2d_did_a1/f2d_did_a2 adapter.
# No old FullAnswer implementation or artifact is replaced.
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
author_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
author_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60 \
author_s72_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72 \
author_s72_u0p5:2:2:16:16:1e-3:1e-3:1:1:0.5:0.5:72:72}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.8}" WEIGHT_A2="${WEIGHT_A2:-1.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0004}" EVAL_BS="${EVAL_BS:-4}" \
    F2R_VARIANT="F2D-AuthorTwin200-FullAnswer-v2" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
