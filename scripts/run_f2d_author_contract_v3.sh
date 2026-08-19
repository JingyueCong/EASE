#!/usr/bin/env bash
# Contract-first author-level V3 generation, audit, and optional FullAnswer training.
# V1 row-wise and V2 author-profile artifacts/runners remain independent.
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
DATA_PATH="${DATA_PATH:-${EASE_ROOT}/ULD/data/ciru/${SPLIT}_author_contract${UNITS}_seed${SEED}_v3.jsonl}"
PROFILES_PATH="${PROFILES_PATH:-${DATA_PATH}.profiles.json}"
STATE_DIR="${STATE_DIR:-${DATA_PATH}.blocks}"
AUDIT_DIR="${AUDIT_DIR:-${EASE_ROOT}/audits/${SPLIT}_author_contract${UNITS}_seed${SEED}_v3}"
SWEEP_NAME="${SWEEP_NAME:-f2d_author_contract${UNITS}_fullanswer_seed${SEED}}"

CF_MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$CF_MODEL}}"
CF_BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$CF_BASE_URL}"
CF_API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$CF_API_KEY_ENV}"
CF_CONCURRENCY="${CF_CONCURRENCY:-2}"
CF_REQUEST_RETRIES="${CF_REQUEST_RETRIES:-3}"
CF_STAGE_RETRIES="${CF_STAGE_RETRIES:-4}"
CF_JSON_MODE="${CF_JSON_MODE:-auto}"
CF_TEMPERATURE="${CF_TEMPERATURE:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
CF_MAX_COMPLETION_TOKENS="${CF_MAX_COMPLETION_TOKENS:-16000}"
RENDER_CHUNK_SIZE="${RENDER_CHUNK_SIZE:-5}"

if [ "$SPLIT" != "forget05" ] || [ "$UNITS" -ne 200 ] || [ "$BLOCK_SIZE" -ne 20 ]; then
    echo "Author-contract V3 currently uses the frozen forget05 manifest: 200 rows, blocks of 20." >&2
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
if [ -z "${!JUDGE_API_KEY_ENV:-}" ]; then
    echo "Set $JUDGE_API_KEY_ENV in $ENV_FILE." >&2
    exit 1
fi

echo "============================================================"
echo "F2D author-contract V3 (Plan -> Contract -> Render -> Judge)"
echo "  split / units      : $SPLIT / $UNITS"
echo "  experimental units : 10 author blocks x 20 immutable C11 QA"
echo "  canonical manifest : $MANIFEST"
echo "  factorial JSONL    : $DATA_PATH"
echo "  resumable state    : $STATE_DIR"
echo "  audit reports      : $AUDIT_DIR"
echo "  generator / judge  : $CF_MODEL / $JUDGE_MODEL"
echo "  block/chunk workers: $CF_CONCURRENCY / $RENDER_CHUNK_SIZE rows"
echo "  placebo contract   : >=10 unique relations, <=2 uses/relation"
echo "  default action     : generate + deterministic audit, then stop"
echo "============================================================"

if [ ! -s "$DATA_PATH" ]; then
    "$GEN_PY" "$EASE_ROOT/ULD/scripts/generate_tofu_author_contract_v3.py" \
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
        --render-chunk-size "$RENDER_CHUNK_SIZE" \
        --json-mode "$CF_JSON_MODE"
else
    echo "Reusing frozen author-contract V3 data: $DATA_PATH"
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
from collections import Counter, defaultdict

data_path, profiles_path, manifest_path = sys.argv[1:]
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
profiles = json.load(open(profiles_path, encoding="utf-8"))
manifest = json.load(open(manifest_path, encoding="utf-8"))
expected_authors = {item["block_id"]: item["canonical_name"] for item in manifest["authors"]}
judge_fields = (
    "target_relation_match", "target_fact_changed", "placebo_parallel",
    "placebo_exclusion", "profile_consistent", "surface_quality",
)

if profiles.get("design_version") != "tofu-author-contract-v3":
    raise SystemExit("Profile ledger is not author-contract V3")
by_block = defaultdict(list)
for row in rows:
    if row.get("design_version") != "tofu-author-contract-v3":
        raise SystemExit(f"Unexpected design_version in {row.get('source_id')}")
    if any(row.get("semantic_judge", {}).get(field) is not True for field in judge_fields):
        raise SystemExit(f"Unapproved semantic judgement in {row.get('source_id')}")
    by_block[row["block_id"]].append(row)

if len(rows) != 200 or set(by_block) != set(expected_authors):
    raise SystemExit("V3 author block coverage does not match the frozen manifest")
for block_id, block_rows in sorted(by_block.items()):
    if len(block_rows) != manifest["block_size"]:
        raise SystemExit(f"Block {block_id} has {len(block_rows)} rows")
    targets = {row["target_entity"].casefold() for row in block_rows}
    replacements = {row["replacement_entity"].casefold() for row in block_rows}
    profile_ids = {row["profile_id"] for row in block_rows}
    placebo_counts = Counter(row["placebo_relation"].casefold().strip() for row in block_rows)
    if targets != {expected_authors[block_id].casefold()}:
        raise SystemExit(f"Block {block_id} target mismatch: {targets}")
    if len(replacements) != 1 or len(profile_ids) != 1:
        raise SystemExit(f"Block {block_id} lacks one coherent replacement profile")
    if len(placebo_counts) < 10 or max(placebo_counts.values()) > 2:
        raise SystemExit(f"Block {block_id} violates placebo diversity: {placebo_counts}")
if len(profiles.get("profiles", [])) != len(expected_authors):
    raise SystemExit("V3 profile ledger must contain one entry per author")
print(f"Author-contract V3 gate OK: rows={len(rows)} blocks={len(by_block)} judges=all-pass")
PY

echo
echo "V3 generation and deterministic audit completed."
echo "Human review is still mandatory before training:"
echo "  $AUDIT_DIR/HUMAN_AUDIT_RANDOM.md"
echo "  $AUDIT_DIR/HUMAN_AUDIT_RISK.md"

if [ "${STOP_AFTER_AUDIT:-true}" = "true" ]; then
    echo "STOP_AFTER_AUDIT=true; no assistant was trained."
    echo "V1/V2 FullAnswer generators, data paths, and checkpoints remain unchanged."
    exit 0
fi
if [ "${AUDIT_APPROVED:-false}" != "true" ]; then
    echo "Refusing to train V3 before human approval. Set AUDIT_APPROVED=true after review." >&2
    exit 2
fi

TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
contract_s48_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:48:48 \
contract_s60_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:60:60 \
contract_s72_u1:2:2:16:16:1e-3:1e-3:1:1:1:1:72:72 \
contract_s72_u0p5:2:2:16:16:1e-3:1e-3:1:1:0.5:0.5:72:72}"

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT="$SPLIT" GPUS="$GPUS" \
    UNITS="$UNITS" SEED="$SEED" VIEWS=1 RESUME="${RESUME:-true}" \
    CIRU_PATH="$DATA_PATH" CF_PATH="$DATA_PATH" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    WEIGHT_A1="${WEIGHT_A1:--1.8}" WEIGHT_A2="${WEIGHT_A2:-1.8}" \
    TOP_FILTER="${TOP_FILTER:-0.0004}" EVAL_BS="${EVAL_BS:-4}" \
    F2R_VARIANT="F2D-AuthorContract200-FullAnswer-v3" \
    bash "$EASE_ROOT/scripts/sweep_f2d_did_training.sh"
