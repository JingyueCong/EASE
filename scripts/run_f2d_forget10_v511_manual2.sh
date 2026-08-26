#!/usr/bin/env bash
# Append-only repair/export of exactly two failed forget10 V5.11 placebo rows.
set -euo pipefail

ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
WORK="$ROOT/ULD/data/ciru/forget10_incremental_v1"
MANIFEST="$WORK/tofu_forget10_author_blocks_exact_v1.json"
BASE_STATE="$WORK/forget10_author_semantic_agent200_seed42_v5_9_new_manualfix_v1.jsonl.blocks"
SOURCE_STATE="$WORK/forget10_author_premisefix200_seed42_v5_11_new_manualfix_v1.jsonl.blocks"
OUTPUT="$WORK/forget10_author_premisefix200_seed42_v5_11_new_manualfix_v2.jsonl"
OUTPUT_STATE="$OUTPUT.blocks"
OUTPUT_PROFILES="$OUTPUT.profiles.json"

MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$MODEL}}"
BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$BASE_URL}"
API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$API_KEY_ENV}"

echo "[1/2] Offline append-only repair of exactly two V5.11 placebo pairs"
"$PY" "$ROOT/scripts/apply_f2d_forget10_v511_manual2.py" \
    --manifest "$MANIFEST" \
    --base-state "$BASE_STATE" \
    --source-state "$SOURCE_STATE" \
    --output-state "$OUTPUT_STATE" \
    --generator-model "$MODEL" --judge-model "$JUDGE_MODEL"

echo "[2/2] Export the ten cached blocks (no generation requests)"
"$PY" "$ROOT/ULD/scripts/generate_tofu_author_premisefix_v5_11.py" \
    --base-state-dir "$BASE_STATE" \
    --split forget10_perturbed --manifest "$MANIFEST" \
    --output "$OUTPUT" --profiles-output "$OUTPUT_PROFILES" \
    --state-dir "$OUTPUT_STATE" --block-ids all --seed 42 \
    --model "$MODEL" --judge-model "$JUDGE_MODEL" \
    --base-url "$BASE_URL" --judge-base-url "$JUDGE_BASE_URL" \
    --api-key-env "$API_KEY_ENV" --judge-api-key-env "$JUDGE_API_KEY_ENV" \
    --temperature 1.0 --judge-temperature 1.0 \
    --max-completion-tokens 18000 --block-concurrency 2 \
    --row-concurrency 5 --request-retries 3 --profile-retries 6 \
    --row-retries 4 --judge-rounds 4 --judge-batch-size 5 \
    --json-mode auto

test "$(wc -l < "$OUTPUT" | tr -d ' ')" = 200
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
echo "Forget10 V5.11 manual2 complete: rows=200 repairs=2 regenerated=0"
echo "output=$OUTPUT"
