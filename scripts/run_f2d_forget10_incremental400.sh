#!/usr/bin/env bash
# Append-only forget10 expansion: exact-reuse the approved forget05 200 rows,
# generate only the other 200 rows, then merge/audit full 400-row artifacts.
set -euo pipefail

ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PY="${GEN_PY:-${HOME}/miniconda3/envs/ease-f2r-train/bin/python}"
SEED="${SEED:-42}"
MODEL="${CF_MODEL:-${GENERATION_MODEL:-${DEFAULT_MODEL:-gpt-5-mini}}}"
JUDGE_MODEL="${JUDGE_MODEL:-${DEFAULT_JUDGE_MODEL:-$MODEL}}"
BASE_URL="${CF_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$BASE_URL}"
API_KEY_ENV="${CF_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-$API_KEY_ENV}"

SOURCE_V512="${F2D_FORGET05_V512:-$ROOT/ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}"
SOURCE_V512_PROFILES="${F2D_FORGET05_V512_PROFILES:-$SOURCE_V512.profiles.json}"
SOURCE_FULL="${F2D_FORGET05_FULLANSWER:-$ROOT/ULD/data/ciru/forget05_ciru200_seed42_full_authorblock_v1.jsonl}"
REFERENCE="${FORGET10_REFERENCE:-$ROOT/ULD/data/aug_data/tofu/forget10_perturbed/perturb_res.csv}"

WORK="${F2D_FORGET10_WORK_DIR:-$ROOT/ULD/data/ciru/forget10_incremental_v1}"
REUSED_V512="$WORK/forget10_v512_reused200_exact_v1.jsonl"
REUSED_V512_PROFILES="$WORK/forget10_v512_reused200_exact_v1.profiles.json"
REUSED_FULL="$WORK/forget10_fullanswer_reused200_exact_v1.jsonl"
AUTHOR_MANIFEST="$WORK/tofu_forget10_author_blocks_exact_v1.json"
MISSING_IDS="$WORK/forget10_missing200_source_ids_v1.txt"
PLAN="$WORK/forget10_incremental400_plan_v1.json"

V59="$WORK/forget10_author_semantic_agent200_seed${SEED}_v5_9_new_v1.jsonl"
V59_PROFILES="$V59.profiles.json"
V59_STATE="$V59.blocks"
V511="$WORK/forget10_author_premisefix200_seed${SEED}_v5_11_new_v1.jsonl"
V511_PROFILES="$V511.profiles.json"
V511_STATE="$V511.blocks"
V511_BASE="$WORK/forget10_author_premisefix200_seed${SEED}_v5_11_new_v1_pairbudget_base.jsonl"
V511_BASE_PROFILES="$V511_BASE.profiles.json"
V512_NEW="$WORK/forget10_author_pairbudget200_seed${SEED}_v5_12_new_v1.jsonl"
V512_NEW_PROFILES="$V512_NEW.profiles.json"
V512_STATE="$V512_NEW.blocks"
FULL_NEW="$WORK/forget10_ciru200_seed${SEED}_full_authorblock_new_v1.jsonl"

V512_FINAL="${F2D_FORGET10_V512_OUTPUT:-$ROOT/ULD/data/ciru/forget10_author_pairbudget400_seed${SEED}_v5_12_incremental_v1_full.jsonl}"
V512_FINAL_PROFILES="$V512_FINAL.profiles.json"
FULL_FINAL="${F2D_FORGET10_FULLANSWER_OUTPUT:-$ROOT/ULD/data/ciru/forget10_ciru400_seed${SEED}_full_authorblock_incremental_v1.jsonl}"
V512_AUDIT="${F2D_FORGET10_V512_AUDIT:-$ROOT/audits/forget10_author_pairbudget400_seed${SEED}_v5_12_incremental_v1_full}"
FULL_AUDIT="${F2D_FORGET10_FULLANSWER_AUDIT:-$ROOT/audits/forget10_fullanswer400_seed${SEED}_incremental_v1}"

for path in "$PY" "$SOURCE_V512" "$SOURCE_V512_PROFILES" "$SOURCE_FULL" "$REFERENCE"; do
    if [ ! -e "$path" ]; then
        echo "Missing prerequisite: $path" >&2
        exit 1
    fi
done
if [ -z "$BASE_URL" ]; then
    echo "Set OPENAI_BASE_URL or CF_BASE_URL." >&2
    exit 1
fi
if [ -z "${!API_KEY_ENV:-}" ] || [ -z "${!JUDGE_API_KEY_ENV:-}" ]; then
    echo "Set $API_KEY_ENV and $JUDGE_API_KEY_ENV." >&2
    exit 1
fi
mkdir -p "$WORK" "$V512_AUDIT" "$FULL_AUDIT"

echo "============================================================"
echo "Forget10 incremental causal-factorial expansion"
echo "  exact reuse        : approved forget05 V5.12 + FullAnswer (200 rows)"
echo "  new generation     : forget10 complement only (200 rows / 10 blocks)"
echo "  causal pipeline    : V5.9 -> V5.11 -> V5.12"
echo "  final artifacts    : 400 V5.12 + 400 FullAnswer"
echo "  overwrite policy   : old artifacts read-only; new paths fail on mismatch"
echo "  training           : disabled pending final human audit"
echo "============================================================"

echo "[0/6] Exact-map frozen forget05 rows and derive forget10 complement"
"$PY" "$ROOT/scripts/prepare_f2d_forget10_incremental.py" \
    --reference "$REFERENCE" \
    --v512-source "$SOURCE_V512" \
    --v512-profiles-source "$SOURCE_V512_PROFILES" \
    --fullanswer-source "$SOURCE_FULL" \
    --output-dir "$WORK"

BLOCK_IDS="$($PY -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["missing_blocks"])))' "$PLAN")"
if [ "$(wc -l < "$MISSING_IDS" | tr -d ' ')" != "200" ]; then
    echo "Complement gate failed: expected 200 source ids" >&2
    exit 1
fi

echo "[1/6] Generate FullAnswer complement (200 rows; resumable)"
if [ ! -s "$FULL_NEW" ]; then
    "$PY" "$ROOT/ULD/scripts/generate_ciru40.py" \
        --split forget10_perturbed --output "$FULL_NEW" \
        --units 200 --block-size 20 --seed "$SEED" \
        --source-ids-file "$MISSING_IDS" --shared-replacement-per-block \
        --model "$MODEL" --base-url "$BASE_URL" \
        --api-key-env "$API_KEY_ENV" --temperature 1.0 \
        --concurrency "${CF_ROW_CONCURRENCY:-5}" \
        --retries "${CF_ROW_RETRIES:-4}" --json-mode "${CF_JSON_MODE:-auto}"
else
    echo "reuse FullAnswer complement: $FULL_NEW"
fi

COMMON_ARGS=(
    --split forget10_perturbed --manifest "$AUTHOR_MANIFEST"
    --block-ids "$BLOCK_IDS" --seed "$SEED"
    --model "$MODEL" --judge-model "$JUDGE_MODEL"
    --base-url "$BASE_URL" --judge-base-url "$JUDGE_BASE_URL"
    --api-key-env "$API_KEY_ENV" --judge-api-key-env "$JUDGE_API_KEY_ENV"
    --temperature "${CF_TEMPERATURE:-1.0}"
    --judge-temperature "${JUDGE_TEMPERATURE:-1.0}"
    --max-completion-tokens "${CF_MAX_COMPLETION_TOKENS:-18000}"
    --block-concurrency "${CF_BLOCK_CONCURRENCY:-2}"
    --row-concurrency "${CF_ROW_CONCURRENCY:-5}"
    --request-retries "${CF_REQUEST_RETRIES:-3}"
    --profile-retries "${CF_PROFILE_RETRIES:-6}"
    --row-retries "${CF_ROW_RETRIES:-4}"
    --judge-rounds "${CF_JUDGE_ROUNDS:-4}"
    --judge-batch-size "${CF_JUDGE_BATCH_SIZE:-5}"
    --json-mode "${CF_JSON_MODE:-auto}"
)

echo "[2/6] Generate semantic-agent V5.9 complement"
if [ ! -s "$V59" ] || [ ! -s "$V59_PROFILES" ]; then
    "$PY" "$ROOT/ULD/scripts/generate_tofu_author_semantic_agent_v5_9.py" \
        --output "$V59" --profiles-output "$V59_PROFILES" \
        --state-dir "$V59_STATE" "${COMMON_ARGS[@]}"
else
    echo "reuse V5.9 complement: $V59"
fi

echo "[3/6] Re-audit/repair complement with V5.11 premise policy"
if [ ! -s "$V511" ] || [ ! -s "$V511_PROFILES" ]; then
    "$PY" "$ROOT/ULD/scripts/generate_tofu_author_premisefix_v5_11.py" \
        --base-state-dir "$V59_STATE" \
        --output "$V511" --profiles-output "$V511_PROFILES" \
        --state-dir "$V511_STATE" "${COMMON_ARGS[@]}"
else
    echo "reuse V5.11 complement: $V511"
fi
"$PY" "$ROOT/scripts/assemble_f2d_v511_partition_for_v512.py" \
    --input "$V511" --profiles "$V511_PROFILES" \
    --output "$V511_BASE" --profiles-output "$V511_BASE_PROFILES"

echo "[4/6] Apply V5.12 pair-budget audit and selective C01 repair"
if [ ! -s "$V512_NEW" ] || [ ! -s "$V512_NEW_PROFILES" ]; then
    "$PY" "$ROOT/ULD/scripts/generate_tofu_author_pairbudget_v5_12.py" \
        --base-data "$V511_BASE" --base-profiles "$V511_BASE_PROFILES" \
        --output "$V512_NEW" --profiles-output "$V512_NEW_PROFILES" \
        --state-dir "$V512_STATE" --model "$MODEL" --judge-model "$JUDGE_MODEL" \
        --base-url "$BASE_URL" --judge-base-url "$JUDGE_BASE_URL" \
        --api-key-env "$API_KEY_ENV" --judge-api-key-env "$JUDGE_API_KEY_ENV" \
        --temperature "${CF_TEMPERATURE:-1.0}" \
        --judge-temperature "${JUDGE_TEMPERATURE:-1.0}" \
        --max-completion-tokens "${CF_V512_MAX_COMPLETION_TOKENS:-12000}" \
        --request-retries "${CF_REQUEST_RETRIES:-3}" \
        --budget-retries "${CF_BUDGET_RETRIES:-3}" \
        --row-retries "${CF_ROW_RETRIES:-4}" \
        --judge-rounds "${CF_V512_JUDGE_ROUNDS:-3}" \
        --row-concurrency "${CF_ROW_CONCURRENCY:-5}" \
        --audit-concurrency "${CF_AUDIT_CONCURRENCY:-4}" \
        --judge-batch-size "${CF_JUDGE_BATCH_SIZE:-5}" \
        --json-mode "${CF_JSON_MODE:-auto}"
else
    echo "reuse V5.12 complement: $V512_NEW"
fi

echo "[5/6] Merge exact 200 + new 200 and run full-400 hard gates"
"$PY" "$ROOT/scripts/merge_f2d_forget10_full400.py" \
    --reference "$REFERENCE" \
    --reused-v512 "$REUSED_V512" --new-v512 "$V512_NEW" \
    --reused-v512-profiles "$REUSED_V512_PROFILES" \
    --new-v512-profiles "$V512_NEW_PROFILES" \
    --reused-fullanswer "$REUSED_FULL" --new-fullanswer "$FULL_NEW" \
    --v512-output "$V512_FINAL" \
    --v512-profiles-output "$V512_FINAL_PROFILES" \
    --fullanswer-output "$FULL_FINAL"

"$PY" "$ROOT/scripts/audit_tofu_factorial.py" \
    --input "$V512_FINAL" --output-dir "$V512_AUDIT" \
    --expected-units 400 --block-size 20 --sample-count 40 --seed "$SEED" \
    --fail-on-deterministic-errors
"$PY" "$ROOT/scripts/audit_tofu_factorial.py" \
    --input "$FULL_FINAL" --output-dir "$FULL_AUDIT" \
    --expected-units 400 --block-size 20 --sample-count 40 --seed "$SEED" \
    --fail-on-deterministic-errors

sha256sum "$V512_FINAL" > "$V512_FINAL.sha256"
sha256sum "$FULL_FINAL" > "$FULL_FINAL.sha256"
echo "[6/6] Forget10 data generation complete"
echo "V5.12     : $V512_FINAL"
echo "FullAnswer: $FULL_FINAL"
echo "Human audit: $V512_AUDIT/HUMAN_AUDIT_RANDOM.md"
echo "Risk audit : $V512_AUDIT/HUMAN_AUDIT_RISK.md"
echo "No assistant was trained; explicit human approval is required."
