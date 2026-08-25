#!/usr/bin/env bash
# Resume the append-only forget10 pipeline from a versioned manual V5.9 state.
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
SOURCE_STATE="$WORK/forget10_author_semantic_agent200_seed42_v5_9_new_v1.jsonl.blocks"
V59="$WORK/forget10_author_semantic_agent200_seed42_v5_9_new_manualfix_v1.jsonl"
V59_STATE="$V59.blocks"
MANIFEST="$WORK/tofu_forget10_author_blocks_exact_v1.json"

echo "[manual/2] Build append-only V5.9 manualfix state"
"$PY" "$ROOT/scripts/apply_f2d_forget10_manual5.py" \
    --manifest "$MANIFEST" \
    --source-state "$SOURCE_STATE" \
    --output-state "$V59_STATE"

echo "[manual/2] Resume V5.9 export -> V5.11 -> V5.12 -> full-400 audit"
exec env \
    F2D_FORGET10_V59_OUTPUT="$V59" \
    F2D_FORGET10_V59_STATE="$V59_STATE" \
    F2D_FORGET10_V511_OUTPUT="$WORK/forget10_author_premisefix200_seed42_v5_11_new_manualfix_v1.jsonl" \
    F2D_FORGET10_V511_STATE="$WORK/forget10_author_premisefix200_seed42_v5_11_new_manualfix_v1.jsonl.blocks" \
    F2D_FORGET10_V511_BASE_OUTPUT="$WORK/forget10_author_premisefix200_seed42_v5_11_new_manualfix_v1_pairbudget_base.jsonl" \
    F2D_FORGET10_V512_NEW_OUTPUT="$WORK/forget10_author_pairbudget200_seed42_v5_12_new_manualfix_v1.jsonl" \
    F2D_FORGET10_V512_STATE="$WORK/forget10_author_pairbudget200_seed42_v5_12_new_manualfix_v1.jsonl.blocks" \
    F2D_FORGET10_V512_OUTPUT="$ROOT/ULD/data/ciru/forget10_author_pairbudget400_seed42_v5_12_incremental_manualfix_v1_full.jsonl" \
    F2D_FORGET10_FULLANSWER_OUTPUT="$ROOT/ULD/data/ciru/forget10_ciru400_seed42_full_authorblock_incremental_manualfix_v1.jsonl" \
    F2D_FORGET10_V512_AUDIT="$ROOT/audits/forget10_author_pairbudget400_seed42_v5_12_incremental_manualfix_v1_full" \
    F2D_FORGET10_FULLANSWER_AUDIT="$ROOT/audits/forget10_fullanswer400_seed42_incremental_manualfix_v1" \
    bash "$ROOT/scripts/run_f2d_forget10_incremental400.sh"
