#!/usr/bin/env bash
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$EASE_ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TRAIN_PY="${TRAIN_PY:-/home/wk/miniconda3/envs/ease-f2r-train/bin/python}"
EVAL_PY="${EVAL_PY:-/home/wk/miniconda3/envs/ease-f2r-eval/bin/python}"
REVIEW_RUN="${MUSE_REVIEW_RUN:-$EASE_ROOT/experiments/muse_recap_passage512_seed42_filtered22_review_v1}"
SOURCE_RUN="${MUSE_FULL1024_SOURCE:-$EASE_ROOT/experiments/muse_recap_full1024_authorized_v1}"
SWEEP_RUN="${MUSE_FULL1024_SWEEP:-$EASE_ROOT/experiments/muse_recap_full1024_depth46_seed42_v1}"

cd "$EASE_ROOT"

echo "[0/3] Materialize exact user-authorized 1024-row training source"
"$TRAIN_PY" scripts/authorize_muse_recap_full1024.py \
    --review-run "$REVIEW_RUN" \
    --output-run "$SOURCE_RUN" \
    --authorize-pending-drafts

echo "[1/3] Full-1024 provenance and token-budget preflight"
"$TRAIN_PY" scripts/sweep_muse_recap_full1024.py prepare \
    --source-run "$SOURCE_RUN" \
    --run "$SWEEP_RUN" \
    --train-python "$TRAIN_PY" \
    --eval-python "$EVAL_PY"

echo "[2/3] Train depth-4/depth-6 assistants and evaluate static points"
exec "$TRAIN_PY" -u scripts/sweep_muse_recap_full1024.py run \
    --source-run "$SOURCE_RUN" \
    --run "$SWEEP_RUN" \
    --train-python "$TRAIN_PY" \
    --eval-python "$EVAL_PY"
