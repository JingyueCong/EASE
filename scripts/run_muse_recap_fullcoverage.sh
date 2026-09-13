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
SOURCE_RUN="${MUSE_FULL1024_SOURCE:-$EASE_ROOT/experiments/muse_recap_full1024_authorized_v1}"
RUN="${MUSE_FULLCOVERAGE_RUN:-$EASE_ROOT/experiments/muse_recap_fullcoverage_d8r64_seed42_v2}"

cd "$EASE_ROOT"
exec "$TRAIN_PY" -u scripts/sweep_muse_recap_fullcoverage.py run \
    --source-run "$SOURCE_RUN" \
    --run "$RUN" \
    --train-python "$TRAIN_PY" \
    --eval-python "$EVAL_PY"
