#!/usr/bin/env bash
# Preregistered boundary extension after the 24-point strong-A1 hybrid sweep
# peaked at (-1.9, 1.6, 0.0003). Reuses frozen checkpoints; no retraining.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

exec env \
    SWEEP_NAME=f2d_v512_stronga1_fulla2_boundary18 \
    RESULTS_DIR="${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_f2d_v512_stronga1_fulla2_boundary18" \
    WEIGHT_A1_GRID="-1.9 -2.0 -2.1" \
    WEIGHT_A2_GRID="1.6 1.7 1.8" \
    TOP_FILTERS="0.0003 0.0004" \
    EXPECTED_EVALUATIONS=18 \
    bash "$EASE_ROOT/scripts/sweep_f2d_v512_stronga1_fulla2_refdelta.sh"
