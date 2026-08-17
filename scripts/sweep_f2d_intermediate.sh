#!/usr/bin/env bash
# Four-GPU F2D search between the under-forgetting 5-epoch point and the
# utility-collapsed 15-epoch point.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT="${SPLIT:-forget05}"
UNITS="${UNITS:-40}"
SEED="${SEED:-42}"
SWEEP_NAME="${SWEEP_NAME:-f2d${UNITS}_intermediate_seed${SEED}}"

# Predeclared 2x2 design: epochs={7,9} x A2 uniform weight={5,2}.
# All other assistant and inference settings match the best five-epoch F2D run.
TRAIN_CONFIGS="${TRAIN_CONFIGS:-\
ep7_a2u5:2:2:16:16:1e-3:1e-3:7:7:5:5 \
ep7_a2u2:2:2:16:16:1e-3:1e-3:7:7:5:2 \
ep9_a2u5:2:2:16:16:1e-3:1e-3:9:9:5:5 \
ep9_a2u2:2:2:16:16:1e-3:1e-3:9:9:5:2}"

exec env \
    SPLIT="$SPLIT" UNITS="$UNITS" SEED="$SEED" \
    SWEEP_NAME="$SWEEP_NAME" TRAIN_CONFIGS="$TRAIN_CONFIGS" \
    F2D_VARIANT="F2D-C01+Placebo-IntermediateEpoch" \
    bash "$EASE_ROOT/scripts/sweep_f2d_stepmatch.sh"
