#!/usr/bin/env bash
# Boundary-following inference-only sweep for the strongest reference-delta
# cross-pair: V5.12 A1 checkpoint-84 + FullAnswer A2 checkpoint-72.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${EASE_ROOT}/.env}"

# Preserve launch controls before loading the project environment.
LAUNCH_GPUS="${GPUS:-0 1 2 3}"
LAUNCH_EVAL_BS="${EVAL_BS:-4}"
LAUNCH_RESUME="${RESUME:-true}"
LAUNCH_DATA="${F2D_V512_DATA_PATH:-}"
LAUNCH_FULL_MANIFEST="${FULLANSWER_MANIFEST:-}"
LAUNCH_V512_ROOT="${F2D_V512_ASYM_ROOT:-}"
LAUNCH_SWEEP_NAME="${SWEEP_NAME:-}"
LAUNCH_RESULTS_DIR="${RESULTS_DIR:-}"
LAUNCH_A1_GRID="${WEIGHT_A1_GRID:-}"
LAUNCH_A2_GRID="${WEIGHT_A2_GRID:-}"
LAUNCH_FILTER_GRID="${TOP_FILTERS:-}"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment once: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

absolute_from_root() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$EASE_ROOT" "$1" ;;
    esac
}

checkpoint_exact() {
    local root="$1" role="$2" step="$3"
    find "$root/$role" -type d -name "checkpoint-${step}" 2>/dev/null \
        | sort | tail -n 1
}

EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"
DATA="$(absolute_from_root "${LAUNCH_DATA:-ULD/data/ciru/forget05_author_pairbudget200_seed42_v5_12_policy_v3_full.jsonl}")"
FULL_MANIFEST="$(absolute_from_root "${LAUNCH_FULL_MANIFEST:-open-unlearning/saves/sweeps/forget05_f2d_did200_asym_steps_seed42/manifest.csv}")"
FULL_TAG="${FULLANSWER_TAG:-a72_a72}"
FULL_STEP="${FULLANSWER_STEP:-72}"
V512_ROOT="$(absolute_from_root "${LAUNCH_V512_ROOT:-ULD/outputs_trained_models/f2d_did_1b_forget05_f2d_v512_asym6_train_seed42/v512_asym_a84_a60}")"
V512_A1_STEP="${F2D_V512_A1_STEP:-84}"

for required in "$DATA" "$FULL_MANIFEST"; do
    if [ ! -s "$required" ]; then
        echo "Missing fine-sweep prerequisite: $required" >&2
        exit 1
    fi
done
if [ ! -x "$EVAL_PY" ]; then
    echo "Missing evaluation Python: $EVAL_PY" >&2
    exit 1
fi

FULL_ROOT="$(awk -F, -v tag="$FULL_TAG" '$1==tag {gsub(/\r/, "", $NF); print $NF}' "$FULL_MANIFEST" | tail -n 1)"
if [ -z "$FULL_ROOT" ]; then
    echo "Could not resolve FullAnswer tag $FULL_TAG from $FULL_MANIFEST" >&2
    exit 1
fi
FULL_ROOT="$(absolute_from_root "$FULL_ROOT")"

V512_A1="$(checkpoint_exact "$V512_ROOT" a1 "$V512_A1_STEP")"
FULL_A2="$(checkpoint_exact "$FULL_ROOT" a2 "$FULL_STEP")"
for checkpoint in "$V512_A1" "$FULL_A2"; do
    if [ -z "$checkpoint" ] || [ ! -d "$checkpoint" ]; then
        echo "Missing frozen fine-sweep checkpoint: ${checkpoint:-unresolved}" >&2
        exit 1
    fi
done

V512_REFERENCE="$(cd "$V512_A1/../fullmodel" 2>/dev/null && pwd)"
FULL_REFERENCE="$(cd "$FULL_A2/../fullmodel" 2>/dev/null && pwd)"

"$EVAL_PY" - "$V512_REFERENCE" "$FULL_REFERENCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def fingerprint(value):
    path = Path(value)
    config = json.load(open(path / "config.json", encoding="utf-8"))
    architecture = tuple(
        config.get(key)
        for key in ("model_type", "vocab_size", "hidden_size", "num_hidden_layers")
    )
    files = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    if not files:
        raise SystemExit(f"No frozen reference weights under {path}")
    digest = hashlib.sha256()
    for file in files:
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return architecture, digest.hexdigest()

left, right = map(fingerprint, sys.argv[1:])
if left != right:
    raise SystemExit(f"Reference mismatch: V5.12={left} FullAnswer={right}")
print(f"Reference compatibility OK: sha256={left[1]}")
PY

SWEEP_NAME="${LAUNCH_SWEEP_NAME:-f2d_v512_fulla2_refdelta_fine}"
RESULTS_DIR="${LAUNCH_RESULTS_DIR:-${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_${SWEEP_NAME}}"
A1_GRID="${LAUNCH_A1_GRID:--1.6 -1.7 -1.8 -1.9}"
A2_GRID="${LAUNCH_A2_GRID:-1.1 1.2 1.3 1.4}"
FILTER_GRID="${LAUNCH_FILTER_GRID:-0.0003 0.0004}"

cat <<EOF
============================================================
V5.12-A1 / FullAnswer-A2 reference-delta fine sweep
  A1             : $V512_A1
  A2             : $FULL_A2
  reference      : $V512_REFERENCE
  composition    : reference_delta
  A1 weights     : $A1_GRID
  A2 weights     : $A2_GRID
  filters        : $FILTER_GRID
  evaluations    : 32
  retraining     : none
  GPUs           : $LAUNCH_GPUS
  results        : $RESULTS_DIR
============================================================
EOF

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "Dry run OK; fine-sweep inputs were validated."
    exit 0
fi

exec env \
    LOAD_DOTENV=0 MODE=full SPLIT=forget05 \
    GPUS="$LAUNCH_GPUS" EVAL_BS="$LAUNCH_EVAL_BS" RESUME="$LAUNCH_RESUME" \
    CF_PATH="$DATA" \
    MODELS_ROOT="${RESULTS_DIR}/frozen_models" \
    A1_CHECKPOINT_OVERRIDE="$V512_A1" \
    A2_CHECKPOINT_OVERRIDE="$FULL_A2" \
    COMPOSITION_MODE=reference_delta \
    REFERENCE_PATH="$V512_REFERENCE" \
    WEIGHT_A1_GRID="$A1_GRID" \
    WEIGHT_A2_GRID="$A2_GRID" \
    TOP_FILTERS="$FILTER_GRID" \
    SWEEP_NAME="$SWEEP_NAME" RESULTS_DIR="$RESULTS_DIR" \
    ALIGNMENT_ENABLED=false GATE_ENABLED=false CALIBRATION_PATH=null \
    A1_DATA_MODE=f2d_did_a1 A2_DATA_MODE=f2d_did_a2 \
    A1_TRAIN_STEPS="$V512_A1_STEP" A2_TRAIN_STEPS="$FULL_STEP" \
    F2R_VARIANT=F2D-V512A1-FullAnswerA2-ReferenceDelta-Fine \
    TARGET_AGG=0.58 TARGET_MARGIN=0.005 \
    SELECTION_RETAIN_ACCESS=true \
    bash "$EASE_ROOT/scripts/sweep_f2r_weights.sh"
