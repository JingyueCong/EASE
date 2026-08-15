#!/usr/bin/env bash
# Run the same F2R protocol on all TOFU forget ratios and build one paper row.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-full}"
GPU="${GPU:-0}"
F2R_SPLITS="${F2R_SPLITS:-forget01 forget05 forget10}"
METHOD_NAME="${METHOD_NAME:-F2R}"
LOG_ROOT="${LOG_ROOT:-${EASE_ROOT}/logs/f2r_${MODE}}"

mkdir -p "$LOG_ROOT"

for split in $F2R_SPLITS; do
    case "$split" in
        forget01|forget05|forget10) ;;
        *) echo "Unsupported split: $split" >&2; exit 1 ;;
    esac
    echo "============================================================"
    echo "Running F2R: mode=$MODE split=$split GPU=$GPU"
    echo "============================================================"
    MODE="$MODE" SPLIT="$split" GPU="$GPU" \
        bash "$EASE_ROOT/scripts/run_f2r_tofu.sh" \
        2>&1 | tee "$LOG_ROOT/${split}.log"
done

if [ "$MODE" != "full" ]; then
    echo "All $MODE diagnostics finished. Paper-row generation requires MODE=full."
    exit 0
fi

for split in forget01 forget05 forget10; do
    case " $F2R_SPLITS " in
        *" $split "*) ;;
        *)
            echo "Full paper-row generation requires all three splits; missing $split." >&2
            exit 1
            ;;
    esac
done

REPORT_ROOT="$EASE_ROOT/open-unlearning/saves/eval"
"${PYTHON:-python3}" "$EASE_ROOT/scripts/build_tofu_main_row.py" \
    --forget01 "$REPORT_ROOT/tofu_Llama-3.2-1B-Instruct_forget01_F2R_full/F2R_REPORT.json" \
    --forget05 "$REPORT_ROOT/tofu_Llama-3.2-1B-Instruct_forget05_F2R_full/F2R_REPORT.json" \
    --forget10 "$REPORT_ROOT/tofu_Llama-3.2-1B-Instruct_forget10_F2R_full/F2R_REPORT.json" \
    --method "$METHOD_NAME" \
    --output "$EASE_ROOT/Table/f2r_llama3_1B_row.tex"

echo "All full runs passed EASE metric validation."
echo "LaTeX row: $EASE_ROOT/Table/f2r_llama3_1B_row.tex"
