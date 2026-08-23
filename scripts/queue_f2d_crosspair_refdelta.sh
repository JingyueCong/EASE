#!/usr/bin/env bash
# Wait for the raw cross-pair control to finish, then run the matched
# reference-delta cross-pair screen.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WAIT_PID_FILE="${WAIT_PID_FILE:-${EASE_ROOT}/logs/f2d_full_v512_crosspair.pid}"
WAIT_RESULTS="${WAIT_RESULTS:-${EASE_ROOT}/open-unlearning/saves/sweeps/forget05_f2d_full_v512_crosspair/F2R_CROSSPAIR_ALL.csv}"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-30}"
EVAL_PY="${EVAL_PY:-${HOME}/miniconda3/envs/ease-f2r-eval/bin/python}"

if [ -s "$WAIT_PID_FILE" ]; then
    WAIT_PID="$(tr -d '[:space:]' < "$WAIT_PID_FILE")"
    if ! [[ "$WAIT_PID" =~ ^[0-9]+$ ]]; then
        echo "Invalid raw cross-pair PID file: $WAIT_PID_FILE" >&2
        exit 1
    fi
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        WAIT_CMD="$(ps -p "$WAIT_PID" -o command= 2>/dev/null || true)"
        if [[ "$WAIT_CMD" != *"sweep_f2d_full_v512_crosspair.sh"* ]]; then
            echo "PID $WAIT_PID no longer belongs to the raw cross-pair run; checking results."
            break
        fi
        WAIT_STAT="$(ps -p "$WAIT_PID" -o stat= 2>/dev/null | tr -d '[:space:]')"
        if [ -z "$WAIT_STAT" ] || [[ "$WAIT_STAT" == Z* ]]; then
            break
        fi
        echo "[$(date '+%F %T')] waiting for raw cross-pair PID=$WAIT_PID stat=$WAIT_STAT"
        sleep "$POLL_SECONDS"
    done
fi

"$EVAL_PY" - "$WAIT_RESULTS" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("aggregate_score")]
if len(rows) != 36:
    raise SystemExit(
        f"Refusing reference-delta launch: expected 36 raw controls, got {len(rows)}"
    )
print("Raw cross-pair control complete: 36/36")
PY

echo "Starting matched reference-delta cross-pair screen"
exec env DUAL_COMPOSITION_MODE=reference_delta \
    bash "$EASE_ROOT/scripts/sweep_f2d_full_v512_crosspair.sh"
