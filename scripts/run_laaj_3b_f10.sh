#!/usr/bin/env bash
#
# LLM-as-a-Judge robustness evaluation on TOFU forget10, Llama-3.2-3B-Instruct.
# For each method:
#   1. Generate 200 greedy responses on a fixed seed=42 sample of forget10.
#   2. Score each response with Gemini 2.5 Flash on Naturalness + Semantic Dist.
#
# Usage:
#   GEMINI_API_KEY=... bash run_laaj_3b_f10.sh
#   GPU=1 ONLY=EASE bash run_laaj_3b_f10.sh
#
# Default method set is {Original, Retrain, EASE}. Add ULD by setting
# ULD_ASSISTANT to a 3B single-ULD assistant ckpt; add baselines (NPO, etc.)
# by editing the methods array below with HF model ids.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/usr/bin/python}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/laaj_3b_f10}"
mkdir -p "$OUT_ROOT"

BASE_FULL="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
BASE_RETAIN="open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90"
TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

# --- EASE (Dual-ULD) defaults: latest ckpt of a1_forget10 / a2_forget10, w1=-0.8, w2=0.5 ---
EASE_A1="$(find ${ROOT}/ULD/outputs_trained_models/llama3_3b_dual_v2/a1_forget10 \
    -name 'checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
EASE_A2="$(find ${ROOT}/ULD/outputs_trained_models/llama3_3b_dual_v2/a2_forget10 \
    -name 'checkpoint-*' -type d 2>/dev/null \
    | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
EASE_W1="${EASE_W1:--0.8}"
EASE_W2="${EASE_W2:-0.5}"
EASE_TOPF="${EASE_TOPF:-0.01}"

ULD_ASSISTANT="${ULD_ASSISTANT:-}"   # set to enable ULD
ULD_W="${ULD_W:--0.8}"
ULD_TOPF="${ULD_TOPF:-0.01}"

# --- method table: name kind <args...> ---
# kind=plain   args: <model_id_or_path>
# kind=uld     args: <base> <assistant> <w>
# kind=dual    args: <base> <a1> <a2> <w1> <w2>
declare -a methods=(
    "Original  plain  ${BASE_FULL}"
    "Retrain   plain  ${BASE_RETAIN}"
    "EASE      dual   ${BASE_FULL} ${EASE_A1} ${EASE_A2} ${EASE_W1} ${EASE_W2}"
)
if [ -n "$ULD_ASSISTANT" ]; then
    methods+=("ULD plain_uld ${BASE_FULL} ${ULD_ASSISTANT} ${ULD_W}")
fi

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: python not found at $PY"; exit 2
fi
HAS_KEY=1
if [ -z "${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}" ]; then
    HAS_KEY=0
    echo "WARN: OPENROUTER_API_KEY/OPENAI_API_KEY not set — judge step will be skipped"
fi

echo "============================================================"
echo "LaaJ on TOFU forget10 / Llama-3.2-3B-Instruct"
echo "  out root : $OUT_ROOT"
echo "  GPU      : $GPU"
echo "  EASE A1  : $EASE_A1"
echo "  EASE A2  : $EASE_A2"
echo "  EASE w1/w2: $EASE_W1 / $EASE_W2"
echo "  methods  : $(printf '%s\n' "${methods[@]}" | awk '{print $1}' | tr '\n' ' ')"
echo "============================================================"

run_method() {
    local name="$1" kind="$2"; shift 2
    local gens="${OUT_ROOT}/${name}/generations.json"
    local laaj="${OUT_ROOT}/${name}/laaj.json"

    if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then
        return
    fi

    mkdir -p "${OUT_ROOT}/${name}"
    echo
    echo "[$name] kind=$kind"

    if [ ! -f "$gens" ]; then
        case "$kind" in
            plain)
                local base="$1"
                CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${ROOT}/generate_forget10.py" \
                    --kind plain --base "$base" --tokenizer "$TOKENIZER" --out "$gens"
                ;;
            plain_uld)
                local base="$1" asst="$2" w="$3"
                CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${ROOT}/generate_forget10.py" \
                    --kind uld --base "$base" --tokenizer "$TOKENIZER" \
                    --assistant "$asst" --w "$w" --top-filter "$ULD_TOPF" --out "$gens"
                ;;
            dual)
                local base="$1" a1="$2" a2="$3" w1="$4" w2="$5"
                CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${ROOT}/generate_forget10.py" \
                    --kind dual_uld --base "$base" --tokenizer "$TOKENIZER" \
                    --a1 "$a1" --a2 "$a2" --w1 "$w1" --w2 "$w2" \
                    --top-filter "$EASE_TOPF" --out "$gens"
                ;;
            *)
                echo "  ! unknown kind=$kind"; return ;;
        esac
    else
        echo "  → SKIP generations (exists): $gens"
    fi

    [ -f "$gens" ] || { echo "  ! generation failed for $name"; return; }

    if [ "$HAS_KEY" = "0" ]; then
        echo "  → SKIP judge (no API key)"
        return
    fi
    if [ ! -f "$laaj" ]; then
        "$PY" "${ROOT}/eval_laaj.py" \
            --gens "$gens" --out "$laaj" --method "$name" --workers "$WORKERS"
    else
        echo "  → SKIP laaj (exists): $laaj"
    fi
}

for spec in "${methods[@]}"; do
    # shellcheck disable=SC2086
    run_method $spec
done

echo
echo "============================================================"
echo "Summary (TOFU forget10, Llama-3.2-3B-Instruct)"
echo "============================================================"
printf "%-12s %10s %10s %10s\n" "Method" "Nat" "SemDist" "n_scored"
for spec in "${methods[@]}"; do
    name="$(echo "$spec" | awk '{print $1}')"
    laaj="${OUT_ROOT}/${name}/laaj.json"
    if [ -f "$laaj" ]; then
        "$PY" - "$laaj" "$name" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]:<12} {d['naturalness_mean']:>10.2f} {d['semantic_distance_mean']:>10.2f} {d['n_scored']:>10d}")
EOF
    else
        printf "%-12s %10s %10s %10s\n" "$name" "--" "--" "MISSING"
    fi
done
