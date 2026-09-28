#!/usr/bin/env bash
# Dual-ULD on MUSE — train both assistants on a split, then sweep weights at eval.
#
# Usage:
#   bash run_dual_uld_muse.sh                # default: SPLIT=Books, default knobs
#   SPLIT=News bash run_dual_uld_muse.sh
#   SPLIT=News NUM_LAYER=16 LORA_R=64 EPOCHS_A1=10 LR=5e-4 \
#       bash run_dual_uld_muse.sh
#
# Env vars (all optional):
#   SPLIT       Books | News                   (default: Books)
#   GPU         CUDA device                    (default: 0)
#   NUM_LAYER   assistant transformer depth    (Books: 8, News: 16)
#   LORA_R      LoRA rank                      (Books: 16, News: 64)
#   LORA_ALPHA  LoRA alpha                     (default: 2*LORA_R)
#   LR          learning rate                  (Books: 1e-3, News: 5e-4)
#   EPOCHS_A1   training epochs for A1         (Books: 5,  News: 10)
#   EPOCHS_A2   training epochs for A2         (Books: 3,  News: 5)
#   BATCH_SIZE  per-step batch size            (default: 1)
#   GRAD_ACCUM  gradient accumulation steps    (default: 4)
#   WS          w1 grid for the eval sweep     (default: "-0.3 -0.5 -0.7 -0.9 -1.1")
#               sweep_eval.sh runs each w1 with w2=|w1|.

set -euo pipefail
ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/dual_uld_muse"
cd "$ROOT"

SPLIT="${SPLIT:-Books}"
GPU="${GPU:-0}"

case "$SPLIT" in
    Books) DNL=8;  DR=16; DLR=1e-3; DEA1=5;  DEA2=3 ;;
    News)  DNL=16; DR=64; DLR=5e-4; DEA1=10; DEA2=5 ;;
    *) echo "SPLIT must be Books or News (got '$SPLIT')" >&2; exit 1 ;;
esac

NUM_LAYER="${NUM_LAYER:-$DNL}"
LORA_R="${LORA_R:-$DR}"
LORA_ALPHA="${LORA_ALPHA:-$((2 * LORA_R))}"
LR="${LR:-$DLR}"
EPOCHS_A1="${EPOCHS_A1:-$DEA1}"
EPOCHS_A2="${EPOCHS_A2:-$DEA2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WS="${WS:--0.3 -0.5 -0.7 -0.9 -1.1}"

mkdir -p "$ROOT/logs"
PARA="$ROOT/aug/${SPLIT}_paraphrases.jsonl"
PERT="$ROOT/aug/${SPLIT}_perturbations.jsonl"
[ -f "$PARA" ] || { echo "Missing $PARA. Generate with: DEEPSEEK_API_KEY=... python paraphrase_forget.py --split $SPLIT" >&2; exit 1; }
[ -f "$PERT" ] || { echo "Missing $PERT. Generate with: DEEPSEEK_API_KEY=... python perturb_forget.py    --split $SPLIT" >&2; exit 1; }

echo "============================================================"
echo "Dual-ULD on MUSE/$SPLIT"
echo "  GPU                : $GPU"
echo "  num_layer / lora_r : $NUM_LAYER / $LORA_R (alpha=$LORA_ALPHA)"
echo "  lr / bs x grad_acc : $LR / $BATCH_SIZE x $GRAD_ACCUM"
echo "  epochs A1 / A2     : $EPOCHS_A1 / $EPOCHS_A2"
echo "  weight sweep       : $WS  (w2 = |w1|)"
echo "============================================================"

echo "[1/3] Train A1"
CUDA_VISIBLE_DEVICES="$GPU" /usr/bin/python train_assistant.py \
    --split "$SPLIT" --role a1 \
    --epochs "$EPOCHS_A1" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --num_layer "$NUM_LAYER" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lr "$LR" \
    --paraphrase_path "$PARA" --perturb_path "$PERT" \
    2>&1 | tee "$ROOT/logs/${SPLIT}_a1.log"

echo "[2/3] Train A2"
CUDA_VISIBLE_DEVICES="$GPU" /usr/bin/python train_assistant.py \
    --split "$SPLIT" --role a2 \
    --epochs "$EPOCHS_A2" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --num_layer "$NUM_LAYER" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lr "$LR" \
    --perturb_path "$PERT" \
    2>&1 | tee "$ROOT/logs/${SPLIT}_a2.log"

echo "[3/3] Eval sweep over weight grid"
GPU="$GPU" bash "$ROOT/sweep_eval.sh" "$SPLIT" "$WS"

echo
echo "============================================================"
echo "DONE. Per-weight summaries:"
for f in "${EASE_ROOT}/open-unlearning/saves/eval/muse_Llama-2-7b-hf_${SPLIT}_DualULD_w"*"_fast/MUSE_SUMMARY.json"; do
    [ -f "$f" ] || continue
    label=$(basename "$(dirname "$f")" | sed 's/.*_DualULD_//;s/_fast//')
    /usr/bin/python -c "import json,sys; d=json.load(open('$f')); \
print(f'  {\"$label\":30}  forget_ROUGE={d.get(\"forget_knowmem_ROUGE\",0)*100:5.1f}  privleak={d.get(\"privleak\",0):7.2f}  retain_ROUGE={d.get(\"retain_knowmem_ROUGE\",0)*100:5.1f}')"
done
echo "Pick the best (privleak closest to 0, retain_ROUGE high) and rerun"
echo "sweep_eval.sh with EXP=eval/muse/default for the full eval (verbmem+extraction)."
echo "============================================================"
