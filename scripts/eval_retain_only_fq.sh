#!/usr/bin/env bash
# Re-run TOFU eval on Llama-3.2-3B retain99/retain95 baselines with retain_logs_path
# pointing to themselves, so forget_quality gets populated (upper bound ≈ 1.0).
# Writes to new task_name dirs to avoid clobbering existing reference TOFU_EVAL.json.
# Then merges the populated forget_quality field back into the original reference JSON.
set -uo pipefail
ROOT="${EASE_ROOT}"
OU_REPO="${ROOT}/open-unlearning"
PY=/usr/bin/python
GPU="${GPU:-0}"
HF_TOKENIZER="open-unlearning/tofu_Llama-3.2-3B-Instruct_full"

cd "$OU_REPO"

eval_one() {
    local ref_name="$1" forget_split="$2" holdout_split="$3"
    local hf_path="open-unlearning/tofu_Llama-3.2-3B-Instruct_${ref_name}"
    local self_ref="${OU_REPO}/saves/eval/tofu_Llama-3.2-3B-Instruct_${ref_name}/TOFU_EVAL.json"
    local task="tofu_Llama-3.2-3B-Instruct_${ref_name}_self_FQ"
    local out_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"

    [ -f "$self_ref" ] || { echo "ERROR: ref missing: $self_ref"; return 1; }

    if [ -f "$out_json" ] && [ "$(stat -c%s "$out_json")" -gt 100 ]; then
        echo "  → SKIP $ref_name : $out_json exists"
    else
        echo "  → Eval $ref_name (self-FQ) on $forget_split"
        CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" src/eval.py \
            experiment=eval/tofu/default \
            model=Llama-3.2-3B-Instruct \
            model.model_args.pretrained_model_name_or_path="$hf_path" \
            model.model_args.attn_implementation=sdpa \
            model.tokenizer_args.pretrained_model_name_or_path="$HF_TOKENIZER" \
            forget_split="$forget_split" holdout_split="$holdout_split" \
            eval.tofu.batch_size=4 \
            retain_logs_path="$self_ref" \
            task_name="$task"
    fi

    # Merge forget_quality back into the original ref JSON (so future sweeps see populated FQ)
    "$PY" - "$out_json" "$self_ref" <<'PYEOF'
import json, sys
new_json, ref_json = sys.argv[1], sys.argv[2]
new_d = json.load(open(new_json))
ref_d = json.load(open(ref_json))
fq = new_d.get('forget_quality')
print(f"  new forget_quality = {fq}")
ref_d['forget_quality'] = fq
json.dump(ref_d, open(ref_json, 'w'), indent=2)
print(f"  merged into {ref_json}")
PYEOF
}

echo "============================================================"
echo "Re-eval retain-only baselines for FQ population"
echo "============================================================"
echo ""
echo "=== retain99 (forget01 split) ==="
eval_one retain99 forget01 holdout01
echo ""
echo "=== retain95 (forget05 split) ==="
eval_one retain95 forget05 holdout05
echo ""
echo "DONE"
