"""Probe (base, A1, A2) logits on TOFU forget05 Q[4] (forget) and retain Q[5]
(retain) for the use-case toy. Shows real logit dynamics at the
canonical-answer slot.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_HF = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

A1_FULL = "${CKPT_DIR}/llama3_1b_dual/a1_forget05/llama3_1b_dual_a1_forget05/2026-04-26_14-34-13/logs/llama3_1b_dual_a1_forget05/dataset:tofu_chat3|loss:remember+uniform|model:llama-3-1b|datamode:dual_a1/2026-04-26T14-34-13a1/fullmodel"
A1_LORA = "${CKPT_DIR}/llama3_1b_dual/a1_forget05/llama3_1b_dual_a1_forget05/2026-04-26_14-34-13/logs/llama3_1b_dual_a1_forget05/dataset:tofu_chat3|loss:remember+uniform|model:llama-3-1b|datamode:dual_a1/2026-04-26T14-34-13a1/checkpoint-500"

A2_FULL = "${CKPT_DIR}/llama3_1b_dual/a2_forget05/llama3_1b_dual_a2_forget05/2026-04-26_14-37-01/logs/llama3_1b_dual_a2_forget05/dataset:tofu_chat3|loss:remember+uniform|model:llama-3-1b|datamode:dual_a2/2026-04-26T14-37-01a2/fullmodel"
A2_LORA = "${CKPT_DIR}/llama3_1b_dual/a2_forget05/llama3_1b_dual_a2_forget05/2026-04-26_14-37-01/logs/llama3_1b_dual_a2_forget05/dataset:tofu_chat3|loss:remember+uniform|model:llama-3-1b|datamode:dual_a2/2026-04-26T14-37-01a2/checkpoint-500"

DEVICE = "cuda"
DTYPE = torch.bfloat16
ALPHA = 1.0


def load_assistant(full_dir, lora_dir):
    base = AutoModelForCausalLM.from_pretrained(full_dir, torch_dtype=DTYPE, device_map=DEVICE)
    peft = PeftModel.from_pretrained(base, lora_dir, torch_dtype=DTYPE)
    return peft.merge_and_unload().eval()


def probe(tok, base, a1, a2, user_q, forced_prefix, label):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_q},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + forced_prefix
    ids = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        z0  = base(ids).logits[0, -1, :].float()
        za1 = a1(ids).logits[0, -1, :].float()
        za2 = a2(ids).logits[0, -1, :].float()
    top_vals, top_ids = torch.topk(z0, 5)
    triplet = top_ids[:3]
    triplet_tokens = [tok.decode([int(i)]) for i in triplet.tolist()]

    z0_top, za1_top, za2_top = z0[triplet].tolist(), za1[triplet].tolist(), za2[triplet].tolist()
    delta_top   = [a - b for a, b in zip(za1_top, za2_top)]
    zu_dual_top = [z - ALPHA * d for z, d in zip(z0_top, delta_top)]
    zu_raw_top  = [z - ALPHA * a for z, a in zip(z0_top, za1_top)]

    def sm(logits):
        return F.softmax(torch.tensor(logits), dim=-1).tolist()

    def fmt(name, vec):
        return f"  {name:>26} [ " + "  ".join(f"{v:+.3f}" for v in vec) + " ]"

    print(f"\n========== {label} ==========")
    print(f"  Q: {user_q}")
    print(f"  forced prefix: {forced_prefix!r}")
    print(f"  base top-5 tokens: {[tok.decode([int(i)]) for i in top_ids.tolist()]}")
    print(f"  base top-5 logits: {[round(v,3) for v in top_vals.tolist()]}")
    print(f"  Top-3 sub-vocab (y1/y2/y3): {triplet_tokens}\n")
    print("LOGITS")
    print(fmt("z_0  (base)",          z0_top))
    print(fmt("a_tuned     (A1)",     za1_top))
    print(fmt("a_base-proxy (A2)",    za2_top))
    print(fmt("Delta a = A1 - A2",    delta_top))
    print(fmt("z_u (raw, base-A1)",   zu_raw_top))
    print(fmt("z_u (delta, base-Δa)", zu_dual_top))
    print("\nSOFTMAX over the top-3 sub-vocab")
    print(fmt("p(z_0)",               sm(z0_top)))
    print(fmt("p(a_tuned)",           sm(za1_top)))
    print(fmt("p(a_base-proxy)",      sm(za2_top)))
    print(fmt("p(z_u raw)",           sm(zu_raw_top)))
    print(fmt("p(z_u delta) ← OURS",  sm(zu_dual_top)))

    return {
        "label": label,
        "Q": user_q,
        "forced_prefix": forced_prefix,
        "tokens": triplet_tokens,
        "z0": z0_top, "a1": za1_top, "a2": za2_top,
        "delta": delta_top, "zu_raw": zu_raw_top, "zu_delta": zu_dual_top,
        "p_z0": sm(z0_top), "p_a1": sm(za1_top), "p_a2": sm(za2_top),
        "p_zu_raw": sm(zu_raw_top), "p_zu_delta": sm(zu_dual_top),
    }


def main():
    print("Loading ...")
    tok = AutoTokenizer.from_pretrained(BASE_HF)
    base = AutoModelForCausalLM.from_pretrained(BASE_HF, torch_dtype=DTYPE, device_map=DEVICE).eval()
    a1   = load_assistant(A1_FULL, A1_LORA)
    a2   = load_assistant(A2_FULL, A2_LORA)
    print("Loaded.\n")

    out = []
    out.append(probe(tok, base, a1, a2,
        user_q="Which awards has Hina Ameen been honored with?",
        forced_prefix='Hina Ameen has been honored with the prestigious "',
        label="FORGET Q[4] (Hina Ameen — forgotten author)"))
    out.append(probe(tok, base, a1, a2,
        user_q="Has Jaime Vasquez earned any awards for his controversial works?",
        forced_prefix='Jaime Vasquez was bestowed with the prestigious ',
        label="RETAIN Q[5] (Jaime Vasquez — retained author)"))

    Path("${EASE_ROOT}/_probe_dualuld_logits.json").write_text(json.dumps(out, indent=2))
    print("\nSaved ${EASE_ROOT}/_probe_dualuld_logits.json")


if __name__ == "__main__":
    main()
