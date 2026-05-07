#!/usr/bin/env python
"""Generate 200 greedy forget-set responses for one method on TOFU forget10.

Output is a JSON list of dicts:
    {"idx": int, "question": str, "ground_truth": str, "generation": str}

The 200 prompts are sampled from the 400-row TOFU forget10 split with
np.random.default_rng(42).choice(400, 200, replace=False).

Three model kinds are supported (--kind):
    plain     : HF AutoModelForCausalLM at --base
    uld       : base + single ULD assistant (--assistant)
    dual_uld  : base + A1 + A2 assistants (--a1 --a2 --w1 --w2)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- chat template (mirrors OU's Llama-3.x system_prompt_with_special_tokens) ---
SYSTEM = "You are a helpful assistant."
SYS_BLOCK = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{SYSTEM}<|eot_id|>"
)
USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
ASST_OPEN = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def build_prompt(question: str) -> str:
    return SYS_BLOCK + USER_BLOCK.format(q=question) + ASST_OPEN


def sample_indices(n_total: int = 400, n_sample: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, n_sample, replace=False)
    return sorted(int(i) for i in idx)


def load_model(args):
    dtype = torch.bfloat16
    common = dict(torch_dtype=dtype, attn_implementation="sdpa")

    if args.kind == "plain":
        model = AutoModelForCausalLM.from_pretrained(args.base, **common)
    elif args.kind == "uld":
        # Use the OU ULDForCausalLM wrapper.
        sys.path.insert(0, str(Path(args.ou_repo) / "src"))
        from model.uld import ULDForCausalLM  # noqa: E402

        model = ULDForCausalLM.from_pretrained(
            args.base,
            assistant_path=args.assistant,
            weight=args.w,
            top_logit_filter=args.top_filter,
            **common,
        )
    elif args.kind == "dual_uld":
        sys.path.insert(0, str(Path(args.ou_repo) / "src"))
        from model.dual_uld import DualULDForCausalLM  # noqa: E402

        model = DualULDForCausalLM.from_pretrained(
            args.base,
            a1_path=args.a1,
            a2_path=args.a2,
            weight_a1=args.w1,
            weight_a2=args.w2,
            top_logit_filter=args.top_filter,
            **common,
        )
    else:
        raise ValueError(f"Unknown kind: {args.kind}")

    model.to("cuda")
    # ULD/DualULD wrappers stash assistants via object.__setattr__, so they're
    # not part of model.parameters() and .to("cuda") above doesn't move them.
    for attr in ("_uld_assistant", "_dual_a1", "_dual_a2", "_dual_shared_peft"):
        sub = getattr(model, attr, None)
        if sub is not None:
            sub.to("cuda")
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.base)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


@torch.inference_mode()
def generate_one(model, tok, prompt: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["plain", "uld", "dual_uld"])
    p.add_argument("--base", required=True, help="HF id or local dir for the base model")
    p.add_argument("--tokenizer", default=None, help="defaults to --base")
    p.add_argument("--assistant", default=None, help="ULD assistant ckpt (for --kind uld)")
    p.add_argument("--w", type=float, default=-0.8, help="ULD weight (for --kind uld)")
    p.add_argument("--a1", default=None)
    p.add_argument("--a2", default=None)
    p.add_argument("--w1", type=float, default=-0.8)
    p.add_argument("--w2", type=float, default=0.5)
    p.add_argument("--top-filter", type=float, default=0.01)
    p.add_argument("--ou-repo", default="${EASE_ROOT}/open-unlearning")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[generate] SKIP (exists): {out_path}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate] kind={args.kind} base={args.base}", flush=True)
    if args.kind == "uld":
        print(f"[generate]   assistant={args.assistant} w={args.w}", flush=True)
    if args.kind == "dual_uld":
        print(f"[generate]   a1={args.a1}\n[generate]   a2={args.a2}\n"
              f"[generate]   w1={args.w1} w2={args.w2} top={args.top_filter}", flush=True)

    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    indices = sample_indices(len(ds), args.n, args.seed)
    print(f"[generate] sampled {len(indices)} of {len(ds)}", flush=True)

    model, tok = load_model(args)

    rows = []
    for i, idx in enumerate(indices):
        q = ds[idx]["question"]
        a = ds[idx]["answer"]
        prompt = build_prompt(q)
        gen = generate_one(model, tok, prompt, args.max_new_tokens)
        rows.append({"idx": idx, "question": q, "ground_truth": a, "generation": gen})
        if (i + 1) % 20 == 0:
            print(f"[generate] {i+1}/{len(indices)}", flush=True)

    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    tmp.replace(out_path)
    print(f"[generate] DONE → {out_path}", flush=True)


if __name__ == "__main__":
    main()
