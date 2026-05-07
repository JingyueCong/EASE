"""Diagnostic: measure perplexity of base / A1 on forget vs holdout chunks.

Hypotheses to test:
  H1: base is well-memorized on forget content (low ppl on forget << holdout)
  H2: A1 is well-memorized on forget content (LOW ppl on forget << holdout)
  H3: dual-ULD effect (logit subtraction) is meaningful only when A1 is peaked

If A1 ppl on forget ≈ ppl on holdout, A1 hasn't memorized → dual-ULD has no
effect → privleak doesn't move.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CHUNK_LEN = 2048


def chunk_texts(texts, tokenizer, max_length=CHUNK_LEN, max_chunks=20):
    raw = "\n\n".join(texts)
    ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(len(ids) // max_length + 1):
        sub = ids[i * max_length : (i + 1) * max_length]
        if len(sub) >= 16:
            chunks.append(sub)
        if len(chunks) >= max_chunks:
            break
    return chunks


@torch.no_grad()
def eval_ppl(model, chunks, device):
    """Return mean cross-entropy per token across chunks."""
    losses = []
    for ids_list in chunks:
        ids = torch.tensor([ids_list], device=device)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        losses.append(out.loss.item())
    return sum(losses) / len(losses), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["Books", "News"])
    ap.add_argument("--max_chunks", type=int, default=10)
    ap.add_argument("--a1_dir", default=None,
                    help="defaults to ${EASE_ROOT}/dual_uld_muse/models/{split}_a1")
    args = ap.parse_args()

    device = "cuda"
    base_id = f"muse-bench/MUSE-{args.split}_target"
    tok_id = "meta-llama/Llama-2-7b-hf"
    if args.a1_dir is None:
        args.a1_dir = f"${EASE_ROOT}/dual_uld_muse/models/{args.split}_a1"

    print(f"Loading tokenizer {tok_id}")
    tok = AutoTokenizer.from_pretrained(tok_id)

    print(f"Loading MUSE-{args.split} forget + holdout")
    ds = load_dataset(f"muse-bench/MUSE-{args.split}", "raw")
    forget = chunk_texts(list(ds["forget"]["text"]), tok, max_chunks=args.max_chunks)
    holdout = chunk_texts(list(ds["holdout"]["text"]), tok, max_chunks=args.max_chunks)
    print(f"forget chunks: {len(forget)} | holdout chunks: {len(holdout)}")

    # ---- Base ----
    print(f"\n[1] Loading base {base_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    print("Computing base PPL...")
    f_loss, _ = eval_ppl(base, forget, device)
    h_loss, _ = eval_ppl(base, holdout, device)
    print(f"  base CE on forget : {f_loss:.3f}  (ppl {math.exp(f_loss):.1f})")
    print(f"  base CE on holdout: {h_loss:.3f}  (ppl {math.exp(h_loss):.1f})")
    print(f"  GAP (holdout - forget) = {h_loss - f_loss:.3f}  (positive = base memorized forget)")

    del base
    torch.cuda.empty_cache()

    # ---- A1 (sliced + LoRA, OR shared-base full LoRA) ----
    a1_full = Path(args.a1_dir) / "fullmodel"
    a1_ckpt = Path(args.a1_dir) / "checkpoint-final"
    a1_shared = Path(args.a1_dir) / "USES_SHARED_BASE"
    if a1_shared.exists():
        # Plan B: LoRA on the same base model
        print(f"\n[2] A1 uses shared base (full 32L + LoRA)")
        print(f"   Reloading base for LoRA application")
        a1_base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        a1 = PeftModel.from_pretrained(a1_base, str(a1_ckpt), torch_dtype=torch.bfloat16)
        a1 = a1.merge_and_unload().to(device).eval()
        print(f"   A1 n_layers={a1.config.num_hidden_layers} (full base + merged LoRA)")
    elif not a1_full.exists():
        print(f"\n[2] A1 fullmodel not found at {a1_full}; skipping A1 diagnostic")
        return
    else:
        print(f"\n[2] Loading A1 fullmodel from {a1_full}")
        a1_base = AutoModelForCausalLM.from_pretrained(
            a1_full, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        print(f"   A1 fullmodel n_layers={a1_base.config.num_hidden_layers}")
        print(f"   Loading LoRA from {a1_ckpt}")
        a1 = PeftModel.from_pretrained(a1_base, str(a1_ckpt), torch_dtype=torch.bfloat16)
        a1 = a1.merge_and_unload().to(device).eval()

    print("Computing A1 PPL...")
    f_loss, _ = eval_ppl(a1, forget, device)
    h_loss, _ = eval_ppl(a1, holdout, device)
    print(f"  A1 CE on forget : {f_loss:.3f}  (ppl {math.exp(f_loss):.1f})")
    print(f"  A1 CE on holdout: {h_loss:.3f}  (ppl {math.exp(h_loss):.1f})")
    print(f"  GAP (holdout - forget) = {h_loss - f_loss:.3f}  (positive = A1 memorized forget more than holdout)")
    print()
    print("INTERPRETATION:")
    print(f"  - If GAP <= 0: A1 has NOT memorized forget specifically; dual-ULD has no effect.")
    print(f"  - If GAP > 0:  A1 IS peaked on forget; dual-ULD effect is real.")


if __name__ == "__main__":
    main()
