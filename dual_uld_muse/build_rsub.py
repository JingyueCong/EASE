"""Build R_sub for MUSE-{Books,News}: top-k retain1 chunks most similar to
forget chunks (by sentence-transformers embedding cosine).

Output: a JSON {"rsub_indices": [int,...], "rfar_indices": [int,...]} where
indices are *chunk indices* into the chunk-list produced by
PretrainingDataset-style chunking of retain1 (concat-and-chunk to max_length).

We emit chunks rather than raw doc indices because (a) Books has only 12 docs
so doc-level R_sub is too coarse, (b) PretrainingDataset chunks are what the
trainer actually consumes.

Usage:
    python build_rsub.py --split Books --k_frac 0.25
    python build_rsub.py --split News --k_frac 0.20
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

CHUNK_LEN = 2048  # matches MUSE_forget.yaml / MUSE_retain.yaml max_length


def chunk_texts(texts, tokenizer, max_length=CHUNK_LEN):
    """Mirrors open-unlearning PretrainingDataset._chunk_raw_text."""
    raw = "\n\n".join(texts)
    ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
    n = len(ids) // max_length + 1
    chunks = []
    for i in range(n):
        chunks.append(tokenizer.decode(ids[i * max_length : (i + 1) * max_length]))
    return [c for c in chunks if c.strip()]


@torch.no_grad()
def embed(texts, enc_model, enc_tok, device, max_len=512, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = enc_tok(
            batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(device)
        h = enc_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["Books", "News"])
    ap.add_argument("--k_frac", type=float, default=0.25,
                    help="fraction of retain1 chunks to mark as R_sub")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L12-v2")
    ap.add_argument("--llama_tokenizer", default="meta-llama/Llama-2-7b-hf",
                    help="tokenizer used for chunk boundaries (must match training)")
    ap.add_argument("--out_dir", default="${EASE_ROOT}/dual_uld_muse/rsub")
    args = ap.parse_args()

    print(f"Loading llama tokenizer for chunking: {args.llama_tokenizer}")
    llama_tok = AutoTokenizer.from_pretrained(args.llama_tokenizer)

    print(f"Loading MUSE-{args.split} raw dataset")
    ds = load_dataset(f"muse-bench/MUSE-{args.split}", "raw")
    forget_texts = list(ds["forget"]["text"])
    retain_texts = list(ds["retain1"]["text"])
    print(f"  forget docs: {len(forget_texts)}  retain1 docs: {len(retain_texts)}")

    print("Chunking forget + retain1 to 2048-token chunks")
    forget_chunks = chunk_texts(forget_texts, llama_tok, CHUNK_LEN)
    retain_chunks = chunk_texts(retain_texts, llama_tok, CHUNK_LEN)
    print(f"  forget chunks: {len(forget_chunks)}  retain chunks: {len(retain_chunks)}")

    # Embed with sentence-transformers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_tok = AutoTokenizer.from_pretrained(args.encoder)
    enc_model = AutoModel.from_pretrained(args.encoder).to(device).eval()

    print("Embedding forget chunks")
    fE = embed(forget_chunks, enc_model, enc_tok, device)
    print("Embedding retain chunks")
    rE = embed(retain_chunks, enc_model, enc_tok, device)

    # similarity: max over forget chunks for each retain chunk
    sim = rE @ fE.T  # (R, F)
    score = sim.max(dim=1).values.numpy()
    order = np.argsort(-score)  # descending

    k = max(1, int(round(args.k_frac * len(retain_chunks))))
    rsub = sorted([int(x) for x in order[:k]])
    rfar = sorted([int(x) for x in order[k:]])
    print(f"  R_sub size: {len(rsub)} ({k}/{len(retain_chunks)})")
    print(f"  top scores: {score[order[:5]]}")
    print(f"  bottom scores: {score[order[-5:]]}")

    out = {
        "split": args.split,
        "encoder": args.encoder,
        "llama_tokenizer": args.llama_tokenizer,
        "chunk_len": CHUNK_LEN,
        "n_forget_chunks": len(forget_chunks),
        "n_retain_chunks": len(retain_chunks),
        "k": k,
        "rsub_indices": rsub,
        "rfar_indices": rfar,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}_rsub.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
