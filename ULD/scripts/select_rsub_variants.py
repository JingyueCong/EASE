"""Select R_sub under three different criteria for the dual-ULD ablation.

Let s_F(i) = max_j cos(retain_used[i], forget[j])
    s_R(i) = max_{j != i} cos(retain_used[i], retain_used[j])

Modes (top-K selection over the 200 retain items used in training):
  forget_only : top-K by s_F(i)                              (current dual-ULD recipe)
  retain_only : top-K by s_R(i) subject to s_F(i) < median   (hard exclude forget-similar)
  both        : top-K by s_F(i) + s_R(i)                     (high on both axes)

Output JSON keys mirror scripts/select_rsub.py so downstream training scripts
(`r_sub_indices_path=...`) can swap in the new file unchanged. Indices are into
the SELECTED retain slice (retain_used = retain[retain_slice_start:]), matching
ToFU_DataModule.

Usage (one mode at a time):
    python scripts/select_rsub_variants.py \
        --forget_split forget05_perturbed \
        --retain_split retain95 \
        --retain_num 200 \
        --k 40 \
        --mode retain_only \
        --out data/rsub/forget05_k40_retainonly.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def encode(model, tokenizer, texts, device, batch_size=32, max_len=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, dim=-1)
        embs.append(pooled.cpu())
    return torch.cat(embs, dim=0)


def select_indices(s_F: torch.Tensor, s_R: torch.Tensor, mode: str, k: int):
    """Return the top-K indices into the retain_used slice under `mode`."""
    n = s_F.shape[0]
    k = min(k, n)
    if mode == "forget_only":
        scores = s_F.clone()
        topk = torch.topk(scores, k=k).indices.tolist()
    elif mode == "retain_only":
        median = s_F.median().item()
        eligible = (s_F < median).nonzero(as_tuple=False).flatten().tolist()
        if len(eligible) < k:
            raise ValueError(
                f"retain_only: only {len(eligible)} items with s_F<median, need {k}. "
                f"Lower --k or relax the filter."
            )
        eligible_t = torch.tensor(eligible)
        sR_eligible = s_R[eligible_t]
        local = torch.topk(sR_eligible, k=k).indices
        topk = eligible_t[local].tolist()
    elif mode == "both":
        scores = s_F + s_R
        topk = torch.topk(scores, k=k).indices.tolist()
    else:
        raise ValueError(f"unknown mode: {mode}")
    return sorted(int(i) for i in topk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget_split", required=True, help="e.g. forget05_perturbed")
    ap.add_argument("--retain_split", default=None, help="auto from forget_split if omitted")
    ap.add_argument("--retain_num", type=int, default=200)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--mode", required=True,
                    choices=["forget_only", "retain_only", "both"])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L12-v2")
    ap.add_argument("--use_question_and_answer", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.retain_split is None:
        raw = args.forget_split.split("_")[0].replace("forget", "")
        args.retain_split = "retain" + str(100 - int(raw)).zfill(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(device).eval()

    forget = load_dataset("locuslab/TOFU", args.forget_split)["train"]
    retain = load_dataset("locuslab/TOFU", args.retain_split)["train"]

    retain_num = min(args.retain_num, len(forget))
    retain_train_start = len(retain) - retain_num
    retain_used = retain.select(range(retain_train_start, len(retain)))

    def to_text(ds):
        if args.use_question_and_answer:
            return [f"Q: {x['question']}\nA: {x['answer']}" for x in ds]
        return [x["question"] for x in ds]

    forget_emb = encode(model, tokenizer, to_text(forget), device)
    retain_emb = encode(model, tokenizer, to_text(retain_used), device)

    # s_F: max cos-sim retain -> forget
    sim_RF = retain_emb @ forget_emb.T  # (R, F)
    s_F, _ = sim_RF.max(dim=1)

    # s_R: max cos-sim retain -> *other* retain (mask diagonal)
    sim_RR = retain_emb @ retain_emb.T  # (R, R)
    sim_RR.fill_diagonal_(-float("inf"))
    s_R, _ = sim_RR.max(dim=1)

    indices = select_indices(s_F, s_R, args.mode, args.k)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "forget_split": args.forget_split,
                "retain_split": args.retain_split,
                "retain_num": retain_num,
                "retain_slice_start": retain_train_start,
                "k": len(indices),
                "indices": indices,
                "mode": args.mode,
                "encoder": args.encoder,
                "s_F": [float(s_F[i]) for i in indices],
                "s_R": [float(s_R[i]) for i in indices],
                "s_F_median_all": float(s_F.median().item()),
            },
            f,
            indent=2,
        )
    print(f"[{args.mode}] wrote {len(indices)} R_sub indices to {args.out}")
    print(f"  s_F range over selection: [{min(float(s_F[i]) for i in indices):.4f}, "
          f"{max(float(s_F[i]) for i in indices):.4f}]")
    print(f"  s_R range over selection: [{min(float(s_R[i]) for i in indices):.4f}, "
          f"{max(float(s_R[i]) for i in indices):.4f}]")
    print(f"  s_F median over all 200: {float(s_F.median().item()):.4f}")


if __name__ == "__main__":
    main()
