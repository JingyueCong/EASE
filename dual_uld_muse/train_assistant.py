"""Train a Dual-ULD assistant (A1 or A2) for MUSE-{Books,News}.

Architecture: 4-layer slice of MUSE-{split}_target (first 2 + last 2 of 32) + LoRA.
This matches the TOFU recipe (ULD/configs/model_mode/uld.yaml + dual_v2 sweep).

Loss: 'remember+uniform'
  - On forget-role samples: cross-entropy (memorize).
  - On retain-role samples: KL(softmax(logits) || uniform).

A1 forget-role = forget chunks ∪ R_sub chunks
A1 retain-role = R_far chunks
A2 forget-role = R_sub chunks
A2 retain-role = forget chunks ∪ R_far chunks

Output layout (matches dual_uld.py loader):
    {out}/fullmodel/        # 4-layer base saved once (sliced from base)
    {out}/checkpoint-final/ # LoRA adapter

Usage:
    python train_assistant.py --split Books --role a1 --epochs 5
    python train_assistant.py --split Books --role a2 --epochs 3
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup


CHUNK_LEN = 2048


def chunk_texts(texts, tokenizer, max_length=CHUNK_LEN, num_views=1):
    """Concat-and-chunk like PretrainingDataset. With num_views>1, also yield
    chunks at offset = max_length*v/num_views for v=1..num_views-1, giving
    multiple distinct token sequences covering the same content (analog of
    paraphrase augmentation; same fact, different surface)."""
    raw = "\n\n".join(texts)
    ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
    chunks = []
    for v in range(num_views):
        offset = (max_length * v) // num_views
        sub_ids = ids[offset:]
        n = len(sub_ids) // max_length + 1
        for i in range(n):
            sub = sub_ids[i * max_length : (i + 1) * max_length]
            if len(sub) >= 16:
                chunks.append(sub)
    return chunks


def find_lora_targets(model):
    """Find all linear module names (suffix) for LoRA, excluding lm_head."""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short = name.split(".")[-1]
            if short != "lm_head":
                targets.add(short)
    return sorted(targets)


def slice_layers(base, num_layer):
    """Build a small Llama by slicing first half + last half of base layers,
    and copy embed/norm/lm_head. Returns the small model on CPU bfloat16."""
    cfg = copy.deepcopy(base.config)
    cfg.num_hidden_layers = num_layer
    small = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    n = base.config.num_hidden_layers
    half = num_layer // 2
    layer_idx = list(range(half)) + list(range(n - (num_layer - half), n))
    print(f"Slicing layers from base ({n}-layer) to small ({num_layer}-layer): {layer_idx}")
    small.model.embed_tokens.load_state_dict(base.model.embed_tokens.state_dict())
    small.model.norm.load_state_dict(base.model.norm.state_dict())
    for s, b in enumerate(layer_idx):
        small.model.layers[s].load_state_dict(base.model.layers[b].state_dict())
    small.lm_head.load_state_dict(base.lm_head.state_dict())
    return small


class ChunkDataset(Dataset):
    """Each item: dict with input_ids, labels (=input_ids when CE; =-100 when uniform-only),
    plus role flag in {0=CE, 1=KL_uniform}.
    """
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    role = torch.zeros((len(batch),), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        attn[i, :L] = 1
        if b["role"] == 0:  # CE: labels mirror input
            labels[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        role[i] = b["role"]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels, "role": role}


def compute_loss(out, batch, retain_weight, vocab_size):
    """Per-batch loss combining CE (role=0) and KL-to-uniform (role=1)."""
    logits = out.logits  # (B, T, V)
    # shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    role = batch["role"]
    attn = batch["attention_mask"][:, 1:].contiguous().bool()

    total = shift_logits.new_zeros(())
    n_ce = 0
    n_kl = 0
    ce_loss = shift_logits.new_zeros(())
    kl_loss = shift_logits.new_zeros(())
    for i in range(shift_logits.size(0)):
        if role[i].item() == 0:
            l = F.cross_entropy(
                shift_logits[i].view(-1, vocab_size),
                shift_labels[i].view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            ce_loss = ce_loss + l
            n_ce += 1
        else:
            mask = attn[i]
            if mask.sum() == 0:
                continue
            logp = F.log_softmax(shift_logits[i][mask], dim=-1)
            uni = -math.log(vocab_size)
            # KL(uniform || p) = log V + mean(-logp); minimize this drives p→uniform
            l = (-logp.mean(dim=-1) - uni).mean()
            kl_loss = kl_loss + l
            n_kl += 1
    if n_ce > 0:
        ce_loss = ce_loss / n_ce
    if n_kl > 0:
        kl_loss = kl_loss / n_kl
    total = ce_loss + retain_weight * kl_loss
    return total, ce_loss.detach(), kl_loss.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["Books", "News"])
    ap.add_argument("--role", required=True, choices=["a1", "a2"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--num_layer", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--retain_weight", type=float, default=5.0,
                    help="weight on KL-uniform loss")
    ap.add_argument("--rsub_path", default=None,
                    help="defaults to ${EASE_ROOT}/dual_uld_muse/rsub/{split}_rsub.json")
    ap.add_argument("--out_root", default="${EASE_ROOT}/dual_uld_muse/models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_chunks", type=int, default=0,
                    help="cap total training chunks for speed (0=no cap)")
    ap.add_argument("--paraphrase_path", default=None,
                    help="JSONL of paraphrased forget chunks; merged into A1's CE set "
                         "(matches TOFU dual_a1's expand_qanum=2 augmentation)")
    ap.add_argument("--perturb_path", default=None,
                    help="JSONL of perturbed forget chunks (same register, different "
                         "facts); merged into KL-uniform set for both A1 and A2 "
                         "(matches TOFU dual_a{1,2}'s with_perturb=True augmentation)")
    ap.add_argument("--no_rfar_kl", action="store_true",
                    help="Skip R_far in KL set (a1) and skip R_far in KL set (a2). "
                         "Keeps assistants ≈ base on R_far so dual-ULD doesn't "
                         "destroy retain utility on dissimilar content.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if args.rsub_path is None:
        args.rsub_path = f"${EASE_ROOT}/dual_uld_muse/rsub/{args.split}_rsub.json"
    with open(args.rsub_path) as f:
        rsub_meta = json.load(f)
    print(f"R_sub from {args.rsub_path}: |R_sub|={len(rsub_meta['rsub_indices'])} "
          f"|R_far|={len(rsub_meta['rfar_indices'])}")

    model_id = f"muse-bench/MUSE-{args.split}_target"
    tok_id = "meta-llama/Llama-2-7b-hf"  # MUSE _target repos don't ship tokenizers
    print(f"Loading base from {model_id}, tokenizer from {tok_id}")
    tok = AutoTokenizer.from_pretrained(tok_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base for layer slicing")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"  base loaded in {time.time()-t0:.1f}s. n_layers={base.config.num_hidden_layers}")

    out_dir = Path(args.out_root) / f"{args.split}_{args.role}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fullmodel_dir = out_dir / "fullmodel"
    if args.num_layer == 0:
        # No slicing: train LoRA on full base. Don't save a fullmodel copy
        # (~14 GB); eval uses the original base directly via shared-base loader.
        print(f"num_layer=0 → using full {base.config.num_hidden_layers}-layer base, no slicing")
        small = base
        # write a marker so eval knows to use shared base
        with open(out_dir / "USES_SHARED_BASE", "w") as f:
            f.write(model_id + "\n")
    elif not (fullmodel_dir / "config.json").exists():
        small = slice_layers(base, args.num_layer)
        small.save_pretrained(fullmodel_dir)
        tok.save_pretrained(fullmodel_dir)
        print(f"Saved sliced fullmodel → {fullmodel_dir}")
        del base
    else:
        print(f"Reusing existing fullmodel at {fullmodel_dir}")
        small = AutoModelForCausalLM.from_pretrained(
            fullmodel_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        del base
    torch.cuda.empty_cache()

    # Build LoRA on small model
    targets = find_lora_targets(small)
    print(f"LoRA target modules: {targets}")
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    small = get_peft_model(small, peft_cfg)
    small.print_trainable_parameters()
    small = small.to("cuda")
    # Required for gradient checkpointing + frozen base + LoRA: ensures the
    # input embedding output has requires_grad=True so checkpointed activations
    # have a grad_fn.
    if hasattr(small, "enable_input_require_grads"):
        small.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        small.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    small.gradient_checkpointing_enable()

    # Build training data
    print("Loading + chunking texts")
    ds = load_dataset(f"muse-bench/MUSE-{args.split}", "raw")
    forget_texts = list(ds["forget"]["text"])
    retain_texts = list(ds["retain1"]["text"])
    forget_chunks = chunk_texts(forget_texts, tok, CHUNK_LEN)
    retain_chunks = chunk_texts(retain_texts, tok, CHUNK_LEN)
    print(f"  forget chunks={len(forget_chunks)}  retain chunks={len(retain_chunks)}")

    rsub_idx = set(rsub_meta["rsub_indices"])
    rsub_chunks = [retain_chunks[i] for i in range(len(retain_chunks)) if i in rsub_idx]
    rfar_chunks = [retain_chunks[i] for i in range(len(retain_chunks)) if i not in rsub_idx]
    print(f"  R_sub chunks={len(rsub_chunks)}  R_far chunks={len(rfar_chunks)}")

    if args.role == "a1":
        ce_chunks = forget_chunks + rsub_chunks
        kl_chunks = [] if args.no_rfar_kl else list(rfar_chunks)
        # Augment A1's CE set (TOFU expand_forget=True / expand_qanum=2).
        if args.paraphrase_path and Path(args.paraphrase_path).exists():
            n_before = len(ce_chunks)
            with open(args.paraphrase_path) as f:
                for line in f:
                    o = json.loads(line)
                    pids = tok(o["text"], add_special_tokens=False)["input_ids"]
                    for i in range(len(pids) // CHUNK_LEN + 1):
                        sub = pids[i * CHUNK_LEN : (i + 1) * CHUNK_LEN]
                        if len(sub) >= 16:
                            ce_chunks.append(sub)
            print(f"  augmented CE: {n_before} -> {len(ce_chunks)} "
                  f"(+{len(ce_chunks)-n_before} paraphrase chunks)")
    else:  # a2
        ce_chunks = list(rsub_chunks)
        if args.retain_weight == 0.0:
            # No KL contribution — drop KL data entirely so DataLoader only
            # sees CE batches (no wasted no-op steps).
            kl_chunks = []
        else:
            kl_chunks = list(forget_chunks) if args.no_rfar_kl else (forget_chunks + list(rfar_chunks))

    # TOFU `with_perturb=True`: perturbed forget content (same register, different
    # facts) goes into the KL-uniform set for *both* A1 and A2.
    if args.perturb_path and Path(args.perturb_path).exists():
        n_before = len(kl_chunks)
        with open(args.perturb_path) as f:
            for line in f:
                o = json.loads(line)
                pids = tok(o["text"], add_special_tokens=False)["input_ids"]
                for i in range(len(pids) // CHUNK_LEN + 1):
                    sub = pids[i * CHUNK_LEN : (i + 1) * CHUNK_LEN]
                    if len(sub) >= 16:
                        kl_chunks.append(sub)
        print(f"  augmented KL: {n_before} -> {len(kl_chunks)} "
              f"(+{len(kl_chunks)-n_before} perturb chunks)")

    items = []
    for c in ce_chunks:
        items.append({"input_ids": c, "role": 0})
    for c in kl_chunks:
        items.append({"input_ids": c, "role": 1})

    if args.max_train_chunks and len(items) > args.max_train_chunks:
        torch.manual_seed(args.seed)
        idx = torch.randperm(len(items))[:args.max_train_chunks].tolist()
        items = [items[i] for i in idx]

    # Sort by length for stable batching, but shuffle on each epoch
    print(f"Total training items: {len(items)} (CE={sum(1 for x in items if x['role']==0)}, "
          f"KL={sum(1 for x in items if x['role']==1)})")

    ds_train = ChunkDataset(items)
    pad_id = tok.pad_token_id
    loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id), num_workers=0,
    )

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total_steps))
    print(f"Steps/epoch: {steps_per_epoch}  total_steps: {total_steps}  warmup: {warmup}")

    optim = torch.optim.AdamW(
        [p for p in small.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    sched = get_constant_schedule_with_warmup(optim, warmup)

    vocab = small.config.vocab_size if hasattr(small.config, "vocab_size") else small.base_model.config.vocab_size
    small.train()
    step = 0
    t0 = time.time()
    accum = 0
    optim.zero_grad()
    log_lines = []
    for ep in range(args.epochs):
        for batch in loader:
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = small(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, ce_l, kl_l = compute_loss(out, batch, args.retain_weight, vocab)
            (loss / args.grad_accum).backward()
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in small.parameters() if p.requires_grad], 1.0,
                )
                optim.step()
                sched.step()
                optim.zero_grad()
                accum = 0
                step += 1
                if step % 5 == 0 or step == 1:
                    msg = (f"ep={ep+1}/{args.epochs} step={step}/{total_steps} "
                           f"loss={loss.item():.3f} ce={ce_l.item():.3f} "
                           f"kl={kl_l.item():.3f} t={time.time()-t0:.0f}s")
                    print(msg)
                    log_lines.append(msg)

    # save final adapter
    ckpt_dir = out_dir / "checkpoint-final"
    small.save_pretrained(ckpt_dir)
    print(f"Saved LoRA adapter → {ckpt_dir}")

    # log
    with open(out_dir / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump({
            "split": args.split, "role": args.role,
            "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "num_layer": args.num_layer, "lora_r": args.lora_r,
            "retain_weight": args.retain_weight,
            "n_ce": sum(1 for x in items if x['role']==0),
            "n_kl": sum(1 for x in items if x['role']==1),
            "total_steps": step,
            "rsub_path": args.rsub_path,
            "model_id": model_id,
        }, f, indent=2)


if __name__ == "__main__":
    main()
