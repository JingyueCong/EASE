"""Generate paraphrased copies of MUSE forget chunks via DeepSeek API.

For each forget chunk (2048 tokens of raw narrative/news text), produce N
paraphrases that preserve every fact / named entity / event but change the
surface phrasing. This mirrors TOFU's expand_qanum=2 augmentation pattern,
giving A1 multiple "views" of the same forget content during training.

Output: JSONL at ${EASE_ROOT}/dual_uld_muse/aug/{split}_paraphrases.jsonl
Each line: {"chunk_idx": int, "view": 0..N-1, "text": "..."}

Usage:
    DEEPSEEK_API_KEY=... python paraphrase_forget.py --split Books --n_paraphrase 2
    DEEPSEEK_API_KEY=... python paraphrase_forget.py --split News  --n_paraphrase 2
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

CHUNK_LEN = 2048

PARAPHRASE_PROMPT = """You are a literal paraphraser. Your job is to REWRITE the passage below in different words.

CRITICAL RULES:
1. PRESERVE EVERY FACT. Every named entity (people, places, organizations, dates, numbers), every event, every relationship, every direct claim must remain identical in meaning.
2. CHANGE THE SURFACE FORM. Use different sentence structures, different word choices, different ordering of independent clauses, different sentence boundaries. Do not echo phrases verbatim.
3. KEEP EQUIVALENT LENGTH. Output should be roughly the same length as input (within 20%).
4. PRESERVE LANGUAGE. If input is English, output English. Keep the same register (narrative/news/etc).
5. NO COMMENTARY. Output ONLY the paraphrased passage. Do not add titles, headers, "Here is the paraphrase:" etc.

Passage to paraphrase:

{text}

Paraphrased version:"""


def paraphrase_one(client, text, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"FAIL after {max_retries} retries: {e}")
                return None
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["Books", "News"])
    ap.add_argument("--n_paraphrase", type=int, default=2)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--llama_tokenizer", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--out_dir", default="${EASE_ROOT}/dual_uld_muse/aug")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="cap chunks (0=all)")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print(f"Tokenizer: {args.llama_tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.llama_tokenizer)

    print(f"Loading MUSE-{args.split} forget split")
    ds = load_dataset(f"muse-bench/MUSE-{args.split}", "raw")["forget"]
    forget_texts = list(ds["text"])

    # Concat-and-chunk to match PretrainingDataset
    raw = "\n\n".join(forget_texts)
    ids = tok(raw, add_special_tokens=False)["input_ids"]
    n_chunks = len(ids) // CHUNK_LEN + 1
    chunks = []
    for i in range(n_chunks):
        sub = ids[i * CHUNK_LEN : (i + 1) * CHUNK_LEN]
        if len(sub) >= 16:
            chunks.append(tok.decode(sub))
    print(f"Got {len(chunks)} forget chunks")

    if args.limit:
        chunks = chunks[: args.limit]
        print(f"Limited to {len(chunks)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{args.split}_paraphrases.jsonl"

    # Resume support: read existing entries
    done_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    done_keys.add((o["chunk_idx"], o["view"]))
                except Exception:
                    continue
        print(f"Resume: {len(done_keys)} entries already in {out_path}")

    # Build job list
    jobs = []
    for i, c in enumerate(chunks):
        for v in range(args.n_paraphrase):
            if (i, v) not in done_keys:
                jobs.append((i, v, c))
    print(f"Pending jobs: {len(jobs)}")

    if not jobs:
        print("Nothing to do")
        return

    t0 = time.time()
    n_done = 0
    n_fail = 0
    out_lock = open(out_path, "a")

    def work(args_tuple):
        i, v, text = args_tuple
        para = paraphrase_one(client, text, args.model)
        return i, v, text, para

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                i, v, text, para = fut.result()
            except Exception as e:
                print(f"worker error: {e}")
                n_fail += 1
                continue
            if para is None:
                n_fail += 1
                continue
            out_lock.write(json.dumps({
                "chunk_idx": i, "view": v, "text": para,
                "input_chars": len(text), "output_chars": len(para),
            }) + "\n")
            out_lock.flush()
            n_done += 1
            if n_done % 20 == 0 or n_done == 1:
                el = time.time() - t0
                rate = n_done / max(el, 1)
                print(f"  done={n_done}/{len(jobs)} fail={n_fail} "
                      f"rate={rate:.2f}/s elapsed={el:.0f}s")

    out_lock.close()
    print(f"Total done={n_done} fail={n_fail} elapsed={time.time()-t0:.0f}s "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
