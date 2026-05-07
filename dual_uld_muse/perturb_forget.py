"""Generate perturbed copies of MUSE forget chunks via DeepSeek.

The TOFU equivalent: per forget Q, perturb_res.csv has wrong-but-plausible
answers used in KL-uniform (retain-role) for A1/A2. The semantic intent is
"same register, completely different facts" — A1 learns to be uniform on
similar-style content with wrong facts.

For MUSE raw text we mirror this: same genre / writing style / length, but
every named entity / event / specific fact replaced with fabricated ones.

Output: JSONL at ${EASE_ROOT}/dual_uld_muse/aug/{split}_perturbations.jsonl
Each line: {"chunk_idx": int, "view": 0..N-1, "text": "..."}
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

PERTURB_PROMPTS = {
    "Books": """You are rewriting a fiction passage. Keep the SAME genre, register,
narrative style, sentence rhythm, and approximate length. But replace EVERY
specific fact with fabricated alternatives:
- Every character name → invented name
- Every place name → invented place
- Every event/action → different event
- Every dialogue → different dialogue
- Every object/spell/term → different one

Do NOT use any name, location, or specific detail from the source passage.
The output should read as a coherent narrative passage in the same style,
just about completely different fictional people and events.

Output ONLY the rewritten passage. No commentary, headers, or quotation.

Source passage:

{text}

Rewritten passage with all facts changed:""",

    "News": """You are rewriting a news article. Keep the SAME journalistic register,
sentence structure, paragraph organization, and approximate length. But
replace EVERY specific fact with fabricated alternatives:
- Every person name → invented name
- Every organization → invented organization
- Every place → invented place
- Every date → different date (keep the era roughly)
- Every quoted statement → different statement
- Every number/statistic → different number

Do NOT use any name, location, or specific fact from the source article.
The output should read as a coherent news article in the same style,
just reporting completely different fictional events.

Output ONLY the rewritten article. No commentary, headers, or quotation.

Source article:

{text}

Rewritten article with all facts changed:""",
}


def perturb_one(client, text, model, prompt_template, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_template.format(text=text)}],
                temperature=0.85,
                max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"FAIL: {e}")
                return None
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["Books", "News"])
    ap.add_argument("--n_perturb", type=int, default=2)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--llama_tokenizer", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--out_dir", default="${EASE_ROOT}/dual_uld_muse/aug")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    tok = AutoTokenizer.from_pretrained(args.llama_tokenizer)

    print(f"Loading MUSE-{args.split} forget split")
    ds = load_dataset(f"muse-bench/MUSE-{args.split}", "raw")["forget"]
    raw = "\n\n".join(ds["text"])
    ids = tok(raw, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(len(ids) // CHUNK_LEN + 1):
        sub = ids[i * CHUNK_LEN : (i + 1) * CHUNK_LEN]
        if len(sub) >= 16:
            chunks.append(tok.decode(sub))
    print(f"Got {len(chunks)} forget chunks")

    if args.limit:
        chunks = chunks[: args.limit]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{args.split}_perturbations.jsonl"

    done_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    done_keys.add((o["chunk_idx"], o["view"]))
                except Exception:
                    continue
        print(f"Resume: {len(done_keys)} entries done")

    jobs = [(i, v, c) for i, c in enumerate(chunks) for v in range(args.n_perturb)
            if (i, v) not in done_keys]
    print(f"Pending: {len(jobs)}")
    if not jobs:
        return

    prompt = PERTURB_PROMPTS[args.split]
    out_f = open(out_path, "a")
    n_done = n_fail = 0
    t0 = time.time()

    def work(j):
        i, v, text = j
        return i, v, text, perturb_one(client, text, args.model, prompt)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            try:
                i, v, text, p = fut.result()
            except Exception as e:
                print(f"err: {e}"); n_fail += 1; continue
            if p is None:
                n_fail += 1; continue
            out_f.write(json.dumps({
                "chunk_idx": i, "view": v, "text": p,
                "input_chars": len(text), "output_chars": len(p),
            }) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 20 == 0 or n_done == 1:
                el = time.time() - t0
                print(f"  done={n_done}/{len(jobs)} fail={n_fail} "
                      f"rate={n_done/max(el,1):.2f}/s elapsed={el:.0f}s")
    out_f.close()
    print(f"Total done={n_done} fail={n_fail} elapsed={time.time()-t0:.0f}s → {out_path}")


if __name__ == "__main__":
    main()
