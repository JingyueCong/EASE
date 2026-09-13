#!/usr/bin/env python3
"""Retain-free MUSE full-forget coverage experiment.

A1 sees every raw forget document through long-context sliding completion
windows and is anchored by the audited C01 control bank.  A2 is trained only
on C10 tokens that differ from C00, with C00 as its reference-preservation
control.  Official MUSE evaluation records are never loaded by preparation or
training; evaluation remains isolated in ``muse_recap_pilot.py``.
"""
import argparse
import concurrent.futures as futures
import difflib
import fcntl
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import authorize_muse_recap_full1024 as authorization
import muse_recap_pilot as base


VERSION = "muse-full-forget-coverage-v1"
CORPORA = ("News", "Books")
DEPTH = 8
RANK = 64
PREFIX_TOKENS = 1024
TARGET_TOKENS = 128
TARGET_STRIDE = 128
MICRO_BATCHES = 4
A1_KL = 1.2
A2_KL = 0.8
A2_EPOCHS = 6
LR = 2e-4
POINTS = {
    "verylight": (0.25, 0.25, 0.001),
    "light": (0.50, 0.50, 0.001),
    "a1light": (0.75, 0.50, 0.001),
    "strong": (1.00, 0.75, 0.001),
}


def raw_arrow_path(corpus):
    slug = corpus.lower()
    pattern = str(
        Path.home()
        / ".cache/huggingface/datasets"
        / f"muse-bench___muse-{slug}/raw/*/*"
        / f"muse-{slug}-forget.arrow"
    )
    paths = sorted(glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one cached raw forget Arrow file for {corpus}: {paths}")
    return Path(paths[0])


def load_raw_documents(corpus):
    from datasets import Dataset

    dataset = Dataset.from_file(str(raw_arrow_path(corpus)))
    if "text" not in dataset.column_names or not len(dataset):
        raise ValueError(f"{corpus}: cached raw forget dataset is invalid")
    values = list(dataset["text"])
    if not all(isinstance(text, str) for text in values):
        raise ValueError(f"{corpus}: raw forget dataset contains non-string text")
    # The official raw split contains a small number of empty records.  They
    # carry no trainable tokens and the original 512-row sampler also skipped
    # them when it required at least 70 words.
    documents = [text for text in values if text.strip()]
    if not documents:
        raise ValueError(f"{corpus}: raw forget dataset has no non-empty documents")
    return documents


def document_digest(documents):
    digest = hashlib.sha256()
    for text in documents:
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def long_completion_windows(tokenizer, documents):
    windows = []
    token_count = target_count = 0
    for document_index, text in enumerate(documents):
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        token_count += len(ids)
        starts = list(range(8, len(ids) - 15, TARGET_STRIDE))
        tail_start = max(8, len(ids) - TARGET_TOKENS)
        if starts and tail_start > starts[-1]:
            starts.append(tail_start)
        for target_start in starts:
            left = max(0, target_start - PREFIX_TOKENS)
            end = min(len(ids), target_start + TARGET_TOKENS)
            if end - target_start < 16:
                continue
            input_ids = ids[left:end]
            labels = [-100] * (target_start - left) + ids[target_start:end]
            windows.append((input_ids, labels, document_index, target_start))
            target_count += end - target_start
    if not windows:
        raise ValueError("No long-context completion windows")
    return windows, {"documents": len(documents), "tokens": token_count,
                     "windows": len(windows), "target_tokens": target_count}


def short_control_views(tokenizer, rows, cell):
    result = []
    for row in rows:
        ids = tokenizer(row["cells"][cell], add_special_tokens=True)["input_ids"]
        for fraction in (0.35, 0.65):
            cut = max(8, min(len(ids) - 16, round(len(ids) * fraction)))
            end = min(len(ids), cut + TARGET_TOKENS)
            if end - cut >= 16:
                result.append(ids[:end])
    if not result:
        raise ValueError(f"No control views for {cell}")
    return result


def changed_token_example(tokenizer, positive, control):
    positive_ids = tokenizer(positive, add_special_tokens=True)["input_ids"]
    control_ids = tokenizer(control, add_special_tokens=True)["input_ids"]
    matcher = difflib.SequenceMatcher(a=positive_ids, b=control_ids, autojunk=False)
    matched = set()
    for left, _, size in matcher.get_matching_blocks():
        matched.update(range(left, left + size))
    labels = [token if index not in matched and index > 0 else -100
              for index, token in enumerate(positive_ids)]
    if not any(label != -100 for label in labels):
        raise ValueError("C10/C00 pair has no changed target token")
    return positive_ids, labels, control_ids


def validate_source(source):
    protected = {}
    for corpus in CORPORA:
        folder = source / corpus
        rows = base.read(folder / "data.json")
        audit = base.read(folder / "audit.json")
        auth = base.read(folder / "training_authorization.json")
        authorization.validate_rows(rows, audit, auth, corpus)
        for name in ("data.json", "audit.json", "training_authorization.json"):
            path = folder / name
            protected[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return protected


def prepare(args):
    args.run.mkdir(parents=True, exist_ok=True)
    plan_path = args.run / "PLAN.json"
    raw = {}
    for corpus in CORPORA:
        documents = load_raw_documents(corpus)
        raw[corpus] = {
            "documents": len(documents),
            "characters": sum(len(text) for text in documents),
            "sha256": document_digest(documents),
            "arrow": str(raw_arrow_path(corpus)),
        }
    identity = {
        "version": VERSION,
        "source_run": str(args.source_run),
        "depth": DEPTH,
        "rank": RANK,
        "prefix_tokens": PREFIX_TOKENS,
        "target_tokens": TARGET_TOKENS,
        "target_stride": TARGET_STRIDE,
        "a1_kl": A1_KL,
        "a2_kl": A2_KL,
        "a2_epochs": A2_EPOCHS,
        "lr": LR,
        "points": POINTS,
        "raw_forget": raw,
        "training_retain_access": False,
        "official_eval_examples_used_for_training": False,
    }
    if plan_path.exists():
        plan = base.read(plan_path)
        for key, value in identity.items():
            if plan.get(key) != json.loads(json.dumps(value)):
                raise ValueError(f"Immutable plan differs at {key}")
        return plan
    if any(path.name != "run.lock" for path in args.run.iterdir()):
        raise ValueError("Unrecognized existing output; refusing overwrite")
    plan = dict(identity, protected_sha256=validate_source(args.source_run))
    (args.run / "assets").symlink_to(args.source_run / "assets", target_is_directory=True)
    marker = dict(base.read(args.source_run / "EXPERIMENT_AUTHORIZATION.json"),
                  derived_experiment=VERSION, raw_forget_full_coverage=True,
                  api_calls=0, training_retain_access=False)
    base.write(args.run / "EXPERIMENT_AUTHORIZATION.json", marker)
    for corpus in CORPORA:
        folder = args.run / corpus
        (folder / "models").mkdir(parents=True)
        (folder / "eval_specs").mkdir()
        for name in ("data.json", "audit.json", "training_authorization.json"):
            shutil.copy2(args.source_run / corpus / name, folder / name)
    base.write(plan_path, plan)
    return plan


def kl_to_reference(model, input_ids):
    import torch
    import torch.nn.functional as F

    tensor = torch.tensor([input_ids], device="cuda")
    with torch.no_grad(), model.disable_adapter():
        model.eval()
        reference = model(input_ids=tensor, use_cache=False).logits[:, :-1].float().softmax(-1)
    model.train()
    logits = model(input_ids=tensor, use_cache=False).logits[:, :-1].float()
    loss = F.kl_div(logits.log_softmax(-1), reference, reduction="batchmean")
    del tensor, logits, reference
    return loss


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(base.ROOT / "dual_uld_muse"))
    from train_assistant import find_lora_targets, slice_layers

    folder = args.run / args.corpus
    rows = base.read(folder / "data.json")
    authorization.validate_rows(rows, base.read(folder / "audit.json"),
                                base.read(folder / "training_authorization.json"), args.corpus)
    plan = base.read(args.run / "PLAN.json")
    documents = load_raw_documents(args.corpus)
    if document_digest(documents) != plan["raw_forget"][args.corpus]["sha256"]:
        raise ValueError("Raw forget corpus changed after preparation")

    output = folder / "models" / args.role
    tokenizer = AutoTokenizer.from_pretrained(args.run / "assets/tokenizer")
    if args.role == "a1":
        examples, coverage = long_completion_windows(tokenizer, documents)
        controls = short_control_views(tokenizer, rows, "C01")
        steps = math.ceil(len(examples) / MICRO_BATCHES)
        epochs = 1
        kl_weight = A1_KL
    else:
        examples = [changed_token_example(tokenizer, row["cells"]["C10"], row["cells"]["C00"])
                    for row in rows]
        controls = None
        coverage = {
            "pairs": len(examples),
            "changed_target_tokens": sum(sum(label != -100 for label in item[1]) for item in examples),
        }
        steps = math.ceil(len(examples) / MICRO_BATCHES) * A2_EPOCHS
        epochs = A2_EPOCHS
        kl_weight = A2_KL
    spec = {
        "version": VERSION, "corpus": args.corpus, "role": args.role,
        "depth": DEPTH, "rank": RANK, "steps": steps, "epochs": epochs,
        "lr": LR, "kl": kl_weight, "coverage": coverage, "seed": 42,
        "raw_forget_sha256": plan["raw_forget"][args.corpus]["sha256"],
        "factorial_data_sha256": base.digest(rows),
        "training_retain_access": False,
        "official_eval_examples_used_for_training": False,
    }
    complete = output / "complete.json"
    if complete.exists():
        if base.read(complete)["spec"] != json.loads(json.dumps(spec)):
            raise ValueError("Completed checkpoint specification differs")
        base.event(f"coverage_train_reuse corpus={args.corpus} role={args.role}")
        return
    if (output / "spec.json").exists() and base.read(output / "spec.json") != json.loads(json.dumps(spec)):
        raise ValueError("Training specification changed; use a new run")
    output.mkdir(parents=True, exist_ok=True)
    base.write(output / "spec.json", spec)

    torch.manual_seed(42)
    full = AutoModelForCausalLM.from_pretrained(
        args.run / "assets" / args.corpus,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    small = slice_layers(full, DEPTH)
    del full
    small.config.use_cache = False
    small.save_pretrained(output / "fullmodel")
    tokenizer.save_pretrained(output / "fullmodel")
    model = get_peft_model(small, LoraConfig(
        r=RANK, lora_alpha=2 * RANK,
        target_modules=find_lora_targets(small), lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
    )).cuda()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    rng = random.Random(42)
    order = []
    control_order = []
    started = time.time()
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        ce_values = []
        selected_controls = []
        for _ in range(MICRO_BATCHES):
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            item = examples[order.pop()]
            if args.role == "a1":
                input_ids, labels = item[0], item[1]
                if not control_order:
                    control_order = list(range(len(controls)))
                    rng.shuffle(control_order)
                selected_controls.append(controls[control_order.pop()])
            else:
                input_ids, labels, control_ids = item
                selected_controls.append(control_ids)
            input_tensor = torch.tensor([input_ids], device="cuda")
            label_tensor = torch.tensor([labels], device="cuda")
            ce = model(input_ids=input_tensor, labels=label_tensor, use_cache=False).loss
            (ce / MICRO_BATCHES).backward()
            ce_values.append(ce.item())
            del input_tensor, label_tensor, ce
        # One preservation example per four positive windows prevents the
        # control distribution from dominating full-corpus coverage.
        control = selected_controls[rng.randrange(len(selected_controls))]
        kl = kl_to_reference(model, control)
        (kl_weight * kl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        warmup = max(1, round(steps * 0.1))
        for group in optimizer.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / warmup)
        optimizer.step()
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == steps:
            base.event(
                f"coverage_train_step corpus={args.corpus} role={args.role} "
                f"step={step+1}/{steps} ce={sum(ce_values)/len(ce_values):.4f} kl={kl.item():.4f}"
            )
        del kl
    model.save_pretrained(output / "checkpoint-final")
    base.write(complete, {"spec": spec, "elapsed_seconds": time.time() - started})
    base.event(f"coverage_train_done corpus={args.corpus} role={args.role}")


def child(args, mode, corpus, item, values, gpu):
    label = values.get("role", values.get("point"))
    log = args.run / "logs" / f"{mode}_{corpus}_{label}.log"
    log.parent.mkdir(exist_ok=True)
    if mode == "train":
        command = [args.train_python, "-u", str(Path(__file__).resolve()), "train",
                   "--run", str(args.run), "--corpus", corpus, "--role", values["role"]]
    else:
        command = [args.train_python, "-u", str(Path(base.__file__).resolve()), "evaluate",
                   "--run", str(args.run), "--corpus", corpus,
                   "--eval-python", args.eval_python]
        for key, value in values.items():
            command.extend(["--" + key, str(value)])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                       TOKENIZERS_PARALLELISM="false")
    base.event(f"coverage_{mode}_start corpus={corpus} item={item} GPU={gpu}")
    with log.open("a") as stream:
        subprocess.run(command, check=True, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT)
    base.event(f"coverage_{mode}_done corpus={corpus} item={item} GPU={gpu}")


def collect(jobs):
    errors = []
    for job in futures.as_completed(jobs):
        try:
            job.result()
        except Exception as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError(" | ".join(errors))


def summarize(args):
    rows = []
    for corpus in CORPORA:
        for point in POINTS:
            spec_path = args.run / corpus / "eval_specs" / f"{point}.json"
            if not spec_path.exists():
                continue
            report = Path(base.read(spec_path)["report"])
            if base.valid_summary(report):
                rows.append({"corpus": corpus, "point": point, "report": str(report),
                             **base.read(report)})
    lines = [
        "# MUSE full-forget coverage experiment", "",
        "Training retain access: false; official evaluation examples used for training: false.",
        "A1 uses the entire raw forget corpus; A2 uses masked audited factorial controls.", "",
        "| Corpus | Point | VerbMem ↓ | KnowMem ↓ | UtilPres ↑ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['corpus']} | {row['point']} | "
            + " | ".join(f"{row[metric]:.6f}" for metric in base.METRICS) + " |"
        )
    base.write(args.run / "RESULTS.json", rows)
    (args.run / "RESULTS.md").write_text("\n".join(lines) + "\n")
    base.event(f"coverage_results completed={len(rows)}/{len(CORPORA)*len(POINTS)}")


def gpu_ready():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = [int(value.strip()) for value in line.split(",")]
        states[index] = (memory, utilization)
    return all(index in states and states[index][0] < 1024 and states[index][1] < 10
               for index in range(4))


def run(args):
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)
        if args.preflight_only:
            base.event("coverage_preflight_ok")
            return
        while not gpu_ready():
            base.event("coverage_wait reason=GPUs_not_idle")
            time.sleep(30)
        base.event("[1/3] Train full-coverage A1 and masked-factorial A2")
        specs = [("News", "a1"), ("News", "a2"), ("Books", "a1"), ("Books", "a2")]
        with futures.ThreadPoolExecutor(max_workers=4) as pool:
            collect([pool.submit(child, args, "train", corpus, role, {"role": role}, gpu)
                     for gpu, (corpus, role) in enumerate(specs)])
        base.event("[2/3] Evaluate four static reference-delta points per corpus")
        evaluations = [(corpus, point, values) for corpus in CORPORA
                       for point, values in POINTS.items()]
        with futures.ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(evaluations), 4):
                jobs = []
                for gpu, (corpus, point, (w1, w2, top_filter)) in enumerate(evaluations[start:start+4]):
                    values = {"point": point, "w1": w1, "w2": w2, "filter": top_filter}
                    jobs.append(pool.submit(child, args, "evaluate", corpus, point, values, gpu))
                try:
                    collect(jobs)
                finally:
                    summarize(args)
        base.event("[3/3] Full-coverage experiment complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "prepare", "train", "summarize"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train-python", default=sys.executable)
    parser.add_argument("--eval-python", default=str(Path.home() / "miniconda3/envs/ease-f2r-eval/bin/python"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--corpus", choices=CORPORA, default="News")
    parser.add_argument("--role", choices=("a1", "a2"), default="a1")
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.source_run is not None:
        args.source_run = args.source_run.resolve()
    if args.mode in ("run", "prepare") and args.source_run is None:
        parser.error("--source-run is required")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
