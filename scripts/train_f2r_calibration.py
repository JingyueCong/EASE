#!/usr/bin/env python3
"""Learn F2R residual alignment and/or a token-level gate.

No retain examples or retain metrics are used.  Alignment is fit on matched
counterfactual answers (C+).  The gate is a balanced logistic classifier whose
positive class is the forget answer and whose negative class is C+ / C-.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OPEN_UNLEARNING_SRC = ROOT / "open-unlearning" / "src"
sys.path.insert(0, str(OPEN_UNLEARNING_SRC))

from model.dual_uld import _load_assistant, _relative_top_filter  # noqa: E402
from model.f2r_calibration import (  # noqa: E402
    FEATURE_NAMES,
    calibrated_residual,
    masked_center,
    residual_features,
)


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str
    label: int
    kind: str


def load_records(path: Path, limit_records: int = 0) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit_records and len(records) >= limit_records:
                    break
    if not records:
        raise ValueError(f"No counterfactual records found in {path}")
    return records


def make_gate_examples(records: Sequence[Dict]) -> List[QAExample]:
    examples: List[QAExample] = []
    seen_sources = set()
    for record in records:
        source_id = record["source_id"]
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            examples.append(
                QAExample(
                    record["source_question"], record["source_answer"], 1, "forget"
                )
            )
        examples.append(
            QAExample(
                record["matched_question"], record["matched_answer"], 0, "matched"
            )
        )
        examples.append(
            QAExample(
                record["mismatched_question"],
                record["mismatched_answer"],
                0,
                "mismatched",
            )
        )
    return examples


def make_alignment_examples(records: Sequence[Dict]) -> List[QAExample]:
    return [
        QAExample(record["matched_question"], record["matched_answer"], 0, "matched")
        for record in records
    ]


def encode_example(tokenizer, example: QAExample, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    system = {"role": "system", "content": "You are a helpful assistant."}
    user = {"role": "user", "content": example.question}
    assistant = {"role": "assistant", "content": example.answer}
    prompt_ids = tokenizer.apply_chat_template(
        [system, user], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        [system, user, assistant], tokenize=True, add_generation_prompt=False
    )
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)[:max_length]
    if len(full_ids) < 2:
        raise ValueError("Tokenized example is too short")
    # Logit position t predicts token t+1. Include the prompt's final position,
    # which predicts the first assistant answer token.
    answer_start = min(max(len(prompt_ids) - 1, 0), len(full_ids) - 1)
    answer_mask = torch.zeros(len(full_ids), dtype=torch.bool)
    answer_mask[answer_start : len(full_ids) - 1] = True
    return torch.tensor(full_ids, dtype=torch.long), answer_mask


def batches(
    tokenizer,
    examples: Sequence[QAExample],
    batch_size: int,
    max_length: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[QAExample]]]:
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(examples), batch_size):
        chunk = list(examples[start : start + batch_size])
        encoded = [encode_example(tokenizer, item, max_length) for item in chunk]
        ids = pad_sequence([item[0] for item in encoded], batch_first=True, padding_value=pad_id)
        answer_masks = pad_sequence(
            [item[1] for item in encoded], batch_first=True, padding_value=False
        )
        attention_mask = ids.ne(pad_id)
        labels = torch.tensor([item.label for item in chunk], dtype=torch.float32)
        yield ids, attention_mask, answer_masks, labels, chunk


@torch.inference_mode()
def assistant_logits(base, a1, a2, ids, attention_mask):
    kwargs = dict(
        input_ids=ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return base(**kwargs).logits, a1(**kwargs).logits, a2(**kwargs).logits


def fit_alignment(
    base,
    a1,
    a2,
    tokenizer,
    examples: Sequence[QAExample],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    top_filter: float,
    weight_a1: float,
    weight_a2: float,
    ridge: float,
    min_observations: int,
    scale_min: float,
    scale_max: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if weight_a2 == 0:
        raise ValueError("Alignment requires a non-zero weight_a2")
    vocab_size = base.config.vocab_size
    numerator = torch.zeros(vocab_size, dtype=torch.float64)
    denominator = torch.zeros(vocab_size, dtype=torch.float64)
    observations = torch.zeros(vocab_size, dtype=torch.int64)
    target_multiplier = -float(weight_a1) / float(weight_a2)

    for batch_index, (ids, attention, answer_mask, _, _) in enumerate(
        batches(tokenizer, examples, batch_size, max_length), start=1
    ):
        ids, attention = ids.to(device), attention.to(device)
        answer_mask = answer_mask.to(device)
        base_logits, a1_logits, a2_logits = assistant_logits(base, a1, a2, ids, attention)
        _, inactive = _relative_top_filter(base_logits, top_filter)
        active = ~inactive
        a1_centered = masked_center(a1_logits, active)
        a2_centered = masked_center(a2_logits, active)
        selected = answer_mask & attention
        x = a2_centered[selected].float()
        y = target_multiplier * a1_centered[selected].float()
        valid = active[selected]
        numerator += (x * y).masked_fill(~valid, 0).sum(dim=0).double().cpu()
        denominator += x.square().masked_fill(~valid, 0).sum(dim=0).double().cpu()
        observations += valid.sum(dim=0).cpu()
        if batch_index % 25 == 0:
            print(f"alignment batches={batch_index}", flush=True)

    scale = (numerator + ridge) / (denominator + ridge)
    scale[observations < min_observations] = 1.0
    scale = scale.clamp(scale_min, scale_max).float().numpy()
    changed = observations.numpy() >= min_observations
    diagnostics = {
        "observed_vocabulary": int(changed.sum()),
        "scale_mean_observed": float(scale[changed].mean()) if changed.any() else 1.0,
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
    }
    return scale, diagnostics


@torch.inference_mode()
def collect_gate_features(
    base,
    a1,
    a2,
    tokenizer,
    examples: Sequence[QAExample],
    alignment_scale: np.ndarray,
    *,
    alignment_enabled: bool,
    device: torch.device,
    batch_size: int,
    max_length: int,
    top_filter: float,
    weight_a1: float,
    weight_a2: float,
    max_tokens_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    state = {
        "alignment_scale": torch.as_tensor(
            alignment_scale, device=device, dtype=next(base.parameters()).dtype
        )
    }
    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    kind_counts: Dict[str, int] = {}
    for batch_index, (ids, attention, answer_mask, labels, chunk) in enumerate(
        batches(tokenizer, examples, batch_size, max_length), start=1
    ):
        ids, attention = ids.to(device), attention.to(device)
        answer_mask = answer_mask.to(device)
        base_logits, a1_logits, a2_logits = assistant_logits(base, a1, a2, ids, attention)
        _, inactive = _relative_top_filter(base_logits, top_filter)
        active = ~inactive
        a1_logits = a1_logits.masked_fill(inactive, 0.0)
        a2_logits = a2_logits.masked_fill(inactive, 0.0)
        delta, _ = calibrated_residual(
            a1_logits,
            a2_logits,
            active,
            weight_a1,
            weight_a2,
            state,
            alignment_enabled=alignment_enabled,
            gate_enabled=False,
        )
        feature_a2 = a2_logits
        if alignment_enabled:
            scale = state["alignment_scale"].to(a2_logits.device, a2_logits.dtype)
            feature_a2 = a2_logits + (scale - 1.0) * masked_center(a2_logits, active)
        features = residual_features(delta, a1_logits, feature_a2, active)
        for row, example in enumerate(chunk):
            selected = answer_mask[row] & attention[row]
            count = int(selected.sum())
            if count:
                feature_chunks.append(features[row, selected].float().cpu())
                label_chunks.append(torch.full((count,), labels[row].item()))
                kind_counts[example.kind] = kind_counts.get(example.kind, 0) + count
        if batch_index % 25 == 0:
            print(f"gate-feature batches={batch_index}", flush=True)

    x = torch.cat(feature_chunks).numpy()
    y = torch.cat(label_chunks).numpy()
    rng = np.random.default_rng(seed)
    selected_indices = []
    for label in (0.0, 1.0):
        indices = np.flatnonzero(y == label)
        if len(indices) > max_tokens_per_class:
            indices = rng.choice(indices, size=max_tokens_per_class, replace=False)
        selected_indices.append(indices)
    keep = np.concatenate(selected_indices)
    rng.shuffle(keep)
    return x[keep], y[keep], kind_counts


def fit_logistic_gate(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Dict[str, float]]:
    torch.manual_seed(seed)
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).float()
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    xz = (x - mean) / std
    coef = torch.zeros(x.shape[1], requires_grad=True)
    intercept = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([coef, intercept], lr=learning_rate)
    positives = y.sum().clamp_min(1.0)
    negatives = (1.0 - y).sum().clamp_min(1.0)
    pos_weight = negatives / positives
    for _ in range(steps):
        optimizer.zero_grad()
        logits = xz @ coef + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        ) + l2 * coef.square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probability = torch.sigmoid(xz @ coef + intercept)
        prediction = probability >= 0.5
        positive_accuracy = prediction[y == 1].float().mean()
        negative_accuracy = (~prediction[y == 0]).float().mean()
        diagnostics = {
            "tokens": int(len(y)),
            "positive_tokens": int(y.sum().item()),
            "balanced_accuracy": float((positive_accuracy + negative_accuracy) / 2),
            "mean_gate_forget": float(probability[y == 1].mean()),
            "mean_gate_controls": float(probability[y == 0].mean()),
            "final_loss": float(loss),
        }
    return (
        coef.detach().numpy(),
        float(intercept.detach()),
        mean.numpy(),
        std.numpy(),
        diagnostics,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("alignment", "gate", "alignment-gate"), required=True)
    parser.add_argument("--counterfactual-path", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--a1-path", required=True)
    parser.add_argument("--a2-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-input", type=Path)
    parser.add_argument("--weight-a1", type=float, default=-1.2)
    parser.add_argument("--weight-a2", type=float, default=0.4)
    parser.add_argument("--top-filter", type=float, default=0.0025)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-tokens-per-class", type=int, default=30000)
    parser.add_argument("--alignment-ridge", type=float, default=10.0)
    parser.add_argument("--alignment-min-observations", type=int, default=8)
    parser.add_argument("--alignment-scale-min", type=float, default=0.25)
    parser.add_argument("--alignment-scale-max", type=float, default=4.0)
    parser.add_argument("--gate-steps", type=int, default=800)
    parser.add_argument("--gate-learning-rate", type=float, default=0.03)
    parser.add_argument("--gate-l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("F2R calibration training requires CUDA")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    records = load_records(args.counterfactual_path, args.limit_records)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading base model: {args.base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    print(f"Loading A1: {args.a1_path}", flush=True)
    a1 = _load_assistant(args.a1_path, dtype, "sdpa").to(device).eval()
    print(f"Loading A2: {args.a2_path}", flush=True)
    a2 = _load_assistant(args.a2_path, dtype, "sdpa").to(device).eval()

    vocab_size = base.config.vocab_size
    alignment_diagnostics: Dict[str, float] = {}
    if args.mode == "alignment":
        alignment_scale, alignment_diagnostics = fit_alignment(
            base,
            a1,
            a2,
            tokenizer,
            make_alignment_examples(records),
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_filter=args.top_filter,
            weight_a1=args.weight_a1,
            weight_a2=args.weight_a2,
            ridge=args.alignment_ridge,
            min_observations=args.alignment_min_observations,
            scale_min=args.alignment_scale_min,
            scale_max=args.alignment_scale_max,
        )
    elif args.mode == "alignment-gate":
        if not args.alignment_input:
            raise ValueError("alignment-gate requires --alignment-input")
        with np.load(args.alignment_input, allow_pickle=False) as artifact:
            alignment_scale = np.asarray(artifact["alignment_scale"], dtype=np.float32)
        if alignment_scale.shape != (vocab_size,):
            raise ValueError(
                f"Alignment input has shape {alignment_scale.shape}; "
                f"expected {(vocab_size,)}"
            )
    else:
        alignment_scale = np.ones(vocab_size, dtype=np.float32)

    gate_coef = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    gate_intercept = np.float32(0.0)
    feature_mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    feature_std = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    gate_diagnostics: Dict[str, float] = {}
    kind_counts: Dict[str, int] = {}
    if args.mode in ("gate", "alignment-gate"):
        features, labels, kind_counts = collect_gate_features(
            base,
            a1,
            a2,
            tokenizer,
            make_gate_examples(records),
            alignment_scale,
            alignment_enabled=args.mode == "alignment-gate",
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_filter=args.top_filter,
            weight_a1=args.weight_a1,
            weight_a2=args.weight_a2,
            max_tokens_per_class=args.max_tokens_per_class,
            seed=args.seed,
        )
        gate_coef, gate_intercept, feature_mean, feature_std, gate_diagnostics = (
            fit_logistic_gate(
                features,
                labels,
                steps=args.gate_steps,
                learning_rate=args.gate_learning_rate,
                l2=args.gate_l2,
                seed=args.seed,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        alignment_scale=alignment_scale.astype(np.float32),
        gate_coef=np.asarray(gate_coef, dtype=np.float32),
        gate_intercept=np.asarray(gate_intercept, dtype=np.float32),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
    )
    metadata = {
        "format_version": 1,
        "mode": args.mode,
        "counterfactual_path": str(args.counterfactual_path),
        "base_model": args.base_model,
        "a1_path": args.a1_path,
        "a2_path": args.a2_path,
        "weight_a1": args.weight_a1,
        "weight_a2": args.weight_a2,
        "top_filter": args.top_filter,
        "records": len(records),
        "feature_names": FEATURE_NAMES,
        "alignment": alignment_diagnostics,
        "gate": gate_diagnostics,
        "gate_kind_token_counts": kind_counts,
        "retain_access": False,
        "selection_retain_access": False,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Calibration: {args.output}")
    print(f"Metadata:    {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
