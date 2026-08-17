#!/usr/bin/env python3
"""Estimate a CIRU DiD subspace and retain-free energy gate from 2x2 units."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "data" / "ciru.py"
spec = importlib.util.spec_from_file_location("ciru_data", MODULE_PATH)
ciru_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ciru_data)


def encode_qa(tokenizer, question: str, answer: str, max_length: int):
    system = {"role": "system", "content": "You are a helpful assistant."}
    user = {"role": "user", "content": question}
    assistant = {"role": "assistant", "content": answer}
    prompt = tokenizer.apply_chat_template(
        [system, user], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        [system, user, assistant], tokenize=True, add_generation_prompt=False
    )
    prompt = list(prompt)
    full = list(full)[:max_length]
    if len(full) < 2:
        raise ValueError("Tokenized CIRU cell is too short")
    if len(prompt) >= len(full) - 1:
        raise ValueError("max_length truncates the CIRU answer span")
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template prompt is not a prefix of the full QA")
    start = len(prompt)
    end = len(full) - 1
    return torch.tensor(full, dtype=torch.long), start, end


@torch.inference_mode()
def collect_representations(
    model,
    tokenizer,
    units: Sequence[Dict],
    layers: Sequence[int],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    encoded: List[Tuple[torch.Tensor, int, int]] = []
    for unit in units:
        for cell in ciru_data.CELLS:
            qa = unit["cells"][cell]
            encoded.append(encode_qa(tokenizer, qa["question"], qa["answer"], max_length))

    by_layer: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    pad_id = tokenizer.pad_token_id
    for start_index in range(0, len(encoded), batch_size):
        chunk = encoded[start_index : start_index + batch_size]
        ids = pad_sequence(
            [item[0] for item in chunk], batch_first=True, padding_value=pad_id
        ).to(device)
        lengths = torch.tensor(
            [len(item[0]) for item in chunk], device=device, dtype=torch.long
        )
        attention = torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]
        outputs = model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        for layer in layers:
            hidden = outputs.hidden_states[layer + 1].float()
            for batch_index, (_, answer_start, answer_end) in enumerate(chunk):
                pooled = hidden[batch_index, answer_start:answer_end].mean(dim=0)
                by_layer[layer].append(pooled.cpu().numpy())
        print(
            f"representation cells={min(start_index + len(chunk), len(encoded))}/{len(encoded)}",
            flush=True,
        )
    return {
        layer: np.stack(values).reshape(len(units), len(ciru_data.CELLS), -1)
        for layer, values in by_layer.items()
    }


def fit_scalar_logistic(feature: np.ndarray, labels: np.ndarray):
    mean = float(feature.mean())
    std = float(max(feature.std(), 1e-6))
    x = (feature - mean) / std
    coefficient = 0.0
    intercept = 0.0
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float((1.0 - labels).sum()), 1.0)
    weights = np.where(labels > 0.5, 0.5 / positives, 0.5 / negatives)
    for _ in range(2000):
        logits = np.clip(coefficient * x + intercept, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = (probability - labels) * weights
        grad_coefficient = float(np.sum(error * x)) + 1e-3 * coefficient
        grad_intercept = float(np.sum(error))
        coefficient -= 0.1 * grad_coefficient
        intercept -= 0.1 * grad_intercept
    return mean, std, coefficient, intercept


def estimate_layer(representations: np.ndarray, rank: int):
    # Cell order is fixed as C11, C01, C10, C00.
    c11, c01, c10, c00 = [representations[:, index, :] for index in range(4)]
    did = (c11 - c01) - (c10 - c00)
    did_center = did.mean(axis=0)
    # Decompose the raw unit-level interactions so the average treatment
    # direction is retained.  Centering before SVD would erase a perfectly
    # consistent causal direction—the strongest possible signal.
    _, singular_values, right = np.linalg.svd(did, full_matrices=False)
    effective_rank = min(rank, right.shape[0])
    basis = right[:effective_rank].T.astype(np.float32)
    control_center = np.concatenate([c01, c10, c00], axis=0).mean(axis=0).astype(np.float32)

    all_cells = representations.reshape(-1, representations.shape[-1])
    projected = (all_cells - control_center) @ basis
    energy = np.linalg.norm(projected, axis=1) / math.sqrt(max(effective_rank, 1))
    labels = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), len(representations))
    energy_feature = np.log1p(energy)
    gate_mean, gate_std, gate_coef, gate_intercept = fit_scalar_logistic(
        energy_feature, labels
    )
    explained = singular_values[:effective_rank] ** 2
    total = float(np.square(singular_values).sum())
    return {
        "basis": basis,
        "control_center": control_center,
        "did_mean": did_center.astype(np.float32),
        "singular_values": singular_values[:effective_rank].astype(np.float32),
        "explained_fraction": float(explained.sum() / max(total, 1e-12)),
        "gate_feature_mean": np.float32(gate_mean),
        "gate_feature_std": np.float32(gate_std),
        "gate_coefficient": np.float32(gate_coef),
        "gate_intercept": np.float32(gate_intercept),
        "positive_energy_mean": float(energy[labels > 0.5].mean()),
        "control_energy_mean": float(energy[labels < 0.5].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 12, 15])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=350)
    args = parser.parse_args()

    units = ciru_data.load_ciru_units(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    bad_layers = [layer for layer in args.layers if layer < 0 or layer >= num_layers]
    if bad_layers:
        raise ValueError(f"Invalid layers {bad_layers}; model has {num_layers} layers")

    representations = collect_representations(
        model,
        tokenizer,
        units,
        args.layers,
        args.batch_size,
        args.max_length,
        device,
    )
    arrays = {
        "layers": np.asarray(args.layers, dtype=np.int64),
        "rank": np.asarray(args.rank, dtype=np.int64),
    }
    diagnostics = {}
    for layer in args.layers:
        result = estimate_layer(representations[layer], args.rank)
        arrays[f"layer_{layer}_basis"] = result.pop("basis")
        arrays[f"layer_{layer}_center"] = result.pop("control_center")
        arrays[f"layer_{layer}_did_mean"] = result.pop("did_mean")
        arrays[f"layer_{layer}_singular_values"] = result.pop("singular_values")
        for name in (
            "gate_feature_mean",
            "gate_feature_std",
            "gate_coefficient",
            "gate_intercept",
        ):
            arrays[f"layer_{layer}_{name}"] = result.pop(name)
        diagnostics[str(layer)] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "method": "CIRU",
        "estimator": "four-cell representation DiD + raw-effect truncated SVD",
        "units": len(units),
        "retain_access": False,
        "selection_retain_access": False,
        "model": args.model,
        "data": str(args.data),
        "layers": args.layers,
        "rank": args.rank,
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
