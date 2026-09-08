#!/usr/bin/env python3
"""Average PEFT LoRA adapters in effective-delta space.

Directly averaging LoRA A/B factors is not invariant to the factor basis.  This
tool instead averages B@A and refactorizes the mean with a compact QR/SVD.  It
can either project back to the source rank or retain the union subspace by
using ``--target-rank source_rank * number_of_adapters``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-rank", required=True, type=int)
    parser.add_argument(
        "--target-alpha",
        type=int,
        help="Defaults to target_rank times the source alpha/r ratio.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_files(directory: Path) -> tuple[Path, Path]:
    config = directory / "adapter_config.json"
    weights = directory / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError(f"not a complete safetensors PEFT adapter: {directory}")
    return config, weights


def compatible_config(config: dict) -> dict:
    """Return the LoRA fields that affect the represented delta.

    PEFT writes ``base_model_name_or_path`` as the local sibling ``fullmodel``
    directory used by each training run.  Independently trained replicas of
    the same sliced model therefore have different path strings even though
    their adapter tensors are mathematically compatible.  Comparing the full
    serialized config incorrectly rejects those replicas.

    Tensor names and shapes are checked separately below.  Here we retain the
    structural LoRA options that change how A/B factors are interpreted, while
    deliberately excluding provenance and training-only fields such as the
    local base path, revision, inference mode, dropout, and initialization
    policy.
    """
    structural_fields = (
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "fan_in_fan_out",
        "bias",
        "use_rslora",
        "use_dora",
        "target_modules",
        "target_parameters",
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
    )
    normalized = {key: config.get(key) for key in structural_fields}
    if isinstance(normalized["target_modules"], list):
        normalized["target_modules"] = sorted(normalized["target_modules"])
    if isinstance(normalized["target_parameters"], list):
        normalized["target_parameters"] = sorted(normalized["target_parameters"])
    if isinstance(normalized["modules_to_save"], list):
        normalized["modules_to_save"] = sorted(normalized["modules_to_save"])
    return normalized


def factor_mean_delta(
    a_tensors: list[torch.Tensor],
    b_tensors: list[torch.Tensor],
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if len(a_tensors) != len(b_tensors) or not a_tensors:
        raise ValueError("A/B tensor lists must be non-empty and equally sized")
    source_rank, input_width = a_tensors[0].shape
    output_width, b_rank = b_tensors[0].shape
    if source_rank != b_rank:
        raise ValueError("LoRA A/B rank mismatch")
    for a_tensor, b_tensor in zip(a_tensors, b_tensors):
        if tuple(a_tensor.shape) != (source_rank, input_width):
            raise ValueError("LoRA A shapes differ across adapters")
        if tuple(b_tensor.shape) != (output_width, source_rank):
            raise ValueError("LoRA B shapes differ across adapters")

    work_a = [tensor.detach().to(device="cpu", dtype=torch.float64) for tensor in a_tensors]
    work_b = [tensor.detach().to(device="cpu", dtype=torch.float64) for tensor in b_tensors]
    b_cat = torch.cat(work_b, dim=1)
    a_cat = torch.cat(work_a, dim=0) / len(work_a)

    # B_cat @ A_cat is the mean effective (unscaled) LoRA delta.  The QR
    # factors reduce the expensive SVD to at most (n * source_rank)^2.
    q_b, r_b = torch.linalg.qr(b_cat, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_cat.T, mode="reduced")
    u, singular, vh = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)

    kept = min(target_rank, singular.numel())
    root_s = singular[:kept].sqrt()
    new_b = (q_b @ u[:, :kept]) * root_s.unsqueeze(0)
    new_a = root_s.unsqueeze(1) * (vh[:kept] @ q_a.T)

    if target_rank > kept:
        new_b = torch.cat(
            [new_b, torch.zeros(output_width, target_rank - kept, dtype=new_b.dtype)],
            dim=1,
        )
        new_a = torch.cat(
            [new_a, torch.zeros(target_rank - kept, input_width, dtype=new_a.dtype)],
            dim=0,
        )

    energy = singular.square().sum().item()
    discarded = singular[kept:].square().sum().item()
    relative_error = math.sqrt(discarded / energy) if energy else 0.0
    return new_a, new_b, relative_error


def average_adapters(
    adapter_dirs: list[Path], output: Path, target_rank: int, target_alpha: int | None
) -> dict:
    if len(adapter_dirs) < 2:
        raise ValueError("at least two adapters are required")
    if target_rank <= 0:
        raise ValueError("target rank must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output}")

    configs: list[dict] = []
    states: list[dict[str, torch.Tensor]] = []
    sources = []
    for directory in adapter_dirs:
        config_path, weights_path = adapter_files(directory)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs.append(config)
        states.append(load_file(str(weights_path), device="cpu"))
        sources.append(
            {
                "adapter": str(directory.resolve()),
                "weights_sha256": sha256(weights_path),
            }
        )

    reference = compatible_config(configs[0])
    for index, config in enumerate(configs[1:], start=2):
        candidate = compatible_config(config)
        if candidate != reference:
            mismatches = {
                key: (reference.get(key), candidate.get(key))
                for key in reference
                if reference.get(key) != candidate.get(key)
            }
            raise ValueError(
                f"adapter config {index} is incompatible with adapter 1: "
                + json.dumps(mismatches, sort_keys=True)
            )
    source_rank = int(configs[0]["r"])
    source_alpha = int(configs[0]["lora_alpha"])
    ratio = source_alpha / source_rank
    if target_alpha is None:
        target_alpha = round(target_rank * ratio)
    if not math.isclose(target_alpha / target_rank, ratio, rel_tol=0, abs_tol=1e-12):
        raise ValueError("target alpha/r must equal source alpha/r to preserve delta scale")

    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("adapter state dictionaries have different keys")
    a_keys = sorted(key for key in keys if ".lora_A." in key)
    b_keys = sorted(key for key in keys if ".lora_B." in key)
    if not a_keys or len(a_keys) != len(b_keys):
        raise ValueError("could not find paired LoRA A/B tensors")

    output_state: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}
    consumed: set[str] = set()
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key not in keys:
            raise ValueError(f"missing paired tensor for {a_key}")
        new_a, new_b, error = factor_mean_delta(
            [state[a_key] for state in states],
            [state[b_key] for state in states],
            target_rank,
        )
        output_state[a_key] = new_a.to(dtype=states[0][a_key].dtype).contiguous()
        output_state[b_key] = new_b.to(dtype=states[0][b_key].dtype).contiguous()
        errors[a_key.rsplit(".lora_A.", 1)[0]] = error
        consumed.update((a_key, b_key))

    for key in sorted(keys - consumed):
        tensors = [state[key] for state in states]
        if tensors[0].is_floating_point():
            output_state[key] = torch.stack(
                [tensor.to(torch.float64) for tensor in tensors]
            ).mean(0).to(tensors[0].dtype)
        elif all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
            output_state[key] = tensors[0]
        else:
            raise ValueError(f"non-floating tensor differs across adapters: {key}")

    output.mkdir(parents=True, exist_ok=True)
    output_config = dict(configs[0])
    output_config.update({"r": target_rank, "lora_alpha": target_alpha, "inference_mode": True})
    (output / "adapter_config.json").write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_file(output_state, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    readme = adapter_dirs[0] / "README.md"
    if readme.is_file():
        shutil.copy2(readme, output / "README.md")

    metadata = {
        "method": "mean_effective_lora_delta_qr_svd",
        "source_count": len(adapter_dirs),
        "source_rank": source_rank,
        "source_alpha": source_alpha,
        "target_rank": target_rank,
        "target_alpha": target_alpha,
        "scale_alpha_over_rank": ratio,
        "target_modules": output_config.get("target_modules"),
        "sources": sources,
        "relative_projection_error_mean": sum(errors.values()) / len(errors),
        "relative_projection_error_max": max(errors.values()),
        "per_module_relative_projection_error": errors,
    }
    (output / "SOUP_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = average_adapters(
        args.adapter, args.output, args.target_rank, args.target_alpha
    )
    print(
        "lora_delta_soup_ready "
        f"sources={metadata['source_count']} rank={metadata['target_rank']} "
        f"alpha={metadata['target_alpha']} "
        f"mean_projection_error={metadata['relative_projection_error_mean']:.8g} "
        f"max_projection_error={metadata['relative_projection_error_max']:.8g}"
    )


if __name__ == "__main__":
    main()
