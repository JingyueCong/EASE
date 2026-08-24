"""Request-scoped sequence routing for DualULD residuals.

The router matches tokenized deletion-request entities only in the user prompt.
It deliberately ignores response tokens so teacher-forced evaluation cannot leak
the reference answer into the routing decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_sequence_router(path: str, *, device: torch.device) -> Dict[str, object]:
    """Load and validate a token-subsequence router artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"F2R sequence-router artifact not found: {path}")
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    if data.get("kind") != "forget_entity_subsequence_v1":
        raise ValueError(f"Unsupported F2R sequence-router kind: {data.get('kind')}")

    raw_patterns = data.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise ValueError("F2R sequence-router artifact requires non-empty patterns")
    patterns = []
    for index, entry in enumerate(raw_patterns):
        token_ids = entry.get("token_ids") if isinstance(entry, dict) else None
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(not isinstance(token, int) or token < 0 for token in token_ids)
        ):
            raise ValueError(f"Invalid sequence-router pattern at index {index}")
        patterns.append(torch.tensor(token_ids, dtype=torch.long, device=device))

    prompt_end_ids = data.get("prompt_end_token_ids")
    if (
        not isinstance(prompt_end_ids, list)
        or not prompt_end_ids
        or any(not isinstance(token, int) or token < 0 for token in prompt_end_ids)
    ):
        raise ValueError("F2R sequence-router artifact requires prompt_end_token_ids")

    return {
        "patterns": tuple(patterns),
        "prompt_end_pattern": torch.tensor(
            prompt_end_ids, dtype=torch.long, device=device
        ),
        "match_scale": float(data.get("match_scale", 1.0)),
        "nonmatch_scale": float(data.get("nonmatch_scale", 0.0)),
        "entities": tuple(data.get("entities", ())),
    }


def _window_matches(
    input_ids: torch.Tensor,
    pattern: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Return a ``[batch, possible_start]`` exact-subsequence match tensor."""
    width = int(pattern.numel())
    if width == 0 or input_ids.shape[1] < width:
        return torch.zeros(
            (input_ids.shape[0], 0), dtype=torch.bool, device=input_ids.device
        )
    token_windows = input_ids.unfold(1, width, 1)
    valid_windows = valid_tokens.unfold(1, width, 1).all(dim=-1)
    return token_windows.eq(pattern.view(1, 1, width)).all(dim=-1) & valid_windows


def _prompt_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    prompt_end_pattern: torch.Tensor,
) -> torch.Tensor:
    """Identify prompt tokens without allowing answer-token routing leakage."""
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid = attention_mask.to(dtype=torch.bool, device=input_ids.device)

    if labels is not None:
        # Open-unlearning labels use -100 for the complete prompt and supervise
        # response tokens.  This is the most exact boundary during scoring.
        return valid & labels.to(input_ids.device).eq(-100)

    # Generation has no labels.  The full prompt ends at the assistant header;
    # because KV cache is disabled, this header remains in every generation call.
    marker_matches = _window_matches(input_ids, prompt_end_pattern, valid)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    prompt_mask = valid.clone()
    if marker_matches.shape[1]:
        sentinel = torch.full_like(marker_matches, input_ids.shape[1], dtype=torch.long)
        starts = torch.where(
            marker_matches,
            torch.arange(marker_matches.shape[1], device=input_ids.device).view(1, -1),
            sentinel,
        ).amin(dim=1)
        prompt_mask &= positions.view(1, -1) < starts.view(-1, 1)
    return prompt_mask


def sequence_router_scale(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    state: Dict[str, object],
    *,
    labels: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-sequence residual scales and entity-match decisions."""
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("sequence routing requires two-dimensional input_ids")
    prompt_mask = _prompt_token_mask(
        input_ids,
        attention_mask,
        labels,
        state["prompt_end_pattern"],
    )
    matched = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for pattern in state["patterns"]:
        matches = _window_matches(input_ids, pattern, prompt_mask)
        if matches.shape[1]:
            matched |= matches.any(dim=1)
    output_dtype = dtype or torch.float32
    match_scale = torch.as_tensor(
        state["match_scale"], device=input_ids.device, dtype=output_dtype
    )
    nonmatch_scale = torch.as_tensor(
        state["nonmatch_scale"], device=input_ids.device, dtype=output_dtype
    )
    scales = torch.where(matched, match_scale, nonmatch_scale)
    return scales, matched
