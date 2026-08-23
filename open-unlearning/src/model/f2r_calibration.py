"""Lightweight residual alignment and gating for F2R.

The calibration artifact is learned only from the forget set and generated
counterfactual controls.  It contains a vocabulary-diagonal A2 alignment map
and/or a six-feature logistic gate.  Keeping this module independent from the
model loader makes the numerical path easy to unit test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


FEATURE_NAMES = (
    "residual_rms",
    "residual_max",
    "a1_rms",
    "a2_rms",
    "a1_a2_cosine",
    "a2_to_a1_rms",
)


def assistant_components(
    a1_logits: torch.Tensor,
    a2_logits: torch.Tensor,
    reference_logits: torch.Tensor | None = None,
    *,
    composition_mode: str = "raw",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the two tensors that enter the weighted DualULD composition.

    ``raw`` preserves the original implementation exactly.  ``reference_delta``
    removes the shared frozen small-assistant initialization so asymmetric
    weights cannot accidentally inject ``(w1 + w2) * reference_logits``.
    """
    if composition_mode == "raw":
        return a1_logits, a2_logits
    if composition_mode != "reference_delta":
        raise ValueError(f"Unsupported DualULD composition mode: {composition_mode}")
    if reference_logits is None:
        raise ValueError("reference_delta composition requires reference logits")
    if reference_logits.shape != a1_logits.shape or a1_logits.shape != a2_logits.shape:
        raise ValueError(
            "reference_delta logits must have identical shapes: "
            f"reference={tuple(reference_logits.shape)} "
            f"a1={tuple(a1_logits.shape)} a2={tuple(a2_logits.shape)}"
        )
    return a1_logits - reference_logits, a2_logits - reference_logits


def masked_center(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Center logits over active vocabulary entries and zero inactive ones."""
    weights = active.to(logits.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (logits * weights).sum(dim=-1, keepdim=True) / count
    return (logits - mean) * weights


def residual_features(
    delta: torch.Tensor,
    a1_logits: torch.Tensor,
    a2_logits: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return the per-position features used by the learned F2R gate."""
    weights = active.to(delta.dtype)
    count = weights.sum(dim=-1).clamp_min(1.0)
    a1_centered = masked_center(a1_logits, active)
    a2_centered = masked_center(a2_logits, active)
    delta_centered = masked_center(delta, active)

    def rms(value: torch.Tensor) -> torch.Tensor:
        return ((value.square() * weights).sum(dim=-1) / count).clamp_min(0).sqrt()

    delta_rms = rms(delta_centered)
    a1_rms = rms(a1_centered)
    a2_rms = rms(a2_centered)
    delta_max = delta_centered.abs().masked_fill(~active, 0.0).amax(dim=-1)
    dot = (a1_centered * a2_centered * weights).sum(dim=-1)
    denom = (a1_rms * a2_rms * count).clamp_min(1e-6)
    cosine = (dot / denom).clamp(-1.0, 1.0)
    ratio = a2_rms / a1_rms.clamp_min(1e-6)
    return torch.stack(
        (delta_rms, delta_max, a1_rms, a2_rms, cosine, ratio), dim=-1
    )


def load_calibration(
    path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
) -> Dict[str, torch.Tensor]:
    """Load and validate an ``.npz`` calibration artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"F2R calibration artifact not found: {path}")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {
            "alignment_scale",
            "gate_coef",
            "gate_intercept",
            "feature_mean",
            "feature_std",
        }
        missing = sorted(required.difference(artifact.files))
        if missing:
            raise ValueError(f"Invalid F2R calibration artifact; missing: {missing}")
        arrays = {name: np.asarray(artifact[name]) for name in required}

    if arrays["alignment_scale"].shape != (vocab_size,):
        raise ValueError(
            "F2R alignment vocabulary mismatch: "
            f"expected {(vocab_size,)}, got {arrays['alignment_scale'].shape}"
        )
    feature_shape = (len(FEATURE_NAMES),)
    for name in ("gate_coef", "feature_mean", "feature_std"):
        if arrays[name].shape != feature_shape:
            raise ValueError(
                f"F2R {name} shape mismatch: expected {feature_shape}, "
                f"got {arrays[name].shape}"
            )

    state = {}
    for name, array in arrays.items():
        tensor_dtype = dtype if name == "alignment_scale" else torch.float32
        # Older torch builds paired with newer NumPy cannot always consume a
        # zero-dimensional ndarray through as_tensor().  Materialise the one
        # scalar explicitly while keeping vector loads zero-copy where possible.
        value = array.item() if array.ndim == 0 else array
        state[name] = torch.as_tensor(value, dtype=tensor_dtype, device=device)
    state["feature_std"] = state["feature_std"].clamp_min(1e-6)
    return state


def calibrated_residual(
    a1_logits: torch.Tensor,
    a2_logits: torch.Tensor,
    active: torch.Tensor,
    weight_a1: float,
    weight_a2: float,
    state: Dict[str, torch.Tensor] | None,
    *,
    alignment_enabled: bool,
    gate_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Compose the F2R residual with optional alignment and learned gate."""
    adjusted_a2 = a2_logits
    if alignment_enabled:
        if state is None:
            raise ValueError("alignment_enabled requires a calibration artifact")
        scale = state["alignment_scale"].to(a2_logits.device, a2_logits.dtype)
        a2_centered = masked_center(a2_logits, active)
        adjusted_a2 = a2_logits + (scale - 1.0) * a2_centered

    delta = float(weight_a1) * a1_logits + float(weight_a2) * adjusted_a2
    gate = None
    if gate_enabled:
        if state is None:
            raise ValueError("gate_enabled requires a calibration artifact")
        features = residual_features(delta, a1_logits, adjusted_a2, active).float()
        mean = state["feature_mean"].to(features.device)
        std = state["feature_std"].to(features.device)
        coef = state["gate_coef"].to(features.device)
        intercept = state["gate_intercept"].to(features.device).reshape(())
        gate_logits = ((features - mean) / std * coef).sum(dim=-1) + intercept
        gate = torch.sigmoid(gate_logits).to(delta.dtype)
        delta = delta * gate.unsqueeze(-1)
    return delta, gate
