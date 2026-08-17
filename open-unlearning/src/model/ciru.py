"""CIRU hidden-state intervention estimated from four-cell DiD controls."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import LlamaForCausalLM


logger = logging.getLogger("model.ciru")


def parse_layer_alphas(value: Optional[str]) -> Dict[int, float]:
    """Parse ``layer:alpha`` pairs separated by slash, comma, or semicolon."""
    if value is None or not str(value).strip():
        return {}
    parsed: Dict[int, float] = {}
    for item in re.split(r"[/,;]", str(value)):
        item = item.strip()
        if not item:
            continue
        try:
            layer_text, alpha_text = item.split(":", 1)
            layer = int(layer_text)
            alpha = float(alpha_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ciru_layer_alphas must contain layer:alpha pairs, "
                "for example 8:0.75/12:1.0/15:1.5"
            ) from exc
        if layer in parsed:
            raise ValueError(f"Duplicate CIRU layer alpha for layer {layer}")
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"Invalid CIRU alpha for layer {layer}: {alpha}")
        parsed[layer] = alpha
    return parsed


class CIRUForCausalLM(LlamaForCausalLM):
    """Llama model with low-rank causal-residual removal at selected layers."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        ciru_artifact_path: str = None,
        ciru_alpha: float = 1.0,
        ciru_layer_alphas: Optional[str] = None,
        ciru_gate_enabled: bool = True,
        **kwargs,
    ):
        if not ciru_artifact_path or ciru_artifact_path == "???":
            raise ValueError("CIRU requires model.model_args.ciru_artifact_path")
        artifact_path = Path(ciru_artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"CIRU artifact not found: {artifact_path}")

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model._ciru_alpha = float(ciru_alpha)
        model._ciru_layer_alphas = parse_layer_alphas(ciru_layer_alphas)
        model._ciru_gate_enabled = bool(ciru_gate_enabled)
        model._ciru_layer_state = {}

        with np.load(artifact_path, allow_pickle=False) as artifact:
            layers = [int(value) for value in artifact["layers"].tolist()]
            for layer in layers:
                basis = np.asarray(artifact[f"layer_{layer}_basis"], dtype=np.float32)
                center = np.asarray(artifact[f"layer_{layer}_center"], dtype=np.float32)
                if basis.ndim != 2 or center.shape != (model.config.hidden_size,):
                    raise ValueError(f"Invalid CIRU layer {layer} artifact shapes")
                if basis.shape[0] != model.config.hidden_size:
                    raise ValueError(
                        f"CIRU hidden-size mismatch at layer {layer}: {basis.shape[0]} "
                        f"vs {model.config.hidden_size}"
                    )
                names = {
                    "basis": f"_ciru_basis_{layer}",
                    "center": f"_ciru_center_{layer}",
                    "gate_mean": f"_ciru_gate_mean_{layer}",
                    "gate_std": f"_ciru_gate_std_{layer}",
                    "gate_coef": f"_ciru_gate_coef_{layer}",
                    "gate_intercept": f"_ciru_gate_intercept_{layer}",
                }
                model.register_buffer(names["basis"], torch.from_numpy(basis), persistent=False)
                model.register_buffer(names["center"], torch.from_numpy(center), persistent=False)
                for short, artifact_name in (
                    ("gate_mean", "gate_feature_mean"),
                    ("gate_std", "gate_feature_std"),
                    ("gate_coef", "gate_coefficient"),
                    ("gate_intercept", "gate_intercept"),
                ):
                    value = float(np.asarray(artifact[f"layer_{layer}_{artifact_name}"]).item())
                    model.register_buffer(
                        names[short], torch.tensor(value, dtype=torch.float32), persistent=False
                    )
                model._ciru_layer_state[layer] = names

        unknown_alpha_layers = set(model._ciru_layer_alphas) - set(model._ciru_layer_state)
        if unknown_alpha_layers:
            raise ValueError(
                "CIRU layer-specific alpha refers to layer(s) absent from the artifact: "
                + ", ".join(str(layer) for layer in sorted(unknown_alpha_layers))
            )

        model._ciru_hook_handles = []
        for layer, names in model._ciru_layer_state.items():
            if layer < 0 or layer >= len(model.model.layers):
                raise ValueError(f"CIRU artifact layer {layer} is outside the model")
            handle = model.model.layers[layer].register_forward_hook(
                model._make_ciru_hook(layer, names)
            )
            model._ciru_hook_handles.append(handle)

        logger.info(
            "CIRU ready: base=%s artifact=%s layers=%s alpha=%s layer_alphas=%s gate=%s",
            pretrained_model_name_or_path,
            artifact_path,
            sorted(model._ciru_layer_state),
            model._ciru_alpha,
            model._ciru_layer_alphas,
            model._ciru_gate_enabled,
        )
        return model

    def _make_ciru_hook(self, layer: int, names):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            basis = getattr(self, names["basis"]).to(hidden.device, hidden.dtype)
            center = getattr(self, names["center"]).to(hidden.device, hidden.dtype)
            coordinates = (hidden - center) @ basis
            projected = coordinates @ basis.transpose(0, 1)
            if self._ciru_gate_enabled:
                energy = torch.linalg.vector_norm(coordinates.float(), dim=-1)
                energy = energy / math.sqrt(max(basis.shape[1], 1))
                feature = torch.log1p(energy)
                mean = getattr(self, names["gate_mean"]).to(feature.device)
                std = getattr(self, names["gate_std"]).to(feature.device).clamp_min(1e-6)
                coef = getattr(self, names["gate_coef"]).to(feature.device)
                intercept = getattr(self, names["gate_intercept"]).to(feature.device)
                gate = torch.sigmoid(coef * ((feature - mean) / std) + intercept)
                projected = projected * gate.to(projected.dtype).unsqueeze(-1)
            layer_alpha = getattr(self, "_ciru_layer_alphas", {}).get(
                layer, self._ciru_alpha
            )
            modified = hidden - layer_alpha * projected
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        hook.__name__ = f"ciru_layer_{layer}_hook"
        return hook
