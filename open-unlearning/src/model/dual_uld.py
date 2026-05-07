"""Dual-ULD wrapper for open-unlearning.

At inference time:
    final_logits = base_logits
                 + weight_a1 * filtered(A1.logits)
                 + weight_a2 * filtered(A2.logits)

A1 is trained on (forget ∪ R_sub) with `remember+uniform` loss; weight_a1 < 0
subtracts its memorisation. A2 is trained on R_sub only with the same loss;
weight_a2 > 0 adds R_sub back so on R_sub the two assistants cancel and base
is preserved. See ULD/configs/data_mode/dual_a{1,2}.yaml.

Mirrors ULDForCausalLM (src/model/uld.py) for two assistants. KV cache is
disabled because base and assistants have different layer counts; for TOFU's
short sequences the O(n^2) cost is acceptable.
"""
import os
import logging
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger("model.dual_uld")


def _relative_top_filter(scores: torch.Tensor, relative_top: float):
    """Same filter as src/model/uld.py:_relative_top_filter (topk variant to
    avoid OOM on long-sequence generation)."""
    min_tokens_to_keep = max(int(relative_top * scores.shape[-1]), 1)
    scores_normalized = scores.log_softmax(dim=-1)
    min_thresh = torch.topk(scores_normalized, min_tokens_to_keep, dim=-1).values[..., -1]
    probs_max = scores_normalized.amax(dim=-1)
    probs_thresh = probs_max + float(np.log(relative_top))
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    mask = scores_normalized < probs_thresh
    return scores, mask


def _is_shared_base_assistant(assistant_path: str) -> bool:
    """Returns True if this assistant was trained as a LoRA on the *same* base
    as the inference base (no slicing / no per-assistant fullmodel/ copy).
    Marker: parent dir contains a USES_SHARED_BASE file."""
    parent = os.path.dirname(assistant_path)
    return os.path.isfile(os.path.join(parent, "USES_SHARED_BASE"))


def _load_assistant(
    assistant_path: str,
    torch_dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
) -> torch.nn.Module:
    extra = {}
    if torch_dtype is not None:
        extra["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation

    adapter_cfg = os.path.join(assistant_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        full_dir = os.path.normpath(os.path.join(assistant_path, "..", "fullmodel"))
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(
                f"Found LoRA adapter at {assistant_path} but no sibling "
                f"`fullmodel/` directory at {full_dir}."
            )
        from peft import PeftModel

        logger.info(f"DualULD: loading assistant fullmodel from {full_dir}")
        base = AutoModelForCausalLM.from_pretrained(full_dir, **extra)
        logger.info(f"DualULD: merging LoRA adapter from {assistant_path}")
        peft = PeftModel.from_pretrained(base, assistant_path, **extra)
        return peft.merge_and_unload()

    logger.info(f"DualULD: loading assistant (merged) from {assistant_path}")
    return AutoModelForCausalLM.from_pretrained(assistant_path, **extra)


class DualULDForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM whose forward() output is the dual-assistant logit sum."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        a1_path: str = None,
        a2_path: str = None,
        weight_a1: float = -1.0,
        weight_a2: float = 1.0,
        top_logit_filter: float = 0.1,
        **kwargs,
    ):
        for name, val in (("a1_path", a1_path), ("a2_path", a2_path)):
            if val is None or val == "???":
                raise ValueError(f"DualULDForCausalLM.from_pretrained requires `{name}`.")

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        device = next(model.parameters()).device

        torch_dtype = kwargs.get("torch_dtype", None)
        attn_impl = kwargs.get("attn_implementation", None)

        # Plan B path: A1 and A2 are LoRA on the *same* base as `model`.
        # Load them as named adapters on `model` (peft) so we don't need to
        # copy 7B weights twice. Forward pass swaps adapters.
        share_a1 = _is_shared_base_assistant(a1_path)
        share_a2 = _is_shared_base_assistant(a2_path)
        shared_peft = None
        a1 = a2 = None
        if share_a1 or share_a2:
            from peft import PeftModel
            logger.info(f"DualULD: shared-base mode for "
                        f"a1={'YES' if share_a1 else 'NO'} a2={'YES' if share_a2 else 'NO'}")
            shared_peft = PeftModel.from_pretrained(
                model, a1_path if share_a1 else a2_path,
                adapter_name=("a1" if share_a1 else "a2"),
            )
            if share_a1 and share_a2:
                shared_peft.load_adapter(a2_path, adapter_name="a2")
            elif share_a1 and not share_a2:
                a2 = _load_assistant(a2_path, torch_dtype, attn_impl)
                a2.to(device); a2.eval()
            elif share_a2 and not share_a1:
                a1 = _load_assistant(a1_path, torch_dtype, attn_impl)
                a1.to(device); a1.eval()
                shared_peft.set_adapter("a2")  # default to a2
            # `model` is now wrapped by PeftModel; set adapter to none for base forward
            shared_peft.eval()
        else:
            a1 = _load_assistant(a1_path, torch_dtype, attn_impl)
            a2 = _load_assistant(a2_path, torch_dtype, attn_impl)
            a1.to(device); a1.eval()
            a2.to(device); a2.eval()

        object.__setattr__(model, "_dual_a1", a1)
        object.__setattr__(model, "_dual_a2", a2)
        object.__setattr__(model, "_dual_shared_peft", shared_peft)
        object.__setattr__(model, "_dual_share_a1", share_a1)
        object.__setattr__(model, "_dual_share_a2", share_a2)
        model._dual_w1 = float(weight_a1)
        model._dual_w2 = float(weight_a2)
        model._dual_top_filter = float(top_logit_filter)

        model.generation_config.use_cache = False
        model.config.use_cache = False

        logger.info(
            f"DualULDForCausalLM ready: base={pretrained_model_name_or_path} "
            f"a1={a1_path} a2={a2_path} "
            f"weight_a1={weight_a1} weight_a2={weight_a2} "
            f"top_logit_filter={top_logit_filter}"
        )
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        **kwargs,
    ):
        common = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        shared_peft = self._dual_shared_peft
        if shared_peft is not None:
            # Shared-base mode: peft wrapped `self`. Calling shared_peft(...)
            # would recurse back into this forward, so bypass via the parent
            # LlamaForCausalLM.forward, which respects LoRA layers swapped via
            # set_adapter / disable_adapter context.
            with shared_peft.disable_adapter():
                base_out = LlamaForCausalLM.forward(self, **common)
            if self._dual_share_a1:
                shared_peft.set_adapter("a1")
                a1_out = LlamaForCausalLM.forward(self, **common)
            else:
                a1_out = self._dual_a1(**common)
            if self._dual_share_a2:
                shared_peft.set_adapter("a2")
                a2_out = LlamaForCausalLM.forward(self, **common)
            else:
                a2_out = self._dual_a2(**common)
        else:
            base_out = super().forward(**common)
            a1_out   = self._dual_a1(**common)
            a2_out   = self._dual_a2(**common)

        base_logits = base_out.logits
        a1_logits   = a1_out.logits.to(base_logits.device)
        a2_logits   = a2_out.logits.to(base_logits.device)

        if self._dual_top_filter > 0.0:
            base_logits, mask = _relative_top_filter(base_logits, self._dual_top_filter)
            a1_logits = a1_logits.clone(); a1_logits[mask] = 0.0
            a2_logits = a2_logits.clone(); a2_logits[mask] = 0.0

        logits = base_logits + self._dual_w1 * a1_logits + self._dual_w2 * a2_logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None,
            hidden_states=None, attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        kwargs.pop("past_key_values", None)
        kwargs["use_cache"] = False
        out = super().prepare_inputs_for_generation(
            input_ids, past_key_values=None, **kwargs
        )
        out["input_ids"] = input_ids
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            out["attention_mask"] = kwargs["attention_mask"]
        out["past_key_values"] = None
        out["use_cache"] = False
        return out
