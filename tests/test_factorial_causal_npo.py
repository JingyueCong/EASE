import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


losses = load_module(
    "factorial_causal_npo_losses",
    ROOT / "ULD/uld/model/forget_losses.py",
)


class FixedModel(torch.nn.Module):
    def __init__(self, logits, trainable=True):
        super().__init__()
        if trainable:
            self.logits = torch.nn.Parameter(logits.clone())
        else:
            self.register_buffer("logits", logits.clone())

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.logits[: input_ids.shape[0]])


class FactorialCausalNPOTest(unittest.TestCase):
    def make_batch(self):
        size, seq, vocab = 4, 3, 5
        labels = torch.full((size, seq), -100, dtype=torch.long)
        labels[:, 1] = 2
        return torch.zeros(size, seq, vocab), {
            "input_ids": torch.zeros(size, seq, dtype=torch.long),
            "attention_mask": torch.ones(size, seq, dtype=torch.long),
            "labels": labels,
            "pair_ids": torch.zeros(size, dtype=torch.long),
            "cell_ids": torch.arange(4, dtype=torch.long),
        }

    def test_loss_is_finite_and_backpropagates(self):
        logits, batch = self.make_batch()
        model = FixedModel(logits)
        oracle = FixedModel(torch.zeros_like(logits), trainable=False)
        objective = losses.FactorialCausalNPOLoss(
            beta=0.1,
            control_kl_weight=0.5,
            locality_weight=0.1,
            locality_margin=0.05,
        )
        result = objective(model, batch, oracle)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(model.logits.grad)
        self.assertGreater(float(model.logits.grad.abs().sum()), 0.0)

    def test_control_kl_penalizes_control_drift(self):
        logits, batch = self.make_batch()
        drifted = logits.clone()
        drifted[1:, 0, 0] = 5.0
        oracle = FixedModel(logits, trainable=False)
        no_control = losses.FactorialCausalNPOLoss(
            control_kl_weight=0.0, locality_weight=0.0
        )(FixedModel(drifted), batch, oracle)["loss"]
        with_control = losses.FactorialCausalNPOLoss(
            control_kl_weight=1.0, locality_weight=0.0
        )(FixedModel(drifted), batch, oracle)["loss"]
        self.assertGreater(float(with_control), float(no_control))

    def test_factory_and_oracle_requirement(self):
        config = {
            "loss_type": "factorial_causal_npo",
            "beta": 0.2,
            "control_kl_weight": 1.0,
        }
        objective = losses.create_unlearn_loss(config)
        self.assertIsInstance(objective, losses.FactorialCausalNPOLoss)
        self.assertEqual(objective.beta, 0.2)
        self.assertTrue(losses.loss_requries_oracle(config))

    def test_rejects_incomplete_group(self):
        logits, batch = self.make_batch()
        batch["cell_ids"][-1] = 2
        with self.assertRaisesRegex(ValueError, "cells 0..3 exactly once"):
            losses.FactorialCausalNPOLoss()(
                FixedModel(logits),
                batch,
                FixedModel(logits, trainable=False),
            )


if __name__ == "__main__":
    unittest.main()
