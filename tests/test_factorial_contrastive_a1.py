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
    "factorial_contrast_losses",
    ROOT / "ULD/uld/model/forget_losses.py",
)
class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.logits[: input_ids.shape[0]])


class FrozenModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits.clone())

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.logits[: input_ids.shape[0]])


class FactorialContrastiveA1Test(unittest.TestCase):
    def test_sampler_keeps_five_cells_per_source(self):
        text = (ROOT / "ULD/uld/data/datamodule.py").read_text()
        self.assertIn("class FactorialFiveSampler", text)
        self.assertIn("for cell_index in range(5)", text)
        self.assertIn("source_index + cell_index * self.num_units", text)

    def make_batch(self):
        batch = 5
        seq = 3
        vocab = 4
        logits = torch.zeros(batch, seq, vocab)
        # Correct answer token is 1. C11 is easy; cross negative is hard.
        logits[0, 0, 1] = 3.0
        logits[2, 0, 1] = -2.0
        logits[3, 0, 2] = 0.5
        logits[4, 0, 2] = -0.5
        labels = torch.full((batch, seq), -100, dtype=torch.long)
        labels[:, 1] = 1
        return logits, {
            "input_ids": torch.zeros(batch, seq, dtype=torch.long),
            "attention_mask": torch.ones(batch, seq, dtype=torch.long),
            "labels": labels,
            "pair_ids": torch.zeros(batch, dtype=torch.long),
            "cell_ids": torch.arange(5, dtype=torch.long),
        }

    def test_loss_is_finite_and_regularizers_are_active(self):
        logits, batch = self.make_batch()
        oracle = FrozenModel(torch.zeros_like(logits))
        base = losses.FactorialContrastiveA1Loss(
            contrast_weight=0.0, placebo_kl_weight=0.0
        )(FixedModel(logits), batch, oracle)["loss"]
        regularized = losses.FactorialContrastiveA1Loss(
            contrast_weight=0.3, placebo_kl_weight=0.03
        )(FixedModel(logits), batch, oracle)["loss"]
        self.assertTrue(torch.isfinite(regularized))
        self.assertGreater(float(regularized), float(base))

    def test_factory_requires_oracle(self):
        config = {
            "loss_type": "factorial_contrastive_a1",
            "retain_weight": 1.5,
        }
        objective = losses.create_unlearn_loss(config)
        self.assertIsInstance(objective, losses.FactorialContrastiveA1Loss)
        self.assertTrue(losses.loss_requries_oracle(config))

    def test_loss_rejects_incomplete_cell_group(self):
        logits, batch = self.make_batch()
        batch["cell_ids"][-1] = 3
        oracle = FrozenModel(torch.zeros_like(logits))
        objective = losses.FactorialContrastiveA1Loss()
        with self.assertRaisesRegex(ValueError, "cells 0..4 exactly once"):
            objective(FixedModel(logits), batch, oracle)


if __name__ == "__main__":
    unittest.main()
