import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "model" / "forget_losses.py"
spec = importlib.util.spec_from_file_location("f2d_losses", MODULE_PATH)
losses = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(losses)


class TinyModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch = input_ids.shape[0]
        return type("Output", (), {"logits": self.logits.expand(batch, -1, -1)})


class FactorialHierarchicalLossTest(unittest.TestCase):
    def batch(self):
        return {
            "input_ids": torch.tensor([[0, 1, 2], [0, 2, 1]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2], [-100, 2, 1]]),
            "retainlabels": torch.tensor([0, 1]),
            "answer_mask": torch.tensor([[0., 1., 1.], [0., 1., 1.]]),
            "claim_mask": torch.tensor([[0., 1., 0.], [0., 1., 0.]]),
            "evidence_mask": torch.tensor([[0., 1., 0.], [0., 1., 0.]]),
        }

    def test_masked_objective_backpropagates(self):
        model = TinyModel(torch.zeros(1, 3, 4))
        objective = losses.FactorialHierarchicalLoss(
            retain_weight=1.0,
            preserve_kl_weight=0.0,
            evidence_weight=1.0,
        )
        result = objective(model, self.batch())
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(model.logits.grad)

    def test_preservation_requires_oracle(self):
        model = TinyModel(torch.zeros(1, 3, 4))
        objective = losses.FactorialHierarchicalLoss(preserve_kl_weight=0.1)
        with self.assertRaisesRegex(ValueError, "requires a frozen oracle"):
            objective(model, self.batch())

    def test_preservation_kl_is_finite(self):
        model = TinyModel(torch.zeros(1, 3, 4))
        oracle_logits = torch.zeros(1, 3, 4)
        oracle_logits[..., 1] = 2.0
        oracle = TinyModel(oracle_logits)
        objective = losses.FactorialHierarchicalLoss(preserve_kl_weight=0.1)
        result = objective(model, self.batch(), oracle_model=oracle)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(float(result["retain_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
