import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "model" / "forget_losses.py"
spec = importlib.util.spec_from_file_location("answer_masked_losses", MODULE_PATH)
losses = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(losses)


class TinyModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch = input_ids.shape[0]
        return type("Output", (), {"logits": self.logits.expand(batch, -1, -1)})


class AnswerMaskedUniformLossTest(unittest.TestCase):
    def batch(self):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 3, -100]]),
        }

    def test_uniform_answer_logits_have_zero_kl(self):
        model = TinyModel(torch.zeros(1, 4, 5))
        loss = losses.AnswerMaskedUniformLossFunc(model, **self.batch())
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_prompt_and_non_predictive_logits_are_ignored(self):
        batch = self.batch()
        base = torch.zeros(1, 4, 5)
        changed = base.clone()
        changed[:, 0, 0] = 25.0  # predicts a prompt token: labels[:, 1] == -100
        changed[:, 2:, 1] = 25.0  # padding/final positions are not supervised

        first = losses.AnswerMaskedUniformLossFunc(TinyModel(base), **batch)
        second = losses.AnswerMaskedUniformLossFunc(TinyModel(changed), **batch)
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_answer_logits_change_loss_and_backpropagate(self):
        logits = torch.zeros(1, 4, 5)
        logits[:, 1, 2] = 8.0  # predicts labels[:, 2], an answer position
        model = TinyModel(logits)
        loss = losses.AnswerMaskedUniformLossFunc(model, **self.batch())
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(model.logits.grad)
        self.assertGreater(float(model.logits.grad[:, 1].abs().sum()), 0.0)
        self.assertEqual(float(model.logits.grad[:, 0].abs().sum()), 0.0)

    def test_factory_selects_answer_masked_retain_loss(self):
        objective = losses.create_unlearn_loss({
            "forget_loss": "GradDescentLossFunc",
            "retain_loss": "AnswerMaskedUniformLossFunc",
            "retain_weight": 1.0,
        })
        self.assertIs(
            objective.retain_loss_func,
            losses.AnswerMaskedUniformLossFunc,
        )


if __name__ == "__main__":
    unittest.main()
