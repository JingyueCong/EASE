import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "open-unlearning" / "src" / "model" / "ciru.py"
spec = importlib.util.spec_from_file_location("ciru_model", MODULE_PATH)
ciru_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ciru_model)


class DummyCIRU(torch.nn.Module):
    _make_ciru_hook = ciru_model.CIRUForCausalLM._make_ciru_hook

    def __init__(self, gate_enabled):
        super().__init__()
        self._ciru_alpha = 1.0
        self._ciru_gate_enabled = gate_enabled
        self.register_buffer("basis", torch.tensor([[1.0], [0.0], [0.0], [0.0]]))
        self.register_buffer("center", torch.zeros(4))
        self.register_buffer("gate_mean", torch.tensor(0.0))
        self.register_buffer("gate_std", torch.tensor(1.0))
        self.register_buffer("gate_coef", torch.tensor(0.0))
        self.register_buffer("gate_intercept", torch.tensor(0.0))


NAMES = {
    "basis": "basis",
    "center": "center",
    "gate_mean": "gate_mean",
    "gate_std": "gate_std",
    "gate_coef": "gate_coef",
    "gate_intercept": "gate_intercept",
}


class CIRUModelTest(unittest.TestCase):
    def test_ungated_hook_removes_only_projected_direction(self):
        model = DummyCIRU(gate_enabled=False)
        hook = model._make_ciru_hook(0, NAMES)
        hidden = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])
        result = hook(None, None, hidden)
        torch.testing.assert_close(result, torch.tensor([[[0.0, 3.0, 4.0, 5.0]]]))

    def test_zero_logit_gate_applies_half_projection_and_preserves_tuple(self):
        model = DummyCIRU(gate_enabled=True)
        hook = model._make_ciru_hook(0, NAMES)
        hidden = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])
        cache = object()
        result = hook(None, None, (hidden, cache))
        torch.testing.assert_close(result[0], torch.tensor([[[1.0, 3.0, 4.0, 5.0]]]))
        self.assertIs(result[1], cache)


if __name__ == "__main__":
    unittest.main()
