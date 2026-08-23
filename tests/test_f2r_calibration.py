import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "open-unlearning" / "src" / "model" / "f2r_calibration.py"
spec = importlib.util.spec_from_file_location("f2r_calibration", MODULE_PATH)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class F2RCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.a1 = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]])
        self.a2 = torch.tensor([[[0.5, 1.0, 0.0, 0.0]]])
        self.active = torch.tensor([[[True, True, False, False]]])
        self.state = {
            "alignment_scale": torch.tensor([2.0, 2.0, 1.0, 1.0]),
            "gate_coef": torch.zeros(6),
            "gate_intercept": torch.tensor(0.0),
            "feature_mean": torch.zeros(6),
            "feature_std": torch.ones(6),
        }

    def test_baseline_is_exact_weighted_sum(self):
        delta, gate = calibration.calibrated_residual(
            self.a1,
            self.a2,
            self.active,
            -1.2,
            0.4,
            None,
            alignment_enabled=False,
            gate_enabled=False,
        )
        torch.testing.assert_close(delta, -1.2 * self.a1 + 0.4 * self.a2)
        self.assertIsNone(gate)

    def test_reference_delta_removes_shared_initialization(self):
        reference = torch.tensor([[[0.4, 0.8, 0.2, -0.1]]])
        a1 = reference + torch.tensor([[[0.1, -0.2, 0.0, 0.3]]])
        a2 = reference + torch.tensor([[[-0.3, 0.1, 0.2, 0.0]]])
        delta1, delta2 = calibration.assistant_components(
            a1,
            a2,
            reference,
            composition_mode="reference_delta",
        )
        torch.testing.assert_close(delta1, a1 - reference)
        torch.testing.assert_close(delta2, a2 - reference)

    def test_raw_components_remain_backward_compatible(self):
        component1, component2 = calibration.assistant_components(
            self.a1, self.a2, composition_mode="raw"
        )
        self.assertIs(component1, self.a1)
        self.assertIs(component2, self.a2)

    def test_reference_delta_requires_reference(self):
        with self.assertRaisesRegex(ValueError, "requires reference logits"):
            calibration.assistant_components(
                self.a1, self.a2, composition_mode="reference_delta"
            )

    def test_alignment_changes_only_centered_active_logits(self):
        delta, _ = calibration.calibrated_residual(
            self.a1,
            self.a2,
            self.active,
            -1.0,
            1.0,
            self.state,
            alignment_enabled=True,
            gate_enabled=False,
        )
        centered = calibration.masked_center(self.a2, self.active)
        expected_a2 = self.a2 + centered
        torch.testing.assert_close(delta, -self.a1 + expected_a2)
        self.assertEqual(delta[..., 2:].abs().sum().item(), 0.0)

    def test_zero_logistic_gate_halves_residual(self):
        raw = -self.a1 + self.a2
        delta, gate = calibration.calibrated_residual(
            self.a1,
            self.a2,
            self.active,
            -1.0,
            1.0,
            self.state,
            alignment_enabled=False,
            gate_enabled=True,
        )
        torch.testing.assert_close(gate, torch.full_like(gate, 0.5))
        torch.testing.assert_close(delta, raw * 0.5)

    def test_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.npz"
            np.savez_compressed(
                path,
                alignment_scale=np.ones(4, dtype=np.float32),
                gate_coef=np.zeros(6, dtype=np.float32),
                gate_intercept=np.asarray(0.0, dtype=np.float32),
                feature_mean=np.zeros(6, dtype=np.float32),
                feature_std=np.ones(6, dtype=np.float32),
            )
            loaded = calibration.load_calibration(
                str(path), device=torch.device("cpu"), dtype=torch.float32, vocab_size=4
            )
            self.assertEqual(tuple(loaded["gate_coef"].shape), (6,))


if __name__ == "__main__":
    unittest.main()
