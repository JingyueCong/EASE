import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "train_f2r_calibration.py"

# The tests below exercise only schema-to-example conversion. Stub the two
# runtime model modules so this lightweight test does not require the server's
# OmegaConf/Hydra evaluation environment.
previous_modules = {
    name: sys.modules.get(name)
    for name in ("model", "model.dual_uld", "model.f2r_calibration")
}
model_package = types.ModuleType("model")
dual_module = types.ModuleType("model.dual_uld")
dual_module._load_assistant = lambda *args, **kwargs: None
dual_module._relative_top_filter = lambda *args, **kwargs: None
calibration_module = types.ModuleType("model.f2r_calibration")
calibration_module.FEATURE_NAMES = ("f0",)
calibration_module.assistant_components = lambda a1, a2, *args, **kwargs: (a1, a2)
calibration_module.calibrated_residual = lambda *args, **kwargs: None
calibration_module.masked_center = lambda value, active: value
calibration_module.residual_features = lambda *args, **kwargs: None
sys.modules["model"] = model_package
sys.modules["model.dual_uld"] = dual_module
sys.modules["model.f2r_calibration"] = calibration_module

spec = importlib.util.spec_from_file_location("train_f2r_calibration", MODULE_PATH)
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)
for name, previous in previous_modules.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


class TrainF2RCalibrationExamplesTest(unittest.TestCase):
    def setUp(self):
        self.factorial = {
            "source_id": "forget05_perturbed-00000",
            "cells": {
                name: {"question": f"{name} question", "answer": f"{name} answer"}
                for name in ("C11", "C01", "C10", "C00")
            },
        }

    def test_factorial_gate_uses_c11_as_only_positive(self):
        examples = trainer.make_gate_examples([self.factorial])
        self.assertEqual([item.kind for item in examples], ["C11", "C01", "C10", "C00"])
        self.assertEqual([item.label for item in examples], [1, 0, 0, 0])

    def test_factorial_alignment_uses_all_three_controls(self):
        examples = trainer.make_alignment_examples([self.factorial])
        self.assertEqual([item.kind for item in examples], ["C01", "C10", "C00"])
        self.assertTrue(all(item.label == 0 for item in examples))

    def test_legacy_schema_remains_supported(self):
        record = {
            "source_id": "legacy-0",
            "source_question": "source q",
            "source_answer": "source a",
            "matched_question": "matched q",
            "matched_answer": "matched a",
            "mismatched_question": "mismatched q",
            "mismatched_answer": "mismatched a",
        }
        gate = trainer.make_gate_examples([record])
        alignment = trainer.make_alignment_examples([record])
        self.assertEqual([item.kind for item in gate], ["forget", "matched", "mismatched"])
        self.assertEqual([item.label for item in gate], [1, 0, 0])
        self.assertEqual([item.kind for item in alignment], ["matched"])

    def test_missing_factorial_control_is_rejected(self):
        del self.factorial["cells"]["C00"]
        with self.assertRaisesRegex(ValueError, "missing factorial cells"):
            trainer.make_gate_examples([self.factorial])
        with self.assertRaisesRegex(ValueError, "missing alignment cells"):
            trainer.make_alignment_examples([self.factorial])

    def test_scalar_rms_alignment_matches_weighted_energy(self):
        scale = trainer.rms_alignment_scale(
            4.0,
            1.0,
            target_multiplier=1.0,
            ridge=0.0,
            scale_min=0.25,
            scale_max=4.0,
        )
        self.assertEqual(scale, 2.0)

    def test_scalar_rms_alignment_requires_opposite_sign_weights(self):
        with self.assertRaisesRegex(ValueError, "opposite-sign"):
            trainer.rms_alignment_scale(
                1.0,
                1.0,
                target_multiplier=-1.0,
                ridge=0.0,
                scale_min=0.25,
                scale_max=4.0,
            )


if __name__ == "__main__":
    unittest.main()
