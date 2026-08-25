import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget013BStrength12Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sweep = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget01_3b_strength12.sh"
        ).read_text()
        cls.weights = (ROOT / "scripts/sweep_f2r_weights.sh").read_text()

    def test_scan_is_frozen_and_3b_specific(self):
        self.assertIn("retraining     : none", self.sweep)
        self.assertIn("TRAIN_MODEL_CONFIG=llama-3-3b", self.sweep)
        self.assertIn(
            "EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD", self.sweep
        )
        self.assertIn("A1_TRAIN_STEPS=32 A2_TRAIN_STEPS=24", self.sweep)

    def test_scan_expands_the_observed_boundary(self):
        self.assertIn('WEIGHT_A1_GRID="-2.0 -2.2 -2.4 -2.6"', self.sweep)
        self.assertIn('WEIGHT_A2_GRID="1.5 1.7 1.9"', self.sweep)
        self.assertIn("TOP_FILTERS=0.0004", self.sweep)

    def test_scan_remains_static_and_retain_free(self):
        self.assertIn("COMPOSITION_MODE=reference_delta", self.sweep)
        self.assertIn("ALIGNMENT_ENABLED=false", self.sweep)
        self.assertIn("GATE_ENABLED=false", self.sweep)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.sweep)
        self.assertNotIn("hf_forget_train.py", self.sweep)

    def test_generic_weight_runner_has_parameterized_task_model(self):
        self.assertIn(
            'TASK_MODEL_NAME="${TASK_MODEL_NAME:-Llama-3.2-1B-Instruct}"',
            self.weights,
        )
        self.assertIn('task_name="tofu_${TASK_MODEL_NAME}_${SPLIT}', self.weights)


if __name__ == "__main__":
    unittest.main()
