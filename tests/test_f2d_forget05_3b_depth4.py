import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget053BDepth4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sweep = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget05_3b.sh"
        ).read_text()

    def test_uses_isolated_3b_artifacts(self):
        self.assertIn("f2d_did_3b_forget05_", self.sweep)
        self.assertIn("TRAIN_MODEL_CONFIG=llama-3-3b", self.sweep)
        self.assertIn(
            "EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD", self.sweep
        )
        self.assertIn("forget05_${SWEEP_NAME}", self.sweep)

    def test_preserves_frozen_data_and_hard_gates_coverage(self):
        self.assertIn("expected 200 unique forget05 rows", self.sweep)
        self.assertIn("units_with_deterministic_errors", self.sweep)
        self.assertIn("incomplete factorial cells", self.sweep)
        self.assertNotIn("audit_tofu_factorial.py --input", self.sweep)

    def test_trains_static_depth4_pair_without_router(self):
        self.assertIn("A1_NUM_LAYER=4", self.sweep)
        self.assertIn("A2_NUM_LAYER=4", self.sweep)
        self.assertIn("A1_TRAIN_STEPS=96", self.sweep)
        self.assertIn("A2_TRAIN_STEPS=72", self.sweep)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.sweep)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.sweep)

    def test_coarse_grid_covers_known_and_stronger_boundaries(self):
        self.assertIn('WEIGHT_A1_GRID="-1.7 -2.0 -2.4 -2.7"', self.sweep)
        self.assertIn('WEIGHT_A2_GRID="1.4 1.7 2.0"', self.sweep)
        self.assertIn("TOP_FILTERS=0.0004", self.sweep)
        self.assertIn("evaluations       : 12", self.sweep)

    def test_retain_data_is_selection_only(self):
        self.assertIn("retain training   : none", self.sweep)
        self.assertIn("SELECTION_RETAIN_ACCESS=true", self.sweep)
        self.assertIn("AUTO_FETCH_RETAIN_LOGS=0", self.sweep)


if __name__ == "__main__":
    unittest.main()
