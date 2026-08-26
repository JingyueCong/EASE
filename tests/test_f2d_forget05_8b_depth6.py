import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget058BDepth6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth6_forget05_8b.sh"
        ).read_text()

    def test_uses_isolated_depth6_artifacts(self):
        self.assertIn("f2d_v512_dual_depth6_8b_train_seed42", self.script)
        self.assertIn("f2d_v512_dual_depth6_8b_pilot_seed42", self.script)
        self.assertIn("old artifacts     : read-only; no overwrite", self.script)

    def test_both_assistants_are_depth6_and_reference_matched(self):
        self.assertIn("A1_NUM_LAYER=6", self.script)
        self.assertIn("A2_NUM_LAYER=6", self.script)
        self.assertIn("shared depth-6 reference_delta", self.script)
        self.assertIn('A1_REFERENCE_PATH="$reference"', self.script)
        self.assertIn('A2_REFERENCE_PATH="$reference"', self.script)

    def test_preregisters_exposure_and_conservative_budgets(self):
        self.assertIn('"exposure:56:48:5e-4:1e-3"', self.script)
        self.assertIn('"conservative:64:54:4e-4:8e-4"', self.script)
        self.assertIn("inference points  : 4 per training budget (8 total)", self.script)

    def test_remains_static_router_free_and_retain_free(self):
        self.assertIn("COMPOSITION_MODE=reference_delta", self.script)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.script)
        self.assertIn("GATE_ENABLED=false", self.script)
        self.assertNotIn("retain95", self.script.split("train_role()", 1)[1].split("echo \"[2/4]", 1)[0])


if __name__ == "__main__":
    unittest.main()
