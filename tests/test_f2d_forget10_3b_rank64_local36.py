import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_f2d_forget10_3b_rank64_local36.sh"


class Forget10ThreeBRank64LocalSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_is_evaluation_only_and_isolated(self):
        self.assertNotIn("TRAIN_ONLY=true", self.text)
        self.assertIn("retraining=false", self.text)
        self.assertIn("forget10_f2d_3b_rank64_local36_seed42", self.text)
        self.assertIn("RESUME=\"$RESUME\"", self.text)

    def test_reuses_exact_rank64_pair(self):
        self.assertIn('checkpoint_at "$MODEL_ROOT/a1" 192', self.text)
        self.assertIn('checkpoint_at "$MODEL_ROOT/a2" 144', self.text)
        self.assertIn("A1_LORA_R=64", self.text)
        self.assertIn("A2_LORA_R=64", self.text)
        self.assertIn("A1_LORA_ALPHA=128", self.text)
        self.assertIn("A2_LORA_ALPHA=128", self.text)
        self.assertIn("target_modules=all_linear", self.text)

    def test_uses_static_reference_delta_without_routing(self):
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertIn('A1_REFERENCE_PATH="$A1_REFERENCE"', self.text)
        self.assertIn('A2_REFERENCE_PATH="$A2_REFERENCE"', self.text)
        self.assertIn("ALIGNMENT_ENABLED=false", self.text)
        self.assertIn("GATE_ENABLED=false", self.text)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.text)

    def test_grid_has_36_points_near_rank64_optimum(self):
        self.assertIn('WEIGHT_A1_GRID="-1.55 -1.60 -1.65 -1.70"', self.text)
        self.assertIn('WEIGHT_A2_GRID="1.75 1.85 1.95"', self.text)
        self.assertIn('TOP_FILTERS="0.00012 0.00017 0.00022"', self.text)
        self.assertIn("evaluations    : 36", self.text)
        self.assertIn("TARGET_AGG=\"${TARGET_AGG:-0.64}\"", self.text)

    def test_uses_forget10_three_b_protocol(self):
        self.assertIn("SPLIT=forget10", self.text)
        self.assertIn("TRAIN_MODEL_CONFIG=llama-3-3b", self.text)
        self.assertIn("Llama-3.2-3B-Instruct_DualULD", self.text)
        self.assertIn("SELECTION_RETAIN_ACCESS=true", self.text)


if __name__ == "__main__":
    unittest.main()
