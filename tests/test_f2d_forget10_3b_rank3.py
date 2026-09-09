import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget103BRank3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            ROOT / "scripts/sweep_f2d_forget10_3b_rank3.sh"
        ).read_text()

    def test_uses_approved_400_row_forget10_artifacts(self):
        self.assertIn("expected = [f\"forget10_perturbed-", self.script)
        self.assertIn("range(400)", self.script)
        self.assertIn("units_with_deterministic_errors", self.script)
        self.assertIn("APPROVED for downstream training", self.script)

    def test_isolated_llama_3b_outputs(self):
        self.assertIn("f2d_forget10_3b_rank3_seed42", self.script)
        self.assertIn("forget10_f2d_3b_rank3_seed42", self.script)
        self.assertIn("TRAIN_MODEL_CONFIG=llama-3-3b", self.script)
        self.assertIn(
            "EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD",
            self.script,
        )

    def test_sweeps_rank_not_seed_or_router(self):
        self.assertIn("for rank in 16 32 64", self.script)
        self.assertIn('A1_LORA_ALPHA="$((2 * rank))"', self.script)
        self.assertIn('A2_LORA_ALPHA="$((2 * rank))"', self.script)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.script)
        self.assertNotIn("average_lora_delta_adapters.py", self.script)

    def test_uses_scale_matched_training_budget(self):
        self.assertIn("A1_TRAIN_STEPS=192", self.script)
        self.assertIn("A2_TRAIN_STEPS=144", self.script)
        self.assertIn("A1_RETAIN_WEIGHT=0.4", self.script)
        self.assertIn("A2_RETAIN_WEIGHT=0.3", self.script)
        self.assertIn(
            "TRAIN_LOSS_CONFIG=factorial_reference_preserving",
            self.script,
        )

    def test_static_reference_delta_grid_is_preregistered(self):
        self.assertIn("COMPOSITION_MODE=reference_delta", self.script)
        self.assertIn('"light:-1.6:1.8"', self.script)
        self.assertIn('"balanced:-1.9:2.0"', self.script)
        self.assertIn('"utility:-2.1:2.2"', self.script)
        self.assertIn('"memory:-2.3:2.3"', self.script)
        self.assertIn("expected 12 reports", self.script)
        self.assertIn("TARGET_AGG:-0.63", self.script)


if __name__ == "__main__":
    unittest.main()
