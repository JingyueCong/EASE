import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_f2d_forget10_1b_rank3.sh"


class Forget10OneBRankSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_approved_forget10_data(self):
        self.assertIn("forget10_author_hybrid400_seed42_v512reused_v511new_incremental_manualfix_v1_full.jsonl", self.text)
        self.assertIn("forget10_ciru400_seed42_full_authorblock_incremental_v1.jsonl", self.text)
        self.assertIn("APPROVED for downstream training", self.text)
        self.assertIn("rows=400 deterministic_errors=0", self.text)

    def test_is_one_b_and_isolated(self):
        self.assertIn("TRAIN_MODEL_CONFIG=llama-3-1b", self.text)
        self.assertIn("Llama-3.2-1B-Instruct_DualULD", self.text)
        self.assertIn("f2d_forget10_1b_rank3_seed42", self.text)
        self.assertIn("forget10_f2d_1b_rank3_seed42", self.text)

    def test_trains_three_rank_pairs_at_fixed_exposure(self):
        self.assertIn("for rank in 16 32 64", self.text)
        self.assertIn("A1_TRAIN_STEPS=192", self.text)
        self.assertIn("A2_TRAIN_STEPS=144", self.text)
        self.assertIn("A1_TRAIN_BS=2", self.text)
        self.assertIn("A2_TRAIN_BS=2", self.text)
        self.assertIn("TRAIN_LOSS_CONFIG=factorial_reference_preserving", self.text)

    def test_uses_static_reference_delta_without_router(self):
        self.assertIn("A1_NUM_LAYER=4 A2_NUM_LAYER=4", self.text)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.text)
        self.assertIn("GATE_ENABLED=false", self.text)

    def test_evaluates_twelve_points_and_keeps_retain_out_of_training(self):
        self.assertIn("4 static points per rank (12 total)", self.text)
        self.assertIn("retain policy  : absent from training", self.text)
        self.assertIn("SELECTION_RETAIN_ACCESS=true", self.text)
        self.assertNotIn('CF_PATH="$RETAIN_LOGS_PATH"', self.text)

    def test_target_matches_forget10_one_b_bss(self):
        self.assertIn('TARGET_AGG="${TARGET_AGG:-0.61}"', self.text)


if __name__ == "__main__":
    unittest.main()
