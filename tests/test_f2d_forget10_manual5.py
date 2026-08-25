import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget10Manual5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.apply = (ROOT / "scripts/apply_f2d_forget10_manual5.py").read_text()
        cls.recover = (
            ROOT / "scripts/run_f2d_forget10_manual5_recover.sh"
        ).read_text()
        cls.base = (
            ROOT / "scripts/run_f2d_forget10_incremental400.sh"
        ).read_text()

    def test_repairs_exactly_five_exhausted_rows(self):
        for suffix in ("00035", "00043", "00095", "00120", "00181"):
            self.assertIn(f"forget10_perturbed-{suffix}", self.apply)
        self.assertIn("repairs={len(repaired)}", self.apply)

    def test_c01_repairs_follow_replacement_profile(self):
        self.assertIn("A Whisper at Dusk", self.apply)
        self.assertIn("Samir Tavakoli's mother's occupation", self.apply)
        self.assertIn("Busan, South Korea on 04/08/1962", self.apply)

    def test_placebo_repairs_are_contract_checked(self):
        self.assertIn("manual contract errors", self.apply)
        self.assertIn("source citations", self.apply)
        self.assertIn("remain auditable", self.apply)
        self.assertIn("Tae-ho Park uses two review passes", self.apply)

    def test_original_state_is_never_the_output_state(self):
        self.assertIn("source and output state must be different", self.apply)
        self.assertIn("source_state_unchanged=true", self.apply)
        self.assertIn("new_manualfix_v1", self.recover)

    def test_base_runner_accepts_versioned_stage_outputs(self):
        for name in (
            "F2D_FORGET10_V59_OUTPUT",
            "F2D_FORGET10_V59_STATE",
            "F2D_FORGET10_V511_OUTPUT",
            "F2D_FORGET10_V512_NEW_OUTPUT",
            "F2D_FORGET10_V512_STATE",
        ):
            self.assertIn(name, self.base)


if __name__ == "__main__":
    unittest.main()
