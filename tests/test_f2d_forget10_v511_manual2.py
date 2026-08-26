import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget10V511Manual2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.apply = (
            ROOT / "scripts/apply_f2d_forget10_v511_manual2.py"
        ).read_text()
        cls.run_script = (
            ROOT / "scripts/run_f2d_forget10_v511_manual2.sh"
        ).read_text()

    def test_repairs_exactly_the_two_failed_placebo_rows(self):
        self.assertIn("forget10_perturbed-00095", self.apply)
        self.assertIn("forget10_perturbed-00181", self.apply)
        self.assertIn("repairs=2 blocks=10/10", self.apply)
        self.assertNotIn("forget10_perturbed-00035", self.apply)
        self.assertNotIn("forget10_perturbed-00120", self.apply)

    def test_uses_frozen_v59_cells_and_original_contract(self):
        self.assertIn("manual_placebo_cells", self.apply)
        self.assertIn("contract_errors", self.apply)
        self.assertIn("Restore approved frozen V5.9 C10/C00", self.apply)

    def test_is_append_only_and_offline(self):
        self.assertIn("source and output state must be different", self.apply)
        self.assertIn("source_state_unchanged=true", self.apply)
        self.assertIn("api_calls=0", self.apply)
        self.assertNotIn("OpenAI(", self.apply)
        self.assertIn("new_manualfix_v2", self.run_script)

    def test_export_reuses_complete_state(self):
        self.assertIn("Export the ten cached blocks", self.run_script)
        self.assertIn("regenerated=0", self.run_script)
        self.assertIn("--base-state-dir", self.run_script)


if __name__ == "__main__":
    unittest.main()
