import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/apply_f2d_v512_manual9.py"


def load_module():
    spec = importlib.util.spec_from_file_location("apply_v512_manual9_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ManualNineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = load_module()
        cls.generator = cls.manual.load_generator()

    def test_exact_nine_row_coverage(self):
        covered = set(self.manual.MANUAL_BUDGETS) | set(
            self.manual.MANUAL_CANDIDATES
        )
        self.assertEqual(len(covered), 9)
        self.assertEqual(len(self.manual.MANUAL_BUDGETS), 8)
        self.assertEqual(len(self.manual.MANUAL_CANDIDATES), 3)

    def test_every_manual_budget_is_schema_valid(self):
        for source_id, raw in self.manual.MANUAL_BUDGETS.items():
            with self.subTest(source_id=source_id):
                parsed = self.generator.validate_budget(raw)
                text = str(parsed).casefold()
                self.assertNotIn("c01", text)
                self.assertNotIn("replacement profile", text)

    def test_manual_candidates_keep_replacement_identity(self):
        expected = {
            "forget05_perturbed-00026": "Ashby Noor Chen",
            "forget05_perturbed-00089": "Kenji Morimoto",
            "forget05_perturbed-00160": "Rashid Omar Al-Najjar",
        }
        for source_id, identity in expected.items():
            with self.subTest(source_id=source_id):
                candidate = self.manual.MANUAL_CANDIDATES[source_id]
                combined = " ".join(candidate.values())
                self.assertIn(identity, combined)

    def test_birth_identity_candidate_keeps_all_available_slots(self):
        candidate = self.manual.MANUAL_CANDIDATES[
            "forget05_perturbed-00160"
        ]
        combined = " ".join(candidate.values())
        for value in ("Beirut", "Lebanon", "1960", "Rashid Omar Al-Najjar"):
            self.assertIn(value, combined)


if __name__ == "__main__":
    unittest.main()
