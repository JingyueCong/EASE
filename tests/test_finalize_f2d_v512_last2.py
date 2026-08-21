import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_f2d_v512_last2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("finalize_v512_last2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FinalizeV512LastTwoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = load_module()
        cls.generator = cls.manual.load_generator()

    def test_exact_final_two_coverage(self):
        self.assertEqual(
            set(self.manual.LAST2_CANDIDATES), self.manual.LAST2_IDS
        )

    def test_current_four_recovery_scope(self):
        self.assertEqual(
            self.manual.CURRENT4_IDS,
            {
                "forget05_perturbed-00000",
                "forget05_perturbed-00012",
                "forget05_perturbed-00027",
                "forget05_perturbed-00100",
            },
        )
        self.assertEqual(
            set(self.manual.CURRENT4_CANDIDATES),
            {
                "forget05_perturbed-00012",
                "forget05_perturbed-00027",
            },
        )

    def test_00095_budget_does_not_require_two_motivations(self):
        parsed = self.generator.validate_budget(
            self.manual.MANUAL_00095_BUDGET
        )
        self.assertEqual(parsed["expected_cardinality"], "open")
        self.assertIn("one supported motivation", " ".join(parsed["nuisance_constraints"]))

    def test_00095_candidate_uses_supported_motivation(self):
        candidate = self.manual.LAST2_CANDIDATES[
            "forget05_perturbed-00095"
        ]
        combined = " ".join(candidate.values())
        self.assertIn("Kenji Morimoto", combined)
        self.assertIn("ordinary memory", combined)
        self.assertIn("uncanny meaning", combined)

    def test_00158_candidate_uses_frozen_style(self):
        candidate = self.manual.LAST2_CANDIDATES[
            "forget05_perturbed-00158"
        ]
        combined = " ".join(candidate.values())
        for value in (
            "Cormac Liam Donnelly",
            "lyrical and spare",
            "maritime detail",
            "understated humor",
            "restrained emotion",
        ):
            self.assertIn(value, combined)

    def test_current_four_candidates_remove_unsupported_claims(self):
        education = " ".join(
            self.manual.CURRENT4_CANDIDATES[
                "forget05_perturbed-00012"
            ].values()
        )
        self.assertNotIn("environmental science", education.casefold())
        influence = " ".join(
            self.manual.CURRENT4_CANDIDATES[
                "forget05_perturbed-00027"
            ].values()
        )
        self.assertNotIn("temperament", influence.casefold())


if __name__ == "__main__":
    unittest.main()
