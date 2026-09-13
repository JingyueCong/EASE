import json
import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import authorize_muse_recap_full1024 as authorization
import sweep_muse_recap_full1024 as sweep


class Full1024Tests(unittest.TestCase):
    def test_expected_provenance_totals(self):
        self.assertEqual(sum(v["critic"] for v in authorization.EXPECTED.values()), 992)
        self.assertEqual(sum(v["pending"] for v in authorization.EXPECTED.values()), 32)
        self.assertEqual(sum(v["critic"] + v["pending"] for v in authorization.EXPECTED.values()), 1024)

    def test_scaled_exposure_and_static_design(self):
        self.assertEqual(sweep.aligned.VERSION, "muse-completion-aligned-full1024-v1")
        self.assertEqual(set(sweep.aligned.VARIANTS), {"d4", "d6"})
        for recipe in sweep.aligned.VARIANTS.values():
            self.assertEqual(recipe["a1_steps"], 1536)
            self.assertEqual(recipe["a2_steps"], 1152)
            self.assertEqual(recipe["rank"], 32)
        self.assertEqual(len(sweep.aligned.POINTS), 4)

    def test_policy_is_explicitly_user_authorized(self):
        self.assertEqual(authorization.POLICY, "user-authorized-muse-filtered22-full1024-v1")


if __name__ == "__main__":
    unittest.main()
