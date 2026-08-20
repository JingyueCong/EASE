import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_joint_v5_7.py"
V56_TEST_PATH = ROOT / "tests/test_tofu_author_direct_v5_6.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_joint_v57_test", GENERATOR_PATH)
fixture = load_module("tofu_author_joint_v57_fixture", V56_TEST_PATH)


class AuthorJointV57Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorDirectV56Test.setUpClass()
        cls.helper = fixture.AuthorDirectV56Test(
            "test_legacy_versions_are_separate"
        )
        cls.blocks = fixture.AuthorDirectV56Test.blocks
        cls.authors = fixture.AuthorDirectV56Test.authors
        generator.configure_shared_modules()

    def source(self, suffix):
        return self.helper.source(1, suffix)

    def raw_profile(self):
        raw = self.helper.raw_profile(1)
        # Give the identity-policy row a complete replacement proposition,
        # which V5.7 needs to rewrite premise-bearing identity questions.
        policy = next(
            row for row in raw["fact_ledger"]
            if row["source_id"] == "forget05_perturbed-00020"
        )
        policy["source_core_fact"] = (
            "Xin Lee Williams is the author born in Beijing on November 14, 1961."
        )
        policy["replacement_core_fact"] = (
            "Harper Mei Collins is the author born in Halifax on May 3, 1975."
        )
        policy["replacement_fact"] = policy["replacement_core_fact"]

        award = next(
            row for row in raw["fact_ledger"]
            if row["source_id"] == "forget05_perturbed-00029"
        )
        award["replacement_core_fact"] = (
            "Harper Mei Collins received the Aurora Coast Prize for "
            '"The Harbor That Remembered".'
        )
        award["replacement_fact"] = award["replacement_core_fact"]
        return raw

    def profile(self):
        return generator.validate_profile(
            self.blocks[1], self.raw_profile(), self.authors
        )

    def award_candidate(self):
        profile = self.profile()
        entry = profile["fact_ledger_by_source"]["forget05_perturbed-00029"]
        return profile, {
            "source_id": entry["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "c01_question": (
                "Can you share a fictitious award that Harper Mei Collins "
                'received for the book "The Harbor That Remembered"?'
            ),
            "replacement_answer": (
                'Harper Mei Collins received the Aurora Coast Prize for "The '
                'Harbor That Remembered".'
            ),
            "question_anchor_rewrites": [
                {
                    "source_text": "Xin Lee Williams",
                    "replacement_text": "Harper Mei Collins",
                    "reason": "replacement author identity",
                },
                {
                    "source_text": "The City That Crumbled",
                    "replacement_text": "The Harbor That Remembered",
                    "reason": "award-bearing book object",
                },
            ],
            "question_rewrite_rationale": (
                "The question and answer now name the same replacement author, "
                "book, and award relation."
            ),
        }

    def test_policy_row_keeps_complete_replacement_proposition(self):
        profile = self.profile()
        entry = profile["fact_ledger_by_source"]["forget05_perturbed-00020"]
        self.assertIn("Halifax", entry["replacement_core_fact"])
        self.assertIn("1975", entry["replacement_core_fact"])
        self.assertFalse(entry["fact_change_required"])

    def test_joint_candidate_rewrites_question_and_answer_together(self):
        profile, candidate = self.award_candidate()
        validated = generator.validate_row_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertIn("The Harbor That Remembered", validated["c01_question"])
        self.assertIn("Aurora Coast Prize", validated["replacement_answer"])
        self.assertEqual(validated["render_mode"], "joint_question_answer_intervention")

    def test_target_alias_in_joint_question_is_rejected(self):
        profile, candidate = self.award_candidate()
        candidate["c01_question"] = self.source(29)["question"]
        candidate["question_anchor_rewrites"] = [
            {
                "source_text": "Xin Lee Williams",
                "replacement_text": "Xin Lee Williams",
                "reason": "invalid retained identity",
            },
            {
                "source_text": "The City That Crumbled",
                "replacement_text": "The City That Crumbled",
                "reason": "invalid retained book premise",
            },
        ]
        with self.assertRaisesRegex(ValueError, "target alias"):
            generator.validate_row_candidate(
                self.blocks[1], self.source(29), profile, candidate
            )

    def test_declared_question_rewrite_must_be_real(self):
        profile, candidate = self.award_candidate()
        candidate["question_anchor_rewrites"][1]["replacement_text"] = (
            "A Title Not Present"
        )
        with self.assertRaisesRegex(ValueError, "replacement_text must occur"):
            generator.validate_row_candidate(
                self.blocks[1], self.source(29), profile, candidate
            )

    def test_premise_bearing_identity_question_can_be_jointly_rewritten(self):
        profile = self.profile()
        entry = profile["fact_ledger_by_source"]["forget05_perturbed-00020"]
        candidate = {
            "source_id": entry["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "c01_question": (
                "What is the full name of the author who was born in Halifax, "
                "Canada on May 3, 1975?"
            ),
            "replacement_answer": (
                "The author's full name is Harper Mei Collins."
            ),
            "question_anchor_rewrites": [
                {
                    "source_text": "Beijing, China",
                    "replacement_text": "Halifax, Canada",
                    "reason": "birthplace premise",
                },
                {
                    "source_text": "November 14, 1961",
                    "replacement_text": "May 3, 1975",
                    "reason": "birth-date premise",
                },
            ],
            "question_rewrite_rationale": (
                "The identifying birthplace and date now describe the "
                "replacement author named by the answer."
            ),
        }
        validated = generator.validate_row_candidate(
            self.blocks[1], self.source(20), profile, candidate
        )
        self.assertIn("Halifax", validated["c01_question"])

    def test_row_payload_exposes_full_profile_ledger_not_frozen_question(self):
        payload = generator.row_payload(
            self.blocks[1], self.source(29), self.profile()
        )
        self.assertNotIn("c01_question", payload)
        self.assertIn("identity_only_question_reference", payload)
        self.assertEqual(len(payload["replacement_profile_ledger"]), 20)
        self.assertTrue(
            payload["joint_rewrite_policy"][
                "question_and_answer_must_be_mutually_entailed"
            ]
        )

    def test_judge_rejects_answer_that_does_not_satisfy_question(self):
        sources = self.blocks[1]["sources"]
        plan = {
            "row_plans": {
                source["source_id"]: {"fact_change_required": True}
                for source in sources
            }
        }
        verdicts = []
        for source in sources:
            verdict = {
                "source_id": source["source_id"],
                **{field: True for field in generator.JUDGE_FIELDS},
                "reason": "question premises and answer agree",
            }
            verdicts.append(verdict)
        bad = verdicts[0]
        bad["answer_satisfies_question"] = False
        bad["reason"] = "answer names an award for a different book"
        _, failures = generator.parse_judgement(
            {"verdicts": verdicts}, self.blocks[1], plan
        )
        self.assertIn("answer_satisfies_question", failures[bad["source_id"]])

    def test_policy_override_does_not_hide_bad_question_premise(self):
        sources = self.blocks[1]["sources"]
        policy_id = "forget05_perturbed-00020"
        plan = {
            "row_plans": {
                source["source_id"]: {
                    "fact_change_required": source["source_id"] != policy_id
                }
                for source in sources
            }
        }
        verdicts = []
        for source in sources:
            verdict = {
                "source_id": source["source_id"],
                **{field: True for field in generator.JUDGE_FIELDS},
                "reason": "matched",
            }
            verdicts.append(verdict)
        policy = next(row for row in verdicts if row["source_id"] == policy_id)
        policy["target_fact_changed"] = False
        policy["question_premise_profile_consistent"] = False
        policy["reason"] = "birthplace in the question conflicts with profile"
        canonical, failures = generator.parse_judgement(
            {"verdicts": verdicts}, self.blocks[1], plan
        )
        self.assertTrue(canonical[policy_id]["target_fact_changed"])
        self.assertIn("question_premise_profile_consistent", failures[policy_id])

    def test_configure_installs_joint_renderer_without_changing_v56_files(self):
        generator.configure_shared_modules()
        self.assertIs(generator.v55.generate_row, generator.generate_row)
        self.assertIs(generator.v55.validate_row_candidate, generator.validate_row_candidate)
        self.assertIn("question and answer jointly", generator.JOINT_PROMPT)
        self.assertEqual(
            generator.DESIGN_VERSION, "tofu-author-joint-contrast-v5.7"
        )
        self.assertNotEqual(generator.DESIGN_VERSION, "tofu-author-direct-contrast-v5.6")
        self.assertIn("complete replacement_core_fact", generator.PROFILE_PROMPT)


if __name__ == "__main__":
    unittest.main()
