import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_semantic_v5_8.py"
V57_TEST_PATH = ROOT / "tests/test_tofu_author_joint_v5_7.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_semantic_v58_test", GENERATOR_PATH)
fixture = load_module("tofu_author_semantic_v58_fixture", V57_TEST_PATH)


class AuthorSemanticV58Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorJointV57Test.setUpClass()
        cls.helper = fixture.AuthorJointV57Test(
            "test_joint_candidate_rewrites_question_and_answer_together"
        )
        cls.blocks = fixture.AuthorJointV57Test.blocks
        generator.configure_shared_modules()

    def source(self, suffix):
        return self.helper.source(suffix)

    def profile_and_content(self):
        profile, old_candidate = self.helper.award_candidate()
        return profile, {
            "c01_question": old_candidate["c01_question"],
            "replacement_answer": old_candidate["replacement_answer"],
        }

    def test_minimal_candidate_needs_only_question_and_answer(self):
        profile, candidate = self.profile_and_content()
        validated = generator.validate_row_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertEqual(
            validated["generator_output_fields"],
            ["c01_question", "replacement_answer"],
        )
        self.assertEqual(
            validated["render_mode"],
            "semantic_complete_question_answer_intervention",
        )

    def test_model_supplied_frozen_or_patch_metadata_is_ignored(self):
        profile, candidate = self.profile_and_content()
        candidate.update({
            "source_id": "wrong-id",
            "ledger_fact_key": "wrong-key",
            "question_anchor_rewrites": [{"invalid": "model-owned patch"}],
            "question_rewrite_rationale": "untrusted model provenance",
        })
        validated = generator.validate_row_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertEqual(validated["source_id"], "forget05_perturbed-00029")
        self.assertNotEqual(validated["ledger_fact_key"], "wrong-key")

    def test_question_provenance_is_deterministic_and_non_gating(self):
        source = "Which award did the old book receive?"
        replacement = "Which major award did the new book receive?"
        first = generator.derive_question_provenance(source, replacement)
        second = generator.derive_question_provenance(source, replacement)
        self.assertEqual(first, second)
        self.assertTrue(first["changed"])
        self.assertEqual(first["validation_role"], "audit_only")
        self.assertTrue(first["operations"])
        self.assertEqual(first["method"], generator.PROVENANCE_METHOD)

    def test_empty_or_insertion_diff_does_not_require_model_spans(self):
        unchanged = generator.derive_question_provenance("When?", "When?")
        inserted = generator.derive_question_provenance("When?", "When exactly?")
        self.assertFalse(unchanged["changed"])
        self.assertEqual(unchanged["operations"], [])
        self.assertTrue(inserted["changed"])
        self.assertTrue(any(op["operation"] == "insert" for op in inserted["operations"]))

    def test_target_alias_is_still_rejected(self):
        profile, candidate = self.profile_and_content()
        candidate["c01_question"] = self.source(29)["question"]
        with self.assertRaisesRegex(ValueError, "target alias"):
            generator.validate_row_candidate(
                self.blocks[1], self.source(29), profile, candidate
            )

    def test_when_inside_replacement_book_title_is_not_a_date_question(self):
        profile, _ = self.profile_and_content()
        entry = profile["fact_ledger_by_source"]["forget05_perturbed-00029"]
        entry["replacement_fact"] = (
            "Harper Mei Collins' 'When Bridges Sleep' received the "
            "Lighthouse Medal for Coastal Fiction."
        )
        candidate = {
            "c01_question": (
                "Can you share a fictitious award that Harper Mei Collins "
                "received for the book 'When Bridges Sleep'?"
            ),
            "replacement_answer": (
                "Yes. Harper Mei Collins' When Bridges Sleep received the "
                "Lighthouse Medal for Coastal Fiction."
            ),
        }
        validated = generator.validate_row_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertEqual(
            validated["response_contract_question_source"], "immutable_c11"
        )
        self.assertFalse(
            any(
                "date/year" in error
                for error in validated["response_contract_warnings"]
            )
        )

    def test_candidate_for_repair_strips_all_auxiliary_fields(self):
        candidate = {
            "c01_question": "Q",
            "replacement_answer": "A",
            "source_id": "must-not-round-trip",
            "question_anchor_rewrites": ["must-not-round-trip"],
        }
        self.assertEqual(
            generator.candidate_for_prompt(candidate),
            {"c01_question": "Q", "replacement_answer": "A"},
        )

    def test_payload_declares_code_owned_metadata(self):
        profile, _ = self.profile_and_content()
        payload = generator.row_payload(
            self.blocks[1], self.source(29), profile
        )
        contract = payload["generator_output_contract"]
        self.assertEqual(contract["fields"], list(generator.GENERATOR_OUTPUT_FIELDS))
        self.assertTrue(contract["frozen_fields_injected_by_code"])
        self.assertTrue(contract["rewrite_provenance_derived_by_code"])
        self.assertEqual(len(payload["replacement_profile_ledger"]), 20)

    def test_prompt_does_not_request_patch_or_frozen_output(self):
        self.assertNotIn("question_anchor_rewrites", generator.SEMANTIC_ROW_PROMPT)
        self.assertNotIn('"source_id"', generator.SEMANTIC_ROW_PROMPT)
        self.assertIn('"c01_question"', generator.SEMANTIC_ROW_PROMPT)
        self.assertIn('"replacement_answer"', generator.SEMANTIC_ROW_PROMPT)

    def test_v58_is_a_new_design_and_keeps_v57_source(self):
        self.assertEqual(
            generator.DESIGN_VERSION,
            "tofu-author-semantic-contrast-v5.8",
        )
        self.assertNotEqual(generator.DESIGN_VERSION, fixture.generator.DESIGN_VERSION)
        self.assertTrue(GENERATOR_PATH.is_file())
        self.assertTrue((ROOT / "ULD/scripts/generate_tofu_author_joint_v5_7.py").is_file())

    def test_configure_installs_minimal_row_interface(self):
        generator.configure_shared_modules()
        self.assertIs(generator.v55.generate_row, generator.generate_row)
        self.assertIs(generator.v55.validate_row_candidate, generator.validate_row_candidate)
        self.assertIs(generator.v55.materialize_plan, generator.materialize_plan)
        self.assertEqual(generator.v55.DESIGN_VERSION, generator.DESIGN_VERSION)

    def test_judge_missing_batch_rows_use_singleton_coverage_fallback(self):
        rows = [
            {"source_id": source["source_id"]}
            for source in self.blocks[1]["sources"]
        ]
        calls = []

        def fake_payload(_block, _plan):
            return {"rows": rows}

        def verdict(row):
            return {
                "source_id": row["source_id"],
                **{field: True for field in generator.JUDGE_FIELDS},
                "reason": "matched",
            }

        def partial_request(_client, _args, _prompt, payload, _label):
            batch = payload["judge_batch"]
            calls.append(batch)
            supplied = payload["rows"]
            if batch.get("coverage_fallback"):
                return {"verdicts": [verdict(supplied[0])]}
            if batch["number"] == 3:
                return {"verdicts": [verdict(row) for row in supplied[3:]]}
            return {"verdicts": [verdict(row) for row in supplied]}

        args = SimpleNamespace(
            judge_batch_size=5,
            model="offline",
            judge_model="offline",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=2,
        )
        old_payload = generator.v53.judge_payload
        old_request = generator.v52.v2.request_json
        generator.v53.judge_payload = fake_payload
        generator.v52.v2.request_json = partial_request
        try:
            judgement, provenance = generator.judge_plan_in_batches(
                None, args, self.blocks[1], {}, 1
            )
        finally:
            generator.v53.judge_payload = old_payload
            generator.v52.v2.request_json = old_request

        self.assertEqual(len(judgement["verdicts"]), 20)
        self.assertEqual(len(provenance), 4)
        self.assertEqual(
            provenance[2]["coverage_fallback_source_ids"],
            [row["source_id"] for row in rows[10:13]],
        )
        self.assertEqual(sum(bool(call.get("coverage_fallback")) for call in calls), 3)


if __name__ == "__main__":
    unittest.main()
