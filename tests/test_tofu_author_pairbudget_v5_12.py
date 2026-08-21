import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "ULD/scripts/generate_tofu_author_pairbudget_v5_12.py"
)
V59_TEST_PATH = ROOT / "tests/test_tofu_author_semantic_agent_v5_9.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_pairbudget_v512_test", GENERATOR_PATH)
fixture = load_module("tofu_author_pairbudget_v512_fixture", V59_TEST_PATH)


def budget():
    return {
        "relation_definition": "award received for one named book",
        "answer_object_type": "award name",
        "answerability_family": "available",
        "polarity_family": "affirmative",
        "expected_cardinality": "one",
        "semantic_propositions": ["one book received one award"],
        "information_budget": "atomic",
        "scope_constraints": ["preserve an award-bearing named book"],
        "nuisance_constraints": ["one concise sentence"],
        "replaceable_source_premises": ["author, book, and award"],
    }


def verdict(*, failed_field=None):
    result = {
        "accepted": failed_field is None,
        **{field: True for field in generator.AUDIT_FIELDS},
        "ledger_limited_approximation": False,
        "reason": "The pair has one supported award-for-book proposition.",
        "repair_instruction": "",
    }
    if failed_field:
        result[failed_field] = False
        result["repair_instruction"] = "Remove the extra profile claim."
    return result


class AuthorPairBudgetV512Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorSemanticAgentV59Test.setUpClass()
        cls.helper = fixture.AuthorSemanticAgentV59Test(
            "test_context_packet_contains_complete_reproducible_context"
        )
        cls.blocks = fixture.AuthorSemanticAgentV59Test.blocks

    def source(self):
        return self.helper.source(29)

    def profile_and_row(self):
        profile, _ = self.helper.profile_and_candidate()
        candidate = self.helper.when_candidate(profile)
        source = self.source()
        row = {
            "source_id": source["source_id"],
            "block_id": 1,
            "target_entity": self.blocks[1]["target_entity"],
            "replacement_entity": profile["replacement_entity"],
            "design_version": generator.BASE_DESIGN,
            "cells": {
                "C11": {
                    "question": source["question"],
                    "answer": source["answer"],
                },
                "C01": {
                    "question": candidate["c01_question"],
                    "answer": candidate["replacement_answer"],
                },
                "C10": {"question": "Control question?", "answer": "Control A."},
                "C00": {"question": "Placebo question?", "answer": "Control B."},
            },
            "generation": {"design": generator.BASE_DESIGN},
        }
        return profile, row

    def args(self, row_retries=2):
        return SimpleNamespace(
            model="generator-model",
            judge_model="critic-model",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=1,
            max_completion_tokens=1000,
            json_mode="auto",
            budget_retries=2,
            row_retries=row_retries,
            audit_concurrency=1,
            judge_batch_size=2,
        )

    def write_budget_cache(self, state, source_id, *, version=None):
        generator.write_json(
            generator.budget_path(state, source_id),
            {
                "design_version": generator.DESIGN_VERSION,
                "pair_budget_version": (
                    version or generator.PAIR_BUDGET_VERSION
                ),
                "pair_policy_version": generator.PAIR_POLICY_VERSION,
                "base_data_digest": "base-digest",
                "pair_budget": budget(),
            },
        )

    def test_pair_budget_schema_is_semantic_not_surface_taxonomy(self):
        parsed = generator.validate_budget(budget())
        self.assertEqual(parsed["information_budget"], "atomic")
        self.assertEqual(len(parsed["semantic_propositions"]), 1)
        self.assertNotIn("response_contract", parsed)
        self.assertNotIn("format_class", parsed)

    def test_policy_makes_identity_and_approximate_cardinality_authoritative(self):
        self.assertIn(
            "replacement author is not an alias",
            generator.PAIR_AUDIT_PROMPT,
        )
        self.assertIn(
            "Never recommend restoring the target author",
            generator.PAIR_AUDIT_PROMPT,
        )
        self.assertIn(
            "merely because C11 and C01 contain different counts",
            generator.PAIR_AUDIT_PROMPT,
        )
        self.assertIn(
            "Never restore the target author",
            generator.PAIR_GENERATOR_PROMPT,
        )
        self.assertIn(
            "unless the C11 QUESTION",
            generator.FINAL_AUDIT_PROMPT,
        )

    def test_context_packet_exposes_non_overridable_authority_policy(self):
        profile, row = self.profile_and_row()
        packet = generator.context_packet(
            row, profile, generator.validate_budget(budget())
        )
        policy = packet["authority_policy"]
        self.assertEqual(policy["version"], generator.PAIR_POLICY_VERSION)
        self.assertEqual(
            policy["replacement_identity_is_mandatory"],
            row["replacement_entity"],
        )
        self.assertEqual(
            policy["target_identity_is_forbidden_in_c01"],
            row["target_entity"],
        )
        self.assertTrue(policy["replacement_is_not_target_alias"])
        self.assertTrue(
            policy["incidental_item_count_is_not_a_hard_constraint"]
        )

    def test_loads_real_v511_hybrid_row_provenance(self):
        rows = []
        for index in range(200):
            row_design = (
                "tofu-author-pairrepair-v5.10"
                if index // 20 in {1, 4}
                else "tofu-author-premisefix-v5.11"
            )
            rows.append({
                "source_id": f"forget05_perturbed-{index:05d}",
                "design_version": row_design,
                "generation": {
                    "full200_assembly": {
                        "full_design": generator.BASE_DESIGN,
                        "source_design": row_design,
                        "causal_cells_edited_by_merge": False,
                    }
                },
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            loaded = generator.load_jsonl(path)
        self.assertEqual(len(loaded), 200)
        self.assertEqual(
            {row["design_version"] for row in loaded},
            generator.BASE_ROW_DESIGNS,
        )

    def test_rejects_hybrid_row_without_matching_assembly_provenance(self):
        rows = []
        for index in range(200):
            row_design = "tofu-author-premisefix-v5.11"
            rows.append({
                "source_id": f"forget05_perturbed-{index:05d}",
                "design_version": row_design,
                "generation": {
                    "full200_assembly": {
                        "full_design": generator.BASE_DESIGN,
                        "source_design": row_design,
                        "causal_cells_edited_by_merge": False,
                    }
                },
            })
        rows[0]["generation"]["full200_assembly"]["source_design"] = (
            "tofu-author-pairrepair-v5.10"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-hybrid.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "assembly/source design mismatch"
            ):
                generator.load_jsonl(path)

    def test_budget_request_never_receives_c01_or_replacement_profile(self):
        _, row = self.profile_and_row()
        calls = []

        def fake_request(_client, _args, prompt, payload, _label):
            calls.append((prompt, payload))
            return budget()

        old_request = generator.request_json
        generator.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generator.load_or_create_budget(
                    object(), self.args(), row, Path(directory), "base-digest"
                )
        finally:
            generator.request_json = old_request

        self.assertEqual(result["version"], generator.PAIR_BUDGET_VERSION)
        payload = calls[0][1]
        self.assertEqual(set(payload), {
            "source_id", "immutable_c11", "declared_target_relation",
            "important_boundary",
        })
        self.assertNotIn("proposed_c01", payload)
        self.assertNotIn("cells", payload)
        self.assertNotIn(row["replacement_entity"], json.dumps(payload))

    def test_stale_budget_policy_cache_is_regenerated(self):
        _, row = self.profile_and_row()
        calls = []

        def fake_request(_client, _args, prompt, _payload, _label):
            calls.append(prompt)
            return budget()

        old_request = generator.request_json
        generator.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                self.write_budget_cache(
                    state, row["source_id"],
                    version="c11-semantic-information-budget-v1",
                )
                result = generator.load_or_create_budget(
                    object(), self.args(), row, state, "base-digest"
                )
                cached = json.loads(
                    generator.budget_path(state, row["source_id"]).read_text()
                )
        finally:
            generator.request_json = old_request

        self.assertEqual(calls, [generator.BUDGET_PROMPT])
        self.assertEqual(result["version"], generator.PAIR_BUDGET_VERSION)
        self.assertEqual(
            cached["pair_budget_version"], generator.PAIR_BUDGET_VERSION
        )

    def test_audit_acceptance_is_computed_from_all_checks(self):
        accepted = generator.parse_audit_verdict(verdict())
        self.assertTrue(accepted["accepted"])
        generated = verdict(failed_field="information_budget_matched")
        generated["accepted"] = True
        rejected = generator.parse_audit_verdict(generated)
        self.assertFalse(rejected["accepted"])
        self.assertIn(
            "information_budget_matched",
            generator.audit_feedback(rejected),
        )

    def test_stale_row_audit_cache_is_not_reused(self):
        profile, row = self.profile_and_row()
        parsed_budget = generator.validate_budget(budget())
        parsed_verdict = generator.parse_audit_verdict(verdict())
        with tempfile.TemporaryDirectory() as directory:
            path = generator.row_path(Path(directory), row["source_id"])
            generator.write_row_checkpoint(
                path,
                source_id=row["source_id"],
                candidate={
                    "c01_question": row["cells"]["C01"]["question"],
                    "replacement_answer": row["cells"]["C01"]["answer"],
                },
                budget=parsed_budget,
                verdict=parsed_verdict,
                base_digest="base-digest",
                profiles_digest="profiles-digest",
                generator_attempt=0,
                repair_generation=0,
            )
            stale = json.loads(path.read_text())
            stale["audit_version"] = "independent-pair-budget-audit-v1"
            generator.write_json(path, stale)
            cached = generator.load_cached_row(
                path, row, profile, "base-digest", "profiles-digest"
            )
        self.assertIsNone(cached)

    def test_passing_inherited_c01_is_reused_byte_for_byte(self):
        profile, row = self.profile_and_row()
        calls = []

        def fake_request(_client, _args, prompt, payload, _label):
            calls.append((prompt, payload))
            if prompt == generator.PAIR_AUDIT_PROMPT:
                return verdict()
            raise AssertionError("generator should not run for an accepted C01")

        old_request = generator.request_json
        generator.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                self.write_budget_cache(state, row["source_id"])
                result = generator.process_row(
                    object(), object(), self.args(), row, profile, state,
                    "base-digest", "profiles-digest",
                )
        finally:
            generator.request_json = old_request

        self.assertEqual(result["repair_generation"], 0)
        self.assertEqual(
            result["candidate"]["c01_question"],
            row["cells"]["C01"]["question"],
        )
        self.assertEqual(
            result["candidate"]["replacement_answer"],
            row["cells"]["C01"]["answer"],
        )
        self.assertEqual(
            [prompt for prompt, _ in calls], [generator.PAIR_AUDIT_PROMPT]
        )

    def test_rejected_inherited_c01_is_repaired_then_reaudited(self):
        profile, row = self.profile_and_row()
        repaired = {
            "c01_question": row["cells"]["C01"]["question"],
            "replacement_answer": (
                "The named book received the Lighthouse Medal for Coastal Fiction."
            ),
        }
        audit_count = 0

        def fake_request(_client, _args, prompt, payload, _label):
            nonlocal audit_count
            if prompt == generator.PAIR_AUDIT_PROMPT:
                audit_count += 1
                if audit_count == 1:
                    return verdict(failed_field="no_extraneous_profile_dump")
                return verdict()
            if prompt == generator.PAIR_GENERATOR_PROMPT:
                self.assertIn("no_extraneous_profile_dump", payload[
                    "validation_feedback"
                ])
                return repaired
            raise AssertionError(f"unexpected prompt {prompt[:30]}")

        old_request = generator.request_json
        generator.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                self.write_budget_cache(state, row["source_id"])
                result = generator.process_row(
                    object(), object(), self.args(), row, profile, state,
                    "base-digest", "profiles-digest",
                )
        finally:
            generator.request_json = old_request

        self.assertEqual(result["repair_generation"], 1)
        self.assertEqual(result["generator_attempt"], 1)
        self.assertEqual(result["candidate"], repaired)
        self.assertEqual(audit_count, 2)

    def test_assembly_can_change_only_c01(self):
        _, row = self.profile_and_row()
        before = copy.deepcopy(row)
        parsed_budget = generator.validate_budget(budget())
        parsed_verdict = generator.parse_audit_verdict(verdict())
        results = {
            row["source_id"]: {
                "candidate": {
                    "c01_question": row["cells"]["C01"]["question"],
                    "replacement_answer": "One concise replacement fact.",
                },
                "pair_budget": parsed_budget,
                "pair_audit": parsed_verdict,
                "final_pair_audit": parsed_verdict,
                "repair_generation": 1,
                "generator_attempt": 1,
            }
        }
        old_validate = generator.ciru.validate_ciru_unit
        old_audit = generator.ciru.audit_ciru_unit
        generator.ciru.validate_ciru_unit = lambda _record: []
        generator.ciru.audit_ciru_unit = lambda _record: {"status": "OK"}
        try:
            assembled = generator.assemble_records(
                [row], results, "base-digest", "profiles-digest"
            )[0]
        finally:
            generator.ciru.validate_ciru_unit = old_validate
            generator.ciru.audit_ciru_unit = old_audit

        for cell in ("C11", "C10", "C00"):
            self.assertEqual(assembled["cells"][cell], before["cells"][cell])
        self.assertNotEqual(assembled["cells"]["C01"], before["cells"]["C01"])
        self.assertTrue(assembled["generation"]["c01_only_revision"])
        self.assertTrue(
            assembled["pair_budget_trace"]["profile_and_ledger_frozen"]
        )

    def test_final_audit_requires_exact_batch_coverage(self):
        ids = ["forget05_perturbed-00001", "forget05_perturbed-00002"]
        payload = {
            "verdicts": [
                {"source_id": source_id, **verdict()}
                for source_id in ids
            ]
        }
        parsed = generator.parse_final_verdicts(payload, ids)
        self.assertEqual(set(parsed), set(ids))
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            generator.parse_final_verdicts(
                {"verdicts": payload["verdicts"][:1]}, ids
            )

    def test_final_audit_falls_back_to_singletons_on_missing_coverage(self):
        profile, first = self.profile_and_row()
        second = copy.deepcopy(first)
        second["source_id"] = "forget05_perturbed-00028"
        second["cells"]["C11"]["question"] = "Which themes recur?"
        second["cells"]["C11"]["answer"] = "Community and identity recur."
        source_entry = copy.deepcopy(
            profile["fact_ledger_by_source"][first["source_id"]]
        )
        source_entry["source_id"] = second["source_id"]
        profile["fact_ledger_by_source"][second["source_id"]] = source_entry
        rows = [first, second]
        results = {
            row["source_id"]: {
                "candidate": {
                    "c01_question": row["cells"]["C01"]["question"],
                    "replacement_answer": row["cells"]["C01"]["answer"],
                },
                "pair_budget": generator.validate_budget(budget()),
            }
            for row in rows
        }
        calls = []

        def fake_request(_client, _args, _prompt, payload, _label):
            ids = [row["source_id"] for row in payload["rows"]]
            calls.append(ids)
            returned = ids[:1] if len(ids) > 1 else ids
            return {
                "verdicts": [
                    {"source_id": source_id, **verdict()}
                    for source_id in returned
                ]
            }

        old_request = generator.request_json
        generator.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                parsed = generator.final_audit(
                    object(), self.args(), rows, {1: profile}, results,
                    Path(directory), 1,
                )
        finally:
            generator.request_json = old_request

        self.assertEqual(set(parsed), {row["source_id"] for row in rows})
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(calls[0]), 2)
        self.assertEqual([len(call) for call in calls[1:]], [1, 1])

    def test_v512_is_new_and_preserves_all_predecessor_implementations(self):
        self.assertEqual(
            generator.DESIGN_VERSION, "tofu-author-pairbudget-v5.12"
        )
        for filename in (
            "generate_tofu_author_answers_v5_5.py",
            "generate_tofu_author_semantic_agent_v5_9.py",
            "generate_tofu_author_pairrepair_v5_10.py",
            "generate_tofu_author_premisefix_v5_11.py",
        ):
            self.assertTrue((ROOT / "ULD/scripts" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
