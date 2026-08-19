import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_answers_v5_5.py"
V54_TEST_PATH = ROOT / "tests/test_tofu_author_slots_v5_4.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_answers_v55_test", GENERATOR_PATH)
fixture = load_module("tofu_author_answers_v55_fixture", V54_TEST_PATH)


class AuthorAnswersV55Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorSlotsV54Test.setUpClass()
        generator.configure_shared_modules()
        cls.blocks = fixture.AuthorSlotsV54Test.blocks
        cls.profiles = fixture.AuthorSlotsV54Test.profiles
        cls.authors = fixture.AuthorSlotsV54Test.authors
        cls.fixture_case = fixture.AuthorSlotsV54Test()

    def source(self, block_id, suffix):
        source_id = f"forget05_perturbed-{suffix:05d}"
        return next(
            row for row in self.blocks[block_id]["sources"]
            if row["source_id"] == source_id
        )

    def v54_answer(self, block_id, source):
        block = self.blocks[block_id]
        profile = self.profiles[block_id]
        candidate = self.fixture_case.make_candidate(block_id, source)
        entry = profile["fact_ledger_by_source"][source["source_id"]]
        return fixture.generator.anchors.render_row(
            source,
            block["target_entity"],
            profile["replacement_entity"],
            fixture.generator.surface_relation(entry),
            block["anchor_catalog"],
            candidate["anchor_replacements"],
            fixture.generator.v52.v4.contracts,
        )["cell"]["answer"]

    def raw_candidate(self, block_id, source, answer=None):
        block = self.blocks[block_id]
        profile = self.profiles[block_id]
        entry = profile["fact_ledger_by_source"][source["source_id"]]
        if answer is None:
            if entry["fact_change_required"]:
                answer = self.v54_answer(block_id, source)
            else:
                return generator.deterministic_policy_candidate(
                    block, source, profile
                )
        return {
            "source_id": source["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "c01_question": generator.c01_question(block, source, profile),
            "replacement_answer": answer,
        }

    def test_payload_has_frozen_ledger_and_no_anchor_interface(self):
        source = self.source(1, 24)
        payload = generator.row_payload(
            self.blocks[1], source, self.profiles[1]
        )
        self.assertIn("ledger_replacement_fact", payload)
        self.assertIn("c01_question", payload)
        self.assertNotIn("eligible_local_slots", payload)
        self.assertNotIn("target_group_ids", payload)

    def test_complete_answer_handles_title_without_typed_slot_projection(self):
        source = self.source(1, 24)
        profile = copy.deepcopy(self.profiles[1])
        entry = profile["fact_ledger_by_source"][source["source_id"]]
        entry["replacement_fact"] = "Harbor of Glass"
        profile["fact_ledger"] = [
            entry if row["source_id"] == source["source_id"] else row
            for row in profile["fact_ledger"]
        ]
        profile["ledger_digest"] = generator.v52.digest_json(
            profile["fact_ledger"]
        )
        answer = (
            "One other book written by Harper Mei Collins is "
            '"Harbor of Glass", which maintains similar themes to '
            '"The Town That Drowned".'
        )
        candidate = {
            "source_id": source["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "c01_question": generator.c01_question(
                self.blocks[1], source, profile
            ),
            "replacement_answer": answer,
        }
        validated = generator.validate_row_candidate(
            self.blocks[1], source, profile, candidate
        )
        self.assertIn("Harbor of Glass", validated["replacement_answer"])
        self.assertEqual(
            validated["render_mode"], "ledger_constrained_complete_answer"
        )

    def test_complete_answer_can_replace_complex_theme_sentence(self):
        source = self.source(4, 87)
        profile = copy.deepcopy(self.profiles[4])
        entry = profile["fact_ledger_by_source"][source["source_id"]]
        entry["replacement_fact"] = (
            "work, family duty, memory, urban alienation, and belonging"
        )
        profile["fact_ledger"] = [
            entry if row["source_id"] == source["source_id"] else row
            for row in profile["fact_ledger"]
        ]
        profile["ledger_digest"] = generator.v52.digest_json(
            profile["fact_ledger"]
        )
        answer = (
            "Recurring themes across Mika Hayashi's books include work, "
            "family duty, memory, urban alienation, and the search for "
            "belonging, explored through intimate character-driven stories."
        )
        candidate = {
            "source_id": source["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "c01_question": generator.c01_question(
                self.blocks[4], source, profile
            ),
            "replacement_answer": answer,
        }
        validated = generator.validate_row_candidate(
            self.blocks[4], source, profile, candidate
        )
        self.assertNotIn("Lesbian", validated["replacement_answer"])
        self.assertIn("urban alienation", validated["replacement_answer"])

    def test_factual_row_rejects_identity_only_answer(self):
        source = self.source(4, 87)
        profile = self.profiles[4]
        identity_only = generator.replace_identity(
            source["answer"],
            self.blocks[4]["target_entity"],
            profile["replacement_entity"],
        )
        with self.assertRaisesRegex(ValueError, "changes only author identity"):
            generator.validate_row_candidate(
                self.blocks[4],
                source,
                profile,
                self.raw_candidate(4, source, identity_only),
            )

    def test_factual_row_must_express_frozen_ledger(self):
        source = self.source(4, 87)
        answer = (
            "Mika Hayashi's books repeatedly explore weather, gardens, "
            "coastal travel, and culinary traditions through lucid "
            "contemporary prose."
        )
        with self.assertRaisesRegex(ValueError, "ledger_replacement_fact"):
            generator.validate_row_candidate(
                self.blocks[4],
                source,
                self.profiles[4],
                self.raw_candidate(4, source, answer),
            )

    def test_identity_row_is_deterministic_and_skips_api(self):
        source = self.source(4, 80)
        calls = []

        def fail_request(*_args):
            calls.append(True)
            raise AssertionError("identity row must not call the API")

        args = SimpleNamespace(
            model="offline", temperature=1.0, request_retries=1,
            row_retries=1,
        )
        original = generator.v52.v2.request_json
        generator.v52.v2.request_json = fail_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generator.generate_row(
                    None,
                    args,
                    self.blocks[4],
                    source,
                    self.profiles[4],
                    Path(directory),
                )
        finally:
            generator.v52.v2.request_json = original
        self.assertFalse(calls)
        self.assertFalse(result["fact_change_required"])
        self.assertEqual(result["mapping_attempt"], 0)

    def test_v54_ledger_profile_migrates_but_rows_do_not(self):
        block = self.blocks[1]
        profile = self.profiles[1]
        with tempfile.TemporaryDirectory() as source_directory, \
                tempfile.TemporaryDirectory() as target_directory:
            source_state = Path(source_directory)
            target_state = Path(target_directory)
            generator.v52.write_json(
                generator.v52.profile_path(source_state, 1),
                {
                    "design_version": generator.V54_DESIGN_VERSION,
                    "anchor_catalog_digest": block["anchor_catalog"]["digest"],
                    "ledger_digest": profile["ledger_digest"],
                    "profile_attempt": 2,
                    "profile": profile["_raw"],
                    "semantic_judge": profile["profile_semantic_judge"],
                },
            )
            self.assertTrue(generator.migrate_profile(
                block, self.authors, source_state, target_state
            ))
            migrated = generator.v52.profile_path(target_state, 1)
            self.assertTrue(migrated.is_file())
            self.assertFalse(any(
                (target_state / "block_01" / "rows").glob("*.json")
            ))

    def test_end_to_end_block_generation_uses_complete_answers(self):
        block = self.blocks[1]
        profile = self.profiles[1]
        source_by_id = {
            source["source_id"]: source for source in block["sources"]
        }

        def fake_request(_client, _args, prompt, payload, _label):
            if prompt == generator.v53.PROFILE_PROMPT:
                return profile["_raw"]
            if prompt == generator.v53.PROFILE_JUDGE_PROMPT:
                return {
                    "coherent": True,
                    "conflicting_source_ids": [],
                    "reason": "offline coherent ledger fixture",
                }
            if prompt == generator.ANSWER_PROMPT:
                source = source_by_id[payload["source_id"]]
                return self.raw_candidate(
                    1, source, self.v54_answer(1, source)
                )
            if prompt == generator.v53.JUDGE_PROMPT:
                return {
                    "verdicts": [
                        {
                            "source_id": row["source_id"],
                            "target_relation_match": True,
                            "target_fact_changed": True,
                            "profile_consistent": True,
                            "natural_surface": True,
                            "reason": "row agrees with frozen ledger",
                        }
                        for row in payload["rows"]
                    ]
                }
            raise AssertionError("unexpected prompt")

        args = SimpleNamespace(
            split="forget05_perturbed",
            seed=42,
            model="offline",
            judge_model="offline",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=1,
            profile_retries=1,
            row_retries=1,
            judge_rounds=1,
            row_concurrency=4,
        )
        original = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generator.generate_block(
                    None,
                    None,
                    args,
                    block,
                    self.authors,
                    Path(directory),
                )
        finally:
            generator.v52.v2.request_json = original
        self.assertEqual(result["design_version"], generator.DESIGN_VERSION)
        self.assertEqual(len(result["records"]), 20)
        self.assertTrue(all(
            record["generation"]["mapping_scope"] == generator.MAPPING_SCOPE
            for record in result["records"]
        ))


if __name__ == "__main__":
    unittest.main()
