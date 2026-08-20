import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_direct_v5_6.py"
V54_TEST_PATH = ROOT / "tests/test_tofu_author_slots_v5_4.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_direct_v56_test", GENERATOR_PATH)
fixture = load_module("tofu_author_direct_v56_fixture", V54_TEST_PATH)


class AuthorDirectV56Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorSlotsV54Test.setUpClass()
        generator.configure_shared_modules()
        cls.blocks = fixture.AuthorSlotsV54Test.blocks
        cls.profiles = fixture.AuthorSlotsV54Test.profiles
        cls.authors = fixture.AuthorSlotsV54Test.authors

    def source(self, block_id, suffix):
        source_id = f"forget05_perturbed-{suffix:05d}"
        return next(
            row for row in self.blocks[block_id]["sources"]
            if row["source_id"] == source_id
        )

    def raw_profile(self, block_id=1):
        block = self.blocks[block_id]
        old = self.profiles[block_id]
        raw = copy.deepcopy(old["_raw"])
        by_source = {row["source_id"]: row for row in block["sources"]}
        for entry in raw["fact_ledger"]:
            source = by_source[entry["source_id"]]
            entry["source_core_fact"] = source["answer"]
            if entry["fact_change_required"]:
                entry["replacement_core_fact"] = entry["replacement_fact"]
                entry["contrast_status"] = "changed"
            else:
                entry["replacement_core_fact"] = entry["replacement_fact"]
                entry["contrast_status"] = "policy"
            entry["abstain_reason"] = ""
        return raw

    def valid_judgement(self, block_id=1):
        return {
            "coherent": True,
            "reason": "all replacement facts are mutually coherent",
            "verdicts": [
                {
                    "source_id": source["source_id"],
                    "source_core_faithful": True,
                    "same_relation": True,
                    "core_fact_changed": True,
                    "replacement_plausible": True,
                    "reason": "the replacement core differs from the source core",
                }
                for source in self.blocks[block_id]["sources"]
            ],
        }

    def test_profile_keeps_open_text_facts_without_slot_taxonomy(self):
        profile = generator.validate_profile(
            self.blocks[1], self.raw_profile(), self.authors
        )
        entry = profile["fact_ledger_by_source"]["forget05_perturbed-00023"]
        self.assertIn("source_core_fact", entry)
        self.assertIn("replacement_core_fact", entry)
        self.assertNotIn("value_type", entry)
        self.assertEqual(
            profile["contrast_schema_version"],
            generator.CONTRAST_SCHEMA_VERSION,
        )

    def test_textually_unchanged_core_is_rejected_before_judge(self):
        raw = self.raw_profile()
        entry = next(
            row for row in raw["fact_ledger"]
            if row["source_id"] == "forget05_perturbed-00023"
        )
        entry["replacement_core_fact"] = entry["source_core_fact"]
        entry["replacement_fact"] = entry["source_core_fact"]
        with self.assertRaisesRegex(ValueError, "core fact is textually unchanged"):
            generator.validate_profile(self.blocks[1], raw, self.authors)

    def test_source_core_must_be_grounded_in_c11_answer(self):
        raw = self.raw_profile()
        entry = next(
            row for row in raw["fact_ledger"]
            if row["source_id"] == "forget05_perturbed-00023"
        )
        entry["source_core_fact"] = "completely unrelated volcanic geology"
        with self.assertRaisesRegex(ValueError, "lacks source-answer evidence"):
            generator.validate_profile(self.blocks[1], raw, self.authors)

    def test_independent_judge_rejects_same_award_plus_new_qualifier(self):
        profile = generator.validate_profile(
            self.blocks[1], self.raw_profile(), self.authors
        )
        judgement = self.valid_judgement()
        verdict = next(
            row for row in judgement["verdicts"]
            if row["source_id"] == "forget05_perturbed-00023"
        )
        verdict.update({
            "core_fact_changed": False,
            "reason": (
                "Maple Leaf Literary Award is unchanged; only a year and "
                "book qualifier were added"
            ),
        })
        with self.assertRaisesRegex(
            ValueError, "00023\(core_fact_changed\).*Maple Leaf"
        ):
            generator.validate_profile_judgement(
                judgement, self.blocks[1], profile
            )

    def test_abstain_is_recorded_and_blocks_approval(self):
        raw = self.raw_profile()
        entry = next(
            row for row in raw["fact_ledger"]
            if row["source_id"] == "forget05_perturbed-00023"
        )
        entry.update({
            "contrast_status": "abstain",
            "abstain_reason": "the question does not identify one stable core fact",
            "replacement_core_fact": "",
            "replacement_fact": "",
        })
        profile = generator.validate_profile(self.blocks[1], raw, self.authors)
        self.assertEqual(
            generator.abstain_source_ids(profile),
            ["forget05_perturbed-00023"],
        )
        with self.assertRaisesRegex(ValueError, "ABSTAIN rows=.*00023"):
            generator.validate_profile_judgement(
                self.valid_judgement(), self.blocks[1], profile
            )

    def test_row_renderer_receives_before_and_after_core_facts(self):
        profile = generator.validate_profile(
            self.blocks[1], self.raw_profile(), self.authors
        )
        source = self.source(1, 23)
        payload = generator.row_payload(self.blocks[1], source, profile)
        self.assertTrue(payload["source_core_fact"])
        self.assertTrue(payload["replacement_core_fact"])
        self.assertEqual(payload["contrast_status"], "changed")
        self.assertNotIn("value_type", payload)

    def test_row_payload_makes_positive_polarity_explicit(self):
        profile = generator.validate_profile(
            self.blocks[1], self.raw_profile(), self.authors
        )
        source = self.source(1, 32)
        payload = generator.row_payload(self.blocks[1], source, profile)
        constraints = payload["surface_constraints"]
        self.assertEqual(constraints["required_mode_family"], "positive")
        self.assertIn("Do not use no, not, never", constraints["mode_instruction"])

    def test_row_payload_forbids_only_new_control_status_markers(self):
        profile = generator.validate_profile(
            self.blocks[1], self.raw_profile(), self.authors
        )
        ordinary = generator.row_payload(
            self.blocks[1], self.source(1, 32), profile
        )["surface_constraints"]
        self.assertIn("fictional", ordinary["forbidden_new_control_status_markers"])

        inherited_source = copy.deepcopy(self.source(1, 32))
        inherited_source["question"] += " about a fictional book"
        inherited = generator.row_payload(
            self.blocks[1], inherited_source, profile
        )["surface_constraints"]
        self.assertIn("fictional", inherited["inherited_control_status_markers"])
        self.assertNotIn(
            "fictional", inherited["forbidden_new_control_status_markers"]
        )

    def test_v56_installs_direct_contrast_answer_prompt(self):
        generator.configure_shared_modules()
        self.assertEqual(generator.v55.ANSWER_PROMPT, generator.ANSWER_PROMPT)
        self.assertIn("required_mode_family", generator.ANSWER_PROMPT)

    def test_profile_cache_requires_independent_contrast_approval(self):
        raw = self.raw_profile()
        block = self.blocks[1]
        source_by_id = {row["source_id"]: row for row in block["sources"]}

        def fake_request(_client, _args, prompt, payload, _label):
            if prompt == generator.PROFILE_PROMPT:
                return raw
            if prompt == generator.PROFILE_JUDGE_PROMPT:
                result = self.valid_judgement()
                verdict = next(
                    row for row in result["verdicts"]
                    if row["source_id"] == "forget05_perturbed-00023"
                )
                verdict["core_fact_changed"] = False
                verdict["reason"] = "same award with a new qualifier"
                return result
            raise AssertionError(
                f"unexpected prompt for {payload.get('source_id', 'profile')}"
            )

        args = SimpleNamespace(
            model="offline", judge_model="offline",
            temperature=1.0, judge_temperature=1.0,
            request_retries=1, profile_retries=1,
        )
        original = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "direct profile failed"):
                    generator.generate_profile(
                        None, None, args, block, self.authors, Path(directory)
                    )
                self.assertFalse(
                    generator.v52.profile_path(Path(directory), 1).exists()
                )
        finally:
            generator.v52.v2.request_json = original
        self.assertEqual(len(source_by_id), 20)

    def test_legacy_versions_are_separate(self):
        self.assertEqual(
            generator.DESIGN_VERSION, "tofu-author-direct-contrast-v5.6"
        )
        self.assertNotEqual(generator.DESIGN_VERSION, generator.v55.V54_DESIGN_VERSION)
        self.assertIn("same award/book", generator.PROFILE_JUDGE_PROMPT)


if __name__ == "__main__":
    unittest.main()
