import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "ULD/scripts/generate_tofu_author_pairrepair_v5_10.py"
)
V59_TEST_PATH = ROOT / "tests/test_tofu_author_semantic_agent_v5_9.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_pairrepair_v510_test", GENERATOR_PATH)
fixture = load_module("tofu_author_pairrepair_v510_fixture", V59_TEST_PATH)


def pair_verdict(*, accepted=True, failed_field=None):
    result = {
        "accepted": accepted,
        **{field: True for field in generator.CRITIC_FIELDS},
        "reason": "The relation, scope, evidence, and granularity are matched.",
        "repair_instruction": "",
    }
    if failed_field:
        result[failed_field] = False
        result["accepted"] = False
        result["repair_instruction"] = "Repair only the failed pair property."
    return result


class AuthorPairRepairV510Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorSemanticAgentV59Test.setUpClass()
        cls.helper = fixture.AuthorSemanticAgentV59Test(
            "test_context_packet_contains_complete_reproducible_context"
        )
        cls.blocks = fixture.AuthorSemanticAgentV59Test.blocks
        generator.configure_shared_modules()

    def source(self, suffix):
        return self.helper.source(suffix)

    def profile_and_candidate(self):
        profile, candidate = self.helper.profile_and_candidate()
        profile["profile_digest"] = "profile-v510"
        profile["ledger_digest"] = "ledger-v510"
        return profile, candidate

    def inherited_candidate(self, profile):
        return self.helper.when_candidate(profile)

    def semantic_brief(self):
        return fixture.semantic_brief()

    def args(self, row_retries=2):
        return SimpleNamespace(
            model="generator-model",
            judge_model="critic-model",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=1,
            row_retries=row_retries,
        )

    def seed_base_row(self, base, source_id, candidate):
        path = generator.v52.row_path(base, 1, source_id)
        generator.v52.write_json(path, {
            "design_version": "tofu-author-semantic-agent-v5.9",
            "candidate": candidate,
        })
        context = generator.v59.semantic_context_path(base, 1, source_id)
        generator.v52.write_json(context, {
            "design_version": "tofu-author-semantic-agent-v5.9",
            "semantic_brief": self.semantic_brief(),
        })

    def test_pair_contract_adds_semantic_checks_not_format_equality(self):
        self.assertIn("question_scope_matched", generator.CRITIC_FIELDS)
        self.assertIn("evidence_status_matched", generator.CRITIC_FIELDS)
        self.assertIn("information_granularity_matched", generator.CRITIC_FIELDS)
        self.assertIn("source_comparison_absent", generator.CRITIC_FIELDS)
        self.assertIn("Do NOT reject only because", generator.PAIR_CRITIC_PROMPT)
        self.assertNotIn("word count must", generator.PAIR_CRITIC_PROMPT)

    def test_nuisance_counts_are_observations_not_hard_gates(self):
        profile, _ = self.profile_and_candidate()
        candidate = self.inherited_candidate(profile)
        candidate["replacement_answer"] = (
            "Yes — " + candidate["replacement_answer"]
        )
        validated = generator.validate_structural_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        observations = validated["deterministic_observations"]
        self.assertTrue(
            observations["counts_are_critic_evidence_not_hard_gates"]
        )
        self.assertGreater(observations["answer_length_ratio"], 0)

    def test_critic_requires_every_pair_check(self):
        parsed = generator.v59.parse_critic_verdict(pair_verdict())
        self.assertTrue(parsed["accepted"])
        rejected = generator.v59.parse_critic_verdict(
            pair_verdict(failed_field="source_comparison_absent")
        )
        self.assertFalse(rejected["accepted"])
        self.assertIn(
            "source_comparison_absent",
            generator.v59.critic_feedback(rejected),
        )

    def test_passing_v59_c01_is_reused_without_generator_call(self):
        profile, _ = self.profile_and_candidate()
        source = self.source(29)
        candidate = self.inherited_candidate(profile)
        calls = []

        def fake_request(_client, _args, prompt, _payload, _label):
            calls.append(prompt)
            if prompt == generator.PAIR_CRITIC_PROMPT:
                return pair_verdict()
            raise AssertionError("passing inherited row must not call generator")

        old_request = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        old_base = generator.BASE_STATE_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, current = root / "base", root / "current"
                self.seed_base_row(base, source["source_id"], candidate)
                generator.BASE_STATE_DIR = base
                generator.BASE_DATA_DIGEST = "a" * 64
                result = generator.generate_row(
                    object(), object(), self.args(), self.blocks[1], source,
                    profile, current,
                )
                checkpoint = json.loads(
                    generator.v52.row_path(
                        current, 1, source["source_id"]
                    ).read_text()
                )
        finally:
            generator.BASE_STATE_DIR = old_base
            generator.v52.v2.request_json = old_request

        self.assertEqual(calls, [generator.PAIR_CRITIC_PROMPT])
        self.assertEqual(
            result["replacement_answer"], candidate["replacement_answer"]
        )
        self.assertEqual(checkpoint["mapping_attempt"], 0)
        self.assertEqual(
            checkpoint["semantic_agent_trace"]["generator_model"],
            "inherited-v5.9",
        )

    def test_rejected_v59_c01_repairs_only_that_row(self):
        profile, _ = self.profile_and_candidate()
        source = self.source(29)
        inherited = self.inherited_candidate(profile)
        repaired = dict(inherited)
        repaired["replacement_answer"] = (
            "Harper Mei Collins' 'When Bridges Sleep' received the "
            "Lighthouse Medal for Coastal Fiction."
        )
        calls = []

        def fake_request(_client, _args, prompt, payload, _label):
            calls.append((prompt, payload))
            if prompt == generator.PAIR_CRITIC_PROMPT:
                critic_calls = [
                    item for item in calls
                    if item[0] == generator.PAIR_CRITIC_PROMPT
                ]
                if len(critic_calls) == 1:
                    return pair_verdict(
                        failed_field="information_granularity_matched"
                    )
                return pair_verdict()
            if prompt == generator.PAIR_GENERATOR_PROMPT:
                return repaired
            raise AssertionError(f"unexpected prompt: {prompt[:40]}")

        old_request = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        old_base = generator.BASE_STATE_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, current = root / "base", root / "current"
                self.seed_base_row(base, source["source_id"], inherited)
                generator.BASE_STATE_DIR = base
                generator.BASE_DATA_DIGEST = "b" * 64
                result = generator.generate_row(
                    object(), object(), self.args(), self.blocks[1], source,
                    profile, current,
                )
        finally:
            generator.BASE_STATE_DIR = old_base
            generator.v52.v2.request_json = old_request

        prompts = [item[0] for item in calls]
        self.assertEqual(prompts.count(generator.PAIR_GENERATOR_PROMPT), 1)
        self.assertEqual(prompts.count(generator.PAIR_CRITIC_PROMPT), 2)
        self.assertEqual(result["repair_generation"], 1)
        generator_payload = next(
            payload for prompt, payload in calls
            if prompt == generator.PAIR_GENERATOR_PROMPT
        )
        self.assertIn("information_granularity_matched", generator_payload[
            "validation_feedback"
        ])
        self.assertEqual(
            generator_payload["previous_candidate"]["c01_question"],
            inherited["c01_question"],
        )

    def test_human_repair_manifest_overrides_a_false_positive_critic(self):
        profile, _ = self.profile_and_candidate()
        source = self.source(29)
        inherited = self.inherited_candidate(profile)
        repaired = dict(inherited)
        repaired["replacement_answer"] = (
            "Harper Mei Collins' 'When Bridges Sleep' received one "
            "coastal-fiction award."
        )
        calls = []

        def fake_request(_client, _args, prompt, _payload, _label):
            calls.append(prompt)
            if prompt == generator.PAIR_CRITIC_PROMPT:
                return pair_verdict()
            if prompt == generator.PAIR_GENERATOR_PROMPT:
                return repaired
            raise AssertionError(f"unexpected prompt: {prompt[:40]}")

        old_request = generator.v52.v2.request_json
        old_base = generator.BASE_STATE_DIR
        old_repairs = generator.HUMAN_REPAIRS
        old_digest = generator.REPAIR_MANIFEST_DIGEST
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, current = root / "base", root / "current"
                self.seed_base_row(base, source["source_id"], inherited)
                generator.BASE_STATE_DIR = base
                generator.HUMAN_REPAIRS = {
                    source["source_id"]: {
                        "category": "PAIR_MISMATCH",
                        "reason": "Human audit found excess granularity.",
                    }
                }
                generator.REPAIR_MANIFEST_DIGEST = "c" * 64
                result = generator.generate_row(
                    object(), object(), self.args(), self.blocks[1], source,
                    profile, current,
                )
        finally:
            generator.BASE_STATE_DIR = old_base
            generator.HUMAN_REPAIRS = old_repairs
            generator.REPAIR_MANIFEST_DIGEST = old_digest
            generator.v52.v2.request_json = old_request

        self.assertEqual(calls.count(generator.PAIR_GENERATOR_PROMPT), 1)
        self.assertEqual(calls.count(generator.PAIR_CRITIC_PROMPT), 2)
        self.assertEqual(
            result["semantic_agent_trace"]["human_audit_repair"]["category"],
            "PAIR_MISMATCH",
        )

    def test_profile_and_ledger_are_migrated_without_generation(self):
        profile, _ = self.profile_and_candidate()
        block = self.blocks[1]
        protected = [
            item["target_entity"] for item in self.blocks.values()
        ]
        old_base = generator.BASE_STATE_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, current = root / "base", root / "current"
                inherited = generator.v52.profile_path(base, 1)
                generator.v52.write_json(inherited, {
                    "design_version": "tofu-author-semantic-agent-v5.9",
                    "anchor_catalog_digest": block["anchor_catalog"]["digest"],
                    "ledger_digest": profile["ledger_digest"],
                    "profile": {
                        key: value for key, value in profile.items()
                        if key not in {
                            "fact_ledger_by_source", "profile_semantic_judge",
                            "profile_digest", "ledger_digest",
                        }
                    },
                    "semantic_judge": {
                        "coherent": True,
                        "conflicting_source_ids": [],
                        "reason": "Frozen V5.9 profile is coherent.",
                    },
                })
                generator.BASE_STATE_DIR = base
                migrated = generator.migrate_profile(
                    object(), object(), self.args(), block, protected, current
                )
                checkpoint = json.loads(
                    generator.v52.profile_path(current, 1).read_text()
                )
        finally:
            generator.BASE_STATE_DIR = old_base

        self.assertEqual(
            migrated["replacement_entity"], profile["replacement_entity"]
        )
        self.assertEqual(
            migrated["fact_ledger"],
            generator.v53.validate_profile(
                block, checkpoint["profile"], protected
            )["fact_ledger"],
        )
        self.assertTrue(checkpoint["frozen_profile"])
        self.assertEqual(checkpoint["design_version"], generator.DESIGN_VERSION)

    def test_v510_is_versioned_and_legacy_generators_remain(self):
        self.assertEqual(
            generator.DESIGN_VERSION, "tofu-author-pairrepair-v5.10"
        )
        for version in (
            "generate_tofu_author_direct_v5_6.py",
            "generate_tofu_author_joint_v5_7.py",
            "generate_tofu_author_semantic_v5_8.py",
            "generate_tofu_author_semantic_agent_v5_9.py",
        ):
            self.assertTrue((ROOT / "ULD/scripts" / version).is_file())
        self.assertIn(
            generator.DESIGN_VERSION,
            generator.ciru.AUTHOR_LEVEL_DESIGNS,
        )


if __name__ == "__main__":
    unittest.main()
