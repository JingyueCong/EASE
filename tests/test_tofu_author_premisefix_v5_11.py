import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "ULD/scripts/generate_tofu_author_premisefix_v5_11.py"
)
V59_TEST_PATH = ROOT / "tests/test_tofu_author_semantic_agent_v5_9.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_premisefix_v511_test", GENERATOR_PATH)
fixture = load_module("tofu_author_premisefix_v511_fixture", V59_TEST_PATH)


def premise_verdict(*, accepted=True, failed_field=None):
    result = {
        "accepted": accepted,
        **{field: True for field in generator.CRITIC_FIELDS},
        "reason": (
            "The question uses replacement-profile premises and the answer "
            "satisfies the rewritten question."
        ),
        "repair_instruction": "",
    }
    if failed_field:
        result[failed_field] = False
        result["accepted"] = False
        result["repair_instruction"] = (
            "Replace the source-specific premise with its replacement-profile "
            "counterpart."
        )
    return result


class AuthorPremiseFixV511Test(unittest.TestCase):
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

    def profile(self):
        profile, _ = self.helper.profile_and_candidate()
        profile["profile_digest"] = "profile-v511"
        profile["ledger_digest"] = "ledger-v511"
        return profile

    def args(self, row_retries=2):
        return SimpleNamespace(
            model="generator-model",
            judge_model="critic-model",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=1,
            row_retries=row_retries,
        )

    def test_policy_preserves_relation_but_replaces_source_factual_qualifiers(self):
        policy = generator.PREMISE_MAPPING_POLICY
        self.assertFalse(policy["source_specific_literals_are_invariants"])
        self.assertTrue(policy["question_must_be_true_of_replacement_profile"])
        self.assertTrue(policy["answer_must_satisfy_rewritten_question"])
        replacements = " ".join(policy["replace_when_source_specific"])
        self.assertIn("award or subject domains", replacements)
        self.assertIn("places, dates", replacements)
        self.assertIn("overrides any legacy", policy["precedence"])

    def test_context_packet_carries_policy_with_explicit_precedence(self):
        packet = generator.build_context_packet(
            self.blocks[1], self.source(29), self.profile()
        )
        policy = packet["premise_mapping_policy"]
        self.assertEqual(policy["version"], generator.PREMISE_POLICY_VERSION)
        self.assertIn("overrides", policy["precedence"])
        self.assertIn(
            "source-specific factual premises required to make C01 coherent",
            packet["pair_contract"]["change_only"],
        )
        authority = packet["evidence_authority_policy"]
        self.assertEqual(
            authority["version"], "frozen-ledger-authority-v1"
        )
        self.assertTrue(
            authority["highest_to_lowest"][0].startswith(
                "frozen_row_ledger"
            )
        )
        self.assertIn("semantic_brief", authority["conflict_rule"])
        self.assertIn("frozen row ledger", authority["conflict_rule"])

    def test_prompts_forbid_literal_source_premise_preservation(self):
        self.assertIn("Do not preserve a literal", generator.SEMANTIC_PLANNER_PROMPT)
        self.assertIn("question must be true", generator.PREMISE_GENERATOR_PROMPT)
        self.assertIn("must replace it", generator.PREMISE_CRITIC_PROMPT)
        self.assertIn("polarity", generator.PREMISE_CRITIC_PROMPT)
        self.assertIn(
            "semantic_brief is advisory only",
            generator.PREMISE_CRITIC_PROMPT,
        )
        self.assertIn(
            "Never reject a candidate for obeying the frozen row ledger",
            generator.PREMISE_CRITIC_PROMPT,
        )

    def test_legacy_brief_is_migrated_but_policy_overrides_must_preserve(self):
        profile = self.profile()
        source = self.source(29)
        old_base = generator.BASE_STATE_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, current = root / "base", root / "current"
                inherited = generator.v59.semantic_context_path(
                    base, 1, source["source_id"]
                )
                brief = fixture.semantic_brief()
                brief["must_preserve"] = [
                    "literal source book and source award domain"
                ]
                generator.v52.write_json(inherited, {
                    "design_version": "tofu-author-semantic-agent-v5.9",
                    "semantic_brief": brief,
                })
                generator.BASE_STATE_DIR = base
                packet, migrated, attempt = generator.load_or_create_semantic_brief(
                    object(), self.args(), self.blocks[1], source, profile, current
                )
                checkpoint = json.loads(
                    generator.v59.semantic_context_path(
                        current, 1, source["source_id"]
                    ).read_text()
                )
        finally:
            generator.BASE_STATE_DIR = old_base

        self.assertEqual(attempt, 0)
        self.assertEqual(
            migrated["schema_version"],
            generator.SEMANTIC_BRIEF_SCHEMA_VERSION,
        )
        self.assertTrue(
            checkpoint["premise_policy_overrides_legacy_must_preserve"]
        )
        self.assertEqual(
            packet["premise_mapping_policy"]["version"],
            generator.PREMISE_POLICY_VERSION,
        )

    def test_missing_v59_row_is_generated_under_new_policy(self):
        profile = self.profile()
        source = self.source(29)
        repaired = self.helper.when_candidate(profile)
        calls = []

        def fake_request(_client, _args, prompt, payload, _label):
            calls.append((prompt, payload))
            if prompt == generator.SEMANTIC_PLANNER_PROMPT:
                return fixture.semantic_brief()
            if prompt == generator.PREMISE_GENERATOR_PROMPT:
                return repaired
            if prompt == generator.PREMISE_CRITIC_PROMPT:
                return premise_verdict()
            raise AssertionError(f"unexpected prompt: {prompt[:50]}")

        old_request = generator.v52.v2.request_json
        old_base = generator.BASE_STATE_DIR
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                generator.BASE_STATE_DIR = root / "empty-base"
                result = generator.generate_row(
                    object(), object(), self.args(), self.blocks[1], source,
                    profile, root / "current",
                )
        finally:
            generator.BASE_STATE_DIR = old_base
            generator.v52.v2.request_json = old_request

        prompts = [prompt for prompt, _ in calls]
        self.assertEqual(prompts.count(generator.PREMISE_GENERATOR_PROMPT), 1)
        self.assertEqual(prompts.count(generator.PREMISE_CRITIC_PROMPT), 1)
        self.assertEqual(result["replacement_answer"], repaired["replacement_answer"])
        self.assertEqual(
            result["semantic_agent_trace"]["premise_policy_version"],
            generator.PREMISE_POLICY_VERSION,
        )
        generator_payload = next(
            payload for prompt, payload in calls
            if prompt == generator.PREMISE_GENERATOR_PROMPT
        )
        self.assertIn("No accepted V5.9 row exists", generator_payload[
            "validation_feedback"
        ])
        self.assertEqual(
            generator_payload["context_packet"]["premise_mapping_policy"][
                "version"
            ],
            generator.PREMISE_POLICY_VERSION,
        )

    def test_explicit_human_repair_is_audited_by_independent_critic(self):
        profile = self.profile()
        source = self.source(29)
        candidate = self.helper.when_candidate(profile)
        directive = {
            "c01_question": candidate["c01_question"],
            "replacement_answer": candidate["replacement_answer"],
            "category": "test_general_repair",
            "reason": "Exercise the manifest-driven repair path.",
        }
        calls = []

        def fake_load(*_args):
            packet = generator.build_context_packet(
                self.blocks[1], source, profile
            )
            return packet, fixture.semantic_brief(), 0

        def fake_critic(*args):
            calls.append(args)
            return premise_verdict()

        old_load = generator.load_or_create_semantic_brief
        old_critic = generator.v59.request_critic_verdict
        old_path = generator.HUMAN_REPAIR_MANIFEST_PATH
        old_digest = generator.HUMAN_REPAIR_MANIFEST_DIGEST
        generator.load_or_create_semantic_brief = fake_load
        generator.v59.request_critic_verdict = fake_critic
        generator.HUMAN_REPAIR_MANIFEST_PATH = Path("repairs.json")
        generator.HUMAN_REPAIR_MANIFEST_DIGEST = "repair-digest"
        try:
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                result = generator.apply_human_candidate(
                    object(), object(), self.args(), self.blocks[1], source,
                    profile, state, directive,
                )
                checkpoint = json.loads(
                    generator.v52.row_path(
                        state, 1, source["source_id"]
                    ).read_text()
                )
        finally:
            generator.load_or_create_semantic_brief = old_load
            generator.v59.request_critic_verdict = old_critic
            generator.HUMAN_REPAIR_MANIFEST_PATH = old_path
            generator.HUMAN_REPAIR_MANIFEST_DIGEST = old_digest

        self.assertEqual(len(calls), 1)
        trace = result["semantic_agent_trace"]
        self.assertEqual(trace["generator_model"], "explicit-human-repair")
        self.assertTrue(trace["critic_independent_call"])
        self.assertEqual(
            trace["human_manual_repair"]["manifest_digest"],
            "repair-digest",
        )
        self.assertEqual(
            checkpoint["semantic_agent_trace"]["human_manual_repair"][
                "category"
            ],
            "test_general_repair",
        )

    def test_rejected_human_repair_is_not_written_as_an_accepted_row(self):
        profile = self.profile()
        source = self.source(29)
        candidate = self.helper.when_candidate(profile)
        directive = {
            "c01_question": candidate["c01_question"],
            "replacement_answer": candidate["replacement_answer"],
            "category": "test_rejection",
            "reason": "Ensure a critic rejection cannot be bypassed.",
        }

        def fake_load(*_args):
            packet = generator.build_context_packet(
                self.blocks[1], source, profile
            )
            return packet, fixture.semantic_brief(), 0

        old_load = generator.load_or_create_semantic_brief
        old_critic = generator.v59.request_critic_verdict
        generator.load_or_create_semantic_brief = fake_load
        generator.v59.request_critic_verdict = lambda *_args: premise_verdict(
            accepted=False, failed_field="question_premises_updated"
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                with self.assertRaisesRegex(
                    ValueError, "failed independent critic"
                ):
                    generator.apply_human_candidate(
                        object(), object(), self.args(), self.blocks[1],
                        source, profile, state, directive,
                    )
                self.assertFalse(
                    generator.v52.row_path(
                        state, 1, source["source_id"]
                    ).is_file()
                )
        finally:
            generator.load_or_create_semantic_brief = old_load
            generator.v59.request_critic_verdict = old_critic

    def test_v511_is_versioned_and_legacy_generators_remain(self):
        self.assertEqual(
            generator.DESIGN_VERSION, "tofu-author-premisefix-v5.11"
        )
        for version in (
            "generate_tofu_author_semantic_agent_v5_9.py",
            "generate_tofu_author_pairrepair_v5_10.py",
        ):
            self.assertTrue((ROOT / "ULD/scripts" / version).is_file())
        self.assertIn(
            generator.DESIGN_VERSION,
            generator.ciru.AUTHOR_LEVEL_DESIGNS,
        )


if __name__ == "__main__":
    unittest.main()
