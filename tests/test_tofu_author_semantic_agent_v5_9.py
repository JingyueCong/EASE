import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "ULD/scripts/generate_tofu_author_semantic_agent_v5_9.py"
)
V58_TEST_PATH = ROOT / "tests/test_tofu_author_semantic_v5_8.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_semantic_agent_v59_test", GENERATOR_PATH)
fixture = load_module("tofu_author_semantic_agent_v59_fixture", V58_TEST_PATH)


def semantic_brief():
    return {
        "same_relation_definition": "award received for a named book",
        "answer_object_type": "award",
        "cardinality": "one",
        "response_mode": "positive",
        "source_specific_premises": ["source author and source book"],
        "replacement_specific_premises": ["replacement author and book"],
        "must_preserve": ["award-for-book relation"],
        "must_change": ["author, book, and award"],
        "risk_notes": ["When is part of the replacement book title"],
    }


def critic_verdict(*, accepted=True, failed_field=None):
    result = {
        "accepted": accepted,
        **{field: True for field in generator.CRITIC_FIELDS},
        "reason": "The candidate preserves the award-for-book relation.",
        "repair_instruction": "",
    }
    if failed_field:
        result[failed_field] = False
        result["accepted"] = False
        result["repair_instruction"] = "Repair the named semantic defect."
    return result


class AuthorSemanticAgentV59Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AuthorSemanticV58Test.setUpClass()
        cls.helper = fixture.AuthorSemanticV58Test(
            "test_minimal_candidate_needs_only_question_and_answer"
        )
        cls.blocks = fixture.AuthorSemanticV58Test.blocks
        generator.configure_shared_modules()

    def source(self, suffix):
        return self.helper.source(suffix)

    def profile_and_candidate(self):
        profile, candidate = self.helper.profile_and_content()
        return profile, candidate

    def when_candidate(self, profile):
        profile["fact_ledger_by_source"][
            "forget05_perturbed-00029"
        ]["replacement_fact"] = (
            "Harper Mei Collins' 'When Bridges Sleep' received the "
            "Lighthouse Medal for Coastal Fiction."
        )
        return {
            "c01_question": (
                "Can you share a fictitious award that Harper Mei Collins "
                "received for the book 'When Bridges Sleep'?"
            ),
            "replacement_answer": (
                "Harper Mei Collins' When Bridges Sleep received the "
                "Lighthouse Medal for Coastal Fiction."
            ),
        }

    def test_context_packet_contains_complete_reproducible_context(self):
        profile, _ = self.profile_and_candidate()
        packet = generator.build_context_packet(
            self.blocks[1], self.source(29), profile
        )
        self.assertEqual(packet["causal_estimand"], "(C11-C01)-(C10-C00)")
        self.assertEqual(
            packet["immutable_c11"]["source_id"],
            "forget05_perturbed-00029",
        )
        self.assertEqual(len(packet["complete_replacement_ledger"]), 20)
        self.assertEqual(
            packet["frozen_row_ledger"]["replacement_fact"],
            profile["fact_ledger_by_source"][
                "forget05_perturbed-00029"
            ]["replacement_fact"],
        )
        self.assertTrue(
            packet["hard_boundaries"]["semantic_properties_are_critic_owned"]
        )

    def test_context_packet_declares_required_identity_literal(self):
        profile, _ = self.profile_and_candidate()
        packet = generator.build_context_packet(
            self.blocks[1], self.source(32), profile
        )
        self.assertEqual(
            packet["hard_boundaries"][
                "required_question_identity_literal"
            ],
            profile["replacement_entity"],
        )

    def test_missing_question_identity_reports_exact_required_literal(self):
        profile, _ = self.profile_and_candidate()
        with self.assertRaisesRegex(
            ValueError, profile["replacement_entity"]
        ):
            generator.validate_structural_candidate(
                self.blocks[1],
                self.source(32),
                profile,
                {
                    "c01_question": (
                        "How does their identity affect the literary scene?"
                    ),
                    "replacement_answer": (
                        f"{profile['replacement_entity']} adds a new "
                        "perspective to the literary scene."
                    ),
                },
            )

    def test_semantic_brief_has_a_strict_schema(self):
        parsed = generator.validate_semantic_brief(semantic_brief())
        self.assertEqual(
            parsed["schema_version"], generator.SEMANTIC_BRIEF_SCHEMA_VERSION
        )
        broken = semantic_brief()
        del broken["answer_object_type"]
        with self.assertRaisesRegex(ValueError, "answer_object_type"):
            generator.validate_semantic_brief(broken)

    def test_when_book_title_is_not_a_deterministic_semantic_reject(self):
        profile, _ = self.profile_and_candidate()
        candidate = self.when_candidate(profile)
        validated = generator.validate_structural_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertEqual(
            validated["response_contract_question_source"],
            "immutable_c11_advisory_only",
        )
        self.assertEqual(
            validated["deterministic_observations"]["validation_role"],
            "critic_context_only",
        )

    def test_fictional_inherits_fictitious_source_instruction(self):
        profile, _ = self.profile_and_candidate()
        candidate = self.when_candidate(profile)
        candidate["c01_question"] = candidate["c01_question"].replace(
            "fictitious", "fictional"
        )
        validated = generator.validate_structural_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertIn("fictional", validated["c01_question"].casefold())

    def test_lexical_overlap_is_an_observation_not_a_hard_gate(self):
        profile, _ = self.profile_and_candidate()
        candidate = {
            "c01_question": (
                "Which honor recognized Harper Mei Collins' replacement novel?"
            ),
            "replacement_answer": (
                "Her replacement novel earned a coastal-fiction distinction."
            ),
        }
        validated = generator.validate_structural_candidate(
            self.blocks[1], self.source(29), profile, candidate
        )
        self.assertIn("deterministic_observations", validated)

    def test_critic_acceptance_is_computed_from_every_check(self):
        accepted = generator.parse_critic_verdict(critic_verdict())
        self.assertTrue(accepted["accepted"])
        rejected = generator.parse_critic_verdict(
            critic_verdict(failed_field="answer_addresses_question")
        )
        self.assertFalse(rejected["accepted"])
        self.assertIn("answer_addresses_question", generator.critic_feedback(rejected))

    def test_agent_repairs_only_after_independent_critic_rejects(self):
        profile, _ = self.profile_and_candidate()
        profile["profile_digest"] = "profile-digest"
        profile["ledger_digest"] = "ledger-digest"
        source = self.source(29)
        calls = []
        candidates = [
            {
                "c01_question": (
                    "What did Harper Mei Collins receive for the replacement book?"
                ),
                "replacement_answer": "It appeared in coastal fiction.",
            },
            self.when_candidate(profile),
        ]

        def fake_request(_client, _args, prompt, payload, _label):
            calls.append((prompt, payload))
            if prompt == generator.SEMANTIC_PLANNER_PROMPT:
                return semantic_brief()
            if prompt == generator.SEMANTIC_GENERATOR_PROMPT:
                return candidates.pop(0)
            if prompt == generator.SEMANTIC_CRITIC_PROMPT:
                critic_calls = [
                    item for item in calls
                    if item[0] == generator.SEMANTIC_CRITIC_PROMPT
                ]
                if len(critic_calls) == 1:
                    return critic_verdict(
                        failed_field="answer_addresses_question"
                    )
                return critic_verdict()
            raise AssertionError("unexpected prompt")

        args = SimpleNamespace(
            model="generator-model",
            judge_model="critic-model",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=1,
            row_retries=2,
        )
        old_request = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generator.generate_row(
                    object(),
                    object(),
                    args,
                    self.blocks[1],
                    source,
                    profile,
                    Path(directory),
                )
                attempt_files = list(
                    Path(directory).rglob("*.attempt_*.json")
                )
                statuses = {
                    json.loads(path.read_text())["status"]
                    for path in attempt_files
                }
        finally:
            generator.v52.v2.request_json = old_request

        self.assertEqual(result["mapping_attempt"], 2)
        self.assertTrue(
            result["semantic_agent_trace"]["critic_verdict"]["accepted"]
        )
        self.assertEqual(statuses, {"accepted", "rejected"})
        second_generator_payload = [
            payload for prompt, payload in calls
            if prompt == generator.SEMANTIC_GENERATOR_PROMPT
        ][1]
        self.assertIn("Independent semantic critic rejected", second_generator_payload[
            "validation_feedback"
        ])
        self.assertIn("previous_candidate", second_generator_payload)
        self.assertEqual(
            second_generator_payload["generator_contract"][
                "required_question_identity_literal"
            ],
            profile["replacement_entity"],
        )

    def test_v59_is_new_and_all_legacy_files_remain(self):
        self.assertEqual(
            generator.DESIGN_VERSION, "tofu-author-semantic-agent-v5.9"
        )
        for version in (
            "generate_tofu_author_direct_v5_6.py",
            "generate_tofu_author_joint_v5_7.py",
            "generate_tofu_author_semantic_v5_8.py",
        ):
            self.assertTrue((ROOT / "ULD/scripts" / version).is_file())

    def test_v58_and_v59_are_registered_as_author_level_designs(self):
        self.assertIn(
            "tofu-author-semantic-contrast-v5.8",
            generator.ciru.AUTHOR_LEVEL_DESIGNS,
        )
        self.assertIn(
            generator.DESIGN_VERSION,
            generator.ciru.AUTHOR_LEVEL_DESIGNS,
        )

    def test_configure_installs_agent_pipeline(self):
        generator.configure_shared_modules()
        self.assertIs(generator.v55.validate_row_candidate, generator.validate_row_candidate)
        self.assertIs(generator.v55.materialize_plan, generator.materialize_plan)
        self.assertIs(generator.v58.generate_block, generator.generate_block)
        self.assertEqual(generator.v58.DESIGN_VERSION, generator.DESIGN_VERSION)

    def test_twenty_row_plan_and_assembly_keep_agent_provenance(self):
        profile, _ = self.profile_and_candidate()
        profile["profile_semantic_judge"] = {
            "coherent": True,
            "contrastive": True,
        }
        candidates = {}
        for index, source in enumerate(self.blocks[1]["sources"]):
            validated = generator.validate_structural_candidate(
                self.blocks[1],
                source,
                profile,
                {
                    "c01_question": generator.v55.c01_question(
                        self.blocks[1], source, profile
                    ),
                    "replacement_answer": (
                        f"{profile['replacement_entity']} has replacement "
                        f"profile fact number {index + 1}."
                    ),
                },
            )
            validated.update({
                "mapping_attempt": 1,
                "repair_generation": 0,
                "semantic_agent_trace": {
                    "agent_protocol_version": generator.AGENT_PROTOCOL_VERSION,
                    "context_digest": "0" * 64,
                    "critic_verdict": {"accepted": True},
                },
            })
            candidates[source["source_id"]] = validated

        plan = generator.materialize_plan(
            self.blocks[1], profile, candidates, 42
        )
        verdicts = {
            source["source_id"]: {
                **{field: True for field in generator.v58.JUDGE_FIELDS},
                "reason": "offline plumbing check",
            }
            for source in self.blocks[1]["sources"]
        }
        assembled = generator.assemble_block(
            self.blocks[1],
            plan,
            verdicts,
            SimpleNamespace(
                split="forget05_perturbed",
                seed=42,
                model="offline",
                judge_model="offline",
            ),
        )
        self.assertEqual(len(assembled["records"]), 20)
        self.assertTrue(all(
            row["generation"]["independent_row_critic"] is True
            for row in assembled["records"]
        ))
        self.assertTrue(all(
            row["design_version"] == generator.DESIGN_VERSION
            for row in assembled["records"]
        ))


if __name__ == "__main__":
    unittest.main()
