import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_contract_v3.py"
CIRU_PATH = ROOT / "ULD/uld/data/ciru.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_contract_v3", GENERATOR_PATH)
ciru = load_module("tofu_author_contract_v3_ciru", CIRU_PATH)


def block_fixture():
    return generator.attach_contracts(
        {
            "block_id": 0,
            "target_entity": "Hina Ameen",
            "sources": [
                {
                    "source_id": "forget05_perturbed-00000",
                    "question": "Where was Hina Ameen born?",
                    "answer": "Hina Ameen was born in Karachi, Pakistan.",
                },
                {
                    "source_id": "forget05_perturbed-00001",
                    "question": "Which award did Hina Ameen win?",
                    "answer": "Hina Ameen won the Harbor Prize.",
                },
            ],
        }
    )


def plan_fixture():
    return {
        "target_entity": "Hina Ameen",
        "replacement_entity": "Mariselle Voss",
        "profile_summary": "A Bellhaven novelist who received the North Star Prize.",
        "row_plans": [
            {
                "source_id": "forget05_perturbed-00000",
                "target_relation": "birthplace",
                "replacement_value": "Bellhaven, Norland",
                "placebo_relation": "preferred writing season",
                "target_placebo_value": "quiet winters",
                "replacement_placebo_value": "calm autumns",
                "placebo_rationale": "Writing season does not reveal birthplace.",
            },
            {
                "source_id": "forget05_perturbed-00001",
                "target_relation": "award won",
                "replacement_value": "North Star Prize",
                "placebo_relation": "preferred writing room",
                "target_placebo_value": "upstairs study",
                "replacement_placebo_value": "garden studio",
                "placebo_rationale": "Writing room does not reveal an award.",
            },
        ],
    }


def rendered_fixture():
    return {
        "rows": [
            {
                "source_id": "forget05_perturbed-00000",
                "C01": {
                    "question": "Where was Mariselle Voss born?",
                    "answer": "Mariselle Voss was born in Bellhaven, Norland.",
                },
                "C10": {
                    "question": "When did Hina Ameen prefer writing?",
                    "answer": "Hina Ameen preferred writing during quiet winters.",
                },
                "C00": {
                    "question": "When did Mariselle Voss prefer writing?",
                    "answer": "Mariselle Voss preferred writing during calm autumns.",
                },
            },
            {
                "source_id": "forget05_perturbed-00001",
                "C01": {
                    "question": "Which award did Mariselle Voss win?",
                    "answer": "Mariselle Voss won the North Star Prize.",
                },
                "C10": {
                    "question": "Where did Hina Ameen prefer writing?",
                    "answer": "Hina Ameen preferred an upstairs study.",
                },
                "C00": {
                    "question": "Where did Mariselle Voss prefer writing?",
                    "answer": "Mariselle Voss preferred a garden studio.",
                },
            },
        ]
    }


def judgement_fixture(accepted=True):
    verdicts = []
    for source_id in ("forget05_perturbed-00000", "forget05_perturbed-00001"):
        verdict = {
            "source_id": source_id,
            **{field: True for field in generator.JUDGE_FIELDS},
            "reason": "The assigned relation and evidence satisfy the contract.",
        }
        verdict["placebo_exclusion"] = accepted
        verdicts.append(verdict)
    return {"verdicts": verdicts}


class AuthorContractV3Test(unittest.TestCase):
    def test_full_name_is_an_identity_intervention(self):
        self.assertTrue(generator.is_identity_relation("full name of the author"))
        self.assertFalse(generator.is_identity_relation("author birthplace"))

        block = generator.attach_contracts({
            "block_id": 1,
            "target_entity": "Xin Lee Williams",
            "sources": [{
                "source_id": "forget05_perturbed-00020",
                "question": "What is the author's full name?",
                "answer": "The author's full name is Xin Lee Williams.",
            }],
        })
        raw_plan = {
            "target_entity": "Xin Lee Williams",
            "replacement_entity": "Harper Mei Collins",
            "profile_summary": "A novelist with a structured editorial practice.",
            "row_plans": [{
                "source_id": "forget05_perturbed-00020",
                "target_relation": "full name of the author",
                "replacement_value": "Harper Mei Collins",
                "placebo_relation": "editorial review workflow",
                "target_placebo_value": "Xin Lee Williams uses two review rounds",
                "replacement_placebo_value": "Harper Mei Collins uses three review rounds",
                "placebo_rationale": "Editorial workflow does not reveal identity.",
            }],
        }
        plan = generator.validate_plan(
            block, raw_plan, ["Xin Lee Williams"],
            min_unique_placebos=1, max_placebo_reuse=1,
        )
        rendered = generator.validate_rendered_rows(
            block,
            plan,
            {"rows": [{
                "source_id": "forget05_perturbed-00020",
                "C01": {
                    "question": "What is the author's full name?",
                    "answer": "The author's full name is Harper Mei Collins.",
                },
                "C10": {
                    "question": "What editorial review workflow does the author use?",
                    "answer": "Xin Lee Williams uses two review rounds.",
                },
                "C00": {
                    "question": "What editorial review workflow does the author use?",
                    "answer": "Harper Mei Collins uses three review rounds.",
                },
            }]},
            ["forget05_perturbed-00020"],
        )
        self.assertIn("forget05_perturbed-00020", rendered)

    def test_contract_is_deterministically_derived_from_c11(self):
        contract = block_fixture()["sources"][0]["contract"]
        self.assertEqual(contract["response_mode"], "affirmative")
        self.assertEqual(contract["answer_format"], "short_prose")
        self.assertEqual(contract["fact_count"], 1)
        self.assertTrue(contract["explicit_target_in_question"])

    def test_plan_requires_diverse_orthogonal_placebos(self):
        block = block_fixture()
        validated = generator.validate_plan(
            block,
            plan_fixture(),
            ["Hina Ameen", "Xin Lee Williams"],
            min_unique_placebos=2,
            max_placebo_reuse=1,
        )
        self.assertEqual(len(validated["placebo_relation_counts"]), 2)

        invalid = plan_fixture()
        invalid["row_plans"][1]["placebo_relation"] = "preferred writing season"
        with self.assertRaisesRegex(ValueError, "diversity|reused"):
            generator.validate_plan(
                block,
                invalid,
                ["Hina Ameen"],
                min_unique_placebos=2,
                max_placebo_reuse=1,
            )

    def test_repeated_relation_cannot_hide_conflicting_profile_facts(self):
        invalid = plan_fixture()
        invalid["row_plans"][1]["target_relation"] = "birthplace"
        with self.assertRaisesRegex(ValueError, "conflicting replacement facts"):
            generator.validate_plan(
                block_fixture(),
                invalid,
                ["Hina Ameen"],
                min_unique_placebos=2,
                max_placebo_reuse=1,
            )

    def test_target_placebo_may_name_its_assigned_target_author(self):
        named = plan_fixture()
        named["row_plans"][0]["target_placebo_value"] = (
            "Hina Ameen preferred quiet winters"
        )
        validated = generator.validate_plan(
            block_fixture(),
            named,
            ["Hina Ameen"],
            min_unique_placebos=2,
            max_placebo_reuse=1,
        )
        self.assertIn(
            "Hina Ameen",
            validated["row_plans"]["forget05_perturbed-00000"]
            ["target_placebo_value"],
        )

    def test_lifestyle_trivia_is_not_a_causal_placebo(self):
        invalid = plan_fixture()
        invalid["row_plans"][0]["placebo_relation"] = "favorite_food"
        with self.assertRaisesRegex(ValueError, "domain-mismatched trivia"):
            generator.validate_plan(
                block_fixture(),
                invalid,
                ["Hina Ameen"],
                min_unique_placebos=2,
                max_placebo_reuse=1,
            )

    def test_render_and_semantic_judge_are_hard_gates(self):
        block = block_fixture()
        plan = generator.validate_plan(
            block,
            plan_fixture(),
            ["Hina Ameen"],
            min_unique_placebos=2,
            max_placebo_reuse=1,
        )
        source_ids = [source["source_id"] for source in block["sources"]]
        rendered = generator.validate_rendered_rows(
            block, plan, rendered_fixture(), source_ids
        )
        self.assertEqual(set(rendered), set(source_ids))
        with self.assertRaisesRegex(ValueError, "semantic judge rejected"):
            generator.validate_judgement(judgement_fixture(False), source_ids)

    def test_assembled_v3_remains_fullanswer_adapter_compatible(self):
        block = block_fixture()
        plan = generator.validate_plan(
            block,
            plan_fixture(),
            ["Hina Ameen"],
            min_unique_placebos=2,
            max_placebo_reuse=1,
        )
        source_ids = [source["source_id"] for source in block["sources"]]
        rendered = generator.validate_rendered_rows(
            block, plan, rendered_fixture(), source_ids
        )
        verdicts = generator.validate_judgement(
            judgement_fixture(True), source_ids
        )
        assembled = generator.assemble_author_block_v3(
            block,
            plan,
            rendered,
            verdicts,
            split="forget05_perturbed",
            seed=42,
            model="generator-test",
            judge_model="judge-test",
        )
        self.assertEqual(assembled["design_version"], ciru.AUTHOR_CONTRACT_DESIGN)
        self.assertEqual(len(assembled["records"]), 2)
        self.assertEqual(ciru.validate_ciru_unit(assembled["records"][0]), [])
        a1_ce, a1_uniform = ciru.factorial_dual_roles(
            assembled["records"], "f2d_did_a1"
        )
        self.assertEqual(a1_ce[0], assembled["records"][0]["cells"]["C11"])
        self.assertEqual(a1_uniform[0], assembled["records"][0]["cells"]["C01"])


if __name__ == "__main__":
    unittest.main()
