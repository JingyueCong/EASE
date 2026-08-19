import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_ledger_v5_3.py"
EVAL_LOG = (
    ROOT / "ULD/data/retain95_llama_wd0.01/eval_results/"
    "ds_size300/eval_log_forget.json"
)
MANIFEST = ROOT / "ULD/configs/data/tofu_forget05_author_blocks.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_ledger_v53_test", GENERATOR_PATH)
anchor = generator.anchors


def source_rows():
    with EVAL_LOG.open(encoding="utf-8") as handle:
        generated = json.load(handle)["generated_text"]
    rows = []
    for index, (prompt, _model_text, ground_truth) in enumerate(generated):
        question = re.sub(r"^\[INST\]\s*|\s*\[/INST\]$", "", prompt).strip()
        rows.append((index, question, ground_truth))
    return rows


def fixture_value(group, ordinal):
    old, kind = group["text"], group["kind"]
    if anchor.normalise(old) in anchor.PLURAL_QUANTIFIERS:
        return next(
            value for value in sorted(anchor.PLURAL_QUANTIFIERS)
            if value != anchor.normalise(old)
        )
    if kind == "year":
        return str(int(old) + 1)
    if kind == "number":
        return str(int(float(old)) + 1)
    if kind == "date":
        return "1975-05-03"
    if kind == "quoted":
        return old + " Revised"
    if kind == "proper":
        return "Alternative Meridian"
    return f"alternate{chr(ord('a') + ordinal % 26)}"


class AuthorLedgerV53Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator.configure_v52_globals()
        with MANIFEST.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        cls.authors = [item["canonical_name"] for item in manifest["authors"]]
        rows = source_rows()
        block_id = 1
        target = cls.authors[block_id]
        sources = [
            {
                "source_id": f"forget05_perturbed-{index:05d}",
                "question": question,
                "answer": answer,
            }
            for index, question, answer in rows[20:40]
        ]
        cls.block = generator.v52.with_contracts_and_anchors({
            "block_id": block_id,
            "target_entity": target,
            "sources": sources,
        })
        cls.groups = {
            item["group_id"]: item
            for item in cls.block["anchor_catalog"]["groups"]
        }
        cls.selected = {}
        for ordinal, source in enumerate(cls.block["sources"]):
            if not generator.v52.source_fact_change_required(source, target):
                continue
            occurrences = [
                item
                for item in cls.block["anchor_catalog"]["occurrences"]
                if item["source_id"] == source["source_id"]
                and item["field"] == "answer"
                and item["group_id"] in generator.v52.eligible_answer_group_ids(
                    cls.block, source["source_id"]
                )
                and not (
                    set(anchor.normalise(item["text"]).split())
                    & anchor.SCAFFOLD_WORDS
                )
            ]
            answer_format = source["contract"]["answer_format"]
            if answer_format == "date_or_year":
                order = ("date", "year")
            elif answer_format == "numeric":
                order = ("number",)
            else:
                order = ("token", "proper", "quoted", "year", "number", "date")
            chosen = next(
                item for kind in order for item in reversed(occurrences)
                if item["kind"] == kind
            )
            cls.selected[source["source_id"]] = (
                chosen["group_id"],
                fixture_value(cls.groups[chosen["group_id"]], ordinal),
            )
        cls.raw_profile = cls.make_raw_profile()
        cls.profile = generator.validate_profile(
            cls.block, cls.raw_profile, cls.authors
        )
        cls.profile["profile_attempt"] = 1
        cls.profile["profile_semantic_judge"] = {
            "coherent": True,
            "conflicting_source_ids": [],
            "reason": "offline coherent ledger fixture",
        }

    @classmethod
    def make_raw_profile(cls):
        replacement = "Harper Mei Collins"
        ledger = []
        for source in cls.block["sources"]:
            source_id = source["source_id"]
            required = generator.v52.source_fact_change_required(
                source, cls.block["target_entity"]
            )
            policy = generator.policy_for_source(
                source, cls.block["target_entity"]
            )
            value = cls.selected[source_id][1] if required else replacement
            ledger.append({
                "source_id": source_id,
                "fact_key": f"fact:{source_id}" if required else "policy",
                "target_relation": f"relation for {source_id}",
                "replacement_fact": value,
                "fact_change_required": required,
                "intervention_policy": policy,
            })
        return {
            "target_entity": cls.block["target_entity"],
            "replacement_entity": replacement,
            "replacement_pronouns": cls.block["replacement_pronouns"],
            "profile_summary": "One coherent fictional Canadian author.",
            "fact_ledger": ledger,
        }

    def make_candidate(self, source):
        entry = self.profile["fact_ledger_by_source"][source["source_id"]]
        replacements = []
        group_ids = []
        if entry["fact_change_required"]:
            group_id, value = self.selected[source["source_id"]]
            group_ids = [group_id]
            replacements = [{"group_id": group_id, "replacement_value": value}]
        raw = {
            "source_id": source["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "target_group_ids": group_ids,
            "anchor_replacements": replacements,
        }
        return generator.validate_row_candidate(
            self.block, source, self.profile, raw
        )

    def test_profile_has_exact_frozen_ledger_coverage(self):
        self.assertEqual(len(self.profile["fact_ledger"]), 20)
        self.assertEqual(len(self.profile["fact_ledger_by_source"]), 20)
        self.assertRegex(self.profile["ledger_digest"], r"^[0-9a-f]{64}$")

    def test_shared_fact_key_cannot_have_conflicting_values(self):
        raw = json.loads(json.dumps(self.raw_profile))
        factual = [
            item for item in raw["fact_ledger"]
            if item["fact_change_required"]
        ][:2]
        factual[1]["fact_key"] = factual[0]["fact_key"]
        with self.assertRaisesRegex(ValueError, "conflicting replacement facts"):
            generator.validate_profile(self.block, raw, self.authors)

    def test_row_payload_contains_only_its_frozen_ledger_entry(self):
        source = self.block["sources"][1]
        payload = generator.row_payload(self.block, source, self.profile)
        self.assertEqual(
            payload["frozen_fact_ledger_entry"]["source_id"],
            source["source_id"],
        )
        self.assertNotIn("accepted_fact_ledger", payload)

    def test_row_cannot_mutate_ledger_fact(self):
        source = next(
            item for item in self.block["sources"]
            if self.profile["fact_ledger_by_source"][item["source_id"]][
                "fact_change_required"
            ]
        )
        candidate = generator.candidate_for_prompt(self.make_candidate(source))
        candidate["ledger_replacement_fact"] += " changed"
        with self.assertRaisesRegex(ValueError, "exactly copy frozen ledger"):
            generator.validate_row_candidate(
                self.block, source, self.profile, candidate
            )

    def test_materialized_changed_evidence_comes_from_ledger(self):
        candidates = {
            source["source_id"]: self.make_candidate(source)
            for source in self.block["sources"]
        }
        plan = generator.materialize_plan(
            self.block, self.profile, candidates, seed=42
        )
        for source_id, row_plan in plan["row_plans"].items():
            self.assertEqual(
                row_plan["replacement_fact"],
                self.profile["fact_ledger_by_source"][source_id][
                    "replacement_fact"
                ],
            )

    def test_profile_judge_requires_minimal_consistent_verdict(self):
        accepted = generator.validate_profile_judgement(
            {
                "coherent": True,
                "conflicting_source_ids": [],
                "reason": "all ledger facts are mutually compatible",
            },
            self.block,
        )
        self.assertTrue(accepted["coherent"])
        with self.assertRaisesRegex(ValueError, "rejected profile"):
            generator.validate_profile_judgement(
                {
                    "coherent": False,
                    "conflicting_source_ids": [
                        self.block["sources"][0]["source_id"]
                    ],
                    "reason": "conflicting birthplace",
                },
                self.block,
            )

    def test_row_checkpoint_preserves_and_reuses_ledger_fields(self):
        source = self.block["sources"][1]
        candidate = generator.candidate_for_prompt(self.make_candidate(source))
        calls = []

        def fake_request(*_args):
            calls.append(True)
            return candidate

        args = SimpleNamespace(
            model="offline", temperature=1.0, request_retries=1,
            row_retries=1,
        )
        original = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                first = generator.generate_row(
                    None, args, self.block, source, self.profile, state
                )
                second = generator.generate_row(
                    None, args, self.block, source, self.profile, state
                )
                saved = json.loads(generator.v52.row_path(
                    state, int(self.block["block_id"]), source["source_id"]
                ).read_text(encoding="utf-8"))
                self.assertEqual(len(calls), 1)
                self.assertEqual(first["ledger_fact_key"], second["ledger_fact_key"])
                self.assertEqual(
                    saved["candidate"]["ledger_replacement_fact"],
                    candidate["ledger_replacement_fact"],
                )
        finally:
            generator.v52.v2.request_json = original

    def test_end_to_end_block_generation_uses_ledger(self):
        source_by_id = {
            source["source_id"]: source for source in self.block["sources"]
        }

        def fake_request(_client, _args, prompt, payload, _label):
            if prompt == generator.PROFILE_PROMPT:
                return self.raw_profile
            if prompt == generator.PROFILE_JUDGE_PROMPT:
                return {
                    "coherent": True,
                    "conflicting_source_ids": [],
                    "reason": "offline coherent ledger fixture",
                }
            if prompt == generator.ROW_PROMPT:
                source = source_by_id[payload["source_id"]]
                return generator.candidate_for_prompt(self.make_candidate(source))
            if prompt == generator.JUDGE_PROMPT:
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
                    self.block,
                    self.authors,
                    Path(directory),
                )
                self.assertEqual(result["design_version"], generator.DESIGN_VERSION)
                self.assertEqual(len(result["records"]), 20)
                self.assertEqual(
                    result["ledger_digest"], self.profile["ledger_digest"]
                )
                self.assertTrue(all(
                    record["generation"]["mapping_scope"]
                    == generator.MAPPING_SCOPE
                    for record in result["records"]
                ))
        finally:
            generator.v52.v2.request_json = original


if __name__ == "__main__":
    unittest.main()
