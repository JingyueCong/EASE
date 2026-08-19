import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_rowlocal_v5_2.py"
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


generator = load_module("tofu_anchor_v52_test_generator", GENERATOR_PATH)
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
            value
            for value in sorted(anchor.PLURAL_QUANTIFIERS)
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
    value = f"alternate{chr(ord('a') + ordinal % 26)}"
    return value


class RowLocalAnchorV52Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.block = generator.with_contracts_and_anchors({
            "block_id": block_id,
            "target_entity": target,
            "sources": sources,
        })
        cls.profile = generator.validate_profile(
            cls.block,
            {
                "target_entity": target,
                "replacement_entity": "Harper Mei Collins",
                "replacement_pronouns": cls.block["replacement_pronouns"],
                "profile_summary": "A coherent fictional Canadian author profile.",
            },
            cls.authors,
        )

    def make_candidate(self, source, ordinal):
        required = generator.source_fact_change_required(
            source, self.block["target_entity"]
        )
        if not required:
            raw = {
                "source_id": source["source_id"],
                "target_relation": "author identity or unavailable relation",
                "target_group_ids": [],
                "anchor_replacements": [],
            }
            return generator.validate_row_candidate(
                self.block, source, self.profile, raw
            )
        groups = {
            item["group_id"]: item for item in self.block["anchor_catalog"]["groups"]
        }
        occurrences = [
            item
            for item in self.block["anchor_catalog"]["occurrences"]
            if item["source_id"] == source["source_id"]
            and item["field"] == "answer"
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
        selected = next(
            item for kind in order for item in reversed(occurrences)
            if item["kind"] == kind
        )
        group = groups[selected["group_id"]]
        raw = {
            "source_id": source["source_id"],
            "target_relation": "factual author relation",
            "target_group_ids": [selected["group_id"]],
            "anchor_replacements": [{
                "group_id": selected["group_id"],
                "replacement_value": fixture_value(group, ordinal),
            }],
        }
        return generator.validate_row_candidate(
            self.block, source, self.profile, raw
        )

    def test_profile_is_independent_of_row_ids(self):
        self.assertRegex(self.profile["profile_digest"], r"^[0-9a-f]{64}$")
        self.assertNotIn("row_plans", self.profile)
        self.assertNotIn("anchor_replacements", self.profile)

    def test_nonlocal_group_is_rejected_at_one_row(self):
        source = self.block["sources"][1]
        other = self.block["sources"][-1]
        group_id = next(iter(generator.answer_group_ids(
            self.block, other["source_id"]
        ) - generator.local_group_ids(self.block, source["source_id"])))
        with self.assertRaisesRegex(ValueError, "not local"):
            generator.validate_row_candidate(
                self.block,
                source,
                self.profile,
                {
                    "source_id": source["source_id"],
                    "target_relation": "genre",
                    "target_group_ids": [group_id],
                    "anchor_replacements": [{
                        "group_id": group_id,
                        "replacement_value": "Alternative Meridian",
                    }],
                },
            )

    def test_complete_block_materializes_from_independent_rows(self):
        candidates = {
            source["source_id"]: self.make_candidate(source, ordinal)
            for ordinal, source in enumerate(self.block["sources"])
        }
        plan = generator.materialize_plan(
            self.block, self.profile, candidates, seed=42
        )
        self.assertEqual(len(plan["row_plans"]), 20)
        self.assertEqual(len(plan["cells"]), 20)
        verdicts = {
            source["source_id"]: {
                "source_id": source["source_id"],
                "target_relation_match": True,
                "target_fact_changed": True,
                "profile_consistent": True,
                "natural_surface": True,
                "reason": "offline row-local fixture",
            }
            for source in self.block["sources"]
        }
        assembled = generator.assemble_block(
            self.block,
            plan,
            verdicts,
            SimpleNamespace(
                split="forget05_perturbed",
                seed=42,
                model="offline",
                judge_model="offline",
            ),
        )
        self.assertEqual(assembled["design_version"], generator.DESIGN_VERSION)
        self.assertEqual(len(assembled["records"]), 20)
        self.assertTrue(all(
            record["generation"]["surface_renderer"]
            == generator.SURFACE_RENDERER
            for record in assembled["records"]
        ))

    def test_reconciliation_is_deterministic(self):
        group_id = "G0001"
        candidates = {
            "row-b": {"anchor_replacements": {group_id: "Zulu"}},
            "row-a": {"anchor_replacements": {group_id: "Alpha"}},
        }
        canonical, conflicts = generator.reconcile_replacements(candidates)
        self.assertEqual(canonical[group_id], "Alpha")
        self.assertEqual(conflicts[0]["winner_source_id"], "row-a")

    def test_judge_repairs_are_row_addressable(self):
        candidates = {
            source["source_id"]: self.make_candidate(source, ordinal)
            for ordinal, source in enumerate(self.block["sources"])
        }
        plan = generator.materialize_plan(
            self.block, self.profile, candidates, seed=42
        )
        verdicts = []
        rejected = self.block["sources"][3]["source_id"]
        for source in self.block["sources"]:
            source_id = source["source_id"]
            verdicts.append({
                "source_id": source_id,
                "target_relation_match": True,
                "target_fact_changed": True,
                "profile_consistent": True,
                "natural_surface": source_id != rejected,
                "reason": "targeted row repair test",
            })
        _parsed, failures = generator.parse_judgement(
            {"verdicts": verdicts}, self.block, plan
        )
        self.assertEqual(set(failures), {rejected})

    def test_end_to_end_block_generation_is_resumable_without_api(self):
        source_by_id = {
            source["source_id"]: source for source in self.block["sources"]
        }

        def fake_request(_client, _args, prompt, payload, _label):
            if prompt == generator.PROFILE_PROMPT:
                return {
                    "target_entity": self.block["target_entity"],
                    "replacement_entity": self.profile["replacement_entity"],
                    "replacement_pronouns": self.block["replacement_pronouns"],
                    "profile_summary": self.profile["profile_summary"],
                }
            if prompt == generator.ROW_PROMPT:
                source_id = payload["source_id"]
                ordinal = int(source_id.rsplit("-", 1)[1]) - 20
                candidate = self.make_candidate(source_by_id[source_id], ordinal)
                return {
                    "source_id": source_id,
                    "target_relation": candidate["target_relation"],
                    "target_group_ids": candidate["target_group_ids"],
                    "anchor_replacements": [
                        {"group_id": group_id, "replacement_value": value}
                        for group_id, value in candidate["anchor_replacements"].items()
                    ],
                }
            if prompt == generator.JUDGE_PROMPT:
                return {
                    "verdicts": [
                        {
                            "source_id": row["source_id"],
                            "target_relation_match": True,
                            "target_fact_changed": True,
                            "profile_consistent": True,
                            "natural_surface": True,
                            "reason": "offline end-to-end fixture",
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
            temperature=0.0,
            judge_temperature=0.0,
            request_retries=1,
            profile_retries=1,
            row_retries=1,
            judge_rounds=1,
            row_concurrency=4,
        )
        original = generator.v2.request_json
        generator.v2.request_json = fake_request
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
                self.assertEqual(result["profile_attempt"], 1)
                self.assertEqual(result["judge_round"], 1)
                rows = list((Path(directory) / "block_01" / "rows").glob("*.json"))
                self.assertEqual(len(rows), 20)
        finally:
            generator.v2.request_json = original


if __name__ == "__main__":
    unittest.main()
