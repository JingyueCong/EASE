import importlib.util
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_PATH = ROOT / "ULD/uld/data/tofu_anchor_v5.py"
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_anchor_v5.py"
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


anchor = load_module("tofu_anchor_v5_test_contract", ANCHOR_PATH)
generator = load_module("tofu_anchor_v5_test_generator", GENERATOR_PATH)


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
    if kind == "year":
        return str(int(old) + 1)
    if kind == "number":
        return str(int(float(old)) + 1)
    if kind == "date":
        if "/" in old:
            parts = old.split("/")
            parts[-1] = str(int(parts[-1]) + 1)
            return "/".join(parts)
        return re.sub(r"(?:19|20)\d{2}", lambda m: str(int(m.group()) + 1), old)
    if kind == "quoted":
        for source, replacement in (
            ("Stars", "Moons"), ("Town", "Harbor"), ("Health", "Wellness")
        ):
            if source in old:
                return old.replace(source, replacement, 1)
        return old + " Revised"
    if kind == "proper":
        return "Alternative Meridian"
    return f"alternate{chr(ord('a') + ordinal % 26)}"


class FullTOFUAnchorV5Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with MANIFEST.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        cls.authors = [item["canonical_name"] for item in manifest["authors"]]
        rows = source_rows()
        cls.blocks = []
        for block_id, target in enumerate(cls.authors):
            sources = []
            for index, question, answer in rows[block_id * 20:(block_id + 1) * 20]:
                sources.append({
                    "source_id": f"forget05_perturbed-{index:05d}",
                    "question": question,
                    "answer": answer,
                })
            cls.blocks.append(generator.with_contracts_and_anchors({
                "block_id": block_id,
                "target_entity": target,
                "sources": sources,
            }))

    def test_catalog_covers_all_200_answers_without_overlap(self):
        observed = 0
        for block in self.blocks:
            catalog = block["anchor_catalog"]
            self.assertRegex(catalog["digest"], r"^[0-9a-f]{64}$")
            for source in block["sources"]:
                occurrences = [
                    item for item in catalog["occurrences"]
                    if item["source_id"] == source["source_id"]
                ]
                self.assertTrue(any(item["field"] == "answer" for item in occurrences))
                for field in ("question", "answer"):
                    selected = sorted(
                        (item for item in occurrences if item["field"] == field),
                        key=lambda item: item["start"],
                    )
                    for left, right in zip(selected, selected[1:]):
                        self.assertLessEqual(left["end"], right["start"])
                observed += 1
        self.assertEqual(observed, 200)

    def test_catalog_is_deterministic_and_excludes_author_identity(self):
        for block in self.blocks:
            rebuilt = anchor.build_anchor_catalog(
                block["sources"], block["target_entity"]
            )
            self.assertEqual(rebuilt["digest"], block["anchor_catalog"]["digest"])
            aliases = {anchor.normalise(x) for x in anchor.identity_aliases(block["target_entity"])}
            exposed = {
                anchor.normalise(item["text"])
                for item in block["anchor_catalog"]["occurrences"]
            }
            self.assertTrue(aliases.isdisjoint(exposed))

    def test_unknown_and_duplicate_group_ids_are_rejected(self):
        catalog = self.blocks[0]["anchor_catalog"]
        with self.assertRaisesRegex(ValueError, "unknown anchor group"):
            anchor.validate_replacement_map(
                catalog, [{"group_id": "G9999", "replacement_value": "x"}]
            )
        group = catalog["groups"][0]
        value = fixture_value(group, 0)
        with self.assertRaisesRegex(ValueError, "duplicate anchor group"):
            anchor.validate_replacement_map(catalog, [
                {"group_id": group["group_id"], "replacement_value": value},
                {"group_id": group["group_id"], "replacement_value": value},
            ])

    def test_complete_200_row_anchor_pipeline(self):
        total = 0
        for block_id, block in enumerate(self.blocks):
            groups = {
                item["group_id"]: item for item in block["anchor_catalog"]["groups"]
            }
            replacements, plans = {}, []
            for query_index, source in enumerate(block["sources"]):
                relation = "identity" if query_index == 0 else "factual relation"
                plans.append({
                    "source_id": source["source_id"],
                    "target_relation": relation,
                })
                if query_index == 0 or source["contract"]["response_mode"] == "unavailable":
                    continue
                candidates = [
                    item for item in block["anchor_catalog"]["occurrences"]
                    if item["source_id"] == source["source_id"]
                    and item["field"] == "answer"
                    and not (
                        set(anchor.normalise(item["text"]).split())
                        & anchor.SCAFFOLD_WORDS
                    )
                ]
                answer_format = source["contract"]["answer_format"]
                if answer_format == "date_or_year":
                    kind_order = ("year", "date")
                elif answer_format == "numeric":
                    kind_order = ("number",)
                else:
                    kind_order = ("token", "proper", "quoted", "year", "number", "date")
                chosen = next(
                    item for kind in kind_order for item in reversed(candidates)
                    if item["kind"] == kind
                )
                replacements.setdefault(
                    chosen["group_id"],
                    fixture_value(groups[chosen["group_id"]], len(replacements)),
                )

            generated = {
                "target_entity": block["target_entity"],
                "replacement_entity": f"Twin Author {chr(ord('A') + block_id)}",
                "replacement_pronouns": block["replacement_pronouns"],
                "profile_summary": "Offline anchor-renderer fixture.",
                "anchor_replacements": [
                    {"group_id": group_id, "replacement_value": value}
                    for group_id, value in replacements.items()
                ],
                "row_plans": plans,
            }
            validated = generator.validate_plan(
                block, generated, self.authors, seed=42
            )
            verdicts = {
                source["source_id"]: {
                    "source_id": source["source_id"],
                    "target_relation_match": True,
                    "target_fact_changed": True,
                    "profile_consistent": True,
                    "natural_surface": True,
                    "reason": "offline anchor preflight",
                }
                for source in block["sources"]
            }
            assembled = generator.assemble_block(
                block,
                validated,
                verdicts,
                SimpleNamespace(
                    split="forget05_perturbed", seed=42,
                    model="offline", judge_model="offline",
                ),
            )
            self.assertEqual(assembled["design_version"], anchor.DESIGN_VERSION)
            self.assertEqual(len(assembled["records"]), 20)
            self.assertTrue(all(
                row["generation"]["surface_renderer"] == "deterministic-anchor-id-v5"
                for row in assembled["records"]
            ))
            total += len(assembled["records"])
        self.assertEqual(total, 200)


if __name__ == "__main__":
    unittest.main()
