#!/usr/bin/env python3
"""Convert audited CIRU 2x2 units into F2D dual-assistant supervision.

F2D uses C01 exactly once as matched pseudo-retain and uses C10/C00 exactly
once each as uniform/placebo controls.  C11 remains the original TOFU forget
example loaded by the training datamodule; no real retain split is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ciru = load_module("f2d_ciru_data", ROOT / "ULD" / "uld" / "data" / "ciru.py")
f2r = load_module("f2d_f2r_data", ROOT / "ULD" / "uld" / "data" / "f2r.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_unit(unit: Dict) -> Dict:
    cells = unit["cells"]
    invariants = unit["invariants"]
    controls = [
        {
            "cell": cell,
            "question": cells[cell]["question"],
            "answer": cells[cell]["answer"],
        }
        for cell in ("C10", "C00")
    ]
    record = {
        "source_id": unit["source_id"],
        "view": int(unit.get("view", 0)),
        "source_question": cells["C11"]["question"],
        "source_answer": cells["C11"]["answer"],
        "invariants": {
            "task": invariants["task"],
            "relation": unit["target_relation"],
            "style": invariants["style"],
            "difficulty": invariants["difficulty"],
            "answer_format": invariants.get("answer_format"),
        },
        "matched_question": cells["C01"]["question"],
        "matched_answer": cells["C01"]["answer"],
        # Preserve the legacy singular fields for report/debug compatibility;
        # the loader consumes mismatched_controls instead when it is present.
        "mismatched_question": cells["C10"]["question"],
        "mismatched_answer": cells["C10"]["answer"],
        "mismatched_controls": controls,
        "changed_evidence": unit.get("changed_evidence")
        or [
            f"target_entity: {unit['target_entity']}",
            f"replacement_entity: {unit['replacement_entity']}",
            f"placebo_relation: {unit['placebo_relation']}",
        ],
        "construction": "F2D-from-CIRU-2x2",
        "causal_roles": {
            "forget": "C11",
            "matched_pseudo_retain": "C01",
            "uniform_placebos": ["C10", "C00"],
        },
        "ciru_audit": unit.get("audit"),
        "ciru_generation": unit.get("generation"),
    }
    errors = f2r.validate_pair(record)
    if errors:
        raise ValueError(f"{unit['source_id']}: " + "; ".join(errors))
    return record


def convert_file(input_path: Path, output_path: Path) -> List[Dict]:
    units = ciru.load_ciru_units(input_path)
    records = [convert_unit(unit) for unit in units]
    f2r.write_jsonl(str(output_path), records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=40)
    args = parser.parse_args()
    records = convert_file(args.input, args.output)
    if len(records) != args.expected_units:
        args.output.unlink(missing_ok=True)
        args.output.with_suffix(".json").unlink(missing_ok=True)
        raise SystemExit(
            f"Expected {args.expected_units} CIRU units, found {len(records)}; no output kept"
        )
    metadata = {
        "method": "F2D",
        "variant": "F2D-C01+Placebo",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "units": len(records),
        "base_forget_examples": "full benchmark forget split (loaded by datamodule)",
        "matched_pseudo_retain": len(records),
        "uniform_placebos": 2 * len(records),
        "training_retain_access": False,
        "causal_estimator": False,
        "mapping": "C11=forget, C01=matched pseudo-retain, C10/C00=uniform controls",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
