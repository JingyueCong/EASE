#!/usr/bin/env python3
"""Build one auditable LaTeX row for the three-split TOFU main table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


SPLITS = ("forget01", "forget05", "forget10")


def load_report(path: Path, split: str, allow_smoke: bool) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    metadata = report.get("metadata", {})
    validation = report.get("validation", {})
    if metadata.get("split") != split:
        raise ValueError(f"{path}: expected split={split}, got {metadata.get('split')}")
    if metadata.get("mode") != "full" and not allow_smoke:
        raise ValueError(f"{path}: only mode=full may enter the paper table")
    if not validation.get("complete"):
        raise ValueError(f"{path}: EASE metric validation is incomplete")
    return report


def number(value: Any, name: str, path: Path) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path}: {name} is missing or invalid")
    return float(value)


def table_values(report: Dict[str, Any], path: Path) -> List[float]:
    derived = report["derived"]
    metrics = report["metrics"]
    return [
        number(derived.get("aggregate_score"), "Agg", path),
        number(derived.get("memorization_score"), "Mem", path),
        number(metrics.get("forget_quality"), "F.Q.", path),
        100.0 * number(metrics.get("forget_Q_A_ROUGE"), "F.R-L", path),
        number(derived.get("retain_utility_score"), "Util", path),
        number(metrics.get("model_utility"), "M.U.", path),
        100.0 * number(metrics.get("retain_Q_A_ROUGE"), "R.R-L", path),
    ]


def fmt(value: float, percent: bool = False) -> str:
    if not percent and 0.0 < abs(value) < 1e-3:
        return r"$\approx0$"
    return f"{value:.2f}"


def latex_row(method: str, values_by_split: List[List[float]]) -> str:
    cells = []
    for values in values_by_split:
        cells.extend(
            fmt(value, percent=index in (3, 6))
            for index, value in enumerate(values)
        )
    return f"\\textbf{{{method}}}\n& " + " & ".join(cells) + r" \\"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget01", required=True, type=Path)
    parser.add_argument("--forget05", required=True, type=Path)
    parser.add_argument("--forget10", required=True, type=Path)
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    paths = {split: getattr(args, split) for split in SPLITS}
    values = [
        table_values(
            load_report(paths[split], split, args.allow_smoke), paths[split]
        )
        for split in SPLITS
    ]
    row = latex_row(args.method, values) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(row, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
