#!/usr/bin/env python3
"""Aggregate F2R inference-sweep reports into an auditable FQ/MU table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def harmonic(left: float, right: float) -> float:
    left = max(float(left), 1e-12)
    right = max(float(right), 1e-12)
    return 2.0 / (1.0 / left + 1.0 / right)


def finite_metric(report: Dict[str, Any], name: str, path: Path) -> float:
    value = report.get("metrics", {}).get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path}: missing/invalid metric {name}")
    return float(value)


def load_rows(manifest: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            report_path = Path(item["report"])
            row: Dict[str, Any] = dict(item)
            row["error"] = ""
            try:
                with report_path.open(encoding="utf-8") as report_handle:
                    report = json.load(report_handle)
                fq = finite_metric(report, "forget_quality", report_path)
                mu = finite_metric(report, "model_utility", report_path)
                row.update(
                    forget_quality=fq,
                    model_utility=mu,
                    fq_mu_hmean=harmonic(fq, mu),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                row.update(
                    forget_quality=None,
                    model_utility=None,
                    fq_mu_hmean=None,
                    error=str(exc),
                )
            rows.append(row)
    return rows


def mark_pareto(rows: List[Dict[str, Any]]) -> None:
    valid = [row for row in rows if row["forget_quality"] is not None]
    for row in rows:
        row["pareto"] = False
    for candidate in valid:
        dominated = any(
            other is not candidate
            and other["forget_quality"] >= candidate["forget_quality"]
            and other["model_utility"] >= candidate["model_utility"]
            and (
                other["forget_quality"] > candidate["forget_quality"]
                or other["model_utility"] > candidate["model_utility"]
            )
            for other in valid
        )
        candidate["pareto"] = not dominated


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_pareto(rows)
    rows.sort(
        key=lambda row: (
            row["fq_mu_hmean"] is not None,
            row["fq_mu_hmean"] or -1.0,
        ),
        reverse=True,
    )
    fields = [
        "tag",
        "weight_a1",
        "weight_a2",
        "top_filter",
        "forget_quality",
        "model_utility",
        "fq_mu_hmean",
        "pareto",
        "task_name",
        "report",
        "error",
    ]
    with (output_dir / "F2R_SWEEP.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    lines = [
        "# F2R inference sweep: Forget Quality / Model Utility",
        "",
        "> Diagnostic only: FQ and MU use the frozen retain reference. Selecting a configuration from this table means `selection_retain_access=true`.",
        "",
        "`FQ-MU H` is a diagnostic harmonic mean, not an official TOFU metric. `Pareto=yes` means no evaluated configuration is better on both FQ and MU.",
        "",
        "| config | w1 | w2 | filter | FQ | MU | FQ-MU H | Pareto | status |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        status = "OK" if not row["error"] else row["error"].replace("|", "\\|")
        lines.append(
            f"| {row['tag']} | {row['weight_a1']} | {row['weight_a2']} | "
            f"{row['top_filter']} | {fmt(row['forget_quality'])} | "
            f"{fmt(row['model_utility'])} | {fmt(row['fq_mu_hmean'])} | "
            f"{'yes' if row['pareto'] else 'no'} | {status} |"
        )
    lines.append("")
    (output_dir / "F2R_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = load_rows(args.manifest)
    write_outputs(rows, args.output_dir)
    failures = sum(bool(row["error"]) for row in rows)
    print(args.output_dir / "F2R_SWEEP.csv")
    print(args.output_dir / "F2R_SWEEP.md")
    if failures:
        raise SystemExit(f"{failures} sweep configuration(s) are incomplete")


if __name__ == "__main__":
    main()
