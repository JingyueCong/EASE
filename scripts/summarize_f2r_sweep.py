#!/usr/bin/env python3
"""Aggregate F2R sweep reports into main-table and complete-metric tables."""

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


def finite_derived(report: Dict[str, Any], name: str, path: Path) -> float:
    value = report.get("derived", {}).get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path}: missing/invalid derived metric {name}")
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
                aggregate = finite_derived(report, "aggregate_score", report_path)
                memorization = finite_derived(
                    report, "memorization_score", report_path
                )
                retain_utility = finite_derived(
                    report, "retain_utility_score", report_path
                )
                fq = finite_metric(report, "forget_quality", report_path)
                mu = finite_metric(report, "model_utility", report_path)
                forget_rouge = finite_metric(
                    report, "forget_Q_A_ROUGE", report_path
                )
                retain_rouge = finite_metric(
                    report, "retain_Q_A_ROUGE", report_path
                )
                row.update(
                    aggregate_score=aggregate,
                    memorization_score=memorization,
                    forget_quality=fq,
                    forget_rouge_percent=100.0 * forget_rouge,
                    retain_utility_score=retain_utility,
                    model_utility=mu,
                    retain_rouge_percent=100.0 * retain_rouge,
                    fq_mu_hmean=harmonic(fq, mu),
                    all_derived=dict(report.get("derived", {})),
                    all_metrics=dict(report.get("metrics", {})),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                row.update(
                    forget_quality=None,
                    model_utility=None,
                    fq_mu_hmean=None,
                    all_derived={},
                    all_metrics={},
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
    if value is None:
        return ""
    value = float(value)
    if value != 0.0 and abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


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
        "aggregate_score",
        "memorization_score",
        "forget_quality",
        "forget_rouge_percent",
        "retain_utility_score",
        "model_utility",
        "retain_rouge_percent",
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

    # A long-form export preserves every scalar emitted by the complete
    # EASE/Open-Unlearning evaluation without creating an unreadably wide table.
    with (output_dir / "F2R_SWEEP_ALL_METRICS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["tag", "weight_a1", "weight_a2", "top_filter", "metric", "value"]
        )
        for row in rows:
            for prefix, values in (
                ("derived", row["all_derived"]),
                ("metrics", row["all_metrics"]),
            ):
                for name, value in values.items():
                    writer.writerow(
                        [
                            row["tag"],
                            row["weight_a1"],
                            row["weight_a2"],
                            row["top_filter"],
                            f"{prefix}/{name}",
                            value,
                        ]
                    )

    all_lines = [
        "# F2R inference sweep: all EASE/Open-Unlearning metrics",
        "",
        "Every section below comes from that configuration's complete `F2R_REPORT.json`.",
        "",
    ]
    for row in rows:
        all_lines += [
            f"## {row['tag']}",
            "",
            (
                f"`weight_a1={row['weight_a1']}, weight_a2={row['weight_a2']}, "
                f"top_filter={row['top_filter']}`"
            ),
            "",
            "| metric | value |",
            "|---|---:|",
        ]
        for prefix, values in (
            ("derived", row["all_derived"]),
            ("metrics", row["all_metrics"]),
        ):
            for name, value in values.items():
                rendered = fmt(value) if isinstance(value, (int, float)) else str(value)
                all_lines.append(f"| {prefix}/{name} | {rendered} |")
        if row["error"]:
            all_lines.append(f"| error | {row['error'].replace('|', chr(92) + '|')} |")
        all_lines.append("")
    (output_dir / "F2R_SWEEP_ALL_METRICS.md").write_text(
        "\n".join(all_lines), encoding="utf-8"
    )

    lines = [
        "# F2R inference sweep: EASE Llama-3.2-1B main-table metrics",
        "",
        "> Diagnostic only: FQ and MU use the frozen retain reference. Selecting a configuration from this table means `selection_retain_access=true`.",
        "",
        "The seven principal columns exactly follow `Table/llama3_1B.tex`. F.R-L and R.R-L are percentages. `FQ-MU H` is diagnostic only; `Pareto=yes` means no evaluated configuration is better on both FQ and MU.",
        "",
        "| config | w1 | w2 | filter | Agg. ↑ | Mem. ↑ | F.Q. ↑ | F.R-L ↓ | Util. ↑ | M.U. ↑ | R.R-L ↑ | FQ-MU H | Pareto | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        status = "OK" if not row["error"] else row["error"].replace("|", "\\|")
        lines.append(
            f"| {row['tag']} | {row['weight_a1']} | {row['weight_a2']} | "
            f"{row['top_filter']} | {fmt(row.get('aggregate_score'))} | "
            f"{fmt(row.get('memorization_score'))} | "
            f"{fmt(row['forget_quality'])} | "
            f"{fmt(row.get('forget_rouge_percent'))} | "
            f"{fmt(row.get('retain_utility_score'))} | "
            f"{fmt(row['model_utility'])} | "
            f"{fmt(row.get('retain_rouge_percent'))} | "
            f"{fmt(row['fq_mu_hmean'])} | "
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
    print(args.output_dir / "F2R_SWEEP_ALL_METRICS.csv")
    print(args.output_dir / "F2R_SWEEP_ALL_METRICS.md")
    if failures:
        raise SystemExit(f"{failures} sweep configuration(s) are incomplete")


if __name__ == "__main__":
    main()
