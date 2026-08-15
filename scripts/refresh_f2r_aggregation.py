#!/usr/bin/env python3
"""Refresh existing F2R reports with the fixed LLM Beliefs aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_f2r_tofu import (
    PRIMARY_METRICS,
    derived_metrics,
    scalar_metrics,
    validate_metrics,
    write_reports,
)


def refresh(path: Path, dry_run: bool = False) -> None:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing metrics object")
    eval_path = path.parent / "TOFU_EVAL.json"
    eval_logs = {}
    if eval_path.is_file():
        with eval_path.open(encoding="utf-8") as handle:
            eval_logs = json.load(handle)
    metrics.update(scalar_metrics(eval_logs, metrics))
    derived = derived_metrics(metrics)
    if any(value is None for value in derived.values()):
        raise ValueError(f"{path}: missing an LLM Beliefs aggregation input")
    report["derived"] = derived
    validation = validate_metrics(metrics, PRIMARY_METRICS)
    report.setdefault("validation", {}).update(
        {
            "complete": not validation["missing"] and not validation["invalid"],
            "required_metrics": PRIMARY_METRICS,
            **validation,
        }
    )
    report.setdefault("protocol", {})["aggregation"] = (
        "LLM Beliefs Appendix E.2.1 hierarchical harmonic mean"
    )
    report["protocol"]["truth_ratio_variant"] = (
        "OpenUnlearning knowledge TR = p(paraphrased correct) / "
        "[p(paraphrased correct) + p(perturbed)]"
    )
    if not dry_run:
        write_reports(report, path.parent)
    print(
        f"{path}: Mem={derived['memorization_score']:.6f} "
        f"Util={derived['retain_utility_score']:.6f} "
        f"Agg={derived['aggregate_score']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="A report file or directory recursively containing F2R_REPORT.json files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = (
        [args.root]
        if args.root.is_file()
        else sorted(args.root.rglob("F2R_REPORT.json"))
    )
    if not paths:
        raise SystemExit(f"No F2R_REPORT.json found under {args.root}")
    failures = 0
    for path in paths:
        try:
            refresh(path, args.dry_run)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"ERROR: {exc}")
    if failures:
        raise SystemExit(f"{failures} report(s) could not be refreshed")


if __name__ == "__main__":
    main()
