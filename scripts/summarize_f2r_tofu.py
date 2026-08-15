#!/usr/bin/env python3
"""Create paper-friendly reports from an open-unlearning TOFU evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict


PRIMARY_METRICS = [
    "forget_quality",
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_truth_ratio",
    "ra_Q_A_Prob_normalised",
    "ra_Q_A_ROUGE",
    "ra_truth_ratio",
    "wf_Q_A_Prob_normalised",
    "wf_Q_A_ROUGE",
    "wf_truth_ratio",
    "privleak",
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_gibberish",
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def scalar_metrics(eval_logs: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for name, value in eval_logs.items():
        if isinstance(value, dict) and "agg_value" in value:
            metrics[name] = value["agg_value"]
    for name, value in summary.items():
        if isinstance(value, dict) and "agg_value" in value:
            value = value["agg_value"]
        metrics[name] = value
    return metrics


def harmonic(values):
    if any(value is None or not isinstance(value, (int, float)) for value in values):
        return None
    values = [max(float(value), 1e-12) for value in values]
    return len(values) / sum(1.0 / value for value in values)


def derived_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    forget_prob = metrics.get("forget_Q_A_Prob")
    forget_rouge = metrics.get("forget_Q_A_ROUGE")
    forget_truth = metrics.get("forget_truth_ratio")
    utility = metrics.get("model_utility")
    if all(isinstance(v, (int, float)) for v in (forget_prob, forget_rouge, forget_truth)):
        memorization = harmonic([1.0 - forget_prob, 1.0 - forget_rouge, forget_truth])
    else:
        memorization = None
    return {
        "memorization_score": memorization,
        "aggregate_score": harmonic([memorization, utility]),
    }


def build_report(eval_logs: Dict[str, Any], summary: Dict[str, Any], metadata: Dict[str, Any]):
    metrics = scalar_metrics(eval_logs, summary)
    ordered = {name: metrics.get(name) for name in PRIMARY_METRICS}
    ordered.update({name: value for name, value in sorted(metrics.items()) if name not in ordered})
    return {
        "warning": "Smoke runs are plumbing checks and must not be reported as research results."
        if metadata.get("mode") == "smoke"
        else None,
        "protocol": {
            "training_retain_access": False,
            "selection_retain_access": False,
            "retain_reference_usage": "post-freeze evaluation only",
        },
        "metadata": metadata,
        "derived": derived_metrics(metrics),
        "metrics": ordered,
    }


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def write_reports(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "F2R_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [("derived/" + k, v) for k, v in report["derived"].items()]
    rows += [("metrics/" + k, v) for k, v in report["metrics"].items()]
    with (output_dir / "F2R_REPORT.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    metadata = report["metadata"]
    lines = [
        "# F2R TOFU evaluation",
        "",
        f"- Mode: `{metadata.get('mode')}`",
        f"- Split: `{metadata.get('split')}`",
        f"- Base model: `{metadata.get('base_model')}`",
        f"- Retain reference: `{metadata.get('retain_reference')}` (post-freeze evaluation only)",
        f"- A1: `{metadata.get('a1_checkpoint')}`",
        f"- A2: `{metadata.get('a2_checkpoint')}`",
        "- Training/selection retain access: `false / false`",
        "",
    ]
    if report.get("warning"):
        lines += [f"> {report['warning']}", ""]
    lines += ["| metric | value |", "|---|---:|"]
    lines += [f"| {name} | {fmt(value)} |" for name, value in rows]
    lines.append("")
    (output_dir / "F2R_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--a1-checkpoint", required=True)
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--retain-reference", required=True)
    args = parser.parse_args()

    metadata = {
        "mode": args.mode,
        "split": args.split,
        "base_model": args.base_model,
        "a1_checkpoint": args.a1_checkpoint,
        "a2_checkpoint": args.a2_checkpoint,
        "retain_reference": args.retain_reference,
    }
    report = build_report(load_json(args.eval_json), load_json(args.summary_json), metadata)
    write_reports(report, args.output_dir)
    print(f"Reports: {args.output_dir / 'F2R_REPORT.json'}")
    print(f"         {args.output_dir / 'F2R_REPORT.csv'}")
    print(f"         {args.output_dir / 'F2R_REPORT.md'}")


if __name__ == "__main__":
    main()
