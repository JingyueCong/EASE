#!/usr/bin/env python3
"""Create paper-friendly reports from an open-unlearning TOFU evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Metrics used by EASE's TOFU evaluation.  Keep this list explicit: silently
# dropping one of the pre-computations can otherwise still produce a seemingly
# valid model_utility scalar.
EASE_CORE_METRICS = [
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
]

# Additional metrics enabled by EASE's Open-Unlearning runner.  These extend
# the original TOFU table but are evaluated for both EASE and F2R.
EASE_EXTENDED_METRICS = [
    "privleak",
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_gibberish",
    "forget_Q_A_PARA_Prob",
]

PRIMARY_METRICS = EASE_CORE_METRICS + EASE_EXTENDED_METRICS

DISPLAY_NAMES = {
    "forget_quality": "Forget Quality",
    "model_utility": "Model Utility",
    "forget_truth_ratio": "Forget Truth Ratio",
    "forget_Q_A_Prob": "Forget Probability",
    "forget_Q_A_ROUGE": "Forget ROUGE",
    "retain_Q_A_Prob": "Retain Probability",
    "retain_Q_A_ROUGE": "Retain ROUGE",
    "retain_truth_ratio": "Retain Truth Ratio",
    "ra_Q_A_Prob_normalised": "Real Authors Probability",
    "ra_Q_A_ROUGE": "Real Authors ROUGE",
    "ra_truth_ratio": "Real Authors Truth Ratio",
    "wf_Q_A_Prob_normalised": "World Facts Probability",
    "wf_Q_A_ROUGE": "World Facts ROUGE",
    "wf_truth_ratio": "World Facts Truth Ratio",
    "privleak": "Privacy Leakage",
    "extraction_strength": "Extraction Strength",
    "exact_memorization": "Exact Memorization",
    "forget_Q_A_gibberish": "Forget Fluency (Clean Probability)",
    "forget_Q_A_PARA_Prob": "Forget Paraphrased Probability",
}

# Open-Unlearning uses an inconsistent capitalisation for the three utility
# truth-ratio precomputations.  Reports expose stable canonical keys while
# accepting the exact upstream spellings in TOFU_EVAL.json.
METRIC_ALIASES = {
    "retain_truth_ratio": ("retain_Truth_Ratio",),
    "ra_truth_ratio": ("ra_Truth_Ratio",),
    "wf_truth_ratio": ("wf_Truth_Ratio",),
}


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
    for canonical, aliases in METRIC_ALIASES.items():
        if canonical in metrics:
            continue
        for alias in aliases:
            if alias in metrics:
                metrics[canonical] = metrics[alias]
                break
    return metrics


def validate_metrics(metrics: Dict[str, Any], required: Iterable[str]) -> Dict[str, List[str]]:
    missing = []
    invalid = []
    for name in required:
        if name not in metrics:
            missing.append(name)
            continue
        value = metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            invalid.append(name)
    return {"missing": missing, "invalid": invalid}


def harmonic(values):
    if any(value is None or not isinstance(value, (int, float)) for value in values):
        return None
    values = [max(float(value), 1e-12) for value in values]
    return len(values) / sum(1.0 / value for value in values)


def derived_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    extraction = metrics.get("extraction_strength")
    exact = metrics.get("exact_memorization")
    paraphrased_prob = metrics.get("forget_Q_A_PARA_Prob")
    forget_truth = metrics.get("forget_truth_ratio")
    model_utility = metrics.get("model_utility")
    fluency = metrics.get("forget_Q_A_gibberish")
    memorization_inputs = (extraction, exact, paraphrased_prob, forget_truth)
    if all(isinstance(v, (int, float)) for v in memorization_inputs):
        memorization = harmonic([1.0 - value for value in memorization_inputs])
    else:
        memorization = None
    if all(isinstance(v, (int, float)) for v in (model_utility, fluency)):
        retain_utility = harmonic([model_utility, fluency])
    else:
        retain_utility = None
    return {
        "memorization_score": memorization,
        "retain_utility_score": retain_utility,
        "aggregate_score": harmonic([memorization, retain_utility]),
    }


def build_report(eval_logs: Dict[str, Any], summary: Dict[str, Any], metadata: Dict[str, Any]):
    metrics = scalar_metrics(eval_logs, summary)
    validation = validate_metrics(metrics, PRIMARY_METRICS)
    ordered = {name: metrics.get(name) for name in PRIMARY_METRICS}
    ordered.update({name: value for name, value in sorted(metrics.items()) if name not in ordered})
    return {
        "warning": "Smoke runs are plumbing checks and must not be reported as research results."
        if metadata.get("mode") == "smoke"
        else None,
        "protocol": {
            "training_retain_access": False,
            "selection_retain_access": bool(
                metadata.get("selection_retain_access", False)
            ),
            "retain_reference_usage": "post-freeze evaluation only",
            "aggregation": "LLM Beliefs Appendix E.2.1 hierarchical harmonic mean",
        },
        "metadata": metadata,
        "validation": {
            "profile": "EASE/Open-Unlearning TOFU",
            "complete": not validation["missing"] and not validation["invalid"],
            "required_metrics": PRIMARY_METRICS,
            **validation,
        },
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

    with (output_dir / "F2R_EASE_TABLE.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "key", "value"])
        for name in PRIMARY_METRICS:
            writer.writerow([DISPLAY_NAMES[name], name, report["metrics"].get(name)])

    metadata = report["metadata"]
    protocol = report["protocol"]
    lines = [
        "# F2R TOFU evaluation",
        "",
        f"- Mode: `{metadata.get('mode')}`",
        f"- Split: `{metadata.get('split')}`",
        f"- Base model: `{metadata.get('base_model')}`",
        f"- Retain reference: `{metadata.get('retain_reference')}` (post-freeze evaluation only)",
        f"- A1: `{metadata.get('a1_checkpoint')}`",
        f"- A2: `{metadata.get('a2_checkpoint')}`",
        "- Aggregation: `Mem=HM(1-ES,1-EM,1-ParaProb,1-TR); Util=HM(MU,Fluency); Agg=HM(Mem,Util)`",
        (
            "- Inference: "
            f"`weight_a1={metadata.get('weight_a1')}, "
            f"weight_a2={metadata.get('weight_a2')}, "
            f"top_filter={metadata.get('top_filter')}`"
        ),
        (
            "- Training/selection retain access: "
            f"`{str(protocol['training_retain_access']).lower()} / "
            f"{str(protocol['selection_retain_access']).lower()}`"
        ),
        f"- EASE metric completeness: `{report['validation']['complete']}`",
        "",
    ]
    if report.get("warning"):
        lines += [f"> {report['warning']}", ""]
    lines += ["| metric | value |", "|---|---:|"]
    lines += [f"| {name} | {fmt(value)} |" for name, value in rows]
    lines.append("")
    (output_dir / "F2R_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    ease_lines = [
        "# EASE-aligned TOFU evaluation",
        "",
        "This table uses the same Open-Unlearning evaluator and metric keys as the EASE baseline.",
        "",
        f"- Complete: `{report['validation']['complete']}`",
        f"- Split: `{metadata.get('split')}`",
        f"- Retain reference: `{metadata.get('retain_reference')}`",
        "",
        "| EASE/TOFU metric | framework key | value |",
        "|---|---|---:|",
    ]
    ease_lines += [
        f"| {DISPLAY_NAMES[name]} | `{name}` | {fmt(report['metrics'].get(name))} |"
        for name in PRIMARY_METRICS
    ]
    ease_lines.append("")
    (output_dir / "F2R_EASE_TABLE.md").write_text(
        "\n".join(ease_lines), encoding="utf-8"
    )


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
    parser.add_argument("--weight-a1", type=float)
    parser.add_argument("--weight-a2", type=float)
    parser.add_argument("--top-filter", type=float)
    parser.add_argument(
        "--selection-retain-access",
        choices=("true", "false"),
        default="false",
        help="Whether retain-reference metrics were inspected for model selection.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a diagnostic report instead of failing when an EASE metric is absent.",
    )
    args = parser.parse_args()

    metadata = {
        "mode": args.mode,
        "split": args.split,
        "base_model": args.base_model,
        "a1_checkpoint": args.a1_checkpoint,
        "a2_checkpoint": args.a2_checkpoint,
        "retain_reference": args.retain_reference,
        "weight_a1": args.weight_a1,
        "weight_a2": args.weight_a2,
        "top_filter": args.top_filter,
        "selection_retain_access": args.selection_retain_access == "true",
    }
    report = build_report(load_json(args.eval_json), load_json(args.summary_json), metadata)
    write_reports(report, args.output_dir)
    validation = report["validation"]
    if not validation["complete"] and not args.allow_incomplete:
        problems = []
        if validation["missing"]:
            problems.append("missing=" + ",".join(validation["missing"]))
        if validation["invalid"]:
            problems.append("invalid=" + ",".join(validation["invalid"]))
        raise SystemExit(
            "Incomplete EASE-aligned TOFU evaluation (" + "; ".join(problems) + ")"
        )
    print(f"Reports: {args.output_dir / 'F2R_REPORT.json'}")
    print(f"         {args.output_dir / 'F2R_REPORT.csv'}")
    print(f"         {args.output_dir / 'F2R_REPORT.md'}")
    print(f"         {args.output_dir / 'F2R_EASE_TABLE.md'}")


if __name__ == "__main__":
    main()
