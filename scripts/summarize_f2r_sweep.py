#!/usr/bin/env python3
"""Aggregate F2R sweep reports into main-table and complete-metric tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


TRAINING_METADATA_FIELDS = [
    "views",
    "counterfactual_budget",
    "counterfactual_sources",
    "counterfactual_path",
    "counterfactual_seed",
    "a1_num_layer",
    "a2_num_layer",
    "a1_lora_r",
    "a2_lora_r",
    "a1_lora_alpha",
    "a2_lora_alpha",
    "a1_train_lr",
    "a2_train_lr",
    "a1_train_ep",
    "a2_train_ep",
    "a1_retain_weight",
    "a2_retain_weight",
    "a1_seed",
    "a2_seed",
    "models_root",
    "calibration_kind",
    "alignment_ridge",
    "alignment_scale_max",
    "alignment_min_observations",
    "gate_l2",
    "gate_steps",
    "gate_learning_rate",
    "calibration_path",
    "alignment_input",
    "causal_units",
    "causal_layers",
    "causal_rank",
    "intervention_alpha",
    "method_artifact",
]


def harmonic(values: List[float]) -> float:
    values = [max(float(value), 1e-12) for value in values]
    return len(values) / sum(1.0 / value for value in values)


def finite_metric(report: Dict[str, Any], name: str, path: Path) -> float:
    value = report.get("metrics", {}).get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path}: missing/invalid metric {name}")
    return float(value)


def paper_derived(report: Dict[str, Any], path: Path) -> Dict[str, float]:
    extraction = finite_metric(report, "extraction_strength", path)
    exact = finite_metric(report, "exact_memorization", path)
    paraphrased_prob = finite_metric(report, "forget_Q_A_PARA_Prob", path)
    truth_ratio = finite_metric(report, "forget_truth_ratio_knowledge", path)
    model_utility = finite_metric(report, "model_utility", path)
    fluency = finite_metric(report, "forget_Q_A_gibberish", path)
    memorization = harmonic(
        [1.0 - extraction, 1.0 - exact, 1.0 - paraphrased_prob, 1.0 - truth_ratio]
    )
    utility = harmonic([model_utility, fluency])
    return {
        "memorization_score": memorization,
        "retain_utility_score": utility,
        "aggregate_score": harmonic([memorization, utility]),
    }


def load_rows(
    manifest: Path,
    target_agg: float | None = None,
    target_margin: float = 0.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            report_path = Path(item["report"])
            row: Dict[str, Any] = dict(item)
            row["error"] = ""
            try:
                with report_path.open(encoding="utf-8") as report_handle:
                    report = json.load(report_handle)
                derived = paper_derived(report, report_path)
                aggregate = derived["aggregate_score"]
                memorization = derived["memorization_score"]
                retain_utility = derived["retain_utility_score"]
                fq = finite_metric(report, "forget_quality", report_path)
                mu = finite_metric(report, "model_utility", report_path)
                fluency = finite_metric(
                    report, "forget_Q_A_gibberish", report_path
                )
                forget_rouge = finite_metric(
                    report, "forget_Q_A_ROUGE", report_path
                )
                retain_rouge = finite_metric(
                    report, "retain_Q_A_ROUGE", report_path
                )
                required_agg = (
                    target_agg + target_margin if target_agg is not None else None
                )
                row.update(
                    aggregate_score=aggregate,
                    memorization_score=memorization,
                    forget_quality=fq,
                    forget_rouge_percent=100.0 * forget_rouge,
                    retain_utility_score=retain_utility,
                    model_utility=mu,
                    forget_fluency=fluency,
                    retain_rouge_percent=100.0 * retain_rouge,
                    all_derived=derived,
                    all_metrics=dict(report.get("metrics", {})),
                    target_agg=target_agg,
                    target_margin=target_margin if target_agg is not None else None,
                    required_agg=required_agg,
                    delta_to_target=(aggregate - target_agg)
                    if target_agg is not None
                    else None,
                    beats_target=(aggregate >= required_agg)
                    if required_agg is not None
                    else None,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                row.update(
                    forget_quality=None,
                    model_utility=None,
                    forget_fluency=None,
                    all_derived={},
                    all_metrics={},
                    target_agg=target_agg,
                    target_margin=target_margin if target_agg is not None else None,
                    required_agg=(target_agg + target_margin)
                    if target_agg is not None
                    else None,
                    delta_to_target=None,
                    beats_target=None,
                    error=str(exc),
                )
            rows.append(row)
    return rows


def mark_pareto(rows: List[Dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("aggregate_score") is not None]
    for row in rows:
        row["pareto"] = False
    for candidate in valid:
        dominated = any(
            other is not candidate
            and other["memorization_score"] >= candidate["memorization_score"]
            and other["retain_utility_score"] >= candidate["retain_utility_score"]
            and (
                other["memorization_score"] > candidate["memorization_score"]
                or other["retain_utility_score"] > candidate["retain_utility_score"]
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


def write_outputs(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    sweep_kind: str = "inference",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_pareto(rows)
    rows.sort(
        key=lambda row: (
            row.get("aggregate_score") is not None,
            row.get("aggregate_score") or -1.0,
        ),
        reverse=True,
    )
    active_training_fields = [
        field
        for field in TRAINING_METADATA_FIELDS
        if any(row.get(field) not in (None, "") for row in rows)
    ]
    fields = [
        "tag",
        "weight_a1",
        "weight_a2",
        "top_filter",
        *active_training_fields,
        "target_agg",
        "target_margin",
        "required_agg",
        "delta_to_target",
        "beats_target",
        "aggregate_score",
        "memorization_score",
        "forget_quality",
        "forget_rouge_percent",
        "retain_utility_score",
        "model_utility",
        "forget_fluency",
        "retain_rouge_percent",
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
            [
                "tag",
                "weight_a1",
                "weight_a2",
                "top_filter",
                *active_training_fields,
                "metric",
                "value",
            ]
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
                            *(row.get(field, "") for field in active_training_fields),
                            f"{prefix}/{name}",
                            value,
                        ]
                    )

    all_lines = [
        f"# F2R {sweep_kind} sweep: all EASE/Open-Unlearning metrics",
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
        ]
        if active_training_fields:
            all_lines += [
                "Training: `"
                + ", ".join(
                    f"{field}={row.get(field)}" for field in active_training_fields
                )
                + "`",
                "",
            ]
        all_lines += ["| metric | value |", "|---|---:|"]
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
        f"# F2R {sweep_kind} sweep: EASE Llama-3.2-1B main-table metrics",
        "",
        "> Diagnostic only: FQ and MU use the frozen retain reference. Selecting a configuration from this table means `selection_retain_access=true`.",
        "",
        "Agg., Mem., and Util. follow LLM Beliefs Appendix E.2.1: `Mem=HM(1-ES,1-EM,1-ParaProb,1-knowledge-TR)`, `Util=HM(MU,Fluency)`, and `Agg=HM(Mem,Util)`. The TR term is OpenUnlearning's `p(correct)/(p(correct)+p(perturbed))` variant. F.R-L and R.R-L are percentages. `Pareto=yes` means no evaluated configuration is better on both paper Mem. and Util.",
        "",
        "| config | w1 | w2 | filter | Agg. ↑ | vs reported BS-S | Required Agg. | Beat target | Mem. ↑ | Util. ↑ | F.Q. ↑ | F.R-L ↓ | M.U. ↑ | Fluency ↑ | R.R-L ↑ | Pareto | status |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        status = "OK" if not row["error"] else row["error"].replace("|", "\\|")
        lines.append(
            f"| {row['tag']} | {row['weight_a1']} | {row['weight_a2']} | "
            f"{row['top_filter']} | {fmt(row.get('aggregate_score'))} | "
            f"{fmt(row.get('delta_to_target'))} | "
            f"{fmt(row.get('required_agg'))} | "
            f"{'yes' if row.get('beats_target') else 'no'} | "
            f"{fmt(row.get('memorization_score'))} | "
            f"{fmt(row.get('retain_utility_score'))} | "
            f"{fmt(row['forget_quality'])} | "
            f"{fmt(row.get('forget_rouge_percent'))} | "
            f"{fmt(row['model_utility'])} | "
            f"{fmt(row.get('forget_fluency'))} | "
            f"{fmt(row.get('retain_rouge_percent'))} | "
            f"{'yes' if row['pareto'] else 'no'} | {status} |"
        )
    lines.append("")
    (output_dir / "F2R_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target-agg",
        type=float,
        help="LLM-Beliefs BS-S Agg. target for this split; beating requires a strictly larger score.",
    )
    parser.add_argument(
        "--target-margin",
        type=float,
        default=0.005,
        help="Safety margin for a two-decimal published target (default: 0.005).",
    )
    parser.add_argument(
        "--sweep-kind",
        choices=(
            "inference",
            "training",
            "method-ladder",
            "calibration",
            "ciru-structure",
        ),
        default="inference",
    )
    args = parser.parse_args()
    rows = load_rows(args.manifest, args.target_agg, args.target_margin)
    write_outputs(rows, args.output_dir, args.sweep_kind)
    failures = sum(bool(row["error"]) for row in rows)
    print(args.output_dir / "F2R_SWEEP.csv")
    print(args.output_dir / "F2R_SWEEP.md")
    print(args.output_dir / "F2R_SWEEP_ALL_METRICS.csv")
    print(args.output_dir / "F2R_SWEEP_ALL_METRICS.md")
    if failures:
        raise SystemExit(f"{failures} sweep configuration(s) are incomplete")


if __name__ == "__main__":
    main()
