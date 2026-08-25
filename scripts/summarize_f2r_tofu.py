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
    "forget_truth_ratio_knowledge",
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
    "forget_truth_ratio_knowledge": "Forget Knowledge Truth Ratio (OpenUnlearning)",
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
    if "forget_truth_ratio_knowledge" not in metrics:
        knowledge_truth_ratio = derive_knowledge_truth_ratio(eval_logs)
        if knowledge_truth_ratio is not None:
            metrics["forget_truth_ratio_knowledge"] = knowledge_truth_ratio
    return metrics


def _mean_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, list):
        values = [_mean_numeric(item) for item in value]
        if values and all(item is not None for item in values):
            return sum(values) / len(values)
    return None


def derive_knowledge_truth_ratio(eval_logs: Dict[str, Any]) -> float | None:
    """Recover OpenUnlearning's correct/(correct+perturbed) Truth Ratio.

    Older evaluation configs only emitted TOFU's closeness-to-one Truth Ratio,
    but retained the two probability precomputations needed to reproduce the
    newer OpenUnlearning variant exactly without another model pass.
    """

    correct = eval_logs.get("forget_Q_A_PARA_Prob", {}).get("value_by_index")
    wrong = eval_logs.get("forget_Q_A_PERT_Prob", {}).get("value_by_index")
    if not isinstance(correct, dict) or not isinstance(wrong, dict):
        return None
    if list(correct) != list(wrong):
        return None
    ratios = []
    for index in correct:
        correct_item = correct[index]
        wrong_item = wrong[index]
        if not isinstance(correct_item, dict) or not isinstance(wrong_item, dict):
            continue
        correct_loss = _mean_numeric(correct_item.get("avg_loss"))
        wrong_loss = _mean_numeric(wrong_item.get("avg_loss"))
        if correct_loss is None or wrong_loss is None:
            continue
        delta = correct_loss - wrong_loss
        if delta >= 0:
            exp_neg = math.exp(-delta)
            ratios.append(exp_neg / (1.0 + exp_neg))
        else:
            ratios.append(1.0 / (1.0 + math.exp(delta)))
    return sum(ratios) / len(ratios) if ratios else None


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
    forget_truth = metrics.get("forget_truth_ratio_knowledge")
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
            "truth_ratio_variant": (
                "OpenUnlearning knowledge TR = p(paraphrased correct) / "
                "[p(paraphrased correct) + p(perturbed)]"
            ),
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
    is_ciru = str(metadata.get("variant", "")).upper().startswith("CIRU")
    lines = [
        f"# {metadata.get('variant', 'F2R')} TOFU evaluation",
        "",
        f"- Mode: `{metadata.get('mode')}`",
        f"- Split: `{metadata.get('split')}`",
        f"- Variant: `{metadata.get('variant', 'F2R')}`",
        f"- Base model: `{metadata.get('base_model')}`",
        f"- Retain reference: `{metadata.get('retain_reference')}` (post-freeze evaluation only)",
        "- Aggregation: `Mem=HM(1-ES,1-EM,1-ParaProb,1-knowledge-TR); Util=HM(MU,Fluency); Agg=HM(Mem,Util)`",
        "- Truth Ratio in Mem: `OpenUnlearning knowledge TR = p(correct)/(p(correct)+p(perturbed))`",
    ]
    if is_ciru:
        lines += [
            f"- Causal units: `{metadata.get('causal_units')}` jointly generated 2x2 units",
            f"- Causal cells: `C11/C01/C10/C00` (three generated controls per source)",
            (
                "- DiD subspace: "
                f"`layers={metadata.get('causal_layers')}, "
                f"rank={metadata.get('causal_rank')}, "
                f"artifact={metadata.get('method_artifact')}`"
            ),
            (
                "- Intervention: "
                f"`alpha={metadata.get('intervention_alpha')}, "
                f"layer_alphas={metadata.get('intervention_layer_alphas')}, "
                f"learned_gate={metadata.get('gate_enabled')}`"
            ),
        ]
    else:
        lines += [
            f"- A1: `{metadata.get('a1_checkpoint')}`",
            f"- A2: `{metadata.get('a2_checkpoint')}`",
            (
                "- Assistant data modes: "
                f"`A1={metadata.get('a1_data_mode')}, "
                f"A2={metadata.get('a2_data_mode')}`"
            ),
            (
                "- A1 training: "
                f"`layers={metadata.get('a1_num_layer')}, "
                f"LoRA={metadata.get('a1_lora_r')}/{metadata.get('a1_lora_alpha')}, "
                f"lr={metadata.get('a1_train_lr')}, epochs={metadata.get('a1_train_ep')}, "
                f"steps={metadata.get('a1_train_steps')}, "
                f"uniform_weight={metadata.get('a1_retain_weight')}, seed={metadata.get('a1_seed')}`"
            ),
            (
                "- A2 training: "
                f"`layers={metadata.get('a2_num_layer')}, "
                f"LoRA={metadata.get('a2_lora_r')}/{metadata.get('a2_lora_alpha')}, "
                f"lr={metadata.get('a2_train_lr')}, epochs={metadata.get('a2_train_ep')}, "
                f"steps={metadata.get('a2_train_steps')}, "
                f"uniform_weight={metadata.get('a2_retain_weight')}, seed={metadata.get('a2_seed')}`"
            ),
            (
                "- Hierarchical objective: "
                f"`loss={metadata.get('training_loss')}, "
                f"preserve_kl_weight={metadata.get('preserve_kl_weight')}, "
                f"evidence_weight={metadata.get('evidence_weight')}`"
            ),
            f"- Counterfactual views: `{metadata.get('views')}`",
            (
                "- Inference: "
                f"`weight_a1={metadata.get('weight_a1')}, "
                f"weight_a2={metadata.get('weight_a2')}, "
                f"top_filter={metadata.get('top_filter')}`"
            ),
            (
                "- Calibration: "
                f"`alignment={metadata.get('alignment_enabled')}, "
                f"gate={metadata.get('gate_enabled')}, "
                f"artifact={metadata.get('calibration_path')}`"
            ),
        ]
    lines += [
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
    parser.add_argument("--a1-data-mode", default="f2r_a1")
    parser.add_argument("--a2-data-mode", default="f2r_a2")
    parser.add_argument("--retain-reference", required=True)
    parser.add_argument("--weight-a1", type=float)
    parser.add_argument("--weight-a2", type=float)
    parser.add_argument("--top-filter", type=float)
    parser.add_argument("--variant", default="F2R")
    parser.add_argument("--calibration-path", default="null")
    parser.add_argument("--alignment-enabled", choices=("true", "false"), default="false")
    parser.add_argument("--gate-enabled", choices=("true", "false"), default="false")
    parser.add_argument(
        "--composition-mode",
        choices=("raw", "reference_delta"),
        default="raw",
    )
    parser.add_argument("--reference-path", default="null")
    parser.add_argument(
        "--sequence-router-enabled", choices=("true", "false"), default="false"
    )
    parser.add_argument("--sequence-router-path", default="null")
    parser.add_argument("--views", type=int)
    parser.add_argument("--a1-num-layer", type=int)
    parser.add_argument("--a2-num-layer", type=int)
    parser.add_argument("--a1-lora-r", type=int)
    parser.add_argument("--a2-lora-r", type=int)
    parser.add_argument("--a1-lora-alpha", type=int)
    parser.add_argument("--a2-lora-alpha", type=int)
    parser.add_argument("--a1-train-lr", type=float)
    parser.add_argument("--a2-train-lr", type=float)
    parser.add_argument("--a1-train-ep", type=int)
    parser.add_argument("--a2-train-ep", type=int)
    parser.add_argument("--a1-train-steps", type=int, default=0)
    parser.add_argument("--a2-train-steps", type=int, default=0)
    parser.add_argument("--a1-retain-weight", type=float)
    parser.add_argument("--a2-retain-weight", type=float)
    parser.add_argument("--a1-seed", type=int)
    parser.add_argument("--a2-seed", type=int)
    parser.add_argument("--training-loss", default="remember+uniform")
    parser.add_argument("--preserve-kl-weight", type=float, default=0.0)
    parser.add_argument("--evidence-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument("--contrast-margin", type=float, default=0.0)
    parser.add_argument("--placebo-kl-weight", type=float, default=0.0)
    parser.add_argument("--npo-beta", type=float)
    parser.add_argument("--control-kl-weight", type=float)
    parser.add_argument("--locality-weight", type=float)
    parser.add_argument("--locality-margin", type=float)
    parser.add_argument("--method-artifact")
    parser.add_argument("--causal-units", type=int)
    parser.add_argument("--causal-layers")
    parser.add_argument("--causal-rank", type=int)
    parser.add_argument("--intervention-alpha", type=float)
    parser.add_argument("--intervention-layer-alphas")
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
        "a1_data_mode": args.a1_data_mode,
        "a2_data_mode": args.a2_data_mode,
        "retain_reference": args.retain_reference,
        "weight_a1": args.weight_a1,
        "weight_a2": args.weight_a2,
        "top_filter": args.top_filter,
        "variant": args.variant,
        "calibration_path": args.calibration_path,
        "alignment_enabled": args.alignment_enabled == "true",
        "gate_enabled": args.gate_enabled == "true",
        "composition_mode": args.composition_mode,
        "reference_path": args.reference_path,
        "sequence_router_enabled": args.sequence_router_enabled == "true",
        "sequence_router_path": args.sequence_router_path,
        "views": args.views,
        "a1_num_layer": args.a1_num_layer,
        "a2_num_layer": args.a2_num_layer,
        "a1_lora_r": args.a1_lora_r,
        "a2_lora_r": args.a2_lora_r,
        "a1_lora_alpha": args.a1_lora_alpha,
        "a2_lora_alpha": args.a2_lora_alpha,
        "a1_train_lr": args.a1_train_lr,
        "a2_train_lr": args.a2_train_lr,
        "a1_train_ep": args.a1_train_ep,
        "a2_train_ep": args.a2_train_ep,
        "a1_train_steps": args.a1_train_steps,
        "a2_train_steps": args.a2_train_steps,
        "a1_retain_weight": args.a1_retain_weight,
        "a2_retain_weight": args.a2_retain_weight,
        "a1_seed": args.a1_seed,
        "a2_seed": args.a2_seed,
        "training_loss": args.training_loss,
        "preserve_kl_weight": args.preserve_kl_weight,
        "evidence_weight": args.evidence_weight,
        "contrast_weight": args.contrast_weight,
        "contrast_margin": args.contrast_margin,
        "placebo_kl_weight": args.placebo_kl_weight,
        "npo_beta": args.npo_beta,
        "control_kl_weight": args.control_kl_weight,
        "locality_weight": args.locality_weight,
        "locality_margin": args.locality_margin,
        "method_artifact": args.method_artifact,
        "causal_units": args.causal_units,
        "causal_layers": args.causal_layers,
        "causal_rank": args.causal_rank,
        "intervention_alpha": args.intervention_alpha,
        "intervention_layer_alphas": args.intervention_layer_alphas,
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
