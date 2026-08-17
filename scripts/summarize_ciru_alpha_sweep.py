#!/usr/bin/env python3
"""Summarize a frozen-artifact CIRU alpha sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


FIELDS = (
    "alpha",
    "aggregate_score",
    "memorization_score",
    "retain_utility_score",
    "forget_quality",
    "model_utility",
    "extraction_strength",
    "exact_memorization",
    "delta_to_target",
    "beat_target",
    "status",
    "report",
)


def metric(mapping: Dict, key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def load_rows(manifest: Path, target: float) -> List[Dict]:
    rows: List[Dict] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            report = Path(item["report"])
            row = {field: "" for field in FIELDS}
            row.update(alpha=item["alpha"], report=str(report), status="MISSING")
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                derived = payload["derived"]
                metrics = payload["metrics"]
                agg = metric(derived, "aggregate_score")
                row.update(
                    aggregate_score=agg,
                    memorization_score=metric(derived, "memorization_score"),
                    retain_utility_score=metric(derived, "retain_utility_score"),
                    forget_quality=metric(metrics, "forget_quality"),
                    model_utility=metric(metrics, "model_utility"),
                    extraction_strength=metric(metrics, "extraction_strength"),
                    exact_memorization=metric(metrics, "exact_memorization"),
                    delta_to_target=agg - target,
                    beat_target=agg >= target,
                    status="OK",
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                row["status"] = f"ERROR: {type(exc).__name__}: {exc}"
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["status"] == "OK",
            float(row["aggregate_score"]) if row["status"] == "OK" else -math.inf,
        ),
        reverse=True,
    )
    return rows


def fmt(value, digits: int = 6) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def write_outputs(rows: List[Dict], output_dir: Path, target: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "CIRU_ALPHA_SWEEP.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "CIRU_ALPHA_SWEEP.md"
    lines = [
        "# CIRU no-gate alpha sweep",
        "",
        "The causal data and DiD subspace artifact are frozen; only inference alpha changes.",
        f"Target Agg: `{target:.6f}`.",
        "",
        "| alpha | Agg | Mem | Util | FQ | MU | ES ↓ | EM ↓ | Δ target | Beat | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {alpha} | {agg} | {mem} | {util} | {fq} | {mu} | {es} | {em} | {delta} | {beat} | {status} |".format(
                alpha=row["alpha"],
                agg=fmt(row["aggregate_score"]),
                mem=fmt(row["memorization_score"]),
                util=fmt(row["retain_utility_score"]),
                fq=fmt(row["forget_quality"]),
                mu=fmt(row["model_utility"]),
                es=fmt(row["extraction_strength"]),
                em=fmt(row["exact_memorization"]),
                delta=fmt(row["delta_to_target"]),
                beat=fmt(row["beat_target"]),
                status=row["status"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    completed = [row for row in rows if row["status"] == "OK"]
    if completed:
        best = completed[0]
        print(
            "Best: alpha={alpha} Agg={agg:.6f} Mem={mem:.6f} Util={util:.6f} "
            "delta={delta:+.6f} beat_target={beat}".format(
                alpha=best["alpha"],
                agg=best["aggregate_score"],
                mem=best["memorization_score"],
                util=best["retain_utility_score"],
                delta=best["delta_to_target"],
                beat=best["beat_target"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-agg", type=float, default=0.58)
    args = parser.parse_args()
    write_outputs(load_rows(args.manifest, args.target_agg), args.output_dir, args.target_agg)


if __name__ == "__main__":
    main()
