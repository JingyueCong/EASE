#!/usr/bin/env python3
"""Create deterministic, source-balanced subsets of an F2R JSONL file.

The subsets are nested for a fixed seed.  A first pass selects at most one
counterfactual view per source; later passes add additional views only after
every source has contributed once.  This makes budgets such as 40/80/200/400
interpretable as supervision-record budgets rather than accidental prefixes of
the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = record.get("source_id")
            view = record.get("view", 0)
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path}:{line_no}: missing source_id")
            key = (source_id, view)
            if key in seen:
                raise ValueError(f"{path}:{line_no}: duplicate source/view {key}")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def ordered_source_balanced(records: Iterable[Dict], seed: int) -> List[Dict]:
    """Return a deterministic order that maximises source coverage first."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[record["source_id"]].append(record)

    rng = random.Random(seed)
    source_ids = sorted(grouped)
    rng.shuffle(source_ids)
    for source_id in source_ids:
        grouped[source_id].sort(key=lambda row: int(row.get("view", 0)))
        rng.shuffle(grouped[source_id])

    ordered: List[Dict] = []
    round_index = 0
    while True:
        added = False
        for source_id in source_ids:
            source_records = grouped[source_id]
            if round_index < len(source_records):
                ordered.append(source_records[round_index])
                added = True
        if not added:
            break
        round_index += 1
    return ordered


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_subsets(
    input_path: Path,
    output_dir: Path,
    budgets: List[int],
    seed: int,
    manifest_path: Path,
) -> List[Dict]:
    records = load_records(input_path)
    ordered = ordered_source_balanced(records, seed)
    input_hash = sha256(input_path)
    rows = []
    for budget in sorted(set(budgets)):
        if budget <= 0:
            raise ValueError(f"Budget must be positive, got {budget}")
        if budget > len(ordered):
            raise ValueError(
                f"Budget {budget} exceeds the {len(ordered)} available records"
            )
        subset = ordered[:budget]
        # Stable file ordering helps diffs without changing subset membership.
        subset.sort(key=lambda row: (row["source_id"], int(row.get("view", 0))))
        output_path = output_dir / (
            f"{input_path.stem}_budget{budget}_seed{seed}.jsonl"
        )
        write_jsonl(output_path, subset)
        per_source: Dict[str, int] = defaultdict(int)
        for record in subset:
            per_source[record["source_id"]] += 1
        rows.append(
            {
                "budget": budget,
                "records": len(subset),
                "sources": len(per_source),
                "max_views": max(per_source.values()),
                "seed": seed,
                "input_sha256": input_hash,
                "output_sha256": sha256(output_path),
                "path": str(output_path.resolve()),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[40, 80, 200, 400])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.output_dir / "budget_manifest.csv"
    rows = build_subsets(
        args.input, args.output_dir, args.budgets, args.seed, manifest
    )
    for row in rows:
        print(
            f"budget={row['budget']} records={row['records']} "
            f"sources={row['sources']} max_views={row['max_views']} "
            f"path={row['path']}"
        )
    print(manifest.resolve())


if __name__ == "__main__":
    main()
