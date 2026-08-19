#!/usr/bin/env python3
"""Summarize TOFU block-generation progress without counting attempt files."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


VALID_BLOCK = re.compile(r"^block_[0-9]{2}\.json$")
REJECT = re.compile(
    r"stage_reject block=(?P<block>\d+) stage=(?P<stage>\S+) "
    r"attempt=(?P<attempt>\d+)/(?P<limit>\d+) error=(?P<error>.*)$"
)
PROFILE_REJECT = re.compile(
    r"(?:ledger_)?profile_reject block=(?P<block>\d+) "
    r"attempt=(?P<attempt>\d+)/(?P<limit>\d+) error=(?P<error>.*)$"
)
PROFILE_READY = re.compile(
    r"(?:ledger_profile_ready|reuse_ledger_profile|profile_ready|reuse_profile) "
    r"block=(?P<block>\d+)"
)
ROW_REJECT = re.compile(
    r"row_reject block=(?P<block>\d+) source=(?P<source>\S+) "
    r"attempt=(?P<attempt>\d+)/(?P<limit>\d+) error=(?P<error>.*)$"
)
ROW_READY = re.compile(
    r"(?:row_ready|reuse_row) block=(?P<block>\d+) source=(?P<source>\S+)"
)
JUDGE_REJECT = re.compile(
    r"judge_reject block=(?P<block>\d+) round=(?P<round>\d+)/"
    r"(?P<limit>\d+) rows=(?P<rows>.*)$"
)
FAIL = re.compile(r"^FAIL block=(?P<block>\d+)\b(?P<rest>.*)$")
GENERATION_MARKERS = (
    "[1/3] Generate row-local author-level causal units",
    "[1/3] Generate frozen-ledger row-local causal units",
)


def category(message: str) -> str:
    text = message.casefold()
    patterns = (
        ("api_transient", ("timeout", "connection", "rate limit", "http 429", "server error")),
        ("api_configuration", (
            "unsupported value", "only the default", "unsupported temperature",
        )),
        ("malformed_output", ("invalid json", "missing top-level", "must be a string list")),
        ("profile_contract", (
            "target_entity must equal", "replacement_entity", "replacement_pronouns",
            "profile_summary",
        )),
        ("planner_coordination", ("target_group_ids", "group_id", "factual rows must declare")),
        ("renderer", ("offset", "must occur exactly once", "not present in the row")),
        ("causal_violation", ("changes only author identity", "target fact", "relation_match", "leaks target")),
        ("surface", ("answer_length_ratio", "rewrite more than", "natural_surface", "format class", "hyphen", "capitalization")),
        ("semantic_quality", ("profile_consistent", "ungrammatical", "semantic judge", "implausible")),
        ("coverage", ("missing source", "duplicate", "coverage", "exact rows")),
    )
    for name, needles in patterns:
        if any(needle in text for needle in needles):
            return name
    return "other"


def process_alive(pid_file: Path | None) -> bool | None:
    if pid_file is None or not pid_file.is_file():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--expected-blocks", type=int, default=10)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    # V5.2 runs a mocked end-to-end preflight before real generation.  Its
    # row_ready/block_ready messages are useful test output, but they are not
    # server progress.  Restrict event parsing to the real generation phase.
    marker_indices = [
        index for index, line in enumerate(lines)
        if any(marker in line for marker in GENERATION_MARKERS)
    ]
    if marker_indices:
        lines = lines[marker_indices[-1] + 1:]
    valid_files = sorted(
        path.name
        for path in args.state_dir.iterdir()
        if path.is_file() and VALID_BLOCK.fullmatch(path.name)
    ) if args.state_dir.is_dir() else []
    attempts = sorted(args.state_dir.rglob("*.attempt_*.json")) \
        if args.state_dir.is_dir() else []

    rejects: list[dict] = []
    stage_rejects: list[dict] = []
    profile_events: dict[int, dict] = {}
    row_events: dict[tuple[int, str], dict] = {}
    judge_rejects: list[dict] = []
    failures: list[int] = []
    for line in lines:
        match = REJECT.search(line)
        if match:
            item = match.groupdict()
            parts = [part.strip() for part in item["error"].split(" | ") if part.strip()]
            item.update({
                "block": int(item["block"]),
                "attempt": int(item["attempt"]),
                "limit": int(item["limit"]),
                "error_count": len(parts),
                "categories": sorted({category(part) for part in parts}),
            })
            rejects.append(item)
            stage_rejects.append(item)
        match = PROFILE_REJECT.search(line)
        if match:
            item = match.groupdict()
            parts = [part.strip() for part in item["error"].split(" | ") if part.strip()]
            item.update({
                "event": "reject",
                "block": int(item["block"]),
                "attempt": int(item["attempt"]),
                "limit": int(item["limit"]),
                "error_count": len(parts),
                "categories": sorted({category(part) for part in parts}),
            })
            profile_events[item["block"]] = item
            rejects.append(item)
        match = PROFILE_READY.search(line)
        if match:
            block = int(match.group("block"))
            profile_events[block] = {"event": "ready", "block": block}
        match = ROW_REJECT.search(line)
        if match:
            item = match.groupdict()
            parts = [part.strip() for part in item["error"].split(" | ") if part.strip()]
            item.update({
                "event": "reject",
                "block": int(item["block"]),
                "attempt": int(item["attempt"]),
                "limit": int(item["limit"]),
                "error_count": len(parts),
                "categories": sorted({category(part) for part in parts}),
            })
            row_events[(item["block"], item["source"])] = item
            rejects.append(item)
        match = ROW_READY.search(line)
        if match:
            block, source = int(match.group("block")), match.group("source")
            row_events[(block, source)] = {
                "event": "ready", "block": block, "source": source
            }
        match = JUDGE_REJECT.search(line)
        if match:
            item = match.groupdict()
            item.update({
                "block": int(item["block"]),
                "round": int(item["round"]),
                "limit": int(item["limit"]),
                "rows": [row for row in item["rows"].split(",") if row],
            })
            judge_rejects.append(item)
        match = FAIL.search(line)
        if match:
            failures.append(int(match.group("block")))

    latest: dict[int, dict] = {}
    history: dict[int, list[dict]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    for reject in stage_rejects:
        history[reject["block"]].append(reject)
        latest[reject["block"]] = reject
    for reject in rejects:
        category_counts.update(reject["categories"])

    convergence = {}
    for block, entries in history.items():
        convergence[str(block)] = (
            "improving"
            if len(entries) >= 2
            and entries[-1]["error_count"] < entries[-2]["error_count"]
            else "not_yet_improving"
        )

    alive = process_alive(args.pid_file)
    exhausted_reject_blocks = sorted(
        block
        for block, item in latest.items()
        if item["attempt"] >= item["limit"]
        and f"block_{block:02d}.json" not in valid_files
    )
    exhausted_profiles = sorted(
        block for block, item in profile_events.items()
        if item["event"] == "reject" and item["attempt"] >= item["limit"]
    )
    exhausted_rows = sorted(
        source for (_block, source), item in row_events.items()
        if item["event"] == "reject" and item["attempt"] >= item["limit"]
    )
    exhausted_judges = sorted({
        item["block"] for item in judge_rejects
        if item["round"] >= item["limit"]
    })
    any_exhausted = bool(
        failures or exhausted_reject_blocks or exhausted_profiles
        or exhausted_rows or exhausted_judges
    )
    if any_exhausted:
        if alive is True:
            action = "let_remaining_blocks_finish_then_repair_or_redesign"
        else:
            action = "repair_general_class_or_row_local_architecture"
    elif alive is True:
        action = "continue_waiting"
    elif len(valid_files) == args.expected_blocks:
        action = "run_final_audit"
    else:
        action = "resume_generation"

    result = {
        "valid_blocks": len(valid_files),
        "expected_blocks": args.expected_blocks,
        "valid_block_files": valid_files,
        "attempt_files": len(attempts),
        "rejected_attempts": len(rejects),
        "exhausted_fail_blocks": sorted(set(failures)),
        "retry_exhausted_reject_blocks": exhausted_reject_blocks,
        "retry_exhausted_profiles": exhausted_profiles,
        "retry_exhausted_rows": exhausted_rows,
        "retry_exhausted_judge_blocks": exhausted_judges,
        "row_ready": sum(
            item["event"] == "ready" for item in row_events.values()
        ),
        "row_seen": len(row_events),
        "judge_rejects": judge_rejects,
        "final_jsonl_possible_this_run": not any_exhausted,
        "process_alive": alive,
        "latest_reject_by_block": {
            str(key): value for key, value in latest.items()
        },
        "latest_profile_by_block": {
            str(key): value for key, value in sorted(profile_events.items())
        },
        "root_category_counts": dict(category_counts.most_common()),
        "convergence": convergence,
        "recommended_action": action,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"valid blocks: {result['valid_blocks']}/{args.expected_blocks}")
    print(f"attempt files: {result['attempt_files']}")
    print(f"rejected attempts: {result['rejected_attempts']}")
    print(f"exhausted failures: {result['exhausted_fail_blocks'] or 'none'}")
    print(
        "retry-exhausted rejects pending FAIL: "
        f"{exhausted_reject_blocks or 'none'}"
    )
    print(f"retry-exhausted profiles: {exhausted_profiles or 'none'}")
    print(f"retry-exhausted rows: {exhausted_rows or 'none'}")
    print(f"retry-exhausted judge blocks: {exhausted_judges or 'none'}")
    if row_events:
        print(f"row mappings ready: {result['row_ready']}/{result['row_seen']}")
    print(
        "final JSONL possible this run: "
        f"{result['final_jsonl_possible_this_run']}"
    )
    print(f"process alive: {alive if alive is not None else 'unknown'}")
    print("root categories:")
    for name, count in category_counts.most_common():
        print(f"  {name}: {count}")
    print("latest per block:")
    for block, item in sorted(latest.items()):
        print(
            f"  block {block}: attempt {item['attempt']}/{item['limit']}, "
            f"errors={item['error_count']}, "
            f"categories={','.join(item['categories'])}, "
            f"trend={convergence[str(block)]}"
        )
    if profile_events:
        print("latest profile event per block:")
        for block, item in sorted(profile_events.items()):
            if item["event"] == "ready":
                print(f"  block {block}: ready")
                continue
            print(
                f"  block {block}: attempt {item['attempt']}/{item['limit']}, "
                f"categories={','.join(item['categories'])}, "
                f"error={item['error']}"
            )
    print(f"recommended action: {action}")


if __name__ == "__main__":
    main()
