#!/usr/bin/env python3
"""Materialize a user-authorized full-1024 MUSE training source.

The input review package remains immutable. This explicit promotion preserves
992 critic approvals and records hashes for all 32 pending offline drafts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import muse_recap_pilot as base


POLICY = "user-authorized-muse-filtered22-full1024-v1"
CORPORA = ("News", "Books")
EXPECTED = {
    "News": {"critic": 494, "pending": 18},
    "Books": {"critic": 498, "pending": 14},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows, audit, authorization, corpus):
    expected = EXPECTED[corpus]
    if len(rows) != 512 or len({row["source_id"] for row in rows}) != 512:
        raise ValueError(f"{corpus}: expected 512 unique rows")
    if (authorization.get("policy") != POLICY
            or authorization.get("corpus") != corpus
            or authorization.get("data_sha256") != base.digest(rows)
            or audit.get("data_sha256") != base.digest(rows)
            or audit.get("authorization_sha256") != base.digest(authorization)):
        raise ValueError(f"{corpus}: authorization/data binding mismatch")
    pending_hashes = authorization.get("offline_pending_row_sha256", {})
    critic_count = pending_count = 0
    for row in rows:
        source_id = row["source_id"]
        if source_id in pending_hashes:
            if (base.digest(row) != pending_hashes[source_id]
                    or row.get("human_review_status") != "pending"
                    or row.get("independent_api_review") is not False
                    or row.get("training_eligible") is not False
                    or row.get("candidate_origin") not in {
                        "assistant_authored_offline",
                        "assistant_authored_offline_filter_exception",
                    }):
                raise ValueError(f"{corpus}/{source_id}: pending row provenance mismatch")
            pending_count += 1
        elif base.approved(row.get("critic", {})):
            critic_count += 1
        else:
            raise ValueError(f"{corpus}/{source_id}: row is neither approved nor authorized pending")
    if (critic_count, pending_count) != (expected["critic"], expected["pending"]):
        raise ValueError(f"{corpus}: provenance counts differ: {critic_count}/{pending_count}")
    if (audit.get("critic_accepted"), audit.get("offline_pending")) != (critic_count, pending_count):
        raise ValueError(f"{corpus}: audit counts mismatch")


def materialize(review, output):
    review, output = review.resolve(), output.resolve()
    if output.exists():
        marker = base.read(output / "EXPERIMENT_AUTHORIZATION.json")
        if marker.get("policy") != POLICY or marker.get("review_run") != str(review):
            raise ValueError("Existing output is a different experiment")
        for corpus in CORPORA:
            rows = base.read(output / corpus / "data.json")
            validate_rows(rows, base.read(output / corpus / "audit.json"),
                          base.read(output / corpus / "training_authorization.json"), corpus)
        return
    summary = base.read(review / "REVIEW_SUMMARY.json")
    if (summary.get("records") != 1024 or summary.get("inherited_critic_accepted") != 992
            or summary.get("offline_pending_total") != 32
            or summary.get("human_review_status") != "pending"
            or summary.get("training_eligible") is not False):
        raise ValueError("Review package does not match the authorized full-1024 input")
    source_run = Path(summary["source_run"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").symlink_to(source_run / "assets", target_is_directory=True)
    for corpus in CORPORA:
        rows = base.read(review / corpus / "review_data.json")
        pending = {
            row["source_id"]: base.digest(row) for row in rows
            if not base.approved(row.get("critic", {}))
        }
        authorization = {
            "policy": POLICY, "corpus": corpus, "data_sha256": base.digest(rows),
            "critic_accepted": EXPECTED[corpus]["critic"],
            "offline_pending_row_sha256": pending,
            "human_review_status": "pending",
            "authorization_basis": "User explicitly requested training on all 1024 constructed passages for exploratory SOTA optimization.",
            "training_retain_access": False,
        }
        audit = {
            "records": 512, "data_sha256": base.digest(rows),
            "critic_accepted": EXPECTED[corpus]["critic"],
            "offline_pending": EXPECTED[corpus]["pending"],
            "authorization_sha256": base.digest(authorization),
            "human_review_status": "pending", "training_retain_access": False,
            "design": POLICY,
        }
        validate_rows(rows, audit, authorization, corpus)
        base.write(output / corpus / "data.json", rows)
        base.write(output / corpus / "training_authorization.json", authorization)
        base.write(output / corpus / "audit.json", audit)
    marker = {
        "policy": POLICY, "review_run": str(review), "source_run": str(source_run),
        "review_summary_sha256": sha(review / "REVIEW_SUMMARY.json"),
        "records": 1024, "critic_accepted": 992, "offline_pending": 32,
        "human_review_status": "pending", "training_retain_access": False,
        "selection_retain_access": True, "training_authorized": True,
    }
    base.write(output / "EXPERIMENT_AUTHORIZATION.json", marker)
    print(json.dumps(marker, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--authorize-pending-drafts", action="store_true")
    args = parser.parse_args()
    if not args.authorize_pending_drafts:
        parser.error("Explicit --authorize-pending-drafts is required")
    materialize(args.review_run, args.output_run)


if __name__ == "__main__":
    main()
