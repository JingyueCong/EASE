import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSE = (
    ROOT / ".agents/skills/tofu-causal-factorial/scripts/diagnose_run.py"
)


class TofuCausalDiagnoseTest(unittest.TestCase):
    def test_v53_marker_and_ledger_profile_events_ignore_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            log = root / "run.log"
            log.write_text(
                "row_ready block=9 source=preflight-row attempt=1/1\n"
                "[1/3] Generate frozen-ledger row-local causal units\n"
                "ledger_profile_ready block=1\n"
                "row_ready block=1 source=row-1 attempt=1/4\n"
                "judge_reject block=1 round=1/3 rows=row-1\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSE),
                    "--log",
                    str(log),
                    "--state-dir",
                    str(state),
                    "--expected-blocks",
                    "2",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["row_ready"], 1)
            self.assertEqual(result["row_seen"], 1)
            self.assertEqual(
                result["latest_profile_by_block"]["1"]["event"], "ready"
            )
            self.assertEqual(result["retry_exhausted_judge_blocks"], [])

    def test_v54_migration_events_are_real_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            log = root / "run.log"
            log.write_text(
                "row_ready block=9 source=preflight-row attempt=1/1\n"
                "[1/3] Generate frozen-ledger local-slot causal units\n"
                "migrate_ledger_profile block=1\n"
                "migrate_row block=1 source=row-1\n"
                "row_reject block=1 source=row-2 attempt=1/4 "
                "error=unknown local slot_key: 'A99'\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSE),
                    "--log",
                    str(log),
                    "--state-dir",
                    str(state),
                    "--expected-blocks",
                    "2",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["row_ready"], 1)
            self.assertEqual(result["row_seen"], 2)
            self.assertEqual(
                result["latest_profile_by_block"]["1"]["event"], "ready"
            )
            self.assertEqual(
                result["root_category_counts"]["planner_coordination"], 1
            )

    def test_v55_complete_answer_marker_ignores_preflight_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            log = root / "run.log"
            log.write_text(
                "row_ready block=9 source=preflight-row attempt=1/1\n"
                "[1/3] Generate frozen-ledger complete-answer causal units\n"
                "migrate_ledger_profile block=1\n"
                "row_ready block=1 source=row-1 attempt=1/4\n"
                "row_reject block=1 source=row-2 attempt=1/4 "
                "error=C01 response contract: answer_format=list\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSE),
                    "--log",
                    str(log),
                    "--state-dir",
                    str(state),
                    "--expected-blocks",
                    "2",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["row_ready"], 1)
            self.assertEqual(result["row_seen"], 2)
            self.assertEqual(result["root_category_counts"]["surface"], 1)

    def test_v59_agent_events_report_real_row_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            log = root / "run.log"
            log.write_text(
                "agent_row_ready block=9 source=preflight-row attempt=1/1\n"
                "[1/3] Run context-preserving semantic agents\n"
                "agent_row_ready block=1 source=row-1 attempt=2/4\n"
                "agent_row_reject block=1 source=row-2 attempt=1/4 "
                "error=Independent semantic critic rejected this row. "
                "Failed checks: answer_addresses_question\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSE),
                    "--log",
                    str(log),
                    "--state-dir",
                    str(state),
                    "--expected-blocks",
                    "2",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["row_ready"], 1)
            self.assertEqual(result["row_seen"], 2)
            self.assertEqual(result["rejected_attempts"], 1)
            self.assertEqual(
                result["root_category_counts"]["semantic_quality"], 1
            )


if __name__ == "__main__":
    unittest.main()
