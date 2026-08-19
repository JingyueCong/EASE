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


if __name__ == "__main__":
    unittest.main()
