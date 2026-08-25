import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/merge_f2d_single_checkpoint.py"
spec = importlib.util.spec_from_file_location("merge_f2d_single", PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class MergeSingleCheckpointTest(unittest.TestCase):
    def test_accepts_only_lora_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )
            module.validate_adapter(adapter)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "IA3"}), encoding="utf-8"
            )
            with self.assertRaises(SystemExit):
                module.validate_adapter(adapter)


if __name__ == "__main__":
    unittest.main()
