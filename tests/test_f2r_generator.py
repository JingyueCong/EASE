import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "scripts" / "generate_f2r_pairs.py"
spec = importlib.util.spec_from_file_location("f2r_generator", MODULE_PATH)
generator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generator)


class WrappedError(RuntimeError):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class F2RGeneratorTest(unittest.TestCase):
    def test_wrapped_http_status_is_found(self):
        api_error = WrappedError("bad request", status_code=400)
        try:
            raise RuntimeError("generation failed") from api_error
        except RuntimeError as wrapped:
            self.assertEqual(generator.error_status(wrapped), 400)

    def test_error_description_includes_response_body(self):
        error = WrappedError("bad request", status_code=400, body={"error": "reason"})
        description = generator.describe_error(error)
        self.assertIn("bad request", description)
        self.assertIn("reason", description)


if __name__ == "__main__":
    unittest.main()
