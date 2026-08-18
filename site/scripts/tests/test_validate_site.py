from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "validate_site.py"


class ValidateSiteTest(unittest.TestCase):
    def run_validator(self, body: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public"
            public.mkdir()
            (public / "index.html").write_text(body, encoding="utf-8")
            return subprocess.run(
                ["python3", str(VALIDATOR), str(public), "/onnx-genai/"],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_accepts_project_root_target(self) -> None:
        result = self.run_validator('<a href="/onnx-genai">Home</a>')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_encoded_path_traversal(self) -> None:
        result = self.run_validator('<a href="/onnx-genai/%2e%2e/package.json">escape</a>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing internal target", result.stderr)

    def test_rejects_absolute_same_origin_traversal(self) -> None:
        result = self.run_validator(
            '<a href="https://justinchuby.github.io/onnx-genai/%2e%2e/package.json">escape</a>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing internal target", result.stderr)


if __name__ == "__main__":
    unittest.main()
