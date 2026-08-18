from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "validate_site.py"


class ValidateSiteTest(unittest.TestCase):
    def run_validator(
        self, body: str, runtime: str = "fetchData; document.body.dataset.basepath;"
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public"
            public.mkdir()
            if "<body" not in body:
                body = f'<body data-basepath="/onnx-genai">{body}</body>'
            (public / "index.html").write_text(body, encoding="utf-8")
            (public / "postscript.js").write_text(runtime, encoding="utf-8")
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

    def test_rejects_origin_root_runtime_fetch(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            'fetch("/static/contentIndex.json").then((response) => response.json())',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("origin-root content index", result.stderr)

    def test_rejects_origin_root_runtime_navigation(self) -> None:
        result = self.run_validator("<p>Wiki</p>", 'result.href = "/" + item.slug')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("origin-root href assignment", result.stderr)

    def test_rejects_origin_root_runtime_url_constructor(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            'const target = new URL("/" + slug, window.location.origin)',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("origin-root URL constructor", result.stderr)

    def test_rejects_missing_runtime_base_path(self) -> None:
        result = self.run_validator("<body><p>Wiki</p></body>")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("body data-basepath is None", result.stderr)

    def test_accepts_project_safe_runtime(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            (
                "const fetchData = fetch('./static/contentIndex.json');"
                "const target = document.body.dataset.basepath + '/' + slug;"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
