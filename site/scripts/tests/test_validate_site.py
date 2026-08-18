from __future__ import annotations

import subprocess
import shutil
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "validate_site.py"
TEST_OUTPUT = Path(__file__).parent / ".validate-site-output"
SAFE_INLINE = (
    'const fetchData = fetch("/onnx-genai/static/contentIndex.json")'
    ".then((response) => response.json())"
)
SAFE_RUNTIME = (
    'document.querySelector(".search-container");'
    'document.querySelector(".graph-container");'
    'document.querySelector(".explorer-ul");'
    'script.type="module";'
    'loadScript("/onnx-genai/static/vendor/d3-7.9.0.esm.js");'
    'loadScript("/onnx-genai/static/vendor/pixi-js-8.19.0.esm.js");'
)


class ValidateSiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public = TEST_OUTPUT / self._testMethodName / "public"
        self.public.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(TEST_OUTPUT / self._testMethodName)
        if TEST_OUTPUT.is_dir() and not any(TEST_OUTPUT.iterdir()):
            TEST_OUTPUT.rmdir()

    def run_validator(
        self,
        body: str,
        runtime_extra: str = "",
        inline_runtime: str | None = SAFE_INLINE,
        include_expected_runtime: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if "<body" not in body:
            body = f'<body data-basepath="/onnx-genai">{body}</body>'
        inline = f"<script>{inline_runtime}</script>" if inline_runtime is not None else ""
        (self.public / "index.html").write_text(body + inline, encoding="utf-8")
        runtime = (SAFE_RUNTIME if include_expected_runtime else "") + runtime_extra
        (self.public / "postscript.js").write_text(runtime, encoding="utf-8")
        (self.public / "prescript.js").write_text("window.addCleanup = () => {}", encoding="utf-8")
        vendor = self.public / "static" / "vendor"
        vendor.mkdir(parents=True)
        (vendor / "d3-7.9.0.esm.js").write_text("globalThis.d3 = {}", encoding="utf-8")
        (vendor / "pixi-js-8.19.0.esm.js").write_text(
            "globalThis.PIXI = {}", encoding="utf-8"
        )
        return subprocess.run(
            ["python3", str(VALIDATOR), str(self.public), "/onnx-genai/"],
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
            "const target = document.body.dataset.basepath + '/' + slug;",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_inline_origin_root_fetch(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';fetch("/static/contentIndex.json")',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline origin-root content index", result.stderr)

    def test_rejects_inline_origin_root_navigation(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';result.href = "/" + item.slug',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline origin-root href assignment", result.stderr)

    def test_external_script_is_not_an_inline_runtime_surface(self) -> None:
        result = self.run_validator(
            '<script src="/onnx-genai/postscript.js"></script>',
            inline_runtime=None,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing inline shared fetchData runtime surface", result.stderr)

    def test_rejects_missing_plugin_runtime_surface(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='document.querySelector(".search-container")',
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Graph runtime surface", result.stderr)
        self.assertIn("missing Explorer runtime surface", result.stderr)

    def test_rejects_inline_floating_cdn_import(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=(
                SAFE_INLINE
                + ';import("https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js")'
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_rejects_external_floating_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_unversioned_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)


if __name__ == "__main__":
    unittest.main()
