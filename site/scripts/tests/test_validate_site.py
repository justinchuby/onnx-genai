from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "validate_site.py"
TEST_OUTPUT = Path(__file__).parent / ".validate-site-output"

KATEX_SCRIPT = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js"
KATEX_STYLE = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
D3_SCRIPT = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
PIXI_SCRIPT = "https://cdn.jsdelivr.net/npm/pixi.js@8.19.0/dist/pixi.js"


class ValidateSiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public = TEST_OUTPUT / self._testMethodName / "public"
        self.public.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(TEST_OUTPUT / self._testMethodName)
        if TEST_OUTPUT.is_dir() and not any(TEST_OUTPUT.iterdir()):
            TEST_OUTPUT.rmdir()

    def write_valid_site(
        self,
        *,
        index_head: str = "",
        index_article: str | None = None,
        index_extra: str = "",
        postscript: str = "",
        readme: bool = True,
    ) -> None:
        article = index_article if index_article is not None else "Knowledge base home. " * 12
        head = (
            "<head><title>onnx-genai Knowledge Base</title>"
            '<link rel="canonical" href="./">'
            f'<link rel="stylesheet" href="{KATEX_STYLE}">'
            f"{index_head}</head>"
        )
        body = (
            '<body data-basepath="/onnx-genai"><article>'
            f"{article}<a href='/onnx-genai/README'>README</a>"
            "</article>"
            '<script src="/onnx-genai/prescript.js"></script>'
            '<script src="/onnx-genai/postscript.js"></script>'
            f'<script src="{KATEX_SCRIPT}"></script>'
            f"{index_extra}</body>"
        )
        (self.public / "index.html").write_text(head + body, encoding="utf-8")
        (self.public / "prescript.js").write_text("window.cleanup = []", encoding="utf-8")
        runtime = (
            f'const d3="{D3_SCRIPT}";const pixi="{PIXI_SCRIPT}";'
            'const data=fetch("/onnx-genai/static/contentIndex.json");'
            + postscript
        )
        (self.public / "postscript.js").write_text(runtime, encoding="utf-8")
        static = self.public / "static"
        static.mkdir()
        (static / "contentIndex.json").write_text("{}", encoding="utf-8")
        if readme:
            (self.public / "README.html").write_text(
                "<head><title>Wiki</title></head>"
                '<body data-basepath="/onnx-genai"><article>Wiki conventions.</article></body>',
                encoding="utf-8",
            )

    def run_validator(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(VALIDATOR), str(self.public), "/onnx-genai/"],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_accepts_reviewed_site(self) -> None:
        self.write_valid_site()
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 referenced local script(s)", result.stdout)
        self.assertIn(D3_SCRIPT, result.stdout)

    def test_rejects_missing_internal_target(self) -> None:
        self.write_valid_site(index_extra='<a href="/onnx-genai/missing">missing</a>')
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing internal target", result.stderr)

    def test_rejects_internal_path_escape(self) -> None:
        self.write_valid_site(index_extra='<a href="/outside">outside</a>')
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("internal URL escapes", result.stderr)

    def test_accepts_custom_domain_absolute_link(self) -> None:
        self.write_valid_site(
            index_extra=(
                '<a href="https://www.justinchuby.com/onnx-genai/README">README</a>'
            )
        )
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validated 5 internal link(s)/asset(s)", result.stdout)

    def test_rejects_missing_custom_domain_absolute_target(self) -> None:
        self.write_valid_site(
            index_extra=(
                '<a href="https://www.justinchuby.com/onnx-genai/missing">missing</a>'
            )
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing internal target", result.stderr)

    def test_rejects_custom_domain_absolute_path_escape(self) -> None:
        self.write_valid_site(
            index_extra='<a href="https://www.justinchuby.com/outside">outside</a>'
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("internal URL escapes", result.stderr)

    def test_ignores_old_github_pages_host(self) -> None:
        self.write_valid_site(
            index_extra=(
                '<a href="https://justinchuby.github.io/onnx-genai/missing">'
                "old deployment</a>"
            )
        )
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_wrong_base_path(self) -> None:
        self.write_valid_site()
        index = self.public / "index.html"
        index.write_text(
            index.read_text(encoding="utf-8").replace(
                'data-basepath="/onnx-genai"', 'data-basepath=""'
            ),
            encoding="utf-8",
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("body data-basepath", result.stderr)

    def test_rejects_missing_referenced_script(self) -> None:
        self.write_valid_site()
        (self.public / "postscript.js").unlink()
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing internal target", result.stderr)

    def test_rejects_unreviewed_external_script(self) -> None:
        self.write_valid_site(
            index_extra='<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>'
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unreviewed external runtime URL", result.stderr)

    def test_rejects_unreviewed_external_stylesheet(self) -> None:
        self.write_valid_site(
            index_head='<link rel="stylesheet" href="https://example.com/site.css">'
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unreviewed external runtime URL", result.stderr)

    def test_rejects_unreviewed_dynamic_runtime_url(self) -> None:
        self.write_valid_site(
            postscript='const moving="https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.js";'
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unreviewed external runtime URL", result.stderr)

    def test_rejects_origin_root_runtime_url(self) -> None:
        self.write_valid_site(postscript='fetch("/static/contentIndex.json")')
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("origin-root content-index fetch", result.stderr)

    def test_rejects_redirect_landing_stub(self) -> None:
        self.write_valid_site(
            index_head=(
                '<meta name="robots" content="noindex">'
                '<meta http-equiv="refresh" content="0; url=./README">'
            )
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("redirect/noindex stub", result.stderr)

    def test_rejects_wrong_landing_title(self) -> None:
        self.write_valid_site()
        index = self.public / "index.html"
        index.write_text(
            index.read_text(encoding="utf-8").replace(
                "onnx-genai Knowledge Base", "README", 1
            ),
            encoding="utf-8",
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("title is 'README'", result.stderr)

    def test_rejects_wrong_landing_canonical(self) -> None:
        self.write_valid_site()
        index = self.public / "index.html"
        index.write_text(
            index.read_text(encoding="utf-8").replace('href="./"', 'href="./README"'),
            encoding="utf-8",
        )
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical URL", result.stderr)

    def test_rejects_thin_landing_article(self) -> None:
        self.write_valid_site(index_article="Short")
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("substantive rendered article", result.stderr)

    def test_rejects_missing_separate_readme(self) -> None:
        self.write_valid_site(readme=False)
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("README.html must remain", result.stderr)


if __name__ == "__main__":
    unittest.main()
