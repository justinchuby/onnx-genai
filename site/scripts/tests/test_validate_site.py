from __future__ import annotations

import hashlib
import json
import subprocess
import shutil
import unittest
from pathlib import Path

VALIDATOR = Path(__file__).parents[1] / "validate_site.py"
TEST_OUTPUT = Path(__file__).parent / ".validate-site-output"

D3_SOURCE = b"globalThis.d3 = {};\n"
PIXI_SOURCE = b"globalThis.PIXI = {};\n"
D3_URL = "/onnx-genai/static/vendor/d3-7.9.0.esm.js"
PIXI_URL = "/onnx-genai/static/vendor/pixi-js-8.19.0.esm.js"

SAFE_INLINE = (
    'const fetchData = fetch("/onnx-genai/static/contentIndex.json")'
    ".then((response) => response.json())"
)

# Mirrors the real emitted Graph script-loader shape (see
# site/quartz/scripts/integrate-graph-runtime.mjs and the compiled Graph
# component): a named function creates a <script>, wires it up as a local
# ESM module, and appends it to the document; two real call sites (one per
# vendor asset) give the resource-graph audit genuine, reachable import
# edges rather than bare signature strings.
def loader_runtime(d3_url: str = D3_URL, pixi_url: str = PIXI_URL) -> str:
    return (
        'function d(o){var s=document.createElement("script");'
        's.src=o,s.type="module",s.crossOrigin="anonymous";'
        "document.head.appendChild(s);}"
        f'd("{d3_url}");d("{pixi_url}");'
    )


SEARCH_RUNTIME = (
    'var searchEl=document.querySelector(".search-container");'
    "searchEl.addEventListener('click',function(){});"
)
GRAPH_SURFACE_RUNTIME = (
    'var graphEls=document.querySelectorAll(".graph-container");'
    "graphEls.forEach(function(el){el.textContent='';});"
)
EXPLORER_RUNTIME = (
    'var explorerEl=document.querySelector(".explorer-ul");'
    "explorerEl.addEventListener('click',function(){});"
)
SAFE_RUNTIME = loader_runtime() + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME


class ValidateSiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public = TEST_OUTPUT / self._testMethodName / "public"
        self.public.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(TEST_OUTPUT / self._testMethodName)
        if TEST_OUTPUT.is_dir() and not any(TEST_OUTPUT.iterdir()):
            TEST_OUTPUT.rmdir()

    def write_manifest(
        self,
        manifest_d3_bytes: bytes = D3_SOURCE,
        manifest_pixi_bytes: bytes = PIXI_SOURCE,
        disk_d3_bytes: bytes | None = None,
        disk_pixi_bytes: bytes | None = None,
    ) -> None:
        """Write the vendor bundles plus the deterministic build manifest.

        `manifest_*_bytes` control the sha256 recorded in manifest.json;
        `disk_*_bytes` (defaulting to the same bytes) control what actually
        lands on disk, so tests can independently exercise a manifest/disk
        hash mismatch without the audit script's own hashing logic hiding
        the bug.
        """
        vendor = self.public / "static" / "vendor"
        vendor.mkdir(parents=True, exist_ok=True)
        (vendor / "d3-7.9.0.esm.js").write_bytes(disk_d3_bytes or manifest_d3_bytes)
        (vendor / "pixi-js-8.19.0.esm.js").write_bytes(disk_pixi_bytes or manifest_pixi_bytes)
        manifest = {
            "generator": "test-fixture",
            "basePath": "/onnx-genai",
            "assets": [
                {
                    "package": "d3",
                    "version": "7.9.0",
                    "file": "static/vendor/d3-7.9.0.esm.js",
                    "url": D3_URL,
                    "sha256": hashlib.sha256(manifest_d3_bytes).hexdigest(),
                },
                {
                    "package": "pixi.js",
                    "version": "8.19.0",
                    "file": "static/vendor/pixi-js-8.19.0.esm.js",
                    "url": PIXI_URL,
                    "sha256": hashlib.sha256(manifest_pixi_bytes).hexdigest(),
                },
            ],
        }
        (vendor / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def run_validator(
        self,
        body: str,
        runtime_extra: str = "",
        inline_runtime: str | None = SAFE_INLINE,
        include_expected_runtime: bool = True,
        write_manifest: bool = True,
        include_runtime_scripts: bool = True,
        head: str = "",
        landing_title: str | None = None,
        required_pages: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        if "<body" not in body:
            body = f'<body data-basepath="/onnx-genai">{body}</body>'
        runtime_scripts = (
            '<script src="/onnx-genai/prescript.js"></script>'
            '<script src="/onnx-genai/postscript.js"></script>'
            if include_runtime_scripts
            else ""
        )
        inline = f"<script>{inline_runtime}</script>" if inline_runtime is not None else ""
        (self.public / "index.html").write_text(
            head + body + runtime_scripts + inline, encoding="utf-8"
        )
        runtime = (SAFE_RUNTIME if include_expected_runtime else "") + runtime_extra
        (self.public / "postscript.js").write_text(runtime, encoding="utf-8")
        (self.public / "prescript.js").write_text("window.addCleanup = () => {}", encoding="utf-8")
        if write_manifest:
            self.write_manifest()
        command = ["python3", str(VALIDATOR), str(self.public), "/onnx-genai/"]
        if landing_title is not None:
            command += ["--landing-title", landing_title]
        for page in required_pages:
            command += ["--require-page", page]
        return subprocess.run(command, check=False, capture_output=True, text=True)

    # ------------------------------------------------------------------
    # Pre-existing link/base-path/origin-root/inline-surface coverage.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Non-vacuous plugin presence: comments, dead code, discarded results,
    # and missing loader import edges must all fail even when the raw
    # signature string is present somewhere in the bundle.
    # ------------------------------------------------------------------

    def test_rejects_missing_plugin_runtime_surface(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='document.querySelector(".search-container");',
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn("missing Graph runtime surface", result.stderr)
        self.assertIn("missing Explorer runtime surface", result.stderr)

    def test_rejects_comment_only_signatures(self) -> None:
        """A signature that only ever appears inside a comment must not
        satisfy Search/Graph/Explorer or the local ESM loader: acorn does
        not even produce AST nodes for comments, so these can never be
        counted as `functional`."""
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=(
                '// document.querySelector(".search-container");\n'
                '/* document.querySelectorAll(".graph-container"); */\n'
                "// var explorerUl = document.querySelector('.explorer-ul');\n"
                f'// d("{D3_URL}"); d("{PIXI_URL}");\n'
                "true;"
            ),
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn("missing Graph runtime surface", result.stderr)
        self.assertIn("missing Explorer runtime surface", result.stderr)
        self.assertIn("missing local ESM script-loader function", result.stderr)

    def test_rejects_unreachable_dead_selector_placeholder(self) -> None:
        """A selector wrapped in a constant-false branch is unreachable and
        must be reported distinctly as dead code, not accepted as
        functional evidence."""
        dead_runtime = "if (false) {" + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME + "}"
        result = self.run_validator(
            "<p>Wiki</p>", runtime_extra=dead_runtime, include_expected_runtime=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn("dead/unreachable", result.stderr)

    def test_rejects_dead_graph_loader_after_return(self) -> None:
        """A script-loader call placed after an unconditional `return` in
        the same block is unreachable and must not count as a real import
        edge."""
        dead_loader = (
            "(function(){return;"
            + loader_runtime()
            + "})();"
            + SEARCH_RUNTIME
            + GRAPH_SURFACE_RUNTIME
            + EXPLORER_RUNTIME
        )
        result = self.run_validator(
            "<p>Wiki</p>", runtime_extra=dead_loader, include_expected_runtime=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)
        self.assertIn(f"missing reachable Graph import edge to {PIXI_URL}", result.stderr)

    def test_rejects_vacuous_discarded_selector(self) -> None:
        """Calling querySelector and immediately discarding the result (a
        bare ExpressionStatement) is exactly the vacuous pattern the
        previous fixture itself used to pass with; it must now fail."""
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='document.querySelector(".search-container");',
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn("vacuous", result.stderr)

    def test_rejects_unused_selector_binding(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=(
                loader_runtime()
                + 'var searchEl=document.querySelector(".search-container");'
                + GRAPH_SURFACE_RUNTIME
                + EXPLORER_RUNTIME
            ),
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn("vacuous", result.stderr)

    def test_accepts_selector_binding_consumed_by_behavior(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=loader_runtime() + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_selector_after_statically_decisive_return(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=(
                loader_runtime()
                + "(()=>{if(true)return;"
                + SEARCH_RUNTIME
                + "})();"
                + GRAPH_SURFACE_RUNTIME
                + EXPLORER_RUNTIME
            ),
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn("dead/unreachable", result.stderr)

    def test_accepts_selector_inside_invoked_function(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=(
                loader_runtime()
                + "function wireSearch(){"
                + SEARCH_RUNTIME
                + "}wireSearch();"
                + GRAPH_SURFACE_RUNTIME
                + EXPLORER_RUNTIME
            ),
            include_expected_runtime=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_selector_inside_never_called_function(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=(
                loader_runtime()
                + "function wireSearch(){"
                + SEARCH_RUNTIME
                + "}"
                + GRAPH_SURFACE_RUNTIME
                + EXPLORER_RUNTIME
            ),
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)

    def test_rejects_missing_graph_import_edge(self) -> None:
        """A real, reachable script-loader that is only ever called for one
        of the two vendor assets must fail: Graph's loader must have real
        import edges to BOTH emitted local vendor modules."""
        one_sided_loader = (
            'function d(o){var s=document.createElement("script");'
            's.src=o,s.type="module",s.crossOrigin="anonymous";'
            "document.head.appendChild(s);}"
            f'd("{D3_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=one_sided_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {PIXI_URL}", result.stderr)

    def test_rejects_loader_missing_module_type(self) -> None:
        """A `<script>` that is created and appended but never marked
        `type="module"` is not Graph's real local ESM loader."""
        fake_loader = (
            'function d(o){var s=document.createElement("script");'
            's.src=o,s.crossOrigin="anonymous";'
            "document.head.appendChild(s);}"
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local ESM script-loader function", result.stderr)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)

    def test_rejects_loader_never_appended(self) -> None:
        """A `<script>` that is fully configured but never inserted into
        the document never actually loads anything."""
        fake_loader = (
            'function d(o){var s=document.createElement("script");'
            's.src=o,s.type="module",s.crossOrigin="anonymous";}'
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local ESM script-loader function", result.stderr)

    def test_rejects_independent_manifest_url_calls_without_loader_edge(self) -> None:
        fake_loader = (
            'function d(o){var s=document.createElement("script");'
            's.src=o,s.type="module";document.head.appendChild(s);}'
            "function other(o){return o;}"
            f'other("{D3_URL}");other("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)

    def test_rejects_loader_with_constant_about_blank_src(self) -> None:
        fake_loader = (
            'function d(o){var s=document.createElement("script");'
            's.src="about:blank";s.type="module";document.head.appendChild(s);}'
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local ESM script-loader function", result.stderr)

    def test_rejects_loader_when_manifest_url_is_wrong_parameter(self) -> None:
        fake_loader = (
            'function d(o,src){var s=document.createElement("script");'
            's.src=src;s.type="module";document.head.appendChild(s);}'
            f'd("{D3_URL}","about:blank");d("{PIXI_URL}","about:blank");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)

    def test_rejects_loader_with_overwritten_src_before_append(self) -> None:
        fake_loader = (
            'function d(o){var s=document.createElement("script");'
            's.src=o;s.src="about:blank";s.type="module";document.head.appendChild(s);}'
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local ESM script-loader function", result.stderr)

    def test_rejects_loader_that_appends_separate_script_object(self) -> None:
        fake_loader = (
            'function d(o){var s=document.createElement("script");var t=document.createElement("script");'
            's.src=o;t.type="module";document.head.appendChild(t);}'
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=fake_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing local ESM script-loader function", result.stderr)

    def test_accepts_loader_alias_dataflow_to_appended_src(self) -> None:
        alias_loader = (
            'function d(o){var local=o;var s=document.createElement("script");'
            's.src=local;s.type="module";document.head.appendChild(s);}'
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=alias_loader + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_functional_runtime(self) -> None:
        result = self.run_validator("<p>Wiki</p>")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_unreferenced_decoy_bundle_as_runtime_evidence(self) -> None:
        (self.public / "decoy.js").write_text(SAFE_RUNTIME, encoding="utf-8")
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra="true;",
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Search runtime surface", result.stderr)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)

    def test_accepts_referenced_root_importing_functional_child(self) -> None:
        (self.public / "child.js").write_text(SAFE_RUNTIME, encoding="utf-8")
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='import "./child.js";',
            include_expected_runtime=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_missing_html_referenced_script(self) -> None:
        result = self.run_validator('<script src="/onnx-genai/missing.js"></script><p>Wiki</p>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing referenced/imported script: missing.js", result.stderr)

    def test_rejects_html_script_path_escape(self) -> None:
        result = self.run_validator('<script src="/onnx-genai/../escape.js"></script><p>Wiki</p>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local URL escapes /onnx-genai", result.stderr)

    def test_rejects_missing_static_import(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='import "./missing-child.js";' + SAFE_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing referenced/imported script: missing-child.js", result.stderr)

    def test_rejects_missing_vendor_manifest(self) -> None:
        result = self.run_validator("<p>Wiki</p>", write_manifest=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime resource audit:", result.stderr)

    def test_rejects_vendor_asset_hash_mismatch(self) -> None:
        """The manifest is emitted by the build itself; if the deployed
        bytes ever drift from what the manifest recorded (a stale copy, a
        hand-edit, or a corrupted asset) validation must fail rather than
        trust the manifest blindly."""
        (self.public / "index.html").write_text(
            '<body data-basepath="/onnx-genai"><p>Wiki</p></body>'
            '<script src="/onnx-genai/prescript.js"></script>'
            '<script src="/onnx-genai/postscript.js"></script>'
            f"<script>{SAFE_INLINE}</script>",
            encoding="utf-8",
        )
        (self.public / "postscript.js").write_text(SAFE_RUNTIME, encoding="utf-8")
        (self.public / "prescript.js").write_text("window.addCleanup = () => {}", encoding="utf-8")
        self.write_manifest(disk_d3_bytes=b"tampered-bytes-do-not-match-manifest-hash")
        result = subprocess.run(
            ["python3", str(VALIDATOR), str(self.public), "/onnx-genai/"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("content hash does not match build manifest", result.stderr)

    def test_rejects_inline_comment_only_fetch_data(self) -> None:
        """A `fetchData` declaration whose real `fetch(...)` call only
        exists in a comment must not satisfy the shared inline runtime
        surface requirement."""
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=(
                '// const fetchData = fetch("/onnx-genai/static/contentIndex.json");\n'
                "var fetchData = null;"
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing inline shared fetchData runtime surface", result.stderr)

    def test_rejects_inline_dead_fetch_data(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=(
                'if (false) { const fetchData = fetch("/onnx-genai/static/contentIndex.json"); }'
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing inline shared fetchData runtime surface", result.stderr)
        self.assertIn("dead/unreachable", result.stderr)

    def test_rejects_inline_never_called_fetch_closure(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime='const fetchData = () => fetch("/onnx-genai/static/contentIndex.json")',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing inline shared fetchData runtime surface", result.stderr)

    def test_accepts_inline_invoked_fetch_closure(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime='const fetchData = (() => fetch("/onnx-genai/static/contentIndex.json"))()',
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_inline_short_circuited_fetch_data(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime='const fetchData = false && fetch("/onnx-genai/static/contentIndex.json")',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing inline shared fetchData runtime surface", result.stderr)

    # ------------------------------------------------------------------
    # Hardened jsDelivr URL parsing: package/version boundary is `/`,
    # end-of-string, `?`, or `#` -- not only `/`. Exact-version allowlisting
    # is keyed to a specific package, not just "looks like a semver".
    # ------------------------------------------------------------------

    def test_rejects_bare_floating_major_cdn_script(self) -> None:
        result = self.run_validator('<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_scheme_relative_floating_major_cdn_script(self) -> None:
        result = self.run_validator('<script src="//cdn.jsdelivr.net/npm/d3@7"></script>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_bare_floating_major_cdn_script_with_query(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3@7?min"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_scheme_relative_floating_major_cdn_script_with_query(self) -> None:
        result = self.run_validator('<script src="//cdn.jsdelivr.net/npm/d3@7?min"></script>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_bare_floating_major_cdn_script_with_fragment(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3@7#sha256-abc"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_bare_unversioned_cdn_script(self) -> None:
        result = self.run_validator('<script src="https://cdn.jsdelivr.net/npm/d3"></script>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_bare_unversioned_cdn_script_with_query(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3?min"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_bare_unversioned_cdn_script_with_fragment(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3#sha256-abc"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_scoped_unversioned_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/@scope/name"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_scoped_floating_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/@scope/name@7"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_scoped_range_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/@scope/name@^7.0.0"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_slash_path_floating_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_slash_path_unversioned_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_exact_version_of_non_allowlisted_package(self) -> None:
        """An exact semver alone is not sufficient: `d3` (and `pixi.js`)
        have zero legitimate jsDelivr references in production because
        they are fully bundled locally, so even a pinned exact version
        must still be rejected."""
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_prerelease_version_of_allowlisted_package(self) -> None:
        """A prerelease/other exact-shaped version of an allowlisted
        package that was never actually reviewed/allowlisted must still be
        rejected -- allowlisting is exact-string, not "any semver"."""
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11-rc.1/dist/katex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_accepts_reviewed_exact_allowlisted_cdn_script(self) -> None:
        """The pinned `latex` Quartz plugin's KaTeX copy-tex asset is the
        one currently reviewed, exact-immutable-semver jsDelivr executable
        reference; it must keep passing."""
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_scheme_relative_reviewed_exact_allowlisted_cdn_script(self) -> None:
        result = self.run_validator(
            '<script src="//cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_allowlisted_package_with_different_asset_path(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_allowlisted_asset_with_query(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js?module"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_allowlisted_asset_with_fragment(self) -> None:
        result = self.run_validator(
            '<script src="//cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js#sha256-abc"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_raw_dot_segment_cdn_package_escape(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/../d3@7/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_encoded_dot_segment_cdn_package_escape(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/%2e%2e/d3@7/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_double_encoded_dot_segment_cdn_package_escape(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/%252e%252e/d3@7/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_single_dot_segment_cdn_package_escape(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/%2e/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_inline_string_floating_cdn_reference(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';const url = "https://cdn.jsdelivr.net/npm/d3@7"',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_rejects_static_import_floating_cdn_reference(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='import "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js";'
            + SAFE_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_inline_floating_cdn_import(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=(
                SAFE_INLINE + ';import("https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js")'
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_rejects_inline_bare_unversioned_cdn_import(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';import("https://cdn.jsdelivr.net/npm/d3")',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_rejects_inline_scoped_unversioned_cdn_import(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';import("https://cdn.jsdelivr.net/npm/@scope/name")',
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

    # ------------------------------------------------------------------
    # Executable CDN host/namespace policy. jsDelivr serves the same mutable
    # package namespace from several hostnames and as ESM through esm.run, so
    # hostnames are canonicalized (lowercase, trailing dot stripped) before
    # matching. Allowlisting stays narrower than detection: only the exact
    # canonical host + `/npm/` namespace + reviewed package/version/asset the
    # pinned latex plugin actually emits is approved.
    # ------------------------------------------------------------------

    def test_rejects_fastly_host_exact_allowlisted_katex_asset(self) -> None:
        """An alternate jsDelivr mirror must not inherit the canonical
        host's reviewed KaTeX allowlist entry."""
        result = self.run_validator(
            '<script src="https://fastly.jsdelivr.net/npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_gcore_host_exact_allowlisted_katex_asset(self) -> None:
        result = self.run_validator(
            '<script src="https://gcore.jsdelivr.net/npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_testingcf_host_exact_allowlisted_katex_asset(self) -> None:
        result = self.run_validator(
            '<script src="https://testingcf.jsdelivr.net/npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_fastly_host_d3_bundle(self) -> None:
        result = self.run_validator(
            '<script src="https://fastly.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_gcore_host_pixi_bundle(self) -> None:
        result = self.run_validator(
            '<script src="https://gcore.jsdelivr.net/npm/pixi.js@8.19.0/dist/pixi.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_testingcf_host_floating_range_bundle(self) -> None:
        result = self.run_validator(
            '<script src="https://testingcf.jsdelivr.net/npm/d3@^7/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_scheme_relative_alternate_host_script(self) -> None:
        result = self.run_validator(
            '<script src="//fastly.jsdelivr.net/npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_uppercase_canonical_host_allowlisted_asset(self) -> None:
        """`CDN.JSDELIVR.NET` is the same origin to a browser, but it is not
        the reviewed canonical emission, so it is detected and rejected."""
        result = self.run_validator(
            '<script src="https://CDN.JSDELIVR.NET/npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_trailing_dot_canonical_host_allowlisted_asset(self) -> None:
        """A trailing DNS root dot resolves to the same origin and must not
        bypass the host policy."""
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net./npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_uppercase_alternate_host_d3_script(self) -> None:
        result = self.run_validator(
            '<script src="https://Fastly.JSDelivr.NET/npm/d3@7/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_esm_run_bare_package_script(self) -> None:
        """`esm.run` always resolves a package specifier to jsDelivr's latest
        matching build, so every reference to it is mutable."""
        result = self.run_validator('<script src="https://esm.run/d3" type="module"></script>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_esm_run_exact_version_script(self) -> None:
        result = self.run_validator(
            '<script src="https://esm.run/katex@0.16.11/dist/contrib/copy-tex.min.js"'
            ' type="module"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_esm_run_bare_specifier_inline_import(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';import("https://esm.run/d3")',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_rejects_esm_run_scheme_relative_inline_string(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE + ';const url = "//esm.run/pixi.js@8"',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_rejects_esm_run_with_query_variant(self) -> None:
        result = self.run_validator(
            '<script src="https://esm.run/d3@7.9.0?bundle" type="module"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_gh_namespace_script(self) -> None:
        """The `/gh/` namespace serves arbitrary GitHub refs and is outside
        the reviewed `/npm/` namespace even on the canonical host."""
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/gh/jquery/jquery@main/dist/jquery.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_gh_namespace_with_exact_tag(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/gh/katex/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_wp_namespace_script(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/wp/some-plugin/trunk/js/script.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_alternate_host_static_import_in_reachable_bundle(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='import "https://fastly.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js";'
            + SAFE_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_alternate_host_dynamic_import_in_reachable_bundle(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra='import("https://gcore.jsdelivr.net/npm/pixi.js@8");' + SAFE_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_alternate_host_dot_segment_escape(self) -> None:
        result = self.run_validator(
            '<script src="https://testingcf.jsdelivr.net/npm/katex@0.16.11'
            '/%2e%2e/d3@7/dist/d3.min.js"></script>'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external script uses mutable jsDelivr", result.stderr)

    def test_rejects_alternate_host_inline_string_reference(self) -> None:
        result = self.run_validator(
            "<p>Wiki</p>",
            inline_runtime=SAFE_INLINE
            + ';const url = "https://testingcf.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inline mutable jsDelivr", result.stderr)

    def test_accepts_canonical_host_reviewed_asset_alongside_other_origins(self) -> None:
        """Non-jsDelivr third-party origins are not silently promoted into the
        executable allowlist, and the reviewed canonical KaTeX script keeps
        passing next to an unrelated preconnect hint."""
        result = self.run_validator(
            '<link rel="preconnect" href="https://cdnjs.cloudflare.com">'
            '<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11'
            '/dist/contrib/copy-tex.min.js"></script>'
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    # ------------------------------------------------------------------
    # Third-party stylesheet policy. External `<link rel="stylesheet">`
    # hrefs are a real runtime load surface, so they carry their own exact
    # allowlist instead of being implicitly out of scope.
    # ------------------------------------------------------------------

    def test_accepts_allowlisted_third_party_stylesheet(self) -> None:
        result = self.run_validator(
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11'
            '/dist/katex.min.css">'
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_local_relative_stylesheet(self) -> None:
        (self.public / "index.css").write_text("body{}", encoding="utf-8")
        result = self.run_validator('<link rel="stylesheet" href="/onnx-genai/index.css">')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_unapproved_version_of_third_party_stylesheet(self) -> None:
        result = self.run_validator(
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10'
            '/dist/katex.min.css">'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("third-party stylesheet is not allowlisted", result.stderr)

    def test_rejects_alternate_host_third_party_stylesheet(self) -> None:
        result = self.run_validator(
            '<link rel="stylesheet" href="https://fastly.jsdelivr.net/npm/katex@0.16.11'
            '/dist/katex.min.css">'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("third-party stylesheet is not allowlisted", result.stderr)

    def test_rejects_unrelated_origin_third_party_stylesheet(self) -> None:
        result = self.run_validator(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/katex/'
            '0.16.11/katex.min.css">'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("third-party stylesheet is not allowlisted", result.stderr)

    def test_rejects_scheme_relative_unapproved_stylesheet(self) -> None:
        result = self.run_validator(
            '<link rel="stylesheet" href="//esm.run/katex@0.16.11/dist/katex.min.css">'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("third-party stylesheet is not allowlisted", result.stderr)

    # ------------------------------------------------------------------
    # Loader lexical identity: vendor/import edges are keyed to the AST node
    # of the binding actually in lexical scope at the call site, never to a
    # function name. Minified bundles reuse short names across scopes.
    # ------------------------------------------------------------------

    def test_rejects_same_name_loader_defined_in_another_scope(self) -> None:
        """Two `d` functions in distinct lexical scopes: only the one that is
        never reachable has loader behavior, while the reachable calls go to a
        same-name pass-through. Name-only credit would accept this."""
        decoy_scope_loader = (
            "function wrapper(){"
            'function d(o){var s=document.createElement("script");'
            's.src=o;s.type="module";document.head.appendChild(s);'
            "}"
            f'd("{D3_URL}");d("{PIXI_URL}");'
            "}"
            "function d(o){return o;}"
            f'd("{D3_URL}");d("{PIXI_URL}");'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=decoy_scope_loader
            + SEARCH_RUNTIME
            + GRAPH_SURFACE_RUNTIME
            + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)
        self.assertIn(f"missing reachable Graph import edge to {PIXI_URL}", result.stderr)
        self.assertIn("missing local ESM script-loader function", result.stderr)

    def test_rejects_loader_name_shadowed_by_parameter_at_call_site(self) -> None:
        """A real top-level loader `d` must not credit calls made to an
        unrelated `d` parameter that shadows it in an inner scope."""
        shadowed = (
            'function d(o){var s=document.createElement("script");'
            's.src=o;s.type="module";document.head.appendChild(s);}'
            f'function run(d){{d("{D3_URL}");d("{PIXI_URL}");}}'
            "run(function(o){return o;});"
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=shadowed + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)
        self.assertIn(f"missing reachable Graph import edge to {PIXI_URL}", result.stderr)

    def test_rejects_same_name_loader_credit_from_dead_code(self) -> None:
        """Unreachable calls to a same-name non-loader must not be reported
        as dead evidence for the real loader either."""
        dead_scope = (
            "function wrapper(){"
            'function d(o){var s=document.createElement("script");'
            's.src=o;s.type="module";document.head.appendChild(s);'
            "}return d;}"
            "function d(o){return o;}"
            f'if(false){{d("{D3_URL}");d("{PIXI_URL}");}}'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=dead_scope + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing reachable Graph import edge to {D3_URL}", result.stderr)
        self.assertNotIn("only found in dead/unreachable code", result.stderr)

    def test_accepts_scoped_loader_binding_invoked_in_its_own_scope(self) -> None:
        """The positive counterpart: a loader declared inside an invoked
        function, called through the binding that is lexically in scope."""
        scoped_loader = (
            "function init(){"
            'function d(o){var s=document.createElement("script");'
            's.src=o;s.type="module";document.head.appendChild(s);'
            "}"
            f'd("{D3_URL}");d("{PIXI_URL}");'
            "}"
            "init();"
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=scoped_loader
            + SEARCH_RUNTIME
            + GRAPH_SURFACE_RUNTIME
            + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_top_level_loader_with_unrelated_same_name_inner_function(self) -> None:
        """The real production shape: a top-level loader plus an unrelated
        same-name function in a nested scope must still validate."""
        mixed = (
            'function d(o){var s=document.createElement("script");'
            's.src=o;s.type="module";document.head.appendChild(s);}'
            "function other(){function d(x){return x;}return d(1);}"
            f'd("{D3_URL}");d("{PIXI_URL}");other();'
        )
        result = self.run_validator(
            "<p>Wiki</p>",
            runtime_extra=mixed + SEARCH_RUNTIME + GRAPH_SURFACE_RUNTIME + EXPLORER_RUNTIME,
            include_expected_runtime=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    # ------------------------------------------------------------------
    # Root landing semantics: `public/index.html` must be rendered content,
    # not a permalink/alias redirect stub, and named pages must stay emitted
    # and reachable.
    # ------------------------------------------------------------------

    LANDING_TITLE = "onnx-genai Knowledge Base"

    def landing_head(
        self,
        title: str | None = None,
        canonical: str | None = "/onnx-genai/",
        extra_meta: str = "",
    ) -> str:
        head = "<!DOCTYPE html><html lang=\"en-us\"><head>"
        head += f"<title>{self.LANDING_TITLE if title is None else title}</title>"
        if canonical is not None:
            head += f'<link rel="canonical" href="{canonical}">'
        return head + extra_meta + "</head>"

    def landing_body(self, article_text: str | None = None, links: str = "") -> str:
        text = article_text if article_text is not None else "Knowledge base landing copy. " * 12
        return (
            '<body data-basepath="/onnx-genai">'
            f"<article><h1>{self.LANDING_TITLE}</h1><p>{text}</p>{links}</article>"
            "</body>"
        )

    def write_linked_page(self, name: str, with_body: bool = True) -> None:
        content = (
            '<body data-basepath="/onnx-genai"><article><p>Wiki conventions.</p></article></body>'
            f"<script>{SAFE_INLINE}</script>"
            if with_body
            else '<meta http-equiv="refresh" content="0; url=./index">'
        )
        (self.public / name).write_text(content, encoding="utf-8")

    def test_accepts_rendered_landing_page(self) -> None:
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Landing audit:", result.stdout)

    def test_accepts_relative_landing_canonical(self) -> None:
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(canonical="./"),
            landing_title=self.LANDING_TITLE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_meta_refresh_landing_stub(self) -> None:
        """The `permalink: index` alias hack emits exactly this: a titled,
        noindex meta-refresh document with no rendered body."""
        stub = (
            "<!DOCTYPE html><html lang=\"en-us\"><head><title>README</title>"
            '<link rel="canonical" href="./README">'
            '<meta name="robots" content="noindex">'
            '<meta http-equiv="refresh" content="0; url=./README">'
            "</head></html>"
        )
        result = self.run_validator(
            "",
            head=stub,
            include_runtime_scripts=False,
            inline_runtime=None,
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("meta-refresh redirect stub", result.stderr)
        self.assertIn("landing page is marked noindex", result.stderr)

    def test_rejects_noindex_landing_page(self) -> None:
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(extra_meta='<meta name="robots" content="noindex, nofollow">'),
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("landing page is marked noindex", result.stderr)

    def test_rejects_landing_title_mismatch(self) -> None:
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(title="README"),
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("landing title is 'README'", result.stderr)

    def test_rejects_landing_without_rendered_article(self) -> None:
        result = self.run_validator(
            '<body data-basepath="/onnx-genai"><p>Knowledge base landing copy.</p></body>',
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no rendered article", result.stderr)

    def test_rejects_landing_with_thin_article(self) -> None:
        result = self.run_validator(
            self.landing_body(article_text="Hi"),
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rendered character(s), expected at least", result.stderr)

    def test_rejects_landing_canonical_pointing_at_another_page(self) -> None:
        self.write_linked_page("README.html")
        result = self.run_validator(
            self.landing_body(links='<a href="/onnx-genai/README">README</a>'),
            head=self.landing_head(canonical="/onnx-genai/README"),
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical URL is", result.stderr)

    def test_rejects_landing_without_canonical(self) -> None:
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(canonical=None),
            landing_title=self.LANDING_TITLE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("declares no canonical URL", result.stderr)

    def test_accepts_required_linked_page(self) -> None:
        self.write_linked_page("README.html")
        result = self.run_validator(
            self.landing_body(links='<a href="/onnx-genai/README">Wiki conventions</a>'),
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
            required_pages=("README.html",),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_missing_required_page(self) -> None:
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
            required_pages=("README.html",),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required generated page: README.html", result.stderr)

    def test_rejects_unlinked_required_page(self) -> None:
        self.write_linked_page("README.html")
        result = self.run_validator(
            self.landing_body(),
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
            required_pages=("README.html",),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not linked from any other page: README.html", result.stderr)

    def test_rejects_required_page_without_body(self) -> None:
        self.write_linked_page("README.html", with_body=False)
        result = self.run_validator(
            self.landing_body(links='<a href="/onnx-genai/README">Wiki conventions</a>'),
            head=self.landing_head(),
            landing_title=self.LANDING_TITLE,
            required_pages=("README.html",),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has no rendered body: README.html", result.stderr)


if __name__ == "__main__":
    unittest.main()
