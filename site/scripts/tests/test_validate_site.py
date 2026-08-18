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
    ) -> subprocess.CompletedProcess[str]:
        if "<body" not in body:
            body = f'<body data-basepath="/onnx-genai">{body}</body>'
        inline = f"<script>{inline_runtime}</script>" if inline_runtime is not None else ""
        (self.public / "index.html").write_text(body + inline, encoding="utf-8")
        runtime = (SAFE_RUNTIME if include_expected_runtime else "") + runtime_extra
        (self.public / "postscript.js").write_text(runtime, encoding="utf-8")
        (self.public / "prescript.js").write_text("window.addCleanup = () => {}", encoding="utf-8")
        if write_manifest:
            self.write_manifest()
        return subprocess.run(
            ["python3", str(VALIDATOR), str(self.public), "/onnx-genai/"],
            check=False,
            capture_output=True,
            text=True,
        )

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

    def test_accepts_functional_runtime(self) -> None:
        result = self.run_validator("<p>Wiki</p>")
        self.assertEqual(result.returncode, 0, result.stderr)

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
            f'<body data-basepath="/onnx-genai"><p>Wiki</p></body><script>{SAFE_INLINE}</script>',
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

    # ------------------------------------------------------------------
    # Hardened jsDelivr URL parsing: package/version boundary is `/`,
    # end-of-string, `?`, or `#` -- not only `/`. Exact-version allowlisting
    # is keyed to a specific package, not just "looks like a semver".
    # ------------------------------------------------------------------

    def test_rejects_bare_floating_major_cdn_script(self) -> None:
        result = self.run_validator('<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutable jsDelivr", result.stderr)

    def test_rejects_bare_floating_major_cdn_script_with_query(self) -> None:
        result = self.run_validator(
            '<script src="https://cdn.jsdelivr.net/npm/d3@7?min"></script>'
        )
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


if __name__ == "__main__":
    unittest.main()
