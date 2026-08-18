#!/usr/bin/env python3
"""Validate a built Quartz site as mounted at a GitHub Pages project path."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.inline_scripts: list[str] = []
        self.script_sources: list[str] = []
        self.has_body = False
        self.body_base_path: str | None = None
        self._script_body: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "body":
            self.has_body = True
            self.body_base_path = attributes.get("data-basepath")
        if tag in {"a", "link"} and attributes.get("href"):
            self.urls.append(attributes["href"] or "")
        if tag in {"img", "script", "source"} and attributes.get("src"):
            self.urls.append(attributes["src"] or "")
        if tag == "script":
            source = attributes.get("src")
            if source:
                self.script_sources.append(source)
            else:
                self._script_body = []

    def handle_data(self, data: str) -> None:
        if self._script_body is not None:
            self._script_body.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_body is not None:
            self.inline_scripts.append("".join(self._script_body))
            self._script_body = None


def output_candidates(public: Path, deployed_path: str, base_path: str) -> list[Path]:
    if deployed_path == base_path.rstrip("/"):
        return [public / "index.html"]
    relative = unquote(deployed_path[len(base_path) :]).lstrip("/")
    if not relative:
        return [public / "index.html"]
    path = public / PurePosixPath(relative)
    if deployed_path.endswith("/"):
        candidates = [path / "index.html"]
    elif path.suffix:
        candidates = [path]
    else:
        candidates = [Path(f"{path}.html"), path / "index.html"]
    safe_candidates: list[Path] = []
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(public)
        except ValueError:
            continue
        safe_candidates.append(candidate)
    return safe_candidates


def page_url(html: Path, public: Path, base_path: str) -> str:
    relative = html.relative_to(public).as_posix()
    if relative == "index.html":
        return base_path
    return f"{base_path}{relative.removesuffix('.html')}"


RUNTIME_ROOT_PATTERNS = {
    "origin-root content index": re.compile(
        r"""(?:fetch\(|new URL\()\s*["']/static/contentIndex\.json"""
    ),
    "origin-root URL constructor": re.compile(r"""new URL\(\s*["']/["']?\s*\+"""),
    "origin-root href assignment": re.compile(r"""\.href\s*=\s*["']/["']?\s*\+"""),
}

# jsDelivr npm URL parsing.
#
# A package/version boundary is any of: `/`, end-of-string, `?`, or `#` (a
# trailing `/` is NOT required, unlike the previous pattern, which missed
# bare executable URLs such as `.../npm/d3@7` or `.../npm/d3` with no
# trailing path segment, and any `?query`/`#fragment` suffix). Scoped
# packages (`@scope/name`) are parsed as a single package identifier so a
# `@version` suffix is never mistaken for part of the scope/name.
JSDELIVR_NPM_PATTERN = re.compile(
    r"""https?://cdn\.jsdelivr\.net/npm/"""
    r"""(?P<package>@[^/@?#"'\s]+/[^@/?#"'\s]+|[^@/?#"'\s]+)"""
    r"""(?:@(?P<version>[^/?#"'\s]*))?""",
    re.IGNORECASE,
)
EXACT_SEMVER_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")

# Runtime executable jsDelivr references must be exact immutable semver
# *and* the exact package must be individually reviewed and allowlisted --
# an exact-semver shape alone is not sufficient, otherwise a prefix,
# prerelease, or an unrelated package could ride along on a coincidentally
# well-formed version string. Local bundling means Graph's d3/pixi.js have
# zero legitimate jsDelivr references in production (see
# integrate-graph-runtime.mjs); the only currently reviewed exception is the
# pinned `latex` Quartz plugin's KaTeX copy-tex asset, embedded at the exact
# commit recorded in quartz.lock.json.
JSDELIVR_ALLOWLIST: dict[str, frozenset[str]] = {
    "katex": frozenset({"0.16.11"}),
}


def mutable_cdn_references(source: str) -> int:
    count = 0
    for match in JSDELIVR_NPM_PATTERN.finditer(source):
        package = match["package"]
        version = match["version"]
        if (
            version is not None
            and EXACT_SEMVER_PATTERN.fullmatch(version) is not None
            and version in JSDELIVR_ALLOWLIST.get(package, frozenset())
        ):
            continue
        count += 1
    return count


AUDIT_SCRIPT = Path(__file__).resolve().parent.parent / "quartz" / "scripts" / "audit-runtime-resources.mjs"
VENDOR_MANIFEST_RELATIVE = PurePosixPath("static/vendor/manifest.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_resource_audit(
    public: Path,
    expected_base_path: str,
    documents: list[tuple[Path, LinkParser]],
    errors: set[str],
) -> dict[str, object]:
    """Run the deterministic AST/resource-graph audit and fold its findings
    into `errors`.

    This is the non-vacuous replacement for substring-counting Search,
    Graph, Explorer, the local ESM loader, and Graph's vendor import edges:
    site/quartz/scripts/audit-runtime-resources.mjs parses every emitted
    `*.js` bundle (and every inline `<script>` body) into a real AST, prunes
    provably unreachable/dead branches, and only counts a signature as
    "functional" when something downstream actually depends on it. A
    comment, an unused string, or a dead `if (false) {...}` placeholder
    cannot satisfy any of these checks. Graph's local vendor assets are
    cross-checked here against the sha256 recorded in the build-emitted
    `static/vendor/manifest.json`, so the manifest itself cannot be
    hand-edited without also matching real deployed bytes.
    """
    manifest_path = public / VENDOR_MANIFEST_RELATIVE
    inline_payload = json.dumps(
        {
            "inlineScripts": [
                {"id": f"{html.relative_to(public).as_posix()}#{index}", "source": source}
                for html, document in documents
                for index, source in enumerate(document.inline_scripts)
            ]
        }
    )
    try:
        result = subprocess.run(
            ["node", str(AUDIT_SCRIPT), str(public), expected_base_path, str(manifest_path)],
            input=inline_payload,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        errors.add(f"runtime resource audit could not run node: {error}")
        return {}
    if result.returncode != 0:
        errors.add(
            f"runtime resource audit failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
        return {}
    try:
        audit = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        errors.add(f"runtime resource audit produced invalid JSON: {error}")
        return {}
    if "manifestError" in audit:
        errors.add(f"runtime resource audit: {audit['manifestError']}")
        return {}

    for entry in audit.get("parseErrors", []):
        errors.add(f"{entry['file']}: runtime bundle failed to parse: {entry['message']}")

    manifest = audit.get("manifest", {})
    assets = manifest.get("assets", [])
    if not assets:
        errors.add(f"{manifest_path}: vendor manifest declares no assets")
    for asset in assets:
        asset_path = public / asset["file"]
        if not asset_path.is_file():
            errors.add(f"missing local Graph runtime asset: {asset_path}")
            continue
        actual_hash = sha256_file(asset_path)
        if actual_hash != asset.get("sha256"):
            errors.add(
                f"{asset_path}: content hash does not match build manifest {manifest_path} "
                f"(expected {asset.get('sha256')}, found {actual_hash})"
            )
        edge = audit.get("vendorEdges", {}).get(asset["url"], {"edges": 0, "dead": 0})
        if edge.get("edges", 0) == 0:
            hint = " (only found in dead/unreachable code)" if edge.get("dead", 0) else ""
            errors.add(f"postscript.js: missing reachable Graph import edge to {asset['url']}{hint}")

    for surface, counts in audit.get("surfaces", {}).items():
        if counts.get("functional", 0) == 0:
            hints = []
            if counts.get("vacuous", 0):
                hints.append(f"{counts['vacuous']} vacuous (result discarded)")
            if counts.get("dead", 0):
                hints.append(f"{counts['dead']} dead/unreachable")
            suffix = f" (found only: {', '.join(hints)})" if hints else ""
            errors.add(f"postscript.js: missing {surface} runtime surface{suffix}")

    if not audit.get("loaderNames"):
        errors.add(
            "postscript.js: missing local ESM script-loader function "
            "(createElement('script') + type='module' + DOM append)"
        )

    fetch_entries = audit.get("sharedFetchData", [])
    for entry in fetch_entries:
        if entry.get("parseError"):
            errors.add(f"{entry['id']}: inline script failed to parse: {entry['parseError']}")
    for html, document in documents:
        if not document.has_body:
            continue
        prefix = f"{html.relative_to(public).as_posix()}#"
        page_entries = [entry for entry in fetch_entries if entry["id"].startswith(prefix)]
        if sum(entry.get("functional", 0) for entry in page_entries) == 0:
            hints = []
            vacuous = sum(entry.get("vacuous", 0) for entry in page_entries)
            dead = sum(entry.get("dead", 0) for entry in page_entries)
            if vacuous:
                hints.append(f"{vacuous} vacuous")
            if dead:
                hints.append(f"{dead} dead/unreachable")
            suffix = f" (found only: {', '.join(hints)})" if hints else ""
            errors.add(f"{html}: missing inline shared fetchData runtime surface{suffix}")

    return audit


def validate_runtime(
    public: Path,
    expected_base_path: str,
    documents: list[tuple[Path, LinkParser]],
    errors: set[str],
) -> dict[str, int]:
    scripts = sorted(public.rglob("*.js"))
    origin_root_count = 0
    mutable_cdn_count = 0
    bundle_present: set[Path] = set()
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        bundle_present.add(script)
        for description, pattern in RUNTIME_ROOT_PATTERNS.items():
            matches = pattern.findall(source)
            origin_root_count += len(matches)
            if matches:
                errors.add(f"{script}: {description} bypasses {expected_base_path}")
        matches = mutable_cdn_references(source)
        mutable_cdn_count += matches
        if matches:
            errors.add(f"{script}: mutable jsDelivr npm runtime dependency")

    required_bundles = (public / "prescript.js", public / "postscript.js")
    for bundle in required_bundles:
        if bundle not in bundle_present:
            errors.add(f"missing required runtime bundle: {bundle}")

    inline_script_count = 0
    for html, document in documents:
        inline_script_count += len(document.inline_scripts)
        for source in document.inline_scripts:
            for description, pattern in RUNTIME_ROOT_PATTERNS.items():
                matches = pattern.findall(source)
                origin_root_count += len(matches)
                if matches:
                    errors.add(f"{html}: inline {description} bypasses {expected_base_path}")
            matches = mutable_cdn_references(source)
            mutable_cdn_count += matches
            if matches:
                errors.add(f"{html}: inline mutable jsDelivr npm runtime dependency")
        for source in document.script_sources:
            matches = mutable_cdn_references(source)
            if matches:
                mutable_cdn_count += matches
                errors.add(f"{html}: external script uses mutable jsDelivr npm dependency")

    audit = run_resource_audit(public, expected_base_path, documents, errors)
    surfaces = audit.get("surfaces", {}) if audit else {}
    vendor_edges = audit.get("vendorEdges", {}) if audit else {}
    manifest_assets = audit.get("manifest", {}).get("assets", []) if audit else []
    fetch_entries = audit.get("sharedFetchData", []) if audit else []

    return {
        "bundles": len(scripts),
        "inline_scripts": inline_script_count,
        "inline_fetch": sum(entry.get("functional", 0) for entry in fetch_entries),
        "origin_root": origin_root_count,
        "mutable_cdn": mutable_cdn_count,
        "local_vendor_assets": sum(
            1 for asset in manifest_assets if (public / asset["file"]).is_file()
        ),
        "local_vendor_references": sum(edge.get("edges", 0) for edge in vendor_edges.values()),
        "module_loaders": len(audit.get("loaderNames", [])) if audit else 0,
        **{name: counts.get("functional", 0) for name, counts in surfaces.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("public", type=Path)
    parser.add_argument("base_path")
    args = parser.parse_args()
    public = args.public.resolve()
    base_path = f"/{args.base_path.strip('/')}/"
    html_files = sorted(public.rglob("*.html"))
    errors: set[str] = set()
    checked = 0
    body_pages = 0
    documents: list[tuple[Path, LinkParser]] = []

    if not (public / "index.html").is_file():
        errors.add("missing public/index.html landing page")

    for html in html_files:
        document = LinkParser()
        document.feed(html.read_text(encoding="utf-8"))
        documents.append((html, document))
        if document.has_body:
            body_pages += 1
            if document.body_base_path != base_path.rstrip("/"):
                errors.add(
                    f"{html}: body data-basepath is {document.body_base_path!r}, "
                    f"expected {base_path.rstrip('/')!r}"
                )
        current_url = f"https://justinchuby.github.io{page_url(html, public, base_path)}"
        for raw_url in document.urls:
            parsed = urlparse(raw_url)
            if parsed.scheme in {"data", "mailto", "tel", "javascript"}:
                continue
            if parsed.scheme in {"http", "https"} and parsed.hostname != "justinchuby.github.io":
                continue
            if raw_url.startswith("#") or not raw_url:
                continue
            checked += 1
            resolved = urlparse(urljoin(current_url, raw_url))
            if resolved.path != base_path.rstrip("/") and not resolved.path.startswith(base_path):
                errors.add(f"{html}: internal URL escapes {base_path}: {raw_url}")
                continue
            candidates = output_candidates(public, resolved.path, base_path)
            if not any(candidate.is_file() for candidate in candidates):
                errors.add(f"{html}: missing internal target: {raw_url}")

    if body_pages == 0:
        errors.add("no generated page has a body for runtime navigation")
    runtime = validate_runtime(public, base_path, documents, errors)
    if errors:
        print("\n".join(sorted(errors)), file=sys.stderr)
        print(f"Found {len(errors)} generated site link error(s).", file=sys.stderr)
        return 1
    print(
        f"Validated {checked} generated internal link(s)/asset(s) across "
        f"{len(html_files)} HTML page(s) and {runtime['bundles']} runtime bundle(s) "
        f"at {base_path}."
    )
    print(
        f"Runtime audit: {runtime['inline_scripts']} inline script body(s), "
        f"{runtime['inline_fetch']} shared fetchData surface(s), "
        f"Search={runtime['Search']}, Graph={runtime['Graph']}, "
        f"Explorer={runtime['Explorer']}, forbidden origin-root="
        f"{runtime['origin_root']}, mutable CDN={runtime['mutable_cdn']}, "
        f"local vendor assets={runtime['local_vendor_assets']}, "
        f"local vendor references={runtime['local_vendor_references']}, "
        f"module loaders={runtime['module_loaders']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
