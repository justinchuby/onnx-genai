#!/usr/bin/env python3
"""Validate a built Quartz site as mounted at a GitHub Pages project path."""

from __future__ import annotations

import argparse
import re
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

JSDELIVR_NPM_PATTERN = re.compile(
    r"""https?://cdn\.jsdelivr\.net/npm/"""
    r"""(?P<package>@[^/"'\s]+/[^@/"'\s]+|[^@/"'\s]+)"""
    r"""(?:@(?P<version>[^/"'\s]+))?/""",
    re.IGNORECASE,
)
EXACT_SEMVER_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")
INLINE_FETCH_DATA_PATTERN = re.compile(
    r"""\b(?:const|let|var)\s+fetchData\s*=\s*fetch\(\s*["']"""
    r"""[^"']*static/contentIndex\.json["']"""
)
RUNTIME_SURFACES = {
    "Search": re.compile(r"\.search-container"),
    "Graph": re.compile(r"\.graph-container"),
    "Explorer": re.compile(r"\.explorer-ul"),
}
MODULE_LOADER_PATTERN = re.compile(r"""\.type\s*=\s*["']module["']""")
VENDOR_ASSETS = (
    "static/vendor/d3-7.9.0.esm.js",
    "static/vendor/pixi-js-8.19.0.esm.js",
)


def mutable_cdn_references(source: str) -> int:
    return sum(
        match["version"] is None
        or EXACT_SEMVER_PATTERN.fullmatch(match["version"]) is None
        for match in JSDELIVR_NPM_PATTERN.finditer(source)
    )


def validate_runtime(
    public: Path,
    expected_base_path: str,
    documents: list[tuple[Path, LinkParser]],
    errors: set[str],
) -> dict[str, int]:
    scripts = sorted(public.rglob("*.js"))
    bundle_sources: dict[Path, str] = {}
    origin_root_count = 0
    mutable_cdn_count = 0
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        bundle_sources[script] = source
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
        if bundle not in bundle_sources:
            errors.add(f"missing required runtime bundle: {bundle}")

    postscript = bundle_sources.get(public / "postscript.js", "")
    surface_counts: dict[str, int] = {}
    for name, pattern in RUNTIME_SURFACES.items():
        surface_counts[name] = len(pattern.findall(postscript))
        if surface_counts[name] == 0:
            errors.add(f"{public / 'postscript.js'}: missing {name} runtime surface")
    module_loader_count = len(MODULE_LOADER_PATTERN.findall(postscript))
    if module_loader_count == 0:
        errors.add(f"{public / 'postscript.js'}: missing local ESM runtime loader")

    inline_script_count = 0
    inline_fetch_count = 0
    local_vendor_references = 0
    for html, document in documents:
        inline_script_count += len(document.inline_scripts)
        page_fetch_count = 0
        for source in document.inline_scripts:
            page_fetch_count += len(INLINE_FETCH_DATA_PATTERN.findall(source))
            for description, pattern in RUNTIME_ROOT_PATTERNS.items():
                matches = pattern.findall(source)
                origin_root_count += len(matches)
                if matches:
                    errors.add(f"{html}: inline {description} bypasses {expected_base_path}")
            matches = mutable_cdn_references(source)
            mutable_cdn_count += matches
            if matches:
                errors.add(f"{html}: inline mutable jsDelivr npm runtime dependency")
        inline_fetch_count += page_fetch_count
        if document.has_body and page_fetch_count == 0:
            errors.add(f"{html}: missing inline shared fetchData runtime surface")
        for source in document.script_sources:
            matches = mutable_cdn_references(source)
            if matches:
                mutable_cdn_count += matches
                errors.add(f"{html}: external script uses mutable jsDelivr npm dependency")

    for relative in VENDOR_ASSETS:
        asset = public / relative
        if not asset.is_file():
            errors.add(f"missing local Graph runtime asset: {asset}")
        deployed_url = f"{expected_base_path.rstrip('/')}/{relative}"
        references = postscript.count(deployed_url)
        local_vendor_references += references
        if references == 0:
            errors.add(f"{public / 'postscript.js'}: missing local runtime import {deployed_url}")

    return {
        "bundles": len(scripts),
        "inline_scripts": inline_script_count,
        "inline_fetch": inline_fetch_count,
        "origin_root": origin_root_count,
        "mutable_cdn": mutable_cdn_count,
        "local_vendor_assets": sum(
            (public / relative).is_file() for relative in VENDOR_ASSETS
        ),
        "local_vendor_references": local_vendor_references,
        "module_loaders": module_loader_count,
        **surface_counts,
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
