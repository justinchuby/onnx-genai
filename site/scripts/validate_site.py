#!/usr/bin/env python3
"""Validate generated Quartz links, project paths, and reviewed runtime URLs."""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse, urlunsplit

SITE_ORIGIN = "https://www.justinchuby.com"
LANDING_TITLE = "onnx-genai Knowledge Base"
REQUIRED_PAGE = "README.html"

ALLOWED_EXTERNAL_SCRIPTS = frozenset(
    {
        "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
        "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/copy-tex.min.js",
        "https://cdn.jsdelivr.net/npm/pixi.js@8.19.0/dist/pixi.js",
    }
)
ALLOWED_EXTERNAL_STYLES = frozenset(
    {"https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"}
)
ALLOWED_RUNTIME_URLS = ALLOWED_EXTERNAL_SCRIPTS | ALLOWED_EXTERNAL_STYLES

RUNTIME_URL = re.compile(
    r"""(?:https:)?//cdn\.jsdelivr\.net/[^"'`()<>\s]+""",
    re.IGNORECASE,
)
ORIGIN_ROOT_PATTERNS = {
    "content-index fetch": re.compile(
        r"""(?:fetch\(|new URL\()\s*["']/static/contentIndex\.json"""
    ),
    "URL constructor": re.compile(r"""new URL\(\s*["']/["']?\s*\+"""),
    "href assignment": re.compile(r"""\.href\s*=\s*["']/["']?\s*\+"""),
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.script_sources: list[str] = []
        self.stylesheets: list[str] = []
        self.inline_scripts: list[str] = []
        self.has_body = False
        self.body_base_path: str | None = None
        self.has_article = False
        self.article_text = ""
        self.title: str | None = None
        self.canonical: str | None = None
        self.meta_refresh = False
        self.noindex = False
        self._article_depth = 0
        self._article_data: list[str] = []
        self._script_data: list[str] | None = None
        self._title_data: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "body":
            self.has_body = True
            self.body_base_path = attributes.get("data-basepath")
        elif tag == "article":
            self.has_article = True
            self._article_depth += 1
        elif tag == "title" and self.title is None:
            self._title_data = []
        elif tag == "meta":
            if (attributes.get("http-equiv") or "").lower() == "refresh":
                self.meta_refresh = True
            if (attributes.get("name") or "").lower() == "robots":
                directives = {
                    value.strip().lower()
                    for value in (attributes.get("content") or "").split(",")
                }
                self.noindex = "noindex" in directives

        if tag == "link":
            rel = {value.lower() for value in (attributes.get("rel") or "").split()}
            href = attributes.get("href") or ""
            if "canonical" in rel and href:
                self.canonical = href
            if "stylesheet" in rel and href:
                self.stylesheets.append(href)

        if tag == "script":
            source = attributes.get("src")
            if source:
                self.script_sources.append(source)
            else:
                self._script_data = []

        if tag in {"a", "link"} and attributes.get("href"):
            self.urls.append(attributes["href"] or "")
        if tag in {"img", "script", "source"} and attributes.get("src"):
            self.urls.append(attributes["src"] or "")

    def handle_data(self, data: str) -> None:
        if self._article_depth:
            self._article_data.append(data)
        if self._script_data is not None:
            self._script_data.append(data)
        if self._title_data is not None:
            self._title_data.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._article_depth:
            self._article_depth -= 1
            if self._article_depth == 0:
                self.article_text = "".join(self._article_data).strip()
        elif tag == "script" and self._script_data is not None:
            self.inline_scripts.append("".join(self._script_data))
            self._script_data = None
        elif tag == "title" and self._title_data is not None:
            self.title = "".join(self._title_data).strip()
            self._title_data = None


def page_url(html: Path, public: Path, base_path: str) -> str:
    relative = html.relative_to(public).as_posix()
    return base_path if relative == "index.html" else f"{base_path}{relative.removesuffix('.html')}"


def output_candidates(public: Path, deployed_path: str, base_path: str) -> list[Path]:
    if deployed_path == base_path.rstrip("/"):
        return [public / "index.html"]
    relative = unquote(deployed_path[len(base_path) :]).lstrip("/")
    path = public / PurePosixPath(relative)
    if deployed_path.endswith("/"):
        candidates = [path / "index.html"]
    elif path.suffix:
        candidates = [path]
    else:
        candidates = [Path(f"{path}.html"), path / "index.html"]
    result = []
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(public)
        except ValueError:
            continue
        result.append(candidate.resolve())
    return result


def normalized_external_url(raw_url: str) -> str | None:
    candidate = f"https:{raw_url}" if raw_url.startswith("//") else raw_url
    parsed = urlparse(candidate)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    if parsed.username or parsed.password or parsed.port or parsed.params:
        return None
    return urlunsplit(
        ("https", parsed.hostname.lower(), parsed.path, parsed.query, parsed.fragment)
    )


def validate_external_url(
    raw_url: str, allowed: frozenset[str], context: str, errors: set[str]
) -> str | None:
    normalized = normalized_external_url(raw_url)
    if normalized is None or normalized not in allowed:
        errors.add(f"{context}: unreviewed external runtime URL: {raw_url}")
        return None
    return normalized


def validate_runtime_source(
    source: str, context: str, base_path: str, errors: set[str]
) -> set[str]:
    found: set[str] = set()
    for description, pattern in ORIGIN_ROOT_PATTERNS.items():
        if pattern.search(source):
            errors.add(f"{context}: origin-root {description} bypasses {base_path}")
    for match in RUNTIME_URL.finditer(source):
        raw_url = match.group(0).rstrip(";,")
        normalized = validate_external_url(
            raw_url, ALLOWED_RUNTIME_URLS, context, errors
        )
        if normalized:
            found.add(normalized)
    return found


def validate_landing(
    public: Path,
    base_path: str,
    documents: dict[Path, PageParser],
    inbound: dict[Path, set[Path]],
    errors: set[str],
) -> None:
    index = (public / "index.html").resolve()
    page = documents.get(index)
    if page is None:
        errors.add("missing public/index.html landing page")
        return
    if page.meta_refresh or page.noindex:
        errors.add("public/index.html must be rendered content, not a redirect/noindex stub")
    if not page.has_body or not page.has_article or len(page.article_text) < 100:
        errors.add("public/index.html must contain a substantive rendered article")
    if page.title != LANDING_TITLE:
        errors.add(f"public/index.html title is {page.title!r}, expected {LANDING_TITLE!r}")
    expected = f"{SITE_ORIGIN}{base_path}"
    if not page.canonical or urljoin(expected, page.canonical).rstrip("/") != expected.rstrip("/"):
        errors.add(f"public/index.html canonical URL must resolve to {expected}")

    readme = (public / REQUIRED_PAGE).resolve()
    readme_page = documents.get(readme)
    if readme_page is None or not readme_page.has_body or not readme_page.has_article:
        errors.add(f"{REQUIRED_PAGE} must remain a separate rendered wiki note")
    elif not inbound.get(readme):
        errors.add(f"{REQUIRED_PAGE} must be linked from another generated page")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("public", type=Path)
    parser.add_argument("base_path")
    args = parser.parse_args()

    public = args.public.resolve()
    base_path = f"/{args.base_path.strip('/')}/"
    html_files = sorted(public.rglob("*.html"))
    errors: set[str] = set()
    documents: dict[Path, PageParser] = {}
    inbound: dict[Path, set[Path]] = {}
    referenced_scripts: set[Path] = set()
    reviewed_urls: set[str] = set()
    checked_links = 0

    for html in html_files:
        page = PageParser()
        page.feed(html.read_text(encoding="utf-8"))
        resolved_html = html.resolve()
        documents[resolved_html] = page
        current_url = f"{SITE_ORIGIN}{page_url(html, public, base_path)}"

        if page.has_body and page.body_base_path != base_path.rstrip("/"):
            errors.add(
                f"{html}: body data-basepath is {page.body_base_path!r}, "
                f"expected {base_path.rstrip('/')!r}"
            )

        for raw_url in page.urls:
            if not raw_url or raw_url.startswith("#"):
                continue
            parsed = urlparse(raw_url)
            if parsed.scheme in {"data", "mailto", "tel", "javascript"}:
                continue
            if parsed.netloc and parsed.hostname != "justinchuby.github.io":
                continue
            checked_links += 1
            deployed = urlparse(urljoin(current_url, raw_url)).path
            if deployed != base_path.rstrip("/") and not deployed.startswith(base_path):
                errors.add(f"{html}: internal URL escapes {base_path}: {raw_url}")
                continue
            candidates = output_candidates(public, deployed, base_path)
            target = next((candidate for candidate in candidates if candidate.is_file()), None)
            if target is None:
                errors.add(f"{html}: missing internal target: {raw_url}")
                continue
            inbound.setdefault(target, set()).add(resolved_html)

        for raw_url in page.script_sources:
            parsed = urlparse(raw_url)
            if parsed.netloc:
                normalized = validate_external_url(
                    raw_url, ALLOWED_EXTERNAL_SCRIPTS, str(html), errors
                )
                if normalized:
                    reviewed_urls.add(normalized)
                continue
            deployed = urlparse(urljoin(current_url, raw_url)).path
            target = next(
                (
                    candidate
                    for candidate in output_candidates(public, deployed, base_path)
                    if candidate.is_file()
                ),
                None,
            )
            if target is not None and target.suffix == ".js":
                referenced_scripts.add(target)

        for raw_url in page.stylesheets:
            if urlparse(raw_url).netloc:
                normalized = validate_external_url(
                    raw_url, ALLOWED_EXTERNAL_STYLES, str(html), errors
                )
                if normalized:
                    reviewed_urls.add(normalized)

        for index, source in enumerate(page.inline_scripts):
            reviewed_urls.update(
                validate_runtime_source(
                    source, f"{html} inline script {index}", base_path, errors
                )
            )

    for script in sorted(referenced_scripts):
        reviewed_urls.update(
            validate_runtime_source(
                script.read_text(encoding="utf-8"), str(script), base_path, errors
            )
        )

    validate_landing(public, base_path, documents, inbound, errors)
    if errors:
        print("\n".join(sorted(errors)), file=sys.stderr)
        print(f"Found {len(errors)} generated site validation error(s).", file=sys.stderr)
        return 1

    print(
        f"Validated {checked_links} internal link(s)/asset(s) across "
        f"{len(html_files)} HTML page(s) at {base_path}; "
        f"{len(referenced_scripts)} referenced local script(s) exist."
    )
    print(
        "Reviewed external runtime URL(s): "
        + ", ".join(sorted(reviewed_urls))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
