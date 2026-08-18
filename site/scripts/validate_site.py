#!/usr/bin/env python3
"""Validate a built Quartz site as mounted at a GitHub Pages project path."""

from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"a", "link"} and attributes.get("href"):
            self.urls.append(attributes["href"] or "")
        if tag in {"img", "script", "source"} and attributes.get("src"):
            self.urls.append(attributes["src"] or "")


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

    if not (public / "index.html").is_file():
        errors.add("missing public/index.html landing page")

    for html in html_files:
        document = LinkParser()
        document.feed(html.read_text(encoding="utf-8"))
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

    if errors:
        print("\n".join(sorted(errors)), file=sys.stderr)
        print(f"Found {len(errors)} generated site link error(s).", file=sys.stderr)
        return 1
    print(
        f"Validated {checked} generated internal link(s)/asset(s) across "
        f"{len(html_files)} HTML page(s) at {base_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
