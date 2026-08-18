#!/usr/bin/env python3
"""Rewrite generated links to repository files as GitHub source links."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import quote, urlparse

ANCHOR = re.compile(r"<a\b(?P<attrs>[^>]*\bdata-slug=\"(?P<slug>[^\"]+)\"[^>]*)>")
BODY_SLUG = re.compile(r"<body\b[^>]*\bdata-slug=\"([^\"]+)\"")
HREF = re.compile(r'\bhref="([^"]*)"')
CLASS = re.compile(r'\bclass="([^"]*)"')
FOOTER_YEAR = re.compile(
    r'(Created with <a href="https://quartz\.jzhao\.xyz/">Quartz</a>) © \d{4}'
)


def repository_target(repository: Path, slug: str) -> Path | None:
    clean_slug = html.unescape(slug).strip("/")
    if not clean_slug:
        return None
    candidates = [repository / clean_slug, repository / f"{clean_slug}.md"]
    if clean_slug.endswith("/index"):
        candidates.append(repository / clean_slug.removesuffix("/index"))
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(repository)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    return None


def github_url(repository: Path, target: Path, fragment: str) -> str:
    relative = target.resolve().relative_to(repository).as_posix()
    kind = "tree" if target.is_dir() else "blob"
    url = f"https://github.com/justinchuby/onnx-genai/{kind}/main/{quote(relative, safe='/')}"
    if fragment:
        url += f"#{quote(fragment, safe='-._~')}"
    return url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("public", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    public = args.public.resolve()
    repository = args.repository.resolve()
    html_files = sorted(public.rglob("*.html"))
    site_slugs: set[str] = set()
    documents: dict[Path, str] = {}

    for path in html_files:
        document = path.read_text(encoding="utf-8")
        documents[path] = document
        if match := BODY_SLUG.search(document):
            site_slugs.add(html.unescape(match.group(1)))

    rewritten = 0
    for path, document in documents.items():

        def replace_anchor(match: re.Match[str]) -> str:
            nonlocal rewritten
            slug = html.unescape(match.group("slug"))
            attrs = match.group("attrs")
            href_match = HREF.search(attrs)
            href = html.unescape(href_match.group(1)) if href_match else ""
            repository_reference = "/./../" in href
            if slug in site_slugs and not repository_reference:
                return match.group(0)
            target = repository_target(repository, slug)
            if target is None:
                return match.group(0)
            fragment = urlparse(html.unescape(href_match.group(1))).fragment if href_match else ""
            url = html.escape(github_url(repository, target, fragment), quote=True)
            if href_match:
                attrs = HREF.sub(f'href="{url}"', attrs, count=1)
            else:
                attrs = f' href="{url}"{attrs}'
            attrs = re.sub(r'\sdata-slug="[^"]*"', "", attrs, count=1)

            def external_class(class_match: re.Match[str]) -> str:
                classes = class_match.group(1).split()
                classes = ["external" if value == "internal" else value for value in classes]
                if "external" not in classes:
                    classes.append("external")
                return f'class="{" ".join(classes)}"'

            if CLASS.search(attrs):
                attrs = CLASS.sub(external_class, attrs, count=1)
            else:
                attrs += ' class="external"'
            rewritten += 1
            return f"<a{attrs}>"

        updated = ANCHOR.sub(replace_anchor, document)
        updated = FOOTER_YEAR.sub(r"\1 · onnx-genai", updated)
        if updated != document:
            path.write_text(updated, encoding="utf-8")

    print(f"Rewrote {rewritten} repository source link(s) to GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
