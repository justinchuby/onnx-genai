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
        self.stylesheet_hrefs: list[str] = []
        self.has_body = False
        self.body_base_path: str | None = None
        self.title: str | None = None
        self.canonical: str | None = None
        self.meta_refresh: str | None = None
        self.robots_noindex = False
        self.has_article = False
        self.article_text = ""
        self._script_body: list[str] | None = None
        self._title_body: list[str] | None = None
        self._article_depth = 0
        self._article_body: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "body":
            self.has_body = True
            self.body_base_path = attributes.get("data-basepath")
        if tag == "title" and self.title is None:
            self._title_body = []
        if tag == "article":
            self.has_article = True
            self._article_depth += 1
        if tag == "meta":
            equiv = (attributes.get("http-equiv") or "").lower()
            if equiv == "refresh":
                self.meta_refresh = attributes.get("content") or ""
            if (attributes.get("name") or "").lower() == "robots":
                directives = {
                    value.strip().lower() for value in (attributes.get("content") or "").split(",")
                }
                if "noindex" in directives:
                    self.robots_noindex = True
        if tag == "link":
            rels = {value.lower() for value in (attributes.get("rel") or "").split()}
            href = attributes.get("href") or ""
            if "canonical" in rels and href:
                self.canonical = href
            if "stylesheet" in rels and href:
                self.stylesheet_hrefs.append(href)
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

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag == "article":
            self._article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._script_body is not None:
            self._script_body.append(data)
        if self._title_body is not None:
            self._title_body.append(data)
        if self._article_depth > 0:
            self._article_body.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_body is not None:
            self.inline_scripts.append("".join(self._script_body))
            self._script_body = None
        if tag == "title" and self._title_body is not None:
            self.title = "".join(self._title_body).strip()
            self._title_body = None
        if tag == "article" and self._article_depth > 0:
            self._article_depth -= 1
            if self._article_depth == 0:
                self.article_text = "".join(self._article_body).strip()


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

# jsDelivr executable-CDN policy.
#
# jsDelivr serves the same mutable package namespace from several hostnames
# and additionally exposes it as ESM through `esm.run`, so a host-literal
# policy that only knows `cdn.jsdelivr.net` is trivially bypassed. Hostnames
# are therefore canonicalized (lowercased, trailing DNS root dot stripped)
# before policy matching, and every canonical jsDelivr executable host is in
# scope. Executable references may be absolute or scheme-relative; both
# browser-equivalent forms are normalized for policy, but raw or decoded dot
# segments are rejected before URL normalization can erase an attempted
# package/path escape.
JSDELIVR_EXECUTABLE_HOSTS = frozenset(
    {
        "cdn.jsdelivr.net",
        "fastly.jsdelivr.net",
        "gcore.jsdelivr.net",
        "testingcf.jsdelivr.net",
        "originfastly.jsdelivr.net",
        "quantil.jsdelivr.net",
        "esm.run",
    }
)

# The only host/namespace pair that may ever carry an allowlisted executable
# reference: the canonical, lowercase, dot-free host and the npm namespace the
# pinned `latex` Quartz plugin actually emits. `/gh/`, `/wp/`, `esm.run` bare
# package specifiers, and every alternate jsDelivr mirror are outside it and
# never inherit allowlisting.
CANONICAL_JSDELIVR_HOST = "cdn.jsdelivr.net"
CANONICAL_JSDELIVR_NAMESPACE = "/npm/"

CDN_REFERENCE = re.compile(
    r"""(?:https?:)?//(?P<host>[A-Za-z0-9._-]+)(?P<path>/[^"'`()<>\s]*)?""",
    re.IGNORECASE,
)
EXACT_SEMVER_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")

# Local bundling means Graph's d3/pixi.js have zero legitimate jsDelivr
# references in production. The only reviewed runtime exception is the exact
# KaTeX copy-tex asset emitted by Quartz's latex plugin, on the canonical host
# and namespace above.
JSDELIVR_ALLOWLIST: dict[tuple[str, str], frozenset[str]] = {
    ("katex", "0.16.11"): frozenset({"dist/contrib/copy-tex.min.js"}),
}

# External `<link rel="stylesheet">` hrefs are a separate, non-executable
# runtime surface. They are still third-party loads, so they are governed by
# their own exact-URL allowlist rather than being silently out of scope.
# Only the stylesheet the pinned `latex` plugin actually emits is approved.
STYLESHEET_ALLOWLIST = frozenset(
    {
        "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
    }
)


def canonical_host(host: str) -> str:
    """Lowercase a hostname and drop the trailing DNS root dot(s).

    `CDN.JsDelivr.NET.` and `cdn.jsdelivr.net` resolve to the same origin in a
    browser, so both must be recognized by policy. Recognition is deliberately
    broader than allowlisting: only the exact canonical spelling is eligible
    for the reviewed KaTeX exception (see `_is_allowlisted_reference`).
    """
    return host.lower().rstrip(".")


def _strip_query_fragment(value: str) -> str:
    return re.split(r"[?#]", value, maxsplit=1)[0]


def _decoded_dot_segment(segment: str) -> bool:
    candidate = segment
    for _ in range(4):
        if candidate in {".", ".."}:
            return True
        decoded = unquote(candidate)
        if decoded == candidate:
            return False
        candidate = decoded
    return candidate in {".", ".."}


def _parse_jsdelivr_tail(tail: str) -> tuple[str | None, str | None, str | None, bool]:
    has_query_fragment = _strip_query_fragment(tail) != tail
    path = _strip_query_fragment(tail)
    segments = path.split("/")
    if any(_decoded_dot_segment(segment) for segment in segments):
        return None, None, None, True
    if has_query_fragment:
        return None, None, None, True
    if not segments or not segments[0]:
        return None, None, None, False

    if segments[0].startswith("@"):
        if len(segments) < 2 or not segments[1]:
            return None, None, None, False
        package_base = f"{segments[0]}/{segments[1].split('@', 1)[0]}"
        version = None
        if "@" in segments[1]:
            _, version = segments[1].rsplit("@", 1)
        asset_segments = segments[2:]
        return package_base, version, "/".join(asset_segments), False

    package_segment = segments[0]
    package = package_segment
    version = None
    if "@" in package_segment:
        package, version = package_segment.rsplit("@", 1)
    asset_path = "/".join(segments[1:])
    return package, version, asset_path, False


def _is_allowlisted_reference(raw_host: str, path: str) -> bool:
    """Whether one executable jsDelivr reference is the reviewed exception.

    Allowlisting is intentionally narrower than detection: it requires the
    exact canonical host spelling (no uppercase or trailing-dot variant), the
    canonical `/npm/` namespace, an exact immutable semver, and an exact
    reviewed asset path. Alternate jsDelivr mirrors and `esm.run` therefore
    never inherit the exception even for byte-identical package paths.
    """
    if raw_host != CANONICAL_JSDELIVR_HOST:
        return False
    if not path.startswith(CANONICAL_JSDELIVR_NAMESPACE):
        return False
    package, version, asset_path, has_dot_segment = _parse_jsdelivr_tail(
        path[len(CANONICAL_JSDELIVR_NAMESPACE) :]
    )
    if has_dot_segment:
        return False
    return (
        package is not None
        and version is not None
        and EXACT_SEMVER_PATTERN.fullmatch(version) is not None
        and asset_path in JSDELIVR_ALLOWLIST.get((package, version), frozenset())
    )


def mutable_cdn_references(source: str) -> int:
    count = 0
    for match in CDN_REFERENCE.finditer(source):
        raw_host = match["host"]
        if canonical_host(raw_host) not in JSDELIVR_EXECUTABLE_HOSTS:
            continue
        if not _is_allowlisted_reference(raw_host, match["path"] or ""):
            count += 1
    return count


def unapproved_stylesheet_references(hrefs: list[str]) -> list[str]:
    """Return every off-origin `<link rel="stylesheet">` href that is not
    exactly allowlisted.

    Same-origin (relative or `justinchuby.github.io`) stylesheets are the
    generated site's own CSS and are covered by the internal link/asset
    existence checks.
    """
    unapproved: list[str] = []
    for href in hrefs:
        candidate = href.strip()
        if not candidate:
            continue
        parsed = urlparse(candidate if not candidate.startswith("//") else f"https:{candidate}")
        if not parsed.netloc:
            continue
        if canonical_host(parsed.hostname or "") == "justinchuby.github.io":
            continue
        if candidate not in STYLESHEET_ALLOWLIST:
            unapproved.append(candidate)
    return unapproved


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
    site/quartz/scripts/audit-runtime-resources.mjs starts at generated HTML
    scripts/inline bodies, follows reachable imports/load edges, parses only
    that reachable script graph into real ASTs, prunes provably unreachable
    branches, and only counts a signature as "functional" when something
    downstream actually depends on it. A comment, unused string, unreferenced
    decoy bundle, or dead `if (false) {...}` placeholder cannot satisfy any of
    these checks. Graph's local vendor assets are cross-checked here against
    the sha256 recorded in the build-emitted `static/vendor/manifest.json`, so
    the manifest itself cannot be hand-edited without also matching real
    deployed bytes.
    """
    manifest_path = public / VENDOR_MANIFEST_RELATIVE
    inline_payload = json.dumps(
        {
            "documents": [
                {
                    "html": html.relative_to(public).as_posix(),
                    "scriptSources": document.script_sources,
                    "inlineScripts": [
                        {
                            "id": f"{html.relative_to(public).as_posix()}#{index}",
                            "source": source,
                        }
                        for index, source in enumerate(document.inline_scripts)
                    ],
                }
                for html, document in documents
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

    for entry in audit.get("resourceGraph", {}).get("errors", []):
        errors.add(f"runtime resource graph: {entry}")

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
            errors.add(f"{script}: mutable jsDelivr CDN runtime dependency")

    required_bundles = (public / "prescript.js", public / "postscript.js")
    for bundle in required_bundles:
        if bundle not in bundle_present:
            errors.add(f"missing required runtime bundle: {bundle}")

    inline_script_count = 0
    allowlisted_stylesheets = 0
    unapproved_stylesheets = 0
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
                errors.add(f"{html}: inline mutable jsDelivr CDN runtime dependency")
        for source in document.script_sources:
            matches = mutable_cdn_references(source)
            if matches:
                mutable_cdn_count += matches
                errors.add(f"{html}: external script uses mutable jsDelivr CDN dependency")
        for href in unapproved_stylesheet_references(document.stylesheet_hrefs):
            unapproved_stylesheets += 1
            errors.add(f"{html}: third-party stylesheet is not allowlisted: {href}")
        allowlisted_stylesheets += sum(
            1 for href in document.stylesheet_hrefs if href.strip() in STYLESHEET_ALLOWLIST
        )

    audit = run_resource_audit(public, expected_base_path, documents, errors)
    surfaces = audit.get("surfaces", {}) if audit else {}
    vendor_edges = audit.get("vendorEdges", {}) if audit else {}
    manifest_assets = audit.get("manifest", {}).get("assets", []) if audit else []
    fetch_entries = audit.get("sharedFetchData", []) if audit else []

    return {
        "bundles": len(scripts),
        "reachable_bundles": audit.get("bundleCount", 0) if audit else 0,
        "ignored_bundles": len(audit.get("resourceGraph", {}).get("ignoredScripts", []))
        if audit
        else 0,
        "inline_scripts": inline_script_count,
        "inline_fetch": sum(entry.get("functional", 0) for entry in fetch_entries),
        "origin_root": origin_root_count,
        "mutable_cdn": mutable_cdn_count,
        "allowlisted_stylesheets": allowlisted_stylesheets,
        "unapproved_stylesheets": unapproved_stylesheets,
        "local_vendor_assets": sum(
            1 for asset in manifest_assets if (public / asset["file"]).is_file()
        ),
        "local_vendor_references": sum(edge.get("edges", 0) for edge in vendor_edges.values()),
        "module_loaders": len(audit.get("loaderNames", [])) if audit else 0,
        **{name: counts.get("functional", 0) for name, counts in surfaces.items()},
    }


LANDING_MIN_ARTICLE_CHARS = 200


def validate_landing(
    public: Path,
    base_path: str,
    documents: dict[Path, LinkParser],
    inbound_links: dict[Path, set[Path]],
    landing_title: str | None,
    required_pages: list[str],
    errors: set[str],
) -> dict[str, object]:
    """Assert that the generated root page is real rendered content.

    A Quartz permalink/alias stub emits `public/index.html` as a `noindex`
    meta-refresh document with no body: the site "has an index" while every
    reader and crawler is bounced somewhere else. These assertions make that
    regression a build failure rather than something only a one-off shell
    check would notice: the root page must render a body and a non-trivial
    article, carry the expected title, declare a canonical URL that is the
    project root beneath the deployed base path, and must not redirect or
    de-index itself. `required_pages` additionally keeps sibling notes (such
    as the wiki README/index-of-notes page) emitted, rendered, and reachable
    by at least one inbound internal link.
    """
    summary: dict[str, object] = {
        "landing_title": None,
        "landing_article_chars": 0,
        "required_pages": len(required_pages),
    }
    index_path = (public / "index.html").resolve()
    document = documents.get(index_path)
    if landing_title is not None:
        if document is None:
            errors.add("missing public/index.html landing page")
        else:
            summary["landing_title"] = document.title
            summary["landing_article_chars"] = len(document.article_text)
            if document.meta_refresh is not None:
                errors.add(
                    "public/index.html is a meta-refresh redirect stub "
                    f"({document.meta_refresh!r}), not a rendered landing page"
                )
            if document.robots_noindex:
                errors.add("public/index.html landing page is marked noindex")
            if not document.has_body:
                errors.add("public/index.html landing page has no rendered body")
            if not document.has_article:
                errors.add("public/index.html landing page has no rendered article")
            elif len(document.article_text) < LANDING_MIN_ARTICLE_CHARS:
                errors.add(
                    "public/index.html landing article has only "
                    f"{len(document.article_text)} rendered character(s), "
                    f"expected at least {LANDING_MIN_ARTICLE_CHARS}"
                )
            if document.title != landing_title:
                errors.add(
                    f"public/index.html landing title is {document.title!r}, "
                    f"expected {landing_title!r}"
                )
            expected_canonical = f"https://justinchuby.github.io{base_path}"
            if not document.canonical:
                errors.add("public/index.html landing page declares no canonical URL")
            else:
                resolved = urljoin(expected_canonical, document.canonical)
                if resolved.rstrip("/") != expected_canonical.rstrip("/"):
                    errors.add(
                        f"public/index.html canonical URL is {resolved!r}, "
                        f"expected {expected_canonical!r}"
                    )

    for relative in required_pages:
        target = (public / PurePosixPath(relative)).resolve()
        try:
            target.relative_to(public)
        except ValueError:
            errors.add(f"required page escapes public/: {relative}")
            continue
        if not target.is_file():
            errors.add(f"missing required generated page: {relative}")
            continue
        required_document = documents.get(target)
        if required_document is None or not required_document.has_body:
            errors.add(f"required generated page has no rendered body: {relative}")
        sources = {source for source in inbound_links.get(target, set()) if source != target}
        if not sources:
            errors.add(f"required generated page is not linked from any other page: {relative}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("public", type=Path)
    parser.add_argument("base_path")
    parser.add_argument(
        "--landing-title",
        default=None,
        help=(
            "Expected <title> of the generated root landing page. Supplying it "
            "enables strict root landing semantics (rendered body/article, "
            "canonical project root URL, no meta-refresh or noindex stub)."
        ),
    )
    parser.add_argument(
        "--require-page",
        action="append",
        default=[],
        dest="required_pages",
        metavar="RELATIVE_HTML",
        help=(
            "Generated page (relative to the output directory) that must exist, "
            "render a body, and be linked from at least one other page. May be "
            "repeated."
        ),
    )
    args = parser.parse_args()
    public = args.public.resolve()
    base_path = f"/{args.base_path.strip('/')}/"
    html_files = sorted(public.rglob("*.html"))
    errors: set[str] = set()
    checked = 0
    body_pages = 0
    documents: list[tuple[Path, LinkParser]] = []
    documents_by_path: dict[Path, LinkParser] = {}
    inbound_links: dict[Path, set[Path]] = {}

    if not (public / "index.html").is_file():
        errors.add("missing public/index.html landing page")

    for html in html_files:
        document = LinkParser()
        document.feed(html.read_text(encoding="utf-8"))
        documents.append((html, document))
        documents_by_path[html.resolve()] = document
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
            if parsed.netloc and parsed.hostname != "justinchuby.github.io":
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
            existing = [candidate for candidate in candidates if candidate.is_file()]
            if not existing:
                errors.add(f"{html}: missing internal target: {raw_url}")
                continue
            inbound_links.setdefault(existing[0].resolve(), set()).add(html.resolve())

    if body_pages == 0:
        errors.add("no generated page has a body for runtime navigation")
    landing = validate_landing(
        public,
        base_path,
        documents_by_path,
        inbound_links,
        args.landing_title,
        args.required_pages,
        errors,
    )
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
        f"allowlisted third-party stylesheets={runtime['allowlisted_stylesheets']}, "
        f"unapproved third-party stylesheets={runtime['unapproved_stylesheets']}, "
        f"local vendor assets={runtime['local_vendor_assets']}, "
        f"local vendor references={runtime['local_vendor_references']}, "
        f"module loaders={runtime['module_loaders']}, "
        f"reachable bundles={runtime['reachable_bundles']}, "
        f"ignored unreferenced bundles={runtime['ignored_bundles']}."
    )
    if args.landing_title is not None:
        print(
            f"Landing audit: root title={landing['landing_title']!r}, "
            f"rendered article characters={landing['landing_article_chars']}, "
            f"required linked page(s)={landing['required_pages']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
