"""Audit static route registrations; never infer behavioral conformance from presence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "8d60873dfb8619f2b5ed31adfbab07cccb40db5c"
FIXTURE = ROOT / "tests/fixtures/llama-server-routes.json"
REPORT = ROOT / "docs/LLAMA_CPP_ENDPOINT_MATRIX.md"
# Tool execution and MCP transport belong to the external harness. Keep their
# upstream contracts visible without counting this deliberate boundary as a gap.
EXTERNAL_HARNESS = {
    (method, path)
    for method in ("GET", "POST")
    for path in ("/tools", "/cors-proxy")
}


def without_comments(source: str) -> str:
    """Mask comments while preserving source lines and quoted route literals."""
    result = list(source)
    i, quoted, escaped = 0, False, False
    while i < len(source):
        c = source[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif source.startswith("//", i) or source.startswith("/*", i):
            block = source.startswith("/*", i)
            end = source.find("*/" if block else "\n", i + 2)
            if end == -1:
                if block:
                    raise ValueError("unterminated source comment; cannot audit routes")
                end = len(source)
            elif block:
                end += 2
            for j in range(i, end):
                if source[j] != "\n":
                    result[j] = " "
            i = end
            continue
        i += 1
    return "".join(result)


def close_paren(source: str, start: int) -> int:
    depth, quoted, escaped = 0, False, False
    for i in range(start, len(source)):
        c = source[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unclosed static route registration; cannot audit routes")


def local_routes(source: str) -> dict[tuple[str, str], str]:
    source = without_comments(source)
    routes = {}
    for match in re.finditer(r'\.route\s*\(\s*"([^"\\]+)"\s*,', source):
        start = source.index("(", match.start())
        body = source[match.end() : close_paren(source, start)]
        i, found = 0, False
        # Only top-level method-router calls count: a layer or handler may itself
        # contain get()/post() calls unrelated to route registration.
        while i < len(body):
            call = re.match(r"(get|post|delete|put|patch|head|options)\s*\(", body[i:])
            if call and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] == "_")):
                paren = body.index("(", i)
                end = close_paren(body, paren)
                key = (call[1].upper(), match[1])
                if key in routes:
                    raise ValueError(
                        f"duplicate local route {key}; resolve registration ambiguity"
                    )
                routes[key] = body[paren + 1 : end].strip()
                found = True
                i = end + 1
            elif body[i] == "(":
                i = close_paren(body, i) + 1
            else:
                i += 1
        if not found:
            raise ValueError(
                f"unsupported method router for {match[1]}; extend the audit grammar"
            )
    if not routes:
        raise ValueError("no static local routes found; verify the server lib.rs path")
    return routes


def upstream_routes(source: str) -> list[dict]:
    source = without_comments(source)
    result = []
    for match in re.finditer(r'ctx_http\.(get|post|del|ws)\s*\(\s*"([^"\\]+)"', source):
        method = {"del": "DELETE", "ws": "WS"}.get(match[1], match[1].upper())
        result.append(
            {
                "method": method,
                "path": match[2],
                "source_line": source.count("\n", 0, match.start()) + 1,
            }
        )
    if not result:
        raise ValueError("no upstream routes found; verify tools/server/server.cpp")
    return result


def contracts(fixture: dict, source: str) -> list[dict]:
    ours = local_routes(source)
    result, seen = [], set()
    for item in fixture["registrations"]:
        path = re.sub(r":([a-zA-Z_][a-zA-Z_0-9]*)", r"{\1}", item["path"])
        key = (item["method"], path)
        if key in seen:
            continue  # Enabled/disabled registrations still describe one contract.
        seen.add(key)
        handler = ours.get(key)
        if key in EXTERNAL_HARNESS and handler is not None:
            raise ValueError(f"external-harness contract {key} must not execute in the inference server")
        status = (
            "external harness"
            if key in EXTERNAL_HARNESS
            else "missing"
            if handler is None
            else "disabled"
            if "disabled_feature" in handler
            else "present"
        )
        result.append({**item, "path": path, "handler": handler, "status": status})
    return result


def render(fixture: dict, rows: list[dict]) -> str:
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# llama.cpp endpoint registration matrix",
        "",
        "Generated by `python3 scripts/audit_llama_server_routes.py --write`.",
        "Run the same command with `--check` to detect stale registrations.",
        "",
        f"Reference: llama.cpp `{fixture['commit']}`, `{fixture['source']}`;",
        f"{len(fixture['registrations'])} registrations / {len(rows)} distinct method-path contracts.",
        f"Local registrations: {counts['present']} present, {counts['disabled']} disabled placeholders, {counts['missing']} missing, {counts['external harness']} delegated to the external harness.",
        "",
        "**Registration presence is not behavioral conformance.** This source audit does not",
        "execute handlers, enable feature gates, compare schemas/results, or prove frontend",
        "compatibility. Every present row still needs the relevant runtime acceptance cases.",
        "Conditional upstream registrations are included and deduplicated. `WS` denotes a",
        "WebSocket protocol, not an ordinary GET handler. Root UI/static assets and dynamic",
        "Vertex AI path rewrites are outside this static registration inventory.",
        "",
        "See [compatibility behavior and remaining gaps](LLAMA_CPP_COMPATIBILITY.md) and",
        "[implementation/validation notes](LLAMA_CPP_IMPLEMENTATION.md) for semantic evidence.",
        "",
        "| Method | Path | Registration | Local handler | Upstream source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        url = f"https://github.com/ggml-org/llama.cpp/blob/{fixture['commit']}/{fixture['source']}#L{row['source_line']}"
        handler = row["handler"] or "—"
        lines.append(
            f"| {row['method']} | `{row['path']}` | {row['status']} | `{handler}` | [source]({url}) |"
        )
    lines += [
        "",
        "Tool execution and MCP transport (`/tools`, `/cors-proxy`, stdio MCP and",
        "container/SSH runtimes) are outside the inference server's scope by design.",
        "Chat tool definitions, calls, streamed deltas and tool-result messages remain",
        "in scope; the external harness owns execution and network/process access.",
        "",
        "## Acceptance work still required",
        "",
        "For each in-scope contract, validate enabled and disabled gates, malformed and valid input,",
        "schema/defaults, actual model execution, status/error bodies, aliases, and",
        "stream framing/cancellation where applicable. Test source presence alone does not",
        "satisfy these cases. Cross-model and backend-specific behavior must use declared",
        "capabilities; a passing narrow fixture does not establish general parity.",
        "",
        "The missing realtime/SystemOne protocols and disabled model download/deletion",
        "remain implementation work. Existing handlers also retain the explicit source,",
        "device-state and multimodal gaps documented in the compatibility ledger.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument(
        "--upstream",
        type=Path,
        help="verify pinned fixture against a clean upstream checkout (no network)",
    )
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text())
    if fixture["commit"] != UPSTREAM_COMMIT:
        parser.error(
            "fixture revision differs from the bundled frontend reference; update the audit deliberately"
        )
    if args.upstream:
        revision = subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != fixture["commit"]:
            parser.error(
                f"upstream checkout is {revision}; select pinned {fixture['commit']}"
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(args.upstream),
                "diff",
                "--exit-code",
                "HEAD",
                "--",
                fixture["source"],
            ],
            check=True,
        )
        actual = upstream_routes((args.upstream / fixture["source"]).read_text())
        if actual != fixture["registrations"]:
            parser.error(
                "upstream route fixture differs from pinned source; regenerate and review all contracts"
            )
    rows = contracts(
        fixture, (ROOT / "crates/onnx-genai-server/src/lib.rs").read_text()
    )
    report = render(fixture, rows)
    if args.write:
        REPORT.write_text(report)
    elif not REPORT.exists() or REPORT.read_text() != report:
        parser.error(
            "endpoint matrix is stale; run scripts/audit_llama_server_routes.py --write and review the changes"
        )
    print(
        f"Audited {len(rows)} route contracts: {dict(Counter(r['status'] for r in rows))}; semantic conformance remains unproven."
    )


if __name__ == "__main__":
    main()
