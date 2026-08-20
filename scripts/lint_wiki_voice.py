#!/usr/bin/env python3
"""Fail when a wiki note talks to a reader it cannot see.

Several of these notes started life as an answer to a question, and the answer
kept its shape: a heading that quotes the question, an opening that says the
reader's observation was correct, a clarification that begins "what you called
...". The person who asked knows what all of that refers to. Every later reader
gets a page that keeps agreeing with a claim it never states, and no way to
find out what the claim was.

This flags the phrasings that only make sense to someone who was present, and
leaves the ordinary second person alone: "your adapter has to drop the
reasoning block" addresses whoever is reading, needs no prior turn, and is
normal technical prose in both languages. What is flagged is second person
attached to *something the reader supposedly said or thought*, plus headings
written as a quoted question, which is the same failure in structural form.

Run with no arguments to check the wiki, or pass paths.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Each pattern matches an attribution to the reader, not a mere mention of
# them, so that generic second-person guidance stays legal.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"你(的)?(观察|理解|说法|直觉|判断|问题|疑问|例子|想法)"),
        "attributes a statement or belief to the reader",
    ),
    (
        re.compile(r"你(说|提到|问|指出|举|给|写)的"),
        "attributes a statement to the reader",
    ),
    (
        re.compile(r"(如你所[说述言]|正如你|按你的|回答你|你之前)"),
        "refers back to something the reader said",
    ),
    (
        re.compile(r"(我们|咱们)(刚才|前面|上面)(说|聊|提)"),
        "refers back to an earlier conversation",
    ),
    (
        re.compile(
            r"\byou(r)?\s+(observation|intuition|understanding|point|question|example)\b",
            re.I,
        ),
        "attributes a statement or belief to the reader",
    ),
    (
        re.compile(r"\b(what\s+)?you\s+(said|called|mentioned|asked|noted|described)\b", re.I),
        "attributes a statement to the reader",
    ),
    (
        re.compile(r"\b(as\s+you\s+(said|noted|pointed)|you(\'re|\s+are)\s+right)\b", re.I),
        "refers back to something the reader said",
    ),
]

# A heading that is a quoted question is the question it was written to answer,
# pasted in verbatim. It tells the reader what was asked instead of what the
# section establishes.
QUOTED_QUESTION_HEADING = re.compile(r"^#{1,6}\s+.*[\"“”].*[?？]")

FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

# A document that teaches the rule has to be able to quote what it forbids.
# The escape hatch is an explicit, greppable marker rather than an exempt-path
# list, so that an exemption is visible in the diff that introduces it and
# covers only the lines it was opened for.
SUPPRESS_OFF = re.compile(r"<!--\s*voice-lint:\s*off\s*-->")
SUPPRESS_ON = re.compile(r"<!--\s*voice-lint:\s*on\s*-->")


def check_text(text: str) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    in_fence = False
    fence: str | None = None
    fence_line = 0
    suppressed = False
    for number, line in enumerate(text.splitlines(), start=1):
        if SUPPRESS_OFF.search(line):
            suppressed = True
            continue
        if SUPPRESS_ON.search(line):
            suppressed = False
            continue
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence, fence, fence_line = True, marker[0], number
            elif fence is not None and marker[0] == fence:
                in_fence, fence = False, None
            continue
        if in_fence:
            # A code block can legitimately contain a rendered transcript that
            # says "you asked"; it is a quotation of data, not the note's voice.
            continue
        if suppressed:
            continue
        if QUOTED_QUESTION_HEADING.match(line):
            findings.append((number, line.strip(), "heading is a quoted question"))
        for pattern, reason in PATTERNS:
            found = pattern.search(line)
            if found:
                findings.append((number, found.group(0), reason))

    if in_fence:
        # Per CommonMark an unclosed fence really does run to the end of the
        # document, so skipping the rest is correct -- and that is the problem.
        # A fence opened by accident, or closed with the other marker, silently
        # exempts everything below it. Saying so turns a quiet gap in coverage
        # into a visible one.
        findings.append(
            (
                fence_line,
                "```" if fence == "`" else "~~~",
                "code fence is never closed, so the rest of the file was not checked",
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.paths or sorted((Path(__file__).resolve().parents[1] / "wiki").rglob("*.md"))
    files = [path for path in paths if path.is_file() and path.suffix == ".md"]
    if not files:
        # Silently checking nothing is how a lint stops being a lint.
        raise SystemExit("no markdown files to check")

    total = 0
    for path in files:
        for number, excerpt, reason in check_text(path.read_text(encoding="utf-8")):
            print(f"{path}:{number}: {reason}: {excerpt}")
            total += 1

    if total:
        print(
            f"\n{total} passage(s) address a reader who was not there. "
            f"A wiki note has to stand on its own; see wiki/README.md, "
            f"'笔记写作规范' item 10.",
            file=sys.stderr,
        )
        return 1
    print(f"Checked {len(files)} note(s); none address an unseen reader.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
