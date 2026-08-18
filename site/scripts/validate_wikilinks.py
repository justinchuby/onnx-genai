#!/usr/bin/env python3
"""Fail when a real Obsidian wikilink does not resolve inside a vault."""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

WIKILINK = re.compile(r"!?\[\[([^\[\]\n]+)\]\]")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
BLOCK_ID = re.compile(r"\^([A-Za-z0-9-]+)\s*$")
YAML_KEY = re.compile(r"^([A-Za-z0-9_-]+):(?:\s*(.*))?$")
LIST_ITEM = re.compile(r"^(?P<indent> {0,3})(?:[-+*]|\d+[.)])(?P<gap> +)")


@dataclass(frozen=True)
class Note:
    path: Path
    relative: str
    keys: frozenset[str]
    headings: frozenset[str]
    blocks: frozenset[str]
    visible_lines: tuple[str, ...]


def normalize_key(value: str) -> str:
    value = unquote(value.strip()).replace("\\", "/")
    if value.lower().endswith(".md"):
        value = value[:-3]
    value = posixpath.normpath(value.lstrip("/"))
    return value.casefold()


def normalize_heading(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[*_~`]", "", value)
    return " ".join(value.split()).casefold()


def split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    return "", text


def parse_frontmatter_keys(frontmatter: str) -> set[str]:
    values: set[str] = set()
    lines = frontmatter.splitlines()
    index = 0
    while index < len(lines):
        match = YAML_KEY.match(lines[index])
        if not match:
            index += 1
            continue
        key, raw_value = match.groups()
        if key not in {"title", "aliases", "alias"}:
            index += 1
            continue
        raw_value = (raw_value or "").strip()
        if key == "title" and raw_value:
            values.add(raw_value.strip("\"'"))
        elif raw_value.startswith("[") and raw_value.endswith("]"):
            for item in raw_value[1:-1].split(","):
                if item.strip():
                    values.add(item.strip().strip("\"'"))
        elif raw_value:
            values.add(raw_value.strip("\"'"))
        else:
            index += 1
            while index < len(lines):
                alias = re.match(r"^\s+-\s+(.+?)\s*$", lines[index])
                if not alias:
                    index -= 1
                    break
                values.add(alias.group(1).strip("\"'"))
                index += 1
        index += 1
    return values


def mask_non_content(text: str) -> tuple[str, ...]:
    text = re.sub(r"<!--.*?-->", lambda match: "\n" * match.group(0).count("\n"), text, flags=re.S)
    visible: list[str] = []
    fence_marker: tuple[str, int] | None = None
    list_content_indent: int | None = None
    for line in text.splitlines():
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker = (marker[0], len(marker))
            elif (
                marker[0] == fence_marker[0]
                and len(marker) >= fence_marker[1]
                and not fence.group(2).strip()
            ):
                fence_marker = None
            visible.append("")
            continue
        if fence_marker is not None:
            visible.append("")
            continue
        if list_item := LIST_ITEM.match(line):
            list_content_indent = len(list_item.group("indent")) + len(
                list_item.group(0)
            ) - len(list_item.group("indent"))
        elif line.strip() and list_content_indent is not None:
            leading_spaces = len(line) - len(line.lstrip(" "))
            if leading_spaces < list_content_indent:
                list_content_indent = None
        leading_spaces = len(line) - len(line.lstrip(" "))
        code_indent = 4 if list_content_indent is None else list_content_indent + 4
        if line.startswith("\t") or leading_spaces >= code_indent:
            visible.append("")
            continue
        visible.append(line)
    joined = "\n".join(visible)
    code_span = re.compile(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", re.S)

    def mask_span(match: re.Match[str]) -> str:
        return "".join("\n" if character == "\n" else " " for character in match.group(0))

    return tuple(code_span.sub(mask_span, joined).split("\n"))


def read_note(vault: Path, path: Path) -> Note:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    relative = path.relative_to(vault).as_posix()
    relative_stem = relative[:-3]
    keys = {
        normalize_key(relative_stem),
        normalize_key(Path(relative_stem).name),
        *(normalize_key(value) for value in parse_frontmatter_keys(frontmatter)),
    }
    visible_lines = mask_non_content(body)
    headings = {
        normalize_heading(match.group(1))
        for line in visible_lines
        if (match := HEADING.match(line))
    }
    blocks = {
        match.group(1).casefold()
        for line in visible_lines
        if (match := BLOCK_ID.search(line))
    }
    return Note(path, relative_stem, frozenset(keys), frozenset(headings), frozenset(blocks), visible_lines)


def split_target(raw: str) -> tuple[str, str | None]:
    target = raw.split("|", 1)[0].strip()
    if "#" in target:
        path, fragment = target.split("#", 1)
        return path.strip(), fragment.strip()
    return target, None


def validate_fragment(note: Note, fragment: str | None) -> bool:
    if not fragment:
        return True
    if fragment.startswith("^"):
        return fragment[1:].casefold() in note.blocks
    return normalize_heading(fragment) in note.headings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vault", type=Path)
    args = parser.parse_args()
    vault = args.vault.resolve()
    notes = [read_note(vault, path) for path in sorted(vault.rglob("*.md"))]

    by_key: dict[str, list[Note]] = {}
    for note in notes:
        for key in note.keys:
            by_key.setdefault(key, []).append(note)

    errors: list[str] = []
    checked = 0
    for source in notes:
        for line_number, line in enumerate(source.visible_lines, start=1):
            for match in WIKILINK.finditer(line):
                backslashes = 0
                cursor = match.start() - 1
                while cursor >= 0 and line[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 1:
                    continue
                checked += 1
                target, fragment = split_target(match.group(1))
                if not target:
                    candidates = [source]
                else:
                    candidates = by_key.get(normalize_key(target), [])
                if not candidates:
                    errors.append(
                        f"{source.path}:{line_number}: unresolved wikilink {match.group(0)}"
                    )
                    continue
                unique = {candidate.path: candidate for candidate in candidates}
                if len(unique) > 1:
                    paths = ", ".join(str(path.relative_to(vault)) for path in sorted(unique))
                    errors.append(
                        f"{source.path}:{line_number}: ambiguous wikilink {match.group(0)}: {paths}"
                    )
                    continue
                destination = next(iter(unique.values()))
                if not validate_fragment(destination, fragment):
                    errors.append(
                        f"{source.path}:{line_number}: missing fragment in "
                        f"{match.group(0)} ({destination.relative}.md)"
                    )

    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"Found {len(errors)} broken wikilink(s).", file=sys.stderr)
        return 1
    print(f"Validated {checked} wikilink(s) across {len(notes)} note(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
