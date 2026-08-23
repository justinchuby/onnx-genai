#!/usr/bin/env python3
"""Give a package's sequence axes the distinct symbols its graphs declare.

Some published packages spell every sequence-like axis `sequence`:

    request.past_key_values.N.key:   [batch, <heads>, sequence, <head_dim>]
    target.present.N.key:            [batch, <heads>, sequence, <head_dim>]
    assistant.shared_kv.<g>.key:     [batch, <heads>, sequence, <head_dim>]
    assistant.attention_mask:        [batch, sequence]
    request.input_ids:               [batch, sequence]

The exported graphs do not. They distinguish, by name:

    target    input_ids               [batch, sequence_len]
    target    past_key_values.N.key   [batch, 1, past_sequence_len, <d>]
    target    present.N.key           [batch, 1, past_sequence_len + sequence_len, <d>]
    assistant inputs_embeds           [batch, q_len, <fused>]
    assistant attention_mask          [batch, kv_len_full_attention]
    assistant shared_kv.full_*        [batch, 1, kv_len_full_attention, <d>]
    assistant shared_kv.sliding_*     [batch, 1, kv_len_sliding_attention, <d>]

Collapsing them costs two distinct, silent failures:

1. A runtime that binds a package's symbols from the tensors it is handed
   rejects any pass whose cache is not exactly as long as its prompt --
   an empty cache binds `sequence` to 0 after `input_ids` bound it to 6.

2. A chained speculative proposal narrows every proposer port that shares the
   fused input's position symbol down to the step's single position. A
   borrowed-KV drafter whose cache axis is *also* called `sequence` therefore
   drafts against one position of the cache instead of all of it -- a proposal
   that is well formed, plausibly tallied, and accepted at a rate of zero.

This renames those axes and only those axes, wherever the document declares
them: workflow input contracts, the state cells that mirror them, and the
component port contracts the steps bind them into. Edits are line-based, so
authored comments, key order and inline flow style survive byte for byte.

Usage:

    python scripts/name_sequence_axes.py <package>/inference_metadata.yaml ...
"""

from __future__ import annotations

import argparse
import pathlib
import re

# `<indent><name>:` -- a section, a component, an input, a cell, or a port whose
# body is a block mapping on the following lines.
ENTRY = re.compile(r"^(\s+)(\S[^:]*):\s*$")
# The same key, with its body inline: `<indent><name>: {dtype: ..., shape: [...]}`.
INLINE_ENTRY = re.compile(r"^(\s+)(\S[^:{]*):\s+\{")
# `shape: [...]` anywhere on the line: a contract may be a block mapping with
# `shape:` on its own line, or an inline flow mapping
# (`contract: {dtype: ..., shape: [...], ...}`). Both spellings occur in
# published packages and in this repository's own fixtures.
SHAPE = re.compile(r"(shape: \[)([^\]]*)(\])")

# Depth of the keys directly under `pipeline.workflow`, and of a component name.
SECTION_DEPTH = 4
COMPONENT_DEPTH = 6

# Entry-name prefix -> (axis, symbol). Applied wherever the entry appears.
CACHE_AXES: dict[str, tuple[int, str]] = {
    # The cache a pass starts from is not the sequence it consumes.
    "past_key_values.": (2, "past_sequence"),
    # The cache a pass ends with is neither of the two.
    "present.": (2, "total_sequence"),
}

# The same, for ports of a component that *borrows* another's cache: its own
# query axis and the cache axis it reads are different extents, and a chained
# proposal's per-position narrowing depends on the package saying so.
#
# Deliberately scoped to `components.<borrower>.ports` and applied nowhere else.
# The narrowing decision reads the *component port* contract, and the two
# namespaces are independent -- the repository's own canonical
# `gemma4_chained` fixture declares `request.shared_kv...` as `sequence` while
# the assistant's port declares `kv_sequence`. Applying an unqualified
# `attention_mask` rule at workflow-input scope would also rename the target's
# mask, which is a different extent again.
BORROWED_AXES: dict[str, tuple[int, str]] = {
    "shared_kv.full_attention.": (2, "full_kv_sequence"),
    "shared_kv.sliding_attention.": (2, "sliding_kv_sequence"),
    # A borrowed-cache reader's mask spans the cache it attends over, which the
    # graph names with the same symbol as that cache.
    "attention_mask": (1, "full_kv_sequence"),
}


def borrowing_components(text: str) -> set[str]:
    """Components declared as read-only readers of another's state.

    Read out of `serving.state_service.groups[*].ports.<component>` rather than
    assumed: a package with no borrowed state has none, and one with a
    differently named proposer must not need this script edited.
    """
    import yaml

    document = yaml.safe_load(text)
    workflow = document.get("pipeline", {}).get("workflow", {})
    serving = workflow.get("serving") or {}
    groups = (serving.get("state_service") or {}).get("groups") or {}
    borrowers = set()
    for group in groups.values():
        for component, aliases in (group.get("ports") or {}).items():
            for alias in aliases.values():
                if alias.get("access") == "read_only":
                    borrowers.add(component)
    return borrowers


def rewrite(text: str, old: str) -> tuple[str, int]:
    borrowers = borrowing_components(text)
    lines = text.splitlines(keepends=True)
    section: str | None = None
    component: str | None = None
    # The entry whose shapes are being rewritten, and how deep its key sat. A
    # nested structural key (`contract:`) belongs to that entry and must not
    # clear it; a sibling at the same depth or shallower ends it.
    current: tuple[int, tuple[int, str]] | None = None
    changed = 0

    def rule_for(entry: str) -> tuple[int, str] | None:
        rules = dict(CACHE_AXES)
        if component in borrowers:
            rules.update(BORROWED_AXES)
        # A workflow input is `request.<port>`; a state cell and a component
        # port are the bare `<port>`. Both name the same tensor.
        name = entry.removeprefix("request.")
        return next(
            (rule for prefix, rule in rules.items() if name.startswith(prefix)),
            None,
        )

    for index, line in enumerate(lines):
        match = ENTRY.match(line)
        if match:
            depth, entry = len(match.group(1)), match.group(2)
            if depth == SECTION_DEPTH:
                section, component, current = entry, None, None
                continue
            if section == "components" and depth == COMPONENT_DEPTH:
                component, current = entry, None
                continue
            axis = rule_for(entry)
            if axis is not None:
                current = (depth, axis)
            elif current is not None and depth <= current[0]:
                current = None
            continue

        # An entry whose body is inline carries its own shape on this line, and
        # ends any block entry at its depth or shallower.
        inline = INLINE_ENTRY.match(line)
        axis = None
        if inline:
            depth, entry = len(inline.group(1)), inline.group(2)
            axis = rule_for(entry)
            if current is not None and depth <= current[0]:
                current = None
        if axis is None:
            if current is None:
                continue
            axis = current[1]
        shape = SHAPE.search(line)
        if not shape:
            continue
        position, symbol = axis
        dims = [dim.strip() for dim in shape.group(2).split(",")]
        if position >= len(dims) or dims[position] != old:
            continue
        dims[position] = symbol
        lines[index] = (
            line[: shape.start()]
            + f"{shape.group(1)}{', '.join(dims)}{shape.group(3)}"
            + line[shape.end() :]
        )
        changed += 1
    return "".join(lines), changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=pathlib.Path, nargs="+")
    parser.add_argument("--from-symbol", default="sequence")
    args = parser.parse_args()

    for path in args.metadata:
        text = path.read_text(encoding="utf-8")
        rewritten, changed = rewrite(text, args.from_symbol)
        if changed == 0:
            print(f"{path}: no sequence axis declared '{args.from_symbol}'; unchanged")
            continue
        path.write_text(rewritten, encoding="utf-8")
        print(f"{path}: {changed} axes renamed off '{args.from_symbol}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
