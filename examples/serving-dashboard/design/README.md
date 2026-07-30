<!-- Copyright (c) Microsoft Corporation. -->

# Design reference — serving dashboard

**This directory does not ship.** It exists so the developers building the demo
can *see* the design contract rendered in a browser instead of inferring it from
prose.

| File | What it is |
|---|---|
| `skeleton.html` | Static, JS-free layout and treatment reference. Open it directly in a browser — no server needed. |
| `../styles/tokens.css` | **The real thing.** The shipping page imports this file. Designer-owned; nobody else edits it. |

## What to copy and what not to copy

**Copy:** the CSS patterns, the DOM structure, the class-naming shape, the
unavailable-state treatments, the failure-state anatomy.

**Do not copy:** any number in `skeleton.html`. Every value there is placeholder
text illustrating a *treatment*. The shipping page has no hardcoded values —
that is acceptance criterion AC6, and it is auditable by grep.

## The one thing to understand before writing code

A measured zero and a fabricated zero must never look the same.

- `prefix cache hit rate: 0.0 %` is a **real measurement**. The cache genuinely
  did not hit. It renders at full contrast, as a number.
- `KV utilization: —` is an **absence**. The server returns a documented zero it
  cannot actually measure. It renders as an em-dash with a dashed underline and
  a hover explaining why.

`skeleton.html` shows both, adjacent, in the hero strip. That pairing is the
whole design.

## Full specification

`demo-ux.md` in the designer's artifact directory — page architecture, the
complete token set, the panel `mount()` contract, the three scenario
visualizations, the unavailable-data design language, and the accessibility
token set.
