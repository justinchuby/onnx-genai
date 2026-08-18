---
title: Wiki
aliases:
  - Knowledge Base
tags:
  - wiki
  - index
status: maintained
---

# onnx-genai Wiki

This directory is an Obsidian-compatible knowledge base for explanatory notes,
learning paths, and links between implementation concepts. It does not replace
the specifications, measured evidence, or accepted designs in `docs/`.

> [!important] Source precedence
> When a wiki note disagrees with formal documentation or code, use this order:
> 1. Current code and reproducible measurements
> 2. Authoritative documents under `docs/`
> 3. Accepted design decisions
> 4. Explanatory wiki notes

## Maps of content

- **Memory:** [[memory/Memory Management for Beginners]]

## Note conventions

Every note should:

1. Use an English filename and `title` so links remain stable across languages.
2. Include YAML frontmatter with `title`, `aliases`, `tags`, and `status`.
3. Begin with a short statement of the question the note answers.
4. Use `[[wikilinks]]` instead of duplicating explanations across notes.
5. Use Obsidian callouts for invariants, warnings, examples, and context.
6. End with links to formal sources under `docs/` or relevant code.
7. Clearly label proposed behavior; never present a target design as implemented.

Notes may be written in the language that best serves their audience. English
titles and paths are required even when the body is written in another language.
