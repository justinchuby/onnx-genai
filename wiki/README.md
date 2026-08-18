---
title: Wiki
permalink: index
aliases:
  - Knowledge Base
tags:
  - wiki
  - index
status: maintained
created: 2026-08-17
updated: 2026-08-17
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

- **Start here:** [[start/Repository Map]]
- **Architecture:** [[architecture/Crate Architecture]]
- **Runtime flow:** [[architecture/Inference Request Lifecycle]]
- **Execution:** [[execution/Execution Backends]]
- **EP contract:** [[execution/Execution Provider Contract]]
- **CPU EP:** [[execution/CPU Execution Provider]]
- **CUDA EP:** [[execution/CUDA Execution Provider]]
- **Plugin EPs:** [[execution/Plugin Execution Providers]]
- **Memory:** [[memory/Memory Management for Beginners]]
- **Tracing:** [[observability/Tracing and Profiling]]
- **Performance engineering:** [[performance/Performance Engineering Playbook]]
- **API design:** [[api/API Design Principles]]
- **Contracts:** [[contracts/Runtime Contracts]]
- **Metadata:** [[metadata/Metadata Driven Runtime]]
- **Model packages:** [[metadata/Model Packages and Variants]]
- **Documentation:** [[start/Documentation Guide]]
- **Wiki maintenance:** [[meta/Using this Wiki]]

## Note conventions

Every note should:

1. Use an English filename and `title` so links remain stable across languages.
2. Include YAML frontmatter with `title`, `aliases`, `tags`, `status`,
   `created`, and `updated`.
3. Begin with a short statement of the question the note answers.
4. Answer one primary question and usually remain readable in roughly 5–10 minutes.
5. Keep a longer tutorial when shortening it would force a beginner to chase
   prerequisite explanations across other files.
6. Use `[[wikilinks]]` instead of duplicating explanations across notes.
7. Use Obsidian callouts for invariants, warnings, examples, and context.
8. Make the note self-contained for its intended reader. Links to `docs/` and code
   are evidence and implementation detail, not required homework.
9. Clearly label proposed behavior; never present a target design as implemented.

Notes may be written in the language that best serves their audience. English
titles and paths are required even when the body is written in another language.

> [!note] Creation and modification dates
> Obsidian can display the version-controlled `created` and `updated` properties
> in the Properties view. Obsidian also knows local filesystem creation and
> modification times, but those are not durable across clones, checkouts, and
> rebases. Core Obsidian does not automatically maintain custom `updated`
> frontmatter on every edit; update it with the note, or use a configured
> automation/plugin. See [[meta/Using this Wiki]].
