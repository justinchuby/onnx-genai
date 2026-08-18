---
title: Using this Wiki
aliases:
  - Wiki Conventions
  - Obsidian Setup
tags:
  - wiki
  - obsidian
  - contributing
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Using this Wiki

> [!summary] Question answered
> How should contributors open, link, date and maintain this repository wiki in Obsidian?

## Open the vault

Open the repository's `wiki/` directory as an Obsidian vault. The notes use
standard Markdown, YAML properties, wikilinks, callouts and Mermaid diagrams.
No community plugin is required to read them.

## Naming

- Use English filenames and `title` properties for stable cross-language links.
- The body may use the language best suited to its audience.
- Use descriptive noun phrases, not issue numbers or temporary project phases.
- Organize by durable domain: `start/`, `architecture/`, `execution/`, `memory/`,
  `meta/`, and future peer domains.

## Required properties

```yaml
---
title: Stable English Title
aliases:
  - Optional alternate title
tags:
  - domain
status: maintained
created: 2026-08-17
updated: 2026-08-17
---
```

Suggested statuses:

| Status | Meaning |
|---|---|
| `maintained` | Intended to track current understanding |
| `proposed` | Explains a target design that is not fully implemented |
| `historical` | Preserved context, not current guidance |
| `draft` | Incomplete and not ready to rely on |

## Can Obsidian show creation and modification times?

Yes, with two different meanings:

### Version-controlled properties

`created` and `updated` appear in Obsidian's Properties view and are stored in
Git. These are the wiki's durable dates.

- Set `created` when adding a note.
- Change `updated` when materially changing its meaning.
- Do not change `updated` for whitespace-only or link-only maintenance unless the
  repository adopts a different convention.

Obsidian Core does not reliably auto-update a custom `updated` property whenever
the note changes. A Linter/Templater-style plugin or repository automation can do
that, but contributors should not need a plugin to produce valid notes.

### Filesystem timestamps

Obsidian and plugins can inspect local file creation/modification timestamps.
These are useful locally but unreliable as repository history:

- cloning creates new local files;
- checkout/rebase can change mtimes;
- different filesystems preserve metadata differently;
- Git does not version filesystem creation time.

Use Git history for exact change provenance:

```bash
git log --follow -- "wiki/path/Note.md"
```

> [!important]
> Frontmatter dates describe note-level editorial history. Git commits remain the
> authoritative record of who changed which lines and when.

## Linking

Prefer wikilinks for wiki concepts:

```markdown
[[architecture/Crate Architecture]]
[[execution/Execution Backends|backend overview]]
```

Use normal relative Markdown links for source files under `docs/` or `crates/`,
because GitHub renders them correctly and they represent formal sources:

```markdown
[Memory Architecture](../../docs/memory/MEMORY_ARCHITECTURE.md)
```

## Avoid duplicated truth

Do not copy large status tables, benchmark numbers or normative contracts into
the wiki. Explain the concept, identify the authoritative source, and link it.
This keeps one place responsible for updating facts.

## Note template

```markdown
---
title: Note Title
aliases: []
tags:
  - domain
status: draft
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Note Title

> [!summary] Question answered
> One sentence describing what the reader will learn.

## Explanation

...

## Formal sources

- [Source](../../docs/path.md)

## Related notes

- [[start/Repository Map]]
```
