# Decision inbox — local runtime queue

With `.squad/config.json` set to `stateBackend: "local"`, individual inbox records
are mutable local state and must remain untracked. This README is the intentional
tracked placeholder; `.gitignore` ignores every other file in this directory.

Agents may create local records here. Scribe promotes accepted, durable decisions to
the canonical `.squad/decisions.md` (and its append-only archive when consolidated);
do not rewrite that history to remove local-state evidence.

Conventions:

- One decision per file. Name it `{agent}-{brief-slug}.md`.
- Keep the format used in `.squad/decisions.md` entries (a dated `###` heading with
  `**By:** / **What:** / **Why:**`, or a short titled record).
- Local drops do not appear in PR diffs. This README stays so the directory is never
  empty and so the local-state policy is discoverable.
