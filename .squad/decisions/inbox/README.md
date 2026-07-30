# Decision inbox — durable queue (tracked)

This directory is **tracked in git**, not gitignored. Agents drop decision records
here as individual files; the Scribe merges them into `.squad/decisions.md` and then
deletes the drop.

Why tracked (changed 2026-07-29):

- **Drops survive worktree deletion.** Previously the inbox was gitignored, so drops
  existed only on the machine that wrote them. Removing a worktree before Scribe ran
  destroyed unmerged drops — this cost real records more than once.
- **Cross-machine visibility.** With several teams writing on different machines, a
  tracked drop is visible everywhere as soon as it is pushed, before it is merged.
- **Concurrent Scribes don't collide.** Each drop is a distinct file, which git merges
  without conflict. The only overlap is the merge-and-delete step, and delete/delete
  resolves cleanly.

Conventions:

- One decision per file. Name it `{agent}-{brief-slug}.md`.
- Keep the format used in `.squad/decisions.md` entries (a dated `###` heading with
  `**By:** / **What:** / **Why:**`, or a short titled record).
- Drops appear in PR diffs (a deliberate cost of durability) and are removed by the
  Scribe once merged. This README stays so the directory is never empty.
