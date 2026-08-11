# Decision: EP plugin export delivered as a single draft PR (#762)

**Date:** 2026-08-11T00:04:18.399+00:00
**By:** @justinchuby (via Copilot coordinator)
**Status:** Accepted

## Context

GitHub CLI became available and authenticated, clearing the long-standing
push blocker. 14 commits across two stacked branches had been local-only.

## Decision

1. Both branches are pushed to origin: `squad/ep-plugin-export` (M1, 9
   commits) and `squad/ep-plugin-parity-cuda` (M2, 5 commits).
2. Work is delivered as a **single draft PR** — #762, base `main`, head
   `squad/ep-plugin-parity-cuda` — tracking the whole EP-compatibility
   milestone. This supersedes the two-stacked-PR recommendation recorded in
   `docs/EP_PLUGIN_EXPORT_PR.md`; a PR comment records the deviation so
   reviewers are not misled.
3. Subsequent milestone commits are pushed to that same draft.
4. **It stays in draft** until the broader EP compatibility milestone is
   complete and validated. It is NOT marked ready yet.

## Why still draft

- Native nxrt dynamic ABI (second half of the extension contract) is
  unimplemented.
- CUDA EP is blocked on both missing hardware and real device-pointer /
  data-transfer work.

## Consequences

- M1 remains independently green on origin and can be split into its own PR
  later if a smaller review surface is wanted.
- The push-early/draft-early cadence directive is now actually executable;
  future milestones push per commit rather than batching.
