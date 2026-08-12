# Decision: fix server registry shrink test assertion (PR #821)

**Date**: 2026-08-12
**Author**: Roy
**Status**: Accepted

## WHAT

Update the assertion in `failed_runtime_shrink_preserves_policy_and_ledger_limit` from
`error.contains("committed bytes")` to `error.contains("cannot satisfy lowered resource limit")`.

## WHY

Commit `c7633eec4` (PR #740) changed the error wording from "committed bytes" to "leased bytes"
without updating the test. The test purpose is to verify shrink-below-usage is rejected and rolled
back — not to pin the exact phrasing. The stable prefix covers all rejection paths.

## Trade-offs

- Looser string match: could mask a future change that turns the rejection into a different error
  category. Mitigated by the test also asserting snapshot rollback (behavioral, not string-based).
- Alternative considered: match on `"leased bytes"` — rejected because there are two code paths
  (`state.rs` pooled-unmapped path and `memory_authority.rs` mapped-or-leased path) with different
  suffixes; the shared prefix is the only stable anchor.

## Introducing commit

`c7633eec4` — "Pool CUDA VMM physical handles across reservations (#740)"
