# Decision: Dispatch-Reachability CI Lint

**Date:** 2026-07-27  
**Author:** Pris (Tester)  
**PR:** TBD (squad/pris-dispatch-reachability-lint)  
**Scope:** `scripts/check_dispatch_reachability.py` + `.github/workflows/ci.yml`

## Rule

Every `static ...TEST_HITS: AtomicUsize` counter in the CPU EP must be read
(`.load(...)`) inside a `#[test]` function in the same file. CI fails if a
counter has no test reading it.

## Why

PR #275 shipped two silent-wrong-answer bugs with green codecov (78% line
coverage). Line coverage cannot detect this class of defect — it measures
"was this line reached?" not "was this branch reached in this configuration?"

The dispatch-reachability pattern (atomic hit counters) asserts the precise
property: **this path really executes for the claimed inputs.** This lint
enforces the pairing so counters cannot exist without tests.

## What it catches

- A counter declared but never tested (lint exit 1, names the counter)
- A `fetch_add` to a name with no matching `static` declaration (coherence)
- Commented-out `.load()` calls (stripped before matching)

## What it cannot catch (documented gap)

A dispatch branch that SHOULD have a counter but doesn't. This requires
human review at PR time. The lint is honest about this: it states the gap
in its docstring and error output.

This mirrors the design of `scripts/check_platform_naming.py` which catches
file-level single-arch omissions but explicitly cannot catch within-file gaps.

## False-positive analysis

- **Non-dispatch statics** (e.g. `BNNS_PREFILL_CALLS`): not matched because
  the regex requires `TEST_HITS` in the name.
- **Helper functions**: the lint only scans for `static.*TEST_HITS` pattern.
- **Test-only code**: counters inside `#[cfg(test)]` ARE scanned — they are
  the mechanism itself.
- **Comments**: `//` line comments are stripped before `.load()` matching.

No false positives on current main (91 files, 5 counters, all paired).

## BNNS-fail fallback (13th combination)

Re-checked after merge: `bnns_matmul_f16_trans_b` and `bnns_matmul_f16`
return false only when `BNNSFilterCreateLayerBroadcastMatMul` returns NULL
or `BNNSFilterApplyTwoInput` returns non-zero. On current Apple Silicon
hardware with valid positive M/K/N dimensions, neither condition occurs.

Adding a fault-injection hook to the BNNS call would:
- Add a branch to the hot path (~500ns overhead per prefill)
- Require `#[cfg(test)]` conditionals in production dispatch
- Test only that the fallback was *reachable*, not that BNNS actually failed

Decision: leave this documented as acceptable risk. If future hardware or
OS versions introduce BNNS failures, the `half_gemm_tile` fallback is already
integration-tested via `matmul_half_dispatch_matches_widened_reference_across_irregular_shapes`.
