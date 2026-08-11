### 2026-08-07: PR #728 round-5 re-review
**By:** Harry
**Verdict:** REJECT

## Blocking finding

1. **The fail-safe classifier is not a hard capture veto, so both new round-4 regressions remain capturable through the legacy runtime-shape heuristic.** The executor passes `node_capture_seq_independent(...)` only as the boolean `set_capture_seq_independent` hint (`crates/onnx-runtime-session/src/executor/kernel_cache.rs:602-625`). CUDA unary and binary pointwise kernels then compute eligibility as `capture_shape_eligible(self.capture_seq_independent, shape)` (`crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs:539-551,812-826`). That helper is an **OR**: `seq_independent || numel == 1 || is_fixed_decode_shape(shape)`, and `is_fixed_decode_shape` accepts any shape whose dimensions except the last multiply to one (`elementwise.rs:1038-1059`).

   Consequently, classifier `false` does not mean eager. The exact new `Reshape([-1])` regression produces rank 1, for which the empty leading-dimension product is 1; the `Flatten` regression produces `[batch, seq_kv*8]`, which is accepted when decode batch is 1. Their downstream `Sigmoid` kernels therefore warm a supported capture signature even though `node_capture_seq_independent` returns false. Replay then retains the warmed `n`/grid while `seq_kv*8` grows—the same stale-geometry corruption the provenance fix is intended to close. The tests at `crates/onnx-runtime-session/src/executor/tests.rs:1110-1251` assert only the metadata predicate, not the kernel's actual `capture_support`, so they are false-positive safety tests.

   Make fail-safe disqualification a hard veto distinct from “metadata did not add eligibility” (for example, a tri-state policy or explicit `capture_forbidden` flag checked before the legacy heuristic), and add an EP/capture-planner regression proving disqualified rank-1 and `[1,N]` growing shapes remain outside capture.

## Confirmed

- The provenance graph itself is coherent: broadcast edges are undirected, derivations flow source→derived, and the combined worklist closure is transitive/order-independent (`kernel_cache.rs:412-442`).
- `inference_symbol_floor` is seeded above all graph symbols and current-pass mints are at/above it (`shape-inference/src/infer.rs:69-75,1527-1542`); repeated inference preserved the Reshape classification in a focused probe.
- Derived-only-from-pinned roots are not reverse-poisoned. Over-eager classification changes performance, not numerical semantics.
- Both new regressions independently failed on `571ea0d9` with only `{SymbolId(1)}` and pass on `0fd87df3`.
- `lower` returns the prior dimensions; added recording is bookkeeping-only. Full shape-inference tests passed (16 + 41 + 275 + doctests), targeted fail-safe/derived tests passed, and `git diff --check` passed.
- The switch-gated denylist residual is non-blocking for this verdict, but consider removing or more strongly gating it.

**Revision owner:** Roy should revise; Cohaagen, Deckard, Leon, Batty, and Sebastian remain locked out.

**REJECT: the shipped fail-safe set is computed correctly but is only an additive hint; the CUDA kernels can still capture disqualified growing rank-1 / `[1,N]` pointwise shapes via the legacy heuristic.**
