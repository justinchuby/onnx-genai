### 2026-08-07: PR #728 revision re-review
**By:** Harry
**Verdict:** REJECT

## Blocking findings

1. **Finding 1 remains open for downstream consumers; input+output checking is not transitive.** Shape inference deliberately replaces two distinct broadcast symbols with the lower-ID representative (`crates/onnx-runtime-shape-inference/src/context.rs:468-482`), and the existing regression proves the named/low-ID symbol wins (`crates/onnx-runtime-shape-inference/tests/op_rules.rs:1721-1738`). Binary inference writes that representative to the output (`handlers/elementwise.rs:20-28`), and graph inference persists it in the IR (`infer.rs:435-452`). A downstream unary/pointwise op then copies that already-aliased shape (`handlers/elementwise.rs:11-16`), so both its input and output contain only the pinned-looking representative. `node_capture_seq_independent` performs exact `SymbolId` membership on those edges (`kernel_cache.rs:234-252`) and therefore wrongly returns true. Runtime execution recomputes the aliasing op's real broadcast extent from concrete inputs (`executor/dynamic_shapes.rs:3-32`, `executor/dispatch.rs:610-650`), while replay launches captured segments without redispatching their nodes (`executor/dispatch.rs:169-182,212-250`), making the stale downstream geometry scenario reachable. The new test covers only the first aliasing op, not its consumer (`executor/tests.rs:889-909`). Require union-find/equivalence-class canonicalization (or equivalent transitive propagation) before growing-set membership, plus a downstream-consumer regression.

2. **Finding 2 is improved but not fully closed for CSA output 5.** The collector adds CSA outputs 1/3 only (`kernel_cache.rs:88-96`) and extracts only the penultimate axis (`:101-114`). Dynamic ratio-4 CSA independently mints a fresh, growing `selections` symbol on output 5's last axis (`crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs:181-195`); the shape test confirms this symbolic output (`tests/op_rules.rs:3491-3505`). The generic `past…`/`present…` rank-4 boundary scan does not cover this internal selected-indices value. The added CSA test exercises output 1 only (`executor/tests.rs:943-985`). Collect output 5's growing axis and test a pointwise consumer.

## Closed / confirmed

- **Finding 3 is closed:** `C1_CAPTURE_TOKEN` is 46283 (`qwen36_35b_a3b_qmoe_divergence.rs:114-121`) and the fatal assertion permits only 33803 or 46283 (`:326-332`).
- The generic rank-4 symbolic-penultimate KV scan excludes static-penultimate recurrent/GDN state (`kernel_cache.rs:171-190`); the targeted recurrent-state test passes, so GDN pointwise eligibility and the reported 34-segment collapse remain consistent.
- Targeted classifier tests, the symbolic representative test, and `git diff --check` passed.

**Revision owner:** Leon should produce the next revision. Cohaagen and Deckard are locked out for this artifact revision cycle.

**REJECT: growing-symbol aliasing still escapes into downstream capturable consumers, and CSA output 5's growing selection axis is still omitted.**
