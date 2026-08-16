# harry — History Archive

## 2026-08-11T03:25:00Z: Scribe compaction from live history

## 2026-07-29T05:55:00+0000 — PR #388 approved

- Verified schema constraints are consumed by llguidance, and checked constraint precedence and validation behavior.
- Targeted HTTP tests and clippy passed in a scratch worktree.
- Confirmed two unrelated full-suite context-limit failures already occur on origin/main.

## 2026-07-30T09:16:00Z — cfg-gated LoopStatePair hotfix

- Landed PR #441, repairing the cfg-gated `LoopStatePair` import.

## 2026-07-30T15:20:00Z — PR #477 merged (shape-inference container types + Sequence)

- PR #477 merged (Lori APPROVED): shape-inference IR container-type model + Sequence foundation (#449). Additive `ValueType` layer, byte-identical tensor path, 4 Sequence ops, 300 tests. Unblocks the previously-deferred Sequence/Optional/Map/ZipMap propagation.

## 2026-07-30T21:15:00Z — PR #486 merged (#449 inc2 sequence ops)

- PR #486 merged: #449 inc2 — SequenceInsert/SequenceErase/SplitToSequence/ConcatFromSequence + seq↔tensor conversion; op catalog 213→217, shape-inference 258→262.
- In flight: #449 inc3 (#527 in review) and inc4.

## 2026-07-31T00:25:00Z — PR #531 merged (#449 inc4); #534 held

- PR #531 merged: #449 inc4 — SequenceMap + Scan container support + cross-subgraph capture. CLOSED issue #449. Container-type shape inference COMPLETE: additive `ValueType{Tensor|Sequence|Optional|Map}`, byte-identical tensor path guaranteed (gated on non-empty container map). Catalog 217 ops/262 entries. Deferred non-load-bearing: Optional/Map handlers, IR-persistence of `ValueType`.
- PR #534 (server contracts #481/#482 — `build_dirty` Option<bool> present-as-null; truncated predicate uses actual scan size): Melina APPROVED but HELD — targets Justin's active branch `feat/genai-demo-dashboard` (PR #476); code exists only there, not main.
- In flight (harry-5): generalize ORT `clone_value`/`clone_owned` to all POD dtypes (unblocks Bool / gemma-3n audio mask).

## 2026-07-31T03:03:15Z — PR #540 merged: generalize ORT value-clone to all POD dtypes

- PR #540 merged (requested by Justin): cloning an ORT cached `Value` now covers ALL POD dtypes via one dtype-agnostic raw-byte fallback — do not re-add per-dtype bail arms. `decode/values.rs::clone_value` and `onnx-genai-ort::value.rs::clone_owned` terminal arms use `Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), shape, dtype)` (typed f32/f16/bf16/i64 fast paths kept). 11 tests.
- Rule: use `as_raw_bytes()` (host-guarded — precise `InvalidArgument` on a device tensor), NEVER `to_raw_bytes()` (reads the data pointer blind). Unblocks the gemma-3n Bool audio mask.

## 2026-07-31T08:48:28Z — #544 test deterministic fix; independent APPROVE #554

- Revised #544 flaky `async_pagein_fence_orders_weight_page_in_consumer` test (bf345904): root cause was the NEGATIVE poison-arm — pure wall-clock race. Fix: event-orders negative transfer after consumer via `record_compute_fence`+`copy_wait_fence`. Test green 5/5 parallel, fails 3/3 without the fix. Production code unchanged.
- Independent APPROVE for PR #554 (Mary's session-reuse recurrent-state fix). MERGED.

## 2026-08-02T10:05:00+0000 — Reviews for #592 and #594

- Approved #592 after verifying the removed flag stayed gone and the no-Scan/no-sibling graph-property gate was mutation-proven.
- Rejected #594 only for rustfmt, confirmed logic/design were otherwise good, and assigned Deckard as revision author under reviewer lockout.

## 2026-08-02T11:40:00+0000 — #595 review

- Independently reviewed #595 and approved it after confirming the dangling reset caller, mutation-verifying reset coverage, checking hot-path behavior, and confirming bench builds.

## 2026-08-02T15:45:00+0000 — PR #597 review

- Approved #597 after independently rebuilding the fp32 acc-1 oracle, confirming native token 1909 matches oracle while ORT device-greedy token 821 is a near-tie artifact.
- Verified the 27-token stream lock is non-vacuous, resolved the ORT-logits-vs-stream puzzle, and merged the regression-lock PR.

## 2026-08-02T19:00:00+0000 — PR #602 review

- Rejected round 1 because the IR inliner failed open for attribute-parameterized functions by not binding call-site/ref_attr_name attrs; Add/Relu tests hid the byte-identity defect.
- Approved round 2 after Deckard added fail-closed attribute-parameter guards, metadata preservation, and mutation-proven `ParamLeakyRelu` coverage; #602 auto-merge armed.

## 2026-08-02T19:50:00+0000 — PR #604 review

- Approved #604 after confirming the phase-profile flake fix was test-only and scoped assertions to a unique phase name instead of the process-global stats map.
- Mutation-proved reset coverage by making `reg.clear()` a no-op and seeing the test fail; reproduced 30/30 clean full-parallel lib runs plus fmt/clippy clean.

## 2026-08-03T02:40:00+0000 — PR #606 review

- Approved #606 after verifying flag-off byte identity, fail-closed mixed-plan behavior, no stateful call path to standalone `hetero::execute`, and honest deferral of integrated execution to #603.
- Mutation-proved the guard by making the `Heterogeneous` arm return `Ok` and seeing `guard_enabled_mixed_fails_closed` fail; fmt, clippy, and C API checks were clean.

## 2026-08-03T03:10:00+0000 — mobius PR #449 review

- Approved #449 after verifying positional wiring, inert bias behavior, and all three call sites passing `None` in slot 4.
- Mutation-proved the new test by moving the bias formal to the end and seeing it fail; confirmed the diff is mobius-only with no onnx-genai changes.

## 2026-08-03T07:40:00+0000 — PR #612 review

- Approved #612 after verifying fp16 TopK writes back original raw fp16 values and only upcasts for total-order compare.
- Confirmed CUDA k-major non-final-axis order is ONNX-spec-correct and mutation-proved tie-break coverage; CPU non-final-axis ordering is a latent separate bug.

## 2026-08-03T10:00:00Z — PR #616 review

- Approved #616 after mutation-verifying both the cuDNN comp-type bug and native device fallback, and checking FFI safety, f32 byte-identity, bf16 NVRTC fallback, and shared dispatch generality.
- Gates clean: fmt/clippy for both crates, GPU parity, engine device tests, and EP reduce lib tests.

## 2026-08-03T12:30:00Z — PR #618 review

- Approved Lever A after mutation-verifying cache-key shape coverage and sync gating, and checking miss-during-capture, warm-before-capture, shape-change aborts, f32/f16 byte behavior, and bf16 NVRTC exclusion.
- Verified fmt/clippy clean and 6/6 capture + 3/3 parity GPU tests; noted non-blocking need for no-cuDNN test guards.

## 2026-08-04T00:40:00Z — PR #625 loader review

- Rejected #625 rev1 for a major initializer-input leak in the native loader metadata path and assigned revision away from the locked-out author.
- Approved rev2 after Quaid added initializer exclusion mirroring `graph_builder.rs` plus metadata==Session KV-geometry parity coverage.
