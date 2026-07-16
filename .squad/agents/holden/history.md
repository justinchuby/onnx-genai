# Holden — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Phases 1-4 done; tool use/grammar/chat-template; Qwen2.5-0.5B runs; Hermes agent E2E; long-context O(1)/token via static-cache in-place KV. Working on DESIGN §26 batched serving + reviews.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-12

## 2026-07-12T13:14:00-07:00 — Security audit merged
Holden's unsafe/resource/supply-chain audit is now in decisions. Current unsafe invariants are documented and sound under today's constraints; cargo audit found 0 vulns and 2 unmaintained transitive tokenizers warnings.


### 2026-07-12T14:50:00-07:00
Recurring audit convention is canonical: `.github/workflows/audit.yml` runs weekly and on dependency changes; fresh cargo-audit found 0 vulnerabilities. Continue periodic security review passes.

## 2026-07-14T02:37:00Z — Reviewed ep-api + ep-cpu (safety)
- **ep-api (65ec9f6):** 🟡 safety — DeviceBuffer ownership, Send/Sync soundness, unsafe construction contracts.
- **ep-cpu (ea30279):** 🟡 safety — strided::view_in_bounds enforcement, isolated unsafe blocks (aligned alloc/dealloc, copy_nonoverlapping, two strided accessors), no cross-EP free.

## 2026-07-14T05:04:00Z — ORT2 safety reviews: session base + dynshape + capi

- **squad/ort2-session** review (🟡): All 5 invariants held (view bounds, single-free, no cross-EP free, copy size ≤ min(src,dst), host-global). Aliasing: in-place ops cause CycleDetected at build. Miri clean. Advisories: A1 mid-run error-path buffer leak; A2 unchecked i64 in `view_in_bounds`; A3 cache key omits dtypes.
- **squad/ort2-session-dynshape** review (🟡): Invariant #1 holds against run-scoped buffers (gate keys off real `buf.len()`). Single-free on realloc verified. 14/14 Miri-clean. Advisories: H-D1 unchecked shape-multiply overflow; H-D2 stale buffer_shapes if allocate fails post-dealloc; Holden-A1 (pre-existing) mid-run leak unchanged.
- **squad/ort2-capi** review (🟢): All 6 FFI axes pass. 12/12 Miri-clean. Advisories: A1 release fns not in guard; A2 storage_bytes unchecked multiply (bounded by prior validation).

## 2026-07-14T06:06:00Z — H-D1 Re-Review (holden-7) → 🟡 SHIP

- **holden-7:** Re-reviewed Deckard's H-D1 fix on `squad/ort2-hardening` @ 852f262.
- Prior 🔴 cleared: all three fix layers (dtype.rs checked_storage_bytes, executor.rs both alloc sites, strided.rs i128 address math) confirmed correct. Original exploit vector `[2^61]`×f64 dead — `ShapeOverflow` at allocation before any buffer exists.
- **Verdict: 🟡 SHIP-with-advisories.** Residual advisories A1 (peripheral panic paths → graceful error) and C1 (addressed_elem_range i64 before i128 widening) are memory-safe non-blocking; suggested fast-follow owner: Leon (neither Deckard nor Batty).
- Fix cherry-picked to main: dbf2d70, 9dcdc04, f749012.

## 2026-07-14T07:20:00Z — ORT2 shape-inference crate review cycle

- **holden-8:** Reviewed `onnx-runtime-shape-inference` DimExpr symbolic-algebra soundness. 🔴 REJECT — `DimExpr` `add`/`sub`/`mul` used unchecked i64: debug panics, release wraps to bogus dim on large tensors (`2^80` product). Secondary: `checked_div` unguarded against `i64::MIN/-1`. All other items HELD. Fix assigned to Deckard.
- **holden-9:** Re-reviewed Deckard's overflow fix (`09988f3`). 🟢 GREEN — all combiners use `checked_*`; `overflow()` sentinel never aliases (SymbolInterner bypasses cache); poison propagates; no wrap in debug or release. 69 tests green debug+release. Advisory noted: `movement.rs` slice-index raw i64 (pre-existing, filed as follow-up).

## 2026-07-14T08:40:00Z — ORT2 shape-inference wiring + IR dtype hardening reviews

- **holden-10:** Reviewed Roy's shape-inference wiring (`f4141b9`). 🟢 GREEN. Symbol-unification sound (overflow sentinel gates representative arm; deterministic order-independent; single topo-pass). Loader seam transactional (graph mutated only after full write-back). JIT fallback comment-only diff. No regressions to view_in_bounds/checked_storage_bytes/unsafe. Full ORT2 suite green debug+release. Non-blocking advisory: stricter fail-fast coupling means op-rule false-positives now block load; Chew to confirm none fire on BERT/opset-12.
- **holden-11:** Reviewed Deckard's IR dtype hardening (`f965f0b`). 🟡 APPROVE-WITH-FOLLOW-UP. Net soundness improvement; no regression; no new unsafe. Float4E2M1 routes through `div_ceil(2)` (overflow-safe). Fail-closed attr hardening safe. 300 tests green debug+release. **Required follow-up:** `graph_builder.rs` value-info (L232,241) and attribute-tensor (L357,365,374) still `.unwrap_or(Float32)` — silent mislabel for unmodeled dtypes. Deckard PROGRESS.md claim overstated. Owner: Roy/Batty/Leon before complex-dtype milestone.

## 2026-07-14T10:00:00Z — ORT2 dtype fail-close review (holden-12)

- **Task:** Review leon-10's dtype fail-close work on `squad/ort2-dtype-failclose` (`a822a21`). Verify closure of own prior holden-11 finding (value-info + attribute-tensor `.unwrap_or(Float32)` sites).
- **Verdict:** 🟢 GREEN — finding fully closed, no over-reach, no regressions.
- **Key checks:** All 8 real-dtype decode sites confirmed fail-closed via `decode_dtype`. No surviving `unwrap_or(Float32)` on real-dtype sites. Signature changes `-> Result<…>` with `?`; transactional-on-failure preserved. Proto bump correct. Full ORT2 suite (262 tests) green debug+release. bert_toy PASS max_abs 1.192e-7. Non-blocking advisory: present-but-UNDEFINED elem_type=0 on value-info now rejected (correct fail-close for typed I/O).

- 2026-07-14T19:05:00Z — Rejected sentinel-leaking UnsupportedOp diagnostics, then approved the final enriched-error plus loader fail-fast validation solution merged in `00cda89`. Also reviewed pipeline API seams GREEN.


## 2026-07-14T20:05:00Z — Loader validation review
Reviewed Batty's unified load-time `validate_model()` checks for unsupported control flow and dangling tensor references 🟢. Both disk/bytes and session load paths are covered; merged as `2a99eec`.

- 2026-07-15 — Reviewed the complete cross-platform oneDNN wheel branch; approved `ef89a95`.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed the threading change; post-fix verdict was needs-review/ship.

- 2026-07-16T00:00:01Z — 🟡 Approved Leon's guarded contiguous-f32 native CPU Mul fast path (`347060f`); aliasing, striding, broadcasting, and non-f32 remain generic. Independent decode result: +6.35%.

### 2026-07-16T00:00:03Z — Decode thread-cap safety review cycle
Rejected Deckard's initial thread-cap environment parsing because unusable or huge values could abort inference or trigger unsafe worker creation; Deckard was locked out of the revision. Re-reviewed Sebastian's pure bounded resolver and cleared it: invalid values fall back safely, valid values cap at available parallelism, prefill remains untouched, and 413 tests passed.
