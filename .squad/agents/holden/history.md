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
