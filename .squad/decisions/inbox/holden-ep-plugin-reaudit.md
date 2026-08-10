# EP Plugin Export — Re-audit Verdict

**From:** Holden (Security Engineer)
**Date:** 2026-08-10T21:30:26Z
**Branch:** `squad/ep-plugin-export` @ `526a883c4`
**Verdict:** 🔴 **RED — ship-blocking**

---

## Original findings disposition

| ID | Finding | Status |
|----|---------|--------|
| C1 | No `catch_unwind` on extern "C" callbacks | **OPEN (partial)** — `compute_execute` unguarded |
| H1 | `static mut HOST_ORT_API` data race | RESOLVED — AtomicPtr Acquire/Release |
| H2 | `graphs` null-deref in `ep_compile` | RESOLVED — null guard at ep.rs:209 |
| H3 | Unsound `Send+Sync` on `OutboundGraphReader` | RESOLVED — impls removed |

## Ship-blocking issues

### CRITICAL — `compute_execute` missing `catch_unwind` (compute.rs:119)

ORT's `OrtNodeComputeInfo::Compute` callback — the per-inference hot path — has no `catch_unwind`. Panics from `entry.kernel.execute()` (user-supplied trait method), the `inputs[input_offset..input_offset + entry.num_inputs]` slice range, `infer_shapes`/`broadcast_shapes`, or OOM inside `read_inputs` will unwind across the C ABI into ORT's process: instant UB, corrupt host process.

**Fix:** Wrap entire `compute_execute` body in `catch_unwind(AssertUnwindSafe(|| { ... }))`.
**Owner: Deckard** (owns `compute.rs`/`kernel_ctx.rs`).

### HIGH — Negative dims wrap to `usize::MAX` in `kernel_ctx.rs:154`

`dims.iter().map(|&d| d as usize)` silently casts ORT's dynamic-dim sentinel `-1i64` to `usize::MAX`. Models with symbolic batch dims trigger this at runtime. Downstream `broadcast_shapes` arithmetic on `usize::MAX` can panic (no catch_unwind → UB). Must be fixed alongside N1.

**Fix:** Checked conversion; return error on negative dim in `read_inputs`.
**Owner: Deckard**.

### MEDIUM — Macro-generated `CreateEpFactories`/`ReleaseEpFactory` lack `catch_unwind` (lib.rs:64–88)

The two top-level `extern "C"` symbols call `create_ep_factories` / `release_ep_factory` without a panic guard. `constructor()` and `ep.name()` can panic at plugin-load time.

**Fix:** Add `catch_unwind` wrappers in the macro expansion.
**Owner: Nabil** (owns `factory.rs`, `lib.rs` macro).

---

## Reviewer Rejection Protocol note

Per protocol: original author Nabil is locked out from revising compute.rs/kernel_ctx.rs (Deckard owns those files). Deckard is locked out from revising the macro in lib.rs/factory.rs (Nabil owns those). No cross-lock conflict; both can proceed in parallel.

Full report: `docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`
