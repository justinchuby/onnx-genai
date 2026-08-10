# Decision: EP Plugin Export — Adapter Crate & Dual-ABI Strategy

**Date:** 2026-08-10T20:12:35.793+00:00
**By:** Roy (Lead)
**Status:** Accepted

## Context

Our EPs implement `ExecutionProvider` (Rust trait) but cannot be loaded by
upstream ORT. The inbound path (loading foreign ORT plugins) is complete; the
outbound path (exporting our EPs as ORT plugins) is missing.

## Decision

1. **Single shared adapter crate** (`onnx-runtime-ep-plugin`, `lib`-only) owns
   100% of the unsafe FFI that projects any `ExecutionProvider` through the ORT
   plugin-EP C ABI (`CreateEpFactories`, `GetCapability`, `Compile`,
   `OrtNodeComputeInfo` callbacks).

2. **Per-EP thin `cdylib` shim crates** (e.g. `onnx-runtime-ep-cpu-plugin`)
   instantiate the concrete EP and invoke the adapter's `export_ep_factories!`
   macro. Each shim is ~5 lines of code and is not a workspace default member.

3. **Dual ABI:** the same `cdylib` exports the ORT plugin ABI today and will
   export the nxrt native dynamic ABI in the future. A normal `cargo build` at
   the workspace root produces no `cdylib` and requires no ORT C library. The
   EP `lib` crate is unchanged.

4. **Reuse of inbound machinery:** `UnionFind`, `SubgraphClaim`,
   `OrtGraphView::query_capabilities`, and `onnx_genai_ort_sys` types are
   shared. New outbound-specific code: `OutboundGraphReader` (reads ORT's
   `OrtGraph*`), `OutboundKernelContext` (ORT `KernelContext` ↔ `TensorView`),
   `ExportedFactory`/`ExportedEp` (heap objects behind opaque C pointers).

5. **Fail closed:** version mismatch → `ORT_FAIL` status with actionable
   message. Missing kernel → `ORT_NOT_IMPLEMENTED`. Compute failure →
   `ORT_FAIL` with `EpError` text. No silent fallback.

## Trade-offs

- **Pro:** Each new EP export is mechanical (one shim crate, one macro call).
- **Pro:** All unsafe FFI is reviewed and tested in one place.
- **Con:** An extra crate per EP. Acceptable given the alternative (bespoke FFI
  per EP).
- **Con:** The adapter depends on `onnx-genai-ort-sys` for type defs. This is
  already the case for the inbound path and adds no new build requirement.

## References

- `docs/EP_PLUGIN_EXPORT.md` — full architecture
- `docs/ORT2.md` §4.1, §4.5 — EP trait and ABI bridge design
- `.squad/decisions.md` — Extension contract standing directive §524
