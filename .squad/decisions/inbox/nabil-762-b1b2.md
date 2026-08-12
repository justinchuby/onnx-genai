# Decision: B1+B2 memory-safety fix for absent optional outputs

**Date:** 2026-08-12  
**Author:** Nabil (FFI/systems)  
**PR:** #762 (stays draft — needs fresh Opus review)  
**Commit:** af45043fd

## B1 — Heap buffer overflow in scratch allocation

**Problem:** Scratch buffers for absent optional outputs used the byte size of the slot's declared dtype (2 bytes for f16/bf16) but the `TensorMut` was hardcoded to `Float32`, so kernels wrote `numel × 4` into a `numel × 2` buffer — a 2× heap overflow on every f16/bf16 op with an omitted optional output.

**Fix:**
- Scratch dtype derived from `output_dtypes[slot]` — no hardcoded `Float32`.
- Buffer sized at `max(byte_size, 8)` per element to tolerate kernels that internally widen (e.g. SkipLayerNorm writes Float32 to mean/inv_std slots regardless of input dtype).
- `TensorMut` gets an `absent` flag; kernels skip dtype validation for absent outputs.
- If dtype is `Undefined` (ORT didn't declare it and no present output exists to propagate from), compute returns `Err` — fail closed.

## B2 — Routed path positional compaction

**Problem:** The routed (fused multi-node) path skipped allocation for absent slots but iterated all sinks via shortened iterators, so any fused node with an omitted output panicked or silently misrouted tensors.

**Fix:** `RoutedSlotKind` enum (Ort/Buffer/Absent) replaces the old continue-based skipping. Every slot gets an entry, so indices stay aligned end-to-end through allocation and view construction.

## Evidence

- **Canary tests:** f16 and bf16 canary byte patterns around scratch allocations; deterministic overflow detection.
- **Miri:** Both canary tests pass under `cargo miri test` (nightly 1.99.0). Miri cannot cross FFI into real ORT — integration tests not coverable.
- **ASAN:** Nightly sanitizer requires rebuilding libstd; not attempted. Canary + Miri provides equivalent coverage for the pure-Rust paths.
- **Test counts:** 277 passed, 0 failed (baseline 269, +8 new tests).
- **Clippy + fmt:** Clean.

## Preserved invariants

All items from the PRESERVE list are intact — `absent_outputs: HashSet<ValueId>`, rank-preserving `Vec<Option<usize>>`, `NodeInputSource::Absent`, runtime `raw_axis` resolution, removed `unwrap_or(Float32)` fallbacks, `resolved >= rank` bounds check, panic containment, `c_char` portability.
