# EP Plugin Compute Path — Design Decision

**By:** Deckard  
**Date:** 2026-08-10  

## Context

The outbound ORT plugin-EP export adapter needed a real `Compute` callback. Previously it returned `ORT_NOT_IMPLEMENTED`.

## Design: Compute/State Architecture

### State (`CreateState` / `ReleaseState`)

State is minimal — a `ComputeState` struct allocated per-session via `Box::into_raw` and freed in `ReleaseState` via `Box::from_raw`. The real execution state (compiled kernels, topological order, shape inference strategies) lives in `ExportedComputeInfo` which outlives all Compute calls. `ComputeState` exists to satisfy the ABI contract and can be extended later for per-session caches.

### Compute Execution

Single linear pass over `ExportedComputeInfo::entries` (topologically sorted at Compile time):

1. Read all inputs from `OrtKernelContext` via `read_inputs()` — extracts data pointer, dtype, shape for each input as `OwnedInput`.
2. For each kernel entry, slice the relevant inputs, infer output shapes, allocate ORT outputs via `KernelContext_GetOutput`, execute the kernel.
3. Convert errors to `OrtStatus` via `fail_status()`.

### Output Shape Resolution Strategy

Two strategies, selected at Compile time per op:

- **`ElementwiseBroadcast`**: numpy-style multi-input broadcast (Add, Mul, Sub, etc.)
- **`SameAsInput(idx)`**: output shape = input[idx] shape (unary ops, LayerNorm, etc.)

Unknown ops default to `SameAsInput(0)`. If that's wrong, the kernel will fail with a shape mismatch rather than silently producing wrong results.

**Why this approach:** For the CPU EP's 166 registered kernels (mostly elementwise and unary), these two strategies cover the vast majority. Ops with complex shape logic (Reshape, Concat, Gather) will need dedicated strategies added incrementally — but they fail closed, never silently wrong.

### Panic Safety

All `extern "C"` callbacks use `std::panic::catch_unwind` (in `CreateState`) or are inherently panic-free (no `.unwrap()` on fallible paths in `Compute`). If Nabil's shared panic-guard lands, consolidation is straightforward.

## Risks

- Multi-node fused subgraphs assume sequential input/output indexing (offset accumulation). This is correct for ORT's fused-node model but should be validated in integration tests.
- The `SameAsInput(0)` default for unknown ops is conservative but may cause failures for ops like Reshape. These need dedicated `ShapeInference` variants.
