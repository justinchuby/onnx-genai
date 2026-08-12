# Decision: Test-integrity gaps closed for EP plugin scratch/routing (PR #762)

**Date:** 2026-08-12  
**Author:** Coco (systems engineer)  
**Status:** Implemented

## Context

PR #762 had three test-integrity gaps flagged in review:
1. `validate_write_dtype` was dead code — nothing called it.
2. The scratch sizing formula existed in 3 places (2 production + 1 test copy).
3. Unroutable multi-node graphs deferred failure to Run instead of Compile.

## Decisions

### 1. validate_write_dtype — wired into tests, not production hot path

The kernel API passes raw pointers via `Kernel::execute`; intercepting writes
at the raw-pointer level would require restructuring the entire trait. Instead:
- Two new tests exercise the validator on absent and present `TensorMut`.
- The function documents the contract: kernels must not write wider than
  `max(byte_size, 8)`. Future refactors (e.g. typed kernel outputs) can
  enforce at runtime; for now the tests prove the mechanism works.

### 2. scratch_alloc_bytes — single public function

`pub fn scratch_alloc_bytes(numel: usize, dtype: DataType) -> usize` in
`compute.rs`. Both production sites and all canary tests call it directly.
The old `production_scratch_alloc` test helper (a copy) is deleted.

### 3. Routing None fails at Compile

`build_subgraph_routing` returning `None` now produces a Compile-time
`fail_status`. ORT falls back cleanly. Dual-role slots (value is both graph
output and consumed by a downstream node) are not representable in the current
`NodeOutputSink` enum; the Compile failure rejects such graphs explicitly.

## Consequences

- Canary test `scratch_buffer_detects_oversized_write` calls the same function
  production does — formula drift is structurally impossible.
- An unroutable graph is caught before execution, where ORT has clean fallback.
