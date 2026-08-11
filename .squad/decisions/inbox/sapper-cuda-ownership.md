# Decision: CUDA Plugin Shared EP Ownership Model

**Author:** Sapper
**Date:** 2026-08-11
**Status:** Implemented (unvalidated on hardware)

## Context

The CUDA plugin's four implementation defects (B4 review) all stemmed from
a fundamental architecture issue: the shared adapter (`onnx-runtime-ep-plugin`)
created independent EP instances for each ORT component (allocator, stream,
data transfer), each constructing its own CUDA runtime and context.

## Decision

Use a single-owner shared EP model:

1. **`ExportedFactory::shared_ep`**: An `Arc<Mutex<Box<dyn ExecutionProvider + Send>>>`
   holding the one EP instance. Components borrow it via raw pointer.
2. **`owns_ep` flags**: Each component tracks whether it owns its EP reference.
   The factory owns the shared EP; components are borrowers.
3. **`ExportedFactory::stream_handle`**: The native stream handle extracted at
   factory creation time, returned by `GetHandle` on all streams.

## Consequences

- CPU path unaffected (shared_ep defaults to None, host_accessible=true).
- CUDA EP constructs once, shares context across all ORT surfaces.
- Hardware validation (#768) is the remaining gate.
