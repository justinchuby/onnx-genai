# Decision: NxrtStatus wire code is u32, not enum

**Date:** 2026-08-11
**Author:** Isidore (Mobile & Bindings Engineer)
**Scope:** `onnx-runtime-ep-nxrt-abi`, `onnx-runtime-ep-nxrt-host`

## Context

The nxrt ABI is a stable plugin boundary. The other side of the `cdylib` may be
a newer plugin version, or a buggy third-party implementation. Transmuting an
unrecognised `u32` discriminant into a Rust `#[repr(u32)]` enum is **undefined
behaviour** per the Rust reference.

## Decision

1. `NxrtStatus.code` is now a raw `u32` (the wire type). All constructors write
   the enum discriminant as `code as u32`.
2. The safe accessor `NxrtStatus::status_code() -> Option<NxrtStatusCode>` does
   checked conversion via `NxrtStatusCode::from_u32()`.
3. Unknown codes → `None` → callers treat as fatal (fail closed).
4. The host validates `struct_size` before calling through any vtable slot.
5. CUDA plugin initialises the ORT host API before running diagnostics.

## Consequences

- No UB from untrusted plugin status codes.
- Newer plugins with new codes degrade gracefully on older hosts.
- Existing tests updated to use `status.status_code()` instead of `status.code`.
- Struct size is 264 bytes unchanged (u32 field, not enum, same repr).
