# Decision: CUDA registry version bounds + nxrt struct_size hardening

**Date:** 2026-08-12  
**Author:** Isidore (FFI/ABI engineer)  
**PR:** #762  

## CUDA kernel-registry `end_version` → `i32::MAX`

**Rationale:** Our CUDA kernels are version-agnostic dispatchers. They operate on
the IR `Node` abstraction which already resolves opset-versioned schema differences
at parse time. The kernel never interprets opset-specific attributes directly.
Therefore no real upper bound exists — `i32::MAX` is correct for the same reason
it's correct on the CPU side. The previous value `99` was an arbitrary cap that
would silently under-claim once ONNX opset exceeds 99.

All ops in `CUDA_COVERED_OPS` fall into two families:
1. **Standard ONNX element-wise/reduction/pool/reshape ops** — schema changes
   across opsets (e.g. Add@7→14) only affect attribute parsing which the IR
   resolves before the kernel sees the node.
2. **Custom/contrib ops** (MatMulNBits, GroupQueryAttention, etc.) — these are
   our own definitions with no upstream schema evolution risk.

## `offset_of!` replaces hand-computed offsets

`memoffset_of_create_ep()` in `provider_adapter.rs` used manual arithmetic
(`size_of::<u32>() * 2 + size_of::<*const u8>() + ...`). Replaced with
`std::mem::offset_of!(NxrtEpFactoryVtable, create_ep)` which is authoritative
regardless of padding/alignment changes.

## struct_size guard on EP vtable release (safety)

**New guard in `NxrtExecutionProvider::drop`:** Before calling
`ep.release(ep.ctx)`, validate that `struct_size` covers both the `release` and
`ctx` fields. If undersized: **deliberately leak** rather than jump through a
bogus pointer.

**Policy:** Leaking is wasteful; jumping through a bogus function pointer from a
malformed or older plugin is arbitrary code execution. We choose the safe side.

**Short-circuit discipline:** The struct_size check comes before any field access
(same as the existing `name` guard in `loader.rs`). Rust's `||`/early-return
ensures no read past declared bounds.
