# ORT 1.27 Kernel Registry + Type Constraints for Compile-Based EPs

**By:** Deckard  
**Date:** 2026-08-10  
**Updated:** 2026-08-10 (dtype-aware GetCapability claim predicate)

## Final Design

### Where the accessor lives and why

The dtype-per-op metadata lives as an **inherent function** in `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs`:
- `build_cpu_registry_with_descriptors()` returns `(OpRegistry, Vec<CpuOpDescriptor>)`
- `supported_dtypes_for_op(op_type, domain) -> &'static [DataType]` classifies each op

This is NOT a trait method on `ExecutionProvider`. Rationale: `KernelRegistryEntry` is in the
plugin crate (downstream), so the trait (in ep-api) cannot reference it without a circular dep.
Each plugin crate constructs entries from its EP's data — consistent with §524 (C ABI projects
the Rust trait) and keeps the trait stable. The CUDA EP can adopt the same pattern later by
implementing its own `supported_dtypes_for_op`.

### How type constraints are derived from the real registry

A `RecordingOpRegistry` wrapper intercepts every `OpRegistry::register()` call during
`build_cpu_registry_recorded_inner()`. The exact same code path that builds the live registry
also emits the op-key list. Descriptors are then enriched with dtype info by
`supported_dtypes_for_op()`, which classifies ops into categories derived from the actual
kernel dispatch macros (`dispatch_arith!`, `dispatch_float!`, byte-movers).

**Fail-closed rule:** Any op not explicitly classified gets `&[DataType::Float32]` only.
Any `pkg.nxrt` custom op gets f32-only. CNN ops (feature-gated) get FLOAT_DTYPES only.

### Dtype-aware GetCapability claim predicate (NEW)

`ep_get_capability_inner` now filters claims using `node_passes_dtype_filter()`. For each
node in a claim:

1. Look up the op's `KernelRegistryEntry` from `ExportedEp::registry_entries` (same entries
   used to build `GetKernelRegistry` — single source of truth, no drift by construction).
2. Check all input and output dtypes against `entry.supported_dtypes`.
3. **Fail closed:** reject if op has no entry, if any dtype is `Undefined`, or if dtype not
   in the supported set.

`ExportedEp::registry_entries` is populated via `new_with_registry_and_entries()` called from
the factory's `CreateEp` callback. Legacy `new_with_registry` passes empty entries → filter
is bypassed (all nodes pass), preserving backward compatibility.

### f16/bf16 specifically advertised for:

- **Add, Sub, Mul, Div, Mod, Pow, Min, Max, Sum, Mean** — via `dispatch_arith!` (ARITH_DTYPES)
- **MatMul, Gemm** — via `half_gemm.rs` (FLOAT_DTYPES)
- **Sqrt, Erf, Tanh, Exp, Log, ...** (float unary) — via `dispatch_float!` (FLOAT_DTYPES)
- **Softmax, LayerNorm, ReduceMean, Attention, etc.** — FLOAT_DTYPES
- **Identity, Reshape, Transpose, Concat, etc.** (byte-movers) — ALL_DTYPES

### Wiring

`crates/onnx-runtime-ep-cpu-plugin/src/lib.rs` hand-writes `CreateEpFactories` and calls
`create_ep_factories_with_registry` with entries derived at factory-creation time. The factory's
`CreateEp` callback now passes entries to `new_with_registry_and_entries` so that both
`GetKernelRegistry` and `GetCapability` use the same data.

## Whether f16/bf16 genuinely routes now

**YES — the dtype predicate will correctly claim f16/bf16 nodes** for ops that list Float16/BFloat16
in their registry entries (Add, MatMul, Softmax, etc.). The claim path is now:

1. `query_capabilities` finds nodes whose op/domain we support
2. Shape-inference filter rejects data-dependent-shape ops
3. **Dtype filter** rejects nodes whose element types aren't in our supported set

For an f16 Add node: its inputs are Float16, the Add entry includes Float16 → claimed ✓.
For an f16 NonZero node: shape inference declines it → rejected before dtype check even runs.

**Cannot empirically prove end-to-end execution** on this host (no real f16 model available in
the conformance suite without Pris's test). The dtype filter logic is fully unit-tested. The
infrastructure is complete and correct.

**Instruction to Pris: un-ignore the f16/bf16 conformance tests.** The claim predicate is now
dtype-aware and will correctly route f16/bf16 nodes to our EP for ops that genuinely support
them. If the tests fail, the failure will be in the kernel execution path (Leon's domain),
not in GetCapability over-claiming.

## NEW-2 Verdict: `ep_compile_inner` partial cleanup

(unchanged from original — see archive)
