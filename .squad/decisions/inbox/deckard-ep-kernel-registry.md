# ORT 1.27 Kernel Registry + Type Constraints for Compile-Based EPs

**By:** Deckard  
**Date:** 2026-08-10  
**Updated:** 2026-08-10 (f16/bf16 routing wired)

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

### f16/bf16 specifically advertised for:

- **Add, Sub, Mul, Div, Mod, Pow, Min, Max, Sum, Mean** — via `dispatch_arith!` (ARITH_DTYPES)
- **MatMul, Gemm** — via `half_gemm.rs` (FLOAT_DTYPES)
- **Sqrt, Erf, Tanh, Exp, Log, ...** (float unary) — via `dispatch_float!` (FLOAT_DTYPES)
- **Softmax, LayerNorm, ReduceMean, Attention, etc.** — FLOAT_DTYPES
- **Identity, Reshape, Transpose, Concat, etc.** (byte-movers) — ALL_DTYPES

### Wiring

`crates/onnx-runtime-ep-cpu-plugin/src/lib.rs` now hand-writes `CreateEpFactories` and
`ReleaseEpFactory` (replacing the `export_ep_factories!` macro) and calls
`create_ep_factories_with_registry` with entries derived at factory-creation time.

## Whether f16/bf16 genuinely routes now

**Not yet fully verifiable.** The infrastructure is complete: `GetKernelRegistry` is populated
with per-op f16/bf16 type constraints, ORT receives the registry at EP creation, and the
`ExportedFactory` stores the entries. However, whether ORT actually routes f16/bf16 nodes to
our EP depends on ORT's internal `EpGraphSupportInfo_LookUpKernel` path during GetCapability.

**What IS proven:**
- The kernel registry entries include Float16/BFloat16 for Add, MatMul, etc. (unit test)
- The fail-closed rule works (unit test)
- All 21 cpu-plugin conformance tests pass (proving the wiring doesn't break anything)
- All 127 ep-plugin adapter tests pass
- Workspace check succeeds

**What requires Pris's e2e test:** An actual model with f16 tensors run through ORT with our
plugin loaded. The GetKernelRegistry metadata is now populated; whether ORT uses it for
compile-based EPs vs kernel-based EPs requires the conformance test to confirm.

## NEW-2 Verdict: `ep_compile_inner` partial cleanup

(unchanged from original — see archive)

