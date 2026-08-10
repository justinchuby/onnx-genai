# ORT 1.27 Kernel Registry + Type Constraints for Compile-Based EPs

**By:** Deckard  
**Date:** 2026-08-10

## NEW-2 Verdict: `ep_compile_inner` partial cleanup

**Determination: Ownership on Compile failure is UNSPECIFIED.**

Header evidence (onnxruntime_ep_c_api.h lines 2179, 2203–2207):
- Line 2179: "ORT calls ReleaseNodeComputeInfos() to release multiple instances in a batch."
- Lines 2203–2207: "Release OrtNodeComputeInfo instances [...] The OrtNodeComputeInfo instances to release."

Neither specifies whether ORT calls `ReleaseNodeComputeInfos` when `Compile` returns a failure status. The "release in a batch" language applies to the success path.

**Resolution:** Implemented the safe-under-both-interpretations strategy:
- On failure at index `i`, free `out_infos[0..i]` AND null them out.
- If ORT calls ReleaseNodeComputeInfos: all slots are null → no double-free (our release callback already skips nulls).
- If ORT does NOT call it: no leak because we freed.

This is the only strategy that avoids both leak and double-free regardless of ORT's behavior.

## GetKernelRegistry and Type Constraints

### Architecture Findings

1. **GetKernelRegistry coexists with Compile.** Header line 1522 documents `EpGraphSupportInfo_LookUpKernel` as "Used within OrtEp::GetCapability()" — the kernel registry provides type-constraint metadata that ORT makes available during capability queries, while Compile still handles fused execution.

2. **Kernel registry is advisory for compile EPs.** Header lines 2382–2383: "Output parameter set to the EP's kernel registry, which must remain valid throughout the lifetime of the EP. Can be NULL if the EP doesn't use a kernel registry." Line: "If set to NULL, ORT assumes the EP compiles nodes." For a compile-based EP, the registry gives ORT type-constraint metadata for node routing.

3. **Building a registry requires:** `CreateKernelRegistry` → `CreateKernelDefBuilder` → set op/domain/version/EP/type constraints → `Build` → `KernelRegistry_AddKernel`. All available since ORT 1.24.

### What Was Implemented

- `GetKernelRegistry` callback in `ep.rs` that returns the pre-built registry
- `OrtKernelRegistryHolder` with proper drop semantics (calls `ReleaseKernelRegistry`)
- `build_ort_kernel_registry()` function that constructs the ORT registry from `KernelRegistryEntry` slices
- `KernelRegistryEntry` public type: `(op_type, domain, since_version, end_version, supported_dtypes)`
- `create_ep_factories_with_registry()` in `factory.rs` for EPs that supply entries
- `noop_kernel_create` as the kernel create function (compile path handles execution)
- `dtype_to_onnx_tensor_elem` mapping (sourced from Leon's `CPU_EP_SUPPORTED_DTYPES` constants)
- Full test coverage for the new code paths

### Proven Blocker for f16/bf16 Routing

The `ExecutionProvider` trait (`crates/onnx-runtime-ep-api/src/provider.rs`) does NOT expose an iterator over registered `(op, domain, version)` triples. The `OpRegistry` in `registry.rs` has no public `iter()` or `entries()` method.

**To complete f16/bf16 routing**, the CPU EP plugin crate must pass `KernelRegistryEntry` slices to `create_ep_factories_with_registry`. This requires either:
1. Adding `fn kernel_registry_entries(&self) -> &[KernelRegistryEntry]` to the `ExecutionProvider` trait, OR
2. Having the CPU EP plugin crate (`lib.rs`) construct the entries from its `OpRegistry` and pass them to the new `create_ep_factories_with_registry` API

Option (2) is recommended: it keeps the trait stable and lets each EP plugin decide its own type constraints. The CPU EP plugin's `export_ep_factories!` macro invocation would become a call to `create_ep_factories_with_registry` with entries derived from `CpuExecutionProvider::registry_entries()`.

### Interaction Summary

| Mechanism | Purpose | Our Use |
|-----------|---------|---------|
| GetCapability | Claim nodes for compilation | ✅ working, claims all supported ops |
| Compile | Compile fused subgraphs | ✅ working, 15 conformance tests |
| GetKernelRegistry | Type-constraint metadata for routing | ✅ implemented, awaiting entries |
| EpGraphSupportInfo_LookUpKernel | In-GetCapability kernel lookup | Available once registry populated |
