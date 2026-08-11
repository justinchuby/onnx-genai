# nxrt Host Loader — Loading Contract & Design

**Author:** Isidore  
**Date:** 2026-08-11  
**Relates to:** §524 extension contract

## Loading Contract

1. **`load_nxrt_plugin(path)`** — opens the `.so`/`.dll`/`.dylib`, resolves all required symbols eagerly, and performs version negotiation. Fails closed on any issue.
2. **Version negotiation** — calls `nxrt_abi_version` and rejects if `plugin_major != host_major` (currently 1). Forward-compatible: higher minor is accepted.
3. **`NxrtExecutionProvider::new(plugin, config_json)`** — calls the factory, validates ≥1 device, wraps as `dyn ExecutionProvider`.

## Required Plugin Symbols

| Symbol | Signature | Purpose |
|--------|-----------|---------|
| `nxrt_abi_version` | `(out_major: *mut u32, out_minor: *mut u32)` | Version negotiation |
| `nxrt_create_ep` | `(config: *const c_char, out: *mut *mut Handle) -> NxrtStatus` | Factory |
| `nxrt_destroy_ep` | `(handle: *mut Handle)` | Cleanup |
| `nxrt_ep_name` | `() -> *const c_char` | Human-readable name |
| `nxrt_device_count` | `(handle, out: *mut u32) -> NxrtStatus` | Device enumeration |

## Lifetime Safety

`Library` is stored in `Arc<Library>`. Both `NxrtPlugin` and `NxrtExecutionProvider` hold an `Arc` clone. The library cannot be unloaded while any EP instance exists — enforced by the type system, not comments.

## Negative Path Behavior

| Scenario | Behavior |
|----------|----------|
| Missing library file | `NxrtHostError::LibraryLoadFailed` with path and OS reason |
| Missing/misspelled symbol | `NxrtHostError::SymbolNotFound` naming the exact symbol |
| Incompatible ABI major version | `NxrtHostError::AbiVersionMismatch` with both versions and rebuild suggestion |
| Factory returns error | `NxrtHostError::FactoryFailed` with status description |
| Zero devices advertised | `NxrtHostError::ZeroDevices` explaining minimum requirement |

## Panic Containment

`Drop` on `NxrtExecutionProvider` wraps `destroy_ep_instance` in `catch_unwind`. Plugin panics never unwind into host code.

## What's Needed from Nabil

The ABI crate (`onnx-runtime-ep-nxrt-abi`) should export:
- The `NxrtStatus` enum, symbol name constants, and function-pointer type aliases defined in `abi_contract.rs`
- Once landed, this host crate's `abi_contract` module will be replaced by a re-export

## Integration Test Guidance for Pris

- Build a minimal `cdylib` test plugin that exports all 5 symbols
- Test: successful load → create → device_count → destroy lifecycle
- Test: plugin with wrong major version → `AbiVersionMismatch`
- Test: plugin whose factory returns `InternalError` → `FactoryFailed`
- Test: plugin that reports 0 devices → `ZeroDevices`
