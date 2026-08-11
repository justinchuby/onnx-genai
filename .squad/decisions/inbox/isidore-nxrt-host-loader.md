# nxrt Host Loader — Reconciled Loading Contract

**Author:** Isidore  
**Date:** 2026-08-11 (updated)  
**Relates to:** §524 extension contract

## Reconciliation Summary

The duplicate `abi_contract.rs` (opaque-handle model) has been **deleted**. The host loader now depends on `onnx-runtime-ep-nxrt-abi` (Nabil's vtable model) as the single source of truth.

## Loading Contract (reconciled to Nabil's ABI)

1. **`load_nxrt_plugin(path)`** — opens the `.so`/`.dll`/`.dylib`, resolves `NxrtNegotiate` and `NxrtCreateEpFactories` eagerly, performs structured negotiation. Fails closed on any issue.
2. **Version negotiation** — calls `NxrtNegotiate` with `NxrtNegotiateRequest::current()`. Host-side validation via `validate_negotiation()`: rejects major mismatch, agreed minor > host minor, or unknown capability flags.
3. **Factory creation** — calls `NxrtCreateEpFactories`, obtains up to 16 factory vtable pointers. Validates non-null, ≥1 factory.
4. **`NxrtExecutionProvider::new(plugin, factory_index)`** — calls `factory.create_ep(ctx, device_ordinal, &mut ep_ptr)`, wraps as `dyn ExecutionProvider`.

## Required Plugin Symbols

| Symbol | Type | Purpose |
|--------|------|---------|
| `NxrtNegotiate` | `NxrtNegotiateFn` | Version + capability negotiation |
| `NxrtCreateEpFactories` | `NxrtCreateEpFactoriesFn` | Factory vtable creation |

## Vtable Ownership (host-side)

| Object | Created by | Released by |
|--------|-----------|-------------|
| Factory | `NxrtCreateEpFactories` | `factory.release(ctx)` in `FactorySet::drop` |
| EP | `factory.create_ep(...)` | `ep.release(ctx)` in `NxrtExecutionProvider::drop` |
| Kernel | `ep.compile(...)` | `kernel.release(ctx)` (when wired) |
| Allocator | `ep.get_allocator(...)` | `allocator.release(ctx)` (when wired) |

## Borrowed-Pointer Rules

- `NxrtTensorDesc.dims`: valid only within the callback frame → must be copied to `Vec<i64>` before returning.
- Factory/EP `name` pointer: valid for object lifetime → copied to `String` at creation time.
- Op type strings in `get_capability`: valid only within the call → copy before retaining.

## Lifetime Safety

`Library` stored in `Arc<Library>`. `NxrtPlugin`, `FactorySet`, and `NxrtExecutionProvider` all hold `Arc` clones. The library cannot be unloaded while any derived object exists.

## Panic Containment

`Drop` on both `FactorySet` and `NxrtExecutionProvider` wraps `release` calls in `catch_unwind`. Plugin panics never unwind into host code.

## Negative Path Behavior

| Scenario | Behavior |
|----------|----------|
| Missing library file | `NxrtHostError::LibraryLoadFailed` with path and OS reason |
| Missing symbol | `NxrtHostError::SymbolNotFound` naming the exact symbol |
| Negotiation failure (version/caps) | `NxrtHostError::AbiVersionMismatch` with both versions |
| Factory returns error | `NxrtHostError::FactoryFailed` with status message |
| Zero factories returned | `NxrtHostError::ZeroDevices` |
| Null factory pointer | `NxrtHostError::FactoryFailed` identifying the index |

## Nothing Missing from Nabil's ABI

The reconciliation is complete. All types needed by the host are exported by `onnx-runtime-ep-nxrt-abi`.
