# nxrt Native Dynamic EP ABI Design (§524 Native Half)

**Date:** 2026-08-11
**Author:** Nabil (ORT Plugin EP Engineer)
**Crate:** `crates/onnx-runtime-ep-nxrt-abi`

## Overview

The native nxrt dynamic ABI is the §524 counterpart to the outbound ORT plugin-EP ABI. It defines a stable C-compatible binary interface that any nxrt EP plugin exports from a `cdylib`, loadable at runtime via `dlopen`/`dlsym`.

## Exported Symbols

A conforming plugin exports exactly **two** symbols:

| Symbol | Purpose |
|--------|---------|
| `NxrtNegotiate` | Version handshake — must be called first |
| `NxrtCreateEpFactories` | Factory creation — called after successful negotiation |

No separate release symbols — release is via vtable function pointers on the returned objects.

## Version Negotiation

- **`NXRT_ABI_VERSION_MAJOR = 1`**, **`NXRT_ABI_VERSION_MINOR = 0`**
- Major version mismatch → **hard reject** with `NxrtStatusCode::VersionMismatch` and an actionable message.
- Minor version: agreed = min(plugin_minor, host_minor). Host newer than plugin is safe; plugin newer than host is rejected.
- Every struct carries a `struct_size: u32` first field for forward compatibility.
- Capability flags (`u64`) validated with `NXRT_CAP_KNOWN_MASK` — unknown bits → reject (fail closed).

## Ownership Rules (explicit, not implicit)

1. **Factory** (`NxrtEpFactoryVtable`): Created by `NxrtCreateEpFactories`. **Host owns.** Must call `factory.release(factory.ctx)`.
2. **EP** (`NxrtEpVtable`): Created by `factory.create_ep()`. **Host owns.** Must call `ep.release(ep.ctx)`.
3. **Kernel** (`NxrtKernelVtable`): Created by `ep.compile()`. **Host owns.** Must call `kernel.release(kernel.ctx)`.
4. **Allocator** (`NxrtAllocatorVtable`): Obtained from `ep.get_allocator()`. **Host owns.** Must call `allocator.release(allocator.ctx)`.
5. **Borrowed pointers** (e.g. `NxrtTensorDesc.dims`, op_types in `get_capability`): Valid only within the callback frame. Must not be stored.
6. **Name pointers** (`ep.name`, `factory.name`): Valid for the object's lifetime. Host must not free them.

## Improvements Over ORT Plugin ABI

| ORT ABI issue | nxrt ABI fix |
|---|---|
| `OrtMemoryInfo` use-after-free (ORT stored a pointer we freed) | Every pointer-returning function documents ownership in the vtable doc. Borrowed vs owned is always explicit. |
| Graph/node handles escaped callback frame | `get_capability` docs state all pointers are borrowed for call frame only. |
| Silent success on unsupported ops | Fail closed everywhere: unknown caps rejected, compile returns `NotImplemented`, capability returns 0 claims. |
| Version negotiation was implicit | Explicit two-step: negotiate first, create only on success. Struct-size guards future fields. |
| Panic UB across C boundary | Every extern "C" wrapped with `catch_unwind` via `catch_status_panic`/`catch_void_panic`. |

## Vtable Lifecycle

```
Host                              Plugin (cdylib)
─────                             ──────
dlopen(plugin.so)
dlsym("NxrtNegotiate")
  → NxrtNegotiate(&req, &mut resp)  → checks version, fills resp
  ← status (Ok or VersionMismatch)
dlsym("NxrtCreateEpFactories")
  → NxrtCreateEpFactories(...)      → returns factory vtable ptr
  ← status + factory*
factory.create_ep(ctx, ord, &mut ep) → returns EP vtable ptr
ep.get_capability(...)               → fills claims array
ep.compile(...)                      → returns kernel vtable ptr
kernel.execute(...)                  → runs computation
kernel.release(...)                  → frees kernel
ep.release(...)                      → frees EP
factory.release(...)                 → frees factory
dlclose(plugin.so)
```

## Public API Surface for Consumers

### For Isidore (host loader):

```rust
use onnx_runtime_ep_nxrt_abi::{
    // Symbol names to dlsym
    NXRT_SYMBOL_NEGOTIATE,           // b"NxrtNegotiate"
    NXRT_SYMBOL_CREATE_EP_FACTORIES, // b"NxrtCreateEpFactories"
    // Function pointer types
    NxrtNegotiateFn,
    NxrtCreateEpFactoriesFn,
    // Version negotiation structs
    NxrtNegotiateRequest,
    NxrtNegotiateResponse,
    NxrtVersionRange,
    NXRT_ABI_VERSION_MAJOR,
    NXRT_ABI_VERSION_MINOR,
    // Vtable types (to read/call through)
    NxrtEpFactoryVtable,
    NxrtEpVtable,
    NxrtKernelVtable,
    NxrtAllocatorVtable,
    NxrtTensorDesc,
    NxrtNodeCapability,
    // Status
    NxrtStatus,
    NxrtStatusCode,
    // Validation helper
    version::validate_negotiation,
    version::NXRT_CAP_KNOWN_MASK,
};
```

### For Pris (testing):

- `NxrtNegotiateRequest::current()` / `NxrtNegotiateResponse::zeroed()` for test setup
- `version::negotiate(req, resp)` callable directly for unit testing
- `vtable::create_ep_factories(...)` callable for in-process factory testing
- `status::catch_status_panic(...)` for panic containment testing
- `version::validate_negotiation(...)` for host-side validation testing
- Negative tests: pass version 99 major, unknown cap bits, null pointers

### For plugin authors (export macro):

```rust
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_factories;
export_nxrt_ep_factories!(|| MyExecutionProvider::new());
```

## Error Model

`NxrtStatus` is a `#[repr(C)]` struct: `{ code: NxrtStatusCode, _reserved: u32, message: *mut c_char }`. Codes are stable `#[repr(u32)]` enum values 0-7. Unknown codes must be treated as fatal (fail closed). Messages are heap-allocated CStrings owned by the receiver.

## Export Macros (shipped 2026-08-11)

### `export_nxrt_ep_factories!` — standard plugin authoring

```rust
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_factories;

export_nxrt_ep_factories!(|| {
    Box::new(MyEp::new()) as Box<dyn ExecutionProvider>
});
```

Emits `NxrtNegotiate` and `NxrtCreateEpFactories` with full panic containment. Fully-qualified paths (`::std::panic::catch_unwind`, `$crate::...`). No reliance on caller-scope names.

### `export_nxrt_ep_negotiate_custom!` — negative-test negotiate override

```rust
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_negotiate_custom;
use onnx_runtime_ep_nxrt_abi::testing::NxrtNegotiateOverride;

// Wrong major:
export_nxrt_ep_negotiate_custom!(NxrtNegotiateOverride::wrong_major(99));
// Unknown caps:
export_nxrt_ep_negotiate_custom!(NxrtNegotiateOverride::unknown_caps(1 << 63));
// Panicking:
export_nxrt_ep_negotiate_custom!(NxrtNegotiateOverride::panicking());
```

### `export_nxrt_ep_create_custom!` — negative-test factory override

```rust
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_create_custom;
use onnx_runtime_ep_nxrt_abi::testing::NxrtCreateFactoriesOverride;

// Error:
export_nxrt_ep_create_custom!(NxrtCreateFactoriesOverride::error(NxrtStatusCode::DeviceError));
// Zero factories:
export_nxrt_ep_create_custom!(NxrtCreateFactoriesOverride::zero());
// Panic:
export_nxrt_ep_create_custom!(NxrtCreateFactoriesOverride::panicking());
```

### Testing module public surface (`onnx_runtime_ep_nxrt_abi::testing`)

- `NxrtNegotiateOverride` — `normal()`, `wrong_major(u32)`, `unknown_caps(u64)`, `panicking()`
- `NxrtCreateFactoriesOverride` — `error(NxrtStatusCode)`, `zero()`, `panicking()`

Both have an `unsafe fn execute(...)` method that Pris can call from her fixture plugins or directly in tests. The macros wrap these with panic containment automatically.

### Complete re-exported public surface

```rust
// Top-level re-exports (lib.rs)
pub use status::{NxrtStatus, NxrtStatusCode};
pub use testing::{NxrtCreateFactoriesOverride, NxrtNegotiateOverride};
pub use version::{
    NXRT_ABI_VERSION_MAJOR, NXRT_ABI_VERSION_MINOR,
    NXRT_CAP_ALLOCATOR, NXRT_CAP_DEVICE_ENUMERATION,
    NXRT_CAP_EP_CONTEXT, NXRT_CAP_KNOWN_MASK, NXRT_CAP_STREAM_SYNC,
    NxrtNegotiateRequest, NxrtNegotiateResponse, NxrtVersionRange,
    validate_negotiation,
};
pub use vtable::{
    NxrtAllocatorVtable, NxrtEpFactoryVtable, NxrtEpVtable,
    NxrtKernelVtable, NxrtNodeCapability, NxrtTensorDesc,
};
// Constants
pub const NXRT_SYMBOL_NEGOTIATE: &[u8];
pub const NXRT_SYMBOL_CREATE_EP_FACTORIES: &[u8];
// Function pointer types
pub type NxrtNegotiateFn;
pub type NxrtCreateEpFactoriesFn;
// Macros
export_nxrt_ep_factories!
export_nxrt_ep_negotiate_custom!
export_nxrt_ep_create_custom!
```
