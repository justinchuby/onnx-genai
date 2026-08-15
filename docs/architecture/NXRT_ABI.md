# nxrt ABI — Specification and Honest Landing Status

**Author:** Roy (Lead)
**Date:** 2026-08-11
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)
**HEAD at time of writing:** `c1d2556b5`

---

## ⚠ Single Source of Truth — How We Broke It and the Rule That Prevents Recurrence

**What happened:** `onnx-runtime-ep-nxrt-host` was written without access to
`onnx-runtime-ep-nxrt-abi`. The host carried its own private `abi_contract.rs`
with a completely different symbol protocol. `onnx-runtime-ep-nxrt-testplugin`
declared its own `[workspace]` so `cargo check --workspace` never built it.
The tree compiled green while nothing was connected.

**The rule going forward:**
> Consumers of the nxrt dynamic ABI depend on `onnx-runtime-ep-nxrt-abi`.
> Nobody redefines the contract locally.

This is now enforced structurally: `onnx-runtime-ep-nxrt-host/Cargo.toml` lists
`onnx-runtime-ep-nxrt-abi = { workspace = true }` as a real dependency; the
duplicate `abi_contract.rs` is deleted; `onnx-runtime-ep-nxrt-testplugin` is a
genuine workspace member and exports through the macro shipped in that crate.

---

## Preamble — What "nxrt ABI" Means Today

| Surface | Committed at HEAD `c1d2556b5` | Location |
|---|---|---|
| **Rust `ExecutionProvider` trait** | ✅ | `crates/onnx-runtime-ep-api/src/provider.rs` |
| **ORT plugin-EP C ABI adapter** | ✅ | `crates/onnx-runtime-ep-plugin/` |
| **Native nxrt dynamic ABI** (`onnx-runtime-ep-nxrt-abi`, `onnx-runtime-ep-nxrt-host`) | ✅ Committed | `crates/onnx-runtime-ep-nxrt-abi/`, `crates/onnx-runtime-ep-nxrt-host/`, `crates/onnx-runtime-ep-nxrt-testplugin/` |

> **Test status as of HEAD `c1d2556b5`:**
> `onnx-runtime-ep-nxrt-abi`: **32/32 passing** (4 ignored doc-tests).
> `onnx-runtime-ep-nxrt-host`: **10/10 round-trip passing** (env-var race fixed via `ENV_MUTEX`).
> See §6.10 for details.

---

## 1. The nxrt Rust Trait ABI

### 1.1 Location

`crates/onnx-runtime-ep-api/src/provider.rs` — trait `ExecutionProvider`, sealed
behind `Send + Sync`.

### 1.2 Required methods (must be implemented by every EP)

| Method | Contract |
|---|---|
| `name() -> &str` | Snake-case identifier; stable across versions (used as map key). |
| `device_type() -> DeviceType` | Declared device class (Cpu, Cuda, …). |
| `device_id() -> DeviceId` | Unique (type, ordinal) pair. |
| `initialize(&mut self, config: &EpConfig) -> Result<()>` | Called once before any dispatch. May fail; EP is unusable on error. |
| `shutdown(&mut self) -> Result<()>` | Symmetric with `initialize`. |
| `supports_op(…) -> KernelMatch` | Capability query per node. Every `Unsupported` must carry an actionable reason. |
| `get_kernel(…) -> Result<Box<dyn Kernel>>` | Returns a runnable kernel for a node the EP claimed. |
| `allocate(size, align) -> Result<DeviceBuffer>` | Device memory allocation. |
| `deallocate(buf)` | Frees a buffer previously returned by `allocate`. |
| `copy(src, dst, size) -> Result<()>` | Synchronous device-to-device copy. |
| `copy_async(src, dst, size) -> Result<Fence>` | Asynchronous copy; caller must wait on the `Fence`. |
| `sync() -> Result<()>` | Synchronize all outstanding device work. |

### 1.3 Optional methods with meaningful defaults

| Method | Default | Override when |
|---|---|---|
| `capabilities()` | Stock (none) | EP supports weight-paging or other executor contract extensions. |
| `page_lazy_weight(key, weight, source)` | `Ok(None)` | EP manages a residency cache (CUDA EP). |
| `prefetch_lazy_weight(key, weight, source)` | `Ok(false)` | EP can pipeline prefetch ahead of need. |
| `wait_fence(fence)` | No-op | Device stream must stall for an upstream event. |
| `record_compute_fence()` | No-op | WAR fence between compute and transfer streams. |
| `copy_wait_fence(fence)` | No-op | Transfer stream wait. |
| `copy_from_host / copy_to_host` | Provided default using `copy` | Can specialise for HTD/DTH paths. |
| `device_argmax_supported()` | `false` | EP has a native argmax kernel. |
| `begin/end/abort_device_graph_capture` | Err (not supported) | EP supports CUDA-style graph capture. |
| `custom_passes()` | None | EP has graph-optimization passes. |

### 1.4 Ownership rules

- **Owner-frees:** every `allocate` pairs with exactly one `deallocate`. `DeviceBuffer`
  has no `Drop`; a dropped handle leaks but never double-frees. The executor holds
  the canonical reference.
- **No cross-EP free:** `deallocate` and `copy` implementations must assert that
  the buffer's device matches this EP's device id. Cross-EP operations are
  undefined behaviour at the hardware level.
- **Kernel lifetime:** a `Box<dyn Kernel>` returned by `get_kernel` is valid for
  the session. The executor does not call `get_kernel` again for a cached kernel.
- **Initialize/shutdown symmetry:** an EP that is `initialize`d will always receive
  a matching `shutdown` call. An EP that fails `initialize` will not receive
  `shutdown`.

### 1.5 Error model

`Result<T>` = `Result<T, EpError>`. Every public error should carry context
identifying the EP, the op, and the reason for decline. Bare `Err(…)` without a
reason string fails Justin's claim-discipline rule.

---

## 2. The ORT Plugin-EP C ABI Adapter

*(Committed at HEAD. Full specification in `docs/ep-plugin/EP_PLUGIN_EXPORT_ABI_TRUTH.md`.)*

### 2.1 Location

`crates/onnx-runtime-ep-plugin/` — the `export_ep_factories!` macro and its
supporting modules (`ep.rs`, `factory.rs`, `compute.rs`, `device.rs`, `status.rs`,
`kernel_ctx.rs`, `transfer.rs`).

### 2.2 Exported symbol names

ORT's `dlopen` loader (`onnxruntime_c_api.h` line 5579) resolves exactly these
two symbols:

| Symbol | Purpose | Source constant |
|---|---|---|
| `CreateEpFactories` | Entry point: create one `OrtEpFactory` per EP. | `EXPORT_SYMBOL_CREATE = b"CreateEpFactories"` |
| `ReleaseEpFactory` | Symmetric release. | `EXPORT_SYMBOL_RELEASE = b"ReleaseEpFactory"` |

### 2.3 Macro usage

```rust
use onnx_runtime_ep_plugin::export_ep_factories;
export_ep_factories!(|| MyExecutionProvider::new());
```

### 2.4 Version negotiation

Reads `OrtApiBase.GetApi(ORT_API_VERSION)` on every `CreateEpFactories` call.
Null return → hard error. `AtomicPtr` with `Acquire`/`Release` ordering. ORT ABI
is backwards-compatible; a plugin at version N works on host at version M ≥ N
(same major).

### 2.5 Panic containment

Every `extern "C"` callback wraps its body in `std::panic::catch_unwind`.
Status-returning callbacks produce an error `OrtStatus` on panic. Void-returning
callbacks swallow the panic (leaking > UB across C boundary).

### 2.6 Ownership and lifetime contracts — ORT ABI lessons

**Lesson 1 — One documented owner.** The `OrtMemoryInfo` use-after-free: ORT
stores the raw pointer passed to `EpDevice_AddAllocatorInfo`. It must outlive the
`OrtEpDevice`. Do NOT call `ReleaseMemoryInfo` after `AddAllocatorInfo`. Root cause
and fix: commit `c92838d`.

**Lesson 2 — Callback-frame lifetimes are not optional.** The `OrtKernelContext`
pointer is valid only for the duration of `compute_execute`. Storing it beyond the
call is undefined behaviour.

### 2.7 The C ABI / Rust trait parity rule (pinned and tested)

```
C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }
```

Nine tests in `crates/onnx-runtime-ep-plugin/tests/trait_cabi_parity.rs`.
Confirmed `Declined`: opset-13 `Unsqueeze` with data-dependent axes, `NonZero`.
Squeeze, ReduceMean, Conv resolve — do not mark Declined.

---

## 3. Device Surfaces (`device.rs`, `transfer.rs`)

`crates/onnx-runtime-ep-plugin/src/device.rs` — committed.
`crates/onnx-runtime-ep-plugin/src/transfer.rs` — committed.

| Type | Role |
|---|---|
| `DeviceAllocator` | `#[repr(C)]` struct with `OrtAllocator` vtable as first field. |
| `DeviceSyncStream` | `OrtSyncStreamImpl` vtable projection. |
| `DeviceSupport` | Maps `DeviceType` → `OrtHardwareDeviceType`; creates `OrtMemoryInfo_V2`, `OrtHardwareDevice`, `OrtEpDevice`. |
| `DeviceDataTransfer` | `OrtDataTransferImpl` vtable projection. Fail-closed: `CanCopy` returns false for unsupported directions. |

---

## 4. Inbound Loader — Loading a Foreign ORT Plugin EP

`crates/onnx-runtime-ep-api/src/abi/runtime.rs` — committed.

`PluginRuntime::load(path, registration_name)`:
1. Opens the shared library (`dlopen` on Linux).
2. Resolves `b"CreateEpFactories"` — hard error if absent.
3. Optionally resolves `b"ReleaseEpFactory"` — if absent, leaks on unload.
4. Calls `CreateEpFactories` with `OrtApiBase` from `ort_api_base()`.
5. Expects at least one factory; errors on zero.

`LegacyOrtEp` wraps a `PluginRuntime` as `dyn ExecutionProvider`.

---

## 5. How nxrt Improves on ORT's ABI

**ORT Lesson 1 — One documented owner.** The nxrt Rust trait statically tracks
ownership: `DeviceBuffer`, `PagedWeight`, `Fence` are owned values. Cannot create
a second owner without unsafe code.

**ORT Lesson 2 — Callback-frame lifetimes are not optional.** `TensorView` borrows
from the kernel context and cannot outlive it. The borrow checker prevents the bug
at compile time.

**A third lesson, from the duplicate-ABI failure (see §0):** The ABI definition
must have a single source of truth. A private `abi_contract.rs` that re-defines
the protocol locally breaks the invariant silently — `cargo check` will never
catch it if the inconsistent crate isn't a workspace member.

**A fourth lesson, from B3 — Status-message ownership (cross-module free):**
Memory allocated by a plugin and freed by the host (or vice versa) is undefined
behaviour when the two sides link against different C runtimes. This is the norm
on Windows and common with Rust `cdylib` boundaries everywhere. The nxrt ABI
enforces this with a **pure-value-type rule**: `NxrtStatus` carries a fixed inline
`[u8; 256]` message buffer (264 bytes total). No heap allocation, no pointers, no
cross-module free. The struct is returned by value and can be `memcpy`'d freely.
See `crates/onnx-runtime-ep-nxrt-abi/src/status.rs`.

> **ABI contract:** No nxrt ABI type may contain a pointer to memory that the
> other side of the boundary is expected to free. If a value must carry variable-
> length data, use a fixed inline buffer with a length field and truncation
> semantics.

**A fifth lesson, from B2/B3 — `c_char` portability:**
`std::ffi::c_char` is `i8` on x86_64 and `u8` on aarch64. Casting a pointer
with `as *const i8` instead of `as *const c_char` is unsound on ARM and causes
type-mismatch errors or UB. This class of bug has bitten this project twice:
once in the `ReleaseEpFactory` ABI test (B2, where an arm64/macOS failure was
misdiagnosed as an ORT ABI discrepancy) and once in nxrt status handling (B3).

> **ABI rule:** All FFI string/byte pointers must use `c_char`, never `i8` or
> `u8` directly. `grep -rn "as \*const i8\|as \*mut i8" crates/` must return
> zero hits in ABI-crossing code.

---

## 6. Native nxrt Dynamic ABI — Committed at `c1d2556b5`

### 6.1 Overview

The native nxrt dynamic ABI ships across three crates:

| Crate | Role |
|---|---|
| `onnx-runtime-ep-nxrt-abi` | Single source of truth: `#[repr(C)]` types, symbol constants, `export_nxrt_ep_factories!` macro, testing override types |
| `onnx-runtime-ep-nxrt-host` | Host loader: `dlopen` → negotiate → create factories → `NxrtPlugin` struct |
| `onnx-runtime-ep-nxrt-testplugin` | Genuine workspace-member `cdylib` fixture; exports through the `export_nxrt_ep_factories!` macro |

### 6.2 Exported symbols

A conforming nxrt plugin exports exactly two symbols:

| Symbol | C signature | Purpose |
|---|---|---|
| `NxrtNegotiate` | `fn(request: *const NxrtNegotiateRequest, response_out: *mut NxrtNegotiateResponse) -> NxrtStatus` | Version handshake — called first |
| `NxrtCreateEpFactories` | `fn(out_factories: *mut *mut NxrtEpFactoryVtable, max_factories: usize, out_num: *mut usize) -> NxrtStatus` | Factory creation — called if negotiation succeeds |

Symbol name constants in the ABI crate: `NXRT_SYMBOL_NEGOTIATE = b"NxrtNegotiate"`,
`NXRT_SYMBOL_CREATE_EP_FACTORIES = b"NxrtCreateEpFactories"`.

### 6.3 Version negotiation rules

ABI version at `c1d2556b5`: **major=1, minor=0**.

1. Host fills `NxrtNegotiateRequest { struct_size, host_range: NxrtVersionRange { major_min, major_max, minor_max } }`.
2. Plugin's `NxrtNegotiate` checks compatibility and fills `NxrtNegotiateResponse { struct_size, agreed_major, agreed_minor, plugin_range, capability_flags }`.
3. Host calls `validate_negotiation` (from `onnx-runtime-ep-nxrt-abi::version`):
   - **Major mismatch → hard reject.** `agreed_major` outside `[host_range.major_min, host_range.major_max]` → reject with actionable error.
   - **Plugin minor newer than host → reject.** `agreed_minor > host_range.minor_max` → reject. The host cannot safely call vtable slots it doesn't know.
   - **Unknown capability bits → reject (fail closed).** Any bit in `capability_flags` outside `NXRT_CAP_KNOWN_MASK` → reject.
4. If validation passes, host calls `NxrtCreateEpFactories`.

**`NXRT_CAP_KNOWN_MASK`** at current version:

| Flag | Value | Meaning |
|---|---|---|
| `NXRT_CAP_DEVICE_ENUMERATION` | `1 << 0` | Factory supports device enumeration |
| `NXRT_CAP_ALLOCATOR` | `1 << 1` | Plugin supports custom allocators |
| `NXRT_CAP_STREAM_SYNC` | `1 << 2` | Plugin supports stream/sync primitives |
| `NXRT_CAP_EP_CONTEXT` | `1 << 3` | Plugin supports compiled kernel caching |

Setting any bit outside this mask causes the host to reject the plugin
(`fail closed` — the comment on `NXRT_CAP_KNOWN_MASK` in `version.rs`).

### 6.4 Forward compatibility via `struct_size`

Every vtable struct (`NxrtEpFactoryVtable`, `NxrtEpVtable`, `NxrtKernelVtable`,
`NxrtAllocatorVtable`) carries `struct_size: u32` as its first field. A host only
reads fields up to `min(its_known_size, reported_size)`. New fields are appended at
the end and guarded by a minor version bump.

**Rules:**
- An older host seeing a larger struct (newer plugin with minor bump) ignores
  trailing bytes — safe because the default for every new field must be a no-op
  when treated as zero/null.
- A newer host seeing a smaller struct (older plugin) treats absent fields as
  zero/null. This means fail closed, not silent success.
- Adding a field WITHOUT bumping the minor version is a bug.

`NxrtNegotiateRequest` and `NxrtNegotiateResponse` also carry `struct_size` for
the same reason.

### 6.5 Ownership contract (host owns everything)

Per `crates/onnx-runtime-ep-nxrt-abi/src/vtable.rs` doc comments:

- **Factory:** created by `NxrtCreateEpFactories`. **Host owns.** Host must call
  `factory.release(factory.ctx)` exactly once when done.
- **EP:** created by `factory.create_ep(ctx, device_ordinal, &mut ep_ptr)`. **Host owns.**
  Host must call `ep.release(ep.ctx)` exactly once.
- **Kernel:** created by `ep.compile(...)`. **Host owns.** Host must call
  `kernel.release(kernel.ctx)` exactly once.
- **Allocator:** obtained from `ep.get_allocator(...)`. **Host owns.** Host must call
  `allocator.release(allocator.ctx)` exactly once.

No separate free symbol. Every type carries its own `release` function pointer.
Symmetric releases prevent the ORT Lesson 1 failure mode.

**Borrowed pointers (valid only within the callback frame):**
- `NxrtTensorDesc.dims` — points into caller-owned memory; do not store.
- `op_types` and `input_descs` passed to `get_capability` — callback-frame only.
- Graph/node handles — do not outlive the `get_capability` call.

**Owned for the object's lifetime:**
- `NxrtEpVtable.name` — valid for the EP's lifetime (owned by EP ctx). Host must
  not free it.
- `NxrtEpFactoryVtable.name` — valid for the factory's lifetime.

### 6.6 Panic containment

Both `NxrtNegotiate` and `NxrtCreateEpFactories` generated by
`export_nxrt_ep_factories!` wrap their bodies in `std::panic::catch_unwind`.

- `NxrtNegotiate` panic → returns `NxrtStatusCode::InternalError`.
- `NxrtCreateEpFactories` panic → sets `*out_num = 0`, returns `InternalError`
  with message `"NxrtCreateEpFactories: constructor panicked (fail-closed)"`.

The two negative-test override macros (`export_nxrt_ep_negotiate_custom!` and
`export_nxrt_ep_create_custom!`) also wrap with `catch_unwind`.

### 6.7 Plugin authoring path — `export_nxrt_ep_factories!`

```rust
// In your cdylib crate's lib.rs:
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_factories;

export_nxrt_ep_factories!(|| {
    Box::new(MyExecutionProvider::new())
        as Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider>
});
```

The closure is called once per `NxrtCreateEpFactories` invocation. The macro
generates `NxrtNegotiate` and `NxrtCreateEpFactories` with full panic containment.
The `Cargo.toml` must have `crate-type = ["cdylib", "rlib"]` — `rlib` is needed
for unit tests to link against the crate's logic without going through dlopen.

### 6.8 Negative-test override macros

For fixture plugins that need to simulate failure conditions:

```rust
// Simulate wrong major version in NxrtNegotiate:
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_negotiate_custom;
use onnx_runtime_ep_nxrt_abi::testing::NxrtNegotiateOverride;
export_nxrt_ep_negotiate_custom!(NxrtNegotiateOverride::wrong_major(99));

// Simulate factory error:
use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_create_custom;
use onnx_runtime_ep_nxrt_abi::testing::NxrtCreateFactoriesOverride;
use onnx_runtime_ep_nxrt_abi::NxrtStatusCode;
export_nxrt_ep_create_custom!(NxrtCreateFactoriesOverride::error(NxrtStatusCode::DeviceError));
```

### 6.9 Host loading path — `load_nxrt_plugin`

`crates/onnx-runtime-ep-nxrt-host/src/loader.rs`:

1. `Library::new(path)` — `dlopen`. Returns `NxrtHostError::LibraryLoadFailed` on failure.
2. Resolve `NxrtNegotiate` symbol. Missing → `NxrtHostError::SymbolNotFound`.
3. Call `NxrtNegotiate(request, &mut response)`. Failure → `AbiVersionMismatch`.
4. `validate_negotiation(&host_range, &response)`. Failure → `AbiVersionMismatch`.
5. Resolve `NxrtCreateEpFactories` symbol. Missing → `SymbolNotFound`.
6. Call `NxrtCreateEpFactories(ptrs, MAX_FACTORIES=16, &mut num)`. Failure → `FactoryFailed`.
7. Zero factories → `NxrtHostError::ZeroDevices`.
8. Any null factory pointer (untrusted plugin) → release already-obtained factories, return `FactoryFailed`.

The returned `NxrtPlugin` holds `Arc<Library>` and `Arc<FactorySet>`. Any EP
instance derived from the plugin structurally shares the `Arc<Library>` — the
library cannot be unloaded while live references exist. `FactorySet::drop` releases
all factory vtables via their `release` pointers with `catch_unwind`.

### 6.10 Current test results (observed at HEAD `c1d2556b5`)

#### `onnx-runtime-ep-nxrt-abi` — 32/32 passing (4 ignored doc-tests)

```
test result: ok. 32 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Tests cover: negotiate success, major reject, minor min, null-pointer fail-closed,
unknown capability flags reject, known capability flags accept, version constants,
macro compile + both symbols, panic containment in constructor, custom negotiate
override, custom create override (zero factories), status inline buffer,
message truncation, struct size stability (264 bytes).

#### `onnx-runtime-ep-nxrt-host` — 10/10 round-trip passing

```
test result: ok. 10 passed; 0 failed; 0 ignored
```

The `full_lifecycle_negotiate_create_release` env-var race that was previously
failing is **fixed** — Pris added an `ENV_MUTEX` that serializes tests setting
`NXRT_TEST_PANIC` / `NXRT_TEST_FACTORY_ERROR` against the lifecycle test.

---

## 7. Running the Tests

```bash
# ORT C ABI adapter (no hardware required)
cargo test -p onnx-runtime-ep-plugin

# Inbound ORT loader (no hardware required)
cargo test -p onnx-runtime-ep-api

# nxrt native ABI crate (no hardware required)
cargo test -p onnx-runtime-ep-nxrt-abi

# nxrt host + roundtrip tests — all passing (see §6.10)
cargo test -p onnx-runtime-ep-nxrt-host

# Full workspace compile check (excludes cuda feature)
cargo check --workspace
```

As of HEAD `c1d2556b5`:
- `cargo check --workspace` — clean.
- `cargo test -p onnx-runtime-ep-plugin` — all tests passing (154 lib + 9 parity).
- `cargo test -p onnx-runtime-ep-nxrt-abi` — 32/32 passing.
- `cargo test -p onnx-runtime-ep-nxrt-host` — 10/10 passing.

CUDA EP tests require hardware; see `docs/execution/CUDA_EP_STATUS.md`.
