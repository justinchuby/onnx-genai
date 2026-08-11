# nxrt ABI — Specification and Honest Landing Status

**Author:** Roy (Lead)
**Date:** 2026-08-11
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)
**HEAD at time of writing:** `4212e090e` (committed)
**Working-tree state:** additional untracked files from parallel agents — see §6

---

## Preamble — What "nxrt ABI" Means Today

> **The code wins. This document describes what exists in the working tree, with
> an explicit distinction between committed (HEAD) and uncommitted (working tree)
> code. Do not read section labels as claiming completion.**

The phrase "nxrt ABI" covers three distinct surfaces. Committed status at HEAD
`4212e090e` is marked separately from the working-tree state (untracked files
from parallel agents, not yet committed).

| Surface | Committed at HEAD | In working tree | Location |
|---|---|---|---|
| **Rust `ExecutionProvider` trait** — the nxrt-native Rust ABI | ✅ | ✅ | `crates/onnx-runtime-ep-api/src/provider.rs` |
| **ORT plugin-EP C ABI adapter** — projects any nxrt EP through ORT's `dlopen` protocol | ✅ | ✅ | `crates/onnx-runtime-ep-plugin/` |
| **Native nxrt dynamic ABI** — `crates/onnx-runtime-ep-nxrt-abi/` (Nabil) + `crates/onnx-runtime-ep-nxrt-host/` (Isidore) | 🔴 Not in HEAD | ⚠️ Untracked files exist but have an integration gap (§6.3) | `crates/onnx-runtime-ep-nxrt-abi/`, `crates/onnx-runtime-ep-nxrt-host/` |
| **Data-transfer adapter** — `transfer.rs` (Leon) | 🔴 Not in HEAD | ⚠️ Untracked | `crates/onnx-runtime-ep-plugin/src/transfer.rs` |

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

*(Committed at HEAD `4212e090e`)*

### 2.1 Location

`crates/onnx-runtime-ep-plugin/` — the `export_ep_factories!` macro and its
supporting modules (`ep.rs`, `factory.rs`, `compute.rs`, `device.rs`, `status.rs`,
`kernel_ctx.rs`, and working-tree `transfer.rs`).

### 2.2 Exported symbol names

ORT's `dlopen` loader (`onnxruntime_c_api.h` line 5579) resolves exactly these
two symbols:

| Symbol | Purpose | Source constant |
|---|---|---|
| `CreateEpFactories` | Entry point: create one `OrtEpFactory` per EP. | `EXPORT_SYMBOL_CREATE = b"CreateEpFactories"` |
| `ReleaseEpFactory` | Symmetric release. | `EXPORT_SYMBOL_RELEASE = b"ReleaseEpFactory"` |

The C typedef is `CreateEpApiFactoriesFn` and `ReleaseEpApiFactoryFn` — those are
type aliases, not the exported names. See `docs/EP_PLUGIN_EXPORT_ABI_TRUTH.md`
for the full header audit.

### 2.3 Macro usage

```rust
// In a cdylib crate (e.g. onnx-runtime-ep-cpu-plugin):
use my_ep::MyExecutionProvider;
use onnx_runtime_ep_plugin::export_ep_factories;

export_ep_factories!(|| MyExecutionProvider::new());
```

The macro generates `CreateEpFactories` and `ReleaseEpFactory` entry points,
both wrapped in `std::panic::catch_unwind`.

### 2.4 Version negotiation

The adapter reads `OrtApiBase.GetApi(ORT_API_VERSION)` on every
`CreateEpFactories` call. Null return → hard error, not null dereference. If
the host is newer, the plugin continues (ORT is backwards-compatible within the
same major version). The resolved `*const OrtApi` is stored in an `AtomicPtr`
with `Acquire`/`Release` ordering.

**Forward compatibility:** new `OrtApi` members are appended at the end of the
struct. A plugin built at version N accesses only the first N fields regardless
of the host's struct size. The plugin never takes `sizeof(OrtApi)` from the host.

### 2.5 Panic containment

Every `extern "C"` callback wraps its body in `std::panic::catch_unwind`. Status-
returning callbacks produce an error `OrtStatus` on panic. Void-returning callbacks
swallow the panic (leaking is preferable to unwinding across the C boundary).

### 2.6 Ownership and lifetime contracts

Two lessons from the ORT side:

1. **`OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo`:** ORT stores the raw
   pointer; it must outlive the `OrtEpDevice`. Do NOT call `ReleaseMemoryInfo`
   after `AddAllocatorInfo`. The use-after-free root cause and fix: commit
   `c92838d`.

2. **Handles must not outlive their callback frame** unless the ABI explicitly
   says otherwise. The `OrtKernelContext` pointer is valid only for the duration
   of `compute_execute`. Storing it beyond the call is undefined behaviour.

Additional contracts documented in `crates/onnx-runtime-ep-plugin/src/device.rs`
header comment for `OrtSyncStreamImpl` and `OrtAllocator`.

### 2.7 The C ABI / Rust trait parity rule (pinned and tested)

```
C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }
```

Nine tests in `crates/onnx-runtime-ep-plugin/tests/trait_cabi_parity.rs` pin this
rule. Confirmed `Declined` cases: opset-13 `Unsqueeze` with data-dependent axes,
`NonZero`. Squeeze, ReduceMean, and Conv resolve — do not mark them Declined.

---

## 3. Device Surfaces (`device.rs`, `transfer.rs`)

`crates/onnx-runtime-ep-plugin/src/device.rs` — committed at HEAD.
`crates/onnx-runtime-ep-plugin/src/transfer.rs` — working tree only (Leon).

| Type | Role | Status |
|---|---|---|
| `DeviceAllocator` | `#[repr(C)]` struct with `OrtAllocator` vtable as first field. | ✅ HEAD |
| `DeviceSyncStream` | `OrtSyncStreamImpl` vtable projection. | ✅ HEAD |
| `DeviceSupport` | Maps `DeviceType` → `OrtHardwareDeviceType`, creates `OrtMemoryInfo_V2`, creates `OrtHardwareDevice` and `OrtEpDevice`. | ✅ HEAD |
| `DeviceDataTransfer` | `OrtDataTransferImpl` vtable projection for copy directions. Fail-closed: `CanCopy` returns false for directions the EP doesn't genuinely support. Device pointers are never dereferenced on the host. | ⚠️ Working tree only |

The `transfer.rs` copy-direction matrix:

| Source | Destination | Method | Supported |
|---|---|---|---|
| Host (CPU) | Device (GPU) | `copy_from_host` | ✅ |
| Device (GPU) | Host (CPU) | `copy_to_host` | ✅ |
| Device (GPU:i) | Device (GPU:i) | `copy` (same device) | ✅ |
| Device (GPU:i) | Device (GPU:j) | `copy` (cross-device) | ❌ — `CanCopy` returns false |
| Host | Host | ORT handles (not our EP) | ❌ — `CanCopy` returns false |

---

## 4. Inbound Loader — Loading a Foreign ORT Plugin EP

`crates/onnx-runtime-ep-api/src/abi/runtime.rs` — committed at HEAD.

`PluginRuntime::load(path, registration_name)`:
1. Opens the shared library via `libloading` (`dlopen` on Linux).
2. Resolves `b"CreateEpFactories"` — hard error if absent.
3. Optionally resolves `b"ReleaseEpFactory"` — if absent, leaks on unload.
4. Calls `CreateEpFactories` with `OrtApiBase` from `ort_api_base()`.
5. Expects at least one factory; errors on zero.

`LegacyOrtEp` wraps a `PluginRuntime` as `dyn ExecutionProvider`. Its
`supports_op` declines individual node dispatch; capability is claimed via
`PluginExecutionPlan::compile` (graph-level).

---

## 5. How nxrt Improves on ORT's ABI

The ORT ABI's two painful lessons:

**Lesson 1 — One documented owner.** The `OrtMemoryInfo` use-after-free was caused
by ambiguous ownership. The nxrt Rust trait eliminates this: `DeviceBuffer`,
`PagedWeight`, and `Fence` are owned values; the borrow checker prevents a second
owner.

**Lesson 2 — Callback-frame lifetimes are not optional.** The nxrt Rust trait
encodes this as a borrow: `TensorView` borrows from the kernel context and cannot
outlive it — the borrow checker prevents the bug at compile time.

**"Evolving the ORT ABI toward nxrt" concretely means:**
1. The plugin adapter is a thin shim — ORT's C types are translated at the
   boundary and not re-exported into core logic.
2. The native nxrt ABI (§6) improves on both lessons by encoding them directly
   into its `#[repr(C)]` surface: struct-size versioning, explicit `release`
   vtable functions as the sole ownership transfer mechanism, and borrowed
   pointers only valid within a documented callback frame.

---

## 6. Native nxrt Dynamic ABI — Working Tree, NOT Committed, Integration Gap

> **This section describes working-tree code that is NOT committed to HEAD
> `4212e090e`. It is present as untracked files from parallel agents.**

### 6.1 Nabil's `crates/onnx-runtime-ep-nxrt-abi/`

Exports two symbols:
- `NxrtNegotiate(request: *const NxrtNegotiateRequest, response_out: *mut NxrtNegotiateResponse) -> NxrtStatus`
- `NxrtCreateEpFactories(out_factories: *mut *mut NxrtEpFactoryVtable, max_factories: usize, out_num: *mut usize) -> NxrtStatus`

Version negotiation via `NxrtNegotiateRequest`/`NxrtNegotiateResponse` structs.
Vtable-based ownership: every factory/EP/kernel/allocator is a `#[repr(C)]` struct
with `release(ctx)` as the free function. `struct_size` field enables forward
compatibility (older hosts ignore trailing bytes; newer hosts treat absent fields
as null/zero). Major.minor rules: same major required, host minor ≥ plugin minor.
Panic containment on both entry points.

The `export_nxrt_ep_factories!` macro generates both symbols from a user-supplied
constructor closure.

ABI version at time of writing: major=1, minor=0.

### 6.2 Isidore's `crates/onnx-runtime-ep-nxrt-host/`

Resolves these symbols (from `abi_contract.rs`):
- `nxrt_abi_version` (writes out-params `major`, `minor`)
- `nxrt_create_ep(config_json, out_handle) -> NxrtStatus`
- `nxrt_destroy_ep(handle)`
- `nxrt_ep_name() -> *const c_char`
- `nxrt_device_count() -> u32`

Uses opaque `NxrtEpHandle` pointers and distinct create/destroy symbols rather
than vtable-embedded release functions.

### 6.3 Integration gap — symbol protocols do not match

**Nabil's ABI exports:** `NxrtNegotiate`, `NxrtCreateEpFactories`
**Isidore's loader expects:** `nxrt_abi_version`, `nxrt_create_ep`, `nxrt_destroy_ep`,
`nxrt_ep_name`, `nxrt_device_count`

These are completely different protocols. A plugin built with Nabil's
`export_nxrt_ep_factories!` macro will NOT be loadable by Isidore's
`load_nxrt_plugin`. Isidore's `abi_contract.rs` acknowledges this with a comment:

> "When Nabil's `onnx-runtime-ep-nxrt-abi` crate lands, this module should be
> replaced by a re-export from that crate."

The host crate was written without the ABI crate available. The two crates need a
reconciliation pass before either can be committed. **This is an integration gap
and the immediate blocker for the native nxrt dynamic ABI.**

### 6.4 Summary

The native nxrt dynamic ABI work has been written (working tree) but is:
1. Not committed to HEAD.
2. Has a symbol-protocol mismatch between the plugin-side ABI and the host loader.
3. Has no round-trip or negative tests committed.
4. Has no GPU hardware validation.

§524 remains incomplete. PR #762 stays draft.

---

## 7. Running the Tests

```bash
# Rust trait + ORT C ABI adapter (no hardware required) — committed
cargo test -p onnx-runtime-ep-plugin

# Inbound ORT loader (no hardware required) — committed
cargo test -p onnx-runtime-ep-api

# Full workspace compile check (excludes cuda feature) — committed
cargo check --workspace
```

As of HEAD `4212e090e`, `cargo check --workspace` is clean and
`cargo test -p onnx-runtime-ep-plugin` passes all tests.

CUDA EP tests require hardware; see `docs/CUDA_EP_STATUS.md`.

The nxrt round-trip tests (Pris) are not committed.


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

### 2.1 Location

`crates/onnx-runtime-ep-plugin/` — the `export_ep_factories!` macro and its
supporting modules (`ep.rs`, `factory.rs`, `compute.rs`, `device.rs`, `status.rs`,
`kernel_ctx.rs`).

### 2.2 Exported symbol names

ORT's `dlopen` loader (`onnxruntime_c_api.h` line 5579) resolves exactly these
two symbols:

| Symbol | Purpose | Source constant |
|---|---|---|
| `CreateEpFactories` | Entry point: create one `OrtEpFactory` per EP. | `EXPORT_SYMBOL_CREATE = b"CreateEpFactories"` |
| `ReleaseEpFactory` | Symmetric release. | `EXPORT_SYMBOL_RELEASE = b"ReleaseEpFactory"` |

The C typedef is `CreateEpApiFactoriesFn` (note the trailing `Fn`) and
`ReleaseEpApiFactoryFn` — those are type aliases, not the exported names.
A plugin that exports `CreateEpApiFactories` (no `Fn`) will NOT be loaded by ORT.
See `docs/EP_PLUGIN_EXPORT_ABI_TRUTH.md` for the full header audit.

### 2.3 Macro usage

```rust
// In a cdylib crate (e.g. onnx-runtime-ep-cpu-plugin):
use my_ep::MyExecutionProvider;
use onnx_runtime_ep_plugin::export_ep_factories;

export_ep_factories!(|| MyExecutionProvider::new());
```

The macro generates:
1. `#[unsafe(no_mangle)] pub unsafe extern "C" fn CreateEpFactories(…)` — calls
   `factory::create_ep_factories` with the user-supplied constructor.
2. `#[unsafe(no_mangle)] pub unsafe extern "C" fn ReleaseEpFactory(…)` — drops the
   factory box.

Both entry points are wrapped in `std::panic::catch_unwind`. See §2.5.

### 2.4 Version negotiation

The adapter reads `OrtApiBase` on every `CreateEpFactories` call:

```
api_base.GetApi(ORT_API_VERSION) -> *const OrtApi
```

`ORT_API_VERSION` is the version the plugin was built against (from
`onnxruntime_c_api.h`). If ORT's host is older than the plugin's build version,
`GetApi` returns null. The adapter fails cleanly with an error status rather than
dereferencing a null vtable. If the host is newer, the plugin continues — ORT is
required to be backwards-compatible at the same major version. The `AtomicPtr`
storing the resolved `OrtApi` pointer uses `Acquire`/`Release` ordering so
concurrent calls on different threads see a consistent pointer.

**Major/minor rule (from ORT's ABI contract, not nxrt's):** ORT's version integer
is `(major * 100 + minor)`. A plugin built against version N may be loaded by a
host at version M ≥ N if same major. A host may add new vtable members after the
plugin's supported version; those are outside the plugin's `OrtApi` struct and
must not be accessed. The adapter never accesses ORT API members beyond what it was
compiled against.

**Adding ABI fields without corruption:** New `OrtApi` members are appended at the
end of the struct. A plugin built at version N sees a struct with exactly N's
fields; the host at version M > N has a larger struct but the plugin's pointer
arithmetic addresses only the first N fields. This is safe in C/Rust because
`OrtApi` is `#[repr(C)]` and accessed through a raw pointer — the plugin never
takes `sizeof(OrtApi)` from the host.

### 2.5 Panic containment

Every `extern "C"` callback generated by or called from the adapter wraps its
body in `std::panic::catch_unwind`. The boundaries are:

| Callback | Location | Panic action |
|---|---|---|
| `CreateEpFactories` | `lib.rs` (macro expansion) | Sets `*out_num = 0`; returns error `OrtStatus`. |
| `ReleaseEpFactory` | `lib.rs` (macro expansion) | Silently swallows (no return channel; leaking > UB). |
| `get_capability` | `ep.rs` | Returns error `OrtStatus`. |
| `ep_compile` | `ep.rs` | Returns error `OrtStatus`; partial `NodeComputeInfo`s cleaned up. |
| `release_node_compute_infos` | `ep.rs` | Swallowed (no return channel). |
| `compute_execute` | `compute.rs:552` | Returns error `OrtStatus` (N1 fix). |
| `compute_release_state` | `compute.rs` | Swallowed (no return channel). |

A panic must never unwind across a C ABI boundary. These wrappers enforce that
guarantee. Leaking resources on panic inside a no-return-channel callback is the
correct trade-off.

### 2.6 Ownership and lifetime contracts

These were learned from the `OrtMemoryInfo` use-after-free bug (commit `c92838d`,
root-cause of "DeviceType:-112 garbage"):

1. **`OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo`:** ORT stores the raw
   pointer and does NOT copy it. It must outlive the `OrtEpDevice`. ORT releases
   it when `ReleaseEpDevice` is called. The plugin must NOT call `ReleaseMemoryInfo`
   after `AddAllocatorInfo`. (ORT `onnxruntime_ep_c_api.h` ~line 1092–1111)

2. **`OrtSyncStreamImpl`:** ORT calls `Release` on the vtable when done. The
   implementation must release resources in its `Release` callback, not before.
   (ORT header ~line 204–258)

3. **`OrtAllocator`:** ORT calls `OrtEpFactory::ReleaseAllocator` to free. The
   factory must track its allocator's lifetime. (header ~line 2835)

4. **`OrtHardwareDevice`:** Created via `CreateHardwareDevice`; the returned
   `OrtEpDevice` array entries are ORT-owned after the call.

5. **Handles must not outlive their callback frame unless the ABI says otherwise.**
   The `OrtKernelContext` pointer passed to `compute_execute` is valid only for
   the duration of that call. Storing it (or any pointer derived from it) beyond
   the call is undefined behaviour.

### 2.7 The C ABI / Rust trait parity rule (pinned and tested)

```
C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }
```

Every node the C ABI claims is also supported by the trait. The converse is NOT
required: a node the trait supports may be excluded if output shapes cannot be
inferred at compile time. This prevents over-claiming. Nine tests in
`crates/onnx-runtime-ep-plugin/tests/trait_cabi_parity.rs` pin this rule.

Confirmed `Declined` cases: opset-13 `Unsqueeze` with data-dependent axes,
`NonZero`. Squeeze, ReduceMean, and Conv resolve — do not mark them Declined.

---

## 3. Device Surfaces (`device.rs`)

`crates/onnx-runtime-ep-plugin/src/device.rs` generalises device enumeration,
allocator, and stream beyond CPU-only. Landed in commit `2da0c4e7f`; finalised
in `3ab0ded68`.

| Type | Role |
|---|---|
| `DeviceAllocator` | `#[repr(C)]` struct with `OrtAllocator` vtable as first field. `*mut DeviceAllocator` → `*mut OrtAllocator` cast is valid. Holds raw EP pointer (EP must outlive allocator per ORT factory lifetime). |
| `DeviceSyncStream` | `OrtSyncStreamImpl` vtable projection. `Release` callback frees the stream resources. |
| `DeviceSupport` | Utility: maps `DeviceType` → `OrtHardwareDeviceType`, creates `OrtMemoryInfo_V2`, creates `OrtHardwareDevice` and `OrtEpDevice`. |

Tested by 30 mock-device tests in `device::tests::*` on a machine with no GPU
(tests use mock EP implementations — not GPU hardware).

---

## 4. Inbound Loader — Loading a Foreign ORT Plugin EP

`crates/onnx-runtime-ep-api/src/abi/runtime.rs` implements the inbound direction:
nxrt hosts a foreign ORT plugin EP.

`PluginRuntime::load(path, registration_name)`:
1. Opens the shared library via `libloading` (`dlopen` on Linux, `LoadLibrary`
   on Windows).
2. Resolves `b"CreateEpFactories"` — hard error if absent.
3. Optionally resolves `b"ReleaseEpFactory"` — if absent, leaks on unload
   (tolerated for compatibility with older plugins).
4. Calls `CreateEpFactories` with `OrtApiBase` from `ort_api_base()`.
5. Expects at least one factory in the output array; errors on zero.

`LegacyOrtEp` wraps a `PluginRuntime` as a `dyn ExecutionProvider`. Its
`supports_op` declines individual node dispatch; capability is claimed via
`PluginExecutionPlan::compile` (graph-level, not op-level).

---

## 5. How nxrt Improves on ORT's ABI — and What "Evolving ORT Toward nxrt" Means

The ORT ABI's two painful lessons:

**Lesson 1 — One documented owner.** The `OrtMemoryInfo` use-after-free was caused
by ambiguous ownership: the plugin created the pointer, passed it to ORT, then
freed it. ORT read a dangling pointer. The nxrt Rust trait eliminates this class
of bug because ownership is statically tracked: `DeviceBuffer`, `PagedWeight`,
and `Fence` are owned values; you cannot accidentally create a second owner without
unsafe code.

**Lesson 2 — Callback-frame lifetimes are not optional.** The `OrtKernelContext`
lesson: a pointer received in a callback is valid for that call only. ORT's ABI
makes this a footgun (raw pointer, no lifetime). The nxrt Rust trait encodes it
as a borrow: `TensorView` borrows from the kernel context and cannot outlive it.
The borrow checker prevents the bug at compile time.

**"Evolving the ORT ABI toward nxrt" concretely means:**
1. The plugin adapter (`onnx-runtime-ep-plugin`) is a thin shim — ORT's C types are
   translated into nxrt Rust types at the boundary and not re-exported into core
   logic. Future ORT ABI versions affect only this shim.
2. When nxrt's native `extern "C"` ABI is eventually implemented (see §6), it will
   export the same semantic contracts as the Rust trait, making the ORT layer
   optional.

---

## 6. Native nxrt Dynamic ABI — NOT YET IMPLEMENTED

Standing directive §524 requires "a first-class native nxrt `extern "C"` ABI" in
addition to the ORT C ABI. This surface does not exist as of HEAD `4212e090e`.

There is no:
- `crates/onnx-runtime-ep-nxrt-abi/` (would hold the `#[repr(C)]` surface,
  version negotiation structs, and export macro)
- `crates/onnx-runtime-ep-nxrt-host/` (would hold the `dlopen` loader and
  adapter back to `dyn ExecutionProvider`)
- Any `extern "C"` symbol named outside the ORT protocol

This gap is accurately recorded in `docs/EP_PLUGIN_EXPORT_PR.md` §524 table row
"Native nxrt dynamic ABI: 🔴 Not implemented."

When designed, it should encode the ownership and lifetime lessons from §5 directly
into its ABI (versioned structs, explicit ownership transfer flags, no naked
`*mut T` across the boundary without a paired release callback).

---

## 7. Running the Tests

```bash
# Rust trait + ORT C ABI adapter (no hardware required)
cargo test -p onnx-runtime-ep-plugin

# Inbound loader (no hardware required)
cargo test -p onnx-runtime-ep-api

# Full workspace compile check
cargo check --workspace
```

As of HEAD `4212e090e`, `cargo check --workspace` is clean and
`cargo test -p onnx-runtime-ep-plugin` passes all tests (see
`docs/EP_PLUGIN_EXPORT_PR.md` Validation section for verbatim output).

CUDA EP tests require hardware; see `docs/CUDA_EP_STATUS.md`.
