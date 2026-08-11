# EP Plugin Export Architecture

> **Status (2026-08-10, Roy re-verified — branch `squad/ep-plugin-export`):**
> Both adapter crates **compile cleanly** (`cargo check -p onnx-runtime-ep-plugin`
> and `cargo check -p onnx-runtime-ep-cpu-plugin` pass with warnings only).
> Upstream ORT 1.27.0 now loads, registers, and executes our Rust CPU EP as a
> real plugin library via `RegisterExecutionProviderLibrary` → `GetEpDevices` →
> `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run`.
> Numerical outputs are correct. See §Validation for exact test results.
>
> **Security status:** Holden's re-audit (2026-08-10T21:30Z) found 3 open
> findings (C1 partially resolved, two new). The EP is **not yet ship-cleared**
> for production. See §Security.
>
> **Author:** Roy (Lead) — 2026-08-10; Implementation: Nabil, Deckard, Leon,
> Pris — 2026-08-10
> **Standing directive:** Extension contract §524 — every extension seam exposes
> a stable dynamic C ABI **and** a first-class Rust trait, shipped in parallel.

## TRUE NOW (verified 2026-08-10)

These facts were verified personally by running the commands below on the branch.

| Claim | Evidence |
|-------|---------|
| Both crates compile | `cargo check -p onnx-runtime-ep-cpu-plugin` exits 0 |
| 82 unit tests pass in `onnx-runtime-ep-plugin` | `cargo test -p onnx-runtime-ep-plugin --lib` |
| 10 ORT integration tests pass | individual `cargo test --test plugin_ort_e2e -- <name>` |
| ORT loads and runs our EP end-to-end | `ort_loads_our_ep_and_runs_model` passes |
| `GetEpDevices` finds `cpu_ep` | `ort_register_ep_library` passes |
| Unsupported ops decline, not crash | `ort_unsupported_op_declines_not_crashes` passes |

## CURRENT STATUS (as of 2026-08-11, commit `c1d2556b5`)

| Item | Status |
|------|--------|
| `conformance_two_sessions` | ✅ **Passing** — was `#[ignore]`d due to test-assertion bug (fixed by Pris); `EpDevice_EpName` returns factory name `"cpu_ep"`, not registration key |
| `stress_register_run_unregister_cycles` (25 cycles) | ✅ **Passing** — UAF fix verified; `DeviceType:-112` regression gone |
| CUDA EP | ⛔ Blocked — no CUDA toolkit/GPU; design work (allocator/stream callbacks) also remains |
| nxrt-native Rust trait ABI for EP | 🟡 Partially wired; not independently tested as a Rust trait surface |
| `OrtCompiledModelCompatibilityInfo` | Returns `None`; deferred |
| Holden security sign-off | ✅ **🟡 YELLOW — May ship** (2026-08-10T22:42Z) — all ship-blockers resolved; 2 LOW advisories recorded for post-merge |

---

## Problem

Our in-repo execution providers (`onnx-runtime-ep-cpu`, `onnx-runtime-ep-cuda`)
implement the `ExecutionProvider` Rust trait
(`crates/onnx-runtime-ep-api/src/provider.rs`). They are consumable natively
inside nxrt, but **upstream ONNX Runtime cannot load them at all**: no crate
exports `CreateEpFactories` / `ReleaseEpFactory`, no `Cargo.toml` sets
`crate-type = ["cdylib"]`, and no code projects the Rust trait through the ORT
plugin-EP C ABI.

The **inbound** direction (nxrt loads foreign ORT plugin EPs) is fully
implemented:

- `abi/runtime.rs` resolves `CreateEpFactories` from a foreign dylib, creates
  an `OrtEpFactory` → `OrtEp`, calls `GetCapability` / `Compile`, and wraps
  the `OrtNodeComputeInfo` callbacks as native `Kernel`s.
- `abi/host.rs` projects our IR through the ORT C graph API (`HostGraph`,
  `HostNode`, `HostSupportInfo`) so the foreign plugin sees a standard
  `OrtGraph`.
- `abi/mod.rs` provides convex partition fusion (`UnionFind`, `SubgraphClaim`,
  `OrtGraphView::query_capabilities`).

The **outbound** direction (upstream ORT loads our EPs as plugins) is the gap.

## 1. Adapter Shape

### New crates

| Crate | `crate-type` | Purpose |
|-------|-------------|---------|
| `onnx-runtime-ep-plugin` | `["lib"]` | Shared adapter: owns 100% of the `unsafe` FFI that projects any `ExecutionProvider` out through the ORT plugin-EP C ABI. Pure Rust library; no `cdylib`. |
| `onnx-runtime-ep-cpu-plugin` | `["cdylib", "lib"]` | Thin shim: instantiates `CpuExecutionProvider`, calls the adapter, and exports `CreateEpFactories` / `ReleaseEpFactory`. |
| `onnx-runtime-ep-cuda-plugin` | `["cdylib", "lib"]` | Same pattern for `CudaExecutionProvider`. |

### Dependency graph

```
onnx-runtime-ep-cpu-plugin (cdylib)
  └── onnx-runtime-ep-plugin (adapter, lib)
        └── onnx-runtime-ep-api (trait + abi types)
        └── onnx-genai-ort-sys (vendored ORT C types)
  └── onnx-runtime-ep-cpu (the real EP)

onnx-runtime-ep-cuda-plugin (cdylib)
  └── onnx-runtime-ep-plugin (adapter, lib)
  └── onnx-runtime-ep-cuda (the real EP)
```

### Why one shared adapter crate

The ORT plugin-EP C ABI has ~15 callback slots and requires careful lifetime
management of factory/EP/compute-info objects. Writing this once in
`onnx-runtime-ep-plugin` means:

1. Each EP's shim crate is **mechanical**: instantiate the EP, call
   `ep_plugin::export(ep)`, done.
2. All `unsafe` FFI lives in one reviewed crate.
3. Bug fixes to ABI projection land once, not per-EP.

### The shim pattern

Each `cdylib` shim crate contains exactly one file:

```rust
// crates/onnx-runtime-ep-cpu-plugin/src/lib.rs
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_plugin::export_ep_factories;

// Generates #[unsafe(no_mangle)] CreateEpFactories and ReleaseEpFactory.
export_ep_factories!(|| CpuExecutionProvider::new());
```

The `export_ep_factories!` macro (defined in `onnx-runtime-ep-plugin`) expands
to the two required C entry points and all interior callback wiring.

## 2. ABI Surface — Outbound Entry Points

The exported `cdylib` must provide these symbols and fill these callback slots.
Types reference `onnx_genai_ort_sys` (the same vendored bindings the inbound
path already uses).

### Top-level exported symbols

| Symbol | Signature | Required |
|--------|-----------|----------|
| `CreateEpFactories` | `(registration_name: *const c_char, api_base: *const OrtApiBase, logger: *const OrtLogger, out_factories: *mut *mut OrtEpFactory, max: usize, out_count: *mut usize) -> *mut OrtStatus` | **v1** |
| `ReleaseEpFactory` | `(factory: *mut OrtEpFactory) -> *mut OrtStatus` | **v1** |

### `OrtEpFactory` callback slots

| Slot | Required | Notes |
|------|----------|-------|
| `ort_version_supported` | **v1** | Set to `ORT_API_VERSION` from vendored `ort-sys`. |
| `GetName` | **v1** | Returns the EP's `name()`. |
| `GetSupportedDevices` | **v1** | Enumerates devices; CPU EP returns one CPU device. |
| `CreateEp` | **v1** | Allocates a boxed `ExecutionProvider`, calls `initialize`, returns an opaque `OrtEp*`. |
| `ReleaseEp` | **v1** | Drops the boxed EP. |

### `OrtEp` callback slots

| Slot | v1? | Adapter strategy |
|------|-----|------------------|
| `GetCapability` | **v1** | Call `OrtGraphView::query_capabilities(ep)` on the **inbound** `OrtGraph*` ORT passes us. This is the **mirror** of the inbound path — we receive an `OrtGraph*` from ORT (not from our IR) and must walk it through the ORT C graph API to ask our EP `supports_op` per node. The adapter reads node attributes, input shapes/dtypes from ORT's graph API callbacks and calls `ep.supports_op()`. |
| `Compile` | **v1** | For each claimed subgraph, call `ep.get_kernel()` per node (or in the future, a fused compile). Return `OrtNodeComputeInfo` with `CreateState`/`Compute`/`ReleaseState` callbacks that dispatch to the Rust `Kernel::execute`. |
| `ReleaseNodeComputeInfos` | **v1** | Drop the adapter-owned kernel state. |

### `OrtNodeComputeInfo` callbacks

| Callback | v1? | Notes |
|----------|-----|-------|
| `CreateState` | **v1** | Allocate per-thread kernel state (or no-op if the kernel is stateless). |
| `Compute` | **v1** | Unpack `OrtKernelContext` inputs → `TensorView`, call `Kernel::execute`, pack outputs. |
| `ReleaseState` | **v1** | Drop kernel state. |

### Deferred (not v1)

| Capability | Reason |
|------------|--------|
| `OrtEp` memory allocator callbacks (`AllocateFunc`, `FreeFunc`) | CPU EP uses host malloc; CUDA needs stream-ordered alloc which is a separate design. |
| Data transfer / `MemCpy` callbacks | Only needed for device EPs where ORT orchestrates H2D/D2H. |
| `OrtEp.SaveContext` / `LoadContext` | EPContext export through ORT's session-option path; requires the §55 plumbing on the ORT side. |
| Custom op registration | Orthogonal to EP export. |

## 3. Symmetry with the Inbound Path

### What is reused

| Component | Location | Reuse |
|-----------|----------|-------|
| `UnionFind` + convex partition fusion | `abi/mod.rs` | **Shared.** `OrtGraphView::query_capabilities` already takes any `&dyn ExecutionProvider`. The outbound `GetCapability` implementation calls this same function after wrapping ORT's inbound `OrtGraph*` in our `GraphView`. |
| `SubgraphClaim` | `abi/mod.rs` | **Shared.** Output of capability discovery, input to `Compile`. Same struct both directions. |
| ORT C type definitions | `onnx_genai_ort_sys` | **Shared.** Both directions use the same vendored bindings. |

### What is new (mirror-image)

| Outbound need | Inbound counterpart | What to write |
|---------------|---------------------|---------------|
| Read ORT's `OrtGraph*` to extract nodes/shapes/dtypes for `supports_op` | `HostGraph` projects **our** IR as an `OrtGraph*` for foreign plugins | **`OutboundGraphReader`**: calls ORT's `OrtGraph` API callbacks to read node/value info from ORT's graph into the shapes/dtypes/layouts that `supports_op` expects. This is the inverse of `HostGraph`. |
| Pack Rust `Kernel::execute` results into `OrtKernelContext` outputs | `PluginCompiledKernel` unpacks `OrtKernelContext` inputs into `TensorView` | **`OutboundKernelContext`**: reads `OrtKernelContext` inputs into `TensorView`, writes `Kernel::execute` outputs back through `OrtKernelContext`. Mirror of the inbound `HostKernelContext`. |
| Fill `OrtEpFactory` / `OrtEp` / `OrtNodeComputeInfo` callback tables | `PluginRuntime` reads these tables from a foreign dylib | **`ExportedFactory`** / **`ExportedEp`**: heap-allocated structs whose raw pointers are returned as opaque `OrtEpFactory*` / `OrtEp*`. Dropped by `ReleaseEpFactory` / `ReleaseEp`. |

### Where it lives

All new outbound code lives in `crates/onnx-runtime-ep-plugin/src/`:

```
crates/onnx-runtime-ep-plugin/
  src/
    lib.rs          -- export_ep_factories! macro, public API
    factory.rs      -- ExportedFactory, CreateEpFactories / ReleaseEpFactory
    ep.rs           -- ExportedEp, GetCapability / Compile dispatch
    graph_reader.rs -- OutboundGraphReader (reads ORT's OrtGraph*)
    kernel_ctx.rs   -- OutboundKernelContext (TensorView ↔ OrtKernelContext)
    compute.rs      -- OrtNodeComputeInfo callback impls
```

## 4. Dual-ABI Coexistence

An EP simultaneously serves three consumers:

| Consumer | Mechanism | Crate involved |
|----------|-----------|---------------|
| **nxrt native (in-process)** | Direct Rust trait call via `EpRegistry::register(Box::new(CpuExecutionProvider::new()))` | `onnx-runtime-ep-cpu` (lib) |
| **Upstream ORT (plugin dylib)** | `dlopen("libnxrt_ep_cpu.so")` → `CreateEpFactories` | `onnx-runtime-ep-cpu-plugin` (cdylib) |
| **Future nxrt dynamic ABI** | `dlopen` + nxrt-native C ABI entry points (not yet designed) | Same cdylib, additional symbols |

### Feature-flag / crate-type strategy

- The EP crate (`onnx-runtime-ep-cpu`) stays `crate-type = ["lib"]`. A normal
  `cargo build` or `cargo test` never produces a dylib and never requires ORT
  headers or an ORT C library.
- The plugin shim crate (`onnx-runtime-ep-cpu-plugin`) adds
  `crate-type = ["cdylib", "lib"]`. Building it produces `libnxrt_ep_cpu.so` /
  `nxrt_ep_cpu.dll` / `libnxrt_ep_cpu.dylib`. The `lib` target lets the shim's
  own unit tests run without `cdylib` link constraints.
- The shim crate is **not** a workspace default member. `cargo build` at the
  workspace root does not build it. An explicit
  `cargo build -p onnx-runtime-ep-cpu-plugin` produces the dylib.
- The adapter crate (`onnx-runtime-ep-plugin`) depends on `onnx-genai-ort-sys`
  for the C type definitions, but does **not** require a linked ORT library at
  build time — it only uses the type definitions and `ORT_API_VERSION` constant,
  which are generated from vendored headers by `ort-sys`'s `build.rs`.

### Evolution toward the nxrt dynamic ABI

The future nxrt ABI will export richer entry points (e.g., structured kernel
metadata, weight negotiation, capture-region policy). The same `cdylib` shim
crate can export both sets of symbols:

```rust
// ORT plugin ABI (backward compat)
export_ep_factories!(|| CpuExecutionProvider::new());

// nxrt native ABI (future)
// export_nxrt_ep!(|| CpuExecutionProvider::new());
```

The ORT ABI evolves toward the nxrt one by adding optional callback slots
to `OrtEpFactory` / `OrtEp` that the adapter populates when the host ORT
version supports them.

## 5. Versioning & Fail-Closed

### ABI version negotiation

1. `CreateEpFactories` receives `*const OrtApiBase`. The adapter calls
   `OrtApiBase::GetApi(ORT_API_VERSION)` to obtain the host's `OrtApi`.
2. If the host's API version is **older** than `ort_version_supported`, the
   adapter returns an `OrtStatus` with code `ORT_FAIL` and message:
   ```
   nxrt EP plugin requires ORT API version {ort_version_supported},
   but the host provides version {host_version}; upgrade the host
   ONNX Runtime or use a plugin built for this ORT version
   ```
3. If `GetApi` returns null, same fail-closed error.
4. `OrtEpFactory::ort_version_supported` is set to the `ORT_API_VERSION`
   constant from our vendored `ort-sys` headers.

### Unsupported-path reporting

- **`GetCapability` claims zero nodes:** The adapter returns an empty
  `OrtEpGraphSupportInfo`. ORT will fall back to its own CPU EP. No silent
  misbehavior.
- **`Compile` receives a subgraph our EP cannot kernel:** The adapter returns
  an `OrtStatus` with code `ORT_NOT_IMPLEMENTED` and a message naming the
  unsupported op/dtype/shape. ORT surfaces this to the user.
- **Kernel `Compute` fails:** Returns `OrtStatus` with code `ORT_FAIL` and
  the `EpError` message. ORT aborts the `Run()`.
- **Memory allocation unsupported (v1 CPU):** The `OrtEp` does not fill
  allocator callbacks. ORT uses its own allocator. If ORT requires device
  alloc callbacks and they are null, ORT will report the missing callback.

### What the adapter must NOT do

- Silently return success when a kernel is missing (fail closed).
- Silently drop nodes from a compile request.
- Return a compute-info with null `Compute` callback.

## 6. What Executes Now (as of 2026-08-10, re-verified by Roy)

### Current compilation state

**Both crates compile cleanly.**

```
cargo check -p onnx-runtime-ep-plugin   # → Finished (1 unused-fn warning)
cargo check -p onnx-runtime-ep-cpu-plugin  # → Finished (no new errors)
```

Previously (before this branch), `ep.rs:34` was missing 9 optional `OrtEp`
fields added in ORT 1.25–1.27 (`CreateProfiler`, `IsGraphCaptureEnabled`,
`IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy`,
`GetAvailableResource`, `OnSessionInitializationEnd`, `GetDefaultMemoryDevice`,
`ReleaseCapturedGraph`). All are now set to `None`. The compile failure is
resolved.

### What executes end-to-end against real ORT 1.27.0

| Component | File | State |
|-----------|------|-------|
| `export_ep_factories!` macro | `lib.rs` | **Exercised** — `panicking_constructor_caught_and_zero_factories_returned` |
| `ExportedFactory` + `CreateEpFactories` / `ReleaseEpFactory` | `factory.rs` | **Exercised** — `ort_register_ep_library` |
| `ExportedEp` + `GetCapability` / `Compile` / `ReleaseNodeComputeInfos` | `ep.rs` | **Exercised** — `ort_loads_our_ep_and_runs_model` |
| `OrtNodeComputeInfo` (`CreateState`, `Compute`, `ReleaseState`) | `compute.rs` | **Exercised** — all conformance tests |
| `OutboundGraphReader` (ORT `OrtGraph*` → nxrt node/shape/dtype) | `graph_reader.rs` | **Exercised** — shape-inference tests |
| `OutboundKernelContext` (`OrtKernelContext` ↔ `TensorView`/`TensorMut`) | `kernel_ctx.rs` | **Exercised** — kernel read/write roundtrip tests |
| CPU EP shim (`export_ep_factories!(|| CpuExecutionProvider::new())`) | `onnx-runtime-ep-cpu-plugin/src/lib.rs` | **Exercised** — full e2e suite |

### Known gaps and open issues

| Gap | Impact | Owner | Status |
|-----|--------|-------|--------|
| `conformance_two_sessions` — previously `#[ignore]`d | ✅ **Fixed** — was test-assertion bug (Pris); UAF root-cause fixed by Deckard (`c92838d`); `stress_register_run_unregister_cycles` (25 cycles) passes | Closed |
| Ops requiring non-trivial shape inference (Reshape, Gather, Concat) | `SameAsInput(0)` fallback may produce wrong shape | Nabil/Deckard | Incremental |
| Multi-output ops (Split, TopK) | Only first output shaped correctly | Nabil/Deckard | Deferred |
| Device (GPU) tensors | Requires allocator callback design | CUDA wave | Blocked — no GPU on this host |
| Holden security re-audit | ✅ **🟡 YELLOW — May ship** (2026-08-10T22:42Z) | All resolved; 2 LOW post-merge advisories | Closed |

## 7. Milestone Plan

### M0: Adapter crate skeleton — ✅ DONE

Created `crates/onnx-runtime-ep-plugin/` with the `export_ep_factories!` macro,
`ExportedFactory` / `ExportedEp` / callback scaffolding, and all ABI fields.
Unit test: macro expands, `CreateEpFactories` returns a valid factory pointer
with correct `ort_version_supported`. 82 unit tests pass.

### M1: CPU EP end-to-end through ORT — ✅ DONE

`crates/onnx-runtime-ep-cpu-plugin/` (cdylib shim) builds and runs:
- `OutboundGraphReader`: reads ORT's `OrtGraph*` into shapes/dtypes.
- `OutboundKernelContext`: `OrtKernelContext` ↔ `TensorView` / `TensorMut`.
- `GetCapability` → `query_capabilities` → shape inference pipeline (22 rules).
- `Compile` → per-node `get_kernel` → `OrtNodeComputeInfo`.
- Integration: ORT loads `libonnx_runtime_ep_cpu_plugin.so`, registers it,
  finds `cpu_ep` via `GetEpDevices`, runs Add/MatMul/broadcast/int32 models,
  outputs verified correct.

### M2: Remaining EPs

| EP | Status | Blocker |
|----|--------|---------|
| `onnx-runtime-ep-cpu-plugin` | **READY (security-pending)** | Holden re-audit: 3 open findings |
| `onnx-runtime-ep-cuda` | **BLOCKED** | No CUDA toolkit/GPU on this host |

### CUDA EP design (no hardware needed)

Work that can proceed without a GPU:
- Allocator callback design (`OrtEpFactory::CreateAllocator`/`ReleaseAllocator`)
- Stream sharing design (`IsStreamAware` + `CreateSyncStreamForDevice`)
- Data transfer registration (`CreateDataTransfer`)
- CUDA graph capture hooks (`IsGraphCaptureEnabled` + `ReplayGraph`)

Work requiring CUDA hardware:
- Runtime validation of the above
- `prefetch_lazy_weight` implementation and testing

## 8. ABI Compatibility Boundary

**ORT version:** 1.27.0 (`ORT_API_VERSION = 27`)

| Item | Value |
|------|-------|
| Required exported symbols | `CreateEpFactories`, `ReleaseEpFactory` |
| Optional exported symbols | none required beyond the two above |
| `OrtEp::ort_version_supported` | Set to 27 |
| `OrtEpFactory::ort_version_supported` | Set to 27 |
| `OrtEp` fields implemented | `GetName`, `GetCapability`, `Compile`, `ReleaseNodeComputeInfos` |
| `OrtEp` fields set to `None` | `GetPreferredDataLayout`, `ShouldConvertDataLayoutForOp`, `SetDynamicOptions`, `OnRunStart`, `OnRunEnd`, `CreateAllocator`, `CreateSyncStreamForDevice`, `GetCompiledModelCompatibilityInfo`, `GetKernelRegistry`, `IsConcurrentRunSupported`, `Sync`, `CreateProfiler`, `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy`, `GetAvailableResource`, `OnSessionInitializationEnd`, `GetDefaultMemoryDevice`, `ReleaseCapturedGraph` |
| ORT host call sequence | `CreateEnv` → `RegisterExecutionProviderLibrary` → `GetEpDevices` → `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run` |
| Minimum ORT version for plugin EP | 1.22 (when `RegisterExecutionProviderLibrary` was added) |

## 9. Hard-Won ABI Contracts — Guidance for Future EP Authors

These two contracts caused real bugs that required multiple agent-days to diagnose.
Every future EP author should read this before touching device descriptors or graph objects.

### Contract A: `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice`

`EpDevice_AddAllocatorInfo(_In_ OrtEpDevice*, _In_ const OrtMemoryInfo*)` **stores
the raw pointer** inside the OrtEpDevice. ORT does NOT copy it. The `EpDevice_MemoryInfo`
API returns this stored pointer directly.

**Bug this caused:** The old code called `ReleaseMemoryInfo` immediately after
`AddAllocatorInfo`. On the first ORT call the freed memory still held valid data (luck).
After ≥6 register/unregister cycles the freed memory was reused, producing garbage
`DeviceType:-112` values — the symptom that identified the bug.

**Rule:** Do NOT release the `OrtMemoryInfo` after a successful `EpDevice_AddAllocatorInfo`.
ORT releases it when it calls `ReleaseEpDevice`. Only release on failure (when the pointer
was not consumed).

**Additionally:** Use `CreateMemoryInfo_V2` with explicit `OrtMemoryInfoDeviceType_CPU` /
`OrtDeviceMemoryType_DEFAULT` / `OrtDeviceAllocator` — the legacy `CreateCpuMemoryInfo`
does not populate the fields the EP device system reads (produces `DeviceType:64,
MemoryType:28` garbage).

### Contract B: `OrtGraph*` / `OrtNode*` must not be cached beyond `Compile`

ORT passes `OrtGraph*` and `OrtNode*` pointers to `GetCapability` and `Compile`.
These pointers are valid only for the duration of the callback. ORT may free or reuse
them after the callback returns.

**Rule:** Copy everything you need from the graph into owned Rust data during `Compile`.
Do not store `OrtGraph*` or `OrtNode*` in your `OrtEp` or `OrtNodeComputeInfo` structs.
The `OutboundGraphReader` design enforces this: it reads graph data into owned `ExportedComputeInfo`
during `Compile` and the `OrtGraph*` reference is never retained.


