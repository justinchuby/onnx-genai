# EP Plugin Export Architecture

> **Status (2026-08-10, Roy re-verified):** Adapter crates `onnx-runtime-ep-plugin` and
> `onnx-runtime-ep-cpu-plugin` exist and contain substantive implementation, but
> **do not compile** (`cargo check -p onnx-runtime-ep-cpu-plugin` fails with
> `error[E0063]: missing fields CreateProfiler, GetAvailableResource,
> GetDefaultMemoryDevice and 8 other fields in initializer of OrtEp`).
> The `OrtEp` struct in ORT 1.27.0 has 24 fields; the adapter initializer is missing
> 11 optional-but-required `Option<fn>` fields added in ORT 1.23–1.27.
> The full `CreateEpFactories → GetCapability → Compile → Compute` path is **not yet
> exercisable** end-to-end. The `Compute` callback for Add/Sub/Mul/Div is written but
> has not been run against real ORT due to the compile failure.
> **Immediate blocker:** Nabil must fill the 11 missing `OrtEp` fields with `None`.
> See `crates/onnx-runtime-ep-plugin/src/ep.rs:34`.
>
> **Author:** Roy (Lead) — 2026-08-10; Implementation: Nabil — 2026-08-10
> **Standing directive:** Extension contract §524 — every extension seam exposes
> a stable dynamic C ABI **and** a first-class Rust trait, shipped in parallel.

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

**`onnx-runtime-ep-cpu-plugin` does NOT compile.**

```
cargo check -p onnx-runtime-ep-cpu-plugin 2>&1 | grep "^error"
error[E0063]: missing fields `CreateProfiler`, `GetAvailableResource`,
  `GetDefaultMemoryDevice` and 8 other fields in initializer of `OrtEp`
  --> crates/onnx-runtime-ep-plugin/src/ep.rs:34:21
```

The `OrtEp` struct in ORT 1.27.0 bindings has **24 fields** (see `EP_PLUGIN_EXPORT_ABI_TRUTH.md` §3).
The adapter at `ep.rs:34` initializes the struct by name and is missing 11 fields added in ORT 1.23–1.27.
All 11 are `Option<fn>` and may be set to `None` for the v1 CPU EP.

**Immediate fix (Nabil):** Add the 11 missing fields as `None` to the `OrtEp` struct initializer in `ep.rs:34`.

### What is written but not yet run

The following code exists and is structured correctly but has not been exercised end-to-end
due to the compile failure:

| Component | File | State |
|-----------|------|-------|
| `export_ep_factories!` macro | `lib.rs` | Written, not compiled |
| `ExportedFactory` + `CreateEpFactories` / `ReleaseEpFactory` | `factory.rs` | Written, not compiled |
| `ExportedEp` + `GetCapability` / `Compile` / `ReleaseNodeComputeInfos` | `ep.rs` | Written, **compile error here** |
| `OrtNodeComputeInfo` callbacks (`CreateState`, `Compute`, `ReleaseState`) | `compute.rs` | Written, not compiled |
| `OutboundGraphReader` (ORT `OrtGraph*` → nxrt node/shape/dtype) | `graph_reader.rs` | Written, not compiled |
| `OutboundKernelContext` (`OrtKernelContext` ↔ `TensorView`/`TensorMut`) | `kernel_ctx.rs` | Written, not compiled |
| CPU EP shim (`export_ep_factories!(|| CpuExecutionProvider::new())`) | `onnx-runtime-ep-cpu-plugin/src/lib.rs` | Written, not compiled |

### What does NOT yet execute

| Gap | Impact | Resolution |
|-----|--------|-----------|
| **11 missing `OrtEp` fields (Nabil's fix)** | Entire plugin path unexercisable | Fill with `None` in `ep.rs:34` |
| Ops requiring non-trivial shape inference (Reshape, Gather, Concat, etc.) | `SameAsInput(0)` fallback may produce wrong shape | Extend `ShapeInference` enum per-op |
| Multi-output ops (Split, TopK) | Only first output shaped correctly | Per-output shape inference |
| Device (GPU) tensors | Data pointer null check fails closed | Requires allocator callback design (M2-cuda) |

## 7. Milestone Plan

### M0: Adapter crate skeleton (prerequisite)

- Create `crates/onnx-runtime-ep-plugin/` with the `export_ep_factories!` macro
  stub and the `ExportedFactory` / `ExportedEp` / callback scaffolding.
- Unit-test: the macro expands, `CreateEpFactories` returns a valid factory
  pointer with correct `ort_version_supported`.

### M1: CPU EP end-to-end through ORT

- Create `crates/onnx-runtime-ep-cpu-plugin/` (cdylib shim).
- Implement `OutboundGraphReader`: read ORT's `OrtGraph*` into
  shapes/dtypes for `supports_op`.
- Implement `OutboundKernelContext`: `OrtKernelContext` ↔ `TensorView` /
  `TensorMut`.
- Implement `GetCapability` → `query_capabilities` pipeline.
- Implement `Compile` → per-node `get_kernel` → `OrtNodeComputeInfo`.
- **Integration test:** Build `libnxrt_ep_cpu.so`, load it into upstream ORT
  via `SessionOptions::RegisterCustomOpsLibrary` or the plugin EP path, run a
  small ONNX model (e.g., a 3-op `Add → Relu → Mul` graph), verify outputs
  match.

### M2: Mechanical remaining EPs

Each additional EP follows the CPU pattern — a `cdylib` shim crate with one
`export_ep_factories!` invocation.

### Per-EP blockers

| EP | Blocker | Notes |
|----|---------|-------|
| `onnx-runtime-ep-cpu` | **Adapter compile error** — `ep.rs:34` missing 9 `OrtEp` optional fields. Mechanical fix. | Nabil owns. |
| `onnx-runtime-ep-cuda` | Same compile error + runtime hardware requirement (`libcuda.so` absent on this host). | Build-time: fixable. Runtime: needs CUDA hardware. |
| `onnx-runtime-ep-cuda` | **Streams:** Our CUDA EP owns its CUDA stream; ORT may pass its own stream via `OrtKernelContext`. The adapter must reconcile stream ownership (use ORT's stream, or sync between them). | Design decision deferred to M2. |
| `onnx-runtime-ep-cuda` | **CUDA context sharing:** nxrt `CudaRuntime` creates its own primary context. ORT also manages CUDA contexts. Must share or reconcile. | Design decision deferred to M2. |
| `onnx-runtime-ep-cuda` | **Data transfer:** ORT orchestrates H2D/D2H copies for device EPs. The adapter must fill `MemCpy` callbacks or declare the EP handles its own transfers. | |
| Future EPs (MLX, etc.) | `../onnxruntime-mlx` is a separate repo not checked out here — out of scope. MLX unified memory simplifies the allocator story when available. | |

### Dependency order

```
M0 (adapter skeleton — done structurally, blocked on compile fix)
 └── M0.1 (Nabil: add 9 missing OrtEp fields as None in ep.rs:34)
       └── M1 (CPU EP end-to-end: compile + run an ONNX model through the plugin ABI)
             └── M2-cpu (integration tests, edge cases, shape inference completeness)
             └── M2-cuda (allocator + stream + CUDA context sharing design)
                   └── M2-cuda-impl (requires CUDA hardware for validation)
```

## 8. Roadmap to Full Provider Compatibility

### Ordered plan: CPU EP to all intended providers

**Step 0 (Immediate, no hardware needed):** Fix the compile error.
- File: `crates/onnx-runtime-ep-plugin/src/ep.rs:34`
- Action: Add the 9 missing `OrtEp` fields as `None` (see `EP_PLUGIN_EXPORT_ABI_TRUTH.md` §6 for the complete field list).
- Verifiable: `cargo check -p onnx-runtime-ep-cpu-plugin` succeeds.
- Also address Holden's CRITICAL finding (C1): wrap all `extern "C"` callbacks in `std::panic::catch_unwind` before any other testing.

**Step 1 (CPU EP unit tests, no hardware needed):** Verify adapter callbacks in isolation.
- Unit-test `OutboundGraphReader`: build a fake `OrtGraph*` (using our inbound `HostGraph` machinery) and confirm it produces the right `GraphView` for a few ops.
- Unit-test `OutboundKernelContext`: confirm tensor extraction from a mock `OrtKernelContext` yields correct `TensorView` data pointers and shapes.
- These tests can run in `cargo test -p onnx-runtime-ep-plugin`.

**Step 2 (CPU EP end-to-end, no hardware needed):** Run a real ONNX model.
- Build `onnx-runtime-ep-cpu-plugin` as a `cdylib`.
- Load via `OrtApi::RegisterExecutionProviderLibrary` (available since ORT 1.22; ORT 1.28.0 wheel is present at `/workspace/dev/onnx-genai/.ort-probe/lib/.../libonnxruntime.so.1.28.0`, backward-compatible with API 27).
- Run a small model (Add → Relu → Mul). Assert outputs.
- This is the end-to-end milestone.

**Step 3 (CPU EP production coverage, no hardware needed):** Extend shape inference.
- Add per-op `ShapeInference` variants for Reshape, Gather, Concat, etc.
- Run the 166-op CPU EP coverage through the adapter.

**Step 4 (CUDA EP adapter — requires design + hardware):**
- Design: CUDA context/stream sharing between nxrt and ORT. Options: adopt ORT's context; create new handles on ORT's stream; or use a context-per-session model.
- Adapter capabilities needed beyond CPU path:
  - `OrtEpFactory::CreateAllocator` and `ReleaseAllocator` — delegate to CUDA EP's `allocate`/`deallocate`.
  - `OrtEpFactory::IsStreamAware` + `CreateSyncStreamForDevice` — expose CUDA stream as `OrtSyncStreamImpl`.
  - `OrtEpFactory::CreateDataTransfer` — H2D/D2H copy registration.
  - `OrtEp::CreateAllocator` (per-session) — session-scoped device allocator.
  - `OrtEp::Sync` — synchronize the device stream.
  - `OrtEp::IsGraphCaptureEnabled` + `IsGraphCaptured` + `ReplayGraph` — expose nxrt's CUDA graph capture capability.
- Hardware: CUDA EP runtime initialization requires `libcuda.so`. This host has no NVIDIA GPU (`nvidia-smi` absent, `/dev/nvidia*` absent). Testing must occur on a CUDA-capable host.
- Note: `prefetch_lazy_weight` stub must be implemented before declaring CUDA plugin production-ready.

**Work that can be done here (no CUDA hardware):**
- Compile error fix (Step 0)
- CPU EP full pipeline (Steps 1–3)
- CUDA adapter design/code review (Step 4 design portion)

**Work requiring CUDA hardware:**
- CUDA EP plugin validation (Step 4 validation)
- `prefetch_lazy_weight` implementation and testing
