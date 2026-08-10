# EP Plugin Export Architecture

> **Status:** v1 implemented (CPU EP) — Compute path live for elementwise ops (Add, Sub, Mul, Div) on f32 as of 2026-08-10.
> See `crates/onnx-runtime-ep-plugin/` and `crates/onnx-runtime-ep-cpu-plugin/`.
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

## 6. What Executes Now (as of 2026-08-10)

### End-to-end operational

The full `CreateEpFactories → GetCapability → Compile → Compute` path is
implemented and tested. The L2 test (`plugin_export_abi.rs`) drives a real f32
Add kernel through the Compute callback with value assertions.

### Ops that genuinely execute through the plugin ABI

| Op | Dtypes | Shape constraint | Status |
|----|--------|------------------|--------|
| `Add` | f32 | any rank, numpy broadcast | ✅ L2 tested |
| `Sub` | f32 | any rank, numpy broadcast | ✅ (same kernel path) |
| `Mul` | f32 | any rank, numpy broadcast | ✅ (same kernel path) |
| `Div` | f32 | any rank, numpy broadcast | ✅ (same kernel path) |
| All other ops supported by `CpuExecutionProvider::supports_op` | f32 | varies | ✅ compile + execute path wired |

### Shape inference strategy (runtime)

Output shapes are inferred at Compute time from actual input shapes using:
- `ElementwiseBroadcast`: for binary elementwise ops — `broadcast_shapes()`
  from `onnx-runtime-ir`.
- `SameAsInput(0)`: for unary / shape-preserving ops (Relu, Sigmoid, Cast, etc.).

### Version check (fail-closed)

`CreateEpFactories` calls `GetApi(ORT_API_VERSION)`. If it returns null (host
too old), the plugin returns an `OrtStatus` error via a v1-API fallback when
available, or writes 0 factories and returns null. The plugin never proceeds
with a partially-understood vtable.

### What does NOT yet execute

| Gap | Impact | Resolution |
|-----|--------|-----------|
| Ops requiring non-trivial shape inference (Reshape, Gather, Concat, etc.) | `SameAsInput(0)` fallback may produce wrong output shape — Compute fails closed with a shape-mismatch error | Extend `ShapeInference` enum per-op |
| Multi-output ops (Split, TopK) | Only first output shaped correctly | Extend per-output shape inference |
| Non-f32 dtypes | `supports_op` may claim support but `from_onnx` mapping covers all dtypes — execution works if the kernel does | No gap for kernels that handle their dtype internally |
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
| `onnx-runtime-ep-cpu` | **None** — pure host memory, no device dependency. This is the v1 candidate. | |
| `onnx-runtime-ep-cuda` | **Device memory:** ORT expects `AllocateFunc`/`FreeFunc` for GPU tensors; our CUDA EP uses `cuMemAlloc`/arena, not ORT's allocator API. The adapter must either (a) fill allocator callbacks that delegate to our EP's `allocate`/`deallocate`, or (b) declare CPU-only I/O and do internal H2D/D2H. Option (a) is correct but requires ORT to route device allocations through our callbacks. | Requires M1 adapter + allocator callback design. |
| `onnx-runtime-ep-cuda` | **Streams:** Our CUDA EP owns its CUDA stream; ORT may pass its own stream via `OrtKernelContext`. The adapter must reconcile stream ownership (use ORT's stream, or sync between them). | Design decision deferred to M2. |
| `onnx-runtime-ep-cuda` | **Data transfer:** ORT orchestrates H2D/D2H copies for device EPs. The adapter must fill `MemCpy` callbacks or declare the EP handles its own transfers. | |
| Future EPs (MLX, etc.) | Mechanical once the adapter exists. MLX unified memory simplifies the allocator story (host-accessible). | |

### Dependency order

```
M0 (adapter skeleton)
 └── M1 (CPU EP end-to-end)
       └── M2-cpu (integration tests, edge cases)
       └── M2-cuda (allocator + stream + transfer design)
             └── M2-cuda-impl
```
