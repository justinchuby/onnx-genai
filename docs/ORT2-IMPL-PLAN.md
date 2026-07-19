# ORT 2.0 — Phase 1 Implementation Plan

> Dependency-ordered execution plan for the **Phase 1 Foundation** of the ORT
> 2.0 runtime (`docs/ORT2.md` §54). This maps the design to buildable, fan-out
> tasks. The scaffold and the `onnx-runtime-ir` public contract already exist on
> branch `squad/ort2-foundation`; everything below builds against that stable
> IR surface.

## Status legend

| Mark | Meaning |
|------|---------|
| ✅ | Landed in the foundation branch (compiles + tested) |
| 🔨 | Skeleton exists (signatures + module layout), body is `todo!()` |
| ⬜ | Not started |

---

## 0. What already landed (this branch)

| Crate | State | Notes |
|-------|-------|-------|
| `onnx-runtime-ir` | ✅ real API | Data model + graph ops fully implemented & unit-tested (34 tests). The **stable contract** all other crates consume. |
| `onnx-runtime-loader` | 🔨 skeleton | Module stubs: `proto`, `graph_builder`, `weights`, `shape_inference`. |
| `onnx-runtime-ep-api` | 🔨 skeleton | Real trait/type signatures; `OpRegistry`/`EpRegistry` logic implemented. |
| `onnx-runtime-ep-cpu` | 🔨 skeleton | `CpuExecutionProvider` stub implements the EP trait; lifecycle tested. |
| `onnx-runtime-session` | 🔨 skeleton | `SessionBuilder`/`InferenceSession` surface; executor is `todo!()`. |
| `onnx-runtime-capi` | 🔨 skeleton | `OrtErrorCode` + version export; Tier-1 entry points TODO. |

All six are wired into the root workspace and build **standalone** (no
`onnx-genai-ort` / ORT C-lib dependency):

```bash
cargo build -p onnx-runtime-ir -p onnx-runtime-loader -p onnx-runtime-ep-api \
            -p onnx-runtime-ep-cpu -p onnx-runtime-session -p onnx-runtime-capi
cargo test  -p onnx-runtime-ir
cargo clippy -p onnx-runtime-ir --lib --tests -- -D warnings
```

---

## 1. Build order & parallelism

```
                ┌──────────────────────┐
                │  onnx-runtime-ir  ✅  │   (contract — DONE)
                └───────────┬──────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                   ▼
 ┌─────────────────┐ ┌────────────────┐   (ep-api re-exports
 │ runtime-loader  │ │ runtime-ep-api │    IR device types)
 └────────┬────────┘ └───────┬────────┘
          │                  │
          │                  ▼
          │          ┌────────────────┐
          │          │ runtime-ep-cpu │
          │          └───────┬────────┘
          ▼                  ▼
        ┌────────────────────────────┐
        │     runtime-session        │  (needs loader + ep-api + ep-cpu)
        └──────────────┬─────────────┘
                       ▼
              ┌────────────────┐
              │  runtime-capi  │
              └────────────────┘
```

**Once IR is landed (it is), these can be built fully in parallel by
independent implementers:**

- **Track A — `onnx-runtime-loader`** (depends only on IR)
- **Track B — `onnx-runtime-ep-api`** (depends only on IR)

Then, serially after their prerequisites:

- **Track C — `onnx-runtime-ep-cpu`** — after B (needs the EP/Kernel traits).
- **Track D — `onnx-runtime-session`** — after A, B, C.
- **Track E — `onnx-runtime-capi`** — after D.

Recommended fan-out: **dispatch A and B now, in parallel.** C starts as soon as
B's trait signatures are frozen (already true on this branch, so C can begin
immediately too, coding against the skeleton). D integrates. E finishes.

---

## 2. Per-crate task breakdown

### A. `onnx-runtime-loader`  (design §19, §12) — **~L**

Turns an ONNX file into an `onnx_runtime_ir::Graph`.

| Module | Deliverable | Design |
|--------|-------------|--------|
| `proto` | `build.rs` running `prost-build` on `onnx.proto3`; `decode_model` | §19.1 |
| `graph_builder` | `GraphProto` → IR: create values/nodes, wire edges, intern symbolic dims by name, populate `opset_imports`, uphold §3.5 invariants | §19.1, §3.5 |
| `weights` | Inline + external-data resolution; `mmap` external files (add `memmap2`); build `WeightRef`s | §19.2, §12 |
| `shape_inference` | Topo-order driver + per-op rule table (MatMul, Conv, elementwise broadcast, Reshape, Gather, LayerNorm, …) | §19.3 |

New deps to add: `prost`, `prost-build` (build-dep), `memmap2`.
**Risk:** the per-op shape-inference table is the long pole — scope it to the
BERT op set first, mark the rest `todo!()`.
**Estimate:** Large (protobuf schema + builder + shape rules).

### B. `onnx-runtime-ep-api`  (design §4) — **~M**

The EP contract. Signatures already frozen on this branch.

| Module | Remaining work | Design |
|--------|----------------|--------|
| `provider` | Finalize `DeviceBuffer` ownership/`Send+Sync` story (currently a documented placeholder `unsafe impl`) | §4.1 |
| `kernel` | Confirm `Cost` shape vs. `onnx-runtime-cost-model` (Phase 2) | §4.2, §6 |
| `registry` | `OpRegistry`/`EpRegistry` ✅ done; `load_legacy` is Phase 2 | §4.3, §4.6 |
| `tensor` | `TensorView`/`TensorMut` ✅ shape; validate against DLPack import path (§5.3) | §5.4 |
| `abi` | `OrtGraphView` C projection — **Phase 2**, `unsafe` FFI | §3.4, §4.5 |

**Risk:** raw-pointer `DeviceBuffer`/`TensorView` safety invariants; needs a
review pass when the first real EP wires memory.
**Estimate:** Medium (mostly done; hardening + DLPack alignment).

### C. `onnx-runtime-ep-cpu`  (design §4.4) — **~L**

CPU kernels through the built-in Rust GEMM implementation (`native-eps/cpu`).

| Piece | Deliverable |
|-------|-------------|
| Native build | No C++ build dependency for the CPU GEMM path |
| `kernels::*` | One `Kernel` + `KernelFactory` per Phase-1 op: `MatMul, Add, Relu, Reshape, Transpose, Gather, LayerNormalization` (see `kernels::PHASE1_OPS`) |
| `provider` | Fill `supports_op`, `get_kernel`, `allocate`/`copy` (host, aligned) |

**Risk:** strided-input handling at the kernel boundary. Start with
`MatMul` (GEMM) end-to-end as the vertical slice.
**Estimate:** Large (C++ interop + per-op ports).

### D. `onnx-runtime-session`  (design §20, §11.3) — **~M**

Ties loader + EPs together with a sequential executor.

| Piece | Deliverable |
|-------|-------------|
| `build()` | load → `auto_detect_device` (CPU only in P1) → optimize (no-op P1) → compile kernels → allocate |
| `run()` | Sequential topo-order executor: materialize inputs, run kernels, collect outputs |
| `Tensor` | Promote the placeholder to the real device-aware tensor (or move to a shared `runtime-tensor` module) — **open question below** |
| `warmup()` | Populate the shape-keyed kernel cache (§11) |

**Estimate:** Medium (executor is straightforward once kernels exist).

### E. `onnx-runtime-capi`  (design §21, §22) — **~M**

Tier-1 C ABI: `OrtGetApiBase` + `CreateSession` + `Run`.

| Piece | Deliverable |
|-------|-------------|
| lib crate-type | Switch to `crate-type = ["cdylib", "staticlib", "lib"]` once symbols exist |
| Entry points | `extern "C"` marshalling for create/run; map `SessionError` → `OrtErrorCode` (`From<&Error>` §22) |
| Status/error | `OrtStatus` object + string message |

**Risk:** binary compatibility with the real ORT header layout — verify against
`onnxruntime_c_api.h`. **Estimate:** Medium.

### Milestone (Phase 1 exit)

`ort2-milestone-bert`: **run BERT on CPU, output matches upstream ORT.** Requires
A + B + C + D (+ E for the C-API path). Add a conformance test comparing against
`onnx-genai-ort` output on a small BERT.

---

## 3. `onnx-runtime-ir` public API contract (landed)

The frozen surface downstream builds against:

- **Types:** `DataType` (ONNX-numbered, byte/bit sizing, sub-byte packing),
  `DeviceType`/`DeviceId` (first-class placement).
- **Shape:** `Shape = Vec<Dim>`, `Dim::{Static,Symbolic}`, `SymbolId`,
  `SymbolConstraints` (min/max/divisible), static-shape helpers.
- **Layout (§5):** `TensorLayout` (optional strides, `MemoryFormat`, alignment),
  `compute_contiguous_strides`, `is_contiguous`, `broadcast_shapes`, lazy
  `transpose`, `storage_size`.
- **Graph model:** `Graph` (arena nodes/values, inputs/outputs, initializers,
  symbol constraints, opset imports, subgraphs), `Node`, `Attribute`, `Value`,
  `TensorData`/`SparseTensorData`/`TypeProto`/`WeightRef`, `Arena`/`ArenaKey`.
- **Ops (real, tested):** `topological_order` (Kahn, deterministic),
  `predecessors`/`successors`/`nodes_between`, mutation
  (`insert_node`/`remove_node`/`replace_node`/`insert_on_edge`/
  `replace_all_uses`/`create_value`) with edge-consistency, `validate`.
- **Errors (§22):** `IrError` + `GraphError`.

**Deferred inside IR (signature-only / by design):** full per-op shape inference
(lives in the loader), and the ORT C ABI graph projection `OrtGraphView` (moved
to `ep-api::abi` because it needs `unsafe` FFI + EP types — see open questions).

---

## 4. Risky / deep parts to watch

1. **Shape inference rule table** (loader) — breadth of ONNX ops; scope to BERT
   first.
2. **CPU GEMM** (ep-cpu) — SIMD dispatch and strided-input contract.
3. **C ABI compatibility** (capi) — must match ORT's header/struct layout for a
   true drop-in `libonnxruntime.so`.
4. **Raw-pointer tensor/buffer safety** (ep-api) — `DeviceBuffer`/`TensorView`
   `Send`/`Sync` invariants need a dedicated review once real memory is wired.
5. **DLPack import/export** (§5.3) — cross-framework zero-copy; align the
   session `Tensor` with `TensorView` strides early.

---

## 5. Open questions for the user

1. **Where does the owned `Tensor` live?** §5.3/§5.4 reference an owned,
   device-aware `Tensor` (DLPack, strided) used by `session::run` and by
   `Kernel` via views. It is not in the Phase-1 crate list. Options: (a) a small
   new `onnx-runtime-tensor` crate depended on by ep-api + session; (b) put it in
   `ep-api`; (c) keep the placeholder in `session`. **Assumed (c) for now**
   (placeholder in `session`); recommend (a) before ep-cpu wires real memory.
2. **`OrtGraphView` placement.** The design lists it under §3.4 (IR), but it
   needs `unsafe` FFI and EP types, conflicting with "IR is safe Rust." **Placed
   the skeleton in `ep-api::abi`.** Confirm this is acceptable.
3. **Protobuf source of truth.** Vendor `onnx.proto3` into the loader crate and
   generate with `prost-build`, or depend on an existing crate? Assumed: vendor +
   `prost-build`.
4. **`DataType::byte_size` for sub-byte types returns 0.** IR exposes
   `storage_bytes(count)` (2-per-byte packing) as the safe path. Confirm kernels
   should use `storage_bytes`/`bit_size` rather than `byte_size` for int4/uint4.
5. **`capi` crate-type.** Left as `lib` in the skeleton so it builds without
   forcing a cdylib link now; switch to `cdylib`+`staticlib` when entry points
   land. OK?
6. **Optimizer coupling.** `ExecutionProvider::custom_passes` returns a local
   placeholder `OptimizerPass` trait to avoid a Phase-2 `onnx-runtime-optimizer`
   dependency. It should move to that crate in Phase 2.
