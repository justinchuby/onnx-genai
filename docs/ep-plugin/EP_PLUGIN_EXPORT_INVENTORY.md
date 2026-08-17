# EP Plugin-Export Inventory

**Purpose:** Pre-export inventory of every in-repo execution provider intended for eventual use as an ORT outbound plugin EP. Analysis only — no implementation changes.

**Author:** Deckard (Systems Dev)  
**Date:** 2026-08-10  
**Requestor:** @justinchuby

---

## 1. Scope Search Results

Search covered: every `impl ExecutionProvider` in the workspace, plus `crates/onnx-runtime-eager`, `crates/mlas-sys`, and any reference to `../onnxruntime-mlx`.

```
grep -rn "impl ExecutionProvider" crates/
```

Results (non-test, non-mock):
| Location | Type |
|---|---|
| `crates/onnx-runtime-ep-cpu/src/provider.rs:118` | **Production EP** — CpuExecutionProvider |
| `crates/onnx-runtime-ep-cuda/src/provider.rs:513` | **Production EP** — CudaExecutionProvider |
| `crates/onnx-runtime-ep-api/src/abi/mod.rs:160` | Inbound adapter — LegacyOrtEp (wraps incoming plugin .so) |
| `crates/onnx-runtime-session/src/plugin_provider.rs:72` | Inbound bridge — PluginExecutionProvider (legacy ORT subgraph dispatch) |

**Non-EPs excluded from inventory:**
- `crates/onnx-runtime-eager/` — orchestrator/dispatcher that holds `Vec<Arc<dyn ExecutionProvider>>`; implements no EP itself.
- `crates/mlas-sys/` — vendored MLAS BLAS kernel library; a dependency of `onnx-runtime-ep-cpu` under its default `mlas` feature, not an EP.
- `../onnxruntime-mlx` — no such sibling directory exists in this workspace. No Metal/MLX EP reference found.

**Production EPs subject to this inventory: 2** — CPU and CUDA.

---

## 2. Inventory Table

### 2.1 `onnx-runtime-ep-cpu` — `CpuExecutionProvider`

| Field | Detail |
|---|---|
| **Crate / path** | `crates/onnx-runtime-ep-cpu/src/` |
| **`impl ExecutionProvider`?** | Yes — `src/provider.rs:118` |
| **Trait methods — required** | `name` ✅ (returns `"cpu_ep"`), `device_type` ✅ (Cpu), `device_id` ✅ (CPU:0), `initialize` ✅ (sets Rayon decode budget), `shutdown` ✅, `supports_op` ✅ (registry-keyed, opset-aware, actionable declines), `get_kernel` ✅ (registry lookup → factory.create), `allocate` ✅ (via `DeviceAllocator` seam, power-of-two alignment check), `deallocate` ✅ (cross-device guard, borrowed-buffer no-op), `copy` ✅ (bounds-checked `copy_nonoverlapping`), `copy_async` ✅ (synchronous; returns `Fence::signalled()`), `sync` ✅ (no-op, correct for CPU) |
| **Trait methods — optional overrides** | `custom_passes` ✅ (`cpu_optimization_passes()`); `with_memory` constructor allows swappable allocator backing |
| **Methods left to trait default (no-ops)** | `page_lazy_weight`, `prefetch_lazy_weight`, `wait_fence`, `record_compute_fence`, `copy_wait_fence`, `device_argmax_supported` (→`false`), `begin/end/abort_device_graph_capture` (→ err), `replay_device_graph*`, `reset_device_graph`, `check_device_capture_error`, `reserve_workspace`, `as_ort_plugin` (→ `None`), `context_source_keys` |
| **Any `todo!()`/`unimplemented!`/stubs?** | **None found.** Every method either does real work or is an intentionally correct no-op or default decline. |
| **Op coverage** | **166 entries** registered in `src/kernels/mod.rs` (verified with `grep -c "reg.register"` → 166). Domains: standard ONNX (`""`, opset-versioned: MatMul, Gemm, Softmax, LayerNorm, GatherND, Reshape, Cast, Conv, …), `com.microsoft` (MatMulNBits, QMoE, GatherBlockQuantized, FusedGemm, MultiHeadAttention, GQA, SkipSimplifiedLayerNorm, CausalConvWithState, RotaryEmbedding, …), `pkg.nxrt` (BlockQuantizedMatMul, BlockQuantizedMoE, IndexShare, VarlenAttention, PackedVarlenAttention, CompressedSparseAttention, …). |
| **Device / memory model** | Host-only. Buffers are `malloc`/`free` via `onnx_runtime_memory_governor::HostAllocator` (swappable). All pointers are dereferenceable host pointers. No streams, no device context, no fences at runtime. |
| **Build deps** | Vendored MLAS (`mlas-sys`, via the default `mlas` feature) linked in as an *internal* backend of this EP — it requires a C++/asm toolchain + cc crate, and every MLAS symbol stays local to the cdylib (0 exported, 0 undefined), so nothing binds to ORT's own copy. `--no-default-features` builds a pure-Rust cdylib for hosts without that toolchain. No CUDA toolkit. |
| **Readiness for outbound ORT plugin export** | ✅ **DONE (M1+M2)** — exported via `export_ep_factories!` macro in `crates/onnx-runtime-ep-cpu-plugin/`. The `as_ort_plugin()` hook in `provider.rs` still returns `None` (not used); the export path goes through the plugin cdylib crate, not through the trait hook. 23 ORT conformance tests pass including f16/bf16; dtype-aware capability claiming wired. |

---

### 2.2 `onnx-runtime-ep-cuda` — `CudaExecutionProvider`

| Field | Detail |
|---|---|
| **Crate / path** | `crates/onnx-runtime-ep-cuda/src/` |
| **`impl ExecutionProvider`?** | Yes — `src/provider.rs:513` |
| **Trait methods — required** | `name` ✅ (returns `"cuda_ep"`), `device_type` ✅ (Cuda), `device_id` ✅ (CUDA:ordinal), `initialize` ✅ (binds CUDA context), `shutdown` ✅, `supports_op` ✅ (same registry-keyed pattern as CPU; actionable declines), `get_kernel` ✅ (registry lookup → factory.create), `allocate` ✅ (VMM arena or `cuMemAlloc` path), `deallocate` ✅ (cross-device guard, borrowed no-op), `copy` ✅ (dtod with size check), `copy_async` ✅ (htod/dtod on dedicated transfer stream, returns real `Fence`), `sync` ✅ (`runtime.synchronize()` at `provider.rs:1500`) |
| **Trait methods — real overrides beyond required** | `capabilities` ✅ (advertises nxrt weight-paging when offload enabled), `page_lazy_weight` ✅ (LRU residency cache page-in), `wait_fence` ✅ (compute stream wait), `record_compute_fence` ✅ (WAR fence), `copy_wait_fence` ✅, `copy_from_host/copy_from_host_at/copy_to_host` ✅, `device_argmax_supported` ✅ (→ `true`), `device_argmax` ✅ (CUDA kernel), `begin/end/abort_device_graph_capture` ✅, `replay_device_graph/replay_device_graph_segment` ✅, `reset_device_graph` ✅, `check_device_capture_error` ✅, `allocate_committed`, `commit_allocation_range/ranges`, `decommit_allocation_range`, `allocation_committed_bytes`, `allocate_with_mapped_growth`, `commit_allocation_ranges_with_mapped_growth`, `mapped_bytes_for_allocation*`, `deallocate_with_unmapped`, `reserve_workspace`, `prepare_mapped_growth`, `release_mapped_growth`, `commits_on_demand`, `set_weight_residency_budget`, `adopt_memory_governor`, `custom_passes` |
| **Stubs / placeholder methods** | `prefetch_lazy_weight` at `provider.rs:564–573`: body is `let _ = (self, key, weight, source); Ok(false)` — acknowledged intentional deferral but **not yet implemented**. Claim-comment says "Phase 2a". |
| **Any other `todo!()`?** | **None found.** |
| **Op coverage** | **109 entries** registered in `src/kernels/mod.rs` (verified with `grep -c "reg.register"` → 109). Covers MatMul, GEMM, custom attention, fused ops, quantized matmul, MoE, activation, norm, element-wise, rotary embedding, etc. — subset of CPU EP coverage. |
| **Device / memory model** | Device buffers (CUDA device-virtual pointers, not host-dereferenceable). Separate compute stream + transfer stream for async weight paging. CUDA events as `Fence`. Optional VMM arena (physical granule mapping). Memory governor integration. cuBLASLt for GEMM. cuDNN for attention. |
| **Build deps** | CUDA toolkit ≥ 12.6, cuBLAS, cuDNN, `cudarc`. Does **not** build without CUDA toolkit. Cannot use `--no-default-features` and still compile this crate. |
| **Readiness for outbound ORT plugin export** | **HARDWARE-VALIDATED (core path)** — PR #832 registered `onnx-runtime-ep-cuda-plugin` with ORT on an NVIDIA H200, discovered 8× `cuda_ep` devices, and executed single- and multi-node graphs on-device with correct results. The four implementation defects (B1–B4) are resolved. It still fails closed by design (zero factories when no GPU is available). Issue #768 now tracks only the residual items in `docs/execution/CUDA_EP_STATUS.md` §7. |

---

## 3. Non-Candidates Confirmed

| Crate | Why excluded |
|---|---|
| `onnx-runtime-eager` | Orchestrator; holds `Vec<Arc<dyn ExecutionProvider>>` (`lib.rs:67`). Dispatches to EPs, is not one. |
| `mlas-sys` | BLAS kernel library (C++/asm). Default-on internal backend for the CPU EP's MatMul. Not an EP. |
| `LegacyOrtEp` (`onnx-runtime-ep-api/src/abi/mod.rs:160`) | **Inbound** adapter: loads an existing ORT plugin `.so` via `CreateEpFactories` and wraps it as a Rust `ExecutionProvider`. Direction is wrong for outbound export. |
| `PluginExecutionProvider` (`onnx-runtime-session/src/plugin_provider.rs:72`) | **Inbound** bridge: claims subgraphs for a loaded plugin EP and delegates unclaimed ops to an embedded CPU EP. Not an in-repo EP to export. |
| `onnxruntime-mlx` | Does not exist in this workspace. No reference found. |

---

## 4. Answers to Justin's Questions

### Q1: Best first candidate for end-to-end ORT plugin-EP ABI export

**Answer: `onnx-runtime-ep-cpu` confirms Justin's prior.**

Evidence:
- **Trait completeness:** Every required `ExecutionProvider` method is real and tested. No `todo!()`. No stubs. The CUDA EP has one real stub (`prefetch_lazy_weight`) and substantially more per-method complexity.
- **Op coverage:** 166 registrations vs 109 for CUDA. More surface area is covered first.
- **Build deps:** Pure-Rust (no external toolchain required for the base build). The ORT plugin-EP adapter crate can build and be integration-tested in standard CI without GPU hardware or CUDA toolkit.
- **Memory model simplicity:** Host pointers are already C ABI–compatible (`*const c_void` / `*mut c_void` + `size_t`). No stream or event handles need to cross the ABI. A `DeviceBuffer` for the CPU EP is literally `(ptr, size, align)` — maps directly to what `OrtMemoryInfo` / `OrtAllocator` expects.
- **Testability:** All existing CPU EP tests run without special hardware. The ORT plugin adapter tests can run in the same environment.

### Q2: Shared vs per-EP split

**Shared (adapter crate owns once):**

| Concern | Evidence |
|---|---|
| `CreateEpFactories` export symbol + `OrtEpFactory` struct layout | ABI fixture at `crates/onnx-runtime-ep-api/tests/fixtures/legacy_plugin_stub.c` defines the canonical shape (name, vendor, get_supported_devices, create_ep, release_ep, ort_version_supported) |
| `EpConfig` → key-value pair serialization | `crate::provider::EpConfig.options: HashMap<String, String>` is already format-agnostic |
| Error → `OrtStatus` conversion | `EpError` to machine-parseable code + message (existing `abi/mod.rs` already does this for inbound; same logic inverted for outbound) |
| `KernelMatch` → capability response | Shared accept/decline wire format |
| Op-capability advertisement (GetSupportedOps / GetCapabilities bridge) | Uses `ExecutionProvider::supports_op` uniformly |
| `OrtPluginExport` scaffolding (currently just `register_symbol: String`) | Needs to grow but lives in `provider.rs` for all EPs to inherit |

**Irreducibly per-EP:**

| Concern | Why per-EP |
|---|---|
| Device enumeration / `GetSupportedDevices` | CPU returns one CPU device; CUDA enumerates `cu_get_device_count()` ordinals |
| Buffer handles crossing the ABI | CPU: raw host `*mut u8`; CUDA: opaque CUDA device pointer (needs `CUdeviceptr` handle exchange) |
| Stream / fence representation | CPU has none; CUDA has compute + transfer stream + event IDs |
| Memory governor wiring | CUDA has VMM arena + mapped-growth protocol; CPU is just `malloc` |
| `prefetch_lazy_weight` | CPU: no-op default; CUDA: deferred stub — bespoke when implemented |

**Implication:** CPU-EP adapter is purely mechanical (no bespoke device logic). The CUDA adapter is non-trivial in the buffer-handle and stream-handle sections but still follows the same contract shape.

### Q3: `ExecutionProvider` trait gaps for the ORT plugin-EP lifecycle

**Missing or wrong-shaped methods (status updated for M1+M2):**

| Gap | Evidence | Severity | M1+M2 status |
|---|---|---|---|
| `as_ort_plugin()` returns `None` by default; no EP overrides it | `provider.rs:942–944` | ~~Blocking~~ | ✅ **Resolved.** Export goes through `export_ep_factories!` macro in `onnx-runtime-ep-cpu-plugin`; the trait hook is not used. |
| `OrtPluginExport` struct carries only `register_symbol: String` | `provider.rs:309–313` | ~~Blocking~~ | ✅ **Resolved.** CPU EP export path does not use `OrtPluginExport`. The plugin cdylib is authoritative. |
| No method for `GetSupportedDevices` (ORT lifecycle step 1) | Absent from trait | **Gap** | ✅ Resolved — `factory.rs` + `device.rs` implement `GetSupportedDevices` in the plugin adapter. |
| No method for capability-claim graph walk (ORT lifecycle step 2: `GetCapabilities`) | The inbound direction is handled by `abi/mod.rs`'s subgraph claim code; the outbound direction has no trait method | **Gap** | ✅ Resolved — `ep.rs` implements `GetCapability` by calling `supports_op`/`supports_node` through the `ShapeInference` filter. |
| `KernelMatch` → Rust enum, cannot cross FFI | `provider.rs` returns `KernelMatch::Supported { cost, … }` | **ABI-safety** | ✅ Resolved in `ep.rs` — mapped to `OrtNodeComputeInfo`. |
| `EpError` / `Result<T, EpError>` → Rust enum, cannot cross FFI | Every required method returns `Result<_, EpError>` | **ABI-safety** | ✅ Resolved in `status.rs` — `fail_status(msg)` converts to `*mut OrtStatus`. |
| `Kernel` trait (returned by `get_kernel`) → Rust vtable, cannot cross FFI | `Box<dyn Kernel>` at `provider.rs` / kernel.rs | **ABI-safety** | ✅ Resolved in `compute.rs` — `ComputeState` wraps kernels; `CreateState`/`Compute`/`ReleaseState` expose `OrtNodeComputeInfo`. |
| `DeviceBuffer` struct itself cannot cross FFI | Contains `NonNull<c_void>`, `usize`, `usize`, `BufferOwner` (private enum) | **ABI-safety** | ✅ Resolved for CPU. CUDA path (device pointers) exercised on an H200 in #832 — see `docs/execution/CUDA_EP_STATUS.md`. |
| Lifetimes in `supports_op`/`get_kernel` (`&Node`, `&[Shape]`, etc.) | These are Rust-lifetime-bearing references that would need to be projected to C structs with explicit ownership | **ABI-shape** | ✅ Resolved — graph is constructed in `ep.rs` from ORT's `OrtGraph` callbacks; lifetimes stay within the callback frame. |

**Not missing, will work as-is:**
- `initialize` / `shutdown` lifecycle maps cleanly to `CreateEp` / `ReleaseEp`.
- `name()` → `GetName` (returns `&'static str`, trivially a C string constant).
- `copy_from_host` / `copy_to_host` defaults are sufficient for host-accessible devices.
- The `OpRegistry` key shape (`op_type`, `domain`, `since_version`) maps to the ORT `GetCapabilities` format.

### Q4: Placeholder honesty check

**Stubs pretending to be complete (updated for M1+M2):**

| Location | What it looks like | What it really is | Status |
|---|---|---|---|
| `provider.rs:309–313` (`OrtPluginExport`) | A named struct suggesting "this EP exports as an ORT plugin" | A marker with only a `register_symbol: String` field. No C function pointers. No factory wiring. | ✅ Not the export path. CPU EP uses `export_ep_factories!` macro in plugin cdylib. CUDA EP still has no working export. |
| `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573` (`prefetch_lazy_weight`) | A method on the CUDA EP trait impl | Body is `let _ = (self, key, weight, source); Ok(false)`. Returns "no prefetch enqueued" unconditionally. Disables double-buffer prefetch for CUDA weight paging even when offload is enabled. Deckard decision: deferred to post-Phase-2a. | 🔴 Still a stub. No change. |
| `as_ort_plugin()` default in both real EPs | Trait method exists, EPs appear to have a path to export | Default returns `None`; neither EP overrides it. | ✅ CPU EP export is done via plugin cdylib. CUDA EP export is incomplete — `onnx-runtime-ep-cuda-plugin` exists as scaffold only. |

**Nothing found that claims op support it does not have.** Both `supports_op` implementations guard unsupported paths with actionable `deny!()` calls. The `registry.supports(...)` check in both EPs is the single authoritative gate. No silent fallback, no fake `Supported` return.

---

## 5. Summary

| EP | Crate | `impl EP`? | `todo!`/stubs | Op registrations | Memory model | Build deps | ORT export readiness |
|---|---|---|---|---|---|---|---|
| **CpuExecutionProvider** | `onnx-runtime-ep-cpu` | Yes (`provider.rs:118`) | None | **166** | Host-only, `malloc`/`free` | Rust + vendored MLAS (default; opt-out) | ✅ **DONE (M1+M2)** — `onnx-runtime-ep-cpu-plugin` is a working ORT plugin EP; 23 conformance tests pass including f16/bf16; dtype-aware capability claiming via `GetKernelRegistry` |
| **CudaExecutionProvider** | `onnx-runtime-ep-cuda` | Yes (`provider.rs:513`) | `prefetch_lazy_weight` stub | **109** | Device pointers, streams, VMM | CUDA ≥ 12.6 at runtime (dynamic-loading build, no build-time dep) | 🟢 **H200-VALIDATED (#832)** — `onnx-runtime-ep-cuda-plugin` loads in ORT on an H200 and executes graphs on-device; it still fails closed (zero factories) without a GPU. All four implementation defects (B1–B4) are resolved. #768 tracks the residual items in `docs/execution/CUDA_EP_STATUS.md` §7. |
| LegacyOrtEp | `onnx-runtime-ep-api` | Yes (inbound only) | — | — | Inbound adapter | — | Not a candidate |
| PluginExecutionProvider | `onnx-runtime-session` | Yes (inbound bridge) | — | — | Inbound bridge | — | Not a candidate |
| onnx-runtime-eager | `onnx-runtime-eager` | No (orchestrator) | — | — | — | — | Not a candidate |
| mlas-sys | `mlas-sys` | No (BLAS lib) | — | — | — | — | Not a candidate |

---

## 6. Roy Verification Note — Updated for B1-B4 corrective wave (2026-08-11 @ c1d2556b5)

Re-ran `grep -rn "impl ExecutionProvider" crates/` independently. Results match Deckard's
inventory exactly: 2 production EPs (CPU + CUDA), 2 inbound adapters, 7 test/mock
implementations (excluded). Inventory is complete and correct.

**CUDA EP status correction:** The CUDA plugin is **hardware-blocked**, not
implementation-blocked. All four defects (B1–B4) are resolved in code. The plugin
fails closed (zero factories) when no GPU is present. Hardware validation is tracked
by #768; the repository has no self-hosted GPU runners.

**CPU EP plugin:** Working end-to-end. 154 lib + 9 parity + 6 ABI + 20 ORT e2e = 189
passing tests, 1 ignored (LayerNorm Mean-shape bug, being fixed by Batty).
