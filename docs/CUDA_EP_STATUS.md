# CUDA EP Status — Honest Capability Table

**Author:** Roy (Lead)
**Date:** 2026-08-11
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)
**HEAD at time of writing:** `087d34888`

---

## Justin's constraint (hard)

> Do not claim working CUDA without real GPU validation.

This host has no CUDA toolkit and no GPU. No capability in this document has been
validated on hardware. "VALIDATED-ON-HARDWARE" in the table below means code
was actually run on a GPU and produced a verified result. Today that column is
empty for every row. Every row marked IMPLEMENTED was written and compiled by a
developer on this host; every row marked COMPILE-CHECKED was verified to build
with `cargo check --workspace` (which excludes the `cuda` feature). No row is
VALIDATED-ON-HARDWARE.

---

## Status column definitions

| Column | Meaning |
|---|---|
| **IMPLEMENTED** | Code exists; logic is complete; correctness is a developer claim, not a measurement. |
| **COMPILE-CHECKED** | Source was checked with `cargo check` or `cargo build --features cuda` with a real CUDA toolkit on a dev machine. Does not imply runtime correctness. |
| **VALIDATED-ON-HARDWARE** | Code was run on a physical CUDA GPU; output was compared to a reference (CPU or oracle); verdict: PASS or FAIL with evidence. |

---

## Capability Status Table

| Capability | IMPLEMENTED | COMPILE-CHECKED | VALIDATED-ON-HARDWARE | Notes |
|---|---|---|---|---|
| **Device enumeration** (`device_type`, `device_id`) | ✅ | ✅ (no-toolkit path via `cargo check`) | ❌ None | Returns `DeviceType::Cuda` and `DeviceId::new(Cuda, ordinal)`. Ordinal from constructor. No `cuDeviceGet` call — does not verify device exists at enumeration time. |
| **Initialize / bind device** (`initialize`) | ✅ | ✅ | ❌ None | Calls `self.runtime.bind()` to confirm device is reachable on current thread. Not run on real hardware. |
| **Device allocator** (`allocate`, `deallocate`) | ✅ | ✅ | ❌ None | VMM arena or `cuMemAlloc` path. Cross-device guard. No-op for borrowed buffers. |
| **Synchronous data transfer** (`copy`) | ✅ | ✅ | ❌ None | DtoD with size check. Uses `CUmemcpyDtoD`. No stream ordering. |
| **Async data transfer** (`copy_async`) | ✅ | ✅ | ❌ None | HTD/DTD on dedicated transfer stream; returns real `Fence` (CUDA event). |
| **Host↔device transfer** (`copy_from_host`, `copy_to_host`, `copy_from_host_at`) | ✅ | ✅ | ❌ None | Uses transfer stream; `copy_from_host_at` supports offset. |
| **Stream synchronization** (`sync`, `wait_fence`, `record_compute_fence`, `copy_wait_fence`) | ✅ | ✅ | ❌ None | `sync` calls `runtime.synchronize()` at `provider.rs:1500`. Fence methods use CUDA events. |
| **Capability advertisement** (`capabilities`) | ✅ | ✅ | ❌ None | Advertises nxrt weight-paging when `ONNX_GENAI_WEIGHT_OFFLOAD` enabled. |
| **Weight paging** (`page_lazy_weight`) | ✅ | ✅ | ❌ None | LRU residency cache page-in via `CudaWeightResidency`. |
| **Prefetch lookahead** (`prefetch_lazy_weight`) | ❌ **STUB** | ✅ | ❌ None | `provider.rs:564–573`: body is `let _ = (self, key, weight, source); Ok(false)`. Deckard's decision: deferred to post-Phase-2a. Returns `false` (no transfer enqueued). |
| **CUDA graph capture** (`begin/end/abort_device_graph_capture`) | ✅ | ✅ | ❌ None | Segmented capture. Assumes ownership of stream — incompatible with ORT plugin model (stream not owned by plugin). |
| **CUDA graph replay** (`replay_device_graph`, `replay_device_graph_segment`) | ✅ | ✅ | ❌ None | |
| **Device argmax** (`device_argmax_supported`, `device_argmax`) | ✅ | ✅ | ❌ None | CUDA kernel. |
| **VMM arena** (`allocate_committed`, `commit_allocation_range/ranges`, `decommit_allocation_range`, etc.) | ✅ | ✅ | ❌ None | `cuMemCreate`/`cuMemMap` VMM allocator. |
| **Op support query** (`supports_op`) | ✅ | ✅ | ❌ None | 109 entries registered in `src/kernels/mod.rs`. Registry-keyed, opset-aware, actionable declines. |
| **Kernel execution** (`get_kernel` + CUDA kernel dispatch) | ✅ | ✅ | ❌ None | cuBLASLt for GEMM; custom kernels for attention, norms, MoE, etc. No end-to-end validated run. |
| **ORT plugin-EP export** (`as_ort_plugin`) | ❌ **ABSENT** | N/A | ❌ None | `onnx-runtime-ep-cuda-plugin` crate exists as scaffold; `CreateEpFactories`/`ReleaseEpFactory` are wired behind `#[cfg(feature = "cuda")]`. Not a working CUDA plugin (device-pointer ABI marshaling and stream/context sharing remain undesigned). |
| **Compile-time build** | ❌ | — | — | Requires CUDA toolkit ≥ 12.6 + cuBLAS + cuDNN + `cudarc`. This host has none. `cargo check --workspace` excludes `cuda` feature. |

---

## `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573`

**Code:**
```rust
fn prefetch_lazy_weight(
    &self,
    key: u64,
    weight: &LazyWeight,
    source: &dyn onnx_runtime_ep_api::MmapRegionSource,
) -> Result<bool> {
    let _ = (self, key, weight, source);
    Ok(false)
}
```

**Deckard's decision:** Explicitly deferred to post-Phase-2a. The Phase-2a CUDA EP
scope is standard GEMM (`MatMul`) via cuBLASLt. Prefetch lookahead requires the
weight-residency cache infrastructure and accurate stream-ordering guarantees that
are not needed for Phase-2a correctness. The `Ok(false)` return is semantically
correct: it means "no transfer was enqueued," which is true for a no-op. It does
not claim to have prefetched and have that claim fail — it honestly declines.

This is a **real functional gap**: a model relying on prefetch lookahead for memory
pressure management will not benefit from it. Not a correctness bug for the current
scope.

---

## Running the Hardware Conformance Runner

**`scripts/cuda_conformance_runner.sh` is committed at HEAD `087d34888`.**

### Preconditions (runner validates these before testing)

1. `nvidia-smi` reachable and a GPU detected.
2. `libcuda.so` loadable (via `ldconfig -p` or standard CUDA paths).
3. `libcublasLt.so.13` loadable.
4. At least one CUDA GPU present.
5. CUDA toolkit ≥ 12.6, cuBLAS, cuDNN, Rust stable.

If any precondition fails, the runner exits with **code 2 (UNVALIDATED)** — not a
test failure, just "can't run here."

### Exit code contract

| Code | Meaning |
|---|---|
| **0** | VALIDATED — all tests passed on real GPU hardware |
| **1** | FAILED — test failures on GPU hardware (real bugs) |
| **2** | UNVALIDATED — preconditions not met (no GPU, no driver, `cuda` feature not enabled) |

### Build and invoke

```bash
# Build with CUDA feature first (requires CUDA toolkit ≥ 12.6):
cargo build --features cuda -p onnx-runtime-ep-cuda

# Run the conformance suite:
./scripts/cuda_conformance_runner.sh

# Target a specific GPU:
CUDA_VISIBLE_DEVICES=0 ./scripts/cuda_conformance_runner.sh
```

### What the runner exercises

- `CudaExecutionProvider::initialize` → device bind → success
- `allocate` / `copy_from_host` / `copy_to_host` → bit-exact roundtrip
- At least one `MatMul` kernel invocation with reference output comparison
- `sync` / `wait_fence` ordering under concurrent dispatch
- `device_argmax` on a known vector
- Weight offload with `ONNX_GENAI_WEIGHT_OFFLOAD=1`

The runner outputs per-test PASS/FAIL, CUDA device name, toolkit version, driver
version, commit SHA, and a validated timestamp.

**This host has no CUDA GPU.** The conformance runner was not run here. Every row
in the VALIDATED-ON-HARDWARE column above remains empty.

---

## Known Hard Blockers for a Working CUDA ORT Plugin

These are engineering gaps, not merely hardware-absent gaps:

| Blocker | Reason | Status |
|---|---|---|
| **Device-pointer ORT ABI marshaling** | ORT's plugin ABI passes device pointers through `OrtKernelContext_GetInput`. nxrt's CUDA EP uses `DeviceBuffer` (opaque CUDA device pointer). Adapting the two requires a shared CUDA context between ORT and the plugin. Design unstarted. | 🔴 Not designed |
| **Stream/context sharing** | nxrt creates its own CUDA primary context and compute stream. ORT provides its own context/stream. Kernels must execute on ORT's stream or synchronize. Must either adopt ORT's context or introduce sync points. | 🔴 Not designed |
| **cuBLAS/cuDNN handle binding** | nxrt's handles are bound to its own stream. Must be re-bound for ORT's context/stream. | 🔴 Not designed |
| **Weight paging in plugin model** | nxrt's `CudaWeightResidency` manages its own VMM pool; in the plugin model, ORT owns weight memory. The paging system cannot operate without a redesign. | 🔴 Not designed |
| **CUDA graph capture in plugin model** | nxrt's graph capture assumes stream ownership; ORT may have its own capture semantics. Must be disabled or coordinated. | 🔴 Not designed |

These are in addition to the hardware requirement. A host with a GPU still cannot
produce a working CUDA ORT plugin without first resolving the design blockers above.
