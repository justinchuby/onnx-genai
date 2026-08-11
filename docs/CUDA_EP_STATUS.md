# CUDA EP Status — Implementation-Blocked

**Author:** Roy (Lead)
**Updated:** 2026-08-11 (post-rejection rewrite)
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762 — REJECTED by review, kept draft)
**HEAD at time of writing:** `62f23440f`

---

## Summary — CUDA is implementation-blocked, not merely hardware-blocked

We have repeatedly described the CUDA EP as "hardware-validation-blocked" — that
the implementation was done and only a GPU was missing. **That was wrong.**

A rubber-duck review of PR #762 found the CUDA plugin was **failing open**: it
advertised a working GPU EP while every component behind it was broken. Four
implementation defects (§1 below) mean that even with a physical GPU present,
this code could not have produced correct results. Issue
[#768](https://github.com/justinchuby/onnx-genai/issues/768) (GPU hardware
validation) remains necessary but is **no longer sufficient**.

The CUDA plugin now **fails closed**: `CreateEpFactories` returns zero factories
and an actionable error status in both `cuda` and `no-cuda` feature
configurations. ORT is never offered an EP we cannot honour.

---

## 1. The Four Implementation Defects (B4 blocker, rubber-duck review)

These are the specific defects found by the review. Each is a specification for
whoever resolves it — they must all be fixed before the CUDA plugin may advertise
any factories.

| # | Defect | Why it breaks | Current state |
|---|--------|---------------|---------------|
| **1** | **Separate CUDA runtime/context per component.** The EP, allocator, and stream each construct independent CUDA runtime and context instances. | Allocations made by the allocator are invisible to kernels on the EP's stream. Stream-ordered copies are incoherent. Memory corruption on any real workload. | 🔴 Not fixed. A correct implementation must share a single `CUcontext` + `cudaStream_t` across EP, allocator, and stream. |
| **2** | **`CreateDataTransfer` returns NULL.** The factory wires a `DeviceDataTransfer`, but it lacks `OrtApi` access and has no shared CUDA stream, so `CopyTensors` returns an error unconditionally. | ORT calls `CanCopy` → `CopyTensors` to move data between host and device. A NULL or non-functional transfer means no data reaches the GPU. | 🔴 Not fixed. Requires `OrtApi` to be stored and a shared `cudaMemcpyAsync` stream. `CanCopy` now returns `false` (fail-closed), so ORT will not attempt the copy. |
| **3** | **`GetHandle` returns NULL stream handle.** `stream_get_handle` returns `null_mut()` — no `cudaStream_t` exists. | ORT and downstream consumers call `GetHandle` to order work on the EP's stream. A null handle causes undefined behaviour or silent fallback to the default stream, breaking ordering guarantees. | 🔴 Not fixed. Requires a real `cudaStream_t` owned or adopted by the plugin. |
| **4** | **`Free` passes `size=0` violating the allocator contract.** `device_free` reconstructs a `DeviceBuffer` with `size=0` because the allocation size is not tracked. | The EP's `deallocate` may require the size for `cudaFree`-style bookkeeping or arena management. Passing `size=0` violates the contract in `onnxruntime_ep_c_api.h`. | 🔴 Not fixed. Requires tracking allocation sizes (e.g. a `HashMap<*mut u8, usize>`) across `Alloc`/`Free`. |

**All four are implementation defects, not hardware-absent gaps.**

---

## 2. Fail-Closed Implementation

**File:** `crates/onnx-runtime-ep-cuda-plugin/src/lib.rs`

With the `cuda` feature enabled:
- `CreateEpFactories` sets `*out_num = 0` and returns an `OrtStatus*` error
  listing all four defects.
- The `cuda_impl` module's `construct_ep`, `device_support`, and
  `build_kernel_registry_entries` exist but are intentionally unused (suppressed
  with `let _ = &...`).
- `CanCopy` returns `false` unconditionally for device EPs
  (`crates/onnx-runtime-ep-plugin/src/transfer.rs`).

Without the `cuda` feature:
- `CreateEpFactories` returns zero factories with a "not available" status.

`ReleaseEpFactory` returns `OrtStatus*` (not `void` — see B2 correction below).

---

## 3. Status of the Standalone CUDA EP (nxrt-native, not ORT plugin)

The standalone CUDA EP (`crates/onnx-runtime-ep-cuda/`) implements the Rust
`ExecutionProvider` trait for direct use within nxrt. This is a separate path
from the ORT plugin export.

### Status levels

| Level | Meaning |
|---|---|
| **CODE EXISTS** | Source code is present and compiles via `cargo check` (using `cudarc` dynamic-loading — no CUDA toolkit at build time). Correctness is a developer claim that has **not survived review**. |
| **STUB** | Method body is a deliberate no-op or placeholder. Semantically honest (returns "not done" rather than claiming success). |
| **VALIDATED** | Code was run on a physical CUDA GPU with output compared to a reference. **No capability has reached this level.** |

### Capability table

| Capability | Status | Notes |
|---|---|---|
| Device enumeration (`device_type`, `device_id`) | CODE EXISTS | Returns `DeviceType::Cuda` and `DeviceId::new(Cuda, ordinal)`. No `cuDeviceGet` call. |
| Initialize / bind device | CODE EXISTS | `runtime.bind()`. Not run on hardware. |
| Device allocator (`allocate`, `deallocate`) | CODE EXISTS | VMM arena or `cuMemAlloc`. Cross-device guard. |
| Synchronous data transfer (`copy`) | CODE EXISTS | DtoD via `CUmemcpyDtoD`. No stream ordering. |
| Async data transfer (`copy_async`) | CODE EXISTS | HTD/DTD on transfer stream; returns `Fence`. |
| Host↔device transfer | CODE EXISTS | Transfer stream; `copy_from_host_at` supports offset. |
| Stream synchronization | CODE EXISTS | `runtime.synchronize()`. Fence methods use CUDA events. |
| Capability advertisement | CODE EXISTS | Advertises weight-paging when offload enabled. |
| Weight paging (`page_lazy_weight`) | CODE EXISTS | LRU residency via `CudaWeightResidency`. |
| Prefetch lookahead (`prefetch_lazy_weight`) | **STUB** | Body: `let _ = (self, key, weight, source); Ok(false)`. Deferred to post-Phase-2a. Returns "no transfer enqueued" — honest decline. |
| CUDA graph capture/replay | CODE EXISTS | Assumes stream ownership — **incompatible** with ORT plugin model. |
| Device argmax | CODE EXISTS | CUDA kernel. |
| VMM arena | CODE EXISTS | `cuMemCreate`/`cuMemMap`. |
| Op support query (`supports_op`) | CODE EXISTS | 109 entries in `src/kernels/mod.rs`. Registry-keyed, opset-aware. |
| Kernel execution | CODE EXISTS | cuBLASLt for GEMM; custom kernels. No validated run. |
| ORT plugin-EP export | **FAILS CLOSED** | `CreateEpFactories` returns 0 factories. See §1–§2. |

**Every "CODE EXISTS" row is unvalidated.** The previous table used "IMPLEMENTED"
which implied a level of confidence the review disproved. "Code exists" is the
honest description: the code compiles, but correctness is unproven.

---

## 4. What Must Happen Before CUDA Advertises Factories

### Phase A: Fix the four plugin defects (§1)

All four defects are implementation work, not hardware work. They can be designed
and partially implemented without a GPU (the shared-context architecture, the
allocation-size tracking, the `OrtApi` storage), but runtime validation requires
a GPU.

### Phase B: Resolve the ORT integration design gaps

| Gap | Description | Status |
|---|---|---|
| **Device-pointer ORT ABI marshaling** | ORT passes device pointers via `OrtKernelContext_GetInput`. nxrt uses `DeviceBuffer`. Requires shared CUDA context. | 🔴 Not designed |
| **Stream/context sharing** | nxrt creates its own primary context and compute stream. ORT provides its own. Must adopt ORT's context or synchronize. | 🔴 Not designed |
| **cuBLAS/cuDNN handle binding** | Handles bound to nxrt's stream; must rebind for ORT's context/stream. | 🔴 Not designed |
| **Weight paging in plugin model** | `CudaWeightResidency` manages its own VMM pool; ORT owns weight memory in the plugin model. | 🔴 Not designed |
| **CUDA graph capture** | Graph capture assumes stream ownership; incompatible with ORT. Must disable or coordinate. | 🔴 Not designed |

### Phase C: Hardware validation

Issue [#768](https://github.com/justinchuby/onnx-genai/issues/768). Requires a
CUDA GPU (compute ≥ 7.0, driver ≥ 535.x). No self-hosted GPU runner exists in
this repository.

**Phases A and B are prerequisites for Phase C.** #768 is necessary but not
sufficient — the code must be correct before hardware validation is meaningful.

---

## 5. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573`

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

Deckard's decision: deferred to post-Phase-2a. Returns `Ok(false)` — "no transfer
enqueued," which is true. Not a correctness bug; a functional gap for models that
rely on prefetch for memory pressure management.

---

## 6. Hardware Conformance Runner

`scripts/cuda_conformance_runner.sh` is committed. Exit codes: 0 = VALIDATED,
1 = FAILED, 2 = UNVALIDATED (preconditions not met).

**This host has no CUDA GPU.** The runner exits 2 (UNVALIDATED). Every capability
in §3 remains unvalidated. No self-hosted GPU workflow exists in this repository.

---

## 7. History — How We Got This Wrong

The original version of this document used a three-column IMPLEMENTED /
COMPILE-CHECKED / VALIDATED-ON-HARDWARE table. "IMPLEMENTED" was a developer
claim that did not survive review. The rubber-duck review of PR #762 found that
the CUDA plugin was failing open: it advertised a GPU EP while every component
behind it was non-functional. We described this as "hardware-validation-blocked"
in multiple documents and sessions. That framing was wrong — the code had
implementation defects that would cause failures on any host, GPU or not.

This rewrite corrects the framing. CUDA is implementation-blocked. The code
exists but is known non-functional. Issue #768 gates the final validation step
but cannot substitute for fixing the code.
