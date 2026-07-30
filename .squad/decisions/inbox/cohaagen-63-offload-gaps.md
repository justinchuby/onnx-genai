# GAP REGISTER — Issue #63 "Complete live GPU weight offload"

Author: Cohaagen (CUDA/systems). Date: 2026-07-30. Branch: `squad/63-live-weight-offload`.
Verified against `origin/main` @ 2975f392 by source-tracing the live decode dispatch path.

## Summary (current-vs-target)

**Target:** models whose int4 weights exceed VRAM still run by paging weights
host↔device on demand during decode, byte-identical to the resident path.

**Current reality:** live CUDA weight/expert paging is a **no-op end-to-end**.
The building blocks exist but are not connected. The `ONNX_GENAI_WEIGHT_OFFLOAD`
knobs only drive the **CPU** QMoE host-cache subsystem
(`crates/onnx-runtime-ep-cpu/src/weight_offload.rs`); on CUDA nothing pages.

## What works today (verified)

1. `CudaWeightPager` (`crates/onnx-runtime-ep-cuda/src/weight_paging.rs`)
   allocates a bounded VRAM page and copies canonical external-data region bytes
   H2D, byte-identical to a resident upload. Implements
   `LazyDeviceWeightBinder::bind_block_quantized_moe`. Region-accurate (offset +
   len), frees on drop. Covered by `tests/weight_offload_gpu.rs` (2 GPU tests).
2. `CudaExecutionProvider::weight_pager()` constructs the pager (provider.rs:117).
3. Executor lazy-weight seam infrastructure exists: `build_lazy_weight_handles`
   (session `executor/build.rs:342`) and dispatch passing `KernelInput::Weight`
   (session `executor/dispatch.rs:1296`).
4. CPU host-cache offload (`ONNX_GENAI_WEIGHT_OFFLOAD` + `_HOST_BYTES` +
   `_PREFETCH`) is a separate, working subsystem for CPU QMoE.

## Gaps (precise seams)

- **GAP A — capability never advertised.** No production EP overrides
  `ExecutionProvider::capabilities()`; both CPU and CUDA return
  `ExecutionProviderCapabilities::stock()` (ep-api `provider.rs:362`). Because
  `build_lazy_weight_handles` early-returns unless the EP advertises
  `NXRT_WEIGHT_PAGING_CAPABILITY` ("nxrt"), **no `WeightHandle::Lazy` is ever
  created** → the entire lazy paging path is dormant in production.
- **GAP B — no kernel consumes lazy weights.** No real kernel (CPU or CUDA)
  overrides `Kernel::execute_with_inputs`; the default adapter errors on a
  `KernelInput::Weight`. The CUDA `BlockQuantizedMoEKernel` never calls the
  pager. So even if a lazy handle reached dispatch, it would fail.
- **GAP C — dispatch never invokes the pager.** `build_input_bindings`
  (dispatch.rs:827) pushes lazy inputs as an *absent* view (`present: false`,
  null ptr). `CudaWeightPager::bind_block_quantized_moe` is called only from
  tests, never in live decode.
- **GAP D — no residency/eviction.** `CudaWeightPage` frees on drop; there is no
  LRU/residency manager, so a paged weight cannot be retained across decode
  steps or evicted under a VRAM budget. Without this, paging would re-upload
  every step.
- **GAP E — boundary too narrow.** `LazyWeightBoundary::matches` only recognizes
  `pkg.nxrt::BlockQuantizedMoE` (ep-api `weight.rs:100`). Real DeepSeek/GLM/Qwen
  exports use `com.microsoft::QMoE`, so no real MoE weight is ever eligible.

## This increment (delivered in this branch)

- **GAP D:** add `CudaWeightResidency` — a bounded-VRAM LRU device-page cache
  built on `CudaWeightPager`, with H2D page-in, cache-hit reuse, and
  strong-count-safe LRU eviction. This is the "page-in + eviction" core the
  issue lists as the missing half of increment (1).
- Env-gated device policy `DeviceOffloadPolicy` reusing the existing
  `ONNX_GENAI_WEIGHT_OFFLOAD` switch plus a new
  `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES` VRAM budget, so residency can be
  enabled without regressing the default fast path (default: disabled → stock
  capabilities → byte-identical, unchanged decode).
- **GAP E:** extend `LazyWeightBoundary` to also recognize
  `com.microsoft::QMoE` (seam-ready; inert until capability is advertised).
- GPU tests forcing a tiny VRAM budget so eviction actually fires, asserting
  byte-identical output vs the resident path.

## Remaining work (next increments, enumerated in PR)

- **GAP A + B + C (end-to-end wiring):** give the CUDA MoE kernel (or a dispatch
  seam) access to an `MmapRegionSource` + `CudaWeightResidency` so a
  `KernelInput::Weight` is paged into VRAM and fed to the fused MoE kernel as a
  real device `TensorView`; then advertise `NXRT_WEIGHT_PAGING_CAPABILITY` from
  the CUDA EP behind the env flag. This is correctness-critical hot-path work and
  must be validated against a real BlockQuantizedMoE/QMoE model (blocked on the
  #384 load chain for 27B; validate mechanism on 7B first).
- Async H2D prefetch overlap (issue #87) and routed-expert paging (issue #82).

## Increment 2 (delivered in this branch — GAP A/B/C CLOSED)

Wired the pager into the **live decode hot-path**, gated behind
`ONNX_GENAI_WEIGHT_OFFLOAD=1`. Chosen architecture: resolve lazy weights to a
device pointer in the **dispatch layer** via a new kernel-agnostic EP trait
method, so all ~6600-line kernels stay **untouched** (correctness risk down).

- **GAP A — capability advertised.** `CudaExecutionProvider` now constructs a
  `CudaWeightResidency` from `DeviceOffloadPolicy::from_env()` and overrides
  `capabilities()` to return `nxrt_weight_paging()` **only when offload is
  enabled** (`residency.is_some()`); otherwise `stock()`. This is what makes
  `build_lazy_weight_handles` mint real lazy handles for boundary-matched
  weights. Default (offload off) → stock caps → byte-identical fast path.
- **GAP B — kernel-side resolution.** New `ExecutionProvider::page_lazy_weight`
  trait method (default `Ok(None)`); the CUDA EP implements it by paging the
  lazy weight through `CudaWeightResidency::resident_materialized` and returning
  a `PagedWeight` whose keep-alive pins the VRAM page for the kernel's lifetime.
  Stream-safety: `admit()` synchronizes the compute stream before evicting a
  page (skipped during graph capture, where offload does not run), so no
  in-flight kernel can reference freed VRAM — no use-after-free.
- **GAP C — dispatch invokes the pager.** `build_input_bindings` now calls
  `page_lazy_weight` for lazy inputs: on `Some` it binds a normal contiguous
  device view over the paged bytes and stores the `PagedWeight` keep-alive in
  `InInfo`; on `None` it leaves the input absent + `lazy_unresolved` so it still
  routes to the kernel as a `KernelInput::Weight`. `has_lazy_inputs` is
  recomputed from the unresolved set so paged weights take the normal compute
  path.
- **GAP E extended:** `LazyWeightBoundary` now also recognizes
  `com.microsoft::MatMulNBits`, so int4 MatMulNBits weights (where the VRAM
  pressure lives on non-MoE models like Qwen) are pageable.
- **Observability:** process-global page-in/hit/eviction counters
  (`global_offload_stats` / `reset_global_offload_stats`) so an opaque
  end-to-end decode can assert paging really happened.

### Validation (real numbers, Qwen3-0.6B int4 MatMulNBits, native CUDA, device 0)

Trusted model whose native-CUDA output is locked by `qwen3_0_6b_native_cuda_e2e`.

- Greedy 32-token decode, offload OFF → baseline token IDs.
- Same with `ONNX_GENAI_WEIGHT_OFFLOAD=1` + `..._DEVICE_BYTES=2097152` (2 MiB).
- **Token IDs IDENTICAL** (offload transparent/correct).
- **page_ins = 12544, evictions = 12541** (paging + eviction really fired).
- Cost: baseline 2.79 tok/s → offloaded 2.30 tok/s (**~1.21× slowdown** from H2D
  traffic under an aggressively tiny budget — expected).
- Test: `weight_offload_native_cuda_e2e::offloaded_native_cuda_decode_is_token_identical_and_pages`
  (`#[ignore]`, GPU + declared-io export required; skips gracefully on the #384
  metadata gap). Plus GPU unit test
  `weight_offload_gpu::residency_materialized_pages_evicts_and_matches_resident`.

### Still remaining after increment 2

- Async H2D prefetch overlap (issue #87) to hide paging latency.
- Routed-expert (per-expert) QMoE paging (issue #82) — page only the experts a
  token actually routes to, not the whole weight.
- 27B/35B-A3B end-to-end validation is blocked on the #384 native load chain
  (metadata `model.io` port auto-wiring); mechanism is proven on 7B/0.6B.

## Risks

- Correctness-critical EP: the default (offload-off) path must stay
  byte-identical. This increment keeps capability un-advertised, so decode is
  unchanged when the flag is off.
- Evicting a page whose VRAM is still referenced by an in-flight kernel would be
  a use-after-free; the residency manager only evicts pages with
  `Arc::strong_count == 1` (cache-only ownership).
- Cross-worktree contention: offload/QMoE area is active (Mary
  large-model-offload-validation, nandez flaky-qmoe-offload-residency). The
  ep-api/session boundary change is additive and inert without capability
  advertisement to minimize conflict.
