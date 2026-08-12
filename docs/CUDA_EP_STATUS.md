# CUDA EP Status — Hardware-Validated Core; Workspace/Teardown Deltas Unvalidated

**Authors:** Roy (Lead), Sapper (GPU/Systems), Nabil (FFI/Systems — B1/B3/S4),
Sebastian (Performance — H200 validation in #832, PR #830 revision 3)
**Updated:** 2026-08-12 (PR #830 revision 3, rebased on `main` @ `8ed44e1cc`)
**Branch:** `squad/cuda-plugin-runtime` (draft PR #830, follows merged #762, #832)

> **Read this first.** The situation changed on 2026-08-12: PR
> [#832](https://github.com/justinchuby/onnx-genai/pull/832) (`2b62c6204`) was
> merged to `main` after being **validated on a physical NVIDIA H200**. It is
> now the authority for the CUDA EP's core execution path — including plugin-EP
> graph execution in ORT. This document describes what #832 established and,
> separately, the narrow delta PR #830 adds on top of it, which is **not**
> hardware-validated.
>
> **No NVIDIA GPU exists in the environment PR #830 was developed in**
> (`nvidia-smi` is absent, `cuInit` returns `CUDA_ERROR_NO_DEVICE`), so nothing
> in §4–§7 is evidence about real CUDA hardware.

---

## 0. Evidence Ledger

| Claim | Evidence class | Where |
|---|---|---|
| Native CUDA EP runs a 52-layer bf16 int4 decoder with zero CPU fallback | **H200-validated** (#832) | §1 |
| ORT loads `libonnx_runtime_ep_cuda_plugin.so` and discovers 8× `cuda_ep` H200 devices | **H200-validated** (#832) | §1 |
| `CreateEp` succeeds for a shared EP and reuses the factory's EP instance | **H200-validated** (#832) — session executed on-device | §2 |
| Fused multi-node subgraph intermediates are device memory via `KernelContext_GetScratchBuffer` | **H200-validated** (#832) — single- and multi-node models produced correct on-device results | §2 |
| `CreateEp` hands back the *same* shared instance (not a name match) | **Host-tested** falsifier for the above | §6.1 |
| Two ORT sessions share exactly one EP instance | **Host-tested** (real ORT) | §6.2 |
| Step-scoped kernel workspaces are served from ORT scratch, correctly aligned | **Host-tested** (real ORT + demonstrated falsifiers) | §4, §6.3 |
| `SessionPersistent` workspaces are declined, never downgraded | **Host-tested** (real ORT + demonstrated falsifier) | §4, §6.3 |
| Workspace over-allocation / align-up / overflow / containment arithmetic | **Host-tested** (unit falsifiers) | §4 |
| Shared-EP teardown runs `shutdown()` exactly once on the normal path | **Host-tested** (real ORT + unit falsifiers) | §5, §6.2 |
| `Alloc(0)` is normalised at the adapter boundary | **Host-tested** (unit falsifier with a zero-hostile allocator) | §3 |
| A poisoned `alloc_sizes` lock does not leak the allocation | **Host-tested** (unit falsifier) | §3 |
| `CanCopy` same-device via `MemoryDevice_GetDeviceId` | **Compile-verified only**; cross-device fails closed | §7 |
| Workspaces satisfy real cuBLASLt / FlashAttention alignment on device | **Not verified** — hardware only | §7 |
| Explicit shutdown ordering on a real CUDA context | **Not verified** — hardware only | §7 |

"Host-tested" means a test in this repository fails if the behaviour regresses,
and that failure has been *demonstrated* by breaking the implementation (§6.3).

---

## 1. What #832 Established on Hardware

PR #832 ran the native CUDA EP and the ORT plugin-EP on an **NVIDIA H200**
(driver 580.105.08, CUDA 13.0):

- **Native EP.** Muse-Glimmer-30B int4 (bf16 activations, KV cache and scales;
  52 layers) runs embedding + decoder entirely on the CUDA EP with
  `ONNX_GENAI_REQUIRE_CUDA=1` and zero CPU fallbacks, producing coherent text.
  Eager decode: median **11.08 tok/s** (~90 ms/token), up from 5.87 tok/s after
  a bf16 `MatMulNBits` path plus a cached grow-only per-kernel scratch arena
  that removed per-call `cuMemAlloc`/`cuMemFree` device syncs. Capture-safe.
- **Plugin EP in ORT.** ORT 1.28 registers the plugin cdylib, discovers 8×
  `cuda_ep` (vendor `nxrt`) H200 devices, and executes **both a single-node and
  a multi-node fused (Add→Mul) graph on-device with correct results**
  (`scripts/validate_plugin_ep_ort.py`).
- **Four ops fixed** that claimed placement but rejected bf16 at runtime: `Clip`
  (integer), `MatMulNBits` (bf16 activations), `GroupQueryAttention` (bf16
  rotary cache), `SkipSimplifiedLayerNormalization` (bf16).

Issue [#768](https://github.com/justinchuby/onnx-genai/issues/768) is no longer
a blanket "CUDA EP unvalidated" tracker. Its remaining scope is the items in §7.

---

## 2. What PR #830 Dropped Because #832 Superseded It

PR #830 was developed in parallel and independently reached the same two
defects. Since #832's fixes carry hardware evidence and #830's do not, **#830
now takes `main`'s implementations verbatim** and drops its own:

| Area | #830's dropped alternative | Kept instead (`main`, #832) |
|---|---|---|
| Shared-EP ownership | Lock-free `Arc<dyn ExecutionProvider>` refactor across `ep.rs`, `factory.rs`, `device.rs`, `transfer.rs` and the CUDA plugin | `EpHandle::{Owned, Shared(Arc<Mutex<Box<dyn ExecutionProvider + Send>>>)}`; `ReleaseEp` shuts down only owned EPs |
| `CreateEp` for shared factories | `ExportedEp` holding a cloned `Arc<dyn …>` | `factory_create_ep` reusing the factory's shared EP through `EpHandle::Shared` |
| Fused-subgraph intermediates | `ExportedComputeInfo.ep` + `ScratchAllocator`/`EpAllocation`, i.e. **per-dispatch EP allocate/free** | `IntermediateBuf::scratch_ptr` via `KernelContext_GetScratchBuffer` |
| Per-value device tagging | `classify_value_device` in `kernel_ctx.rs` | Not needed: ORT scratch is already placed on the operands' memory device |

The per-dispatch allocate/free design was not merely redundant — it would have
put a synchronous `cuMemAlloc`/`cuMemFree` pair on **every node of every decode
step**, which is the exact cost #832 measured and removed, and a device free is
illegal inside CUDA-graph capture.

---

## 3. Retained: Allocator Boundary Hygiene

`Alloc(0)` is legal for an `OrtAllocator` and must yield a unique, non-null,
freeable pointer. Whether a backing EP allocator honours that is **not**
guaranteed — `cudaMalloc(0)` and third-party CUDA allocators (RMM, PyTorch's
caching allocator) differ. Normalisation therefore lives at the **adapter
boundary**, not in any allocator: `device.rs::normalize_alloc_size` rewrites
only 0 → `ZERO_SIZE_ALLOC_BYTES`, and `device_alloc` applies it before calling
the EP. The falsifier is a mock EP whose `allocate()` **rejects** zero bytes, so
an alternate CUDA allocator cannot regress this.

`device_alloc` also records the size under a poisoned lock rather than skipping
the record (`PoisonError::into_inner`). The previous `if let Ok(..)` /
`.lock().ok()` pair meant one panicking thread turned every subsequent free into
a silent no-op — an unbounded device-memory leak. `device_free` recovers from
poison for the same reason, and still no-ops on pointers it does not know rather
than fabricating `size = 0`.

---

## 4. Retained: Governed Kernel Workspaces

`compute.rs` dispatched every kernel through `Kernel::execute()`, so a kernel
declaring a workspace via `workspace_requirement()` — cuBLASLt GEMMs, tree
reductions, FlashAttention scratch — could never receive one. Every dispatch
(single-node and routed) now goes through `prepare_workspace` →
`Kernel::execute_with_workspace`.

**Mechanism.** Workspaces come from `KernelContext_GetScratchBuffer` against the
memory info of the kernel's own operands — the same call #832 validated on H200
for intermediates. ORT owns those bytes for the duration of `Compute`, which is
exactly a step-scoped workspace's lifetime, so this executor never issues a
device free and nothing here is unsafe during graph capture.

**Alignment.** ORT promises no alignment beyond its allocator's, so a stricter
request is met by **checked over-allocation** (`bytes + alignment - 1`) followed
by align-up inside the block. `workspace_block_bytes` and
`align_workspace_window` are separate, unit-tested functions: a non-power-of-two
or zero alignment is rejected before any allocation, `u64 → usize` conversion is
checked, every add is `checked_add`, the scratch pointer is null-checked, and
the aligned window is re-verified to lie inside the allocation before it is
handed out.

**Lifetime.** `StepScoped` is served as above. `SessionPersistent` is
**declined** (`None`) — never downgraded. Serving it from scratch would hand the
kernel a block ORT recycles when `Compute` returns, which a persistent consumer
would then reuse on the next `Run`. Alignment is still validated before the
decline, so a malformed requirement is an error rather than a silent `None`.

Declining is **exactly behaviour-preserving against `main`**, which is why it is
safe to ship without hardware: `main`'s executor calls bare `execute()` at both
dispatch sites (`compute.rs:1080`, `:1205`), and the
`Kernel::execute_with_workspace` default forwards to `execute()`, so passing
`None` reproduces `main` node for node.

What that behaviour *is* varies by kernel, and it is **not** uniformly a
self-owned fallback. Of the five `SessionPersistent` declarers in
`onnx-runtime-ep-cuda`:

| Kernel | Behaviour when declined |
|---|---|
| `GroupQueryAttention` | Self-owned pooled score scratch (documented compatibility path) |
| `StandardAttention` | Self-owned pooled/per-call score + staged-K/V scratch |
| `MatMulNBits` (via `blas::governed_workspace_requirement`) | Self-owned `Bf16Scratch` for bf16 staging; the f32 dequant-cuBLASLt path errors in `governed_workspace_ptr` |
| `BlockQuantizedMoE` | Hard error — no self-owned path |
| `IndexShare` | Hard error — no self-owned path |

The last two, and the `MatMulNBits` dequant-cuBLASLt path, **already fail this
way on the plugin path on `main` today**. This executor neither fixes nor
worsens them, and no claim is made here that they work. Making them work needs a
real session-persistent device arena at this seam, which is **future work**.

> Hard-failing on `SessionPersistent` instead of declining was tried and
> rejected: it would have turned every GQA-bearing model into a plugin-path
> error on the hardware #832 validated.

---

## 5. Retained: Explicit Shared-EP Shutdown Semantics

A shared EP is reachable from four ORT-owned surfaces — the `OrtAllocator`, the
`OrtSyncStreamImpl`, the `OrtDataTransferImpl`, and one `OrtEp` per session.
Each holds an `Arc` clone, so **no individual `Release*` callback may call
`shutdown()`**: doing so would tear down a runtime another live session still
needs. `factory_release_ep` accordingly only shuts down an `EpHandle::Owned`.

`ReleaseEpFactory` is the one point in the ORT lifecycle that happens after
every other surface has been released, so that is where explicit shutdown
belongs. `release_ep_factory_with_teardown` takes the factory's own `Arc` and
reports:

| Situation | Outcome | Behaviour |
|---|---|---|
| Factory is the last owner (normal teardown) | `ShutdownCalled` | `shutdown()` runs exactly once — the explicit documented cleanup path |
| `shutdown()` returned an error | `ShutdownFailed` | Reported on stderr, not swallowed; the EP is still dropped |
| A surface is still alive (ORT contract violation / leaked handle) | `StillReferenced { strong_count }` | Diagnostic printed, **no** `shutdown()`; falls back to the codified Drop-only invariant |
| Non-shared (owned, e.g. CPU) factory | `NotShared` | Unchanged |

**Compute-info holder audit.** On `main`, `ExportedComputeInfo` holds *no* EP
reference — workspaces and intermediates both come from ORT scratch — so a live
compute info can never keep the EP alive or block teardown, in either ordering.
What a compiled kernel does capture is its backend runtime (e.g.
`Arc<CudaRuntime>`), and `CudaExecutionProvider::shutdown()` only clears an
initialisation flag; real device teardown happens in `Drop`, which runs when the
last `Arc` goes away. Inverted teardown (compute infos released *after*
`ReleaseEpFactory`) is therefore not a use-after-free hazard, and is pinned by a
unit test.

---

## 6. Conformance

### 6.1 Vtable-level (`onnx-runtime-ep-plugin/tests/shared_gpu_conformance.rs`)

Drives our `extern "C"` callbacks directly, using a real dlopen'd ORT only to
build genuine `OrtValue`/`OrtMemoryDevice` objects, backed by a `MockCudaLikeEp`
tagged `DeviceType::Cuda`, `stream_aware: true`, `host_accessible: false`.

Covers `CreateAllocator` alloc/free including `Alloc(0)`; a caller-owned opaque
stream handle round-tripping unchanged; `CanCopy` H2D/D2H/H2H/same-device
classification; byte-accurate `CopyTensors`; and `CreateEp` returning an
`EpHandle::Shared` whose `Arc` is `ptr_eq` to the one every other surface holds.

This is the **CPU-runnable falsifier for #832's `CreateEp` fix** — it fails on
any host if the shared-instance invariant or release ordering regresses. It
proves ABI wiring and ownership; it proves nothing about CUDA.

### 6.2 Real ORT end-to-end (`onnx-runtime-ep-shared-mock-plugin`)

A shared-EP plugin (`publish = false`) built as a real cdylib and loaded by ORT
through the supported path:

```
RegisterExecutionProviderLibrary → GetEpDevices
  → SessionOptionsAppendExecutionProvider_V2 → CreateSession → Run
  → ReleaseSession → UnregisterExecutionProviderLibrary
```

`WorkspaceAddKernel` (op `Add`) declares a 256-byte-aligned workspace and its
`execute()` **always fails**, so a model containing `Add` can only run if the
executor honours the workspace contract. `PlainMulKernel` (op `Mul`) declares
`WorkspaceRequirement::NONE`, keeping the no-workspace path covered.

| Test | What it pins |
|---|---|
| `shared_ep_session_runs_and_workspace_is_plumbed` | step-scoped workspace served, aligned, correct results |
| `shared_ep_routed_subgraph_intermediates_come_from_ort_scratch` | routed multi-node values correct; `alloc_calls == 0` |
| `session_persistent_workspace_is_declined_not_downgraded` | `persistent_downgraded == 0` **and** `persistent_declined > 0` |
| `shared_ep_two_sessions_share_one_instance` | one EP instance across two sessions |
| `shared_ep_shutdown_runs_once_at_library_unregister` | exactly one `shutdown()` at unregister |

Every EP-allocation assertion is now `alloc_calls == 0`: workspaces and
intermediates must come from ORT scratch, so **this suite fails if per-dispatch
EP allocate/free is ever reintroduced**.

**The mock EP is deliberately CPU-typed.** `factory_get_supported_devices` can
only match hardware ORT actually enumerates; a GPU-typed mock is never selected
on a GPU-less host, so the suite would silently degrade into a skip. It
exercises the *plugin protocol*, not device memory.

### 6.3 Demonstrated falsification

Each claim was checked by breaking the implementation and confirming the tests
go red:

- returning `Some(workspace)` for a `SessionPersistent` request →
  `session_persistent_workspace_is_declined_not_downgraded` fails with
  *"the executor served a SessionPersistent workspace request from step-scoped
  memory"*;
- removing the align-up in `align_workspace_window` → four of the five ORT E2E
  tests fail with *"workspace pointer 0x… is not 256-byte aligned as requested
  by workspace_requirement"*;
- reverting either `execute_with_workspace` call site to `execute()` → every
  workspace test fails with *"WorkspaceAddKernel::execute called directly"*;
- `device.rs`'s falsifier uses an EP whose `allocate()` **rejects** zero bytes,
  and a second poisons `alloc_sizes` before freeing, so both the normalisation
  and the poison-recovery paths fail if removed;
- `factory.rs`'s falsifiers fail if `shutdown()` is skipped on normal teardown,
  called twice, or called while another surface is alive.

> **Stale-cdylib hazard.** `cargo test -p <pkg> --test <name>` builds the test
> binary and the rlib but does **not** refresh the crate's `cdylib`, so an
> integration test can silently validate a stale `.so` — this masked the first
> falsification attempt. `onnx_runtime_ort_testkit::find_plugin_cdylib` always
> runs `cargo build -p <package>` first (memoised; `NXRT_SKIP_PLUGIN_REBUILD=1`
> opts out), and derives the target dir, profile and `--target` triple from
> `current_exe()` so the rebuild refreshes the artifact the test actually loads.
> The old code read `PROFILE`, which cargo sets **only for build scripts**, so a
> `--release` test run resolved `target/debug/…`.

### 6.4 Running it

```sh
NXRT_REQUIRE_ORT_TESTS=1 cargo test \
  -p onnx-runtime-ep-plugin -p onnx-runtime-ep-cpu-plugin \
  -p onnx-runtime-ep-shared-mock-plugin -p onnx-runtime-ort-testkit \
  --no-fail-fast
```

`NXRT_REQUIRE_ORT_TESTS=1` turns an "ORT not found" skip into a hard failure.
`NXRT_<PLUGIN>_PLUGIN_PATH` pins a cdylib explicitly (for CI); without it the
testkit derives the path from the running test binary. The two test-only crates
are workspace `members` but not `default-members`, so they do not affect default
builds or publishing.

---

## 7. Still Hardware-Only — Not Verified by Anything Here

Everything in this list is about the PR #830 delta or was never in #832's scope:

- whether a governed workspace served from `KernelContext_GetScratchBuffer`
  satisfies real cuBLASLt / FlashAttention alignment and size requirements on
  device, and whether ORT's CUDA-side scratch is arena-backed (if it is not,
  each request is a real `cuMemAlloc` and the capture-safety argument narrows to
  "no free", not "no allocation");
- whether declining `SessionPersistent` leaves `GroupQueryAttention`,
  `StandardAttention` and `MatMulNBits` on a correct self-owned path under the
  plugin executor specifically (#832 validated the *native* path for those
  kernels, and the plugin path for `Add`/`Mul` and the 30B decoder, but not a
  plugin-path model that exercises `GroupQueryAttention`). `BlockQuantizedMoE`
  and `IndexShare` are known *not* to run on the plugin path either here or on
  `main` — that is a stated gap, not an open question;
- whether explicit `ReleaseEpFactory` shutdown ordering is correct against a
  live CUDA context with multiple sessions;
- whether `CanCopy`'s `MemoryDevice_GetDeviceId` comparison behaves correctly
  across real devices (compile-verified only);
- `prefetch_lazy_weight` remains a stub (§8).

**To validate the remaining workspace path on an H200**, use the same path #832
used:

```sh
cargo build --release -p onnx-runtime-ep-cuda-plugin --features cuda
python scripts/validate_plugin_ep_ort.py \
    target/release/libonnx_runtime_ep_cuda_plugin.so
# and a workspace-exercising model (GroupQueryAttention or a cuBLASLt GEMM)
# placed on cuda_ep, asserting numerics plus no per-node cuMemAlloc/cuMemFree
# in an nsys trace.
```

That was **not run for this revision** — the development environment has no
NVIDIA device (`cuInit` → `CUDA_ERROR_NO_DEVICE`). PR #830 stays **Draft** until
it is.

---

## 8. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs`. Returns
`Ok(false)` — "no transfer enqueued". Deferred to post-Phase-2a.
