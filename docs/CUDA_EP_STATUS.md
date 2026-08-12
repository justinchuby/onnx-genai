# CUDA EP Status — Compiles and Passes Host Conformance; Unvalidated on GPU Hardware

**Authors:** Roy (Lead), Sapper (GPU/Systems), Nabil (FFI/Systems — B1/B3/S4),
Deckard (Systems Dev — `CreateEp` ownership unification),
Leon (Engine Dev, KV & Buffers — workspace/intermediate plumbing, ORT conformance)
**Updated:** 2026-08-12 (PR #830 revision 2, post-review)
**Branch:** `squad/cuda-plugin-runtime` (draft PR #830, follows merged #762)

> **Scope note, read this first.** Everything below is either (a) verified by
> tests that run on this host, (b) verified only by compilation, or (c) not
> verified at all. §0 states which is which for every claim. **No NVIDIA GPU
> exists in this environment** (`nvidia-smi` is not installed), so *nothing*
> in this document is evidence of correctness on real CUDA hardware.
> [#768](https://github.com/justinchuby/onnx-genai/issues/768) remains **open**
> and is the sole tracker for hardware validation.

---

## 0. Evidence Ledger

| Claim | Evidence class | Where |
|---|---|---|
| `CreateEp` succeeds for a shared EP and hands back the same `Arc` | **Host-tested** (real dlopen'd ORT, real `CreateSession`/`Run`) | §6.2 |
| Two ORT sessions share exactly one EP instance | **Host-tested** (real ORT) | §6.2 |
| Kernel workspaces are allocated on the EP and passed via `execute_with_workspace` | **Host-tested** (real ORT + unit falsifiers) | §4, §6 |
| Fused/routed subgraph intermediates are EP-allocated and EP-device-tagged | **Host-tested** (real ORT + unit falsifiers) | §5, §6 |
| Non-host-accessible EP without a usable allocator fails closed before execute | **Host-tested** (unit falsifier) | §5 |
| Shared-EP teardown runs `shutdown()` exactly once on the normal path | **Host-tested** (real ORT + unit falsifiers) | §7, §6.2 |
| `Alloc(0)` is normalised at the adapter boundary | **Host-tested** (unit falsifier with a zero-hostile allocator) | §8 |
| `CopyTensors` direction classification (H2D/D2H/D2D) | **Host-tested with a CPU-backed mock** — the *dispatch* is proven, the CUDA memcpy is not | §6.1 |
| `CanCopy` same-device via `MemoryDevice_GetDeviceId` | **Compile-verified only**; cross-device fails closed | §2 |
| `CudaExecutionProvider` construction, `libcuda.so` loading, stream validity | **Not verified** — hardware only | §9 |
| Real CUDA kernel numerics, device memory correctness, perf | **Not verified** — hardware only | §9 |

"Host-tested" means a test in this repository fails if the behaviour
regresses, and that failure has been demonstrated (see §6.3).

---

## 1. What This Revision Changes

PR #762 merged the CPU EP and the native nxrt ABI and left the CUDA plugin
deliberately fail-closed. PR #830 addresses the defects that made that
fail-closed state *permanent*, and then the plumbing gaps that would have made
a working `CreateEp` insufficient anyway.

**Revision 1 (commits `62a7d3547`, `5c46794fd`, `77b60a3f0`):**
`factory_create_ep` unconditionally returned an error whenever
`ExportedFactory::shared_ep` was set, so no ORT session could ever execute a
compiled subgraph through a shared-EP factory. The ownership model was unified
from `Arc<Mutex<Box<dyn ExecutionProvider + Send>>>` to a lock-free
`Arc<dyn ExecutionProvider>` so `CreateEp` clones the same `Arc` the
allocator/stream/transfer surfaces already hold. See §3.

**Revision 2 (commits `434ea677b`, `c7769e9dc`, `49af06f6c`, `4dc9ee908`):**
a working `CreateEp` exposed three further defects that would have made
execution *wrong* rather than merely impossible, plus a conformance gap:

- The plugin executor always called `Kernel::execute()`, so any kernel needing
  a governed workspace (cuBLASLt, reductions, FlashAttention scratch) failed at
  dispatch. Fixed in §4.
- Routed/fused subgraph intermediates were host `Vec<u8>` buffers tagged
  `DeviceId::cpu()` and handed to device kernels. Fixed in §5.
- Shared-EP shutdown had no explicit path on normal teardown. Fixed in §7.
- Conformance drove our own vtable directly rather than going through ORT.
  Complemented with a real `RegisterExecutionProviderLibrary` → `GetEpDevices`
  → `CreateSession` → `Run` suite. See §6.2.

---

## 2. Defect Ledger

| # | Defect | Resolution | Evidence class |
|---|--------|------------|----------------|
| **1** | **Shared runtime/context use-after-free (B1)** — raw pointer derived from a `MutexGuard` dangled once the guard dropped. | Every surface holds a lock-free `Arc<dyn ExecutionProvider>` clone. No raw pointers from guards; no mutex. | Compile-verified + host-tested ownership identity (`Arc::ptr_eq`) |
| **2** | **`CopyTensors` did not classify direction (B3)** — both src and dst were wrapped as device buffers. | `transfer_full_copy_tensors` classifies each tensor with `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType`, then dispatches `copy_from_host` / `copy_to_host` / `copy`. | Dispatch host-tested with a CPU-backed mock; the CUDA memcpy itself is hardware-only |
| **3** | **Panic bomb made success unreachable (S4)** — `create_ep_factories` called the constructor just to read the EP name, and CUDA's constructor was a panic bomb. | `create_ep_factories_for_shared_ep` takes `ep_name` directly. | Host-tested |
| **4** | **`Free` passed `size=0`, violating the allocator contract.** | `DeviceAllocator` tracks sizes; unknown pointers are no-op'd rather than fabricating `size=0`. | Host-tested |
| **5** | **`CreateEp` unconditionally failed for shared EPs.** | `factory_create_ep` clones the shared `Arc` into a real `ExportedEp`. | Host-tested through a real ORT session |
| **6** | **Governed kernel workspaces were never allocated.** | Executor honours `workspace_requirement` / `execute_with_workspace`. §4. | Host-tested + demonstrated falsifier |
| **7** | **Subgraph intermediates were host buffers on device EPs.** | EP-allocated, RAII-freed, EP-device-tagged; fail-closed when impossible. §5. | Host-tested + demonstrated falsifier |
| **8** | **No explicit shared-EP shutdown on normal teardown.** | `ReleaseEpFactory` owns explicit shutdown; Drop-only is the codified fallback. §7. | Host-tested |

### Gaff's review items

| # | Issue | Disposition |
|---|-------|-------------|
| **S1** | `device_free` fell back to `size=0` for unknown pointers | ✅ Fixed — unknown pointers early-return. |
| **S2** | `Mutex::lock().unwrap()` across `extern "C"` | ✅ Superseded — the shared-EP path has no mutex to poison. |
| **S3** | `factory_create_ep` ignored `shared_ep` | ⚠️ The original "fix" returned an actionable error status, which *was* defect #5. Failing closed on every `CreateEp` is not a fix for an EP that must eventually run inference. Now genuinely fixed. |
| **S4** | CUDA constructor panicked during factory creation | ✅ Fixed — defect #3. |
| **B2** | `CanCopy` same-device used pointer equality | ✅ Code fixed to use `MemoryDevice_GetDeviceId` (ORT 1.27 bindings, `OrtEpApi` offset 96); **compile-verified only**. Cross-device (P2P) fails closed, as does a `None` `GetDeviceId`. |
| **N1** | "mock" wording in production comments | ✅ Fixed. |
| **N2** | `factory_get_vendor_id` always returned 0 | ✅ Fixed — reads `device_support.vendor_id`. |

---

## 3. Ownership Architecture

```
ExportedFactory {
    shared_ep: Option<Arc<dyn ExecutionProvider>>,
    ...
}

// Every surface clones the SAME Arc — no locking, no MutexGuard lifetime:
DeviceAllocator        { ep_ref: EpRef::Shared(Arc::clone(shared)), ... }
DeviceSyncStream       { ep_ref: EpRef::Shared(Arc::clone(shared)), ... }
DeviceDataTransferFull { ep_ref: EpRef::Shared(Arc::clone(shared)), ... }
ExportedEp             { ep: Arc::clone(shared), ... }   // one per session
```

**Why this is sound without a mutex:** `ExecutionProvider: Send + Sync`, and
every method used post-construction (`allocate`, `deallocate`, `copy`,
`copy_async`, `sync`, `get_kernel`, `supports_op`, …) takes `&self`. Only
`initialize`/`shutdown` need `&mut self`. `initialize` runs once before the EP
is wrapped in the `Arc` (the caller's responsibility, documented on
`create_ep_factories_for_shared_ep`). `shutdown` is reached only through
`Arc::get_mut`, i.e. only from the sole remaining owner — see §7 for exactly
who that is and when.

**Why `Arc`, not a raw pointer or a `Mutex`:** a raw pointer from a
`MutexGuard` is valid only while the guard is held (the B1 defect); a
`Mutex<Box<dyn ExecutionProvider>>` cannot yield the `Box` to `CreateEp`
without `Option::take`, which would strand every other holder with a dangling
`EpRef`. `Arc<dyn ExecutionProvider>` is `Clone`, needs no lock for
`&self`-only usage, and keeps the allocation alive for every holder
independently.

---

## 4. Governed Kernel Workspaces (blocker 1)

`crates/onnx-runtime-ep-plugin/src/compute.rs` previously called
`Kernel::execute()` for every dispatch. Kernels that declare a workspace via
`Kernel::workspace_requirement()` — cuBLASLt GEMMs, tree reductions,
FlashAttention scratch — have no way to obtain scratch through that entry point
and fail. The executor now, for **both** the single-node and the routed
subgraph path:

1. builds `TensorMetadata` for the dispatch and calls
   `kernel.workspace_requirement(&meta)`;
2. for a non-zero requirement, allocates `bytes` at `alignment` **through the
   owning EP's allocator** (`EpAllocation`), so device kernels get device
   memory and never a host pointer;
3. verifies the returned pointer actually satisfies `alignment`, failing closed
   with an actionable message if not;
4. calls `kernel.execute_with_workspace(inputs, outputs, Some(view))`, or
   passes `None` when the requirement is zero — no pointless allocation;
5. frees the allocation via `Drop`, so **every** early return — including the
   error paths — releases it. There is no per-call leak.

**Lifetime.** Workspaces are allocated per dispatch rather than cached as
`WorkspaceLifetime::SessionPersistent`. Caching would require locking, because
ORT may `Run()` one session concurrently and a shared workspace would be a data
race. This is a **performance** note, not a correctness gap; revisit alongside
a per-stream arena.

**Alignment.** A zero or non-power-of-two alignment is rejected before any
allocation. A zero-byte request is normalised to one byte so the pointer is
unique, non-null and freeable (§8).

---

## 5. Fused/Routed Subgraph Intermediates (blocker 2)

Multi-node subgraph routing allocated intermediates as host `Vec<u8>` and
tagged every `TensorView` `DeviceId::cpu()`. On a host-accessible EP that is
merely redundant; on CUDA it hands a host pointer to a device kernel.

Now:

- intermediates come from `ScratchBuf`, which allocates through the owning EP
  whenever one is present, and are freed by RAII on every exit path;
- each `IntermediateBuf` records the device its bytes actually live on, and
  `IntermediateBuf::view()` tags the `TensorView` with that `DeviceId` —
  `ep.device_id()` for EP-backed memory, not a blanket `DeviceId::cpu()`;
- zero-fill is applied **only** to host-accessible allocations. Memsetting
  device memory from the host is undefined behaviour;
- **fail-closed rule:** if the EP is not host-accessible and a real EP
  allocation cannot be made, multi-node routing is rejected *before* compile or
  execute with an actionable error naming the EP and the reason. A host pointer
  is never handed to a device kernel.

Per-tensor device tagging for ORT-provided inputs/outputs queries ORT itself
(`Value_GetMemoryDevice` / `MemoryDevice_GetDeviceType`) rather than assuming
`ep.device_id()`, because ORT legitimately keeps some tensors (e.g. shape
inputs) in CPU memory even for a GPU EP. Host-accessible EPs skip the query, so
the CPU path is byte-identical to before.

---

## 6. Conformance

### 6.1 Vtable-level (`onnx-runtime-ep-plugin/tests/shared_gpu_conformance.rs`)

Drives our `extern "C"` callbacks directly, using a real dlopen'd ORT only to
build genuine `OrtValue` / `OrtMemoryDevice` objects. Backed by
`MockCudaLikeEp` — tagged `DeviceType::Cuda`, `stream_aware: true`,
`host_accessible: false`, but holding ordinary host memory.

Covers: `CreateAllocator` alloc/free round-trip including `Alloc(0)`;
a caller-owned opaque stream handle round-tripping unchanged through
`CreateSyncStreamForDevice` → `GetHandle`; `CanCopy` H2D/D2H/H2H/same-device
classification; `CopyTensors` byte-accurate H2D and D2H through real
`OrtValue`s; and `CreateEp` returning the same `Arc` allocation as every other
surface (asserted with `Arc::ptr_eq`, not name equality).

**This proves ABI wiring and ownership. It proves nothing about CUDA.** Calling
our own vtable is also not the same as ORT calling it — which is what §6.2 is
for.

### 6.2 Real ORT end-to-end (`onnx-runtime-ep-shared-mock-plugin/tests/shared_ep_ort_e2e.rs`)

A shared-EP plugin (`crates/onnx-runtime-ep-shared-mock-plugin`,
`publish = false`) built as a real cdylib and loaded by ORT through the
supported path:

```
RegisterExecutionProviderLibrary → GetEpDevices
  → SessionOptionsAppendExecutionProvider_V2 → CreateSession → Run
  → ReleaseSession → UnregisterExecutionProviderLibrary
```

The plugin exposes `WorkspaceAddKernel`, whose `workspace_requirement` demands
a 256-byte-aligned scratch buffer and whose `execute()` — the non-workspace
entry point — **always fails with an explicit message**. If the executor does
not honour the workspace contract, every test that runs a model through this EP
fails. It also exposes a zero-workspace `PlainMulKernel` so the `None` path is
exercised, and `#[no_mangle]` counters for `initialize` / `shutdown` /
`CreateEp` so the E2E tests can assert on real lifecycle ordering.

| Test | Blocker |
|---|---|
| `shared_ep_session_runs_and_workspace_is_plumbed` | 1 |
| `shared_ep_routed_subgraph_intermediates_are_ep_allocated` | 2 |
| `shared_ep_two_sessions_share_one_instance` | 3 |
| `shared_ep_shutdown_runs_once_at_library_unregister` | 4 |

**The mock EP is deliberately CPU-typed.** `factory_get_supported_devices` can
only match hardware ORT actually enumerates; a GPU-typed mock is never selected
on a GPU-less host, so no session could be created and the suite would silently
degrade into a skip. A CPU-typed mock is the only way to get ORT to genuinely
drive this path here. It therefore exercises the *plugin protocol*, not device
memory.

ORT/cdylib discovery is centralised in `crates/onnx-runtime-ort-testkit`
(`publish = false`); the two byte-identical `tests/common/ort_discovery.rs`
copies and `tests/ort_path.rs` were deleted in favour of it.

### 6.3 Demonstrated falsification

Each claim above was checked by breaking the implementation and confirming the
tests go red:

- reverting both `execute_with_workspace` call sites in `compute.rs` to
  `execute()` turns all four §6.2 tests red with
  `WorkspaceAddKernel::execute called directly: … The executor is not honouring
  the workspace contract.`;
- the `compute.rs` unit falsifiers fail if intermediates are tagged
  `DeviceId::cpu()`, if a workspace allocation leaks across dispatches, or if a
  non-host-accessible EP is allowed to reach `execute` with host scratch;
- the `device.rs` falsifier uses an EP whose `allocate()` **rejects** zero-byte
  requests, so it fails if the normalisation moves out of the adapter;
- the `factory.rs` falsifiers fail if `shutdown()` is skipped on normal
  teardown, or called while another surface is still alive.

> **Stale-cdylib hazard.** `cargo test -p <pkg> --test <name>` builds the test
> binary and the rlib but does **not** refresh the crate's `cdylib`, so an
> integration test can silently validate a stale `.so` — this masked the first
> falsification attempt. `onnx_runtime_ort_testkit::find_plugin_cdylib` now
> always runs `cargo build -p <package>` first (memoised per package;
> `NXRT_SKIP_PLUGIN_REBUILD=1` opts out).

### 6.4 Running it

```sh
cargo test -p onnx-runtime-ep-plugin -p onnx-runtime-ep-cpu-plugin \
           -p onnx-runtime-ep-cuda-plugin -p onnx-runtime-ep-shared-mock-plugin \
           -p onnx-runtime-ort-testkit --no-fail-fast
```

Set `NXRT_REQUIRE_ORT_TESTS=1` to turn an "ORT not found" skip into a hard
failure. The two new crates are workspace `members` but not `default-members`,
so they do not affect default builds or publishing.

---

## 7. Shared-EP Shutdown Semantics (blocker 4)

A shared EP is reachable from four ORT-owned surfaces — the `OrtAllocator`, the
`OrtSyncStreamImpl`, the `OrtDataTransferImpl`, and one `OrtEp` per session.
Each holds an `Arc` clone, so **no individual `Release*` callback may call
`shutdown()`**: doing so would tear down a runtime another live session still
needs. `factory_release_ep` accordingly shuts down only when its `ExportedEp`
is the sole owner, which for a shared EP is never true.

`ReleaseEpFactory` is the one point in the ORT lifecycle that happens after
every other surface has been released, so that is where explicit shutdown
belongs. `release_ep_factory_with_teardown` takes the factory's own `Arc`, and:

| Situation | Outcome | Behaviour |
|---|---|---|
| Factory is the last owner (normal teardown) | `ShutdownCalled` | `shutdown()` runs exactly once — the **explicit documented cleanup path** |
| `shutdown()` returned an error | `ShutdownFailed` | Reported on stderr, not swallowed; the EP is still dropped |
| A surface is still alive (ORT contract violation / leaked handle) | `StillReferenced { strong_count }` | Diagnostic printed, **no** `shutdown()`; falls back to the **codified Drop-only invariant** — `Arc` drops the EP when the last surface goes, and the EP's own `Drop` frees device resources |
| Non-shared (owned, e.g. CPU) factory | `NotShared` | Unchanged |

Real CUDA resource teardown is not gated on `shutdown()` in either case: it
happens in `Drop` impls (e.g. `CudaRuntime`), which run exactly once when the
last `Arc` strong reference goes away. Both branches are pinned by unit tests
in `factory.rs` and by `shared_ep_shutdown_runs_once_at_library_unregister`
under real ORT.

> This corrects the previous revision of this document, which asserted that
> `ExportedFactory::shared_ep` "is never taken/cleared". It *is* now taken,
> precisely so that normal teardown has an explicit cleanup path.

---

## 8. Size-Zero Allocation (blocker 5)

`Alloc(0)` is legal for an `OrtAllocator` and must yield a unique, non-null,
freeable pointer. Whether a backing EP allocator honours that is **not**
guaranteed — `cudaMalloc(0)` and third-party CUDA allocators (RMM, PyTorch's
caching allocator) differ.

The normalisation therefore lives at the **adapter boundary**, not in any
allocator: `device.rs` exposes `normalize_alloc_size`, which rewrites only 0 to
`ZERO_SIZE_ALLOC_BYTES`, and `device_alloc` applies it before calling the EP.
`compute.rs` applies the mirrored `MIN_SCRATCH_BYTES` to scratch and workspace
requests. `device_free` no-ops on pointers it does not know rather than
fabricating `size = 0`.

The falsifier is a mock EP whose `allocate()` **returns an error** for zero
bytes: the adapter test still requires a non-null, freeable pointer, so an
alternate CUDA allocator cannot regress this.

---

## 9. Hardware-Only — Not Verified by Anything Here

- whether `cudarc` dynamic loading finds `libcuda.so` / `libcudart.so`;
- whether `CudaRuntime::stream_ptr()` returns a valid `cudaStream_t`;
- whether ORT's `Value_GetMemoryDevice` reports the right device type for
  tensors allocated by *our* allocator on a real GPU;
- whether `copy_from_host` / `copy_to_host` / `copy` are correct on real CUDA
  memory;
- whether `CanCopy`'s `MemoryDevice_GetDeviceId` comparison behaves correctly
  across real devices (B2 is compile-verified only);
- whether a real `CudaExecutionProvider` compiles and runs a real `OrtGraph`
  end to end with real kernels and real device memory;
- whether governed workspaces satisfy real cuBLASLt / FlashAttention alignment
  and size requirements;
- whether the lock-free `Arc` sharing model holds under concurrent ORT sessions
  on a real device.

Tracked by **open** issue
[#768](https://github.com/justinchuby/onnx-genai/issues/768).
`scripts/cuda_conformance_runner.sh` exits 2 (UNVALIDATED) on this host. No
self-hosted GPU workflow exists.

---

## 10. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573`.
Returns `Ok(false)` — "no transfer enqueued". Deferred to post-Phase-2a.
