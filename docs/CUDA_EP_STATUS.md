# CUDA EP Status — Hardware-Validated Core; Workspace/Teardown Deltas Unvalidated

**Authors:** Roy (Lead), Sapper (GPU/Systems), Nabil (FFI/Systems — B1/B3/S4),
Sebastian (Performance — H200 validation in #832, PR #830 revision 3),
Batty (Engine — PR #830 revision 4)
**Updated:** 2026-08-13 (PR #830 revision 4, rebased on `main` @ `8ed44e1cc`)
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
> (`nvidia-smi` is absent, there are no `/dev/nvidia*` nodes, and `cuInit`
> returns `CUDA_ERROR_NO_DEVICE`), so nothing in §4–§7 is evidence about real
> CUDA hardware.

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
| Which CUDA kernels decline, and what each does about it | **Source audit** of `onnx-runtime-ep-cuda` (not a runtime measurement) | §4.1 |
| A workspace plan is computed once per (node, operand signature), not once per dispatch | **Host-tested** (real ORT counter + 10 unit falsifiers, all demonstrated red) | §4, §6.3 |
| Workspace memory info comes from the dispatching node's own ORT **inputs**, resolved only after the zero-byte and `StepScoped` gates | **Host-tested** (compiles + ORT E2E; two placement-query falsifiers); the multi-input agreement check is **compile-verified only** — no host test produces divergent operands | §4 |
| Fused-subgraph *intermediates* use subgraph input 0's memory info | **H200-validated as-is** (#832); **no** device guard claimed | §4 |
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
memory info of the **dispatching node's own ORT-bound operands**
(`operand_mem_info`). ORT owns those bytes for the duration of `Compute`, which
is exactly a step-scoped workspace's lifetime, so this executor never issues a
device free and nothing here is unsafe during graph capture.

**Inputs only.** "The node's operands" means, precisely, the handles ORT returns
from `KernelContext_GetInput` for that node. Outputs are never consulted: ORT
does not require an output to be materialised before `Compute` runs, and the
tensor the kernel is about to write is not evidence about where its compute
happens. So this is an **input-derived** placement, and no claim is made about
output placement anywhere in this PR.

For a node with more than one ORT-bound input, the derivation additionally
verifies the inputs agree, via `OrtApi::CompareMemoryInfo`. Mixed placement is
legal in ORT (`OrtMemTypeCPUInput` operands sit in host memory), so the
executor **fails closed** rather than guessing: a node whose inputs disagree,
or one where `CompareMemoryInfo` is unavailable and the node has several
inputs to compare, gets an error naming the node instead of a workspace from
an arbitrarily chosen device. That error is only reachable when a workspace is
actually requested — nodes that need none are untouched.

**Derived lazily.** A derivation costs up to `2n` ORT FFI calls for `n`
ORT-bound inputs plus `n-1` comparisons — and none at all when every operand is
a fused-subgraph intermediate — and the answer is only ever used to place a
workspace. `prepare_workspace` therefore resolves it **after** the zero-byte and
`StepScoped` gates, so a dispatch that requests **zero bytes** and one whose
`SessionPersistent` request is **declined** resolve no placement at all. The
gates are on the *requirement*, not on the device: a node that genuinely needs a
step-scoped workspace resolves placement wherever it runs, including on CPU,
because that is precisely what tells the executor to serve it from host memory.

Two real-ORT falsifiers pin this from both sides: a declined dispatch must
record **zero** placement resolutions, and a mixed `chain_add_mul` subgraph must
record **exactly** as many as workspaces served — higher means the zero-byte
`Mul` node is paying, lower means a workspace was placed without working out
where its kernel runs. Both go red when the derivation is moved back above the
gates. The counter measures *resolutions*, not FFI calls, since the
intermediates path resolves without calling ORT.

**Status handling at the new call site.** The `CompareMemoryInfo` call releases
the `OrtStatus` it may return, inline, via `OrtApi::ReleaseStatus`. The two
older `GetMemoryInfo`/`GetScratchBuffer` status sites reached from here
(`ort_input_mem_info`, `alloc_scratch`) are carried over verbatim from `main`
and still drop their status without releasing it or reading its message; that is
a **pre-existing leak on a failure path**, is out of scope for this PR, and is
recorded here as follow-up rather than fixed behind a new shared abstraction.

Stated exactly, because this PR does change the site *count*: on `main` the
`GetMemoryInfo` leak was reachable once per dispatch (subgraph input 0); with
per-node derivation it is reachable up to once per ORT-bound input of each
**served** node, so a subgraph with `n` such inputs across its served nodes has
up to `n` reachable sites instead of one. The leak is unchanged in kind and
still confined to a failure path that **aborts the `Run`** — an unreadable
placement is a hard error, not a retry loop — so it cannot accumulate across
steps of a healthy session. Fixing it means releasing and reading the message at
all three sites, which is follow-up scope, not this PR.

> **Scope limit, stated precisely.** This per-node derivation covers the
> *workspace* path only. Fused-subgraph **intermediate** buffers still derive
> their memory info from subgraph input 0 (`device_mem_info` →
> `ort_input_mem_info(.., 0)`). That is deliberate and unchanged: that exact
> allocation is what #832 validated on H200, and it is not re-derived here
> because there is no hardware in this branch's CI to re-validate a change to
> it. For a fused subgraph whose interior nodes are placed on a different
> device from input 0, intermediates would be allocated against input 0's
> memory info. **No device guard is claimed for that path** — it is a known,
> unexercised limitation, not a checked invariant.

**Alignment.** ORT promises no alignment beyond its allocator's, so a stricter
request is met by **checked over-allocation** (`bytes + alignment - 1`) followed
by align-up inside the block. `workspace_block_bytes` and
`align_workspace_window` are separate, unit-tested functions: a non-power-of-two
or zero alignment is rejected before any allocation, `u64 → usize` conversion is
checked, every add is `checked_add`, and the aligned window is re-verified to
lie inside the allocation before it is handed out. A null scratch pointer is
rejected in exactly **one** place — `alloc_scratch`, which returns `Err` for
both a failed status and a null block. `prepare_workspace` used to re-check the
same pointer afterwards; that second check was unreachable and has been removed,
because two guards for one condition invite a future edit that "fixes" one of
them and leaves the other looking authoritative. The same reasoning removed a
second dead guard added earlier in this PR: `device_alloc` null-checked the
pointer out of a successful `ExecutionProvider::allocate`, but
`DeviceBuffer::as_ptr` unwraps a `NonNull<c_void>`, so that branch could not be
taken. The zero-size normalisation it sat next to is unchanged and still
falsifier-tested.

**Workspace-plan memoization.** `Kernel::workspace_requirement` is not free: on
the cuBLASLt-backed kernels it runs `plan_gemm` →
`cublasLtMatmulAlgoGetHeuristic`. Revision 3 of this PR introduced a call to it
on every dispatch, had the answer declined, and left the kernel to plan again
inside `execute` — two searches per dispatch where one result was used.
`prepare_workspace` now routes the call through a per-node
`WorkspacePlanCache`, keyed on the full operand metadata — `(dtype, present,
shape)` per operand, which is exactly and only what `TensorMetadata` shows a
kernel, so a cache hit means the kernel would have been asked an identical
question.

**What that is worth, against the right baseline.** The cache removes
*revision 3's own* second search. It does **not** touch the kernel-side plan:
`blas::governed_gemm` still plans once per dispatch inside `execute`, and this
seam cannot reach into it. Against `main` — which never called
`workspace_requirement` at all — the steady state is therefore approximately
**neutral**, not a halving: a repeated operand signature costs a mutex acquire
plus a linear scan of at most 8 entries, and each *new* signature costs one
heuristic search `main` did not perform. No speedup over `main` is claimed here,
and none has been measured.

**Where hits actually occur.** Hit rate is a property of the shapes, not of the
kernel. A stable geometry (fixed batch and sequence length, or a GEMM whose
cuBLASLt signature does not move between decode steps) hits after its first
dispatch. A growing-KV `StepScoped` attention, whose operand shapes change every
token, can **miss on every step**; there the cache is bounded overhead — one
extra search per step — rather than a saving. Both cases are correct; only the
first is faster.

Properties, each pinned by a test that has been shown to go red without it:

- Repeated `Run`s of an unchanged shape plan **once**
  (`workspace_plans_do_not_repeat_for_an_unchanged_shape`, real ORT: 12 `Run`s,
  1 plan; without the cache, 12).
- Any change of shape, dtype, presence or operand count **re-plans** and is
  never served a stale entry (`a_changed_shape_gets_its_own_workspace_plan` on
  a dynamic-batch model; four unit falsifiers).
- Concurrent dispatches never read each other's plans, and the lock is never
  held across the planning call, so a slow plan cannot serialize other nodes.
- A planning **error** is propagated and never cached.
- A poisoned lock is recovered (`PoisonError::into_inner`), matching
  `EpHandle::with` and factory teardown, rather than turning one panic into a
  permanently dead cache.
- Capacity is bounded (8 signatures/node, move-to-front). Overflow is a **miss**
  — never a wrong answer — and the hot signature survives a flood of one-off
  shapes.

Honest residual: **every** dispatch of a served kernel still plans once inside
the kernel, and each distinct signature plans one extra time here. Removing the
kernel-side plan is not possible from this seam — `plan_gemm` returns the
algorithm and matrix layouts, not just a byte count, and the kernel re-validates
the supplied size independently. So the cache bounds this seam's cost at one
search per signature per node; it does not make the plugin path faster than
`main`.

**Lifetime.** `StepScoped` is served as above. `SessionPersistent` is
**declined** (`None`) — never downgraded. Serving it from scratch would hand the
kernel a block ORT recycles when `Compute` returns, which a persistent consumer
would then reuse on the next `Run`. Alignment is still validated before the
decline, so a malformed requirement is an error rather than a silent `None`.
Note the evaluation order: a requirement of **zero bytes returns `Ok(None)`
before the lifetime is consulted at all**, which is why the declared lifetime of
a kernel that asks for nothing never matters.

Declining is **exactly behaviour-preserving against `main`**, which is why it is
safe to ship without hardware: `main`'s executor calls bare `execute()` at both
dispatch sites, and the `Kernel::execute_with_workspace` default forwards to
`execute()`, so passing `None` reproduces `main` node for node.

### 4.1 What declining actually means, per kernel

The previous revisions of this document carried a five-row table that was wrong
in both directions. The corrected audit — read out of the kernel sources, not
inferred — is below. The distinction that matters is that **most of the
hard-error paths are not unconditional**: they depend on what cuBLASLt's
heuristic picks for that shape.

**(a) Self-owned fallback — declining is harmless.**

| Kernel | Condition | Fallback |
|---|---|---|
| `GroupQueryAttention` (`group_query_attention.rs:3026`) | `SessionPersistent` whenever its composite layout totals > 0 | Pooled `GqaWorkspace` slots (scores, packed Q/K/V, BNSH staging) inside `run` |
| `StandardAttention` (`standard_attention.rs:864`) | `SessionPersistent` **only** for single-token, single-batch decode (`batch == 1 && q_seq == 1`); otherwise `StepScoped` | Pooled or per-call self-owned score/staging scratch inside `run` |

`StandardAttention` is therefore *both* a decliner and a served consumer,
depending on geometry: **prefill and batched decode are served a real
step-scoped workspace by this executor.**

**(b) Heuristic-dependent hard error — only when cuBLASLt asks for bytes.**

These kernels declare their requirement through
`blas::governed_workspace_requirement(bytes)` (`blas.rs:305`), where `bytes`
comes from `plan_gemm`'s `cublasLtMatmulAlgoGetHeuristic` result:

- `bytes == 0` → `WorkspaceRequirement::NONE`. `prepare_workspace` returns
  `Ok(None)` at the zero-byte gate, `governed_workspace_ptr` returns `Ok(0)`,
  and the kernel runs normally. **Nothing is lost.**
- `bytes > 0` → `SessionPersistent`, which this executor declines, and
  `governed_workspace_ptr` (`blas.rs:430`) then returns
  `"cuda_ep {op}: governed cuBLASLt workspace requires {n} bytes, but none was
  supplied"`.

| Kernel | Site |
|---|---|
| `MatMul` | `matmul.rs:649` |
| `Gemm` (f32 path) | `gemm.rs:411` |
| `FusedEpilogue` | `fused_gemm.rs:321` |
| `MatMulNBits` f32 dequant-cuBLASLt path | `matmul_nbits.rs:6390` |

Which branch a given node takes depends on shape, dtype, device and cuBLASLt
version — it is a property of the heuristic, not of this executor. Many
GEMM shapes, decode-shaped ones especially, select **0 bytes** and are entirely
unaffected. `WORKSPACE_BYTES` (32 MiB) is a *ceiling* handed to `MatmulPref`,
not an amount anyone requires. No claim is made here about which fraction of
real models lands on which branch, because that has not been measured on
hardware.

**Correction to the previous table:** it listed `MatMulNBits`' **bf16 staging**
as a self-owned decline fallback. That is wrong. `uses_dequant_cublas_workspace`
requires `DataType::Float32`, so `MatMulNBits` returns `NONE` for BFloat16 and
never requests a governed workspace on that path at all. Its `Bf16Scratch` arena
is *always* self-owned, in every revision, and the decline neither helps nor
harms it.

**(c) Unconditional hard error — every geometry.**

| Kernel | Site |
|---|---|
| `BlockQuantizedMoE` | `block_quantized_moe.rs:771` |
| `IndexShare` | `index_share.rs:720` |

Both declare `SessionPersistent` for all shapes, and both `execute()` and
`execute_with_workspace(None)` return an error.

**(d) True `StepScoped` consumers — served by this executor.**

| Kernel | Site |
|---|---|
| default-domain `Attention` Phase-2a scratch | `attention.rs:973` |
| `StandardAttention` prefill / batched decode | `standard_attention.rs:864` |

These are the only paths where this change does something observable on
hardware, and they are what H200 validation must exercise. **GQA decode is not
one of them** — it declines, so measuring it would validate nothing.

Everything in (b) with `bytes > 0`, and everything in (c), **already fails this
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
| A surface is still alive (ORT contract violation / leaked handle) | `StillReferenced { strong_count, weak_count }` | Diagnostic printed, **no** `shutdown()`; falls back to the codified Drop-only invariant |
| Non-shared (owned, e.g. CPU) factory | `NotShared` | Unchanged |

**Why `weak_count` is reported.** `Arc::get_mut` refuses for *either* of two
reasons: another strong owner, or any outstanding `Weak`. The diagnostic
previously printed `strong_count - 1` as "other reference(s) still alive", so in
the weak-only case — factory holds the sole strong reference, someone holds a
`Weak` — it printed "0 other reference(s) are still alive" while simultaneously
skipping shutdown, which reads as a contradiction and points an investigator at
the wrong thing. The outcome now carries both counts and the message names
whichever condition actually applies. No `Weak` to the shared EP exists in this
tree today (grep confirms), so this is a correctness-of-diagnostic fix, not a
live bug; it is pinned by
`releasing_the_factory_reports_weak_handles_as_the_blocker`, which builds the
weak-only case explicitly.

**And when the counts race.** `Arc::get_mut` and the two count reads are three
separate atomic operations, so a concurrent owner can release *between* them and
leave the diagnostic reading `strong=1, weak=0` — i.e. claiming that nothing
blocked an access that was nevertheless refused. Exclusive access is now retried
once in that exact case: if the blocker really did disappear, the retry succeeds
and `shutdown()` runs (which is legal, since `get_mut` succeeding *is* the proof
of exclusivity). If it is refused a second time with the counts still reading
exclusive, the message says so — references are being manipulated concurrently
with `ReleaseEpFactory` — instead of naming a blocker the counts do not
support.

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
| `workspace_plans_do_not_repeat_for_an_unchanged_shape` | 12 `Run`s of one shape ⇒ the plan counter never moves past what Run 1 needed |
| `a_changed_shape_gets_its_own_workspace_plan` | on a dynamic-batch model, alternating `[1,4]`/`[3,4]` plans once per *distinct* shape and never serves a stale plan — the kernel rejects an undersized workspace, so a stale plan is a `Run` failure, not a silent pass |
| `a_declined_workspace_never_asks_ort_where_the_operands_live` | a declined `SessionPersistent` dispatch records **zero** placement resolutions, with `persistent_declined >= RUNS` guarding against a vacuous pass |
| `only_the_nodes_that_receive_a_workspace_query_placement` | on `chain_add_mul`, placement resolutions **equal** workspaces served — the zero-workspace `Mul` never resolves, and every served `Add` does |
| `shared_ep_two_sessions_share_one_instance` | one EP instance across two sessions |
| `shared_ep_shutdown_runs_once_at_library_unregister` | exactly one `shutdown()` at unregister |

The mock `WorkspaceAddKernel` counts its own `workspace_requirement` calls
(`nxrt_mock_shared_ep_workspace_plans`), standing in for the cuBLASLt heuristic
search a real GEMM kernel runs there. It is the only way to observe the plan
cache through a real ORT `Run` without hardware.

`nxrt_mock_shared_ep_placement_queries` is different in kind: it re-exports the
*executor's* own counter (`compute::workspace_placement_queries`), not a
mock-side one, because the property under test is what the executor does before
it decides to serve. That counter is deliberately not `cfg(test)`-gated — the
cdylib ORT loads is built without `cfg(test)`, so a gated counter would leave
the claim tested only in a configuration nobody ships.

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

Falsifications demonstrated for this revision, with the exact observed output:

| Mutation | Result |
|---|---|
| Drop `shape` from `OperandKey::matches` | 4 unit tests red (*"the larger geometry must get its own plan, not the smaller one's — left: 128, right: 256"*) **and** `a_changed_shape_gets_its_own_workspace_plan` red through real ORT with *"WorkspaceAddKernel: workspace too small — need 48 bytes, got 16"* |
| Drop `dtype` + `present` from the key | *"an f16 dispatch must not be served the f32 plan — left: 256, right: 128"*; *"an absent optional operand must not be served the plan that charged for it — left: 128, right: 64"* |
| Delete the `WorkspacePlanCache::lookup` fast path | 5 unit tests red (*"16 dispatches of one unchanged shape must run the planner once, not 17"*) and both E2E plan tests red: *"12 Runs of one unchanged shape re-planned the workspace: 12 plans where the first Run already needed 1"* — i.e. exactly linear growth |
| Restore the pre-fix `weak_count = strong_count - 1` | both teardown tests red; the diagnostic prints the self-contradicting *"…but 0 Weak handle(s) … are outstanding, which is also enough to block exclusive access"* |
| Move `operand_mem_info` back above the zero-byte / `StepScoped` gates | both placement falsifiers red: *"the executor resolved where the operands live for a node whose workspace request it then declined … 4 wasted placement resolutions per dispatch"* and *"placement was resolved 3 times for 2 served workspaces"* |

Each mutation was reverted and the suite re-run green before committing.

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
- whether declining `SessionPersistent` leaves `GroupQueryAttention` and
  `StandardAttention`'s single-token decode geometry on a correct self-owned
  path under the plugin executor specifically (#832 validated the *native* path
  for those kernels, and the plugin path for `Add`/`Mul` and the 30B decoder,
  but not a plugin-path model that exercises `GroupQueryAttention`);
- which real model shapes drive `MatMul`/`Gemm`/`FusedEpilogue`/`MatMulNBits`
  into cuBLASLt's `bytes > 0` branch, where declining is a hard error rather
  than a no-op (§4.1(b)). This is a property of the heuristic and can only be
  measured on device. `BlockQuantizedMoE` and `IndexShare` are known *not* to
  run on the plugin path either here or on `main` — that is a stated gap, not
  an open question;
- whether the per-node `operand_mem_info` derivation agrees with the
  subgraph-level derivation on real fused CUDA subgraphs (they are expected to,
  since every node in a fused subgraph is placed on the same EP; the
  `CompareMemoryInfo` guard exists so that a case where they do *not* fails
  loudly instead of silently allocating on the wrong device). Fused-subgraph
  **intermediates** deliberately keep the #832-validated subgraph-level
  derivation and carry **no** device guard — see §4;
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
```

then run a model that exercises a **true `StepScoped` consumer** — per §4.1(d),
either default-domain `Attention` Phase-2a, or `StandardAttention` on a
**prefill / batched** geometry (`batch > 1` or `q_seq > 1`). Assert numerics
against the native CUDA EP, and capture

```sh
nsys profile --trace=cuda,nvtx -o ws_check <runner>
nsys stats --report cuda_api_sum ws_check.nsys-rep | grep -E 'cuMemAlloc|cuMemFree'
```

expecting **no** per-node `cuMemAlloc`/`cuMemFree` inside the steady-state
decode region.

> **Do not use GQA decode as the workspace evidence.** `GroupQueryAttention`
> *declines* (§4.1(a)), so it exercises the self-owned fallback and would prove
> nothing about a served workspace. Likewise `StandardAttention` at
> `batch == 1 && q_seq == 1` declines; only its prefill/batched geometry is
> served.

That was **not run for this revision** — the development environment has no
NVIDIA device (`nvidia-smi` absent, no `/dev/nvidia*`, `cuInit` →
`CUDA_ERROR_NO_DEVICE`). PR #830 stays **Draft** until it is.

---

## 8. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs`. Returns
`Ok(false)` — "no transfer enqueued". Deferred to post-Phase-2a.
