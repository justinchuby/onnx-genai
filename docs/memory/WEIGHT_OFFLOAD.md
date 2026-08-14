# Weight Offload and Paging for Huge MoE Models

> **Status:** Design approved 2026-07-14 (owner @justinchuby); Phase 1 partially
> implemented. **§0 records what is measured to work today and what does not** —
> read it before relying on anything in this document. Open questions resolved;
> see §11.
>
> **Primary targets:** GLM-5.2 and DeepSeek-V4-Flash-class sparse MoE models.
>
> **Reference UX:** llama.cpp/Unsloth-style memory mapping and partial GPU offload:
> the same model runs under a small memory budget, while larger machines
> automatically keep more weights resident.
>
> **Date:** 2026-07-14

## 0. North star

**One model package runs on any machine that can physically hold one activation
set, and gets faster as the machine gets bigger — without a different build, a
different file, or a different code path.**

That is the whole objective. Everything else in this document is a means to it,
and any part of it can be replaced by something that serves it better.

### 0.1 What that commits us to

Four properties, in priority order. When they conflict, the earlier one wins.

1. **Capacity degrades into latency, never into failure.** A model that does not
   fit should get slower, not refuse to load. The floor is one activation set
   plus one bounded weight tile — not the whole model, not a whole expert.
2. **The user states one number: how much of their device they are willing to
   give up.** Everything else is derived. A knob that must be discovered,
   enabled, and then sized again elsewhere is a design failure even when each
   piece works.
3. **We must beat the operating system.** Every platform already has a way to
   overflow device memory — WDDM shared memory, UVM, swap. Ours is only worth
   having if it is *faster*, and it should be, because we know the access
   schedule in advance and the OS does not. **If we are slower than doing
   nothing, the honest recommendation is to do nothing.**
4. **One authority accounts for every byte.** Any path that allocates without
   the ledger seeing it is a second set of books, and a system with two sets of
   books cannot make an admission decision.

### 0.2 What would falsify this design

Stated up front so it stays testable rather than becoming a slogan:

- A model that fits in host RAM but cannot be made to load at all.
- Our managed offload measured slower than the platform's own overflow mechanism
  on the same model and machine (property 3 — this is a **quantitative** bar, and
  it is currently failing; see §0.4).
- A user needing to know more than one memory number to run a model that fits.
- Any allocation the governor cannot see.

### 0.3 What is deliberately out of scope

- Making a model fit that cannot fit — there is a physical floor and this design
  does not pretend otherwise.
- Beating a machine that has enough memory. When everything is resident, the
  right behaviour is to get out of the way entirely: allocate once, use stable
  pointers, and add no per-token cost.
- Changing numerics to save memory. Quantization is chosen by the model package,
  not by the residency system.

### 0.4 Current state

**This subsection is a snapshot and is expected to rot. The rest of §0 is not.**
It is kept deliberately short and links out rather than restating numbers, so
that stale figures live in issue threads where they can be superseded, not in
the design.

| north-star property | status |
|---|---|
| 1. degrades into latency | **holds** — a model exceeding the card loads and answers correctly |
| 2. one number | **now largely holds under #798/#755** — managed no-spill VMM is the default on native CUDA, `--vram-limit` **overrides** the inferred device budget rather than gating the managed path, and weight streaming auto-enables when weights exceed the budget (no separate offload opt-in). The legacy two-opt-in path ([#712]) remains only behind `ONNX_GENAI_LEGACY_ALLOCATOR=1` |
| 3. beats the OS | **fails, and the cause is now structural rather than unknown** — measured directly on `qwen14b-zp` (byte-identical output, solo, `nvidia-smi` verified empty before each run): **WDDM 8.09 tok/s with `htod_bytes_per_token = 0`** against **managed streaming 0.11**, so #874 made the OS path the Windows default and stopped auto-enabling managed streaming there. The reason is not a tuning gap: each weight is read **exactly once per decode step**, so both paths move the same bytes over the same link while ours adds a CPU memcpy into pinned staging, a VRAM allocation, a `cuMemMap`, an eviction and a synchronize — buying residency that is discarded before it is ever re-read. **Copying to VRAM only pays when the data is re-read from VRAM before eviction, and there is no intra-step reuse.** The presumed route to actually beating the OS was the #864 hybrid (a resident hot set, chosen knowing the layer walk, plus **zero-copy** cold reads). **It was built end-to-end and it failed on measurement (#912).** Its two *blocking* risks were indeed measured away first — a real strided GEMV reads host-mapped weights at ~5.6 GB/s against ~100–133 GB/s resident with bit-identical outputs, and CUDA graph capture with a host-mapped pointer is supported and replays bit-identically (#880) — so what killed it was neither: **aggregate distinct host-mapped reads above ~0.44–0.65 GB/step silently return stale data** (48 cold weights collapsed generation 16 → 3 tokens; a single read is bit-identical at 1/8/16/32, and a copy-instead A/B pins the fault on the read, so it is an aperture ceiling), and capped at a provably safe 256 MiB budget the arm ran **0.73 tok/s against WDDM's 7.84**. The safe aperture (0.26 GB/step) is structurally smaller than the traffic it must displace (~0.6 GB/step) — too short **by construction**, not by tuning. It ships default-OFF behind `ONNX_GENAI_ZERO_COPY_HYBRID` as instrumentation for parts with larger host apertures, which must be re-measured rather than assumed. **So this row's honest status is: on WDDM consumer hardware we do not beat the OS, we correctly decline to compete with it, and no route currently on the table changes that** — the remaining levers are batching (which creates the intra-step reuse that makes residency pay at all, #884/#891) and the admission side of the residency gap (~90% of it, #901). **That verdict is WDDM-scoped and does not transfer to Linux, where this row's whole framing changes and the answer is now measured (#925/#936):** there is no shared-memory fallback, so an over-budget model that does not stream fails outright — managed streaming competes with "does not run", and the #864 hybrid competes with managed streaming rather than with a 7.84 tok/s OS path. **On Linux the hybrid wins by ~8×: 67 tok/s against ~8.5 median for managed streaming.** The aperture ceiling that closed the hybrid on Windows is a VidMm behaviour, as #863 suggested (it measured VidMm demoting our own VMM granules; a host registration being silently remapped is the same family) — measured absent on H200 (driver 580.105.08, CUDA 13, kernel 6.6), byte-identical with `fallbacks=0` up to **6.795 GB** of distinct host-mapped weights per step (704 binds, n=3), ~15× the WDDM onset. The default budget is now platform-conditional: 256 MiB on Windows, 2 GiB elsewhere — bounded rather than unbounded, >3× under the measured-safe figure because only one GPU class was tested. #783's lesson applied in both directions: the negative was not inherited onto Linux, and the positive is not extrapolated past its hardware. The code keeps the platforms apart throughout: `shared_memory_weight_fallback` is `cfg!(windows)`, Linux still auto-enables managed streaming, unit-locked by `managed_default_no_flag_over_budget_model_auto_streams`. ([#705], [#783], [#864], [#874], [#877], [#880], [#912]) |
| 4. one authority | **holds on the paths that are wired, but only as accounting** — the platform can allocate behind our back on Windows ([#704]), and #863 measured that it also **pages out our own VMM granules**: a `cuMemCreate`+`cuMemMap`+`cuMemSetAccess` allocator mirroring the engine arena committed *and touched* 9,984 MiB on an 8,188 MiB card, with device-resident capped at ~7,942 MiB while the host working set reached 9.49 GB. So `peak_committed_physical_bytes < managed_limit_bytes` and `oversubscribed_bytes == 0` describe **our ledger, not physical residency**. Refined: solo and under `managed_limit`, no-spill *does* hold physically (`nvidia-smi` tracks us 1:1); spill is specifically the **system-wide over-commit** case, invisible to our counters because our own committed bytes do not change. Scoped to WDDM — under TCC `cuMemCreate` should fail at the physical limit rather than spill ([#704], [#863]) |

Two structural notes that are not merely "not done yet":

- **The memory paths do not compose.** Some combinations of the arena, offload
  and the baseline fail to load ([#704]). Until that is fixed, any A/B that
  toggles one of them may be comparing against a broken configuration.
- **The platform is part of the system.** On Windows WDDM, `cudaMalloc` spills
  past dedicated VRAM into system RAM and `cuMemGetInfo` cannot see it. Device
  capacity is therefore not a number we can read, and floor comparisons taken on
  such a machine do not mean what they appear to ([#704]).
- **Weight offload and CUDA graph capture were mutually exclusive; #796 removed
  that.** The pager's alloc/copy/free operations are capture-illegal, so under
  the legacy pointer-returning pager, enabling offload disabled capture — and
  large models that need offload forfeited **all** of the capture-fragmentation
  wins #708/#728/#757 landed (35B-A3B decode from **154 to 34** graph segments).
  #796 fixes it by paging weights under a **stable virtual address**: a page-in
  remaps physical granules at the same VA (survivable per #727) instead of
  returning a new pointer, so the captured graph is preserved. Capture-under-
  offload is now gated on `weight_offload_enabled && !weight_offload_stable_va`,
  i.e. allowed on the stable-VA path and pinned by the #796 unit tests. The
  legacy pointer-returning pager still implies capture-off; the mutual exclusion
  is a property of that path, not of offload as such.
- **The residency cache had a 0% hit rate for its entire life** — `evict_to_fit`
  was a plain LRU evicting against a cyclic sequential layer scan, so it always
  evicted the page needed soonest (0 hits across 6,936 page-ins, at both 3 GB and
  6 GB budgets). Fixed by a scan-resistant stable-subset policy in **PR #723**
  (74.18% hit rate; evictions 6,286 → 0). This is the named mechanism behind the
  "beats the OS" failure. See [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md).
- **Public budgets are committed physical bytes, not nominal content bytes.** A
  nominal 6 GB content budget was measured to physically consume ~**6.51 GB**
  (~8.5% hidden), because reservation is committed at the 2 MiB granule. Size and
  report budgets in committed bytes.

- **Managed no-spill VMM is now the default (#798), not an opt-in.** On native
  CUDA the authority-governed VMM path is selected without a flag; a model that
  fits stays `FullResident` and does **not** page (measured: offload off, **0
  page-ins**), and weight streaming auto-engages only when the resolved budget
  cannot hold the weights. The `ONNX_GENAI_WEIGHT_OFFLOAD=1` opt-in and the
  phased rollout in §8/§10 below describe the **legacy** route-first pager, which
  the managed default supersedes for the fitting and auto-streaming cases; treat
  those sections as the pager mechanism, not the current default policy. See
  [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md).

What is confirmed working, and worth not re-litigating: **prefix reuse**, and
**multi-request concurrency** up to the point where admission becomes the limit.
See `MEMORY_ARCHITECTURE.md` for the per-component status table.

[#704]: https://github.com/justinchuby/onnx-genai/issues/704
[#705]: https://github.com/justinchuby/onnx-genai/issues/705
[#712]: https://github.com/justinchuby/onnx-genai/issues/712
[#783]: https://github.com/justinchuby/onnx-genai/issues/783
[#863]: https://github.com/justinchuby/onnx-genai/issues/863
[#864]: https://github.com/justinchuby/onnx-genai/issues/864
[#874]: https://github.com/justinchuby/onnx-genai/pull/874
[#877]: https://github.com/justinchuby/onnx-genai/pull/877
[#880]: https://github.com/justinchuby/onnx-genai/pull/880
[#912]: https://github.com/justinchuby/onnx-genai/pull/912

## 1. Executive recommendation

Treat immutable model weights as a three-tier hierarchy:

```text
read-only external-data mmap        bounded host cache              bounded device cache
Disk / filesystem backing   <---->  RAM (pageable or pinned)  <----> VRAM
        cold                              warm                         hot
```

The runtime should keep one canonical compressed representation on disk, identify
independently addressable weight regions, and lease only the regions needed by the
next operation. Dense/shared tensors receive the highest residency priority. MoE
expert tensors are expert-major, divided into bounded transfer pages, and admitted
by observed routing heat. A lease pins its pages until the CPU kernel returns or the
device completion fence signals.

The recommended implementation is a **parallel, weight-specific residency system**,
not storage of experts in `onnx-genai-kv`. It should reuse or extract the KV
subsystem's generic ideas—tier identities, byte budgets, LRU/priority admission,
promotion, prefetch hints, and pin/lease state—but not its token/sequence keys,
mutable KV payloads, copy-on-write rules, or tensor geometry. Existing design already
states that expert weights are immutable model data, not KV, and calls for a separate
weight API while reusing KV concepts
([DESIGN.md lines 9195-9213](../architecture/DESIGN.md#L9195-L9213)).

Keep the loader's existing `WeightStore` as the immutable backing catalog. Add a
weight-residency layer beside it, preferably a dedicated runtime crate/module rather
than expanding `onnx-runtime-memory`: that crate is deliberately a pure,
EP-independent activation-liveness planner
([lib.rs lines 1-23](../../crates/onnx-runtime-memory/src/lib.rs#L1-L23)). The residency
layer serves a narrow `ExpertStore` facade to fused MoE kernels and a more general
`WeightResidencyManager` to layer placement.

Two fast paths are mandatory:

1. **Fully resident:** if the planned weights fit, allocate/upload once, use stable
   pointers, and avoid eviction or per-token copies.
2. **Paged:** if they do not fit, the same kernels consume bounded leases from mmap,
   host RAM, or VRAM. Capacity degrades into latency rather than model-load failure.

“Any-size machine” still has a physical floor: enough storage for the package,
address space for its mappings, and memory for one activation/scratch set plus one
bounded weight tile. The design removes the requirement that RAM or VRAM hold the
whole model or even a whole active expert.

## 2. Problem framing

### 2.1 Total bytes, active bytes, and resident bytes are different

A dense model reads nearly every parameter for every token, so disk-backed execution
is possible but quickly becomes storage-bandwidth-bound. Sparse MoE changes the
capacity equation:

```text
total model bytes = shared dense bytes + all expert bytes
active token bytes = shared dense bytes + top-k experts per MoE layer
resident bytes     = policy choice constrained by RAM/VRAM budgets
```

GLM-5.2 is the motivating extreme: the quantization design records 744B total
parameters but about 40B active per token, and cites community packages around
223–245 GB for dynamic 1–2-bit variants rather than roughly 1.5 TB uncompressed
([SUB4BIT_QUANT.md lines 8-16](../quantization/SUB4BIT_QUANT.md#L8-L16)). DeepSeek-V4-Flash is the
second target because it has the same useful systems property: many expert parameters,
but sparse expert activation.

Sparse activation makes offload tractable only if the graph preserves the MoE unit.
A decomposed graph exposes every expert as ordinary initializers; a fused MoE op can
compute routes first, union selected expert IDs across the batch, acquire only those
weight slices, and release them after compute
([SUB4BIT_QUANT.md lines 281-302](../quantization/SUB4BIT_QUANT.md#L281-L302)).

### 2.2 Two operating regimes

#### Tiny machine: storage-backed execution

- External weights remain in read-only mmap files.
- CPU kernels read compressed blocks directly from mapped expert pages.
- The explicit host cache may be small or zero; clean mmap pages remain reclaimable by
  the OS.
- Shared weights are assigned highest priority but may still be streamed when the RAM
  budget cannot hold them.
- GPU use is optional. If present, one or a few bounded tiles are staged to VRAM.
- Prefill may process active experts in waves so the union of a large batch never has
  to fit simultaneously.

This mode optimizes for bounded owned memory and OS-reclaimable mapped residency, not
high tokens/second.

#### Big machine: resident execution with an offloaded tail

- Shared weights and the hottest/fullest useful layer set are resident.
- If the complete compressed model fits VRAM, paging is disabled after startup.
- Otherwise the planner keeps whole layers and/or hot experts on the GPU, retains the
  remainder in host RAM, and uses disk only as immutable backing.
- Transfers use pinned staging and asynchronous H2D prefetch when profitable.
- On an H200-class system, the default should maximize stable VRAM residency, preserve
  headroom for KV/activations/scratch, and page only the expert tail.

The same package and operator semantics serve both regimes. Placement changes
latency, never routing, quantization format, or numerical policy.

## 3. Current codebase foundations and gaps

### 3.1 Disk foundation: external-data mmap already works

The loader records every external initializer as `(path, offset, length, dtype, dims)`,
maps each backing file read-only, validates each range, and returns borrowed slices
from the live mapping
([weights.rs lines 19-83](../../crates/onnx-runtime-loader/src/weights.rs#L19-L83),
[lines 113-167](../../crates/onnx-runtime-loader/src/weights.rs#L113-L167)). `WeightRef::External`
already carries the range and shape needed to derive expert subranges
([tensor.rs lines 72-106](../../crates/onnx-runtime-ir/src/tensor.rs#L72-L106)).

For a host-accessible EP, executor construction aliases aligned initializer bytes
with a borrowed `DeviceBuffer` rather than allocating and copying them. The comments
explicitly rely on OS demand paging so weights may exceed RAM
([executor.rs lines 691-733](../../crates/onnx-runtime-session/src/executor.rs#L691-L733)).
The CPU EP correctly treats borrowed buffers as foreign mmap memory and does not free
them
([provider.rs lines 176-200](../../crates/onnx-runtime-ep-cpu/src/provider.rs#L176-L200)).
The EP API defines the same borrowed-buffer ownership and read-only lifetime contract
([provider.rs lines 68-87](../../crates/onnx-runtime-ep-api/src/provider.rs#L68-L87),
[lines 121-145](../../crates/onnx-runtime-ep-api/src/provider.rs#L121-L145)).

This is the cold-tier foundation. Mapping a file does not mean all bytes are resident;
file-backed pages enter RAM on demand and can be reclaimed cleanly.

### 3.2 The current executor is not a device offloader

The executor still creates one initializer binding for every graph initializer during
build. Host EPs can borrow mmap, but a non-host EP takes the allocate-and-copy branch,
which uploads every initializer eagerly
([executor.rs lines 698-743](../../crates/onnx-runtime-session/src/executor.rs#L698-L743)).
It also owns one EP for the whole plan, not per-node CPU/GPU placement
([executor.rs lines 220-236](../../crates/onnx-runtime-session/src/executor.rs#L220-L236),
[lines 677-681](../../crates/onnx-runtime-session/src/executor.rs#L677-L681)). Therefore
partial GPU offload needs both lazy initializer binding and multi-EP/layer placement;
it cannot be implemented as an allocator tweak.

The transfer API has the right shape but is incomplete for overlap. `ExecutionProvider`
offers `copy_async` and `Fence`
([provider.rs lines 237-241](../../crates/onnx-runtime-ep-api/src/provider.rs#L237-L241),
[lines 258-295](../../crates/onnx-runtime-ep-api/src/provider.rs#L258-L295)), while the
CUDA implementation currently performs a synchronous copy and returns an already
signalled placeholder fence
([provider.rs lines 220-225](../../crates/onnx-runtime-ep-cuda/src/provider.rs#L220-L225)).
True prefetch requires stream-ordered host-to-device copies and awaitable completion.

> **Update (issue #87, Phase-4 mechanism landed).** The placeholder fence is gone.
> The CUDA EP now issues a real stream-ordered async H2D copy on a dedicated transfer
> stream with pinned host staging and records a genuine CUDA completion event; the
> generic trait gained `wait_fence`, which makes the compute stream wait on that event
> (a non-host-blocking `cuStreamWaitEvent`), plus `record_compute_fence` / `copy_wait_fence`
> for the write-after-read (WAR) direction. RAW ordering (compute waits for the transfer)
> is GPU-tested in `onnx-runtime-ep-cuda`. WAR safety for double-buffer reuse is **enforced
> by the shipped `drive_double_buffer` driver itself** — before a reuse copy overwrites a
> slot it makes the transfer stream wait on the prior consumer's compute event — and is
> GPU-tested through the public driver path by
> `drive_double_buffer_war_safe_across_waves` (session `cuda` feature), which corrupts if
> that fence is removed. The executor-side double-buffering *schedule* ships as a
> standalone, unit-tested strategy
> (`onnx_runtime_session::plan_double_buffer` / `drive_double_buffer`); wiring it into
> the live MoE decode loop depends on Phase-3b live device weight binding (follow-up).

### 3.3 Existing MoE representation is suitable for slicing, not yet paging

The CPU `com.microsoft::MoE` kernel accepts ORT's expert-major canonical tensors and
validates shapes whose first dimension is `experts`
([moe.rs lines 1-6](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L1-L6),
[lines 155-180](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L155-L180)). Its
per-row execution already indexes contiguous per-expert FC1/FC2 slices
([moe.rs lines 231-317](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L231-L317)).
That layout is the correct storage boundary.

The current correctness kernel nevertheless materializes the complete FC1 and FC2
inputs before routing
([moe.rs lines 225-229](../../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L225-L229)).
Likewise, `BlockQuantizedMatMul` dequantizes a constant packed matrix into a full f32
`OnceLock<Vec<f32>>`
([block_quantized_matmul.rs lines 77-83](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L77-L83),
[lines 151-166](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L151-L166),
[lines 189-212](../../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L189-L212)).
Those are valid correctness baselines but must not be used for huge offloaded experts.
The paging fast path needs fused MoE and compressed-domain kernels.

`MOE_SUPPORT.md` already requires expert-major contiguous external data so each expert
slice can be computed from the initializer descriptor without materializing the tensor
([MOE_SUPPORT.md lines 289-302](../quantization/MOE_SUPPORT.md#L289-L302)). This document adopts that
contract.

### 3.4 KV tiering provides concepts, not a reusable weight implementation

The KV crate names GPU, CPU, and disk tiers
([lib.rs lines 53-69](../../crates/onnx-genai-kv/src/lib.rs#L53-L69)) and its page table
tracks page identity, refcount, device, last access, and LRU promotion/demotion
([page_table.rs lines 309-340](../../crates/onnx-genai-kv/src/page_table.rs#L309-L340),
[lines 870-925](../../crates/onnx-genai-kv/src/page_table.rs#L870-L925)). The paged cache
can promote a requested logical range
([paged_cache.rs lines 509-553](../../crates/onnx-genai-kv/src/paged_cache.rs#L509-L553)).
The connector API also contains useful lookup/fetch/prefetch/pin/evict vocabulary
([connector.rs lines 409-458](../../crates/onnx-genai-kv/src/connector.rs#L409-L458)).

Direct reuse is unsafe and misleading:

- `PageTable` is sequence/token-oriented and stores mutable KV-specific f32/int8/fp8
  vectors and per-token scales
  ([page_table.rs lines 309-351](../../crates/onnx-genai-kv/src/page_table.rs#L309-L351)).
- `KvCacheConnector` keys data by model, token-prefix hash, chunk index, and layer range,
  not immutable file regions
  ([connector.rs lines 65-98](../../crates/onnx-genai-kv/src/connector.rs#L65-L98)).
- The shipped hot/cold page movement is currently bookkeeping over host-owned payloads;
  the tier module says both tiers are in host RAM
  ([tiered.rs lines 1-8](../../crates/onnx-genai-kv/src/tiered.rs#L1-L8)).
- `LocalTieredConnector` explicitly does not implement real disk spill and retains an
  authoritative owned host payload
  ([local_tiered.rs lines 53-59](../../crates/onnx-genai-kv/src/local_tiered.rs#L53-L59),
  [lines 107-123](../../crates/onnx-genai-kv/src/local_tiered.rs#L107-L123)).
- `fp8.rs` is a software E4M3FN/E5M2 codec for KV payload compression
  ([fp8.rs lines 1-10](../../crates/onnx-genai-kv/src/fp8.rs#L1-L10)); it is not the
  weight-format layer for MXFP4 or IQ blocks.

The right reuse boundary is generic policy primitives after they are factored away
from KV semantics. Weight storage needs immutable external ranges, representation-aware
byte accounting, alignment, I/O and device-copy state, and completion-fenced leases.

## 4. Proposed architecture

> **Consolidated.** See [MEMORY_ARCHITECTURE.md §2-3](MEMORY_ARCHITECTURE.md) for the
> consolidated weight residency and governor design. The three-tier hierarchy,
> `WeightRegionCatalog`, `WeightResidencyManager`, `ExpertStore`, lease semantics,
> expert paging, and governor integration are now maintained there.

## 5. MoE expert paging and batching

> **Consolidated.** See [MEMORY_ARCHITECTURE.md §3.4-3.6](MEMORY_ARCHITECTURE.md).


## 6. Quantization synergy

Canonical resident and transferred bytes should stay compressed. MXFP4, IQ formats,
and affine int2/int4 reduce all three important quantities:

- disk footprint and storage bandwidth;
- host-cache footprint;
- H2D traffic and VRAM residency.

`BlockQuantizedMatMul` preserves native GGUF blocks in an opaque `uint8` tensor so
external-data slices remain mmap-able
([SUB4BIT_QUANT.md lines 201-225](../quantization/SUB4BIT_QUANT.md#L201-L225)). Fused expert tensors
should preserve the same expert-major native blocks.

### Dequantize in the kernel — default

Preferred for decode and constrained machines:

- direct IQ/MXFP4/int2/int4 GEMV/GEMM reads compressed pages;
- no full-expert f16/f32 buffer;
- minimum transfer and resident bytes;
- tiles can be released immediately after their dot products complete.

This is required for the “run from mmap” path.

### Dequantize/prepack on load — opt-in derived cache

Potentially useful for a hot expert on a large CPU machine or for a device library that
requires a prepacked layout:

- pays conversion once and reuses the result;
- consumes much more RAM/VRAM;
- must be keyed by format/kernel/device version;
- must be separately budgeted and evictable;
- should be admitted only after observed reuse exceeds a measured threshold.

The current full-f32 constant cache demonstrates the performance idea but also the
memory hazard. Huge-model mode must default away from it.

## 7. Device offload and partial GPU placement

### 7.1 User model: budget first, layer count as an override

The automatic planner should select the largest stable GPU placement that fits after
KV, activation, scratch, and EP headroom. Also expose an explicit llama.cpp-like
control for repeatability:

```text
device_policy = auto | cpu | gpu_layers:<N> | device_bytes:<SIZE>
```

`gpu_layers:N` is now parsed by the engine and translated into a byte-capped
whole-layer placement plan during native model load. The plan is reported in
`--profile`, including the planner's human-readable explanation. Enforcement is
still advisory in this increment: the native executor still owns one EP per session,
so device-planned layers are not yet made resident while host-planned layers execute
from CPU mmap/warm pages. A follow-up must consume the plan at the executor boundary.
An advanced expert pin list may be added later, but raw expert-count configuration is
less stable than a byte budget because expert sizes can differ.

### 7.2 Required execution changes

Partial offload requires:

- per-node or per-partition EP placement rather than one EP per executor;
- explicit host/device transfer edges at partition boundaries;
- lazy initializer bindings so a GPU kernel can receive selected expert pages without
  allocating the complete initializer;
- a paging-aware fused MoE kernel contract (`WeightHandle`/`ExpertStore`), or an
  engine-owned fused op that acquires leases before dispatch;
- stream/event lifetime integration so eviction cannot free in-flight device memory.

For the CPU mmap phase, a fused kernel can index selected subranges from the existing
borrowed full initializer. For GPU paging, a `TensorView` of a hypothetical complete
VRAM tensor is not honest; the kernel API needs a lazy weight handle or selected-page
binding.

### 7.3 H200-class path

- Detect real free HBM and reserve conservative headroom.
- Prefer whole-layer stable placement, then fill an expert cache by heat.
- Keep compressed blocks in VRAM unless a measured device kernel requires another
  representation.
- Keep a bounded pinned-host staging ring.
- ~~Prefetch the next exact/predicted expert wave on a transfer stream while the current
  wave computes.~~ **Superseded (#715, #718):** sequential weight prefetch was
  implemented and measured to produce no usable compute/transfer overlap on the
  dev box, and was removed in PR #715; AirLLM's own analysis (#718) attributes
  the cost to disk/transfer bandwidth rather than missing overlap. Do not present
  prefetch as an existing lever. Expert-wave prefetch remains a *design proposal*
  contingent on a measured win — see Phase 4.
- If all planned weights fit, eagerly load and pin them, disable eviction, and match a
  conventional resident runtime's hot path.

The design must benchmark fully resident performance separately; offload machinery is
not successful if it slows that case materially.

## 8. Configuration and UX

Use the existing `serving.memory.limits` surface as the global authority; it already
accepts byte, fraction, and `auto` values. The implemented weight-policy surface is
`serving.memory.weights.device_policy`; invalid values are load-time errors, not
silent fallback:

```yaml
serving:
  memory:
    limits:
      vram_limit: auto
      host_ram_limit: auto
      disk_spill_limit: auto
    weights:
      device_policy: auto        # or gpu_layers:48 / device_bytes:120GiB
```

Supported `device_policy` values:

- `auto`: plan from the governor-coordinated device weight budget.
- `cpu`: plan every discovered layer for host placement.
- `gpu_layers:<N>`: translate the first `N` discovered layers into bytes, capped by
  the governor-coordinated device weight budget.
- `device_bytes:<SIZE>`: plan from the explicit byte size, still capped by the
  governor-coordinated device weight budget. `<SIZE>` accepts the same byte suffixes
  as `vram_limit` (`MiB`, `GiB`, `MB`, `GB`, or raw bytes), but not fractions.

Current layer discovery covers native `com.microsoft::QMoE` expert regions, which are
the pageable MoE weight regions the existing planner understands. Models without such
regions run normally and produce no weight-placement profile row.

Environment aliases for command-line deployments:

```text
ONNX_GENAI_WEIGHT_OFFLOAD=1                    # Phase-1 route-first mmap CPU MoE
ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=<bytes>   # owned Phase-2 warm-cache override
ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=1       # opt-IN async fence-ordered device page-in (default: sync)
```

The older environment aliases sketched here for `ONNX_GENAI_WEIGHT_BUDGET`,
`ONNX_GENAI_WEIGHT_DEVICE_BUDGET`, `ONNX_GENAI_WEIGHT_HOST_BUDGET`,
`ONNX_GENAI_GPU_LAYERS`, and `ONNX_GENAI_WEIGHT_PREFETCH` are not implemented and
should not be documented as active
configuration until they are wired to the same parser.

`ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN` is **opt-in** (default: synchronous
page-in). Set it to a truthy value (`1`/`true`/`yes`/`on`) to use the
asynchronous, fence-ordered device page-in; unset or any other value keeps the
synchronous `cuMemcpyHtoD`. The synchronous path is the default because a
measured A/B (qwen3-0.6b-int4, 96 MiB device budget, eviction/thrash regime)
showed sync is faster — **15.84 tok/s sync vs 12.16 tok/s async**. When every
admit must evict, the async path re-serializes on the eviction compute-stream
drain and still pays a non-overlappable pinned-staging alloc+copy, so it cannot
hide the transfer. The async path stays fully available behind this flag and is
expected to become a net win once a warm-host materialize cache lands (removing
the per-page-in staging cost from the critical path). Both paths are byte-exact:
this flag never changes output, only the page-in mechanism.

`ONNX_GENAI_WEIGHT_OFFLOAD` is opt-in in Phases 1 and 2. When set to `1`, pageable
expert-major external QMoE tensors use route-first execution and bypass
full-pool materialization/dequantization caches. Phase 2 inserts a bounded
derived-f32 expert cache between mmap and compute. Its default owned-byte cap is
the Resource Governor's resolved host-RAM sub-budget; the host-bytes environment
variable can lower or override that cap. A zero-byte cap preserves the Phase-1
map-and-dequantize-per-use path. Unset (the default), or for any non-pageable
tensor, execution follows the existing resident QMoE path.

Precedence: explicit API > environment > YAML > auto defaults. Per-tier weight caps
are subordinate to global ceilings and may be reduced by the governor when KV or
scratch needs guaranteed space.

Suggested auto behavior:

1. inventory compressed shared/expert bytes and minimum scratch;
2. query real free/total RAM, VRAM, and filesystem capacity;
3. reserve safety headroom and the minimum KV/activation budget;
4. if all weights fit VRAM, choose fully resident;
5. else if all weights fit RAM, keep a host-resident backing set and maximize stable
   GPU layers/cache;
6. else choose mmap backing, size warm/device caches from remaining headroom, and
   print an estimated bytes/token and likely storage-bound warning.

At startup, print an explainable plan: total/shared/expert bytes, selected tiers,
resident layer count, cache caps, expected minimum working set, and whether async
prefetch is actually supported. Do not claim asynchronous overlap while the active EP
still implements synchronous copies.

`--profile` now prints the `PlacementPlan::explanation` for the computed static
weight plan. Because enforcement is not wired yet, that row is a planning/diagnostic
row rather than proof that the executor consumed the placement.

## 9. Observability and correctness invariants

Required metrics:

- mapped bytes versus resident RSS;
- owned host bytes and pinned bytes;
- device resident shared/expert bytes;
- hits/misses by tier and layer;
- disk-read and H2D bytes/token;
- page faults, load latency, and compute stall time;
- active experts, unique experts/batch, and tokens/expert;
- evictions, promotions, lease wait time;
- prefetch issued/hit/late/wasted bytes;
- expanded/dequantized derived-cache bytes;
- fully resident fast-path overhead versus baseline.

Correctness invariants:

1. Backing ranges are bounds-checked and immutable for the session lifetime.
2. Residency never changes quantization format or router/aggregation semantics.
3. A leased page cannot be evicted, unmapped, overwritten, or deallocated.
4. Device release occurs after the completion fence.
5. The manager never allocates an unbudgeted full expert/model expansion.
6. Every derived representation is reproducible from canonical backing bytes.
7. A prefetch miss or cancellation cannot change output.
8. Budget failure is reported before OOM with the minimum required working set.

## 10. Phased rollout

### Phase 1 — mmap disk tier and active-expert CPU access

**Ship independently:**

- formalize a `WeightRegionCatalog` over existing `WeightStore`/`WeightRef` ranges;
- require/validate expert-major contiguous external data for paging-capable MoE;
- add a fused CPU MoE path that computes routes first and reads only selected
  compressed expert slices from mmap;
- disable full-expert/full-pool dequant caches in huge-model mode;
- add mapped/RSS/fault/read-byte and active-expert metrics.

**Measure:** run a model/package larger than RAM with bounded owned memory; verify exact
routes/logits against the dense reference; report cold and warmed tokens/s and bytes
read/token. No GPU or explicit host cache is required.

### Phase 2 — bounded host-RAM expert LRU

**Status (2026-07-17): implemented for the fused CPU QMoE path.** The process-wide
warm cache stores immutable `Arc`-backed derived expert entries and charges their
expanded f32 FC1/FC2/FC3 byte size before allocation. Compute leases retain the
`Arc`, so an entry removed from the index cannot be freed during use. Admission
requires repeated use; frequency and recency select victims, while recently hot
entries receive a short policy pin to prevent a rare route from displacing them.
The cache reserves/evicts before fallible dequant allocation and never admits an
entry that would exceed the owned-host cap. Entries larger than the current cap
stream directly from mmap rather than making model loading fail.

The native engine seeds the cache cap from the Resource Governor's resolved
host-RAM limit. `WeightOffloadStats` reports hits, misses, evictions, current and
peak owned bytes, and the active cache budget separately from mmap size and Linux
RSS/page-fault counters. `ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=0` explicitly
selects the Phase-1 fallback. Device residency and asynchronous prefetch remain
Phase 3 and Phase 4 work respectively.

**Ship independently:**

- add byte-based host pages, leases, LFRU admission, pinning, and hysteresis;
- support zero-byte cache fallback to direct mmap;
- optionally use pinned staging and CPU-prepacked/dequantized derived entries with
  honest expanded-byte accounting;
- integrate host sub-budget reporting with the Resource Governor.

**Measure:** enforce configured owned-host cap; show convergence on a repeated routing
working set; compare direct mmap, pageable cache, and pinned cache hit rate/latency;
verify no cache thrash under rare routes.

### Phase 3 — device cache and partial GPU offload

**Status:** Phase 3a has landed the CPU-testable byte-budget placement planner,
the `WeightHandle`/`nxrt` capability seam, resident fallback, and
quant-block-aligned tile sizing. Its VRAM sub-budget arbitration has since been
removed in favour of the memory governor's ledger, which arbitrates for every
holder on a tier rather than for these three. Phase 3b remains responsible
for live device allocation, H2D transfer, lazy `pkg.nxrt::BlockQuantizedMoE` binding,
and device execution.

**Ship independently:**

- [x] add deterministic whole-layer placement planning and explicit planned transfer boundaries;
- [x] add the lazy device initializer/weight-handle capability seam and stock-EP resident fallback;
- [x] implement `gpu_layers:N`/byte-budget planning and quant-block-aligned tile sizing;
- [ ] implement bounded live VRAM expert pages and device binding (Phase 3b);
- [ ] connect CPU execution for planned non-GPU layers or waves (Phase 3b);
- [x] enforce coordinated weight/KV/scratch VRAM sub-budget arbitration — **now
  served by the memory governor's ledger** (`onnx-runtime-memory-governor`),
  not by this module. Phase 3a landed its own arbitration here; the ledger
  answers the same question for every holder on a tier, and two authorities
  dividing the same memory is what that work exists to end. The arbitration
  types were removed; `plan_placement` stays and takes its weight budget as an
  argument, so Phase 3b should ask the governor for it rather than deriving one;
- [ ] connect the plan and arbitration decisions to live device execution (Phase 3b).

**Measure:** models larger than VRAM complete without whole-session CPU fallback or
OOM; sweep GPU layer/device cache budgets; report H2D bytes, stalls, tok/s, and peak
VRAM. On a fitting model, fully resident performance must remain near baseline.

### Phase 4 — asynchronous and predictive prefetch

> **Status: design proposal, and its central premise is measured false so far.**
> Sequential weight prefetch was implemented and **removed in PR #715** because it
> produced no usable compute/transfer overlap on the dev box; AirLLM's own
> analysis (#718) independently attributes the cost to disk/transfer bandwidth,
> not to a lack of overlap, and rates prefetch at ~10% by their own account. On a
> model that does not fit there is not enough independent compute to hide the
> transfer behind. Treat everything below as a proposal that must clear a measured
> end-to-end win before it is built, not as a planned or landed capability. The
> "landed" annotations below refer only to the low-level plumbing (stream-ordered
> H2D, awaitable fences, a standalone double-buffer scheduler), **not** to
> prefetch producing a throughput gain.

**Ship independently (plumbing only; predictive prefetch itself is unproven):**

- implement true stream-ordered H2D and awaitable fences; **[landed — issue #87]**
- double-buffer expert panels; **[strategy landed as a standalone, unit-tested
  scheduler; live MoE wiring pending Phase-3b device binding]**
- add exact-next-wave, heat-based, then opt-in router-predicted prefetch
  — **only if measured to help; sequential prefetch was measured not to (#715)**;
- budget reservations and cancel low-value work under pressure.

**Measure:** prefetch hit/late/waste, hidden transfer percentage, p50/p99 token latency,
and throughput across decode and prefill. Predictive mode graduates to default only
when it improves end-to-end performance without increasing memory violations or tail
latency.

## 11. Open questions for owner review

1. **Lazy initializer boundary:** should paging-aware weights enter kernels through a
   new executor `WeightHandle`, through an EP/custom-op context, or through an
   engine-owned fused MoE path? The current all-inputs-are-`TensorView` contract cannot
   honestly represent a partially resident GPU initializer.
   **Resolution:** Use a general executor `WeightHandle` from the start, compatible with
   existing ORT plugin EPs through capability detection. Paging-capable EPs advertise an
   `nxrt` capability flag and receive a lazy `WeightHandle`; stock ORT EPs receive a
   materialized resident-tensor fallback. Paging is opt-in, never a correctness dependency.
2. **ORT integration:** can upstream `QMoE`/plugin EPs lazily access external expert
   slices, or does practical offload require the private `BlockQuantizedMoE` boundary?
   **Resolution:** `pkg.nxrt::BlockQuantizedMoE` is the offload boundary and alone honors
   lazy expert leases, capability-negotiated with a plain `QMoE` fallback. Mobius emits
   `BlockQuantizedMoE` when the `nxrt` capability is present, otherwise `QMoE`; file an
   upstream ORT issue for lazy-external-weight `QMoE`.
3. **Exporter contract:** which metadata is required beyond expert-major shape to bind
   FC1/FC2/FC3, scales, zero points, shared experts, nonuniform expert sizes, and
   format/layout versions without name inference?
   **Resolution:** Use a hybrid contract: numeric bindings (FC1/FC2/FC3, scales,
   zero-points, shared-expert flag, and per-expert sizes) are explicit op inputs or
   attributes, never name-inferred; residency metadata lives in the package manifest; and
   format/layout version is mandatory and explicit, with the loader hard-rejecting a
   mismatch. Residency metadata is a compact model- or layer-group-level layout descriptor
   (stride, tiling, page size, and expert-range formula) referenced by a small region-group
   ID on each op—O(1)–O(layers), not per-expert. Compute concrete byte ranges from
   `WeightStore` offsets plus the descriptor.
4. **Host budget semantics:** do we promise a cap on owned cache bytes only, or a best-
   effort process RSS cap using mmap advice? OS page-cache residency is not strictly
   controllable by the runtime.
   **Resolution:** The cross-platform contract is a hard cap on owned cache bytes.
   RSS-tightening is advisory, off the hot path, and acts only on already-evicted pages so
   it cannot regress performance, behind a `PageAdvisor` trait (`madvise` on POSIX,
   `Offer` + `DiscardVirtualMemory` on Windows, and a no-op fallback).
5. **Partial-GPU policy:** is `gpu_layers:N` required as a stable public compatibility
   knob, or should bytes plus an explainable placement plan be the primary API?
   **Resolution:** Make a byte budget plus an explainable placement plan the primary API.
   Retain `gpu_layers:N` as a compatibility override and report it back in bytes.
6. **Mixed CPU/GPU MoE:** may one fused layer execute some expert waves on CPU and
   others on GPU, or should the first device phase keep each layer on one compute
   device to simplify ordering and numerics?
   **Resolution:** Phase 3 uses a single device per layer; defer intra-layer expert splits
   to a later measured optimization.
7. **Minimum tile size:** what transfer-page/panel sizes best balance NVMe readahead,
   pinned-memory pressure, direct compressed kernels, and GPU occupancy across MXFP4,
   IQ, and affine int2/int4?
   **Resolution:** Default to expert-FC panels comprising whole quant blocks—a tile must
   never split a quant block. Provide a byte-size override that snaps to block boundaries,
   with per-format minimums for MXFP4, IQ, and affine int2–4. Defer auto-tuning to Phase 4.
8. **Governor arbitration:** what minimum KV guarantee and rebalancing hysteresis
   prevent KV/expert-cache oscillation under continuous batching?
   **Resolution:** Use dynamic arbitration: a hard KV floor sized to committed in-flight
   sequences, watermark hysteresis, a minimum rebalance dwell, and admission control at
   batch formation. Thoroughly test oscillation/thrash, KV-floor breaches, and admission
   under continuous batching; these are hard test gates.
9. **Prefetch predictor:** can GLM-5.2/DeepSeek routing be predicted early enough to
   hide storage/H2D latency without duplicating router compute or wasting bandwidth?
   **Resolution:** Use layered opt-in escalation: (a) exact-next-wave by default, (b) a
   heat warm-set, and (c) router prediction as opt-in, graduating to default only when
   measured to help. Provide a trait-based public `ResidencyPolicy` extension point:
   policy advises hints, priorities, and eviction candidates; the Resource Governor remains
   authoritative for budgets, the KV floor, and leases, and cancels low-value work.
   “Policy proposes, Governor disposes”: a bad policy may hurt performance but cannot
   violate memory safety or correctness.
10. **Integrity/lifetime:** should package validation pin file identities/hashes and
    reject replacement/truncation while mmaps and derived cache entries are live?
    **Resolution:** Pin file identity cheaply at load using size plus mtime/inode, or a
    fast header-plus-region-table signature—O(1), with no full re-hash. Offer opt-in
    full hashing for attestation, translate `SIGBUS` to a clean runtime error, and reject
    live truncation or replacement of a mapped package.

## 12. Decision

Proceed, after owner approval, with a **weight-specific parallel residency subsystem**
backed by the loader's existing mmap `WeightStore`. Reuse generalized policy
primitives from `onnx-genai-kv` only after removing KV-specific token, sequence, and
payload assumptions. Make fused MoE the paging boundary, preserve compressed blocks
through all tiers, use leases for pointer lifetime, and let the Resource Governor
coordinate weight, KV, activation, scratch, and EP budgets.
