# MoE Expert-Offload Architecture — What We Measured and What to Build

**Status:** Measurement synthesis / design guidance
**Date:** 2026-08-18
**Scope:** Residency and transfer of mixture-of-experts weights when the expert
bank exceeds VRAM — the "what to do optimally when you cannot fit all the
experts" problem. Composes three independent workstreams that converged this
week. Register follows [`MEMORY_MANAGEMENT_MODEL_DESIGN.md`](MEMORY_MANAGEMENT_MODEL_DESIGN.md):
state what was measured, on what hardware, with what model, and mark what
remains inferred.

## Hardware and provenance

Every measured figure below was taken on one machine unless stated otherwise:

> **This box:** RTX 4060 Laptop (8 GB, driver 591.55, CUDA 13.1 runtime, WDDM) ·
> i7-13800H (14C/20T) · **63.8 GB DDR5-6400** · KIOXIA KXG80 NVMe Gen4 ·
> ~218 GB free on C:. This is an 8 GB WDDM laptop, **not** the Linux datacentre
> GPU the production target runs on — a gap called out wherever it changes a
> conclusion.

Three workstreams, so a reader can trace any claim to its evidence:

- **Mechanism** (this document's author): what the weights are backed by, whether
  the OS/storage can be handed paging, the three-regime page-cache sweep, the
  host/PCIe crossover, CUDA UM on WDDM, DirectStorage/GDS, and the layout
  constraint. All measured on *this box*.
- **Policy** (MoE agent, merged as **#1321 / #1322 / #1326 / #1331**): the routing
  concentration curve, static hot-pin vs the Belady oracle, the composed PCIe
  floor, the absence of a knee, route-aware scheduling, and the prediction-window
  gate. Trace-driven from a real routing trace; cited here, not re-derived.
- **Batching** (batching agent): batching as the largest lever on bytes/token, and
  the fixed batch-decode overhead that currently blocks it.

---

## Part I — Mechanism (measured on this box)

### 0. The layout constraint gates the entire storage-to-device family

Before any transfer question: **can a kernel read raw file bytes at all?** Only if
the on-disk layout is the layout the kernel consumes.

- **Canonical MatMulNBits int4** (`b_shape = [n, k_blocks, blob_size]`) is read
  directly by the default CUDA kernel — proven by a **bit-identical** zero-copy
  GEMV straight from a host-mapped file region (`cuMemHostRegister` DEVICEMAP|READ_ONLY).
  **File-backed-usable.**
- **CUDA `BlockQuantizedMoE` expert offload** copies *"canonical compressed bytes …
  never a full host expansion … byte-identical"* — stored bytes stay in on-disk
  block-quant layout; dequant is an in-kernel/transient compute step, not a
  load-time repack. **File-backed-usable.**
- **Marlin** (`repack_int4_weights`, device-side) and **MLAS-prepacked** tensors
  need a layout that **does not exist on disk**. Streaming raw file bytes yields no
  kernel-usable buffer for them (and both are resident-only anyway). **Excluded.**

This single answer settles mmap, DirectStorage and GDS together: the family is
available for the default/MoE canonical path and unavailable for the repack paths.
It is the cheapest disqualifier and it is checked first for that reason.

### 1. What the weights are backed by today

`git grep` across the tree: **zero managed/unified memory anywhere.** Weights are
device memory via two allocators — `CudaDeviceAllocator` (`cuMemAlloc`) and the
VMM arena (`cuMemAddressReserve` + 2 MiB PINNED device granules + `cuMemMap`).
The streaming/offload path stages from a **file-backed mmap** of `model.onnx.data`
(`WeightStore::map_external`) through host-pinned buffers, and the shipped #864
zero-copy hybrid registers the cold over-budget fraction of that mapping directly.
So the loader already hands kernels pointers into file-backed memory — the seam a
storage-to-device design would build on already exists.

### 2. The three regimes — how much the free OS page cache captures

The decisive question was not "does mmap work" but **how much of the achievable win
the Windows cache manager captures for free, and how much headroom is left for a
routing-aware cache.** Measured with a **routing-skewed random** access pattern
(so readahead cannot help), driven from the measured MoE skew (25% of experts
carrying 45.4% of reads), over a 15 GB bank of 16 MiB "experts", **under
ballast-induced memory pressure** swept so the answer is a curve. Model
gemma-3-27b `model.onnx.data`; single-thread; host tier; decode regime. Warm/cold
calibration was cleanly bimodal (warm ≈ 1.1–2.5 ms vs cold ≈ 7–31 ms per 16 MiB),
so hit classification is unambiguous.

| ballast | usable cache | cache vs hot-set (3.75 GB) & bank (15 GB) | **hot warm-hit** | OS/oracle capture |
|---|---|---|---|---|
| 24 GB | ~21 GB | cache ≥ bank (all fits) | 85.8% | 83% |
| 32 GB | ~8 GB | hot < cache < bank | 87.9% | 88% |
| 38 GB | ~5 GB | cache ≈/< hot-set | **33.1%** | 76% |

Three regimes fall out:

1. **cache ≥ full bank** — everything fits. OS captures ~100%; residual misses are
   *compulsory cold-start*, which a pinned cache would not avoid either. Routing
   knowledge adds **≈0**.
2. **hot-set < cache < bank** — bank overflows but the hot set fits. **LRU keeps the
   hot set for free because frequent ≈ recent → ~88% hot-warm.** The general-purpose
   cache already approximates the oracle; routing knowledge adds little.
3. **cache ≲ hot-set** — the OS cannot hold the hot set; tail traffic evicts it.
   **OS collapses to 33% hot-warm.** This is the only regime with real headroom: a
   routing-aware cache that *pins the hottest experts* takes hot-access latency from
   3.89 ms median / 29 ms p90 back to ~1.3 ms — a **~3× win on hot-expert reads**.

This **refines** the earlier prediction that "MoE beats the OS." It does not,
uniformly: the OS exploits skew for free whenever the hot set fits in cache. MoE
beats the OS **specifically in the oversubscribed-hot-set regime** — which is
exactly the hundreds-of-GB-MoE case the target names.

### 3. The ~45 GB crossover on 64 GB DDR5 (inferred)

Usable file cache on this box is ~40–50 GB. So the crossover into regime 3 requires
the **hot working set** (not the full bank) to exceed ~45 GB:

- Dense models and MoE whose full weights ≤ ~45 GB → regime 1, OS wins, zero
  headroom.
- MoE whose hot expert set alone > ~45 GB (GLM/DeepSeek-class) → regime 3,
  routing-aware pinning has real headroom.
- A 128–192 GB workstation pushes the crossover proportionally higher, so *more*
  models fall into regimes 1–2 where the free cache suffices.

**Inferred, not measured:** the specific ~45 GB figure and its linear scaling come
from the measured cache/hot-set relationship, not an end-to-end benchmark at that
size — the 17 GB test file caps a direct test. The *shape* (three regimes, crossover
at cache ≈ hot-set) is measured; the *coordinate* on a 64 GB machine is projected.

### 4. The host/PCIe crossover — where a routing-aware cache earns its complexity

Warm host reads (~10.5 GB/s, single-thread page-cache hit) **exceed every
host→device path** measured on this box: zero-copy `cuMemHostRegister` device read
**~5.9 GB/s**, staged pinned H2D **~10.9 GB/s** (this laptop's PCIe ceiling),
resident VRAM **~178 GB/s**. So in regimes 1–2 the host tier is *not* the bottleneck
— **PCIe is**, matching the policy workstream's independently-derived "the wall is
PCIe" conclusion (§II).

Regime 3 is different: eviction-miss traffic drags achieved host throughput to
**1.3–5 GB/s, below PCIe**, and the host tier itself becomes the wall. That is the
one place the DRAM tier stops being free and a routing-aware cache pays for itself.

### 5. CUDA Unified Memory is empirically closed on WDDM (hard negative — do not re-propose)

Recorded so nobody re-proposes it. Measured on this box (591.55 / CUDA 13.1 / WDDM):

- `CONCURRENT_MANAGED_ACCESS = 0` — no hardware demand paging.
- A 12 GiB managed allocation on the 8 GiB card completes but at **~0.6 GB/s**
  (WDDM software paging / whole-allocation migration, not page-granular fault-in).
- **`cuMemAdvise_v2` and `cuMemPrefetchAsync_v2` both return
  `CUDA_ERROR_INVALID_DEVICE`** — the steering hints the owner's "hint, don't manage"
  thesis would rely on are rejected outright on this platform.

CUDA UM cannot page weights on WDDM. This is consistent with the shipped choice to
**stop managing and let the platform demand-page** when a model exceeds budget on
Windows (#864/#874). Linux discrete and TCC behave differently (they fail at the
physical limit rather than demand-paging), so this is a WDDM-specific negative.

### 6. DirectStorage and GDS — Regime-B backstops, not breakthroughs

Both change the *source* of a device fill (storage → device, bypassing the host
copy) rather than the *rate* (still PCIe-bound). Neither exceeds the PCIe link.

- **DirectStorage → D3D12 → `cudaImportExternalMemory`** is the Windows twin of GDS
  and is testable here in principle: `d3d12.dll` and the CUDA import types are
  present, but the DirectStorage runtime redist is not installed. Prior art
  (`dstorage_gpu`) headline numbers are all **load-regime** (500 MB / 42 ms ≈
  11.7 GB/s ≈ PCIe; its "16×" is `torch.load` being CPU-copy-bound at 0.73 GB/s,
  not DirectStorage exceeding PCIe). The **D3D12→CUDA interop link itself is
  inferred-not-measured** on this box; a minimal upload→`CreateSharedHandle`→
  `cuImportExternalMemory` probe would close it, and stays cheap to revisit if
  Regime B goes live. Structural cost: a D3D12 dependency, a Windows-only path, and
  fence/semaphore synchronisation — a real race surface.
- **GDS / `cuFile`** is **Linux-only**; untestable here. It is the same *shape* as
  the DirectStorage path (storage populates a device buffer we then read), so a
  design built around file-backed weights is portable in architecture across both
  even though the code is not.

Both matter **only in Regime B — expert bank exceeds host RAM as well as VRAM** —
where there is no host copy to page from and NVMe→device is the only path
(measured cold-NVMe floor here: **3.7 ms per 16 MiB**, ~4.5 GB/s). When the bank
fits in host RAM (Regime A), they equal the staged copy on throughput and save only
CPU overhead. Label which regime any DirectStorage/GDS claim belongs to.

---

## Part II — Policy (MoE agent; measured trace-driven, cited)

Merged as **#1321 / #1322 / #1326 / #1331**; numbers reproduced for composition, not
re-derived.

- **Concentration curve** (real routing trace): top **12.5% → 27.1%** of read volume,
  **25% → 45.6%**, **50% → 74.4%**. Materially **milder than the assumed 80/20** —
  skew is real but not extreme, which bounds how much any pinning policy can win.
- **Static hot-pin** sits **~17–27% above the Belady oracle** — a simple,
  routing-informed static pin captures most of the achievable policy win; the
  dynamic-optimal ceiling is close enough that the complexity of chasing it is hard
  to justify. The **DRAM tier removes avoidable SSD traffic entirely** (once DRAM
  holds the bank, the SSD leaves the critical path).
- **Composed PCIe floor:** **1.63 GB/token → ~125 ms/token → ~8 tok/s** from expert
  transfer alone at 25% of experts VRAM-resident, batch 1, PCIe x8 — *before*
  compute. This is the independent derivation that meets the mechanism workstream's
  "the wall is PCIe" from the opposite direction.
- **No sharp knee:** VRAM keeps paying well past where folk wisdom expects, so
  "how much VRAM" is a smooth cost/benefit dial, not a cliff to sit just above.
- **Route-aware scheduling** adds only **17–39%** over plain batching, at a p99 cost
  of **102 queue positions unbounded / ~7–25 when bounded** — a modest lever with a
  real tail cost.
- **Prediction-window gate (#1331):** the window between "router output for layer L
  known" and "layer L experts needed" is **≈0 intra-layer** (topology: `TopK` router
  indices feed the expert MatMuls directly, no intervening work). Cold-NVMe (3.7 ms)
  and even PCIe H2D (1.5 ms per 16 MiB, mechanism-measured) exceed any realistic
  per-layer decode window. **Dynamic demand-prefetch cannot beat static hot-pin:**
  the only experts prefetchable with both certainty and lead time are the always-on
  core, which a static pin already keeps resident. Dynamic prefetch is foreclosed by
  the window, not by prediction quality.

---

## Part III — The gate (batching agent)

Batching is the **single largest lever on expert transfer**: **−61% bytes/token**
going W=2→8, because sequences sharing a fused forward read each weight once for
many tokens. But that saving is **in bytes only**. This stack currently pays
**~5.4× per step going M=1→M=2** (~2.55 → ~14 ms), with GEMV excluded as the cause —
a fixed batch-decode overhead. So the biggest MoE lever is **blocked behind a
batch-decode overhead bug**: the bytes/token win cannot be realised until stepping
at M≥2 stops costing 5.4×. That dependency belongs on the critical path to any MoE
throughput work.

---

## Part IV — What this means we should and should not build

Composing all three workstreams:

1. **Build routing-aware expert pinning only for Regime 3.** It is near-worthless in
   regimes 1–2, where the free OS page cache already captures **83–88%** and LRU
   exploits skew for free. It earns its complexity only once the hot expert set
   exceeds usable DRAM cache (~45 GB on this 64 GB box; proportionally more on a
   larger machine). Build the *static hot-pin* form — the policy workstream shows it
   is within ~17–27% of the Belady oracle, and the prediction-window gate (#1331)
   shows **dynamic prefetch is foreclosed**, so there is no reason to build the
   dynamic machinery.

2. **The target (hundreds-of-GB MoE) sits in Regime 3, so pinning is worth building
   — but it cannot be validated on this hardware.** An 8 GB WDDM laptop with a 17 GB
   test file cannot reach a >45 GB hot set. What *would* validate it: a Linux
   datacentre GPU with a real hundreds-of-GB MoE, measuring hot-warm-hit and
   achieved tok/s across VRAM-resident fractions against the composed PCIe floor
   (~8 tok/s at 25%, batch 1, x8). Until then the pinning win is **projected from the
   regime shape, not measured at target scale.**

3. **Quantisation is likely the best return per unit of complexity.** It attacks the
   PCIe wall **linearly** (fewer bytes/token → higher tok/s at the same residency)
   and needs **no residency policy at all** — no pinning, no eviction, no scheduler.
   The MoE agent is pricing it now; the cost is stated accuracy, not engineering
   surface. Against a 1.63 GB/token PCIe floor with no knee, halving bytes/token
   roughly doubles the transfer-bound ceiling for free of policy.

4. **DirectStorage / GDS are a Regime-B backstop, not a breakthrough.** They help
   only when the bank exceeds host RAM too; they never beat PCIe; and they inherit
   the same layout constraint (canonical only). Worth keeping as a portable
   file-backed design *shape* (Windows DirectStorage ↔ Linux GDS), not worth
   building until Regime B is the live case. The one open empirical gap is the
   D3D12→CUDA interop link, left inferred.

5. **Do not re-propose CUDA Unified Memory on WDDM.** Measured dead (§I.5).

**Ordering implied by the evidence:** fix the M≥2 batch-decode overhead (unblocks the
−61% bytes lever) and pursue quantisation (linear PCIe relief, no policy) *before*
building any residency policy; build static hot-pin only when a Regime-3 model is the
actual target and can be validated on Linux hardware. Challenge this if a measurement
says otherwise.

---

## Corrections and cross-links to `MEMORY_MANAGEMENT_MODEL_DESIGN.md`

That document said the dense/MoE reuse question **"should be measured before any
MoE-specific residency policy is designed."** It now has been — this document and
#1321/#1322/#1326/#1331 are that measurement. Its **dense conclusions stand
unchanged** (`reads_per_step = 1.000` across 867 keys #944; the ~30× loss to the OS
#864/#874). The regime framing here explains *why* the OS won there: **dense has no
reuse, so nothing clever was possible — that is regime 1 with a single-touch access
pattern, where the OS is optimal by construction.** MoE has reuse, but the OS still
wins in regimes 1–2 and loses only in regime 3.
