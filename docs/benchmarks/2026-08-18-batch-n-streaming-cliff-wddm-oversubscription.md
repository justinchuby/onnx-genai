# The batch-N streaming cliff is WDDM oversubscription thrash, not our span churn

**Date:** 2026-08-18
**Author:** Copilot (VMM/offload slice)
**Owner directive:** "继续推进 vmm、offload、streaming、multi-request batching 对大模型的支持,
提高速度,实现简洁高效" — the **vmm/offload** slice.
**Issue:** #1295 — "Streaming-regime batch-N ceiling is VMM map/unmap churn, not the
batching path (N_max~4-5 vs projected ~19 on 8GB)". Handover from the batching slice.
**Companion:** `2026-08-18-batch-n-scaling-8gb-limiters.md` (#1295's record),
`2026-08-18-moe-per-expert-paging-churn.md` (#1308, the concurrent per-expert MoE slice).

## Question this answers

#1295 established (and this slice verified, did not re-derive) that in the weight-streaming
regime the batch-N wall-clock ceiling is `N_max ≈ 4–5` on 8 GB — far below the `~19` #884/#891
projected — and attributed the limiter to **`vram_free_ms` exploding 9.7 s → 90 s** once
`mapped_physical` (8.39 GB @ N=8) crosses physical VRAM (8.19 GB). The 1/N htod amortization is
confirmed real (contention-invariant `htod_bytes/token` tracks 1/N); the batching path is not the
limiter.

This doc **characterises the cliff before proposing a fix**, answering #1295's two questions:

1. **What exactly costs the time** — `cuMemUnmap`, `cuMemRelease`, driver granule reclaim, WDDM
   paging, or *our* bookkeeping over many small spans?
2. **Is it our churn or the driver's?** If it is WDDM thrashing on oversubscription, no amount of
   our bookkeeping fixes it; the right answer is to **stop over-subscribing** (admission control).

The "many small spans" hypothesis had specific supporting evidence to *check, not assume*: loading
`qwen05b-q4` under VMM leaves the arena holding **440 MiB across 517 spans**, and the concurrent
per-expert MoE workstream (#1308) deliberately creates *more, smaller* regions.

## Hardware / method (house rule §32.2)

- **Box:** RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1, **WDDM**), Intel i7-13800H (14C/20T).
  CUDA 13 runtime on PATH via the anaconda `nvidia/cu13` + `cudnn` wheels. Native CUDA EP.
  `cuMemGetInfo` at start: **total 8188 MiB, free ~7959 MiB** (≈229 MiB held by the desktop
  compositor — the *usable* device budget is ~7959 MiB, **not** the nominal 8188).
- **GPU idle before each run** (`nvidia-smi`: 0 MiB used, 0 %); single process. But note: this box
  is shared with build agents and the OS itself pages our VMM granules under system-wide pressure
  (#863) — **wall-clock driver timing here is not contention-immune**, which is itself part of the
  finding below.
- **Instrument:** `bench-seqmajor/src/bin/churn_cliff.rs` — a **bare driver-API** microbenchmark
  (cudarc `driver::sys`, **no `Arena`, no `Spans` BTreeMap, no `granule_refs`, none of
  `vmm_allocator.rs`**). It reserves 11 GiB of VA (physical is 8 GiB, so it *can* oversubscribe) and
  times each op — `cuMemCreate` / `cuMemMap` / `cuMemSetAccess` / touch (`cuMemsetD8`+
  `cuCtxSynchronize`) / `cuMemUnmap` / `cuMemRelease` — **separately**, by VRAM-fill level. 2 MiB
  device granule (same as the real arena, #776).
- **Reference baseline:** the non-oversubscribed regime, measured in the **same binary** (Phase 3
  "under" 0.80× vs "over" 1.03×). Prior in-tree reference: the #1308 MoE churn doc measured
  ~2.6 µs/unmap at a 128 MiB (non-oversubscribed) peak.
- **Repetitions:** Phase 3 was run **5 times**, Phase 1 deciles **3 times**, because the first run
  produced a result that did not reproduce — see below. Ranges are reported, not a single number.

## Result — and a retracted first reading

**A single first run showed a 27× unmap+release explosion at oversubscription (1264 ms/cycle vs
45 ms). It did not reproduce and is retracted as an outlier.** Reporting it as the characterisation
would have repeated the exact failure this project keeps correcting (measurement-discipline #6:
wall-clock on a box that pages its own memory). What survives repetition is narrower and points the
same way.

### Phase 3 — steady per-step churn (hold R resident, churn a fixed 96-granule/192 MiB chunk)

Span count is **identical** across all three rows; only resident fill changes. Median ms/cycle of
8 cycles, across 5 runs (min–max):

| scenario | peak/VRAM | map+touch ms/cycle (5 runs) | unmap+release ms/cycle (5 runs) |
|----------|----------:|-----------------------------|---------------------------------|
| under | 0.80× | 28–49 (med ~46) | 21–47 (med ~31) |
| near  | 0.97× | 45–171 (med ~59) | 18–348 (med ~30) |
| over  | 1.03× | 51–402 (med ~60) | 19–**1264** (med ~33) |

- **`map+touch` (page fault-in) degrades reliably in *direction*** — `over > near > under` in every
  run — but modestly in the common case (~1.3–2× at the median) with occasional severe spikes.
- **`unmap+release` shows no reliable trend with fill.** Its median barely moves (31→30→33 ms); the
  only large values are non-deterministic storms (one 1264 ms, one 348 ms), not a systematic
  oversubscription cost. **"cuMemUnmap got slow" is not the reproducible mechanism.**

### Phase 1 — commit + touch, per-op µs/granule by VRAM-fill decile

Quiet run: `touch` is flat ~210 µs/granule (2 MiB page = ~9.5 GB/s) up to ~90% fill, then rises to
**407 µs at 90–100%** and **618 µs at 100–110%** (~2.3 GB/s ≈ PCIe) — a clean page-in cliff **at the
usable-VRAM boundary**. `create`/`map`/`setacc` stay small (tens of µs / single-digit µs / tens of
µs) throughout. A second run under background pressure was **chaotic** — `touch` spiked to
2200–2900 µs at *mid* fill (30–50%), nowhere near oversubscription — i.e. external system-memory
pressure destabilises the whole curve regardless of our fill level.

### Phase 2 — isolated unmap+release, top-down teardown

Freeing monotonically from an oversubscribed peak (no re-mapping pressure) is **sub-millisecond per
granule at every fill level** (unmap 166–681 µs, release 82–319 µs). An isolated unmap/release is
not the cliff.

## Reading (measured)

1. **Span count is not the driver of the cliff.** Phase 3 holds span count constant and varies only
   fill; the reproducible effect comes entirely from crossing the VRAM boundary. This **refutes the
   "440 MiB / 517 spans" hypothesis** — many small spans are not what costs the time.
2. **It is the driver / WDDM, not our code.** `churn_cliff` is the bare driver API with none of our
   allocator bookkeeping, yet it reproduces the oversubscription page-in cliff. Our `Spans` /
   `granule_refs` / coalescing cannot be the cause because none of it is present here.
3. **The reproducible cost is page *fault-in* (`touch`) under oversubscription, not `unmap`.** Once
   the touched working set crosses **usable** VRAM (~0.97× of nominal 8188, i.e. the ~7959 MiB
   `cuMemGetInfo`-free budget, *not* nominal total), the driver backs the overflow with system
   memory and every access faults over PCIe (~9.5 → ~2.3 GB/s).
4. **Beyond the boundary the cost is not just larger, it is *unpredictable*.** The same "over" config
   ranged from ~50 ms to >1200 ms/cycle depending on system-wide memory pressure. That variance
   **is** the cliff signature: once you hand residency to WDDM you lose bounded, attributable cost.

## Reading (inferred)

- **The 90 s `vram_free_ms` in #1295 is most plausibly one or both of:** (a) a WDDM eviction storm
  once over the usable-VRAM line — the outlier regime this microbenchmark reproduced once and could
  not stabilise; and/or **(b) a probable mis-attribution in the instrument.** The real free path,
  `CudaWeightPage::release_allocation(true)` (`weight_paging.rs`), calls
  `self.runtime.synchronize()` **and** `copy_stream().synchronize()` *inside* the
  `GLOBAL_VRAM_FREE_NS` bracket (`Drop::drop` → `add_duration(&GLOBAL_VRAM_FREE_NS, …)`). Under
  oversubscription the enqueued compute/copy is itself PCIe-fault-slowed (finding #3), so the
  "free" counter can be **waiting on paging-slowed work, not unmapping.** This is the same class of
  error as the retired `h2d_enqueue_copy_ms` (measurement-discipline #2). **Actionable:** time
  `cuMemUnmap`/`cuMemRelease` separately from the pre-free stream drain in `weight_paging.rs` before
  quoting `vram_free_ms` as an unmap cost.
- **The #1295 candidate menu — coalesce spans, retain/reuse handles, defer/batch unmap — targets
  our-side churn, which the data shows is not the bottleneck.** They cannot move a driver-side
  page-in penalty that reproduces with zero of our bookkeeping present. (The tree already coalesces
  unmap over contiguous runs and retains handles in a `PhysicalHandlePool`, consistent with those
  not being the lever.)
- **The only mechanism-addressing fix is to stop over-subscribing: admission control.** Cap the sum
  of mapped-physical consumers (weights + KV + activations) at a safe fraction of **usable** VRAM
  (`cuMemGetInfo` free × ~0.90) and **refuse the growth that would cross it** — the (N+1)th
  sequence's KV/activation admission. Keeping the peak in the "under" regime keeps page-in at
  ~9.5 GB/s and cost bounded; crossing it hands cost to WDDM and makes it both larger and
  unpredictable.
- **Honest success criterion (per #1295):** admission control does **not raise `N_max`** — it
  converts the catastrophic, non-deterministic **cliff into a plateau** (a graceful refusal / stable
  ceiling). That is the achievable and far better failure mode. Not a raised ceiling.

## Where the fix goes (code pointers, for the follow-up)

The refusal machinery already exists and does **not** need to be built:

- `CudaVmmAllocator::try_commit_span` (`crates/onnx-runtime-cuda-memory/src/vmm_allocator.rs`)
  already enforces `max_additional_mapped_bytes` and returns a refusal `MemoryError`.
- The memory governor already grants/refuses a `MappedAllowance` per tier (G3 refusal,
  `crates/onnx-runtime-memory-governor/src/lib.rs`).
- The device VRAM tier is already measured honestly —
  `engine/governor.rs::device_vram_capacity` → `FixedCapacity::new(total, free)` — so `free` is in
  hand.
- **The gap is what the cap is sourced from.** #1295 capped the *weight* budget at 6 GiB
  (`ONNX_GENAI_VRAM_LIMIT`), but KV/activation mapped growth is admitted against a *separate*
  headroom, so the **combined** mapped physical reached 8.39 GB > usable VRAM. The change is to
  bound the **sum** of mapped-physical consumers at `free × safe_fraction` and let the existing
  `try_commit_span` refusal fire on the KV/activation admission path. Plus the instrumentation fix
  above so `vram_free_ms` measures unmap, not a sync.

## Interaction with the concurrent per-expert MoE workstream (#1295 item 4)

Per-expert MoE paging (#1308 → executor wiring) deliberately creates **more, smaller VMM regions**.
#1295 flagged the risk that if span count drove this cliff, that change would worsen churn. **This
measurement clears it:** span count is *not* the driver (measured #1, and the bare-driver-API
reproduction). Moreover per-expert paging **lowers** peak mapped physical (only top-k experts
resident vs the whole bank), moving the system *away* from the oversubscription boundary — the safe
side of this finding. The surviving caveat: per-expert paging must respect the *same*
total-mapped-physical admission cap, because its benefit is precisely keeping the peak below usable
VRAM. Recommend the owner connect the two slices on the shared cap; no edits to their work from here.

## Reproduce

```powershell
$nv = "$env:LOCALAPPDATA\anaconda3\Lib\site-packages\nvidia"
$env:PATH = "$nv\cu13\bin\x86_64;$nv\cudnn\bin;$env:PATH"
cd bench-seqmajor
cargo run --release --bin churn_cliff   # run several times — Phase 3 magnitude varies with
                                         # system memory pressure; the DIRECTION (over>under for
                                         # map+touch) is the stable signal, not any single number.
```
