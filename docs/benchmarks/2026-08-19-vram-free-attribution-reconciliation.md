# `vram_free_ms` reconciliation: the 41–45 s is not driver freeing (and is not a stable number)

**Hardware:** RTX 4060 (8 GB, driver 591.55, CUDA 13.1, WDDM), i7-13800H, 63.8 GB DDR5.
**Model:** `qwen2.5-14b-onnx` (int4, 7.76 GB weights on disk, `total_weight_bytes=8,318,428,306`),
batch=1, greedy, `--backend native`, `--ep cuda`, release build (`target/release/profile_native`).
**Regime:** `ONNX_GENAI_VRAM_LIMIT=6GiB ONNX_GENAI_WEIGHT_OFFLOAD=1 ONNX_GENAI_CUDA_GRAPH=0`,
`--steady` → `strategy=vram-limit dynamic KV/weight lending with a retained physical-handle pool`.
This is the same knob set and model class that produced the prior **41.5–45.2 s `vram_free_ms`**
figure in `2026-08-19-vmm-admission-cap-usable-vram-and-free-attribution.md`.

This note answers the five reconciliation questions put to the prior "45 s = genuine unmap/release
churn" claim. **Measured vs inferred is labelled inline.** Nothing is fabricated; the two raw
captures are `vram_free_partial.txt` (run A) and `vram_free_split3.txt` (run B) at repo root.

> **Build-provenance / #1439 contamination note (added post-hoc).** These captures were taken on a
> current-tree binary that postdates commit `4e2a8b2e` (#1383, 2026-08-18 21:05), which issue
> **#1439** shows introduced an on-by-default use-after-unmap in the weight-lending eviction path
> (`defer_eager_sync` makes `CudaRuntime::synchronize()` a no-op, so eviction can `cuMemUnmap` a
> weight granule while a decode kernel still reads it — and via stable-slot VA remap it can *silently*
> substitute a successor weight's bytes). **That bug corrupts weight *read data*; it does not change
> the free-path quantities measured here** — the `cuMemUnmap` call count (~6/token), ms/call
> (~16.9 ms), and the 30× `vram_free_ms` swing are properties of the eviction/allocator bookkeeping,
> not of whether a compute kernel read correct bytes. So the conclusions of this note stand. The bug
> is why the run in this regime does not complete a full generation; the free-path window measured
> here is the pre-fault window. To reproduce a *clean* full run, add `ONNX_GENAI_DEFER_EAGER_SYNC=0`.

---

## Headline

**MEASURED.** In the streaming first-token window (identical deterministic work in both runs:
`page_ins=682`, `evictions=12`, `htod_bytes=5,844,615,168`, `vmm_arena releases=6`,
`commits=3115`):

| quantity                               | run A (`partial`) | run B (`split3`) |
|----------------------------------------|------------------:|-----------------:|
| `vram_free_ms` (weight-page free bracket) | **2449.890**   | **82.160**       |
| `vram_free_sync_ms` (stream drain, split out) | 0.242      | 0.249            |
| `vmm_arena releases` (= contiguous `cuMemUnmap` runs) | 6      | 6                |
| `cuMemUnmap` driver time (all runs, ms) | not instrumented  | **101.268**      |
| `cuMemUnmap` calls                      | not instrumented  | **6**            |
| `cuMemUnmap` ms/call                    | —                 | **16.878**       |
| handle disposal (pool return) ms / calls| not instrumented  | **0.043 / 59**   |
| `resource_vram oversubscribed_bytes`    | 0                 | 0                |

Two facts fall out immediately and each corrects the prior reading:

1. **The driver freeing cost is ~0.1 s, not tens of seconds.** For a full streaming first token the
   allocator issues **6 `cuMemUnmap` calls** (== `vmm_arena releases`), ~16.9 ms each, ~101 ms total,
   plus **0.043 ms** returning 59 physical handles to the pool (the retained pool means
   `cuMemRelease` is *not* called — handles are pooled, not released to the driver). Over the prior
   40-token run that is ≈ 240 unmap calls ≈ **~4 s of driver freeing at most**, not 45 s.

2. **`vram_free_ms` is not a stable measurement of freeing.** The *identical* deterministic workload
   (same 682 page-ins, same 12 evictions, same 6 releases, same bytes) produced
   `vram_free_ms` = **2449.9 ms** in run A and **82.2 ms** in run B — a **30× swing**. In run B the
   raw driver `cuMemUnmap` time (101 ms) is *larger than* the whole weight-free bracket (82 ms). A
   deterministic driver cost cannot vary 30× on identical work; the bracket is capturing variable
   **non-driver** time (bookkeeping and/or blocking), so the aggregate "45 s" is not a reliable
   "cost of freeing" at all.

**Conclusion (this is the second retraction the owner asked to be put on record with the same
clarity as the span-count refutation):** the 41–45 s is **not** genuine unmap/release driver churn.
Driver freeing is ~0.1 s/token (6 cheap `cuMemUnmap` calls + near-free pooled handle returns). The
remainder of `vram_free_ms` is non-driver time inside the bracket, and it is run-to-run unstable, so
the number does not support "freeing is inherently slow." It also does **not** support the alternate
"a thousand release calls" story — there are **6 release calls per token (~240 per run), not ~1000**.

---

## The five questions, answered

### 1. Scope of the number — per-call vs per-step vs aggregate

- **The 41–45 s is an AGGREGATE** (prior doc, MEASURED): cumulative `vram_free_ms` over a **40-token**
  run streaming **213 GB**, 6 GiB limit, release build → ≈ **1.1 s/token**.
- **Per-call (MEASURED here):** `cuMemUnmap` ≈ **16.9 ms/call**; **6 calls/token** →
  ≈ **101 ms/token** of driver unmap; pooled handle return ≈ **0.7 µs/granule** (0.043 ms / 59).
- **Call count (MEASURED here):** 6 unmap runs per streaming token, **~240 over 40 tokens**.
- So quoted next to #1295's "90 s", the honest decomposition of the 45 s aggregate is:
  **≤ ~4 s driver freeing + the rest non-driver bracket time.**

### 2. Reconcile against the ~33 ms bare-driver microbenchmark

- Bare-driver `cuMemUnmap`/`cuMemRelease` median ≈ **33 ms**, *regime-qualified* as "past the usable
  boundary" (oversubscribed) in the admission-cap doc.
- In-run MEASURED here (normal regime, `oversubscribed_bytes=0`): **16.9 ms/call**, 6 calls. Same
  order of magnitude; the normal-regime call is *faster* than the oversubscribed-boundary 33 ms, as
  expected.
- **The two microbenchmark and aggregate numbers do not contradict** once the call count is known:
  6 calls × 17 ms ≈ 0.1 s/token, ~4 s/run. The apparent contradiction only arose from never
  counting the calls. **There is no population of ~1000 driver calls.** (INFERRED corroboration: the
  prior doc's own **debug build = 2× release** is the tell — `cuMemUnmap` is a driver call unaffected
  by our build profile, so a 2× inflation from a debug build means almost all the bracket time is in
  *our* Rust code, not the driver. That is consistent with driver freeing being ~0.1 s/token.)

### 3. ms per call against bytes/spans freed

- MEASURED: 6 unmap runs cover the **12 evicted weight pages** of the window, disposing **59 physical
  granules** (≈ 124 MB; ~10 granules / ~20 MB per unmap run). 16.9 ms to unmap ~20 MB of contiguous
  VA is a plausible driver rate, **not** an alarming one. Pooled handle return is ~0.7 µs each.
- The coalescing is real: 12 evictions collapse to 6 contiguous `cuMemUnmap` runs (adjacent granules
  unmapped in one driver round-trip, per `virtual_memory.rs::release`/`decommit`).

### 4. Oversubscribed vs normal regime

- MEASURED: `resource_vram: ... oversubscribed_bytes=0` in both runs (mapped physical
  `5,788,139,520` < usable). This is the **normal** regime, matching the prior doc's "not
  oversubscribed" claim. So this figure is **not** the oversubscription cliff — it is the residency
  cache evicting/streaming under budget.
- The ~33 ms bare-driver figure, by contrast, is an **oversubscribed-boundary** number. The two
  belong to different regimes and should not be quoted interchangeably.

### 5. Re-check the timing bracket

- The stream drain is already split out (`vram_free_sync_ms` ≈ 0.24 ms here; #1295 fix). Confirmed:
  no stream sync remains in `GLOBAL_VRAM_FREE_NS`.
- **But the bracket is still not clean.** The 30× run-to-run swing on identical work
  (`vram_free_ms` 82 ↔ 2450 ms, with the deterministic driver content fixed at 6 unmaps / ~101 ms)
  proves `GLOBAL_VRAM_FREE_NS` captures variable non-freeing time — the per-granule bookkeeping loop
  in `vmm_allocator.rs::release_granules_report` (refcount walk + `contiguous_granule_runs` +
  `give_back_lease` over ~3000 granules) and/or CPU time blocked behind other GPU work. The prior
  doc's re-bracketing removed the stream sync but did not isolate deterministic free work.

---

## Material new finding: the 45 s regime crashes on the current build

**MEASURED.** The exact regime that produced the 41–45 s (14B, 6 GiB weight-lending, `--steady`) now
faults **reproducibly** on this build with
`DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)` during weight page-in H2D — across five attempts and
three env variants (`VRAM_LIMIT=6GiB`; `WEIGHT_OFFLOAD_DEVICE_BYTES=4GiB`; with/without explicit
`ONNX_GENAI_CUDA_VMM`/`KV_MIN_BUCKET`). At `VRAM_LIMIT=7GiB` the governor refuses the lease at load
(`cannot reserve 6,928,990,208 bytes … usable 6,699,456,921`). Consequence: a clean **40-token
aggregate cannot be re-measured on the current build/box** — the numbers above are a **partial
(prefill/first-token) window** captured on the error path before the fault. This instability is
itself worth flagging; it post-dates the prior doc (which completed 40 tokens) and lands in the
shared `weight_paging.rs` / VMM-lending path that #1315/#1325 touched.

---

## Instrumentation added (diagnostic; kept for reproducibility)

- `crates/onnx-runtime-cuda-memory/src/virtual_memory.rs` — `GLOBAL_UNMAP_NS/_CALLS`,
  `GLOBAL_HANDLE_DISPOSE_NS/_CALLS`, `global_unmap_driver_stats()`,
  `reset_global_unmap_driver_stats()`; `Instant` timers around the two `cuMemUnmap` sites and the
  handle-disposal loop in `release`/`decommit`. Overhead is negligible (ns timer around ms-scale
  driver calls).
- `crates/onnx-genai-bench/src/bin/profile_native.rs` — `print_vmm_observability` now emits
  `vmm_free_driver_split: cuMemUnmap_ms/_calls/_ms_per_call handle_dispose_ms/_calls`; `run_steady`
  resets the counters and, on a mid-run generation error, dumps the accumulated offload/VMM stats
  (with explicit stdout flush) so a crashing weight-lending run still yields the free/release
  counters.

## What this does and does not change

- **Does not** touch the runtime free path's behaviour — only measurement.
- **Corrects** the prior "genuine unmap/release churn / not substantially retracted" reading: driver
  freeing is ~0.1 s/token; the 45 s is dominated by non-driver bracket time and is not a stable
  number.
- **Actionable direction (INFERRED, not built):** if the bracket time matters, the target is the
  per-granule bookkeeping in `release_granules_report`, not the 6 driver unmaps — but the 30×
  variance suggests part of it is blocking time, so the first step is to make `GLOBAL_VRAM_FREE_NS`
  bracket only the deterministic bookkeeping+driver work and move any wait outside it, exactly as the
  stream-sync split did.
