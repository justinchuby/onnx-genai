# The admission cap: source the mapped-physical ceiling from *usable* VRAM, and stop mis-attributing stream-drain to "free"

**Date:** 2026-08-19
**Author:** Copilot (VMM/offload slice)
**Owner directive:** "继续推进 vmm、offload、streaming、multi-request batching 对大模型的支持,
提高速度,实现简洁高效" — the **vmm/offload** slice.
**Issue:** #1295 — "Streaming-regime batch-N ceiling is VMM map/unmap churn, not the
batching path (N_max~4-5 vs projected ~19 on 8GB)". This is the **fix** phase; the
**characterisation** is `2026-08-18-batch-n-streaming-cliff-wddm-oversubscription.md`.
**Companion:** `2026-08-18-moe-per-expert-paging-churn.md` (#1308, per-expert MoE slice —
shares the single cap built here).

## What this delivers

Two things the owner greenlit, kept honest and separated **measured** from **inferred**:

1. **A combined admission cap** over the sum of mapped physical (weights + KV + activations),
   sourced from **usable** (`cuMemGetInfo` *free*) VRAM at a default **0.90**, routed through the
   **existing** device-authority ledger — no parallel mechanism. It converts a catastrophic driver
   cliff into a graceful governed refusal. **It does not raise `N_max`; it makes the failure mode a
   plateau instead of a crash.** Said plainly so the next reader does not expect the ceiling to move.
2. **A fix to the `vram_free_ms` instrument** that had the stream synchronisation *inside* the free
   bracket, so "free" time could absorb paging-slowed compute/copy. Corrected and re-measured.

## Hardware / method (house rule §32.2)

- **Box:** RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1, **WDDM**), Intel i7-13800H (14C/20T).
  CUDA 13 runtime on PATH via the anaconda `nvidia/cu13` + `cudnn` wheels. Native CUDA EP.
  **Shared box** (build/other agents run concurrently), so `cuMemGetInfo` *free* is dynamic and was
  observed between **~7100 and ~7959 MiB** of a nominal **8188** across runs — itself part of the
  finding: *usable* is not nominal, and it moves.
- **Model:** `qwen14b-zp` (Qwen2 14B, int4, 48 layers, ~7.8 GB weights on device).
- **Baseline / reference:** pre-change behaviour is the device ceiling resolved from **nominal
  total** (`Fraction(0.90)` of 8188 MiB = 7369 MiB); the fix resolves from **measured free**.
- **Build:** `--release` for every headline number (a debug build inflates our-side per-granule
  bookkeeping ~2×, see below). Feature set `bench-native,cuda`. Single process, single thread.
- **Forced-growth recipe:** `ONNX_GENAI_CUDA_VMM=1 ONNX_GENAI_CUDA_GRAPH=0 ONNX_GENAI_KV_MIN_BUCKET=8`.
- **Tests:** `onnx-genai-engine` lib, governor module — **13 passed, 0 failed, 0 ignored**
  (3 new). The GPU decode runs below are functional runs, not the `cuda,native-backend` unit suite;
  pre-existing failures out of scope here are #1284 (three harness) and #1305 (four kernel).

---

## Part 1 — The cap: driver crash → governed refusal (measured)

### The bug the cap fixes

The device-authority ledger already bounds the **sum** of every Device-tier lease — weight
residency admits against `governor.available(Tier::Device)` (weight_paging.rs), and KV/activation
mapped growth admits through `prepare_mapped_growth` against the same tier. So a single ceiling is
already the shared authority the owner asked MoE to reuse. What was wrong was its **source**: a
`Fraction` resolved against **nominal total** VRAM. On this box that is 0.90 × 8188 = **7369 MiB** —
which **exceeds the measured usable free** (≈7100 MiB on the shared box this session; ≈7959 MiB
idle). A ceiling *above* usable means the ledger permits leases the driver cannot satisfy, so the
**driver** faults first.

### Measured: before (nominal-sourced ceiling)

Loading `qwen14b-zp` under VMM with the mapped ceiling at-or-above usable, the driver refuses:

```
cannot reserve 12582912 bytes of device memory:
  cuMemMap: cuMemSetAccess failed: CUDA_ERROR_OUT_OF_MEMORY (os error 2)
```

That is the cliff in its most brutal form — a raw driver `CUDA_ERROR_OUT_OF_MEMORY`, not a
recoverable signal. (Reproduced across the no-limit batch-1 load and the governed batch-4/8
sweeps, which all OOM at load on this 8 GB box: the 14B's 7.8 GB of weights alone brush usable.)

### Measured: after (usable-sourced cap, default 0.90)

Same load, same binary, cap sourcing the ceiling from **measured free × 0.90**:

```
cannot reserve 33554432 bytes of device memory for Workspace { step_scoped: false }:
  6673137664 of 6699456921 bytes are already leased, leaving 26319257;
  free memory by closing sessions, lower the demand, or raise the device limit
```

- Ceiling = **6699456921 B** = floor(measured_free × 0.90).
- The refusal is a **clean, governed `MemoryError`** that names the tier, the shortfall, and three
  remedies — an actionable ceiling, not a crash.

### Measured: the 0.90 headroom is what keeps admission off the boundary

Same binary, `ONNX_GENAI_VMM_MAPPED_FRACTION=1.0` (ceiling = measured free, **no** safety margin):

```
cannot reserve 12582912 bytes ... for Workspace:
  7440695296 of 7443841024 bytes are already leased, leaving 3145728; ...
```

- Ceiling = **7443841024 B** = measured free exactly. Still a **governed** refusal (good — sourcing
  from free, not total, is what moves the refusal from the driver to the ledger), but it admits
  right up to the usable edge (leaves 3 MB), exactly where #1295 measured the WDDM fault-in cost
  turning unpredictable.
- The default **0.90** stops admission ~740 MB earlier — the margin that keeps mapped physical off
  the fault-in cliff.

### Measured: the cap admits the runnable config

The cap is not blanket refusal. With a streaming weight limit (`ONNX_GENAI_VRAM_LIMIT=6GiB`, so
mapped weights ≈ 5.9 GB budget, well under the 0.90 cap), the same 14B **loads and decodes**:

```
throughput=0.37 tok/s   mapped_physical=5.31 GB   budget=5.90 GB   (batch=1, release)
```

So the cap **admits what fits and refuses what would oversubscribe** — the cliff→plateau
conversion, at the admission layer, measured live.

### Inferred (not measured live)

- A live **multi-request N=8 throughput plateau** could not be captured: the governed batch-N
  sweep **OOMs at load** on this 8 GB box for N ≥ 4 (the 14B's weights alone ≈ usable VRAM), the
  same oversubscription the cap addresses, one layer earlier than sequence admission. The plateau
  mechanism — the identical ledger refusal firing on the (N+1)th sequence's KV/activation growth —
  is proven by the unit test `device_authority_ceiling_is_usable_free_and_refuses_growth_past_it`
  and by the live load-time refusal above; the specific 0.80→peak tok/s curve #1295 wanted is
  **not** measurable here and is not claimed.

### The fraction is derivable and hardware-recorded, not magic (cf #1261)

`DEFAULT_VMM_MAPPED_FRACTION = 0.90`, override `ONNX_GENAI_VMM_MAPPED_FRACTION` (finite, `(0,1]`).
The concrete ceiling is `measured_free × fraction`, **recomputed per device at load** from the
driver's own query — never a machine-specific constant. The 0.90:
- **Protects against** the WDDM fault-in cliff: #1295 measured its onset at ~0.97× nominal ≈ the
  usable boundary, past which `vram_free` cost is unpredictable (median ~33 ms with
  non-deterministic multi-hundred-ms storms). Sourcing from *free* already excludes the standing
  desktop-compositor reserve (~229 MiB idle here); the remaining 10 % is margin below the onset.
- **Would justify changing:** raise toward 1.0 on a headless box (no compositor reserve, steadier
  WDDM paging); lower it if a device shows eviction storms *below* the boundary.

---

## Part 2 — The `vram_free_ms` instrument: corrected, and the headline mostly survives

### The bug

`weight_paging.rs::release_allocation(true)` synchronised **both** streams *inside* the
`GLOBAL_VRAM_FREE_NS` bracket. Under streaming/oversubscription that in-flight work is
PCIe-fault-slowed, so "free" absorbed paging-stalled compute/copy — the same class as the retired
`h2d_enqueue_copy_ms` bug (measurement-discipline #2). #1295's 90 s headline drove the whole
investigation, so an instrument that mis-attributes other work to "free" would mislead the next
reader exactly as it nearly misled us. Fix: the stream drain is timed **separately**
(`GLOBAL_VRAM_FREE_SYNC_NS`, surfaced as `vram_free_sync_ms`); the `vram_free_ms` bracket now spans
only unmap/release/free.

### Measured: corrected split (release, batch=1, 6GiB limit, streaming 213 GB / 40 tok)

| build         | vram_free_ms | vram_free_sync_ms | sync share of bracket | tok/s |
|---------------|-------------:|------------------:|----------------------:|------:|
| release (n=1) | 45193.091    | 1893.232          | ~4.0 %                | 0.37  |
| release (n=2) | 41487.582    | 1628.175          | ~3.9 %                | 0.41  |
| debug         | 105246.709   | 7228.884          | ~6.4 %                | 0.19  |

Two release samples (same box, same 6GiB-limit streaming regime, single-thread) give
`vram_free_ms` = 41.5–45.2 s with `vram_free_sync_ms` = 1.6–1.9 s — the sync share is a stable
~4 %, not a run-to-run artefact. The corrected free time is genuinely tens of seconds in both.

- **The stream-drain that used to be inside "free" is only ~4 % (release) of the bracket.** The
  corrected `vram_free_ms` (45.2 s) still dominates: it is **genuine unmap/release churn**, not
  waiting on paging-slowed work. **The headline is not substantially retracted.** An honest
  instrument was still worth building — but the correction refines the number, it does not overturn
  the conclusion.
- **This run is *not* oversubscribed** (mapped 5.31 GB < usable): the 45 s is the residency cache
  evicting 213 GB of weight pages across 40 steps to hold the 5.9 GB streaming budget — ≈1.1 s/token
  of real unmap/release tax that the 1/N amortisation must overcome, present *before* any
  oversubscription. #1295's 90 s figure is the same phenomenon at N=8 with oversubscription added.
- Debug inflates absolute free ~2× (per-granule bookkeeping over ~2500 granules/step), which is why
  headline numbers here are release-only.

---

## Cross-slice notes

- **#1308 (per-expert MoE):** the cap is a **single shared authority** — the device-tier ceiling
  bounds weights + KV + activations together, so MoE's per-expert regions admit against the same
  ledger with nothing to duplicate. Confirmed earlier: per-expert paging does **not** worsen this
  cliff (span count is not the driver) and *lowers* peak mapped physical.
- **OS-paging / residency-hint direction:** this session did not add hints, but the measurements
  bear on it. The refusal moved from the **driver** to the **ledger** purely by sourcing the ceiling
  from *free* rather than *total* — i.e. once the ceiling is ≤ usable, the driver is never asked for
  memory it lacks. That is evidence that the binding constraint is **usable VRAM**, not our
  bookkeeping: refusing to oversubscribe is correct whoever does the paging. If WDDM is to be
  *hinted* to page instead, the admission cap remains the correct floor under that architecture too.

## Files

- `crates/onnx-genai-engine/src/engine/governor.rs` — the cap: `clamp_ceiling_to_usable_vram`,
  `usable_mapped_safe_fraction`, `DEFAULT_VMM_MAPPED_FRACTION`, applied at the device-ceiling
  derivation; 3 new tests.
- `crates/onnx-runtime-ep-cuda/src/weight_paging.rs` — `GLOBAL_VRAM_FREE_SYNC_NS`, the
  `release_allocation` re-bracketing, `vram_free_sync_ns` on `GlobalOffloadStats`.
- `crates/onnx-genai-engine/src/native_decode/cuda.rs`,
  `crates/onnx-genai-bench/src/bin/profile_native.rs` — surface `vram_free_sync_ms`.
