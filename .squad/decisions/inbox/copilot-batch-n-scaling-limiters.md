### 2026-08-18: Batch-N large-model scaling — 1/N HtoD amortization is REAL; wall-clock is capped by two independent limiters (VMM churn + an M≥2 decode-GEMM cliff)

**By:** Copilot (multi-request batching slice)

**Hardware (on every number):** RTX 4060 Laptop 8 GB, i7-13800H (14C/20T),
CUDA 13.1, native CUDA EP, greedy device-argmax, byte-identical uniform-token
sweep (`--native-decode-batch-sweep`), medians + range (wall time is noisy under
streaming pressure, #863). Instrument landed in PR #1291.

---

#### MEASURED

**1. `qwen14b-zp`, `ONNX_GENAI_VRAM_LIMIT=6GiB` (weight streaming engaged, budget≈5.85 GB):**
- `htod_bytes_per_token` = **5.11 / 2.61 / 1.34 / 0.72 GB/tok** for N = 1/2/4/8.
  Tracks **1/N** within the #866 elastic offset; `htod_bytes` *per step* stays
  ~flat (5.11 → 5.79 GB). → **The 1/N weight-stream amortization is confirmed by
  a contention-invariant deterministic counter.** Each weight IS read once per
  step regardless of N.
- Wall `aggregate_tok_s` = **0.49 / 0.87 / 0.93 / 0.80** for N = 1/2/4/8.
  Climbs N=1→2, saturates N≈2–4, **regresses at N=8**.
- `vram_free_ms` explodes **9.7 s → 90 s** as `mapped_physical_bytes` (8.39 GB
  @ N=8) exceeds physical VRAM (8.19 GB). → On 8 GB, **N_max ≈ 4–5**, far below
  the doc's projected `N_max ~ 19 @ 2048 ctx` (#884/#891).

**2. `qwen05b-q4` (fully resident, htod_bytes=0, no VMM churn):**
- `median_ms_per_step`: N=1 **2.68 ms**; N=2 **71–101 ms**; N=4/8/16 **~96–100 ms
  (flat)**. `aggregate_tok_s`: **373 / ~28 / 40 / 84 / 166** — batch-16 aggregate
  (166) is **below** batch-1 (373).
- Split timing (per step): N=1 forward **0.2 ms** (async graph replay) + readback
  2.3 ms; N=2 forward **50–100 ms (BLOCKING)** + readback 2.7 ms. → **The ~33×
  per-step cliff is entirely in the forward, not the argmax readback.** Readback
  is identical for N=1 and N=2.

**3. Code-level root cause (`crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`):**
- `m == 1` → specialized decode **GEMV** family (lines 6386 / 6657): wide-load,
  fp16, split-K, interleaved-dequant, fused int4 dequant-in-register.
- `m > 1` → **prefill tiled GEMM** (line 6987; file header line 2: "block-wise
  dequantization and f32 cuBLASLt GEMM fallback used for prefill"). Batch decode
  (M≥2) takes this path **every step**.

---

#### INFERRED (labelled)

- The **flat ~100 ms for M=2..16** (vs 33× above M=1) is consistent with the
  prefill GEMM's cost being dominated by its N×K grid pass with a tile height
  that pads small M — so M=2 costs ~the same as M=16 until M exceeds the tile.
  I.e. batch decode pays a fixed, M-independent full-weight-grid pass per step,
  wasting occupancy at decode M. **Not yet confirmed by nsys**; consistent with
  the dispatch + the split-timing + the flat-M signature.
- On the **8 GB + 14B streaming target**, this compute cliff is *hidden* behind
  ~2000 ms/step of HtoD streaming; the binding limiter there is **VMM map/unmap
  churn** (`vram_free_ms`), which is the **offload/VMM agent's lever**, not
  batching. The batch path's own compute cliff would only surface as the
  wall-clock limiter once the model is **resident** (small models today; large
  models that fit in larger VRAM).
- The compute cliff is **compatible-to-fix under streaming**: the weight is
  materialized in VRAM once per step (htod 1/N preserved); a batched-decode
  kernel reads that resident weight M times from VRAM, which is *not* extra HtoD.
  So fixing it does not break the confirmed 1/N mechanism.

---

#### PROPOSAL — held for owner sign-off (structural: touches a CUDA kernel dispatch)

The honest answer to "does batch-N deliver 1/N on a streaming-bound large model
today?" is: **the data mechanism yes, the wall-clock no**, for two *independent*
reasons that must not be conflated:

- **(A) Streaming regime (8 GB + 14B):** VMM `vram_free` churn caps N_max≈4–5.
  This is the **offload/VMM agent's** territory (keep granules mapped / cut
  unmap churn once mapped_physical ≥ physical VRAM). **Flagging for the
  coordination the owner offered** — not a batching change.
- **(B) Resident regime (any model that fits):** the M≥2 decode-GEMM cliff. The
  smallest change that moves the number: for small M (2..~8) dispatch batch
  decode to the existing **M=1 GEMV looped/broadened over M rows** instead of the
  prefill GEMM — restores per-row throughput toward the M=1 rate and near-linear
  aggregate scaling, preserves htod 1/N. The optimal (larger) version is a true
  **multi-row decode GEMV** (one threadblock emits M outputs per N-tile, reading
  each weight once for M rows) — the kernel-level realization of the doc's 1/N.

**简洁高效 caution:** proposal (B) does **NOT** move the specific 8 GB + 14B
streaming number (that's VMM-bound, limiter A). It moves the number for
**resident** batch-N. Per the owner's rule against "a large refactor that does
not move the measured number," I am **not building the kernel** until the owner
picks the regime to target. I have no resident large model on this box
(models dir is 0.5B and 14B variants only), so a resident-large data point to
size (B)'s payoff would need a mid-size model or a bigger-VRAM box.

**Do NOT** build the device-sampling producer (killed, #1282). No DRY guard
touched; native/ORT decode loops remain shared (the 3 `batched.rs` sites are
constructor selection).
