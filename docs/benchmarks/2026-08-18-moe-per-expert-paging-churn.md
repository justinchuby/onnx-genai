# Per-expert MoE weight paging: churn is the cost, skew is the win, granule is the floor

**Date:** 2026-08-18
**Author:** Copilot (streaming slice → MoE expert paging)
**Owner directive:** "继续推进 vmm、offload、streaming、multi-request batching 对大模型的支持，
提高速度，实现简洁高效" — the "streaming" slice, clarified to mean **MoE expert-weight
streaming**, not HTTP/SSE.

**Question this answers (WEIGHT_OFFLOAD / MEMORY_MANAGEMENT_MODEL_DESIGN open item):**
before designing any MoE residency policy, and before wiring per-expert dispatch into
the executor, what does per-expert VMM paging **cost** on this box? The doc records that
dense is settled at `reads_per_step = 1.000` (no hot subset) and that MoE is the first
case a residency policy could exploit — *if* the paging layer can see the skew. Today
`bind_block_quantized_moe` pages the whole expert bank as **one key**, so a QMoE run
reports whole-bank `reads_per_step ≈ 1.0` (dense-like) and the measured router skew is
**invisible**. Per-expert paging would expose it, but at the price of more, smaller VMM
regions — and VMM `cuMemMap`/`cuMemUnmap` churn is a known binding limiter here
(`vram_free_ms` 9.7 s → 90 s once mapped physical exceeds VRAM, see
`2026-08-18-batch-n-scaling-8gb-limiters.md`). This measures that price directly.

## Hardware / method (house rule §32.2)

- **Box:** RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1), Intel i7-13800H (14C/20T).
  CUDA runtime on PATH via the anaconda `nvidia/cu13` + `cudnn`.
- **Path exercised:** the real `CudaWeightResidency` VMM arena
  (`crates/onnx-runtime-ep-cuda/src/weight_paging.rs`) — stable-VA slots (#716),
  physical granules mapped on page-in / unmapped on eviction at the **2 MiB device
  granule** (#776). Page-in = VMM map + stream-ordered H2D; eviction = VMM unmap. Same
  mechanism the live decoder would use.
- **Harness:** `crates/onnx-runtime-ep-cuda/tests/weight_offload_churn_gpu.rs`
  (`gpu-tests`, `--nocapture`). Each arm builds 32 distinct expert weights in one mmap,
  requests routed experts per step through `residency.resident(expert_id, …)`, and reads
  `residency.stats()` + `global_offload_stats()`. Warm-up step excluded from timing.
- **Routing shape:** `granite-3.0-1b-a400m` — **32 routed experts, top-8, no shared
  expert** (`inference_metadata.yaml`). Two per-step routed sequences bracket reality:
  - **uniform** — top-8 picked uniformly (churn ceiling, no cross-step reuse);
  - **skewed** — a fixed hot core of 4 experts always selected (per-step hot share
    4/8 = 0.50 ≈ the **measured** granite top-8 read share 0.45–0.54).
- **Baseline:** arm A (whole-bank, one key, resident) — today's behaviour — in the same
  binary/run (same-binary A/B).
- **Model behind the routing:** `granite-1b-a400m-f16-mobius` (the only local MoE, a real
  IBM trained router; skew independently measured — see the router-skew decision drop).
  Expert *byte sizes* are swept (not model-specific) to span the granule.
- **Note on realism:** this measures the **paging mechanism** driven by a
  realistically-sized/-routed harness, **not** an end-to-end native `BlockQuantizedMoE`
  decode. No runnable native QMoE model is available on this box and the
  executor→pager per-expert dispatch is the deferred seam (#82/#87). The H2D volumes and
  map/unmap counts are exact; the per-step *wall* time excludes decode compute, which a
  live run would add.

## Result — 200 steps, 32 experts, top-8, hot core 4

Per-step figures; "map/unmap+ovh" = wall − H2D − materialize (materialize was 0 — the
harness reads from the mmap source, no host rebuild).

| expert | arm | page-ins | unmaps | H2D (MiB) | H2D ms/step | map/unmap ms/step | wall ms/step | peak res |
|--------|-----|----------|--------|-----------|-------------|-------------------|--------------|----------|
| 2 MiB (=1 granule) | A whole-bank (resident) | 0 | 0 | 0 | 0.00 | 0.001 | 0.001 | 64 MiB |
| | B per-expert, full budget | 24¹ | 0 | 48 | 0.03 | 0.013 | 0.043 | 64 MiB |
| | C per-expert@top-k **uniform** | 1345 | 1345 | 2690 | 2.36 | 3.48 | 5.84 | 16 MiB |
| | C per-expert@top-k **skewed** | 719 | 719 | 1438 | 1.33 | 2.01 | 3.34 | 16 MiB |
| 3 MiB (=2 granules, granite f16) | A whole-bank | 0 | 0 | 0 | 0.00 | 0.001 | 0.001 | 96 MiB |
| | C@top-k uniform | 1229 | 1227 | 3687 | 3.61 | 3.26 | 6.87 | 30 MiB |
| | C@top-k skewed | 649 | 647 | 1947 | 1.98 | 1.85 | 3.84 | 30 MiB |
| 16 MiB (GLM/DeepSeek-class) | A whole-bank | 0 | 0 | 0 | 0.00 | 0.001 | 0.001 | 512 MiB |
| | B per-expert, full budget | 24¹ | 0 | 384 | 0.33 | 0.038 | 0.369 | 512 MiB |
| | C@top-k uniform | 1345 | 1345 | 21520 | 20.8 | 6.17 | 26.95 | 128 MiB |
| | C@top-k skewed | 719 | 719 | 11504 | 10.9 | 3.39 | 14.34 | 128 MiB |

¹ one-time first-touch page-ins (32 experts − 8 warm-up), then all hits.

Sub-granule row (0.75 MiB, granite **int4** expert): rounds up to the 2 MiB granule
(**2.7× physical waste**) and cannot be isolated below granule; the residency's
content-byte budget then admits more than top-8, accidentally churning less but wasting
physical. Reported for completeness, not comparable to the ≥-granule rows.

## Reading (measured)

1. **Per-expert *keying* is neutral.** Arm B (all experts resident, per-expert keys)
   costs a one-time 24 page-ins then all hits — ≤0.37 ms/step even at 16 MiB. The owner's
   "more keys, more bookkeeping" concern: measured, negligible **when the experts fit**.
2. **Per-expert *paging* churn is significant and bandwidth-dominated.** At top-8 budget
   the working set turns over every step. H2D re-streaming dominates (10.9 ms/step at
   16 MiB skewed); pure VMM map/unmap adds **3–6 ms/step** at GLM-class sizes — the
   binding-limiter churn, now quantified per-expert.
3. **Skew is a real, bankable win.** Skewed routing cuts page-ins **~46%** vs uniform,
   consistently across every expert size (719 vs 1345 etc.) — the hot core stays
   resident. 46% ≈ the independently-measured granite top-8 read share, i.e. the residency
   headroom a policy could exploit.
4. **The 2 MiB granule is a hard floor.** Experts smaller than a granule (granite int4,
   ~0.75 MiB) cannot be paged individually and waste 2.7× physical. Per-expert VMM paging
   is a **large-expert** technique (GLM-5.2 / DeepSeek-V4-class, where an expert is many
   granules), not a small-MoE one.

## Reading (inferred)

- **Arm A ("whole-bank resident, 0 churn") is not available in the target regime.** It
  only wins because the whole bank stays resident (24–512 MiB here; tens of GB for a real
  30B+ MoE — will not fit 8 GB). When offload actually engages, the honest baseline is
  *whole-bank streamed* = all 32 experts H2D per step. Per-expert@top-8 streams only
  8/32 = **25%** of that, and skew removes ~46% of the remaining page-ins. So per-expert
  paging is not merely "adds churn": it is what makes MoE offload tractable at all, and
  the skew is a second-order win on top. This is inferred from the H2D ratios, not from a
  live whole-bank-streamed decode (not run).
- Whether the per-step map/unmap + H2D (≈14 ms/step skewed at 16 MiB) is affordable
  depends on decode compute per step for the target model, which was **not** measured here
  (no runnable native QMoE model on this box).

## What this gates

- **Plumbing (per-expert keys/regions):** neutral to land on its own (arm B). The
  representation already exists (`placement.rs` `ExpertTensorLayout`); the residency layer
  already pages per key (arms B/C prove it).
- **The remaining work is structural** — wiring the router's per-step top-k output into
  per-expert `resident(expert_id, …)` calls in the executor decode loop (the deferred
  #82/#87 seam; the design doc states it "cannot be implemented as an allocator tweak").
  Returned to the owner before implementation.
- **Policy** (pin the hot core, layers 1–2 always-on experts as free hits) is sized
  against #3 above but must wait until the skew is confirmed to survive a *live*
  per-expert round trip on a model where paging engages.
