# MLAS vs Native CPU EP on Apple Silicon — Strategy Decision

**Author:** Sebastian (Performance Engineer)
**Date:** 2026-07-27 (Q6 added same day)
**Status:** RECOMMENDATION — awaiting Justin's decision
**Requested by:** Justin Chu

---

## 🔥 BATCH DECODE: A Bigger Strategic Opening Than Single-Stream

**Date:** 2026-07-27T08:48 | **Load:** 4.2–5.9/10 cores | **Corroborated:** mach_absolute_time + clock_gettime (agreement <1%)

> **⚠️ Correction (2026-07-27T08:59):** Commit `ad920725` title states "15× advantage over ORT at B=32". This is **overstated**. The BNNS side (1663 tok/s at B=32) is measured and corroborated. The ORT batch-decode throughput has **not been measured** — the compare harness has no batch support, and we have not yet driven ORT directly with a batched input. The "~108 tok/s" ORT estimate in the table below is derived indirectly (assumed ~120 GFLOPS for MLAS multi-threaded NEON, applied to the full-model FLOPs) and has not been validated. The **mechanism** is sound — ORT's CPU EP does not link Accelerate (`otool -L` confirmed), so it cannot reach AMX — but the specific ratio is unquantified until ORT batch decode is measured. All ORT batch figures below are marked **(est.)**.

Justin asked: *"What about batch decode? Can we optimize that together?"* Answer: **batch decode shifts the workload into compute-bound territory where BNNS/AMX dominates and ORT structurally cannot follow. Our measured BNNS throughput at B=32 is 1663 tok/s. ORT's batch throughput is unmeasured; the structural argument strongly favours us but the ratio is unquantified.**

### The roofline shifts with batch size

Batch decode (M=B) reuses weights across B sequences, raising arithmetic intensity:

| B | AI (f16 weights) | Bandwidth ceiling | Compute ceiling | **Binding** |
|---|---|---|---|---|
| 1 | 1.0 | 399 GFLOPS | 2500 GFLOPS | **BANDWIDTH** |
| 2 | 2.0 | 796 | 2500 | **BANDWIDTH** |
| 4 | 4.0 | 1585 | 2500 | **BANDWIDTH** |
| 8 | 7.8 | 3139 | 2500 | **COMPUTE** |
| 16 | 15.4 | 6160 | 2500 | **COMPUTE** |
| 32 | 29.7 | 11874 | 2500 | **COMPUTE** |

Ridge point on M1 Max (400 GB/s, ~2500 GFLOPS): **B ≈ 6–7**. On M1 Air (~100 GB/s, ~2000 GFLOPS): **B ≈ 10**. Above the ridge, more batch size = more AMX utilization = bigger advantage over NEON-only ORT.

### Measured: BNNS vs sgemm at batch decode shapes

Per-step time includes **all 121 MatMul calls** (5 per layer × 24 layers + lm_head), measured per-call to capture real dispatch overhead. Qwen2.5-0.5B shapes.

| B | sgemm per-step | BNNS per-step | Ratio | Per-token (BNNS) | Tok/s (BNNS) | BNNS GFLOPS |
|---|---|---|---|---|---|---|
| 1 | **48.9 ms** | 203.2 ms | sgemm 4.2× | — | — | 5 |
| 2 | 46.3 ms | **22.3 ms** | BNNS 2.1× | 11.15 ms | **90** | 89 |
| 4 | 47.0 ms | **22.9 ms** | BNNS 2.1× | 5.73 ms | **175** | 173 |
| 8 | 43.7 ms | **22.2 ms** | BNNS 2.0× | 2.78 ms | **360** | 355 |
| 16 | 43.8 ms | **17.6 ms** | BNNS 2.5× | 1.10 ms | **907** | 896 |
| 32 | 34.4 ms | **19.2 ms** | BNNS 1.8× | 0.60 ms | **1663** | 1643 |

**NEON blocked GEMM (half_gemm.rs, single-threaded) at B=32: 528 ms per step → 60 GFLOPS. BNNS is 28× faster.**

### The dispatch-overhead trap did NOT materialize

Justin flagged: 49 calls × ~40 µs ≈ 2 ms overhead per step. Actual measurement: BNNS per-call overhead is ~50 µs for small ops (QKV, O_proj) and ~130–200 µs for large ops (Gate/Up/Down), but the AMX compute within those calls is doing useful work. The total BNNS time (17.6–22.3 ms at B≥2) is well below sgemm (34.4–47.0 ms). The overhead is real but the throughput advantage overwhelms it.

Per-call detail at B=8 (corroborated):

| Op | K | N | sgemm µs | BNNS µs | Winner |
|---|---|---|---|---|---|
| QKV | 896 | 1152 | **24** | 50 | sgemm (overhead dominates small op) |
| O_proj | 896 | 896 | **20** | 40 | sgemm (same) |
| Gate | 896 | 4864 | 446 | **199** | **BNNS 2.2×** (AMX throughput dominates) |
| Up | 896 | 4864 | 445 | **201** | **BNNS 2.2×** |
| Down | 4864 | 896 | 541 | **202** | **BNNS 2.7×** |
| lm_head | 896 | 151936 | 8289 | **5640** | **BNNS 1.5×** |

Pattern: BNNS loses on small ops (~50 µs fixed overhead), wins big on large ops (AMX throughput). Large ops dominate the total (Gate+Up+Down = 72 calls of 121).

### ORT batch decode: MEASURED (2026-07-27T09:02, load 18–23/10 cores ⚠️)

ORT batch decode was measured directly using `onnxruntime` 1.27.0 Python API against `models/qwen2.5-0.5b/model.onnx` (f32), CPUExecutionProvider, 200 iterations × 3 runs, median reported. **The machine was heavily contended** (load 18–23), which disproportionately affects ORT (CPU-threaded) more than BNNS (AMX coprocessor). B=1 cross-check: 40.2 tok/s measured here vs 46.01 in quiet compare harness = 0.87×, suggesting ~13% load penalty on ORT.

| B | Our BNNS (measured, load 4–6) | ORT (measured, load 18–23 ⚠️) | Raw ratio | Notes |
|---|---|---|---|---|
| 1 | N/A (GEMV: 60 tok/s) | 40.2 tok/s (46 quiet) | **1.4× (quiet)** | Both sides measured in quiet conditions |
| 2 | 90 tok/s | 73.4 tok/s | 1.2× | ORT load-penalized |
| 4 | — | 148.2 tok/s | — | BNNS B=4 not measured separately |
| 8 | 360 tok/s | 224.6 tok/s | 1.6× | ORT B=8 agreement poor (11.6%) ⚠️ |
| 16 | 907 tok/s | 292.4 tok/s | 3.1× | |
| 32 | 1663 tok/s | 345.3 tok/s | 4.8× | ORT B=32 spread 50.8% ⚠️ |

**⚠️ The BNNS and ORT measurements were taken under different load conditions.** The BNNS numbers (load 4–6) are more reliable. The ORT numbers (load 18–23) are depressed by ~13% (B=1 cross-check). Even load-adjusted (÷ 0.87), ORT B=32 ≈ ~400 tok/s → ratio ≈ 4.2×. **The real advantage at B=32 is approximately 4–5×, not 15× as previously estimated.** The 15× estimate was based on assuming MLAS NEON peaks at ~120 GFLOPS; actual ORT (with graph fusion, thread pool, and MLAS combined) achieves ~185 GFLOPS effective at B=32 shapes.

**My earlier estimate of ~108 tok/s for ORT at B=32 was off by ~3×.** The error came from estimating MLAS's isolated kernel throughput (~120 GFLOPS) while ignoring that ORT's graph fusion and thread pool add substantial value at batch shapes. ORT runs fused subgraphs where we run 434 individual ops — at B=32, ORT's fusion advantage compounds. This is exactly the kind of error the corroboration rule was designed to catch.

**Corrected assessment:** Batch decode favours us at B≥2, with the advantage growing from ~1.2× (B=2) to ~4–5× (B=32). This is a real and significant advantage — but it is 4–5×, not 15×, and ORT is not standing still (graph fusion gives it a substantial efficiency edge that partially offsets MLAS's NEON-vs-AMX throughput deficit).

**What remains true:**
- ✅ Our BNNS: 1663 tok/s at B=32 (measured, load 4–6, corroborated)
- ✅ ORT: ~345 tok/s at B=32 (measured, load 18–23, 3 runs but high spread)
- ✅ ORT does not link Accelerate (verified, `otool -L`)
- ✅ Advantage grows with B (mechanism: compute-bound regime favours AMX)
- ⚠️ Both sides should be re-measured **under identical load conditions** before final publication

### Three-regime dispatch rule

| Regime | Condition | Kernel | Why |
|--------|-----------|--------|-----|
| Single-stream decode | M=1 | `neon_gemv_f16_col_parallel` | BW-bound, multi-threaded columns, reads f16 directly |
| Batch decode / prefill | M≥2, macOS | `BNNSMatMul` f16→f32 | AMX, 90–2450 GFLOPS, scales with B |
| Fallback | M≥2, non-Mac | `half_gemm.rs` NEON | Portable, ~50–160 GFLOPS multi-threaded |

The threshold is **M=2** — the same for batch decode and prefill. No separate batch-decode threshold needed, because BNNS's per-call overhead (~50 µs) is already absorbed at M=2. **Runtime query:** `geom.m >= 2 && cfg!(target_os = "macos")` — no per-chip calibration needed.

**On low-bandwidth chips (M1 Air, ~100 GB/s):** the ridge point is higher (~B=10), but BNNS still wins at B=2 because fp16 input halves bandwidth and AMX is available on all Apple Silicon. The GFLOPS numbers scale down proportionally, but the relative advantage over NEON-only (ORT) is the same.

### Batch decode design compatibility (Q5)

| Component | Batch-friendly? | Notes |
|-----------|----------------|-------|
| **SPMD decode pool** | ✅ Neutral | Not used for BNNS MatMul (BNNS has own threading). Still useful for non-MatMul ops (RMSNorm, SiLU). |
| **transposed_b_f16 prepack** | ✅ Helps more at batch | Transpose amortizes across B sequences. At B=1, one GEMV reuses the transpose. At B=32, 32 sequences reuse it. Cost is unchanged (one-time lazy init). |
| **434 ops/token dispatch** | ✅ Amortizes | 434 dispatches × ~2 µs = ~0.87 ms/step. At B=32: 0.87/32 = 0.03 ms/token. Dispatch overhead becomes negligible. |
| **Continuous batching scheduler** | ✅ Exists | `max_batch_size: 32` (default). Scheduler already manages batch formation. |
| **half_gemm.rs at small M** | ⚠️ Wrong for Mac | MR=4 tiling at B=2: 50% tile utilization, single-threaded for M<8. But on Mac, BNNS supersedes it entirely. |
| **BNNS threading** | ⚠️ Do NOT call from Rayon | Same constraint as cblas_sgemm. BNNS call must be from dispatch level, not inside par_iter. |

**Nothing in the current design is hostile to batching.** The SPMD pool, prepack cache, and scheduler all work correctly at B>1. The only fix needed is routing: `try_matmul_half` should fall through to BNNS at M≥2 on Mac (same fix as the prefill dispatch).

### Does ORT batch-decode well?

**Cannot measure directly.** The compare harness (`compare.rs`) has no `--batch` flag and does not support concurrent sequence generation. ORT's scheduler (if any) is internal to the ORT runtime and not exposed via our harness.

Structurally: ORT uses MLAS on ARM, which is NEON-only (~120 GFLOPS multi-threaded). Even if ORT's graph fusion reduces dispatch count from 434 to ~300 ops, the GEMM throughput is the bottleneck at batch decode — and MLAS cannot reach AMX. ORT's batch decode would be bandwidth-bound up to ~B=6 and compute-bound above, hitting a ceiling of ~120 GFLOPS regardless of batch size.

### Apple Silicon generality

- **The BNNS batch-decode throughput (1663 tok/s at B=32) is measured on M1 Max.** The magnitude will differ on other chips (AMX throughput varies). ORT's batch decode throughput has not been measured on any chip, so no specific ratio can be stated.
- **The qualitative conclusion is family-wide:** AMX exists on all Apple Silicon, ORT cannot reach it (does not link Accelerate), and batch decode is compute-bound at B≥8 on all chips. The structural advantage exists on every Apple Silicon part; its magnitude is unmeasured.
- **M4+ (SME):** BNNS abstracts SME routing. The advantage grows on M4+ because SME has higher throughput than AMX, and Accelerate routes to it automatically.

---

## ⚡ CRITICAL UPDATE: BNNS fp16 Matmul Reaches AMX — half_gemm.rs is Wrong for Mac Prefill

**Date:** 2026-07-27T08:28

Justin asked: *"Didn't you say GEMM can use Accelerate? If it's faster than NEON."* He is right. This section supersedes the prefill analysis in Q2 and the half_gemm.rs assessment below.

### The question answered

Standard BLAS has no half-precision GEMM (`cblas_sgemm` is f32). But Apple's **BNNS** (part of Accelerate) exposes `BNNSMatMul` which accepts `BNNSDataTypeFloat16` inputs. I measured it. **It reaches AMX.**

### The decisive table

Measured on M1 Max, load averages 3.9–6.7/10 cores. All numbers corroborated with both `mach_absolute_time` and `clock_gettime(CLOCK_MONOTONIC)` (agreement within 0.1%). Qwen2.5-0.5B shapes summed over all layers.

| M | cblas_sgemm (f32) | BNNS f16→f16 | BNNS f16→f32 | widen+sgemm | NEON 4×8 (1T) | ORT bar |
|---|---|---|---|---|---|---|
| | ms / GFLOPS | ms / GFLOPS | ms / GFLOPS | ms / GFLOPS | ms / GFLOPS | |
| 1 | **48.9** / 20 | 203.2 / 5 | *(not tested)* | 92.4 / 11 | 80.2 / 12 | — |
| 2 | 45.1 / 44 | **22.4** / 88 | — | 92.7 / 21 | — | — |
| 4 | 43.7 / 90 | **22.2** / 178 | — | 89.0 / 44 | — | — |
| 10 | 43.8 / 226 | **23.4** / 422 | — | 89.5 / 110 | — | — |
| 40 | **39.9** / 990 | 31.6 / 1250 | — | 87.8 / 450 | — | 107 ms |
| 128 | 70.6 / 1791 | 56.8 / 2225 | **~55** / **2451** | 124.4 / 1017 | skip | 107 ms |
| 512 | 238.4 / 2122 | **215.7** / 2345 | — | 295.6 / 1711 | skip | — |

**BNNS f16→f32 mixed precision** (fp16 inputs, f32 output): tested at M=128 K=896 N=4864 → **2451 GFLOPS** (mach: 0.455 ms, clock: 0.455 ms). Faster than both homogeneous f16→f16 (2120 GFLOPS) and cblas_sgemm f32 (1972 GFLOPS) at this shape. This is the optimal path: native fp16 inputs (half bandwidth), f32 accumulation (full precision).

### Crossover threshold

| M | BNNS vs sgemm (Gate: 896×4864) | Winner |
|---|---|---|
| 1 | sgemm 0.386 ms vs BNNS 0.900 ms | **sgemm** (BNNS dispatch overhead) |
| 2 | sgemm 0.467 ms vs BNNS 0.206 ms | **BNNS 2.3×** |
| 3 | sgemm 0.435 ms vs BNNS 0.198 ms | **BNNS 2.2×** |
| 4 | sgemm 0.442 ms vs BNNS 0.201 ms | **BNNS 2.2×** |
| 8 | sgemm 0.442 ms vs BNNS 0.208 ms | **BNNS 2.1×** |

**The crossover is exactly M=2.** BNNS has high fixed overhead (~0.9 ms at M=1 from GCD thread pool wake-up, same issue as cblas_sgemm). At M=2+, AMX fp16 throughput overwhelms the overhead. This is a binary threshold, not a sliding scale — runtime detection needs only `geom.m >= 2`, not per-chip calibration.

### Three verdicts

**1. Should prefill f32 GEMM go to `cblas_sgemm`?**
**Yes** — already implemented (matmul.rs:286–293). At M≥2, sgemm achieves 900–2100 GFLOPS. At M=1, our NEON GEMV is better (dispatch overhead too high for sgemm). ✅ No change needed for f32.

**2. Should `half_gemm.rs`'s NEON path be superseded for Mac?**
**Yes — by BNNS `BNNSMatMul` with f16→f32 mixed precision.** half_gemm.rs achieves ~12–52 GFLOPS single-threaded on NEON. Even with 8-core Rayon parallelism (~100–160 GFLOPS), it is **15–25× slower than BNNS** at prefill shapes. At M=128 Gate: BNNS=0.474 ms (2354 GFLOPS) vs NEON=skip (would be ~20 ms, ~55 GFLOPS). **On Mac, hand-written NEON GEMM for compute-bound prefill is the wrong investment.**

half_gemm.rs is NOT wrong in general — it is the right kernel for **non-Mac ARM platforms** (Linux ARM, Windows ARM, Android) where neither BNNS nor Accelerate is available. But on Mac, dispatch should route fp16 M≥2 to BNNS, not to the NEON blocked GEMM.

**3. Where do the thresholds sit?**

| Regime | Dtype | M | Kernel | Why |
|--------|-------|---|--------|-----|
| Decode | f16 | =1 | `neon_gemv_f16_col_parallel` | BW-bound, reads f16 directly, multi-threaded columns |
| Decode | f32 | =1 | `neon_gemv_parallel` | BW-bound, avoids Accelerate dispatch overhead |
| Prefill | f16 | ≥2 | **`BNNSMatMul` f16→f32** | AMX, 2000–2450 GFLOPS, f32 precision output |
| Prefill | f32 | ≥2 | `cblas_sgemm` | AMX, 990–2100 GFLOPS |
| Prefill | f16 | ≥2 (non-Mac) | `half_gemm.rs` | Only path on Linux/Windows ARM |

**Runtime queries for Apple Silicon generality:** The M=2 threshold is chip-independent — AMX is present on all Apple Silicon (M1+), and BNNS routes to the best available hardware. No per-chip calibration needed. Future SME-equipped chips (M4+) would only widen the gap. The specific GFLOPS numbers are M1-Max-specific, but the **ranking** (BNNS f16 > sgemm f32 > NEON) is family-wide.

### BNNS API status

`BNNSMatMul` is deprecated in macOS 15 in favor of `BNNSGraph*` APIs. It still compiles and runs. Migration to `BNNSGraph` is a future maintenance item (the graph API provides the same matmul capability). The deprecation does NOT affect the performance numbers — Apple is consolidating the API surface, not removing the AMX fp16 path.

### Threading constraint (critical)

**Do NOT call BNNS from inside a Rayon parallel region.** BNNS uses GCD internally, and calling it from within Rayon's thread pool causes the same 4× bandwidth collapse I measured earlier with `cblas_sgemm`. The BNNS call must be made from the dispatch level (single Rayon task or main thread), not from within `par_iter`.

### Widen-then-sgemm: the ironic anti-pattern

Widening fp16→f32 and calling `cblas_sgemm` (what ORT does in the graph optimizer) is **1.6× slower than BNNS f16→f32** at M=128 (124.4 ms vs 56.8 ms total). The widening step costs ~30 ms of unnecessary memory traffic. ORT does this because its graph optimizer (`FuseFp16InitializerToFp32NodeTransformer`) widens fp16 weights before they reach the GEMM layer. On Mac, this is the worst possible strategy for prefill: it prevents AMX fp16, doubles memory traffic, and forfeits the 1.24× speedup of native fp16.

The irony: ORT's widening is *also* wrong for decode (it prevents their own hgemm). It is wrong in both regimes, for different reasons.

### TTFT implication

At M=40 (typical short prompt), BNNS f16→f32 total MatMul time: ~31.6 ms. Adding ~5 ms for non-MatMul ops: **~37 ms TTFT**. Compare:
- Our current: 1034 ms (NEON-only GEMM for all M)
- ORT bar: 107 ms
- cblas_sgemm f32: ~45 ms
- **BNNS f16→f32: ~37 ms → 2.9× faster than ORT, 28× faster than our current NEON-only path.**

---

## UPDATE: `half_gemm.rs` Analysis (main merge e104664b)

Main landed `crates/onnx-runtime-ep-cpu/src/kernels/half_gemm.rs` (898 lines): a blocked f16/bf16 GEMM with f32 accumulation. This is now directly load-bearing for Q6, Q1, and Q5. Three findings:

### 1. Architecture and expected GFLOPS

The kernel uses classical GEBP blocking: MR=4, NR=8, KC=128, NC=64. Both A and B are packed from f16/bf16 storage into f32 panels (widening during pack via NEON `vcvt_f32_f16` or scalar fallback). The NEON microkernel (`micro_kernel_neon`, line 625–677) accumulates 4×8 tiles using `vmulq_f32` + `vaddq_f32` — **separate multiply and add, not fused `vfmlaq`**. Rayon parallelizes over row blocks of C.

Estimated single-thread GFLOPS on M1 Max NEON (architectural analysis):
- Per depth step: 2 `vld1q` (B low/high) + 4 rows × (1 `vdup` + 2 `vmul` + 2 `vadd`) = 22 instructions for 64 FLOPs.
- At M1's ~2 insn/cycle, 3.2 GHz: ~18–20 GFLOPS per core.
- Using `vfmaq_f32` instead of separate mul+add would yield ~34 GFLOPS (45% headroom).
- With 8 P-cores via Rayon: ~100–160 GFLOPS multi-threaded.

**Comparison at prefill shapes:**

| Kernel | M=40 GFLOPS | M=128 GFLOPS | M=512 GFLOPS |
|--------|-------------|--------------|--------------|
| Accelerate/AMX | **926** | **1853** | **2053** |
| half_gemm.rs (est. 8-core) | ~120 | ~140 | ~150 |
| MLAS HalfGemmKernelNeon (est.) | ~150–200 | ~180–220 | ~200–250 |

half_gemm.rs is 6–15× slower than Accelerate at prefill. MLAS's `HalfGemmKernelNeon.S` would be ~1.3–1.7× faster than ours because it uses native fp16 arithmetic (8 half-precision elements per FMLA vs 4 single-precision), but this is moot on Mac — **Accelerate dominates both by an order of magnitude.** (Note: MLAS hgemm accumulates in fp16, accepting lower precision; ours accumulates in f32.)

### 2. ⚠️ Dispatch ordering bug: `try_matmul_half` intercepts fp16 M=1 decode

The Qwen2.5-0.5B fp16 model has **both MatMul inputs as Float16** (confirmed: RMSNorm output and weight are both Float16). This means `try_matmul_half` (matmul.rs:488) fires BEFORE the optimized `neon_gemv_f16_col_parallel` path (matmul.rs:497–514).

At M=1, half_gemm is **structurally inferior** to the GEMV path:

| Property | half_gemm at M=1 | neon_gemv_f16_col_parallel |
|----------|-----------------|---------------------------|
| Threading | **Single-threaded** (1 row → 1 chunk in par_chunks_mut) | **Column-parallel** across all cores |
| Memory traffic | Packs f16→f32 panels then reads f32 again: ~3× source data | Reads f16 directly from mmap, widens in-register: ~1× |
| Allocations | Two `Vec<f32>` panels per gemm_block call | Zero (writes into pre-allocated output) |

For the Gate projection (1×896×4864): half_gemm reads ~17 MB of f32 panel data (single-threaded); GEMV reads ~8.7 MB of f16 data (multi-threaded across 8 cores). Estimated **4–8× slower for M=1 decode.**

**Impact on the 60.41 tok/s headline:** This number was measured BEFORE half_gemm.rs landed. If this code ships as-is, **fp16 decode may regress significantly** on the current branch. The fix is straightforward: gate `try_matmul_half` on `m > 1`, or add a `geom.m == 1` carve-out that falls through to the GEMV path. I have flagged this to Iran.

### 3. Strategic impact on Q6, Q1, Q5

**Q6 (strengthened):** half_gemm.rs provides a portable f16 GEMM for non-Mac ARM platforms (Linux ARM, Windows ARM) without vendoring MLAS. For Mac, it's irrelevant — Accelerate handles prefill and the existing GEMV handles decode. The case for vendoring MLAS's ARM assembly is now **even weaker**: we have our own f16 GEMM for the portable path.

**Q1 (sharpened):** When ORT eventually fixes its routing gap and activates `hgemm`, the relevant comparison changes:
- ORT's prefill through MLAS hgemm: ~150–250 GFLOPS (NEON fp16, no AMX).
- Our prefill through Accelerate: ~900–2100 GFLOPS.
- **Our prefill advantage would actually GROW**, because ORT's newly activated hgemm is still NEON-only while we use AMX.
- For decode: the advantage depends on the GEMV path remaining active (requires the dispatch fix above). If GEMV is active, we still win on bandwidth (2 bytes/weight vs MLAS's potential 2 bytes/weight + pack overhead). If the dispatch bug is not fixed, we lose the advantage.

**Q5 (unchanged, one caveat):** The split architecture (NEON GEMV for decode, Accelerate for prefill) remains correct. The one caveat: the try_matmul_half dispatch ordering must be fixed to preserve the decode path. This is a ~5-line code change, not a strategic concern.

---

## Q6 — Would Vendoring MLAS's ARM Kernels Actually Buy Us Anything?

**Answer: No. MLAS's ARM GEMV kernel is tied with ours. Vendoring it buys 0–5% on f32 decode and exactly 0% on prefill. The cost vastly exceeds the gain.**

### Head-to-head microbenchmark (isolated kernel, no graph fusion or thread pool confound)

Three implementations at identical Qwen2.5-0.5B shapes (M=1 decode), single-threaded, measured with `mach_absolute_time`. Two runs at different system loads for corroboration:

| Run | Load avg | Our GEMV (ms) | MLAS-style GEMV (ms) | Ratio |
|-----|----------|---------------|---------------------|-------|
| 1 | 24.9 | 30.18 | 28.70 | **1.05×** |
| 2 | 7.0 | 30.39 | 30.52 | **1.00×** |

Per-shape breakdown (Run 2, lower contention):

| Op | K | N | Ours ms | MLAS ms | Winner |
|---|---|---|---|---|---|
| QKV | 896 | 1152 | 0.064 | 0.078 | **Ours +22%** |
| O_proj | 896 | 896 | 0.051 | 0.063 | **Ours +24%** |
| Gate | 896 | 4864 | 0.415 | 0.359 | **MLAS +16%** |
| Up | 896 | 4864 | 0.362 | 0.416 | *(noise)* |
| Down | 4864 | 896 | 0.375 | 0.357 | **MLAS +5%** |

**Pattern**: Our dot-product-on-transposed-B wins on small N (QKV, O_proj). MLAS's outer-product-on-row-major-B wins on large N (Gate). They cancel out. The TOTAL is tied.

**Methodology note**: The "MLAS-style" kernel is a C reimplementation of the exact algorithm in `SgemvKernelNeon.S` — 64-column outer-product loop with `ld1r`+`fmla` broadcast pattern and NEON 16-register accumulator usage. It does NOT include MLAS's panel packing or KleidiAI's hand-scheduled assembly, so it slightly underestimates MLAS's actual kernel. Even granting MLAS 10–15% for assembly scheduling, the gain is **at most 5–15% on the subset of decode shapes where MLAS wins** (Gate/Up), which translates to **~3–5% overall** after averaging with shapes where we win.

### For prefill (M>1): MLAS is irrelevant

| M | Accelerate GFLOPS (measured) | MLAS NEON (est. ~120) | Notes |
|---|---|---|---|
| 40 | 926 | ~120 (est.) | MLAS value not measured directly |
| 128 | 1853 | ~120 (est.) | MLAS value not measured directly |

MLAS cannot reach AMX for compute-bound work. The MLAS NEON estimate (~120 GFLOPS) is based on NEON theoretical throughput, not a direct measurement of MLAS at these shapes. Regardless, vendoring MLAS ARM SGEMM would add **zero** prefill value on Mac — the Accelerate path is structurally superior.

### KleidiAI is load-bearing and would also need vendoring

ORT's ARM performance does not come from MLAS's `.S` files alone. The shipped dylib (`libonnxruntime.1.27.0.dylib`) dispatches to **KleidiAI** microkernels:

- `GetKleidiAISGemmUKernel`, `GetKleidiAISGemvUKernel` — f32 GEMM/GEMV
- `GetKleidiAIQGemmUKernel` — int4 quantized GEMM
- `ArmKleidiAI::UseSME`, `ArmKleidiAI::UseSME2` — runtime SME detection

KleidiAI's source is referenced from our vendor snapshot (`kai_ukernel_interface.h/cpp`) but the actual header-only microkernel files (`kai/ukernels/...`) are **not vendored**. To reproduce ORT's ARM speed, we would need to vendor KleidiAI too.

**KleidiAI details:**
- **License**: MIT (ARM Limited, `SPDX-License-Identifier: MIT`). ✅ Permissive.
- **Scope**: NEON + DotProd + i8mm + SME + SME2 kernels for f32 GEMM/GEMV, int4/int8 quantized GEMM, bf16 SBGEMM, f16 HGEMM. ~25 kernel headers included from `kai_ukernel_interface.cpp`.
- **Our vendor snapshot**: Has the interface file but NOT the kernel headers. A full vendor drop would need the entire `kai/ukernels/` tree.
- **Build impact**: KleidiAI is header-only (function definitions in `.h` files), so it compiles with the translation units that `#include` it — no separate assembly step, but it would increase compile time for `platform.cpp` and related files.

### Cost of default-enabling `mlas` with ARM support

| Cost | Detail |
|------|--------|
| **Vendor size** | Current x86-only: 3.7 MB. Add: ~500K aarch64 assembly + KleidiAI headers (~estimated 1–2 MB). Total ~5–6 MB. |
| **Build toolchain** | Requires C++ compiler + assembler for aarch64 on every build. macOS: Xcode clang handles it. Linux: needs `aarch64-linux-gnu-gcc` for cross-compile. Windows ARM64: MSVC `.asm` files differ from GAS `.S`. |
| **Build time** | 20 aarch64 `.S` files + ~30 C++ files = ~30–60s additional compile time per build. Default-enabled means this hits every developer, not just opt-in. |
| **CI matrix fragility** | We broke CI on non-aarch64 yesterday. Adding aarch64 assembly to a default-on feature means every x86 CI build must skip it (feature gating) or conditionally compile. `build.rs` must handle `cfg(target_arch)` correctly for ALL targets: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `aarch64-pc-windows-msvc`, `aarch64-unknown-linux-gnu`. One miss = CI break. |
| **Maintenance drift** | MLAS upstream changes ~monthly. Our x86 snapshot is already drifting (no KleidiAI headers). An ARM drop would also drift. Syncing requires manual review of assembly changes. |
| **Platform.cpp complexity** | MLAS's `platform.cpp` has ~15 ARM-specific references and conditional compilation. Adding it to the build doubles the dispatch-table complexity. |

### Strategy ranking (Q6 verdict)

| Rank | Strategy | Decode gain | Prefill gain | Cost | Verdict |
|------|----------|-------------|-------------|------|---------|
| **1** | **Ours for decode + Accelerate for prefill (option 2)** | Baseline (already 1.42× over ORT fp16) | Accelerate: 926–2053 GFLOPS (measured) vs MLAS ~120 (est.) | Zero | **Best.** Already implemented. |
| **2** | **Option 2 + graph-level op fusion (option 3)** | +3–5% from fusing QKV + reducing 435→~300 dispatches | Same as option 2 | Medium (Sapper + Iran) | **Second best.** Addresses the actual gap source. |
| **3** | **Vendor ARM MLAS, default-enable (option 1, Justin's proposal)** | +0–5% f32 decode from kernel quality | 0% (Accelerate already dominates) | High (vendor, CI, maintenance) | **Not worth it.** Gains do not justify costs. |

### Why option 1 is wrong for Mac

Justin's instinct — "selectively pick the fastest path" — is correct. But on Mac, **the fastest path for decode is already ours** (tied with MLAS for f32, ahead for f16), and **the fastest path for prefill is Accelerate** (which MLAS cannot reach). Vendoring MLAS ARM doesn't add a faster path for either regime. It adds a *tied* path for one regime and a *slower* path for the other.

The 8% f32 decode gap between us and ORT is not a kernel problem — it's a graph-fusion problem (435 ops/token vs ~300). Iran's attribution confirmed: ~0.9 ms/token dispatch overhead vs ~0.5 ms/token kernel quality. Vendoring MLAS fixes the 0.5 ms (at most), not the 0.9 ms. Graph fusion fixes the 0.9 ms.

### When option 1 would be right

If we needed MLAS on a **non-Mac ARM platform** (Linux ARM server, Windows ARM, Android) where Accelerate is unavailable and our NEON GEMM is the only option — then vendoring MLAS ARM would add value for GEMM-bound prefill. But on Mac, Accelerate eliminates this need entirely. **Update:** the new `half_gemm.rs` kernel now provides a portable f16/bf16 GEMM for non-Mac ARM platforms (~100–160 GFLOPS with Rayon, vs MLAS's estimated ~150–250 GFLOPS for hgemm). While MLAS would be ~1.3–1.7× faster on pure NEON, half_gemm.rs eliminates the "no GEMM at all" gap that was the strongest argument for vendoring.

---

## TL;DR — Q5 Answer

**Yes, the native CPU EP can stand alone on Apple Silicon.** The split architecture is:

| Regime | Kernel | Why |
|--------|--------|-----|
| Decode (M=1, bandwidth-bound) | Our NEON f16 GEMV | 88–100 GB/s single-threaded, half the DRAM reads of f32, already 1.42× faster than ORT/MLAS |
| Prefill f16 (M≥2, compute-bound) | **BNNS `BNNSMatMul` f16→f32** | AMX, 2000–2450 GFLOPS, f32 precision, 2.9× faster than ORT |
| Prefill f32 (M≥2, compute-bound) | Accelerate `cblas_sgemm` → AMX | 900–2100 GFLOPS on this M1 Max |

ORT structurally cannot reach AMX (no Accelerate linkage in the shipped dylib, confirmed by `otool -L`). Under this split we never need MLAS on Mac — we are faster than MLAS for decode (fp16 bandwidth advantage) and faster than anything MLAS can deliver for prefill (AMX vs NEON).

**However, our fp16 decode advantage is fragile (see Q1 below). Justin needs to understand this before betting the roadmap.**

---

## Q1 — Fragility of the FP16 Advantage

### Finding: Our 1.42× fp16 lead rests on an ORT routing gap, not a capability gap.

**Evidence:**

1. ORT's MLAS binary contains a fully-wired HGEMM path:
   - `MlasHGemmDispatchNeon` dispatch table (symbol at 0x15f2600)
   - `HGemmOperation`, `MlasHGemmSupported` — runtime dispatch for half-precision GEMM
   - `MlasGemmBatch` accepting `MLAS_HGEMM_DATA_PARAMS` — batched hgemm ready
   - `hw.optional.arm.FEAT_FP16` check string — runtime capability detection present

2. ORT's graph optimizer **intercepts fp16 before it reaches MLAS**:
   - `FuseFp16InitializerToFp32NodeTransformer` — proactively widens fp16 weights to fp32 at graph-optimization time
   - `InsertCastTransformer` — inserts Cast nodes for type mismatches
   - `IsIsolatedFp16NodeOnCpu` — identifies and converts "isolated" fp16 nodes

3. MLAS hgemm layout constraint (from error strings):
   - `"hgemm currently only support A x Transpoe(B) or A x B"` (sic, typo in MLAS source)
   - Standard ONNX MatMul uses `A × B` (no transpose), which IS in the supported set

**Fragility assessment:**

The fix for ORT upstream is straightforward: suppress `FuseFp16InitializerToFp32NodeTransformer` for MatMul-type ops when `MlasHGemmSupported(CblasNoTrans, CblasNoTrans)` returns true. This is a graph-optimizer config change, not new kernel work. Likelihood of appearing in ORT 1.28 or 1.29: **moderate to high** — the kernel exists, the capability check exists, only the routing is missing.

If ORT fixes this, their fp16 decode would approach our performance (same bandwidth advantage: read f16, compute in f32). Our remaining edge would be:
- Our SPMD decode pool (~50 ns barrier) vs ORT's ThreadPool (~2–5 µs fork-join)
- Our direct mmap-to-GEMV path (zero-copy f16 transpose) vs MLAS's copy+pack path

Estimated residual advantage after hypothetical ORT fix: **10–20%** instead of 42%.

**Update (half_gemm.rs):** Our own blocked f16 GEMM now exists (`half_gemm.rs`). At the kernel level, it gets ~18–20 GFLOPS/core (f32 accumulation) vs MLAS hgemm's estimated ~25–30 GFLOPS/core (fp16 accumulation with 2× element width per instruction). MLAS has a kernel-quality edge of ~1.3–1.5× on pure NEON, but this is irrelevant for Mac prefill (Accelerate wins both by 10×+). For decode (M=1), the comparison is between GEMV implementations, not GEMM — and our GEMV is already tied with MLAS's (see Q6 head-to-head). **The FP16 moat is fragile for decode, but we have a prefill moat that ORT cannot reach.** ⚠️ The decode moat requires the dispatch fix flagged in the half_gemm.rs update above.

**Mitigation:** Our real moat is Accelerate/AMX for prefill, not fp16 decode. Even if ORT closes the fp16 decode gap, they still can't touch our prefill performance unless they link Accelerate, which is a much larger upstream change.

---

## Q2 — Prefill: Accelerate/AMX Performance at Real Shapes

### Measured: cblas_sgemm at Qwen2.5-0.5B shapes on M1 Max (8 P-cores)

All numbers corroborated with both `clock_gettime(CLOCK_MONOTONIC)` and `mach_absolute_time()`.

**Per-op GFLOPS by prompt length (Accelerate, all layers summed):**

| Prompt (M) | QKV GFLOPS | Gate/Up GFLOPS | Down GFLOPS | Total MatMul ms | Implied TTFT |
|------------|-----------|---------------|-------------|-----------------|-------------|
| 10 | 803 | 167–195 | 170 | 45 ms | ~50 ms |
| 40 | 1168 | 1060 | 1053 | 42 ms | ~47 ms |
| 128 | 1936 | 1566–1911 | 896 | 87 ms | ~95 ms |
| 512 | 2303 | 2016–2018 | 1419 | 263 ms | ~290 ms |

**Comparison: NEON-only GEMM (no AMX, single-threaded):**

| M | NEON GFLOPS | Accelerate GFLOPS | Speedup |
|---|-------------|-------------------|---------|
| 10 | 21–23 | 167–803 | 8–35× |
| 40 | 15–21 | 1053–1168 | 50–70× |
| 128 | 13–21 | 896–1936 | 42–92× |

### AMX M-threshold: There is no lower threshold.

AMX pays off even at M=10 (10-token prompt), delivering 170–800 GFLOPS vs NEON's 21 GFLOPS. The transition is binary:
- **M=1 → NEON GEMV** (bandwidth-bound, AMX dispatch overhead exceeds compute)
- **M≥2 → Accelerate sgemm** (compute-bound, AMX dominates)

No hybrid strategy needed. The crossover is exactly at the decode/prefill boundary.

### Implied TTFT vs ORT:

- Our current TTFT: **1034 ms** (NEON-only GEMM for prefill, ~20 GFLOPS)
- With Accelerate sgemm: **~45 ms** at M=40 (MatMul only = 40 ms + ~5 ms other ops)
- **With BNNS f16→f32: ~37 ms** at M=40 (MatMul only = 31.6 ms + ~5 ms other ops) — **best available path**
- ORT bar: **107 ms** TTFT
- **We'd beat ORT by ~2.9× on prefill** with BNNS fp16, and ORT structurally cannot close this gap (no BNNS usage, no Accelerate linkage).

### Apple Silicon generality:

| Chip | FEAT_FP16 | AMX | cblas_sgemm | Notes |
|------|-----------|-----|-------------|-------|
| M1/M1 Pro/Max/Ultra | ✅ | Gen 1 | ✅ | Measured here |
| M2/M2 Pro/Max/Ultra | ✅ | Gen 2 | ✅ | Higher GFLOPS expected |
| M3/M3 Pro/Max/Ultra | ✅ | Gen 3 | ✅ | Higher GFLOPS expected |
| M4/M4 Pro/Max | ✅ | Gen 4 + SME | ✅ | SME gives further uplift |

Accelerate is the stable API across all generations. No runtime detection needed beyond `#[cfg(target_os = "macos")]` — Apple guarantees `cblas_sgemm` routes to the best available hardware (AMX or SME). KleidiAI also has `UseSME`/`UseSME2` checks, but we don't need KleidiAI since Accelerate abstracts this.

---

## Q3 — What Makes MLAS Fast on ARM, and Is Any of It Worth Porting?

### MLAS ARM internals (from ORT 1.27.0 dylib analysis):

| Component | What it does | Gain available to us | Effort | Verdict |
|-----------|-------------|---------------------|--------|---------|
| **KleidiAI microkernels** | ARM's hand-tuned asm GEMM/GEMV (`GetKleidiAISGemmUKernel`, `GetKleidiAIGemvUKernel`). Optimized for Cortex-X1/X2, with SME awareness. | GEMV: ~5–15% over our NEON. GEMM: moot (Accelerate beats any NEON kernel). | High (C++ FFI, ARM-specific asm) | **Skip.** Accelerate makes GEMM moot; GEMV gains don't justify the FFI complexity. |
| **B-panel packing** | `MlasGemmPackB`, `MlasSgemmCopyPackB` — pre-pack weights for cache-optimal access patterns in GEBP tiling. | Relevant for GEMM only. Our Accelerate path doesn't need it (Apple packs internally). | Medium | **Skip.** Moot under Accelerate. |
| **Cache-aware tiling** | KC/NC/MC blocking per L1/L2 size. Built into the dispatch loop with MLAS's thread pool. | Same as packing — only matters for NEON-only GEMM path. | Medium | **Skip.** |
| **Thread pool** | ORT's `concurrency::ThreadPool` — work-stealing, QoS-aware scheduling. | Our SPMD pool already achieves ~50 ns barrier (measured). ORT's ThreadPool adds ~2–5 µs per dispatch. We're faster here. | N/A | **Keep ours.** We win. |
| **HGEMM (half GEMM)** | `MlasHGemmDispatchNeon` — NEON fp16 GEMM with FEAT_FP16. | For decode: we already do f16→f32 in-register GEMV (same approach, less overhead). For prefill: moot (Accelerate). | High | **Skip.** Our f16 GEMV is already competitive. |
| **Quantized GEMM** | `MlasSymmQgemmPackB`, `MlasDynamicQgemmPackB` — int8/int4 quantization-aware GEMM. `MlasQ4GemmPackB`. | Potentially useful for int4/int8 quantized models. We have `MatMulNBits` but haven't benchmarked MLAS's quant GEMM vs ours on ARM. | High (major FFI) | **Evaluate later.** Relevant if we ship int4 on Mac. |

### Bottom line:

Accelerate makes most of MLAS moot for compute-bound work (prefill), and our own NEON GEMV already suffices for bandwidth-bound decode. The only potentially valuable piece is the quantized GEMM for future int4 models, but that's a separate decision and MLAS's ARM int4 path would need evaluation.

**Porting hand-written assembly is not worth it.** The effort-to-payoff ratio is terrible: ~5–15% decode improvement from KleidiAI GEMV vs months of FFI maintenance. If we need that last 15%, it's cheaper to optimize our Rust NEON kernel (add 8-row batching, prefetch hints) than to vendor KleidiAI.

---

## Q4 — The 8% FP32 Decode Gap

### Measurements: 42.30 tok/s (ours) vs 46.01 tok/s (ORT/MLAS)

**Where the gap lives:**

| Source | Estimated contribution | Evidence |
|--------|----------------------|----------|
| Op dispatch overhead | 2–4% | We dispatch 435 ops/token (446 nodes – 11 initial). ORT fuses subgraphs (QKV into one MatMul: 1152-wide, SiLU, LayerNorm) → ~250–300 dispatches. ~150 extra dispatches × ~2 µs = ~0.3 ms on a ~24 ms token. |
| GEMV kernel quality | 2–4% | MLAS KleidiAI has more aggressive register scheduling and prefetch tuning than our 4-row batched NEON GEMV. Measured our single-threaded at 52 GB/s (f32); MLAS likely achieves 60–65 GB/s. |
| B-panel pre-packing | 1–2% | MLAS pre-packs B for cache-line-aligned streaming. Our pre-transposed B is close but not identical to packed layout. |

**Total: ~5–10%**, consistent with the measured 8%.

### Is it reachable?

Yes, but not worth the effort if Mac default becomes fp16:
- **With fp16 (recommended default):** We lead 60.41 vs 42.45 → 42% advantage. The 8% f32 gap is irrelevant.
- **If we still wanted to close it:** Fuse QKV (Sapper, low-priority), add prefetch to NEON GEMV, tune SPMD partition sizes. Estimated 2–3 days engineering for ~5% recovery.

**Verdict: Concede it.** Ship fp16 as the Mac default. The 8% fp32 gap is unmeasurable in the fp16 world where we lead by 42%.

---

## Q5 — Strategic Recommendation (Full)

### The native CPU EP can stand alone on Apple Silicon.

The architecture:

```
                    ┌──────────────────────┐
                    │   Decode (M=1)       │
                    │   Bandwidth-bound    │
                    │   NEON f16 GEMV      │
                    │   88–100 GB/s/core   │
                    │   → 60+ tok/s        │
                    └──────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │   auto-detect M    │
                    └─────────┬──────────┘
                              │
                    ┌──────────────────────┐
                    │   Prefill (M≥2)      │
                    │   Compute-bound      │
                    │   f16: BNNS f16→f32  │
                    │   f32: cblas_sgemm   │
                    │   → AMX coprocessor  │
                    │   2000–2450 GFLOPS   │
                    │   → ~37 ms TTFT @M=40│
                    └──────────────────────┘
```

### Why we don't need MLAS on Mac:

1. **Decode:** Our fp16 NEON GEMV reads 2 bytes/weight (half the DRAM traffic of MLAS's fp32 path), achieving 88 GB/s single-threaded. MLAS's fp32 path achieves ~60 GB/s. We win structurally.

2. **Prefill:** Accelerate/AMX delivers 900–2100 GFLOPS. MLAS's NEON GEMM delivers ~20 GFLOPS (single-threaded, ~80 with threading). We win by 10–100×. **ORT cannot reach AMX** — the shipped dylib has no Accelerate linkage.

3. **Thread pool:** Our SPMD decode pool has ~50 ns barrier latency vs ORT's ThreadPool at ~2–5 µs. We win on dispatch overhead.

### What must ship to realize this:

1. **Accelerate integration for prefill (already coded):** `accelerate_gemm::sgemm` is already implemented and gated on `CpuBackend::Accelerate`. Confirm it activates for M>1 in `gemm_with_backend`. ✅ Already done (matmul.rs line 286–293).

2. **FP16 as Mac default:** Ship fp16 models as the standard Mac model format. Our fp16 GEMV path is already the hot path when `CpuBackend::Accelerate` is selected and `inputs[1].dtype == Float16` (matmul.rs line 496–514). ⚠️ **Dispatch fix needed:** `try_matmul_half` (line 488) now intercepts this path for fully-fp16 models. Gate it on `geom.m > 1` to preserve the GEMV decode path.

3. **Verify Accelerate TTFT E2E:** The microbenchmarks predict ~47 ms TTFT at M=40. Measure this with the compare harness using the fp32 model and Accelerate enabled. If it matches, the story is closed.

### Caveats and risks:

| Risk | Severity | Mitigation |
|------|----------|------------|
| ORT fixes fp16 routing → our decode moat shrinks from 42% to ~15% | Medium | Our real moat is prefill (AMX). Decode advantage is bonus. |
| Apple changes AMX access pattern in future macOS | Low | Accelerate is the stable API; Apple guarantees it. |
| Small models (M=10 prompt) show lower AMX utilization (170 GFLOPS vs 2100 at M=512) | Low | Even at 170 GFLOPS we beat MLAS's ~80 GFLOPS by 2×. |
| Future int4/int8 quantized models may need MLAS's quant GEMM | Medium | Evaluate if/when we ship int4 on Mac. Separate decision. |

### What we do NOT need from MLAS:

- ❌ KleidiAI microkernels (Accelerate beats them for GEMM; marginal for GEMV)
- ❌ B-panel packing (Accelerate handles this internally)
- ❌ HGEMM (our f16 GEMV is already competitive)
- ❌ Cache tiling (Accelerate handles this internally)
- ❌ Thread pool (ours is faster)

### What we might want later (separate decisions):

- ❓ MLAS quantized GEMM for int4 Mac models (evaluate when relevant)
- ❓ KleidiAI's SME2 awareness for M4 (check if Accelerate already routes to SME)

---

## Raw Data (Corroborated Measurements)

### Benchmark environment
- **Machine:** Apple M1 Max, 8 P-cores + 2 E-cores, 32 GB LPDDR5
- **FEAT_FP16:** Yes (hw.optional.arm.FEAT_FP16 = 1)
- **FEAT_BF16:** No (M1)
- **DRAM BW (theoretical):** 400 GB/s
- **Timing:** clock_gettime(CLOCK_MONOTONIC) and mach_absolute_time(), both reported

### Accelerate sgemm — full prefill TTFT (MatMul only)

| M | QKV ms×24 | O ms×24 | Gate ms×24 | Up ms×24 | Down ms×24 | lm_head ms | Total ms |
|---|-----------|---------|------------|----------|------------|------------|----------|
| 10 | 0.62 | 0.49 | 12.56 | 10.74 | 12.33 | 8.34 | 45.1 |
| 40 | 1.70 | 1.33 | 7.83 | 7.89 | 7.95 | 15.32 | 42.0 |
| 128 | 3.28 | 2.80 | 14.01 | 17.10 | 29.88 | 19.42 | 86.5 |
| 512 | 11.02 | 8.47 | 53.07 | 53.12 | 75.49 | 61.79 | 262.9 |

### NEON GEMV — single-threaded decode (M=1)

| | f32 total ms | f16 total ms | f16/f32 speedup |
|---|---|---|---|
| All layers + lm_head | 62.35 | 10.97 | 5.68× |
| f32 BW | 30–62 GB/s | — | — |
| f16 BW | — | 86–100 GB/s | — |

### Accelerate sgemm M=1 decode (why we DON'T use it for decode)

| Op | ms | GB/s |
|---|---|---|
| Gate | 0.455 | 38.3 |
| lm_head | 17.53 | 31.1 |

Accelerate's M=1 path is 2–3× slower than our NEON GEMV due to dispatch overhead. Confirmed: GEMV stays with us, GEMM goes to Accelerate.
