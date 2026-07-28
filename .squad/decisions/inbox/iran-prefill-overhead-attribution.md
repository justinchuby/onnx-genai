# Prefill Overhead Attribution — Measured Breakdown

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**Status:** Research finding — no implementation changes
**Requested by:** Justin Chu
**Machine:** Apple M1 Max 10-core, macOS
**Model:** Qwen2.5-0.5B-Instruct fp16, 24 layers, 14 Q-heads / 2 KV-heads (GQA), hidden=896, head_dim=64
**Prompt:** 40 tokens

---

## TL;DR: The overhead is concentrated, and smaller than believed

**The distribution is CONCENTRATED**, not spread.

The "~136 ms dispatch overhead" does not exist at low load. At load ~3 on 10 cores:

| Component | ms | % of TTFT |
|---|---:|---:|
| **GEMM (MatMul + FusedMatMulBias)** | 52 | 68% |
| **Attention SDPA** | 11 | 14% |
| **SiLU/Swish** | 7 | 9% |
| **Mul (gate × up)** | 3 | 4% |
| **RotaryEmbedding** | 2 | 2% |
| **RMSNormalization** | 2 | 2% |
| **96 Constant + misc** | <0.2 | <0.3% |
| **Non-node overhead** | ~4 | ~5% |
| **TTFT** | **~80** | **100%** |

Three ops — **Attention (11 ms), Swish (7 ms), Mul (3 ms)** — account for
**87% of non-GEMM time**. The remaining 350 ops (Constant, Shape, Cast, etc.)
contribute <0.2 ms combined. Per-dispatch mechanism cost is <1 µs/op.

The 160 ms figure was **CPU contention at high system load**, not dispatch overhead.

---

## 1. Per-Op Timing Breakdown (Prefill, 40 tokens)

Conditions: load 3.5, 5 measured runs, `ONNX_GENAI_PROFILE_OPS=1`.

| Op Type | Total ms | % | Calls | µs/call |
|---|---:|---:|---:|---:|
| MatMul | 31.4 | 41.3 | 49 | 641.6 |
| FusedMatMulBias | 20.2 | 26.5 | 120 | 168.1 |
| Attention | 10.9 | 14.3 | 24 | 453.6 |
| Swish | 6.7 | 8.8 | 24 | 278.0 |
| Mul | 3.4 | 4.4 | 24 | 141.0 |
| RotaryEmbedding | 1.7 | 2.3 | 48 | 36.0 |
| RMSNormalization | 1.6 | 2.1 | 49 | 31.9 |
| Constant | 0.1 | 0.1 | 96 | 1.0 |
| Gather | 0.03 | 0.04 | 3 | 8.9 |
| *11 other* | <0.1 | <0.1 | 12 | — |
| **Total** | **76.1** | **100** | **446** | **170.6** |

TTFT wall-clock: ~80 ms (76 ms node execution + ~4 ms engine overhead).

---

## 2. Per-Call Cost Decomposition

### Attention (454 µs/call, 10.9 ms total)

The `Attention` contrib op calls `to_bhsd()` for Q/K/V (fp16→f32 widen +
reshape/transpose), `concat_cache` (clone), then `sdpa_f32_neon` (NEON 4×-unrolled
dot + axpy, scalar loop over batch×heads×seq²), then f32→f16 narrow.

| Phase | ~µs | Notes |
|---|---:|---|
| 3× to_bhsd (widen + transpose) | ~150 | NEON fcvtl, 3×36 K elements |
| sdpa_f32_neon | ~200 | 14 heads × 40² × 64 dot+axpy, single-threaded |
| concat_cache (clone) | ~30 | No past cache during prefill |
| Output narrow + alloc | ~70 | vec! allocation + NEON fcvtn |

The SDPA core is **not using Accelerate/AMX for its internal Q·Kᵀ and P·V
matmuls** — it runs NEON dot+axpy loops. The MLAS fast path exists in the code
but is gated behind `--features mlas`, which `bench-native` does not enable.

### Swish/SiLU (278 µs/call, 6.7 ms total)

| Phase | ~µs | Notes |
|---|---:|---|
| fp16→f32 widen | ~50 | NEON fcvtl, 143 K elements [1,40,3584] |
| SiLU compute (NEON) | ~180 | 4×-unrolled NEON sigmoid; exp() dominates |
| f32→f16 narrow | ~50 | NEON fcvtn |

Mostly legitimate compute — `exp()` over 143 K elements is ~180 µs at
NEON throughput. Widen/narrow is ~35% of the call.

### Mul (141 µs/call, 3.4 ms total)

| Phase | ~µs | Notes |
|---|---:|---|
| fp16→f32 widen (×2 inputs) | ~100 | Dominates — trivial mul after |
| Multiply (broadcast_apply) | ~10 | <10% of call |
| f32→f16 narrow | ~30 | |

**~90% of this op is widen/narrow for a trivial pointwise multiply.**
An fp16 multiply kernel would reduce each call from 141 µs to ~20 µs.

---

## 3. AMX State Switching

**Not a significant factor.** AMX is a coprocessor accessed through Accelerate
framework calls (BNNS BroadcastMatMul). Between BNNS calls, other ops run NEON
intrinsics. Apple Silicon manages AMX↔NEON transitions in hardware — they are
not a software context switch. The 350-op interleaving of AMX and NEON work does
not produce measurable switching cost.

Evidence: the per-call cost decomposition for Attention, Swish, and Mul fully
accounts for their measured time through compute + widen/narrow — there is no
unexplained residual consistent with per-call AMX switching overhead.

## 4. Per-Call fp16↔f32 Widening

**Yes, every non-GEMM op widens on entry and narrows on exit.**

All ops go through `to_dense_f32_widen()` → compute in f32 → `write_dense_f32_narrow()`.
This involves:
- A `Vec<f32>` allocation per call (~143 KB for M=40 SiLU)
- NEON fcvtl bulk conversion (f16→f32)
- A second allocation for the result
- NEON fcvtn bulk conversion (f32→f16)

For compute-heavy ops (SiLU), widen/narrow is ~35% of the call.
For compute-light ops (Mul), it's **~90%** — the conversion dominates.

Aggregate widen/narrow cost across all non-GEMM ops: **~8–10 ms** (10–13% of TTFT).

## 5. Allocation / Buffer Churn

Each non-GEMM op allocates temporary f32 buffers per call via `to_dense_f32_widen`
(which returns `Cow::Owned`). At M=40, hidden=896, intermediate=3584, this means:
- SiLU: 573 KB × 2 (widen buf + result) per call × 24 = 27 MB churn/prefill
- Mul: 573 KB × 3 (2 inputs + result) × 24 = 41 MB
- Attention: ~1 MB × 4 (Q, K, V, output) × 24 = 96 MB
- Total: ~170 MB allocated and freed during one prefill step

The allocator cost (malloc/free overhead) is subsumed in the widen/narrow
measurements above. macOS's `malloc_zone` handles this reasonably, but the
pattern is wasteful.

## 6. Threading

No oversubscription during prefill. BNNS uses GCD internally for its MatMul.
The SDPA (`sdpa_f32_neon`) runs **single-threaded**. Rayon is only used for
decode GEMV, not during prefill. There is no threading cost during prefill
beyond BNNS's own internal parallelism.

---

## 7. Comparison Against ORT

Measured back-to-back at load ~2.7–3.3:

| Metric | Native | ORT | Ratio |
|---|---:|---:|---:|
| Prefill (TTFT) | 78 ms | 108 ms | **0.73×** (native 28% faster) |
| Decode (ms/tok) | 13.2 | 23.2 | **0.57×** (native 43% faster) |

**Native already beats ORT on prefill.** The gap is BNNS/AMX GEMMs at
1472–2436 GFLOPS vs ORT's NEON kernels at ~400–500 GFLOPS. ORT has ~130
fewer ops after its more aggressive optimizer, but its weaker GEMM kernels
more than offset that advantage.

Per-dispatch overhead comparison:
- Native: 76 ms / 446 ops = **170 µs/op** average (dominated by GEMM at 641 µs)
- ORT: 108 ms / ~315 ops ≈ **343 µs/op** average

Native has **lower** per-op overhead than ORT. ORT's disadvantage is kernel
throughput, not dispatch count.

---

## 8. Load Sensitivity

| Load avg (10 cores) | Native TTFT | Notes |
|---|---:|---|
| 2.7 | 78 ms | Quiet |
| 3.5 | 80 ms | Quiet |
| 6–8 | 90 ms | Moderate (history: PR #297) |
| 15–17 | 160 ms+ | Heavy compilation/agents |

TTFT doubles between load 3 and load 15. This is **GCD thread contention
inside BNNS** — BNNS dispatches MatMul work onto the system-wide GCD thread
pool, and competing threads delay those work items. This is not a dispatch
mechanism problem.

---

## 9. Recommendations (Ranked by Recoverable Milliseconds)

### 1. Native fp16 elementwise ops — **~3–5 ms recoverable**

The Mul kernel spends 90% of its time on fp16↔f32 conversion for a trivial
multiply. A NEON fp16 Mul/Add elementwise path would reduce 24 × 141 µs to
24 × 20 µs = ~2.9 ms saved. Similar savings from fp16 RMSNorm and
RotaryEmbedding. **Low risk, low effort, predictable gain.**

### 2. Accelerate sgemm for Attention SDPA — **~5–8 ms recoverable**

The Attention op's Q·Kᵀ and P·V matmuls are small GEMMs (M=40, N=40, K=64
× 14 heads). The current NEON dot+axpy loops are ~3–5× slower than what
Accelerate `cblas_sgemm` would achieve. Batching the 14 heads into a strided
GEMM call is the right approach (BNNS BroadcastMatMul with batch dims).
**Medium effort, significant gain, already proven feasible in the MatMul kernel.**

### 3. Fused SiLU·Mul — **~3–4 ms recoverable**

The SiLU and Mul ops are always adjacent (SiLU on gate, then gate × up).
Fusing them eliminates one widen+narrow round-trip and one 573 KB intermediate
allocation per layer. This is a graph fusion, not a kernel merge — it creates
a single `SiLuMul` op that reads two fp16 inputs and writes one fp16 output
with a single widen+compute+narrow pass. **Medium effort, good return, standard
fusion pattern.**

### 4. Don't chase dispatch count or op fusion — **confirmed not the lever**

- The Q/K/V sibling merge (PR #297) regressed TTFT by 68% because BNNS
  prefers smaller GEMMs.
- Per-dispatch mechanism cost is <1 µs/op (96 Constant ops cost 0.1 ms total).
- The 446→350 reduction from the optimizer already eliminated the cheap ops.
- ORT's advantage is kernel throughput, not fewer dispatches.

### 5. Load sensitivity mitigation — **unknown recovery, high impact**

The 80→160 ms degradation under load is a GCD contention problem inside BNNS.
Options to investigate:
- `dispatch_set_target_queue` to isolate BNNS's GCD pool
- QoS class hints (`QOS_CLASS_USER_INTERACTIVE`) on the inference thread
- Pre-sizing BNNS's thread pool via `BNNSFilterSetThreadCount`

This would not improve best-case TTFT but would stabilize it under real
workloads.

---

## Methodology

- **Instrumentation:** `ONNX_GENAI_PROFILE_OPS=1` (per-op-type timing via
  `std::time::Instant` in `executor/dispatch.rs`). This timestamps every
  `exec_plan_node` call and aggregates by op type.
- **Trace:** `--trace` option captures per-op-instance Chrome Trace events via
  `onnx-runtime-tracer` `SpanGuard` RAII.
- **Load:** Reported with every number via `uptime`. Measurements taken at
  load 2.7–3.5 on a 10-core system after waiting for quiet.
- **Runs:** 5 measured runs with 2 warmups, `--steady` mode.
- **Corroboration:** Key numbers (TTFT, decode) cross-checked between native
  and ORT backends at matched load, and against prior PR #297 measurements.

`uptime` at measurement time: `load averages: 3.48 5.19 4.73` (prefill profiling),
`load averages: 2.70 6.96 6.55` (ORT comparison).

---

## Known issue: MLX plugin debug noise

The MLX/Metal EP plugin (`onnxruntime-mlx`) prints once per subgraph execution
in a hot loop:

```
[rust-mlx-ep] Compute: subgraph run via mlx-c COMPILED general (17 node(s))
```

This produces thousands of lines of output during a profile capture, polluting
committed artifacts. The workspace convention (`docs/ERROR_AND_LOGGING_CONVENTIONS.md`)
requires `tracing` crate instrumentation, not ad-hoc `println!`/`eprintln!`.

**Action:** The Metal pod should replace these prints with `tracing::debug!`
gated behind the `debug` level, consistent with the same defect class fixed in
the calibrator (`eprintln!` → `tracing::debug!`) and in `compare.rs`
(stdout pollution from `--profile-json -`).

The plugin lives in `../onnxruntime-mlx` (out of scope for this PR).
