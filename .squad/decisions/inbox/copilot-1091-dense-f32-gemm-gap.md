# EARLY NOTE (#1091): dense f32 GEMM — measured gap + mechanism (before writing a kernel)

**Host:** RTX 4060 laptop, 20 logical CPUs, AVX2+FMA+F16C+AVX-VNNI, no AVX-512, CPU only.
**Measured in one binary (`--features native-backend,mlas`), same-binary A/B via the existing
`NXRT_CPU_GEMM_BACKEND` toggle (`mlas` vs `simd`), process via `gemm_with_backend`, min-of-20, wall GFLOP/s.**

## The gap I actually measured (NOT #1045's 4.4x)

| shape (M×K×N) | M | simd/mlas |
| --- | --- | --- |
| 1×5120×5120 | 1 | **4.59x** |
| 1×5120×7168 | 1 | 3.58x |
| 1×5120×13824 | 1 | 3.44x |
| 1×13824×5120 | 1 | 2.49x |
| 1×5120×152064 (lm_head) | 1 | 2.23x |
| 128×5120×5120 | 128 | 0.88x (simd faster) |
| 128×5120×13824 | 128 | 0.87x (simd faster) |
| 128×13824×5120 | 128 | 1.15x |

**#1045's 4.4x prefill dense-f32 gap does NOT reproduce on this host.** At M=128 our built-in
`SimdX86` 6×16 packed microkernel is already at parity with MLAS (0.87–1.15x, net slightly favoring
SimdX86). #1045's 4.4x was on an AMD EPYC without AVX-512; here it's gone.

**The entire reproducible gap is at M=1 (decode GEMV): 2.2–4.6x.**

## Mechanism (source-cited, both sides)

MLAS `sgemm.cpp:1108-1130`:
> "Handle the special case of a small M. The data from matrix B is not referenced multiple times,
> so using a local packed buffer is a wasted memory copy."
> `if (M == 1 && ...) { SgemmKernelM1Routine(A, B, C, K, N, ldb, beta); return; }`
i.e. MLAS routes M==1 to `SgemmKernelM1Avx.asm`, which reads B **in place** at stride `ldb`
(16 columns/register-tile, K unrolled ×4) — no pack, **no resident buffer**.

Ours (`x86_sgemm.rs::sgemm_simd`): calls `pack_b` into a `bpack` scratch **unconditionally**. At M=1
there is a single A-panel, so each packed B panel is reused **zero** times — the pack is a wasted
full copy of B (K·N f32). That extra read+write of all of B (≈3× the memory traffic of a straight
GEMV) is the whole 2.2–4.6x. It is memory traffic, not arithmetic and not layout.

## How much is reachable without a resident copy

**All of it.** The fix is a dedicated M=1 GEMV that streams B in place (each 64-byte cache line read
once, fully consumed by a 16-col register tile), C in registers, A broadcast — **zero added
footprint** (it actually *removes* the `bpack` allocation at M=1). Exactly the #1104-desirable
property. No `GovernedWeightCache`, no `OnceLock`, nothing to admit/decline.

## Plan

Port MLAS's M1 mechanism natively into `SimdX86`: register-tiled column-blocked GEMV, no pack.
Same-binary A/B toggle `ONNX_GENAI_CPU_MM_SIMD_M1_GEMV` (default off, like #1104's
`ONNX_GENAI_CPU_MM_INT4_NBLK`). Numerics change accumulation order vs the current SimdX86 packed
path (a few ULP), so NOT byte-identical to it — will validate against the Generic/f64 reference and
quantify. No int4 model exercises this path, so measurement is a synthetic in-binary driver
(`bench_f32_gemm_ab`); reported honestly as such.

---

# RE-MEASUREMENT (#1116 review): control-gated, process CPU time

The reviewer's key catch: on the first wall-clock run the **M=128 control rows moved** (they *cannot*,
being an M==1-only route), proving up-to-1.5x background noise. Two methodology fixes adopted here:

**1. The M=128 rows are a built-in control, reported every run.** For `m >= 2` both simd arms run the
*identical* packed kernel, so `gemv/packed` must be ~1.0; if it drifts past a threshold the machine
was not quiet and **no conclusion is drawn**. `bench_f32_gemm_ab` prints this control + verdict on
every run. Recommended as the standard for every A/B harness in this repo.

**2. Gate the control on process CPU time, not wall clock.** On this shared box the 20-thread M=128
GEMMs have a ~25% *wall-clock* noise floor (min-of-N never clears sustained background load), so the
in-process wall control kept — correctly — refusing to certify. Process CPU time reproduces to ~2-4%
under the same contention (the reviewer's own 39/26/16 s-wall vs ~2% CPU observation). So the
authoritative numbers come from per-arm isolated processes measured by `TotalProcessorTime`, with the
M=128 control required to match across arms.

## Clean, control-passing tables (process CPU seconds, this host)

**CONTROL (M=128 prefill, identical code in every arm) — 2 reps:**

| rep | simd_packed | simd_gemv | mlas | gemv/packed drift | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | 80.06 | 81.77 | 79.75 | 2.1% | OK |
| 2 | 79.25 | 82.14 | 74.98 | 3.6% | OK |

Both < 5% ⇒ runs usable. Peak RSS 293 MB (packed==gemv), mlas 286 MB. **M=128 untouched** and our
packed prefill already ≈ MLAS (confirms #1116: prefill is at parity, the whole gap is M=1).

**DECODE (M=1) aggregate CPU s — 2 reps:** gemv/packed = 0.33x, 0.36x (≈3x faster than our default);
gemv/mlas = 0.85x, 0.93x (gemv edges *ahead* of MLAS in aggregate); peak RSS 2977 vs 2982 MB
(identical — zero footprint, `bpack` removed at M=1).

**DECODE per-shape CPU s (gemv/packed, gemv/mlas):**

| shape (M=1) | mlas | packed | gemv | gemv/packed | gemv/mlas |
| --- | --- | --- | --- | --- | --- |
| 1×5120×7168 (QKV) | 5.20 | 7.08 | 3.55 | **0.50** | 0.68 |
| 1×5120×5120 (o_proj) | 2.34 | 6.47 | 2.17 | 0.34 | 0.93 |
| 1×5120×13824 (gate/up) | 6.95 | 12.52 | 5.33 | 0.43 | 0.77 |
| 1×13824×5120 (down) | 4.59 | 13.31 | 5.63 | 0.42 | 1.22 |
| 1×5120×152064 (lm_head) | 37.28 | 130.19 | 44.03 | 0.34 | 1.18 |

## Adjudication of `1x5120x7168` (review item 3)

On robust CPU time it is **gemv/packed = 0.50 (2x faster than packed) and gemv/mlas = 0.68 (1.5x
faster than MLAS)** — one of the *best* rows, not a loss. The earlier "1.71x → 4.52x" was wall-clock
noise on a busy box, which the control now catches *before* concluding. **gemv strictly dominates the
packed default on all five M=1 shapes**, so there is deliberately no dispatch fall-back to packed
(it would be uniformly slower). The only residual gap is versus MLAS — not our shipping default — on
the two largest shapes (down_proj K=13824, lm_head N=152064: 1.18–1.22x), documented as a known
boundary at the dispatch site in `sgemm_simd_variant`.

## Numerics (unchanged stance)

Toggle stays **default-off**. GEMV reassociates the f32 sum vs the packed path (not byte-identical),
but matches the f64/Generic reference within `1e-3·(1+|e|)` — the same tolerance the existing SimdX86
tests use. "Default off, numerically different"; flipping it on is a separate future decision that
must restate the deviation.
