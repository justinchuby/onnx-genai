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
