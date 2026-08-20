# `Gemm`'s `bf16` decode was excluded by dtype, and it cost 3.0x-24x

**Date:** 2026-08-21
**Host:** AMD EPYC 9V74, 32 vCPU / 16 physical cores, AVX2 + FMA + F16C. No AVX-512,
no AVX512-BF16, no VNNI, no AMX. Linux, `--release`, default features (no `mlas`).
**Kernel:** `Gemm`, `M = 1` (decode), contiguous 2-D `[1, K] x [K, N]`, `bf16` storage.
**Harness:** `bench_gemm_half_decode_route` in `crates/onnx-runtime-ep-cpu/src/kernels/gemm.rs`,
driven through `Kernel::execute` so the dispatch decision is part of what is timed.

## The gap

`GemmKernel::try_half_fast_path` opened with

```rust
if a.dtype != DataType::Float16 || b.dtype != DataType::Float16 {
    return Ok(None);
}
```

so a `bf16` `Gemm` never reached the decode GEMV, no matter the shape. `MatMul` has served
`bf16` decode from that same GEMV since 2026-08-19 (`half_storage_format` admits
`BFloat16`). The identical decode therefore got a different kernel depending on which op
the exporter emitted -- and the one `Gemm` got is the portable blocked half GEMM, the
slowest dense region measured anywhere in this EP.

This was recorded as open in the
[2026-08-21 GEBP retirement](2026-08-21-half-decode-gebp-retired.md); this is the fix.

## The measurement

Both arms come out of **one build**, selected by `ONNX_GENAI_CPU_MM_HALF_GEMV`, which is a
process-wide `OnceLock` -- so one arm per process. With the knob off, `Gemm` declines the
GEMV and lands in the blocked half GEMM, which is exactly the pre-change `bf16` route.
`steady_ms` is the median of 25 executes after a cold one. Every row also runs the same
shape in `f32`, a path neither arm can move; `ctl` divides that out, so `corrected` is a
route difference and not machine drift.

### 8 threads (`RAYON_NUM_THREADS=8`)

| shape | `K x N` | before (ms) | after (ms) | raw | ctl | **corrected** |
|---|---|---|---|---|---|---|
| `k1024n768` | 1024x768 | 0.264 | 0.060 | 4.40x | 0.95 | **4.63x** |
| `k1024n1024` | 1024x1024 | 0.378 | 0.080 | 4.72x | 1.08 | **4.37x** |
| `k1024n2048` | 1024x2048 | 0.765 | 0.092 | 8.32x | 1.48 | **5.62x** |
| `k2048n1024` | 2048x1024 | 0.710 | 0.115 | 6.17x | 1.17 | **5.28x** |
| `k512n4096` | 512x4096 | 0.752 | 0.055 | 13.67x | 1.03 | **13.32x** |
| `qwen_qkv` | 3584x4608 | 9.974 | 0.523 | 19.07x | 0.88 | **21.66x** |
| `llama_mlp` | 4096x11008 | 36.066 | 1.798 | 20.06x | 0.83 | **24.07x** |
| `llama_qkv` | 4096x4096 | 9.854 | 0.470 | 20.97x | 0.98 | **21.34x** |

### 32 threads

| shape | before (ms) | after (ms) | raw | ctl | **corrected** |
|---|---|---|---|---|---|
| `k1024n768` | 0.261 | 0.063 | 4.14x | 1.09 | **3.79x** |
| `k1024n1024` | 0.364 | 0.086 | 4.23x | 1.03 | **4.11x** |
| `k1024n2048` | 0.725 | 0.155 | 4.68x | 1.14 | **4.11x** |
| `k2048n1024` | 0.747 | 0.115 | 6.50x | 1.00 | **6.50x** |
| `k512n4096` | 0.715 | 0.219 | 3.26x | 1.10 | **2.97x** |
| `qwen_qkv` | 8.385 | 0.639 | 13.12x | 0.85 | **15.38x** |
| `llama_mlp` | 34.845 | 1.671 | 20.85x | 0.93 | **22.33x** |
| `llama_qkv` | 7.644 | 0.644 | 11.87x | 1.06 | **11.22x** |

Every shape wins at both thread counts, and the win grows with the weight: the blocked GEMM
holds 2.5-6.0 GB/s of weight bandwidth regardless of shape, while the GEMV reaches
26-76 GB/s -- 50-100% of what the same shape's `f32` control gets, against a 75.8 GB/s DRAM
ceiling. The `f16` arm is unchanged and still lands on the GEMV (`llama_mlp` 1.752 ms
against `bf16`'s 1.798 ms at 8 threads, as it should be -- both read `2 * K * N` bytes).

## What changed

- The dtype gate now calls `matmul::half_storage_format`, the same helper `MatMul` uses, so
  the two ops admit exactly the same `(a, b)` dtype pairs.
- The decode arm passes that `HalfFormat` to `half_gemv::simd_available` and
  `half_gemv::gemv_half_kn` instead of hard-coding `F16`.
- `Gemm` now honours `ONNX_GENAI_CPU_MM_HALF_GEMV` as well. It did not, so the documented
  field kill-switch for this route silently only covered `MatMul` -- and there was no way to
  A/B the `Gemm` side of the shipped binary. Both are fixed by one term.

### The one asymmetry that stays

A **transposed** (`[N, K]`) `bf16` decode still declines. That is kernel coverage, not
policy: `gemv_f16_nk` reads `f16` bit patterns and has no `bf16` twin, so admitting it
would silently reinterpret the weight. It falls into the blocked GEMM exactly as before,
and `transposed_bf16_decode_declines_the_gemv_but_stays_correct` pins both the route and
the numerics. Writing that kernel is the obvious next increment; it is a separate change
with its own numerics burden.

## Correctness

`bf16_decode_takes_the_same_gemv_in_both_ops` asserts, at 64x64, 1024x1024 and 2048x3072,
that (1) the `Gemm` route counter shows the GEMV, (2) the `MatMul` counter does too,
(3) the two outputs are **bit-identical**, and (4) both are within 8e-3 of an `f64`
reference computed over the *bf16-narrowed* operand values -- so the only error the
tolerance absorbs is accumulation order, not the storage rounding.

**Mutation-verified**, both directions:

| mutation | test that fails |
|---|---|
| restore the `format != F16 -> None` gate | `bf16_decode_takes_the_same_gemv_in_both_ops` |
| drop the `!trans_b \|\| format == F16` term | `transposed_bf16_decode_declines_the_gemv_but_stays_correct` (on numerics) |

`a_multi_row_half_gemm_is_not_counted_as_decode` pins the instrument itself: the counter
must stay at 0 for an `m = 4` half `Gemm`, so a change that made it fire for prefill cannot
make the two tests above pass vacuously.

## Remaining losses

- Transposed `bf16` decode, above -- still on the blocked GEMM, so still ~4-21x off.
- The GEMV's own distance from roofline is unchanged by this: 26-76 GB/s against 75.8 GB/s,
  so the large shapes are close and the small ones are latency-bound. Separate work.
- Nothing here touches `bf16` **prefill** (`m > 1`), which still uses the blocked half GEMM
  in both ops.

## Reproducing

```bash
for arm in "" ONNX_GENAI_CPU_MM_HALF_GEMV=0; do
  env $arm PROBE_DTYPE=bf16 RAYON_NUM_THREADS=8 \
    cargo test -p onnx-runtime-ep-cpu --release --lib \
    bench_gemm_half_decode_route -- --ignored --nocapture
done
```

Drop `RAYON_NUM_THREADS` for the 32-thread rows; drop `PROBE_DTYPE` for the `f16` control.
