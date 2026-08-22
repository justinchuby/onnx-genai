# A transposed `bf16` decode was 21x-101x off, because the `[N, K]` GEMV existed only in an `f16` spelling (2026-08-21)

**Date:** 2026-08-21 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c/2t), AVX2+FMA+F16C, **no AVX-512/VNNI**, L3 64 MiB, 75.8 GB/s DRAM,
shared. 8 rayon threads.

## Verdict

**Merged.** `Gemm` with `transB = 1` on a `bf16` weight declined the decode GEMV
and fell to the portable path: **0.038 ms -> 0.810 ms** at `k=1024, n=768` and
**3.42 ms -> 201.9 ms** at a 896x151936 `lm_head`. The fix is not a new kernel.
The `[N, K]` GEMV was already the right kernel; it was written once in an `f16`
spelling, so a `bf16` weight had nowhere to go.

## What this closes

§15 of the ledger closed `Gemm`'s *untransposed* `bf16` decode and recorded what
it could not close:

> **Still asymmetric:** a *transposed* `bf16` decode declines, because
> `gemv_f16_nk` reads `f16` bit patterns and has no `bf16` twin — kernel
> coverage, not policy.

That reading was correct, and it is the whole bug. The `[K, N]` stripe kernel had
already been made per-format by a macro (`stripe_simd_fn!`, instantiated as
`stripe_simd_f16` / `stripe_simd_bf16`) precisely because `#[target_feature]` is
an attribute on a concrete function and `bf16` must not be made to ask for the
`f16c` unit it does not use. The `[N, K]` row kernel never got the same
treatment. So `trans_b` was selecting on **dtype** when it should have been
selecting on **layout**.

The change instantiates the `[N, K]` row dot from one macro the same way
(`dot_row_simd_f16` / `dot_row_simd_bf16`), threads `HalfFormat` through
`gemv_f16_nk` -> `gemv_half_nk` and `dot_row_scalar`, and deletes the
`(!trans_b || format == HalfFormat::F16)` term from the decode gate. Both
operators, both stored orders and both 16-bit formats now reach the same
backend.

## Measurement

`crates/onnx-runtime-ep-cpu/benches/half_decode_gemv_ab.rs`, the production A/B
harness, extended with `PROBE_OP=gemm_transb` — a `Gemm` node carrying
`transB = 1` against an `[N, K]` weight. **The transposed route had no benchmark
row before this**, which is exactly how a `bf16` decode sat on the portable path
unmeasured. The weight is generated in `[K, N]` and transposed into `[N, K]`, so
the values, the `f64`-referenced `max_rel` and the digest are comparable to the
untransposed row of the same shape.

Arms are selected by the documented field kill-switch
`ONNX_GENAI_CPU_MM_HALF_GEMV`, one binary, `PROBE_SHAPE=full`, `m = 1`, steady
p50 of 7 reps after 2 warmups. **`GEMV=0` reproduces the old transposed `bf16`
behaviour exactly**, because that route declined unconditionally before this
change.

### `bf16`, `transB = 1`, steady ms

| shape | `k` | `n` | before (portable) | after (GEMV) | speedup | after GB/s |
|---|---:|---:|---:|---:|---:|---:|
| k1024n768 | 1024 | 768 | 0.810 | **0.038** | **21.3x** | 41.0 |
| k2048n2048 | 2048 | 2048 | 4.391 | **0.073** | **60.2x** | 114.6 |
| k4096n1024 | 4096 | 1024 | 4.378 | **0.078** | **56.1x** | 107.3 |
| k2048n4096 | 2048 | 4096 | 8.830 | **0.103** | **85.7x** | 163.1 |
| k4096n2048 | 4096 | 2048 | 8.830 | **0.108** | **81.8x** | 155.3 |
| k4096n4096 | 4096 | 4096 | 19.659 | **0.195** | **100.8x** | 171.8 |
| k4096n8192 | 4096 | 8192 | 42.479 | **0.635** | **66.9x** | 105.7 |
| k4096n11008 | 4096 | 11008 | 57.491 | **0.942** | **61.0x** | 95.7 |
| k896n151936 | 896 | 151936 | 201.902 | **3.424** | **59.0x** | 79.5 |

`f16` transposed rows move by the same order in this table, but that is **not**
a claim of this change: `f16` already took the GEMV, so its `GEMV=0` arm is a
counterfactual that never shipped. It is reported only to show the two formats
now land on the same kernel — `f16` 0.042/0.073/0.074/0.103/0.100/0.193/0.603/
0.904/3.298 ms against `bf16` 0.038/0.073/0.078/0.103/0.108/0.195/0.635/0.942/
3.424 ms, i.e. within a few percent at every shape, which is what "one backend"
should look like.

### Null control

`f32` rows run the same harness and never touch the half GEMV, so the switch must
not move them. Measured across the two arms:

| shape | ratio (off/on) |
|---|---:|
| k1024n768 | 0.862 |
| k2048n2048 | 0.795 |
| k4096n1024 | 0.799 |
| k2048n4096 | 0.985 |
| k4096n2048 | 1.072 |
| k4096n4096 | 1.008 |
| k4096n8192 | 1.128 |
| k4096n11008 | 1.191 |
| k896n151936 | 1.264 |

**The null band is 0.795-1.264, i.e. +-26%** on a shared host — wide, and stated
rather than hidden. The effect being claimed is 21x-101x, two orders of magnitude
outside it. No cell of this change's verdict rests on a margin near the band.

### On the GB/s column

Several rows read above the host's 75.8 GB/s DRAM figure (up to 171.8), and the
reason is **not** the same in every row. The mid-size rows are cache-resident: a
4096x4096 `bf16` weight is 33.6 MB against a 64 MiB L3, so after the warmups DRAM
is not the binding constraint. The 896x151936 `lm_head` is **272 MB** and cannot
be L3-resident at all — its 79.5 GB/s simply exceeds the quoted DRAM figure by
~5%, which says the 75.8 GB/s number is approximate rather than that anything
beat it. Neither row is a roofline violation, and neither is evidence of one. The same denominator error is what
[`2026-08-20-int4-nibble-i16-negative.md`](2026-08-20-int4-nibble-i16-negative.md)
committed in the other direction, and it is flagged here for the same reason:
**a bandwidth percentage means nothing until the denominator is shown to bind.**

## Correctness

Reading a `bf16` weight through the `f16` kernel does not fault — it silently
reinterprets every bit pattern — so a route assertion alone would be worthless
here. Three pins, all mutation-checked against exactly that mutation
(`gemv_half_nk` forced to `dot_row_simd_f16`):

| test | catches |
|---|---|
| `a_transposed_bf16_decode_takes_the_same_gemv_as_f16` | route **and** `f64`-referenced numerics |
| `kn_and_nk_agree_on_the_same_bf16_weight` | the two stored orders disagreeing |
| `nk_simd_and_scalar_rows_agree_bit_for_bit` (now both formats) | vector vs documented reduction order |

Forcing the `f16` kernel fails the first two; the suite is otherwise green, which
is the point — before this change nothing in the tree would have caught it.

The scalar fallback widens with `widen_scalar(format, ..)`, the same shift the
vector path uses, so the two agree **bit for bit** in either format, signalling
NaN included. That is asserted, not assumed.

## What this does and does not establish

**Does:** a transposed `bf16` decode now reaches the same `[N, K]` GEMV an `f16`
one does, at 21x-101x, with the numerics pinned against an `f64` reference.

**Does not:** say anything about `m > 1`, which is prefill and is not routed
here; about non-x86 hosts, where this module does not build; or about whether the
GEMV is *optimal* — it reaches 79-172 GB/s against an L3-resident weight, and no
attempt was made here to establish what the L3 ceiling actually is.

## Reproduce

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
BIN=$(ls -t target/release/deps/half_decode_gemv_ab-* | grep -v '\.d$' | head -1)
RAYON_NUM_THREADS=8 PROBE_OP=gemm_transb PROBE_SHAPE=full $BIN                        # after
RAYON_NUM_THREADS=8 ONNX_GENAI_CPU_MM_HALF_GEMV=0 PROBE_OP=gemm_transb PROBE_SHAPE=full $BIN  # before
```
