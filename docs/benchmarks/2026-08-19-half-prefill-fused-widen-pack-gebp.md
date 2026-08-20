# f16/bf16 prefill: fused widen-pack GEBP for the native CPU path

**Date:** 2026-08-19
**Host:** AMD EPYC 9V74, 32 vCPU / 16 physical cores. AVX2 + FMA + F16C only — no AVX-512,
no AVX512-BF16, no VNNI, no AMX. Linux.
**Build:** `--release`, default features (**`mlas` off**, which is the shipped default).

## What was wrong

`MatMul` with contiguous `f16`/`bf16` operands lands on the blocked half GEMM
(`kernels/half_gemm.rs`). That kernel splits its work **only over rows of C**, and each row block
re-widens and re-packs the whole of `B`:

```
split_mc = m.div_ceil(threads * 2).clamp(1, MAX_MC)
```

At `m = 64` on a 32-thread host that evaluates to `1`, so a 64-token prefill makes **64 full passes
over the weight**. The cost is structural, not microkernel quality — the same defect class as the
int4 prefill path in #1117.

The MLAS build does not have this problem: #1080 routes constant-weight f16 prefill through MLAS
SGEMM on a once-widened, once-packed `B`. That gate is `#[cfg(feature = "mlas")]`, and `mlas` is
**off by default**, so the default native build has been on the row-blocked path the whole time.
This work gives the native build its own answer.

## What changed

`x86_sgemm::half_prefill_gebp`: widen `B` **directly into** the L1 `KC x NR` f32 panel that the
existing `micro_6x16` microkernel already consumes. The weight is traversed once per column strip
regardless of `m`, and the tuned f32 microkernel, the packing layout and the strip parallelism are
all reused unchanged. `A` is widened once into an `m*k` f32 transient (bounded by the activation,
not the weight) and packed by the shared `pack_a`.

No f32 copy of `B` is ever materialized or retained, so this adds **no weight-derived cache** —
unlike the MLAS route, which keeps a widened `4*K*N`-byte `B`.

Because the panels are element-for-element what `pack_a`/`pack_b` produce from the widened
operands, the result is **bit-identical** to `sgemm_simd(widen(a), widen(b))` for every finite
operand. That is asserted with `assert_eq!`, not a tolerance
(`half_gebp_matches_the_widened_f32_kernel`).

The one exception is a `bf16` **signalling** NaN: widening by shift keeps the payload, while
`half::bf16::to_f32` canonicalizes it to a quiet NaN, so 126 of the 65536 `bf16` patterns widen to a
different NaN *encoding*. No finite value differs, NaN still propagates as NaN, and the same shift is
what the blocked half GEMM already used — so this is inherited behaviour, not new.
`widening_matches_the_half_crate_over_the_whole_domain` sweeps all 65536 patterns of both formats
and pins the count at 0 for `f16` and exactly 126 for `bf16`, so the divergence is written down
rather than left to be discovered.

Gate: `k*n >= 1_048_576` elements, and `m >= 2` (`bf16` also at `m == 1`, see below).
`ONNX_GENAI_CPU_MM_HALF_GEBP=0` restores the blocked route for the whole process.

## Production A/B

`cargo bench -p onnx-runtime-ep-cpu --bench half_prefill_route_ab`, run twice in the same build —
default (fused) and `ONNX_GENAI_CPU_MM_HALF_GEBP=0` (blocked). Every number is the **production**
kernel through `ExecutionProvider::get_kernel` + `Kernel::execute`, p50, 2 warmups then 5 reps for
steady, fresh kernel per rep for cold.

`k = 4096, n = 11008` (a Llama-class MLP weight):

| dtype | m | blocked steady ms | fused steady ms | gain | blocked cold | fused cold |
|---|---|---:|---:|---:|---:|---:|
| f16 | 8 | 36.20 | **2.59** | **14.0x** | 36.58 | 2.68 |
| f16 | 64 | 118.33 | **7.50** | **15.8x** | 124.76 | 7.80 |
| f16 | 256 | 169.41 | **26.27** | **6.4x** | 172.49 | 25.66 |
| bf16 | 1 | 31.36 | **1.96** | **16.0x** | 31.46 | 2.33 |
| bf16 | 8 | 32.19 | **2.53** | **12.7x** | 30.71 | 2.62 |
| bf16 | 64 | 108.79 | **7.66** | **14.2x** | 113.69 | 7.19 |
| bf16 | 256 | 135.08 | **26.34** | **5.1x** | 133.99 | 27.79 |

`k = n = 2048`:

| dtype | m | blocked steady ms | fused steady ms | gain |
|---|---|---:|---:|---:|
| f16 | 8 | 2.74 | **0.47** | **5.8x** |
| f16 | 64 | 7.74 | **1.71** | **4.5x** |
| f16 | 256 | 11.01 | **4.67** | **2.4x** |
| bf16 | 1 | 1.31 | **0.36** | **3.7x** |
| bf16 | 8 | 2.30 | **0.55** | **4.2x** |
| bf16 | 64 | 6.93 | **1.59** | **4.4x** |
| bf16 | 256 | 9.72 | **4.60** | **2.1x** |

Cold and steady agree in both arms, as they must: neither route caches anything.

### The f32 control

The same harness runs the identical shapes in `f32`, which **neither arm can move**. It does not
move: `4096x11008` f32 steady is 3.84 / 7.42 / 23.07 ms (m = 8 / 64 / 256) with the fused route on
and 3.81 / 7.64 / 23.25 ms with it off — inside noise, and the f32 output digests are identical
across the two processes. So the half-route difference above is the route, not the machine.

It is also the scale to read the result against: after the change `f16` at `m = 64` is 7.50 ms
against f32's 7.42 ms (parity), and at `m = 8` it is **faster** than f32 (2.59 vs 3.84 ms) because
the operands are half the bytes. At `m = 256` it is 26.27 vs 23.07 ms — the residual ~14% is the
widen-pack of `B` that f32 does not pay.

### Conformance

The harness reports `max_rel`, the largest relative deviation from the same GEMM run in `f32`
through the same production kernel on the *same* (already narrowed) operand values. The two half
routes sum in different orders, so their digests differ by design; what matters is that the fused
route is no further from the f32 answer than the blocked route it replaces. It is not — the two
arms agree to three significant figures on every row:

| dtype | shape | blocked max_rel | fused max_rel |
|---|---|---|---|
| f16 | 2048², m=8..256 | 2.87e-5 | 2.87e-5 |
| f16 | 4096x11008, m=8..256 | 4.81e-4 … 4.86e-4 | 4.81e-4 … 4.86e-4 |
| bf16 | 2048², m=8..256 | 2.29e-4 … 4.33e-4 | 2.29e-4 … 4.33e-4 |
| bf16 | 4096x11008, m=8..256 | 3.86e-3 … 3.88e-3 | 3.86e-3 … 3.88e-3 |

The f32 control rows read exactly `0`.

## Where the gate came from

`bench_half_prefill_gebp_crossover` (ignored test in `matmul.rs`, run with `--release`) times the
two routes directly, interleaved rep-by-rep, touching no environment variable. Ratio is
`blocked / gebp`, `T=32 / T=4`, f16:

| K x N | elements | m=2 | m=4 | m=8 | m=16 |
|---|---|---|---|---|---|
| 256 x 256 | 65_536 | 0.60 / 0.77 | 0.59 / 0.86 | 0.56 / 1.02 | 3.42 / 1.09 |
| 512 x 512 | 262_144 | 1.18 / 1.84 | 3.76 / 2.39 | 3.59 / 3.18 | 4.21 / 1.29 |
| 768 x 768 | 589_824 | 4.42 / 3.08 | 5.44 / 3.39 | 3.84 / 3.71 | 5.76 / 1.64 |
| **1024 x 1024** | **1_048_576** | **5.01 / 3.36** | **7.69 / 3.42** | **6.61 / 4.24** | **4.67 / 1.95** |
| 1536 x 1536 | 2_359_296 | 6.79 / 3.59 | 6.63 / 5.53 | 7.67 / 5.53 | 7.38 / 2.47 |
| 2048 x 2048 | 4_194_304 | 6.58 / 3.47 | 7.37 / 3.15 | 6.39 / 6.79 | 8.69 / 3.53 |

End to end through the production kernel the ratios are damped by the per-call fixed cost, and
`512 x 512` *loses* at `T=32` for `m = 2..8` (0.63x-0.98x) while still winning at `T=4`.
`1_048_576` is the smallest weight that wins in **both** harnesses, at both thread counts, in both
formats — so that is the gate. `512 x 512` and `768 x 768` are left on the blocked route rather
than claimed.

## `M = 1`: why the two formats differ

`f16` has a dedicated decode GEMV (`half_gemv::gemv_f16_kn`) that intercepts contiguous 2-D `m = 1`
before this tile is reached, so `f16` decode is **bit-for-bit unchanged**; the `f16 m=1` rows in the
A/B move by less than run-to-run noise because both arms take that GEMV.

`bf16` has no such GEMV. A single `bf16` decode row falls onto the blocked half GEMM and pays a full
widen-pack of `B` regardless — 31.4 ms for **one token** at `k=4096, n=11008`. So `bf16` takes the
fused route at `m == 1` too, for a 16x gain. A native `bf16` GEMV (the analogue of
`half_gemv::gemv_f16_kn`, reading `B` in place with no packing at all) should beat both and is left
as separate work.

## Remaining losses

* **`bf16` decode has no GEMV.** 1.96 ms per token at `k=4096, n=11008` is now GEMM-shaped work
  where a GEMV would be bandwidth-bound; the packed route reads and writes a panel it uses once.
* **~14% behind f32 at `m = 256`** — the widen-pack of `B`. A prepacked, session-lifetime widened
  panel cache would remove it, at `4*K*N` resident bytes per weight, which is exactly the residency
  the borrowed paths exist to avoid. Not taken.
* **AVX512-BF16 hosts are untouched.** `x86_bf16::native_available()` still wins the dispatch there,
  and that kernel's own row-blocking has not been measured — this host cannot run it.
* **aarch64 is untouched.** The kernel is x86-64 (it reuses the AVX2/FMA microkernel); the NEON half
  path keeps its current behaviour.
