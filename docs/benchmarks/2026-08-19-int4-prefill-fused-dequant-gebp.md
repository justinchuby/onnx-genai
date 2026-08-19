# int4 prefill: fusing the dequant into the packed-panel GEMM

**Date:** 2026-08-19
**Issue:** #1117
**Host:** 32-core x86_64 (AVX2 + FMA), Linux, contended (other jobs resident)
**Build:** `--release`, default features (`mlas` **off**), pure-native CPU EP
**Method:** `cargo bench -p onnx-runtime-ep-cpu --bench int4_prefill_route_ab` —
the real kernel via `ExecutionProvider::get_kernel` + `Kernel::execute`, arms
selected by environment inside **one** build. Medians of 7 timed calls per arm
per shape, and the reported figure is the median of 3 interleaved arm-pairs.

## What was wrong

`borrowed_affine_int4_matmul` visits activation rows outer-most, so each of the
`m` prompt rows makes a complete pass over the packed weight. Prefill therefore
had the arithmetic intensity of `m` independent decodes: it gained nothing from
batching, and #1117 measured it costing 2.3x *more* CPU per token than decode,
which is backwards.

The row-blocking added in #1126 (`ONNX_GENAI_CPU_MM_INT4_PREFILL`) removed the
per-row fork/join and the per-row allocation but deliberately kept rows
outer-most, so the weight traffic was unchanged. Measured here, that knob is
worth ~1.09x — real, but not the shape of the problem.

## The measurement that framed the fix

Three routes, same shape (`k=4096, n=11008`, block 32, symmetric int4), same
build, steady state:

| m | row-serial borrowed | +#1126 blocking | dense f32 (resident weight) |
|---:|---:|---:|---:|
| 8 | 17.8 ms — 40.4 GFLOP/s | 16.7 — 43.3 | 3.6 — 198 |
| 64 | 152.5 — 37.8 | 131.6 — 43.8 | 7.8 — 741 |
| 256 | 573.8 — 40.2 | 529.7 — 43.6 | 23.2 — 996 |

(The dense arm here is measured against the #1176 branch, where the dense route
keeps its dequantized weight instead of rebuilding it per call.)

A 25x gap, and the dense column shows where it goes: the machine will do ~1000
GFLOP/s on this shape once B is a packed panel. The dense route buys that by
materializing an f32 `[n, k]` weight — 8x the packed bytes resident, exactly the
residency #979 removed. So the question was never "GEMM or borrow", it was
whether the panel can be built from the *packed* weight.

## The fix

Fuse the dequantization into `pack_b`. Each `KC x NR` panel is expanded from
packed nibbles straight into the L1-resident f32 panel the `6x16` microkernel
already consumes, and every one of the `m` rows then reuses that panel. Each
packed byte is read once per call, no f32 weight is ever resident, and the
panels are bit-identical to what the f32 SGEMM would have packed from a
dequantized weight (asserted in
`int4_gebp_matches_the_sgemm_on_the_dequantized_weight`).

## Result

`k=4096, n=11008`, block 32, symmetric int4:

| m | row-serial (previous default) | fused GEBP | speedup |
|---:|---:|---:|---:|
| 8 | 18.04 ms — 40.0 GFLOP/s | **6.04 — 119.5** | **3.0x** |
| 64 | 144.1 — 40.0 | **9.63 — 599.6** | **15.0x** |
| 256 | 587.8 — 39.3 | **24.76 — 932.3** | **23.7x** |

Cold (a fresh kernel per repetition, i.e. what time-to-first-token actually
pays) tracks steady state within noise on the fused arm — 5.9/9.9/24.5 ms — as
it must, because the route caches nothing. The previous route's cold and steady
columns also agree, for the same reason; the win is not a caching artifact.

At `m=256` the fused route lands within ~7% of the dense f32 ceiling (24.8 vs
23.2 ms) **while keeping the weight borrowed**.

## Thread pool

Prefill runs on the global pool, not the decode pool. Dispatched inside
`with_decode_pool` the same kernel measured 61.2 ms at `m=256` against 24.8 ms
outside it — 2.4x, purely from the decode pool's deliberately narrow default
(the flat pool caps at eight workers, sized for per-token latency). A prefill
GEMM is the opposite regime: one bulk parallel region per call that wants the
whole machine, which is the policy the f32 dense fallback already uses.

## Crossover

The panel only pays for itself once enough rows reuse it, so the route is gated
on `m >= 4`:

| m | k=2048,n=2048 | k=4096,n=11008 |
|---:|---|---|
| 2 | 0.83 -> 1.11 ms (**regresses**) | 8.21 -> 7.80 ms |
| 4 | 1.64 -> 1.32 ms | 16.4 -> 7.51 ms (2.2x) |
| 8 | 3.36 -> 1.14 ms (2.9x) | 33.5 -> 8.42 ms (4.0x) |

`m = 4` is the lowest row count that never loses. Below it the row-serial path
stays.

## Scope and caveats

* x86_64 with AVX2 + FMA only (the packed microkernel's contract). Other
  targets keep the row-serial path unchanged.
* Reached from the `bits=4, accuracy_level=0` borrowed route — symmetric and
  asymmetric both, since the zero point is applied during the pack.
* `ONNX_GENAI_CPU_MM_INT4_GEBP=0` restores the previous route.
* The host was contended, so absolute times are pessimistic; both arms were
  interleaved within each round so the ratios are not.
* No end-to-end TTFT number is claimed here: this host has no model weights, so
  every figure is the kernel measured through the production entry point, not a
  generation run.
