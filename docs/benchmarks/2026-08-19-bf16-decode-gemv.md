# A decode GEMV for `bf16`, and the weight size where packing beats not packing

**Date:** 2026-08-19
**Host:** AMD EPYC 9V74, 32 vCPU / 16 physical cores, AVX2 + FMA + F16C. No AVX-512, no
AVX512-BF16, no VNNI, no AMX. Linux, `--release`, default features (no `mlas`).
**Kernel:** `MatMul`, `M = 1` (decode), contiguous 2-D `[1, K] x [K, N]`, 16-bit storage.
**Harness:** `crates/onnx-runtime-ep-cpu/benches/half_decode_gemv_ab.rs`, driven through
`ExecutionProvider::get_kernel` + `Kernel::execute` — the dispatch decision and the output
narrowing are inside the timed region, because production pays for both.

## What was wrong

`f16` decode has had a dedicated GEMV since #1082: at `M = 1` no packed panel of `B` is reused, so
packing is pure overhead on a problem that is purely memory-bound. `half_gemv::gemv_f16_kn` reads
the weight in place in its stored `[K, N]` order and allocates nothing.

`bf16` had no such path. A single decode token on a `bf16` weight fell through to
`try_matmul_half`, and from there either to the fused prefill GEBP (#1365, on weights of at least
1M elements) or, below that, to the row-blocked half GEMM — which widens and packs the entire
weight to multiply it by one row.

## Arms

All three are reachable from **one build**, by environment alone, so nothing here is a
cross-build comparison:

| arm | how | route at `M = 1` |
|---|---|---|
| `GEMV` | `ONNX_GENAI_CPU_MM_HALF_GEBP=0` | read `B` in place, `[K, N]` order, no packing, no copy |
| `GEBP` | `ONNX_GENAI_CPU_MM_HALF_GEMV=0` (`bf16`), threshold (`f16`) | widen `B` into packed L1 panels |
| `blocked` | both `=0` | the row-blocked half GEMM |
| `shipped` | default | GEMV under 33.6M weight elements, GEBP at or above |

An `f32` control row runs the same shapes through the f32 path in every arm. No arm can move it,
so it is the check that a difference between the half arms is the route and not the machine.

## The result, and the direction it went

The hypothesis was that a GEMV — one pass over the weight, nothing packed — must be the floor at
`M = 1`, so the work started as "give `bf16` the GEMV `f16` already has". That is true at small and
medium weights and **false at large ones**. Steady p50, 7-9 interleaved repetitions per arm,
minimum of the same samples in parentheses where the two statistics disagree:

| dtype | `K x N` | elements | GEMV | fused GEBP |
|---|---|---:|---:|---:|
| f16 | 4096x1024 | 4.2M | **0.39 ms** | 0.59 ms |
| bf16 | 4096x2048 | 8.4M | **0.86 ms** (0.51) | 0.71 ms (0.59) |
| f16 | 4096x4096 | 16.8M | 1.13 ms | **1.06 ms** |
| f16 | 4096x8192 | 33.6M | 1.92 ms | 1.91 ms |
| bf16 | 4096x8192 | 33.6M | 1.87 ms | 2.01 ms |
| f16 | 4096x11008 | 45.1M | 2.47 ms (2.95) | **2.46 ms** (2.16) |
| bf16 | 4096x11008 | 45.1M | 3.32 ms | **2.45 ms** |
| f16 | 896x151936 | 136M | 6.94 ms | **5.84 ms** |
| bf16 | 896x151936 | 136M | 6.61 ms | **5.56 ms** |

The mechanism is parallelism, not traffic. The GEMV hands each worker a `STRIPE = 512` column slice
and has that worker walk all `k` rows: at `n = 11008` that is 21 stripes on a 32-thread pool, so a
third of the cores idle while the rest issue `k` strided visits each. The GEBP's extra panel writes
are L1-resident and its column strips keep every worker on a contiguous run. Below ~32M elements
the GEMV's zero-overhead traversal is worth more than the parallel efficiency; above it, it is not.

## Where the threshold went, and why not at the win

`HALF_DECODE_GEBP_MIN_WEIGHT = 1 << 25` (33.6M elements — a 64 MiB `f16` weight).

That is the **wash**, not the first win: at exactly 33.6M the two routes are inside run-to-run
noise in both formats, and only above it does the GEBP pull clearly ahead. Placing the threshold at
the wash means no shape is left on the slower route, and no shape is moved onto the allocating
route for a gain that could not be measured. `1 << 24` would have claimed 16.8M on a 6% median
difference that this host cannot resolve; `1 << 26` would have left `4096 x 11008` — a 7B-class MLP
projection — on the slower arm.

## Shipped routing, and what it changes

| case | before | after | change |
|---|---|---|---|
| `f16` `M = 1`, weight < 33.6M | GEMV | GEMV | none |
| `f16` `M = 1`, weight >= 33.6M | GEMV | GEBP | wash to **1.28x** (lm_head) |
| `bf16` `M = 1`, weight < 1M | blocked | GEMV | **3.0x** |
| `bf16` `M = 1`, 1M <= weight < 33.6M | GEBP | GEMV | wash; tie broken toward the route that allocates nothing |
| `bf16` `M = 1`, weight >= 33.6M | GEBP | GEBP | none |

Full production A/B, 7 interleaved repetitions per arm, steady p50 (min):

| dtype | shape | `K x N` | shipped | GEMV | blocked | vs blocked |
|---|---|---|---:|---:|---:|---:|
| f32 | attn_out | 1024x768 | 0.121 (0.112) | 0.113 | 0.109 | control |
| f32 | square | 2048x2048 | 0.258 (0.154) | 0.414 | 0.395 | control |
| f32 | mlp | 4096x11008 | 4.999 (3.927) | 5.526 | 5.307 | control |
| f32 | lm_head | 896x151936 | 9.896 (8.713) | 9.859 | 10.262 | control |
| f16 | attn_out | 1024x768 | 0.133 (0.095) | 0.150 | 0.429 | **3.2x** |
| f16 | square | 2048x2048 | 0.439 (0.230) | 0.447 | 2.552 | **5.8x** |
| f16 | mlp | 4096x11008 | 3.570 (2.872) | 3.466 | 56.497 | **15.8x** |
| f16 | lm_head | 896x151936 | 7.466 (6.442) | 9.592 | 161.327 | **21.6x** |
| bf16 | attn_out | 1024x768 | 0.128 (0.101) | 0.147 | 0.386 | **3.0x** |
| bf16 | square | 2048x2048 | 0.450 (0.378) | 0.461 | 2.189 | **4.9x** |
| bf16 | mlp | 4096x11008 | 3.089 (2.278) | 5.973 | 46.449 | **15.0x** |
| bf16 | lm_head | 896x151936 | 6.452 (5.801) | 8.486 | 140.744 | **21.8x** |

The `f32` control rows move by up to 1.6x between arms on identical code. **That is the noise
floor of this host** — it is shared, and load average during the sweep ranged from 14 to 30. Every
ratio in the "vs blocked" column is far outside it; the 1.28x-1.32x `lm_head` gains are outside it
by median and by minimum across 7-9 repetitions but are not order-of-magnitude claims, and the
"wash" rows are called washes precisely because they are not.

## Assignment == execution

Timing alone cannot say which route ran — the arms agree to half-precision rounding. Two
independent checks do:

1. **Output digests.** The bench prints an order-sensitive digest per row. At `attn_out` and
   `square` the shipped digest equals the **GEMV** arm's digest in both formats; at `mlp` and
   `lm_head` it equals the **GEBP** arm's. That is the threshold, read off the values.
2. **Route counters.** `half_decode_gemv_calls()` and `half_prefill_gebp_calls()` are asserted in
   `decode_on_a_small_weight_keeps_the_gemv`, which also asserts the GEMV materialised neither a
   widened `B` nor a transposed weight cache.

`no_decode_is_handed_to_the_blocked_gemm` pins the invariant that makes the split safe: the two
gates live in different functions with different thresholds, and every shape the GEMV declines must
be one the fused GEBP accepts. If they drift apart, decode silently lands on the 16x-21x slower
blocked kernel with identical numbers.

## Numerics

`bf16 -> f32` is a left shift by 16 — `bf16` *is* the top half of an `f32` — so widening is exact
for every finite value, and accumulation is `f32`, matching every other route's accumulator width.
Each output element accumulates over `p` in strictly increasing order, so the SIMD path, the scalar
fallback and a naive reference agree **bit for bit**; the tests assert equality, not a tolerance.

The one exception is a signalling NaN: the shift keeps the payload where `half::bf16::to_f32`
canonicalizes to a quiet NaN. 126 of the 65536 `bf16` patterns therefore widen to a different NaN
*encoding*. No finite value differs, NaN still propagates, and the blocked half GEMM this replaces
widens by the same shift — so a decode that switches routes sees no change.
`bf16_widening_matches_the_half_crate_over_the_whole_domain` sweeps all 65536 patterns and pins the
count.

## Reproducing

```bash
cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
ONNX_GENAI_CPU_MM_HALF_GEBP=0 cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
ONNX_GENAI_CPU_MM_HALF_GEMV=0 ONNX_GENAI_CPU_MM_HALF_GEBP=0 \
    cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
```

`PROBE_SHAPE=tiny|square|mlp|lm_head|sweep|cross` selects a shape set; `sweep` is the `k = 4096`
column sweep the threshold was read from. `ONNX_GENAI_CPU_MM_HALF_GEMV=0` (or `off`) is also the
field bisect knob: it sends decode back through the blocking path for the whole process without a
rebuild.

## What is still open

- **The 33.6M threshold is one host's number.** It is a parallel-efficiency crossover, so it
  depends on core count and `n`; a 4-core host would cross much later. The gate is a constant, not
  a measurement at runtime, and it should be revisited on a machine with more cores or AVX-512.
- **The GEMV leaves cores idle at moderate `n`.** `STRIPE = 512` was chosen for cache-line
  efficiency, not for occupancy. Splitting the `k` range as well (with a per-worker partial sum
  reduced at the end) would use every core, at the cost of the strict `p`-order accumulation that
  currently makes the kernel bit-identical to a naive reference. That trade was not taken here.
- **AVX512-BF16 hosts are untested.** `x86_bf16::native_available()` still claims `bf16` before
  either route, and this host cannot run it, so the decode gate explicitly declines to divert those
  calls.
