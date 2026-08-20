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
packing is pure overhead on a problem that is purely memory-bound. `half_gemv::gemv_half_kn` reads
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
| `shipped` | default | GEMV under `HALF_PREFILL_GEBP_MIN_WEIGHT` (1M weight elements), GEBP at or above |

An `f32` control row runs the same shapes through the f32 path in every arm. No arm can move it,
so it is the check that a difference between the half arms is the route and not the machine.

## The result, and the direction it went

The hypothesis was that a GEMV -- one pass over the weight, nothing packed -- must be the floor at
`M = 1`, so the work started as "give `bf16` the GEMV `f16` already has". **That is false above one
megaelement, in both formats.** The GEMV's remaining job is much narrower than the hypothesis
assumed: it is the right route exactly where the fused widen-pack GEBP declines the weight
outright, and nowhere else.

Measured on latest `main` (`d4cb7341d`) through the production kernel, `M = 1`, median of the
per-run steady p50 over 5 interleaved repetitions per arm. `ratio` is `GEMV / GEBP`, so > 1 means
the GEBP is faster. `/ctl` divides out the `f32` control row of the same runs, which no arm here
can move -- on a shared host that correction is what makes 10% differences readable:

| `K x N` | elements | f16 ratio | f16 /ctl | bf16 ratio | bf16 /ctl |
|---|---:|---:|---:|---:|---:|
| 1024x768 | 0.79M | 0.39 | 0.40 | 0.36 | 0.37 |
| **1024x1024** | **1.05M** | **1.77** | **1.97** | **1.93** | **2.14** |
| 2048x1024 | 2.10M | 1.24 | 1.18 | 1.52 | 1.45 |
| 1024x2048 | 2.10M | 1.69 | 1.75 | 1.93 | 1.99 |
| 512x4096 | 2.10M | 2.17 | 2.27 | 2.13 | 2.23 |
| 2048x2048 | 4.19M | 1.67 | 1.78 | 1.27 | 1.35 |
| 4096x1024 | 4.19M | 0.89 | 0.98 | 1.34 | 1.47 |
| 2048x4096 | 8.39M | 0.91 | 1.01 | 1.82 | 2.02 |
| 4096x2048 | 8.39M | 1.19 | 1.25 | 1.38 | 1.45 |
| 4096x4096 | 16.8M | 1.15 | 1.19 | 1.20 | 1.24 |
| 4096x8192 | 33.6M | 0.94 | 1.00 | 1.00 | 1.06 |
| 4096x11008 | 45.1M | 1.18 | 1.18 | 1.26 | 1.26 |
| 896x151936 | 136M | 1.26 | 1.26 | 1.37 | 1.37 |

The 0.79M row is the GEMV's whole remaining job. There the GEBP declines on
`HALF_PREFILL_GEBP_MIN_WEIGHT` and the "GEBP" column is really the row-blocked GEMM, which the GEMV
beats by 2.5x. One row further down -- 1024x1024, the smallest weight the GEBP accepts -- the
ranking has already inverted, by 2.0x (`f16`) and 2.1x (`bf16`). Nothing at or above that weight
favours the GEMV once the control is divided out: the three sub-1.00 raw ratios (0.89, 0.91, 0.94)
each sit inside their own control's drift and correct to 0.98-1.01.

The mechanism is parallelism, not traffic. The GEMV hands each worker a `STRIPE = 512` column slice
and has that worker walk all `k` rows: at `n = 11008` that is 21 stripes on a 32-thread pool, so a
third of the cores idle while the rest issue `k` strided visits each. The GEBP's extra panel writes
are L1-resident and its column strips keep every worker on a contiguous run. That trade only ever
pays off in the GEMV's favour while the weight is small enough that the GEBP will not touch it.

### The threshold this replaced

An earlier revision of this work put the handover at `HALF_DECODE_GEBP_MIN_WEIGHT = 1 << 25`
(33.6M elements), read off a sweep that varied only `n` at `k = 4096`. Re-measured on current
`main` across both axes, that constant was wrong in the same direction everywhere it applied: it
left every shape from 1.05M to 33.6M on the slower route. The worst case was `bf16` at 2048x2048,
where it took decode off the GEBP that `main` already used and put it on the GEMV, a measured
**2.1x regression** against `main`. It is retired, not retuned -- there is no second threshold, and
`half_decode_prefers_gebp` now asks exactly the question `half_prefill_gebp_selected` asks.

The earlier sweep was not wrong about its own points; it was wrong to generalize a `k = 4096` line
to a rule about `k * n`. At a fixed element count the two routes rank differently depending on how
that count is split between `k` and `n` -- 2048x2048 and 4096x1024 are both 4.19M and disagree by a
factor of two.

## Where the threshold went

There is no decode-specific constant. Decode takes the GEMV exactly when the GEBP declines the
weight, which is `k * n < HALF_PREFILL_GEBP_MIN_WEIGHT` (1_048_576 elements). Because the crossover
measured above lands on that same boundary to within one sweep point, the two facts are now one
constant and `the_decode_handover_tracks_the_gebp_weight_gate` asserts the two predicates agree
shape by shape.

## Shipped routing, and what it changes

| case | before | after | change |
|---|---|---|---|
| `f16` `M = 1`, weight < 1M | GEMV | GEMV | none |
| `f16` `M = 1`, weight >= 1M | GEMV | GEBP | **1.20x - 1.34x** |
| `bf16` `M = 1`, weight < 1M | blocked | GEMV | **2.9x** (3.0x control-corrected) |
| `bf16` `M = 1`, weight >= 1M | GEBP | GEBP | none |

Full production A/B against `main` at `d4cb7341d`, two binaries interleaved rep by rep, 5
repetitions each, median of the per-run steady p50, default shipped routing on both sides:

| dtype | shape | `K x N` | main | this PR | gain | f32 ctl | gain/ctl | route |
|---|---|---|---:|---:|---:|---:|---:|---|
| f16 | attn_out | 1024x768 | 0.096 | 0.112 | 0.86 | 0.97 | 0.88 | unchanged |
| f16 | square | 2048x2048 | 0.223 | 0.177 | 1.26 | 0.99 | **1.27** | GEMV -> GEBP |
| f16 | mlp | 4096x11008 | 2.441 | 2.029 | 1.20 | 0.98 | **1.23** | GEMV -> GEBP |
| f16 | lm_head | 896x151936 | 6.172 | 4.611 | 1.34 | 1.00 | **1.34** | GEMV -> GEBP |
| bf16 | attn_out | 1024x768 | 0.252 | 0.086 | 2.93 | 0.97 | **3.01** | blocked -> GEMV |
| bf16 | square | 2048x2048 | 0.189 | 0.172 | 1.10 | 0.99 | 1.11 | unchanged |
| bf16 | mlp | 4096x11008 | 1.947 | 1.945 | 1.00 | 0.98 | 1.03 | unchanged |
| bf16 | lm_head | 896x151936 | 4.534 | 4.603 | 0.99 | 1.00 | 0.99 | unchanged |

The `f16 attn_out` row is the only one below 1.00, and it is **provably noise**: the output digest
is byte-identical across all ten runs on both binaries, so the same route ran on both sides and
nothing in this change can have caused the difference. The two sample sets overlap (main
0.075-0.121 ms, this PR 0.068-0.117 ms) and the fastest single sample of the ten is on the PR side.
At 0.1 ms this shape is at the floor of what the harness can resolve.

The `bf16 mlp` and `bf16 lm_head` rows were the headline of the earlier revision of this PR; they
are now neutral, because `main` already routes them through the fused GEBP (#1365) and this change
leaves them there. The `bf16` win that survives is the one below the weight gate, where `main` had
no vectorised route at all.

## Re-measured at review time, on a quiet host

Because that noise floor is wide enough to swallow the `lm_head` claims, the two shapes those
claims rest on were re-run at review time with load average at 4.6 instead of 14-30. Same one
build, same interleaving, **median of the per-run steady p50** on both sides (the minimum of the
same samples in parentheses), 9 repetitions per arm for the small shapes and 7 for `lm_head`:

| dtype | shape | `K x N` | arm A | arm B | ratio | control |
|---|---|---|---:|---:|---:|---:|
| bf16 | attn_out | 1024x768 | blocked 0.252 (0.243) | shipped 0.083 (0.077) | **3.04x** (3.16x) | 1.02x |
| bf16 | small | 512x512 | blocked 0.078 (0.077) | shipped 0.014 (0.014) | **5.57x** (5.50x) | 0.97x |
| f16 | lm_head | 896x151936 | GEMV 6.131 (5.599) | shipped 4.598 (4.474) | **1.33x** (1.25x) | 0.99x |
| bf16 | lm_head | 896x151936 | GEMV 6.218 (5.932) | shipped 4.563 (4.404) | **1.36x** (1.35x) | 0.99x |

The `control` column is the same-run `f32` ratio between the two arms — identical code, so its
distance from 1.00 is this run's noise. At 1-3% it is an order of magnitude tighter than the sweep
above, and every ratio in the table survives: the small-shape gains are unchanged, and the
`lm_head` gains come out slightly **larger** than the 1.28x/1.32x originally claimed. The one
number that moves against the claim is `f16` `lm_head` by minimum (1.25x rather than 1.28x), so
1.25x is the honest floor for that row. Absolute milliseconds differ between the two sweeps because
the host does; the ratios do not.

## Assignment == execution

Timing alone cannot say which route ran — the arms agree to half-precision rounding. Two
independent checks do:

1. **Output digests.** The bench prints an order-sensitive digest per row. At `attn_out` (0.79M,
   below the gate) the shipped digest equals the **GEMV** arm's in both formats; at `square`,
   `mlp` and `lm_head` it equals the **GEBP** arm's. That is the threshold, read off the values.
   The same digests are what prove the one sub-1.00 row of the A/B table is noise: it is identical
   on both binaries, so no route moved.
2. **Route counters.** `half_decode_gemv_calls()` and `half_prefill_gebp_calls()` are asserted in
   `decode_on_a_small_weight_keeps_the_gemv`, which also asserts the GEMV materialised neither a
   widened `B` nor a transposed weight cache.

`no_decode_is_handed_to_the_blocked_gemm` pins the invariant that makes the split safe: the two
gates live in different functions, and every shape the GEMV declines must be one the fused GEBP
accepts. If they drift apart, decode silently lands on the 16x-21x slower blocked kernel with
identical numbers. `the_decode_handover_tracks_the_gebp_weight_gate` strengthens that to equality
in both directions, which is what folding the two thresholds into one constant now permits.

`switching_off_the_gebp_leaves_decode_on_the_gemv` pins the other half of that invariant, the one
the first test cannot reach: `ONNX_GENAI_CPU_MM_HALF_GEBP=0` is a bisect knob for *prefill*
packing, so the decode handover must consult it too, or turning the knob off would push every large
decode onto the blocked kernel instead of back onto the GEMV. `half_prefill_gebp_enabled` is a
process-wide `OnceLock` that no test can flip, so the decision is asked of
`half_decode_prefers_gebp_when`, which takes that answer as an argument.

On 32-bit `x86` the handover does not exist at all — the fused GEBP is `x86_64`-only — so
`half_decode_prefers_gebp` declines unconditionally there and every decode stays on the GEMV,
exactly as before this change. Verified by compiling the crate for `i686-unknown-linux-gnu`.

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

`PROBE_SHAPE=tiny|square|mlp|lm_head|low|full|sweep|cross` selects a shape set. `low` and `full`
are the two-axis sweeps the threshold was read from -- `low` straddles the weight gate itself
(0.79M to 2.1M, varying `k` and `n` independently at fixed element counts) and `full` carries it up
to 136M. `sweep` is the older `k = 4096`-only column sweep, kept because it is the one that
produced the retired 33.6M threshold. `ONNX_GENAI_CPU_MM_HALF_GEMV=0` (or `off`) is also the
field bisect knob: it sends decode back through the blocking path for the whole process without a
rebuild.

## What is still open

- **The gate is still one host's number**, inherited from the prefill sweep. It is a
  parallel-efficiency crossover, so it depends on core count and on how `k * n` is split; a 4-core
  host would cross elsewhere. That the decode crossover and the prefill weight gate coincide here
  to within one sweep point is a measured coincidence on this machine, not a derivation -- the
  reason to fold them into one constant is that a second constant was measurably worse, not that
  the two must be equal in principle. `the_decode_handover_tracks_the_gebp_weight_gate` pins the
  coupling so that a future host-specific split is a deliberate edit rather than a drift.
- **1.05M is the first weight measured above the gate, not the crossover itself.** The crossover is
  somewhere in (0.79M, 1.05M]; this sweep cannot place it more finely, and did not need to, because
  the GEBP declines everything below 1.05M anyway.
- **The GEMV leaves cores idle at moderate `n`.** `STRIPE = 512` was chosen for cache-line
  efficiency, not for occupancy. Splitting the `k` range as well (with a per-worker partial sum
  reduced at the end) would use every core, at the cost of the strict `p`-order accumulation that
  currently makes the kernel bit-identical to a naive reference. That trade was not taken here.
- **AVX512-BF16 hosts are untested.** `x86_bf16::native_available()` still claims `bf16` before
  either route, and this host cannot run it, so the decode gate explicitly declines to divert those
  calls.
