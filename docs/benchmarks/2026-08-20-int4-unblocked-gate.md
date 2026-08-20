# The int4 prefill gate nobody could measure

**Date:** 2026-08-20
**Host:** AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), L1d 32 KiB/core, L2 1 MiB/core,
L3 2 x 32 MiB. AVX2 + FMA, **no** AVX-512, **no** VNNI. 75.8 GB/s DRAM. Shared host.
**Build:** `cargo build --release -p onnx-genai-bench --no-default-features
--features bench-native,cuda-13000 --bin bench_generic`. Bundled ORT 1.28.0.
**Base:** `134731003` (#1560).

## Why this was open

`INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED` is the row threshold for weights whose block size the
column-blocked int4 kernels reject. It has been 4 since before #1356. #1556 and #1560 both re-derived
the *other* two int4 prefill gates; #1560 closed by naming this one as "untouched", and #1556 did
not name it at all -- its closing "the lowest gate of any route is 4" has been stale since #1560 set
the pair to (3, 6).

It was untouched because no harness could express it:

| harness | pinned | consequence |
|---|---|---|
| `benches/int4_prefill_route_ab.rs` | `block_size = 32` | cannot reach the branch at all |
| `scripts/ort_ab/gen_gemm.py` | `BLOCK_SIZE = 32` | no production model has a 16-element block |
| `scripts/ort_ab/gen_gemm.py` | `NBITS_TOKENS = (1, 8, 128, 256, 512)` | steps `1 -> 8`, straddling every row gate |

The third one is worth stating separately: the int4 prefill row gates have lived at 4, 12, 5, 3, 6
and 2 across the last three PRs, and **the production A/B token grid has never contained a single
row any of them gate**. Every one of those retunes was validated on cells that could not observe it.

This is the same defect as `PROBE_BITS` in #1558, which is what let an 8-bit "5% win" stand
unchallenged until the control was run. Third instance. The rule it implies is in the tree now: a
constant that no harness parameter can reach is not conservative, it is unowned.

## The measurement

`PROBE_BLOCK=16`, native-alone, steady phase, median of five interleaved reps, GEBP against the
route it displaces. Both arms come from **one binary** with the gate forced to 1, selected by
`ONNX_GENAI_CPU_MM_INT4_GEBP`, so this is not a cross-build comparison.

| m | 2048x2048 | 4096x11008 |
|---|---|---|
| 1 | 2.51x | 4.62x |
| 2 | **4.06x** | **8.96x** |
| 3 | 6.81x | 14.6x |
| 4 | 10.4x | 18.1x |
| 6 | 13.2x | 30.1x |
| 8 | 13.8x | 30.6x |

Raw medians on `4096x11008`, in ms:

| m | GEBP | per-block dot |
|---|---|---|
| 1 | 2.034 | 9.393 |
| 2 | 2.078 | 18.613 |
| 3 | 1.934 | 28.208 |
| 4 | 2.050 | 37.133 |
| 6 | 1.938 | 58.375 |
| 8 | 2.516 | 76.934 |

The shape of this is the whole result: the dot arm is **linear in `m`** and GEBP is **flat**, because
GEBP's cost here is the pack and the pack does not depend on `m`. There is no crossover to find in
the measured range — the two curves have already crossed below `m = 1`.

That is why this gate reads so differently from the other two. Below the 32-element gates sits a
rival *vectorized* kernel, and the crossover is a real contest. Below this one sits
`borrowed_affine_int4_matmul`, a per-block scalar dot: both `borrowed_affine_int4_matmul_prefill` and
`borrowed_affine_int4_matmul_nblock` require 32-element blocks and decline. The old comment argued
that raising the threshold "would not hand them a better kernel, it would hand them no kernel" — true,
and the same asymmetry says the threshold should be as *low* as the surrounding invariants allow.

## Production A/B

25 int4 models at `block_size = 16` through the real dispatch path, `t = 8` and `t = 16`,
`--native-only --null-control`. The rows the gate moved, `t2` and `t3`, at 5 x 15:

| cell | t=8 | t=16 |
|---|---|---|
| `llama3_8b_qkv_t2` | -70.6% | -62.6% |
| `llama3_8b_qkv_t3` | -75.9% | -60.9% |
| `qwen3_0p6b_qkv_t2` | -80.1% | -73.7% |
| `qwen3_0p6b_qkv_t3` | -86.7% | -81.4% |
| `qwen3_0p6b_mlp_t2` | -78.7% | -63.4% |
| `qwen3_0p6b_mlp_t3` | -79.3% | -80.3% |
| `qwen3_8b_square_t2` | -63.0% | -54.0% |
| `qwen3_8b_square_t3` | -75.3% | -63.0% |

Paired against ORT, 7 x 20, `t = 8`, parity **PASS** on every cell:

| cell | native before | native after | ratio before | ratio after |
|---|---|---|---|---|
| `llama3_8b_qkv_t3` | 19.832 ms | 2.407 ms | 20.30x | **2.60x** |
| `llama3_8b_mlp_t3` | 45.008 ms | 5.601 ms | 17.98x | **2.48x** |
| `qwen3_0p6b_qkv_t3` | 2.481 ms | 0.415 ms | 16.48x | **2.99x** |
| `llama3_8b_qkv_t2` | 13.234 ms | 2.447 ms | 13.63x | **2.53x** |
| `qwen3_8b_square_t2` | 6.507 ms | 1.190 ms | 13.20x | **2.34x** |

2.3x-3.0x is the band 32-element weights already reach after #1560, so this closes an outlier class
rather than opening a front. 16-element blocks were paying 13-20x where everything else paid ~2.5x.

## The controls, and a noise lesson

`t1`, `t4` and `t8` are structurally identical between the arms: gates 4 and 2 both exclude `m = 1`
and both admit `m >= 4`, so the executed code is the same. Any delta there is measurement error, and
at 5 x 15 there were several large ones — `llama3_8b_qkv_t4 t=16` read **-31.3%** against a -1.3%
null. At 11 x 40 it collapsed to +0.60%.

`llama3_8b_qkv_t1 t=16` across three independent runs: **+0.95%, +5.49%, -11.80%**, the last against
a 0.06% null. Opposite signs on a code path that provably cannot differ. Two rules hold up here and
both were needed:

- a structural argument beats any single measurement — `m = 1` cannot reach code gated at `m >= 2`;
- sign flips across runs are the signature of host noise, and a tight null does not make a delta
  real. A 0.06% null next to an 11.8% reading means the noise is *bursty*, not absent.

## Positive proof the route moved

The routes either side of this threshold agree to within a few f32 ulps, so a numerics test passes
whichever one ran. `INT8_PREFILL_GEBP_TEST_CALLS` already existed for exactly this reason on the
8-bit side; int4 had no equivalent, and the consequence was visible: the 32-element parity test
carried a comment claiming "4 is the dispatch threshold" for a weight whose gate had since moved to
6, and had been quietly covering routes other than the one it named.

`INT4_PREFILL_GEBP_TEST_CALLS` plus `matmulnbits_int4_prefill_gebp_covers_the_unblocked_block_size`
now assert the route *taken* at `block_size = 16`, for `m` either side of the gate. The test is
non-vacuous: with `ONNX_GENAI_CPU_MM_INT4_GEBP=0` it fails at `m = 2`, which is the newly re-pointed
row.

## Why 2 and not 1

The measurement says GEBP wins at `m = 1` too, by 2.51x/4.62x. It is still set to 2.

`m = 1` is decode. This bench times one op in isolation; it cannot see the per-token pool contention
the narrow decode pool exists to avoid, and GEBP returns *before* `with_decode_pool` and drives the
global pool. `borrowed_affine_int4_matmul_prefill` is gated at `m >= 2` for the same reason. Moving
decode on the strength of a prefill bench is the error this file would otherwise be recording, so
`m = 1` stays where it is and the question stays open, pinned by a `const` assert so a future retune
has to argue with it rather than slide past it.

## Reproduce

```bash
# kernel sweep, both arms from one binary
PROBE_BLOCK=16 PROBE_SHAPE=big PROBE_M_LIST=1,2,3,4,6,8 <bench> --bench
ONNX_GENAI_CPU_MM_INT4_GEBP=0 PROBE_BLOCK=16 PROBE_SHAPE=big PROBE_M_LIST=1,2,3,4,6,8 <bench> --bench

# production models at the rows the gate covers
python3 scripts/ort_ab/gen_gemm.py --out models16 --block-size 16 --tokens 1 2 3 4 8
python3 scripts/ort_ab/ab.py --arms base=<base> new=<new> --native-only --null-control \
  --threads 8 16 --trials 11 --runs 40 --csv out.csv --models models16/gemm_nbits_b16_*.onnx
```

## Remaining losses

- Weights far below the 2048x2048 floor measured here. GEBP forks the global pool where the scalar
  fallback stays on the narrow decode pool, so for a few-KB weight the fork/join could dominate.
  Correctness is unaffected (partial panels are zero-padded) and no real projection at
  `block_size = 16` is that small, but the sweep does not cover it.
- `m = 1` at every block size, and whether decode should route through GEBP at all.
- The residual 2.3x-3.0x to ORT: structural, MLAS `SQNBitGemm` CompInt8 wants VNNI this host lacks.
- 8-bit at 16-element blocks needed no retune — `INT8_PREFILL_GEBP_MIN_ROWS` is already 2.
