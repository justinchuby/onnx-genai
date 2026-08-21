# int4 acc4 on AVX2 pays an int8 repack and still loses 8.8x (2026-08-20)

> **Outcome of the proposal in §5 (2026-08-21):** the packed-nibble `madd_epi16`
> kernel sketched here was built twice. The first attempt lost 1.5x-2.2x and was
> rejected ([`2026-08-20-int4-nibble-i16-negative.md`](2026-08-20-int4-nibble-i16-negative.md));
> the second is **1.2x-2.4x faster** than the int8 repack and is merged
> ([`2026-08-21-int4-packed-nibble-avx2.md`](2026-08-21-int4-packed-nibble-avx2.md)).
> The 4.7x scratch-prototype figure below did **not** transfer: it was an
> inner-loop-only measurement, and the deciding cost turned out to be per-**block**
> overhead outside that loop. Treat it as directional, as this document says.

Host: AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C, **no
AVX-512, no VNNI**, L2 16 MiB, **L3 64 MiB**, 75.8 GB/s DRAM, ORT 1.28.0. All
cells `t=8`, `--trials 3 --runs 20 --warmups 5`, medians. Native arm is the
default no-MLAS artifact.

## 1. The gap is concentrated at small `m`, and it is flat there

`gemm_nbits_llama3_8b_qkv` (K=4096, N=6144, block 32, `accuracy_level=0`):

| m | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| ratio vs ORT | **4.58** | 2.60 | 2.33 | 1.87 | 1.85 | 1.47 | 1.63 | 1.45 |
| native ms | 2.38 | 2.63 | 2.33 | 2.85 | 3.67 | 4.60 | 8.91 | 14.66 |

The m=1 cell here reads 2.38 ms / 4.58 where §3 reads 1.86 ms / 4.14 and a
later confirmation run read 1.76 ms / 3.91. These are **separate sweeps on a
shared host**, not repeats of one measurement, and the spread across them
(1.76-2.38 ms) is the honest run-to-run envelope for this cell. Nothing below
turns on a difference smaller than that envelope; the claims that matter --
flatness in m, and the acc0/acc4 contrast in §3 -- are far larger than it.

`native` is essentially **flat from m=1 to m=8** (2.33-2.85 ms) and only starts
tracking `m` past 16. Time that does not respond to work is a fixed cost --
here, walking the whole weight once per call -- not a slow multiply. Every
optimisation below therefore targets *cost per weight byte touched*, not FLOPs.

## 2. Not DRAM-bound: the weight is cache-resident in both arms

At m=1 the int4 weight is 15.7 MB (packed nibbles + f32 scales) and L3 is
64 MiB, so after the harness warmups it is L3-resident. The measured ORT rate
confirms this rather than assuming it:

| arm (`accuracy_level=4`) | time | bytes walked | implied rate |
|---|---|---|---|
| native | 1.569 ms | 28.3 MB (int8 repack) | 18.0 GB/s |
| ORT | 0.179 ms | 15.7 MB (int4) | **87.9 GB/s** |

(The native figure counts values + scales only; `int8_matmul` also walks
per-block sums, so 28.3 MB understates its traffic. The understatement is
conservative -- it works against the argument being made here.)

87.9 GB/s **exceeds the 75.8 GB/s DRAM ceiling**, which is only possible from
cache. So DRAM bandwidth is not the binding constraint for either arm and a
DRAM roofline is the wrong yardstick here; the deficit is inner-loop issue cost
and the number of bytes each arm chooses to walk.

## 3. What each `accuracy_level` actually dispatches

`accuracy_level` selects the ONNX compute type. For `bits == 4` on x86_64:

| accuracy_level | route | bytes/weight |
|---|---|---|
| 0 (CompFp32) | borrowed int4, `borrowed_int4_block_dot_avx2` — unpacks nibbles to f32, `_mm256_fmadd_ps` | 0.5 |
| 4 (CompInt8) | `matmul_nbits.rs:1766` → `prepack_int8_weight` → `int8_matmul` → `dot_u8_i8` → `dot_u8_i8_avx2` | **1.0** |

The int4-direct chain (`supports_int4_direct`) additionally requires VNNI on
x86_64, so on this host it is unreachable and plays no part in either row.

The acc4 row is the important one and it is **not** a missing route: int4 at
acc4 does get a CompInt8 kernel without VNNI. What it does *not* get is an int4
one. `prepack_int8_weight` expands every 4-bit weight to a whole int8 byte, so
the CompInt8 path walks **1.8x the bytes** of the int4 weight it started from
(28.3 MB vs 15.7 MB) — and then still loses:

| model | accuracy_level | native | ORT | ratio |
|---|---|---|---|---|
| int4 qkv m=1 | 0 (CompFp32) | 1.86 ms | 0.45 ms | 4.14 |
| int4 qkv m=1 | **4 (CompInt8)** | 1.57 ms | **0.18 ms** | **8.77** |

ORT is 2.5x faster at acc4 than at acc0; we improve only 1.2x. The int8 repack
buys a denser multiply but pays 2x the weight traffic for it, and on a workload
whose cost is *per byte touched* (§1) that trade is close to a wash.

## 4. The shape of the opportunity

The instruction sequences per 32 weights:

| path | widen/convert | arithmetic | ratio |
|---|---|---|---|
| int4 acc0 (`borrowed_int4_block_dot_avx2`) | 4x `cvtepu8_epi32` + 4x `cvtepi32_ps` | 4x `fmadd_ps` | 2:1 overhead |
| 8-bit i16 (`block_dot_u8_i16_avx2`) | 2x `cvtepu8_epi16` | 2x `madd_epi16` | 1:1 |

The 8-bit int16-activation kernel is the template worth copying: it reaches 1:1
widen-to-arithmetic using `_mm256_madd_epi16`, which needs **no VNNI**. Applied
to int4 it would consume packed nibbles directly, keeping 0.5 bytes/weight
*and* getting the denser multiply, instead of choosing between them.

A scratch prototype of exactly that inner loop, versus the current acc0 kernel,
over the full K=4096/N=6144 weight, single-threaded, best of 4 timed passes:

| inner loop | time | rate |
|---|---|---|
| current `borrowed_int4_block_dot_avx2` (f32 fmadd) | 4.235 ms | 3.0 GB/s |
| prototype int4 -> i16 `madd_epi16` | **0.900 ms** | **14.0 GB/s** |

**Provisional: 4.7x.** This prototype is *not committed* — it was a standalone
`rustc -O` file cross-checked to produce the identical dot, and it models only
the inner loop, with no zero-point term, no per-group activation scales and no
threading. Treat it as an argument that the direction is worth building, not as
a benchmark result. The real number has to come from a committed bench.

### Where such a kernel may not go

`borrowed_affine_int4_matmul` (`matmul_nbits.rs:7776`) carries an explicit
precision contract for the code it calls, including
`borrowed_int4_output_element`:

> Precision contract: this helper is reached only from the `accuracy_level == 0`
> borrowed int4 route, i.e. ONNX CompFp32. Every path below therefore has to
> keep the activations in f32. It must never dispatch an int8-activation kernel
> [...] -- that is CompInt8, and delivering it where CompFp32 was requested
> costs ~1e-3 relative error. An `aarch64` `m == 1, block_size == 32` NEON-SDOT
> diversion used to sit here and did exactly that; it was removed rather than
> gated [...]

So a reduced-precision int4 kernel belongs behind
`reduced_precision_activation_allowed` (`accuracy_level >= 2`, `:9297`), on the
acc4 side, exactly as `eight_bit_int16_activation` is gated for 8-bit — never
in the acc0 borrowed path.

## 5. Why this was invisible: a fifth pinned harness parameter

`gen_gemm.py` hard-coded `accuracy_level=0` with no way to override it, so
every model the A/B has ever generated requested CompFp32, and the entire acc4
column of §3 — the route that actually ships for CompInt8 models — had never
been measured.

This is the fifth instance of the same defect class (after `bits` pinned to 4,
`block_size` pinned to 32, `BLOCK_SIZE = 32`, and `NBITS_TOKENS`). The rule
already recorded in the tree holds again: *a constant no harness parameter can
express is not conservative, it is unowned.*

`--accuracy-level` now exists, and non-zero levels are tagged into the model
stem (`_a4`) so both sets can share a directory as distinct cells.

### Caveat recorded honestly

The `nbits8 accuracy_level=4` cells report `PARITY_FAIL=3/3` against ORT and
their timings are therefore **not published** here. That is a real semantic
divergence, not harness noise: at acc4 the 8-bit path is allowed its
int16-activation route while ORT quantizes activations to int8 (int8 flips
qwen3's near-tie token 1479 -> 3988). The two arms compute different things, so
comparing their speed would be meaningless.

Note also that the 8-bit *acc0* cells are **not** evidence for the int16 kernel:
at acc0 `reduced_precision_activation_allowed` is false, so 8-bit runs
`gemv_nk_u8` with f32 activations. Those cells show ORT being slow at 8-bit
MatMulNBits, nothing about our int16 path.

## Reproduction

Two generator runs are needed -- one per accuracy level, since each run emits
only its own set:

```
python3 scripts/ort_ab/gen_gemm.py --out <dir> --block-size 32 --tokens 1 8
python3 scripts/ort_ab/gen_gemm.py --out <dir> --block-size 32 --tokens 1 8 \
    --accuracy-level 4
python3 scripts/ort_ab/ab.py --arms main=<bench_generic> \
    --models <dir>/gemm_nbits_llama3_8b_qkv_t1.onnx \
             <dir>/gemm_nbits_a4_llama3_8b_qkv_t1.onnx \
    --threads 8 --trials 3 --runs 20 --warmups 5 --csv out.csv
```
