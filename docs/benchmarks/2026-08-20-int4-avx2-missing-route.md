# int4 MatMulNBits has no reduced-precision route on AVX2 (2026-08-20)

Host: AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C, **no
AVX-512, no VNNI**, 75.8 GB/s DRAM, ORT 1.28.0. All cells `t=8`, `--trials 3
--runs 20 --warmups 5`, medians. Native arm is the default no-MLAS artifact.

## 1. The gap is concentrated at small `m`, and it is flat there

`gemm_nbits_llama3_8b_qkv` (K=4096, N=6144, block 32, `accuracy_level=0`):

| m | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| ratio vs ORT | **4.58** | 2.60 | 2.33 | 1.87 | 1.85 | 1.47 | 1.63 | 1.45 |
| native ms | 2.38 | 2.63 | 2.33 | 2.85 | 3.67 | 4.60 | 8.91 | 14.66 |

`native` is essentially **flat from m=1 to m=8** (2.33-2.85 ms) and only starts
tracking `m` past 16. Time that does not respond to work is the signature of a
fixed cost -- here, streaming the whole weight -- not of a slow multiply.

## 2. It is not bandwidth

At m=1 the weight is 15.7 MB (packed int4 + f32 scales), read once:

| arm | time | achieved | % of 75.8 GB/s |
|---|---|---|---|
| native | 1.718 ms | 9.2 GB/s | **12%** |
| ORT | 0.444 ms | 35.4 GB/s | 47% |

Both arms read the same bytes. We are at 12% of the roofline, so the deficit is
inner-loop issue cost, and ~3.9x is available before bandwidth binds.

## 3. The mechanism: bits, not shape

Same shape, same `m=1`, only `bits` differs:

| model | ratio vs ORT | native | ORT |
|---|---|---|---|
| **8-bit** qkv | **0.183** (we are 5.5x *faster*) | 2.35 ms | 12.8 ms |
| **int4** qkv | **3.868** (we are 3.9x slower) | 1.72 ms | 0.44 ms |

The 8-bit decode GEMV has an int16-activation route (`gemv_nk_u8_i16` ->
`block_dot_u8_i16_avx2`, `_mm256_madd_epi16`) and beats ORT outright. int4 has
no AVX2 equivalent: `supports_int4_direct` requires VNNI, so on this host the
whole int4-direct chain is unreachable and int4 falls to
`borrowed_int4_block_dot_avx2`, which unpacks nibbles to **f32** and issues
`_mm256_fmadd_ps`.

Counting the inner loop per 32 weights:

| path | widen/convert | arithmetic | ratio |
|---|---|---|---|
| int4 f32 (`borrowed_int4_block_dot_avx2`) | 4x `cvtepu8_epi32` + 4x `cvtepi32_ps` | 4x `fmadd_ps` | 2:1 overhead |
| 8-bit i16 (`block_dot_u8_i16_avx2`) | 2x `cvtepu8_epi16` | 2x `madd_epi16` | 1:1 |

## 4. The route is missing, not mistuned

`accuracy_level` selects the ONNX compute type, and the EP gates every
reduced-precision activation on `reduced_precision_activation_allowed`
(`accuracy_level >= 2`). Measuring both levels:

| model | accuracy_level | native | ORT | ratio |
|---|---|---|---|---|
| int4 qkv m=1 | 0 (CompFp32) | 1.86 ms | 0.45 ms | 4.14 |
| int4 qkv m=1 | **4 (CompInt8)** | 1.57 ms | **0.18 ms** | **8.77** |

ORT gets **2.5x faster** when the model permits int8 compute. We barely move,
because there is nothing to move to. So the honest statement of the int4 m=1
deficit is not "our kernel is 4x slow"; it is:

> at `accuracy_level >= 2` we silently deliver the CompFp32 kernel where
> CompInt8 was requested and available, and pay 8.8x for the privilege.

The fp32 answer is the *more* accurate one, but the model asked for the cheaper
one and we are not offering it.

## 5. The fix is viable and needs no VNNI

Prototype of an int4 x int16-activation dot using the same `_mm256_madd_epi16`
the 8-bit path already uses, versus the current f32 kernel, over the full
K=4096/N=6144 weight, single-threaded, best of 4 timed passes:

| inner loop | time | achieved |
|---|---|---|
| current `borrowed_int4_block_dot_avx2` (f32 fmadd) | 4.235 ms | 3.0 GB/s |
| prototype int4 -> i16 `madd_epi16` | **0.900 ms** | **14.0 GB/s** |

**4.7x** on the inner loop, cross-checked to produce the identical dot. This
reuses machinery that already exists and is already validated for 8-bit:
`quantize_block_i16`, `activation_quant_group`, and the per-group-scale
structure of `block_dot_u8_i16`.

### Where it may *not* go

`borrowed_int4_output_element` carries an explicit precision contract: it is
reached only from the `accuracy_level == 0` route, i.e. CompFp32, and its
comment records that an aarch64 int8-activation diversion once sat there and
was **removed rather than gated**, because delivering CompInt8 where CompFp32
was requested costs ~1e-3 relative error. Any int16-activation int4 kernel must
therefore sit behind `reduced_precision_activation_allowed`, exactly as
`eight_bit_int16_activation` does -- never in the acc0 path.

## 6. Why this was invisible: a fifth pinned harness parameter

`gen_gemm.py` hard-coded `accuracy_level=0` with no way to override it, so
every model the A/B has ever generated requested CompFp32. Every
reduced-precision route in the EP was therefore not merely unmeasured but
**unreachable** by the harness -- including the int4-direct chain and the 8-bit
int16-activation path, whose gate is `accuracy_level >= 2`.

This is the fifth instance of the same defect class (after `bits` pinned to 4,
`block_size` pinned to 32, `BLOCK_SIZE = 32`, and `NBITS_TOKENS`). The rule
already recorded in the tree holds again: *a constant no harness parameter can
express is not conservative, it is unowned.*

`--accuracy-level` now exists, and non-zero levels are tagged into the model
stem (`_a4`) so both sets can share a directory as distinct cells.

### Caveat recorded honestly

The `nbits8 accuracy_level=4` cells report `PARITY_FAIL=3/3` against ORT and
their timings are therefore **not published** here. That is a real semantic
divergence, not harness noise: at `accuracy_level=4` ORT quantizes activations
to int8 while our 8-bit path deliberately uses int16 (int8 flips qwen3's
near-tie token 1479 -> 3988). The two arms are computing different things, so
comparing their speed would be meaningless.

## Reproduction

```
python3 scripts/ort_ab/gen_gemm.py --out <dir> --block-size 32 \
    --accuracy-level 4 --tokens 1 8
python3 scripts/ort_ab/ab.py --arms main=<bench_generic> \
    --models <dir>/gemm_nbits_a4_llama3_8b_qkv_t1.onnx \
             <dir>/gemm_nbits_llama3_8b_qkv_t1.onnx \
    --threads 8 --trials 3 --runs 20 --warmups 5 --csv out.csv
```
