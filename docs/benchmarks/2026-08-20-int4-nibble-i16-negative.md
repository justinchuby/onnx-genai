# Packed-nibble int4 x int16 AVX2 decode: built, measured, rejected

> **SUPERSEDED (2026-08-21) — the verdict below is wrong, and one of its
> explanations is wrong.** A second packed-nibble kernel is **1.2x-2.4x faster**
> than the incumbent on the same host and shapes, and is merged. See
> [`2026-08-21-int4-packed-nibble-avx2.md`](2026-08-21-int4-packed-nibble-avx2.md),
> which reconciles the two in detail. In short:
>
> * The kernel measured here spent ~4 uops per 32 weights restoring `k` order in
>   the **weights** (`unpack` x2 + `cvtepu8_epi16` x2). Deinterleaving the
>   **activation** once per row instead removes all of them: 14 uops -> 10.
> * Neither kernel's real bottleneck was the inner loop. Per-**block** cost was
>   ~17 cycles against ~2.5 cycles of arithmetic at `block_size = 32`; hoisting a
>   per-block `is_x86_feature_detected!` and tiling four blocks through one
>   reduction and one vector tail took it to 8.5 and produced most of the win.
> * **"The incumbent is already at the memory roofline" does not hold.** The
>   `acc4_int8` arm's 98-102%-of-DRAM figure at block 128 is an L3-resident
>   working set (58.7 MB expanded, 64 MiB L3) measured against a DRAM
>   denominator — the exact error this document warns about elsewhere. The new
>   kernel is 2.37x faster than that arm on that cell, which no bandwidth-
>   saturated kernel reading half the bytes could be.
>
> What still holds: `madd_epi16` retires half the products of `maddubs_epi16`,
> int16 activations cost two loads per 32 weights, and the accuracy pinning test
> this work landed. Kept unedited below as the record of what was measured.


**Date:** 2026-08-20 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c/2t), AVX2+FMA+F16C, **no AVX-512/VNNI**, L1d 512 KiB, L2 16 MiB,
**L3 64 MiB**, 75.8 GB/s DRAM, shared. ORT 1.28.0.

## Summary

A complete packed-nibble `int4 x int16` AVX2 kernel for `accuracy_level = 4`
`MatMulNBits` decode was implemented, validated against an `f64` reference, and
benchmarked against the route it was meant to replace. **It is 1.5x-2.2x slower
in every cell measured and has been rejected.** It is not merged.

The idea was to stop the incumbent `accuracy_level = 4` path from expanding each
4-bit weight into a whole `int8` byte (`prepack_int8_weight`), and instead
consume the ONNX packed nibbles directly at 0.5 B/weight. The premise --
recorded in `2026-08-20-int4-acc4-int8-repack.md` -- was that a decode GEMV's
cost is dominated by weight bytes streamed, so halving the bytes should roughly
halve the time.

**The premise is false on this host, and the measurement says why: at m=1 the
incumbent is already at the memory roofline while the nibble kernel is nowhere
near it.** Halving the bytes cannot help a kernel that cannot issue instructions
fast enough to saturate bandwidth in the first place.

One durable result is kept and merged: a test pinning *how much* accuracy
`accuracy_level = 4` actually costs int4 decode, which turns out to be ~9,700x
the error of `accuracy_level = 0` and ~2 orders of magnitude worse than the
8-bit route that shares the same attribute.

## What was built

Enough to be a fair test of the idea, not a sketch:

* `Int4I16Weight { values, scales, scaled_zero_points }` -- `values` byte-identical
  to the ONNX initializer (`[n][k_blocks][block_size/2]`, low nibble first, already
  block-padded), so the "prepack" is a copy plus two `n * k_blocks` side tables
  rather than an `O(weights)` rewrite. Cached in a `OnceLock` on the kernel behind
  `can_prepack`, so steady-state decode does not repack per `Run`.
* `block_dot_int4_i16_avx2` -- 16-byte load, `and`/`srli`/`and` to split nibbles,
  `unpacklo/unpackhi_epi8` to restore element order, `cvtepu8_epi16`, then
  `_mm256_madd_epi16` (no VNNI needed). A 16-element step so `block_size == 16`
  still vectorizes, and an exact scalar tail for odd `K` / partial blocks.
* Zero points applied outside the integer dot via
  `sum (q - zp)*a = sum q*a - zp * sum a`, against the block's exact `f32`
  activation sum. Absent `zero_points` is the implicit midpoint 8, which *is* the
  signed-nibble reading (`q - 8` maps `0..=15` to `-8..=7`), so unsigned+zp
  subsumes signed exactly and there is no separate signed path.
* Overflow: `q <= 15`, `|a| <= 32767`, so a `madd_epi16` lane holds at most
  983,010 and an `i32` absorbs 4,369 products; a lane accumulates `group/16` of
  them. Four orders of magnitude of margin.
* Gated on `accuracy_level == 4` **and** `reduced_precision_activation_allowed`,
  with mutation tests proving `accuracy_level = 0` could not reach it.

Correctness was established before timing: bit-exact AVX2-vs-scalar over all 16
nibble values in every lane position, block sizes 16/32/64/128, partial and odd
tails, and `i16::MIN`/`i16::MAX` activations; all 256 packed byte values unpacked
in ONNX order; and **1.0e-5** relative error against an `f64` reference at
K=4096, N=256, block 32.

The full implementation (942 insertions) is the **first commit of the pull
request that added this document**, kept deliberately so a rejected experiment
stays auditable: GitHub keeps a PR's commits browsable after a squash merge, so
anyone wanting to re-run this on different hardware can recover the kernel
without reconstructing it from this prose.

## The measurement

Three arms, same binary, `m = 1`, `t = 8`, 30 runs / 10 warmups, native-only
(ORT's intra-op pool spin-waits and depresses a paired native arm):

* `acc0` -- `accuracy_level = 0`, the exact fp32-activation borrowed int4 path.
* `acc4_int8` -- the incumbent `accuracy_level = 4` CompInt8 repack.
* `acc4_nibble` -- this kernel.

Median native ms, lower is better:

| cell | acc0 | acc4_int8 | acc4_nibble | nibble/int8 | nibble/acc0 |
|---|---|---|---|---|---|
| llama3_8b_qkv b32 | 1.309 | **0.986** | 1.579 | 1.60 | 1.21 |
| llama3_8b_qkv b128 | 0.751 | **0.361** | 0.750 | 2.08 | 1.00 |
| llama3_8b_mlp b32 | 2.882 | **2.216** | 3.616 | 1.63 | 1.25 |
| llama3_8b_mlp b128 | 1.528 | **0.806** | 1.663 | 2.06 | 1.09 |
| qwen3_0p6b_qkv b32 | 0.310 | **0.150** | 0.239 | 1.59 | 0.77 |
| qwen3_0p6b_qkv b128 | 0.213 | **0.113** | 0.117 | 1.04 | 0.55 |
| qwen3_8b_square b32 | 0.746 | **0.541** | 0.814 | 1.50 | 1.09 |
| qwen3_8b_square b128 | 0.451 | **0.199** | 0.446 | 2.24 | 0.99 |

Paired A/B with a null control on the same cells put the noise floor at
0.2%-5.1%, against deltas of 21%-53%, so the losses are far outside noise.

**The kernel is dominated.** It never beats the route it would replace. It beats
`acc0` only on the two smallest cells, and `acc0` is *exact* while this kernel is
not, so that is not a trade anyone should take.

## Why: it is instruction-bound, not byte-bound

The falsifying observation is that **`nibble/int8` is flat across an 18x range of
weight footprint** -- 1.59, 1.60, 1.63, 1.50 at block 32 for weights of 1.6, 12.6,
29.4 and 6.4 MB. If bytes were the constraint, the ratio would improve as the
weight grew. It does not move.

The cleanest single cell: at `llama3_8b_mlp b32` the int8 arm's working set is
**73.4 MB and spills this host's 64 MiB L3**, while the nibble arm's is 44.1 MB
and does not. That is the byte thesis's best case, and the nibble kernel is still
**1.63x slower**.

Achieved weight-stream throughput (weight bytes / median time) against the
75.8 GB/s DRAM ceiling:

| cell | arm | MB | ms | GB/s | % DRAM |
|---|---|---|---|---|---|
| llama3_8b_qkv b128 | acc4_int8 | 26.7 | 0.361 | 74.0 | 97.6% |
| llama3_8b_mlp b128 | acc4_int8 | 62.4 | 0.806 | 77.4 | 102.1% |
| llama3_8b_qkv b32 | acc4_int8 | 31.5 | 0.986 | 31.9 | 42.1% |
| llama3_8b_mlp b32 | acc4_nibble | 44.1 | 3.616 | 12.2 | 16.1% |
| llama3_8b_mlp b32 | acc0 | 29.4 | 2.882 | 10.2 | 13.5% |

(The `MB` column is not defined identically across rows: the nibble arm's 44.1 MB
is its full working set -- packed values plus the two f32 side tables -- while the
`acc0` row's 29.4 MB is weight bytes alone. The looseness runs *against* the
thesis being tested, since it flatters the nibble kernel's GB/s, so no conclusion
below depends on tightening it.)

(The 102.1% is not an error bar -- it is L3 assistance, and it is the reminder
that a roofline percentage means nothing until you check whether the denominator
is the binding constraint. That mistake was made earlier in this workstream and
is not repeated here.)

The incumbent at block 128 is running at **98-102% of DRAM bandwidth**: it is
already at the roofline and there is no headroom for a byte reduction to buy.
The nibble kernel reaches **16%**. It is not waiting on memory at all; it is
waiting on its own instruction stream.

The instruction counts explain it. Per 32 weights:

* **int8 arm:** one 32-byte weight load, one 32-byte activation load,
  `maddubs_epi16` (32 products in one instruction), `madd_epi16`, `add` -- about
  5 vector ops.
* **nibble arm:** one 16-byte load, `and` + `srli` + `and`, two `unpack`, two
  `cvtepu8_epi16`, **two** 32-byte activation loads, two `madd_epi16`, two `add`
  -- about 14.

Two independent factors, and both are structural rather than fixable by tuning:

1. `_mm256_madd_epi16` retires 16 products per instruction; `_mm256_maddubs_epi16`
   retires 32. Any int16-activation kernel needs **2x** the multiply instructions
   of an int8-activation kernel at the same vector width.
2. int16 activations are 2 bytes, so a fixed 32-weight step reads **2x** the
   activation bytes and needs two loads where the int8 path needs one.

The nibble unpack itself (~4 ops per 32 weights) is the smaller part of the loss.

### What was tried to close it

Folding each activation group's `i32` partial into a single `f32x8` block
accumulator -- the structure `block_dot_u8_i16_avx2` uses, replacing a full
horizontal reduction per group with a convert/scale/add and one reduction per
block. Measured gain: **1-3 pp** (llama3_8b_mlp -8.16% -> -5.23%; qkv -33.28% ->
-31.64%). It also costs bit-exactness against the scalar reference, because it
scales eight lanes separately instead of one exact `i32` sum. A 2 pp gain on a
kernel losing by 60% does not buy an exact contract, so it was reverted.

No arrangement of AVX2 instructions closes a 2x multiply-throughput deficit plus
a 2x activation-load deficit. The idea is dead on AVX2.

## The parity warning, and why the timings above are still real

The A/B harness reported `PARITY_FAIL` on every nibble cell, and it must not be
read as "the kernel is wrong". `bench_generic` checks agreement **with ORT**, and
ORT's `accuracy_level = 4` makes the same int8-activation approximation the
incumbent does. Measuring all three against an `f64` reference at K=4096, N=256,
block 32:

| path | max abs error | relative |
|---|---|---|
| nibble int16 | 7.94e-4 | **1.04e-5** |
| incumbent int8 | 2.19e-1 | **2.86e-3** |

The nibble kernel is **276x more accurate**; the incumbent passes parity because
it reproduces ORT's error, and the nibble kernel "fails" it by being closer to
the truth than ORT is. This is a real limitation of the harness: it cannot
validate a kernel more accurate than its reference. The timing comparison is
unaffected -- both arms compute the same operator over the same bytes.

## What is kept

`accuracy4_int4_decode_error_envelope_is_pinned_against_f64` (merged) pins the
incumbent's accuracy against an `f64` reading of the operator, two-sided:

* `accuracy_level = 0`: **4.89e-7** relative -- exact, f32 rounding only.
* `accuracy_level = 4`: **4.75e-3** relative -- a **9,703x** ratio.

`accuracy_level = 4` is a contract to be less accurate, so no output-comparison
test can fail when its error drifts; the number was untracked. The lower bound
matters as much as the upper one: int4 acc4 quantizes activations to **int8**
while 8-bit acc4 quantizes to **int16** (`gemv_nk_u8_i16`, ~1e-5), a ~2
order-of-magnitude asymmetry between two routes chosen by the same attribute.
This kernel would have closed that asymmetry and was rejected on speed, so the
asymmetry is now a deliberate standing trade. If it ever closes, the lower bound
fires and forces this document to be revisited.

## Reproduce

```bash
for bs in 16 32 64 128; do for al in 0 4; do
  python3 scripts/ort_ab/gen_gemm.py --out roy_k_models \
    --block-size $bs --accuracy-level $al --tokens 1 2 4 8 16
done; done

cargo build --release -p onnx-genai-bench \
  --no-default-features --features bench-native,cuda-13000 --bin bench_generic

# acc0 / acc4 arms, one cell
./target/release/bench_generic --model roy_k_models/gemm_nbits_llama3_8b_qkv_t1.onnx \
  --native-only --runs 30 --warmups 10
./target/release/bench_generic --model roy_k_models/gemm_nbits_a4_llama3_8b_qkv_t1.onnx \
  --native-only --runs 30 --warmups 10
```

Non-obvious: `gen_gemm.py` emits int4 as `gemm_nbits_*` and 8-bit as
`gemm_nbits8_*`; block 32 carries no `b` tag; `--accuracy-level 4` tags `_a4` into
the stem and a single generator run emits only the level it was given.

## Bearing on the ORT gap

This closes one candidate explanation for the residual int4 decode gap and
sharpens where the rest must be. ORT does `llama3_8b_qkv` acc4 m=1 in 0.18 ms
where our best arm takes 0.361-0.986 ms. Our int8 arm is already at 98-102% of
DRAM bandwidth at block 128, so ORT is **not** winning by streaming faster --
it must be touching fewer bytes per output or reusing more from cache. Weight
byte width is now excluded as the lever: the narrower representation is
available, correct, and slower. The next probe should be what ORT's working set
per output row actually is, not how fast it moves bytes.
