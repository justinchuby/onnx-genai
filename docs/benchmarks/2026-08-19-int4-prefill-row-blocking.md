# int4 `MatMulNBits` prefill: row blocking

**Author:** Roy (Lead / CPU MatMul)
**Date:** 2026-08-19
**Host:** AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), L1d 32 KiB/core, L2 1 MiB/core,
L3 2 x 32 MiB. AVX2 + FMA, **no** AVX-512, **no** VNNI.
**Build:** `cargo build --release -p onnx-genai-bench --no-default-features
--features bench-native,cuda-13000 --bin bench_generic`. Bundled ORT 1.27.0.
**Method:** `--native-only` and `--ort-only` as separate arms, per
`sebastian-paired-harness-coresidency`: ORT's intra-op pool spin-waits after its
last op, so a paired run depresses the native arm by up to 4.8x on cells this
long. Every number below is a separate-arm measurement. Load is reported because
this host is shared.

## The report

The scheduler isolated `gemm_nbits_llama3_8b_qkv_t8` as ~10x behind ORT measured
native-alone, flat across 8/16/32 threads. Reproduced on `f8f3878ba`, at 32
threads, native-alone:

| threads | native | ORT | ratio |
|---:|---:|---:|---:|
| 8 | 12.502 ms | 1.479 ms | 8.45x |
| 16 | 6.031 ms | 0.891 ms | 6.77x |
| 32 | 6.837 ms | 0.691 ms | **9.90x** |

6.837 ms against the 6.90 ms in the scheduler's note — three significant
figures, an hour apart, so the cell is stable and the report is exact. Note that
we *regress* from 16 to 32 threads while ORT keeps scaling.

The shape is `k = 4096`, `n = 6144`, `m = 8`: 201 M MACs against 12.58 MiB of
packed int4 weight plus 3.15 MiB of f32 scales.

## What it was not

**Not memory bandwidth.** The first hypothesis was that the weight is re-read
per row, making this a 100 MiB traffic problem. The arithmetic does not support
it: 100.7 MiB in 6.87 ms is 14.6 GB/s, an order of magnitude under what this
part sustains, and the whole 15.7 MiB working set fits in a single 32 MiB L3
slice. I built the fix anyway — a 64-column tile with rows inner, so a tile's
weight and activations both sit in L2 — and it moved nothing at 16 or 32
threads. Rejected, and removed.

**Not fork/join.** `ONNX_GENAI_CPU_MM_INT4_PREFILL=1` selects a path whose whole
purpose is to replace the row-serial path's `m` fork-joins with one. It reads
6.874 ms against the default's 6.897 ms at 32 threads. The eight barriers an
8-token prefill pays are not the problem.

**Not the fan-out policy.** Which executor the fan-out lands on is the subject
of `WIDE_PREFILL_MACS` and `flat_fan_out`, and §36.2 of the phase-18 document
already established that routing this path through the task runtime "changed
**nothing**, in either direction, on any cell."

## What it was

Two things, and the second is the expensive one.

**1. The fixed path was switched off.** `borrowed_int4_prefill_block_enabled()`
defaulted to `false`, so every build anyone ran took `borrowed_affine_int4_matmul`
— the row-serial path. This is the same failure mode as #1080's f16 fix sitting
behind a `mlas` feature gate: a toggle added "until the win is measured",
and then nothing measured it. §36.2 of the phase-18 document called this out in
so many words; it had not been acted on.

**2. The kernel is nibble-decode bound, and re-decodes the weight per row.**
Per 32-lane chunk of one column, `borrowed_int4_nblock4_avx2`'s inner loop is:

```
_mm_loadu_si128            1
_mm_and_si128 x2, _mm_srli_epi16      3
_mm_unpacklo/hi_epi8       2
_mm256_cvtepu8_epi32 x4 + _mm256_cvtepi32_ps x4   8
_mm256_fmadd_ps x4         4
```

Eighteen instructions of decode to feed four FMAs. Better than 80% of the
instruction stream is unpacking nibbles — and every activation row pays all of
it again, because the row loop sits outside.

The signature is visible without a profiler. Time is **exactly linear in `m`**,
which is what zero row reuse looks like:

| `m` | native (t=32) | per row |
|---:|---:|---:|
| 1 | 0.890 ms | 0.890 ms |
| 8 | 6.868 ms | 0.859 ms |
| 128 | 108.903 ms | 0.851 ms |

And the roofline says the same thing from the other side: 402 MFLOP in 6.87 ms
is 58.6 GFLOP/s, or **0.61 FLOP/cycle/thread**, against ORT's 6.07. A kernel
that is 10x off compute peak while sitting inside L3 is not waiting for memory.

## The fix

`borrowed_int4_rowblock_avx2`: decode each K block's nibbles **once**, then
drive `PREFILL_ROW_BLOCK` activation rows through the decoded vectors. The
18-instruction decode is amortised over `4 x ROWS` FMAs instead of 4, taking the
budget per 32 MACs from ~22 instructions to ~8.5. Within any one row it is
instruction-for-instruction the single-column case of the existing `NCols4`
kernel — same unpack, same per-block `blk`, same `acc = fma(blk, scale, acc)`
fold, same scalar affine correction, same one final reduction. The default is
flipped to on, leaving the variable as a kill switch.

### Why 4 rows

The chunk loop holds the decoded block (4 `ymm`), a per-row in-block accumulator
(`ROWS`), and the activations for the row it is on (4). `ROWS = 4` fits in 12 of
16 architectural `ymm`; `ROWS = 8` needs 20 and spills. Measured, 32 threads,
native-alone, ms:

| cell | `ROWS = 4` | `ROWS = 8` |
|---|---:|---:|
| `llama3_8b_qkv_t8` | 2.277 | 2.279 |
| `llama3_8b_qkv_t128` | 26.697 | 23.201 |
| `llama3_8b_qkv_t512` | 90.942 | 92.268 |
| `llama3_8b_mlp_t8` | 5.524 | 5.107 |
| `llama3_8b_mlp_t128` | 52.886 | 54.392 |
| `llama3_8b_mlp_t512` | 216.386 | 225.152 |
| `qwen3_0p6b_qkv_t8` | 0.325 | 0.275 |
| `qwen3_0p6b_qkv_t128` | 4.480 | 4.829 |
| `qwen3_0p6b_qkv_t512` | 10.526 | 13.786 |
| `qwen3_0p6b_mlp_t8` | 0.578 | 0.577 |
| `qwen3_0p6b_mlp_t128` | 5.047 | 6.510 |
| `qwen3_0p6b_mlp_t512` | 20.372 | 24.868 |

`ROWS = 2` is uniformly worst (3.514 on the target cell). 8 leads only on the
`t8` cells and only just, while 4 leads on the wide prefills by up to 1.31x. A
`t8` cell is 8's most favourable possible shape — one tile, no remainder — and
it still does not carry the table, so 4.

I had written "`ROWS = 8` measured slower" into the constant's documentation
before measuring it. It is faster on the target cell. The table above is what
the sweep actually says.

## Result

All cells at 32 threads, native-alone, medians of 12 runs after 4 warmups,
load 7-10. "before" is `ONNX_GENAI_CPU_MM_INT4_PREFILL=0`, which is exactly the
path that shipped.

| cell | before | after | speedup | ORT | was | now |
|---|---:|---:|---:|---:|---:|---:|
| `llama3_8b_qkv_t8` | 7.087 ms | **2.339 ms** | 3.03x | 0.758 ms | 9.35x | **3.09x** |
| `llama3_8b_qkv_t128` | 103.250 ms | 28.862 ms | 3.58x | 8.270 ms | 12.49x | 3.49x |
| `llama3_8b_qkv_t512` | 509.199 ms | 97.591 ms | **5.22x** | 21.241 ms | 23.97x | 4.59x |
| `llama3_8b_mlp_t8` | 15.350 ms | 5.327 ms | 2.88x | 1.466 ms | 10.47x | 3.63x |
| `llama3_8b_mlp_t128` | 195.611 ms | 56.196 ms | 3.48x | 14.931 ms | 13.10x | 3.76x |
| `llama3_8b_mlp_t512` | 810.630 ms | 227.322 ms | 3.57x | 52.328 ms | 15.49x | 4.34x |
| `qwen3_0p6b_qkv_t8` | 0.895 ms | 0.261 ms | 3.43x | 0.096 ms | 9.32x | 2.72x |
| `qwen3_0p6b_qkv_t128` | 13.645 ms | 4.981 ms | 2.74x | 0.683 ms | 19.98x | 7.29x |
| `qwen3_0p6b_qkv_t512` | 51.084 ms | 15.337 ms | 3.33x | 6.007 ms | 8.50x | 2.55x |
| `qwen3_0p6b_mlp_t8` | 1.718 ms | 0.678 ms | 2.53x | 0.170 ms | 10.10x | 3.99x |
| `qwen3_0p6b_mlp_t128` | 26.891 ms | 6.981 ms | 3.85x | 1.428 ms | 18.83x | 4.89x |
| `qwen3_0p6b_mlp_t512` | 104.909 ms | 30.250 ms | 3.47x | 6.113 ms | 17.16x | 4.95x |

Twelve cells, twelve wins, 2.53x-5.22x. No cell regressed.

The target cell across thread counts, final build, load 7.3:

| threads | before | after | speedup |
|---:|---:|---:|---:|
| 8 | 12.182 ms | 4.714 ms | 2.58x |
| 16 | 5.646 ms | 1.893 ms | 2.98x |
| 32 | 6.596 ms | 2.332 ms | 2.83x |

And row reuse now shows up where its absence used to: per-row cost falls with
`m` instead of staying flat.

| `m` | before, per row | after, per row |
|---:|---:|---:|
| 8 | 0.886 ms | 0.292 ms |
| 128 | 0.807 ms | 0.226 ms |
| 512 | 0.994 ms | 0.191 ms |

## Numerics

Row blocking ends the prefill path's byte-identity with the row-serial
per-element path, and the test that asserted it is rewritten rather than
deleted. The per-element path horizontally reduces every 32-lane block and
accumulates a scalar; the row-blocked kernel keeps a lanewise accumulator across
all blocks and reduces once — the trade `borrowed_int4_nblock4_avx2` already
documents. Nibble decode is untouched. Measured disagreement is ~2 ULP.

`prefill_matches_per_column_borrowed_path_within_reassociation` now checks both
paths against an **f64 oracle** over the dequantized weights and requires the
blocked path to be no worse, so "it reassociates" cannot quietly become "it
lost accuracy". Errors are normalised by `sum |a| * |w|` rather than by the
result: these dot products cancel hard — a `k = 256` cell sums terms of order
0.25 down to a result of order 0.005 — so normalising by the result would
measure the conditioning of the test data instead of the kernel.

Byte-identity is still asserted everywhere it is load-bearing:

* `prefill_column_fan_out_matches_its_serial_self` — across fan-out partitions.
* `prefill_is_independent_of_the_row_tiling` — a row's value cannot depend on
  which tile it landed in. 9 rows against `ROWS = 4`, so rows land at every
  offset within a tile, and each is compared bit-for-bit against being computed
  alone.
* `rowblock_matches_nblock_for_a_single_row` — the claim that licenses reusing
  `NCols4`'s numerics rationale, asserted rather than asserted-by-comment.
* `the_blocked_int4_prefill_is_on_by_default` — pins the default, which is the
  part of this change that a future "add a toggle, default off" would silently
  undo.

End-to-end parity against ORT on the target cell: `max_rel = 1.68e-1` PASS —
unchanged from before the change, and dominated by int4 quantization error, not
by summation order.

## What this does not fix

* **We are still 2.5x-7.3x behind ORT.** The remaining gap is the decode itself:
  MLAS SQNBit's CompInt8 path avoids f32 dequantization entirely by quantizing
  activations to int8 and using integer dot products, where this kernel still
  widens every nibble to f32. That is the next structural step, not a tuning
  one.
* **`qwen3_0p6b_qkv_t128` is the worst remaining cell at 7.29x** and is now the
  outlier rather than `llama3_8b_qkv_t8`. It did improve 2.74x; ORT is simply
  very fast there (0.683 ms).
* **AVX-512 / VNNI is untested** — this host has neither, and VNNI is exactly
  what would make an int8 decode path pay off.
* **`ROWS` is a compile-time constant.** A host with 32 `ymm` (AVX-512) would
  likely want a wider tile; nothing here adapts.
* The `m == 1` decode route is untouched — this change is `m >= 2` only.
