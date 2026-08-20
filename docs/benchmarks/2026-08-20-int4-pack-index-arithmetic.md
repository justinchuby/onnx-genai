# The int4 prefill pack was bookkeeping-bound: 1.53x more, and the row gates move a third time

**2026-08-20 · AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2+FMA+F16C, no AVX-512/VNNI,
75.8 GB/s DRAM, shared host · ORT 1.28.0**

## Where the cost was

After the pack was vectorized, its fitted fixed cost at `4096x11008` was **2.16 ms** against 22.5 MB
of packed weight. That is **10.8 GB/s — 14% of the 75.8 GB/s roofline** — and the pack is already
parallel across column strips, so it is not bandwidth and not serialization. Meanwhile the
microkernel beside it runs at **1161 GFLOPS**, ~95% of this host's AVX2 FMA peak. The pack was the
entire remaining opportunity at small `m`, where it is ~70% of the call.

Neither of the two costs found was arithmetic.

### 1. Four bounds-checked byte indexes instead of one load

```rust
let raw = u32::from_le_bytes([
    self.packed[byte_at], self.packed[byte_at + 1],
    self.packed[byte_at + 2], self.packed[byte_at + 3],
]);
```

Four loads, four bounds checks, and the shift/or chain to reassemble — per eight nibbles, i.e. per
`__m256`. Replaced by `self.packed[byte_at..byte_at + 4].try_into()`, which is one unaligned 32-bit
load behind one bounds check.

**fixed 2.155 -> 1.775 ms (1.21x).** Interleaved, 5 reps, non-overlapping: base
[2.86, 2.93, 2.95, 3.11, 2.95] vs new [2.65, 2.72, 2.67, 2.59, 2.64] at `m = 8`.

### 2. Scale and zero point re-derived every eight depths

Both are constant across a whole block, but the group is eight depths, so at `block_size = 32` each
was recomputed four times per block — and the int4 zero point lookup is itself a nibble extract.
Hoisting both to block scope restructures the loop to walk whole groups inside one block.

**fixed 1.881 -> 1.407 ms (1.34x).**

**Cumulative: 2.155 -> 1.407 ms, 1.53x**, on top of the previous 2.2x.

### The invariant this introduces

Walking groups inside a block requires `block_size - pc % block_size` to stay a multiple of 8. It
does: `pc` is a multiple of `KC = 256` and both `KC` and `block_size` are multiples of 8, so the
remainder is too. The bit-identity test now covers `block_size` **24 and 40** — multiples of the
group that do *not* divide `KC` — so this is pinned, not assumed.

## The row gates move for the third time

Measured with the gate forced to 1 so both routes run at every `m`, GEBP against the column-blocked
route (`ONNX_GENAI_CPU_MM_INT4_GEBP=0`), 5 interleaved reps.

**Non-resident, `4096x11008`:**

| m | column-blocked | GEBP | GEBP / row |
|---:|---:|---:|---:|
| 1 | 2.094 ms | 1.787 ms | 0.853x |
| 2 | 1.798 ms | 1.768 ms | 0.983x |
| 3 | 1.944 ms | 1.798 ms | 0.925x |
| 4 | 2.271 ms | 1.774 ms | 0.781x |
| 5 | 3.747 ms | 1.802 ms | 0.481x |
| 6 | 3.981 ms | 1.765 ms | 0.443x |

**L2-resident, `2048x2048`:**

| m | column-blocked | GEBP | GEBP / row |
|---:|---:|---:|---:|
| 2 | 0.2020 ms | 0.3690 ms | 1.827x |
| 4 | 0.2680 ms | 0.3660 ms | 1.366x |
| 6 | 0.4150 ms | 0.3830 ms | 0.923x |
| 8 | 0.5030 ms | 0.4810 ms | 0.956x |
| 12 | 0.6800 ms | 0.4890 ms | 0.719x |
| 16 | 0.8880 ms | 0.5760 ms | 0.649x |

| regime | scalar pack | vectorized pack | now |
|---|---:|---:|---:|
| `INT4_PREFILL_GEBP_MIN_ROWS` | 12 | 5 | **3** |
| `INT4_PREFILL_GEBP_MIN_ROWS_L2_RESIDENT` | 24 | 12 | **6** |

On the non-resident shape GEBP is ahead at every `m >= 1`, so that crossover has fallen off the
bottom of the sweep. `3` rather than `1` is deliberate: `m = 2` is a 1.7% difference, inside the
noise, and `m = 1` is decode, which has its own dedicated route and should not be re-pointed on the
strength of a prefill bench.

The split between the two regimes is still load bearing, and by more than before: a flat `3` costs
the small shape 1.37x at `m = 4` and 1.83x at `m = 2`; a flat `6` gives up 1.48x-2.08x on the large
shape at `m = 3..5`.

## Production A/B

25 real `MatMulNBits` models x 3 thread counts, `--native-only`, null control, 7 trials x 30 runs.
Metric column is `native`. Parity `PASS` on every row.

**60 improved · 15 within noise · 0 surviving regressions**

| model | threads | base ms | new ms | delta | null | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3_0p6b_mlp_t8` | 32 | 1.585 | 0.556 | -64.92% | 24.48% | 2.851x |
| `qwen3_0p6b_qkv_t8` | 32 | 1.086 | 0.422 | -61.14% | 19.24% | 2.573x |
| `qwen3_0p6b_qkv_t8` | 8 | 0.945 | 0.389 | -58.84% | 4.13% | 2.429x |
| `qwen3_0p6b_qkv_t8` | 16 | 0.942 | 0.441 | -53.18% | 8.81% | 2.136x |
| `qwen3_0p6b_mlp_t8` | 8 | 1.444 | 0.733 | -49.24% | 19.60% | 1.970x |
| `qwen3_0p6b_mlp_t8` | 16 | 1.253 | 0.708 | -43.50% | 6.46% | 1.770x |
| `qwen3_0p6b_qkv_t1` | 32 | 0.675 | 0.435 | -35.56% | 14.96% | 1.552x |
| `llama3_8b_qkv_t8` | 8 | 3.125 | 2.193 | -29.82% | 0.29% | 1.425x |
| `qwen3_8b_square_t8` | 8 | 2.100 | 1.477 | -29.67% | 1.90% | 1.422x |
| `llama3_8b_mlp_t8` | 8 | 7.629 | 5.373 | -29.57% | 0.08% | 1.420x |
| `llama3_8b_mlp_t8` | 16 | 3.957 | 2.815 | -28.86% | 0.78% | 1.406x |
| `llama3_8b_mlp_t8` | 32 | 3.795 | 2.847 | -24.98% | 0.18% | 1.333x |
| `qwen3_0p6b_mlp_t1` | 32 | 1.006 | 0.774 | -23.06% | 35.09% | 1.300x |
| `llama3_8b_qkv_t8` | 32 | 1.981 | 1.528 | -22.87% | 0.30% | 1.296x |
| `llama3_8b_qkv_t8` | 16 | 1.829 | 1.439 | -21.32% | 1.26% | 1.271x |
| `qwen3_8b_square_t8` | 32 | 1.166 | 0.958 | -17.84% | 2.40% | 1.217x |
| `qwen3_0p6b_mlp_t1` | 16 | 0.686 | 0.594 | -13.41% | 4.23% | 1.155x |
| `qwen3_8b_square_t1` | 32 | 1.170 | 1.040 | -11.11% | 9.57% | 1.125x |
| `qwen3_0p6b_qkv_t128` | 16 | 1.645 | 1.494 | -9.18% | 0.73% | 1.101x |
| `llama3_8b_mlp_t128` | 8 | 27.581 | 25.620 | -7.11% | 0.05% | 1.077x |
| `qwen3_0p6b_mlp_t128` | 8 | 3.417 | 3.176 | -7.05% | 0.15% | 1.076x |
| `llama3_8b_qkv_t128` | 8 | 13.386 | 12.471 | -6.84% | 0.12% | 1.073x |
| `llama3_8b_qkv_t128` | 16 | 8.084 | 7.550 | -6.61% | 0.99% | 1.071x |
| `qwen3_0p6b_qkv_t128` | 8 | 1.931 | 1.825 | -5.49% | 4.56% | 1.058x |
| `llama3_8b_mlp_t128` | 16 | 16.800 | 15.940 | -5.12% | 0.76% | 1.054x |
| `qwen3_8b_square_t128` | 16 | 4.746 | 4.523 | -4.70% | 0.44% | 1.049x |
| `llama3_8b_mlp_t128` | 32 | 15.695 | 14.990 | -4.49% | 0.31% | 1.047x |
| `llama3_8b_qkv_t256` | 8 | 24.236 | 23.241 | -4.11% | 0.14% | 1.043x |
| `qwen3_0p6b_mlp_t256` | 8 | 6.095 | 5.846 | -4.09% | 0.15% | 1.043x |
| `qwen3_8b_square_t256` | 16 | 8.601 | 8.256 | -4.01% | 0.45% | 1.042x |
| `llama3_8b_mlp_t256` | 16 | 30.926 | 29.703 | -3.95% | 0.11% | 1.041x |
| `qwen3_0p6b_qkv_t1` | 8 | 0.383 | 0.368 | -3.92% | 0.78% | 1.041x |
| `llama3_8b_mlp_t256` | 8 | 49.392 | 47.465 | -3.90% | 5.18% | 1.041x |
| `qwen3_0p6b_mlp_t128` | 16 | 2.177 | 2.093 | -3.86% | 0.64% | 1.040x |
| `qwen3_0p6b_qkv_t256` | 8 | 3.239 | 3.118 | -3.74% | 0.25% | 1.039x |
| `llama3_8b_qkv_t128` | 32 | 7.682 | 7.403 | -3.63% | 0.44% | 1.038x |
| `llama3_8b_qkv_t256` | 16 | 14.387 | 13.865 | -3.63% | 0.33% | 1.038x |
| `qwen3_0p6b_qkv_t128` | 32 | 1.573 | 1.516 | -3.62% | 0.06% | 1.038x |
| `qwen3_0p6b_mlp_t256` | 16 | 3.836 | 3.698 | -3.60% | 0.34% | 1.037x |
| `qwen3_0p6b_mlp_t128` | 32 | 2.324 | 2.249 | -3.23% | 0.22% | 1.033x |
| `llama3_8b_mlp_t256` | 32 | 28.564 | 27.660 | -3.16% | 0.25% | 1.033x |
| `qwen3_8b_square_t128` | 8 | 7.098 | 6.889 | -2.94% | 0.10% | 1.030x |
| `qwen3_8b_square_t128` | 32 | 4.707 | 4.580 | -2.70% | 0.00% | 1.028x |
| `qwen3_0p6b_mlp_t512` | 16 | 7.271 | 7.101 | -2.34% | 0.21% | 1.024x |
| `qwen3_8b_square_t1` | 8 | 1.013 | 0.990 | -2.27% | 0.99% | 1.023x |
| `qwen3_0p6b_mlp_t1` | 8 | 0.601 | 0.588 | -2.16% | 0.67% | 1.022x |
| `llama3_8b_mlp_t512` | 8 | 103.714 | 101.518 | -2.12% | 0.02% | 1.022x |
| `llama3_8b_qkv_t512` | 8 | 45.945 | 45.016 | -2.02% | 0.11% | 1.021x |
| `llama3_8b_qkv_t256` | 32 | 13.795 | 13.524 | -1.96% | 0.16% | 1.020x |
| `llama3_8b_mlp_t1` | 32 | 2.165 | 2.123 | -1.94% | 16.21% | 1.020x |
| `qwen3_0p6b_mlp_t512` | 8 | 11.558 | 11.335 | -1.93% | 0.03% | 1.020x |
| `llama3_8b_mlp_t512` | 16 | 57.263 | 56.177 | -1.90% | 0.85% | 1.019x |
| `qwen3_0p6b_qkv_t512` | 8 | 6.115 | 6.000 | -1.88% | 0.11% | 1.019x |
| `qwen3_0p6b_qkv_t512` | 16 | 4.451 | 4.373 | -1.75% | 0.88% | 1.018x |
| `qwen3_0p6b_mlp_t256` | 32 | 3.916 | 3.848 | -1.74% | 0.20% | 1.018x |
| `llama3_8b_mlp_t1` | 8 | 3.631 | 3.568 | -1.74% | 5.43% | 1.018x |
| `qwen3_8b_square_t512` | 8 | 24.863 | 24.437 | -1.71% | 0.11% | 1.017x |
| `qwen3_0p6b_qkv_t256` | 16 | 2.413 | 2.374 | -1.62% | 0.75% | 1.016x |
| `qwen3_0p6b_qkv_t512` | 32 | 4.500 | 4.432 | -1.51% | 0.02% | 1.015x |
| `qwen3_8b_square_t256` | 32 | 8.169 | 8.048 | -1.48% | 0.07% | 1.015x |
| `qwen3_0p6b_qkv_t256` | 32 | 2.555 | 2.522 | -1.29% | 0.70% | 1.013x |
| `llama3_8b_qkv_t1` | 8 | 1.749 | 1.731 | -1.03% | 0.06% | 1.010x |
| `qwen3_8b_square_t512` | 16 | 16.184 | 16.025 | -0.98% | 0.08% | 1.010x |
| `qwen3_8b_square_t256` | 8 | 12.786 | 12.662 | -0.97% | 0.09% | 1.010x |
| `llama3_8b_qkv_t1` | 32 | 1.710 | 1.695 | -0.88% | 2.16% | 1.009x |
| `llama3_8b_mlp_t512` | 32 | 54.085 | 53.662 | -0.78% | 0.10% | 1.008x |
| `qwen3_0p6b_mlp_t512` | 32 | 7.015 | 6.965 | -0.71% | 0.14% | 1.007x |
| `qwen3_8b_square_t512` | 32 | 15.323 | 15.228 | -0.62% | 0.44% | 1.006x |
| `llama3_8b_qkv_t512` | 32 | 26.123 | 25.962 | -0.62% | 0.07% | 1.006x |
| `qwen3_8b_square_t1` | 16 | 1.015 | 1.010 | -0.49% | 1.77% | 1.005x |
| `llama3_8b_qkv_t1` | 16 | 1.741 | 1.737 | -0.23% | 0.17% | 1.002x |
| `llama3_8b_qkv_t512` | 16 | 27.559 | 27.981 | +1.53% | 0.12% | 0.985x |
| `qwen3_8b_square_t8` | 16 | 1.408 | 1.483 | +5.33% | 3.62% | 0.949x |
| `llama3_8b_mlp_t1` | 16 | 1.776 | 1.889 | +6.36% | 10.47% | 0.940x |
| `qwen3_0p6b_qkv_t1` | 16 | 0.398 | 0.430 | +8.04% | 0.00% | 0.926x |

### The three flagged cells, re-measured

| cell | threads | at 7x30 | at 11x40 | verdict |
|---|---:|---:|---:|---|
| `llama3_8b_qkv_t512` | 16 | +1.53% | **-1.73%** | improvement |
| `qwen3_8b_square_t8` | 16 | +5.33% | **-21.21%** | improvement |
| `qwen3_0p6b_qkv_t1` | 16 | +8.04% | +2.42% (null 2.18%) | noise |

`qwen3_8b_square_t8`'s own t=8 and t=32 cells improved 1.42x and 1.22x in the same run — opposite
signs on the same shape at different thread counts is the noise signature, not a real effect.

`qwen3_0p6b_qkv_t1` is `m = 1` on a 1.57 MB weight, so its gate is the L2-resident one, `6` both
before and after: it **provably cannot reach any changed code**. At 15 trials x 60 runs it reads
-3.16% against a -7.22% null. A 0.00% null on a 0.4 ms cell, as the first run reported, is not a
credible noise floor.

## Versus ORT

Paired before/after from the same invocation, 8 threads.

| cell | before | after |
|---|---:|---:|
| `qwen3_0p6b_qkv_t8` | 4.69x | **2.59x** |
| `llama3_8b_qkv_t8` | 2.93x | **2.05x** |
| `llama3_8b_mlp_t8` | 2.57x | **2.03x** |
| `llama3_8b_qkv_t128` | 1.53x | 1.53x |

`t128` unchanged is the control: there the pack is a small fraction of the call.

## What is not claimed

- `m = 1` is untouched; decode does not route here.
- 8-bit prefill keeps its per-column scalar pack. Vectorizing it is *measured* to be worth nothing
  (`2026-08-20-quant-pack-cost-attribution.md`).
- `INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED = 4` was not re-measured; it gates a different competitor.
- The residual gap to ORT at small `m` is structural: MLAS `SQNBitGemm` CompInt8 wants VNNI, which
  this host does not have.

## Reproduce

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench int4_prefill_route_ab
PROBE_SHAPE=big PROBE_M_LIST=8,16,64 ./target/release/deps/int4_prefill_route_ab-*
# crossover: rebuild with both gates forced to 1, then
ONNX_GENAI_CPU_MM_INT4_GEBP=0 PROBE_SHAPE=big PROBE_M_LIST=1,2,3,4,5,6,8,12 ./...
```
