# The int4 prefill's fused dequant pack was scalar

Issue #1471. Host: AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C; **no AVX-512, no
VNNI**. 75.8 GB/s sustained DRAM. Shared machine, so every number is a median of interleaved reps
with a null control. Bundled ONNX Runtime 1.28.0.

## The issue's premise was already fixed; its measurement was not

#1471 says the int4 prefill "expands the full weight into an f32 panel before running the GEMM" and
proposes fusing the dequant into an L2 tile. That fusion **already landed**, in #1356: `pack_b_quant`
dequantizes straight into the `KC x NR` panel the microkernel consumes, and `quant_prefill_gebp`
reuses each panel across all `m` rows. There is no 180 MB f32 panel on `main` to remove.

The measurement behind the issue is still real. Fitting `t = fixed + marginal * m` over the GEBP arm
at `4096x11008`:

| | fixed | marginal |
|---|---:|---:|
| before | **4.80 ms** | 0.0783 ms/row |
| after | **2.24 ms** | 0.0804 ms/row |

4.80 ms against 22.5 MB of packed weight is **4.7 GB/s**, 6% of this host's roofline. The fixed term
is not memory. `marginal` barely moves, which is the control: the microkernel is untouched.

## What it actually was

`Int4Weight::dequant_column` fills one column of the panel at a time. Per element that is a shift, a
mask, a scalar widen, a subtract, a multiply — and a store to `dst[p * NR + slot]`, i.e. **one f32
every 64 bytes**. A separate cache line, and a separate scalar store, for every element.

The fix dequantizes `DEQUANT_GROUP = 8` columns at once into eight `__m256`, transposes them in
registers, and writes eight contiguous 32-byte stores: **one store per eight elements instead of
eight**. Eight is not a tuning knob, it is the f32 lane count.

## Numerics

Bit-identical, and asserted as such. The widen/subtract/multiply is the same arithmetic in the same
order; the subtract and multiply stay separate `_mm256_sub_ps` / `_mm256_mul_ps` so they are never
contracted into an FMA. `int4_dequant_panel_is_bit_identical_to_the_per_column_path` compares the
vector panel against the scalar one with `to_bits()` equality over block sizes 2/4/8/32/128 (the
first two fall back), symmetric and asymmetric zero points, `nr` = 1/5/7/8/9/13/15/16 (whole groups,
every scalar tail width, and panels too narrow for a group), `kc` = 1/7/8/9/16/33/64/130, and two
`pc` offsets — and also asserts the packer never writes past `nr` into the caller's zero-padding.

The test was mutation-checked: perturbing the zero point by 1e-4 and swapping the nibble order both
fail it.

## The row gate had to move with it

`INT4_PREFILL_GEBP_MIN_ROWS` is a crossover between two kernels, and it was measured against the
scalar pack. Halving the GEBP's fixed cost moves it. Re-derived, GEBP's speedup over the
column-blocked kernel, steady phase, median of five interleaved reps:

| m | 2048x2048 (2.1 MB, L2-resident) | 4096x11008 (22.5 MB) |
|---|---:|---:|
| 4 | 0.54x | 0.98x |
| 5 | 0.89x | **1.48x** |
| 6 | 0.89x | 1.52x |
| 8 | 0.86x | 1.49x |
| 12 | **1.19x** | 2.21x |
| 16 | 1.38x | 2.60x |

Large-weight crossover **12 -> 5**; L2-resident **24 -> 12**. Reproduce with
`PROBE_M_LIST=4,5,6,8,12,16` on the `int4_prefill_route_ab` bench, once with the default env and
once with `ONNX_GENAI_CPU_MM_INT4_GEBP=0`.

## Results: production A/B on real `MatMulNBits` models

`ab.py --native-only --null-control`, 5 trials x 15 runs, `null` is the base binary under a second
name in the same invocation.

| cell | threads | base ms | new ms | delta | null | verdict |
|---|---:|---:|---:|---:|---:|---|
| `llama3_8b_mlp_t1` | 8 | 3.711 | 3.720 | +0.24% | 16.90% | within noise |
| `llama3_8b_mlp_t1` | 16 | 2.880 | 2.825 | -1.91% | 0.21% | **1.02x** |
| `llama3_8b_mlp_t128` | 8 | 35.901 | 27.579 | -23.18% | 0.38% | **1.30x** |
| `llama3_8b_mlp_t128` | 16 | 20.869 | 16.698 | -19.99% | 1.27% | **1.25x** |
| `llama3_8b_mlp_t512` | 8 | 102.036 | 93.378 | -8.49% | 0.39% | **1.09x** |
| `llama3_8b_mlp_t512` | 16 | 60.184 | 55.818 | -7.25% | 2.94% | **1.08x** |
| `llama3_8b_mlp_t8` | 8 | 8.811 | 6.793 | -22.90% | 0.09% | **1.30x** |
| `llama3_8b_mlp_t8` | 16 | 4.465 | 3.670 | -17.81% | 0.74% | **1.22x** |
| `llama3_8b_qkv_t1` | 8 | 1.806 | 1.801 | -0.28% | 0.94% | within noise |
| `llama3_8b_qkv_t1` | 16 | 2.140 | 1.736 | -18.88% | 20.37% | within noise |
| `llama3_8b_qkv_t128` | 8 | 15.092 | 12.171 | -19.35% | 0.40% | **1.24x** |
| `llama3_8b_qkv_t128` | 16 | 9.229 | 7.768 | -15.83% | 0.10% | **1.19x** |
| `llama3_8b_qkv_t512` | 8 | 45.496 | 41.752 | -8.23% | 0.12% | **1.09x** |
| `llama3_8b_qkv_t512` | 16 | 28.702 | 26.932 | -6.17% | 0.30% | **1.07x** |
| `llama3_8b_qkv_t8` | 8 | 3.808 | 2.897 | -23.92% | 1.21% | **1.31x** |
| `llama3_8b_qkv_t8` | 16 | 2.235 | 2.024 | -9.44% | 16.02% | within noise |
| `qwen3_0p6b_qkv_t1` | 8 | 0.373 | 0.379 | +1.61% | 3.75% | within noise |
| `qwen3_0p6b_qkv_t1` | 16 | 0.455 | 0.490 | +7.69% | 6.15% | above null (see below) |
| `qwen3_0p6b_qkv_t128` | 8 | 2.443 | 3.007 | +23.09% | 27.18% | within noise |
| `qwen3_0p6b_qkv_t128` | 16 | 2.300 | 1.785 | -22.39% | 16.39% | **1.29x** |
| `qwen3_0p6b_qkv_t512` | 8 | 5.945 | 5.572 | -6.27% | 0.10% | **1.07x** |
| `qwen3_0p6b_qkv_t512` | 16 | 4.339 | 4.389 | +1.15% | 5.76% | within noise |
| `qwen3_0p6b_qkv_t8` | 8 | 0.885 | 0.894 | +1.02% | 3.50% | within noise |
| `qwen3_0p6b_qkv_t8` | 16 | 1.125 | 0.991 | -11.91% | 26.04% | within noise |

**14 cells improve (1.02x - 1.31x), 9 within noise, 1 above its null.**

### The one cell above its null did not survive re-measurement

`qwen3_0p6b_qkv_t1` at 16 threads read +7.69% against a 6.15% null. That cell is `m = 1`, which
**cannot reach any changed code**: the lowest row gate is 4 (`INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED`,
unchanged here; the gates this PR moves are 5 and 12), so `quant_prefill_gebp` is never
called, and the threshold change does not move `m = 1` either (1 < 12 before, 1 < 5 after). It had
to be noise, and at 11 trials x 40 runs it is:

| cell | threads | base ms | new ms | delta | null | verdict |
|---|---:|---:|---:|---:|---:|---|
| `llama3_8b_qkv_t1` | 8 | 1.689 | 1.654 | -2.07% | 0.18% | **1.02x** |
| `llama3_8b_qkv_t1` | 16 | 0.910 | 0.954 | +4.84% | 1.76% | above null |
| `qwen3_0p6b_qkv_t1` | 8 | 0.367 | 0.356 | -3.00% | 1.63% | **1.03x** |
| `qwen3_0p6b_qkv_t1` | 16 | 0.495 | 0.504 | +1.82% | 2.63% | within noise |
| `qwen3_0p6b_qkv_t128` | 8 | 1.997 | 1.653 | -17.23% | 0.40% | **1.21x** |
| `qwen3_0p6b_qkv_t128` | 16 | 1.620 | 1.413 | -12.78% | 0.74% | **1.15x** |

`qwen3_0p6b_qkv_t1 t=16` becomes **+1.82%, within noise**. The two `llama3_8b_qkv_t1` cells read
-2.07% and +4.84% — opposite signs on the same shape at different thread counts, which is noise, on
a path that structurally cannot execute changed code. And `qwen3_0p6b_qkv_t128 t=8`, whose null was
an unusable 27.18% in the short run, resolves to a **1.21x improvement** with a 0.40% null.

## Remaining gap to ORT

Paired invocation, so the native arm is depressed by ORT's spin-waiting pool — read the before/after
pair, not the absolute:

| cell | base / ORT | new / ORT |
|---|---:|---:|
| `llama3_8b_qkv_t8` | 4.61x | **3.56x** |
| `llama3_8b_mlp_t8` | 3.42x | **2.73x** |
| `llama3_8b_qkv_t128` | 1.98x | **1.57x** |
| `llama3_8b_mlp_t128` | 1.84x | **1.44x** |
| `llama3_8b_qkv_t512` | 1.25x | 1.17x |
| `llama3_8b_mlp_t512` | 1.20x | 1.11x |

## What this does not fix

- **8-bit is untouched.** `Int8Weight` keeps the per-column default `dequant_panel`, so its packed
  panel is byte-for-byte what it was. The same transpose would help it and is not done here.
- **`m = 1` is unchanged**, by construction.
- **Still behind ORT at small `m`.** The reason is the one section 5 of the ledger already gives:
  MLAS SQNBit's CompInt8 path quantizes activations to int8 and uses integer dot products, where
  this kernel widens every nibble to f32. That is structural, and VNNI — which this host lacks — is
  what would make it pay.
- **The `block_size` not a multiple of 8 case falls back** to the scalar pack. ONNX's minimum block
  size is 16, so in practice only synthetic weights take it.

