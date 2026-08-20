# What the int4 pack vectorization actually bought — and why the 8-bit twin buys nothing

**2026-08-20 · AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2+FMA+F16C, no AVX-512/VNNI, 75.8 GB/s
DRAM, shared host · `benches/int4_prefill_route_ab`, `PROBE_SHAPE=big` (`k=4096, n=11008`),
`block_size=32`, 5 interleaved reps, medians**

## Result

A negative result and a correction to a merged claim.

- Vectorizing the **8-bit** prefill dequant pack the same way #1556 vectorized the 4-bit one is
  **worth nothing measurable**. Discarded, not merged.
- That null is itself the measurement that says #1556's stated *mechanism* was wrong. The win was
  the scalar nibble unpack, not the strided store.

## Why the 8-bit pack is the right control

`Int4Weight` and `Int8Weight` fill the identical `KC x NR` panel, through the identical
`dst[p * NR + slot]` store pattern, feeding the identical microkernel. The single difference is
that 8-bit has **no unpack**: eight depths of one column are eight consecutive bytes, already
byte-aligned. So vectorizing the 8-bit pack removes the strided store *and nothing else*. It
isolates the store term exactly.

## The 8-bit A/B — scalar pack vs SIMD pack

| m | scalar ms | SIMD ms | delta | speedup |
|---:|---:|---:|---:|---:|
| 8 | 3.354 | 3.347 | -0.21% | 1.002x |
| 16 | 3.724 | 3.687 | -0.99% | 1.010x |
| 32 | 4.984 | 4.937 | -0.94% | 1.010x |
| 64 | 7.147 | 7.202 | +0.77% | 0.992x |
| 256 | 22.537 | 22.612 | +0.33% | 0.997x |

Every cell is inside the noise, and the sign is not even consistent. Fitting
`t = fixed + marginal * m` over `m = 16..64`: **fixed 2.583 -> 2.515 ms**, a 0.07 ms difference.

The vector path really did run. The panel is bit-identical by construction, so no timing can
distinguish "ran and did not help" from "silently fell back" — that had to be settled separately.
`int8_dequant_panel_is_bit_identical_to_the_per_column_path` passed, and perturbing the widened
value inside `dequant_panel_avx2` by `+1.0` made it fail
(`panel mismatch at depth 0 slot 0 (block=8, asym=false, nr=8, kc=8, pc=0): -4.329 vs -4.292`).
The path is live; it just does not pay.

## The 4-bit A/B — same harness, same session

| m | scalar pack ms | SIMD pack ms | speedup |
|---:|---:|---:|---:|
| 16 | 6.051 | 3.334 | 1.815x |
| 32 | 7.247 | 4.741 | 1.529x |
| 64 | 9.431 | 7.057 | 1.336x |

**fixed 4.924 -> 2.093 ms, a 2.83 ms saving.**

## The decomposition

| term | cost at `4096x11008` | share of #1556's win |
|---|---:|---:|
| strided panel store | ~0.07 ms | 2.4% |
| scalar nibble unpack | ~2.76 ms | **97.6%** |

#1556 said the store was the cause. It was not. The reasoning that misled me was "one f32 every 64
bytes, so a separate cache line per element" — true about the *addresses*, irrelevant here, because
the panel is `KC * NR * 4` = 16 KB and sits in L1. Strided stores into 16 KB are L1 store-port
traffic, roughly one per cycle, not cache-line traffic. The 8-bit measurement puts a number on it:
0.07 ms.

What #1556 actually removed was per-element scalar work — a shift, a mask and a scalar widen for
every nibble — replaced by four AVX2 instructions per eight values. Grouping eight columns and
transposing them in registers is still necessary, but for **orientation**, not for the store: a
vector of eight dequantized values is eight depths of one column, and the panel wants one depth of
eight columns.

## What this changes

- #1556's measured outcome is unaffected: 4.80 -> 2.24 ms, the retuned row gates, and the 1.31x on
  `llama3_8b_qkv_t8` all stand. Only the explanation was wrong, and it is corrected in place.
- The 8-bit pack keeps the per-column scalar default. ~90 lines of unsafe SIMD for no measured gain
  is not a trade worth making, and "it is at least not slower" is not a reason to carry unsafe code.
- The generalizable rule: **vectorize a packer when its per-element arithmetic vectorizes, not
  because its stores are strided.** An L1-resident panel absorbs the strided store. The next
  candidate packer should be screened by asking what per-element ALU work it does, not by looking
  at its store addresses.

## Reproduce

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench int4_prefill_route_ab
PROBE_BITS=8 PROBE_SHAPE=big PROBE_M_LIST=8,16,32,64,256 \
  ./target/release/deps/int4_prefill_route_ab-*
```

`PROBE_BITS` is added by this change; before it the harness could only drive 4-bit weights, which
is why the control was never run.

## What is not claimed

- Only `4096x11008`, `block_size = 32`, this host. The share attributed to the store could differ
  where the panel does not fit in L1 — but `KC` and `NR` are compile-time constants here, so it
  cannot on this kernel.
- Nothing here narrows the residual gap to ORT on 8-bit prefill; it says where the gap is *not*.
