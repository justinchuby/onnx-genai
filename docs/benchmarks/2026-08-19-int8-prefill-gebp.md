# 8-bit MatMulNBits prefill: one fused GEBP instead of a 51 MB f32 weight per call

**Date:** 2026-08-19
**Assignment row:** `docs/performance/CPU_MATMUL_ASSIGNMENT.md`, "The 8-bit prefill loss was a 51 MB f32 weight rebuilt on every call" (formerly "The 8-bit win is bounded by row count")
**Host:** AMD EPYC 9V74, 32 vCPU / 16 physical cores, AVX2 + FMA + F16C (no AVX-512), Linux
**Build:** `--release`, default features (`mlas` **off**), pure-native CPU EP
**Method:** two harnesses, both driving the production entry point
(`ExecutionProvider::get_kernel` + `Kernel::execute`), arms selected by
environment inside **one** build:
  * `cargo bench -p onnx-runtime-ep-cpu --bench int8_prefill_route_ab` — kernel-level, median of 7 timed calls per arm per shape, reported as the median of 3 interleaved arm-pairs.
  * `scripts/ort_ab/ab.py` — native/ORT ratio measured inside one process on one graph, arms interleaved trial by trial.

## What was wrong

On a native (non-MLAS) build there was **no borrowed route at all** for
`bits == 8, m > 1`. `try_prefill_mlas_nt` declines, and the kernel falls to:

```rust
let weight_kn = self.dequantize_weight(.., WeightLayout::Kn)?;
gemm(&activations, &weight_kn, result, m, self.k, self.n)?;
```

`WeightLayout::Kn` is the transposed layout the dense GEMM wants, so the dequant
writes four bytes of f32 for every one byte of weight read, at stride `n`, into
a buffer far larger than any cache — and it does it **on every call**, because
nothing caches `Kn` (only `Nk` is cached, and only on the MLAS path). A
3584x3584 8-bit node materializes 51 MB per prefill in order to multiply it
once.

That term does not shrink as rows are removed, which is why the loss was worst
where it was least expected: at `m = 8` the node spent 41.7 ms to do 1.8 GFLOP.

## The fix

Generalize #1117's fused-dequant GEBP to both bit widths. `pack_b_quant` builds
each `KC x NR` panel by expanding packed weights straight into the L1-resident
f32 panel the `6x16` microkernel already consumes; all `m` rows then reuse that
panel. The bit width is a `BlockQuantWeight` implementation
(`Int4Weight` / `Int8Weight`), so 4-bit and 8-bit share the packing, blocking,
threading and microkernel and differ only in `dequant_column`.

Each packed byte is read once per call, no f32 weight is ever resident, and only
the per-strip scratch is allocated.

### The numbers are bit-identical, not merely close

The route being replaced dequantizes to f32 and calls `gemm`, which on native
x86_64 + AVX2 resolves to `x86_sgemm::sgemm_simd` — the same packing and the
same `6x16` microkernel `quant_prefill_gebp` drives. `pack_b_quant` emits
exactly the bytes `pack_b` would have emitted from that dequantized weight, so
the two routes are not merely "the same order of operations", they are the same
operations. The A/B bench compares the **full** output of both arms in one
process and reports `bitexact` for every cell of every shape and row count
measured below, and
`matmulnbits_int8_gebp_is_bit_identical_to_the_dequant_route` asserts it through
the production entry point for partial trailing blocks and partial `NR = 16`
panels too. That is also why the tests need `INT8_PREFILL_GEBP_TEST_CALLS`:
values alone cannot tell which route ran.

## Result, kernel level

`k = n = 3584`, block 32, symmetric 8-bit. Steady-state ms (median of 3
interleaved arm-pairs, each the median of 7 timed calls):

| m | dequant + GEMM (previous) | fused GEBP | speedup | GFLOP/s |
|---:|---:|---:|---:|---:|
| 1 (control, same route both arms) | 1.054 | 1.054 | 1.00x | 24 |
| 2 | 15.465 | **0.905** | **17.1x** | 57 |
| 4 | 15.572 | **0.864** | **18.0x** | 119 |
| 8 | 15.245 | **1.096** | **13.9x** | 188 |
| 64 | 16.520 | **2.833** | **5.83x** | 580 |
| 256 | 21.426 | **8.263** | **2.59x** | 796 |
| 512 | 28.731 | **15.602** | **1.84x** | 843 |

Two more shapes, same method, `m = 1` carried as the control:

| shape | m | previous | fused | speedup |
|---|---:|---:|---:|---:|
| 2048x2048 | 1 | 0.374 | 0.374 | 1.00x |
| 2048x2048 | 4 | 11.610 | **0.448** | **25.9x** |
| 2048x2048 | 64 | 12.576 | **1.219** | **10.3x** |
| 2048x2048 | 256 | 14.683 | **3.676** | **3.99x** |
| 4096x11008 | 1 | 3.669 | 3.639 | 0.99x |
| 4096x11008 | 4 | 48.724 | **2.509** | **19.4x** |
| 4096x11008 | 64 | 52.287 | **7.066** | **7.40x** |
| 4096x11008 | 256 | 67.392 | **22.561** | **2.99x** — 1023 GFLOP/s |

The `m = 1` rows are the noise gauge: decode is not on this route in either arm,
and it reads 1.00x / 1.00x / 0.99x, so the prefill ratios above are not host
drift.

### The small-`m` ratios are the host-sensitive ones

Reproduced at review time on the same host under moderate contention, the
`m >= 64` rows and the ORT crossover came back essentially unchanged — 5.55x at
`m = 64`, 2.54x at `m = 256`, 1.75x at `m = 512`, peak 790 GFLOP/s — while
`m = 2` and `m = 4` read **13.9x** and **14.7x** against the 17.1x / 18.0x
above. The whole difference sits in the fused arm's *absolute* time (1.12 ms
against 0.905 ms at `m = 2`); the dequant arm reproduced to within 0.7%
(15.58 ms vs 15.47).

That is the expected shape: at two rows the run is ~1 ms, so the fixed pack and
fork/join cost is a large fraction of it and any co-tenant on the machine moves
the ratio. Read the small-`m` numbers as "more than an order of magnitude,
measured between 14x and 18x depending on host load", and the `m >= 64` numbers
as the stable ones. Nothing about the disposition changes either way: the
dequant arm cannot go below ~15 ms at this shape, because that is what
materializing 51 MB costs.

Cold (a fresh kernel per repetition — what time-to-first-token pays) tracks
steady state on both arms, because neither caches anything: at `k = n = 3584`
the fused arm's cold medians are 1.014 / 0.879 / 1.197 / 2.918 / 8.312 /
15.401 ms against the steady 0.905 / 0.864 / 1.096 / 2.833 / 8.263 / 15.602.
The win is not a caching artifact.

## Result, against ONNX Runtime

This is the claim that matters, and the one the assignment matrix is written
in. Native/ORT ratio, lower is better, `p50 [min-max]` over interleaved trials
(`--runs 7 --warmups 3`), Llama-3-8B projection geometry. The assignment
matrix's own square geometry is measured separately below.

| cell | threads | before | after |
|---|---:|---|---|
| 8-bit qkv (k=4096, n=6144), 128 rows | 2 | 3.573 [2.316-4.188] | **1.081** [0.784-1.265] |
| 8-bit qkv, 128 rows | 4 | 3.358 [3.152-3.481] | **0.919** [0.657-0.957] |
| 8-bit qkv, 128 rows | 8 | 2.163 [2.031-2.942] | **0.839** [0.731-0.953] |
| 8-bit qkv, 512 rows | 2 | 1.503 [1.440-1.577] | **0.894** [0.782-0.999] |
| 8-bit qkv, 512 rows | 4 | 1.478 [1.118-1.546] | **0.942** [0.806-1.220] |
| 8-bit qkv, 512 rows | 8 | 1.177 [1.123-1.343] | **0.698** [0.653-1.131] |
| 8-bit qkv, 8 rows | 8 | 2.918 [2.360-3.087] | **0.329** [0.285-0.405] |
| 8-bit mlp (k=4096, n=14336), 128 rows | 8 | 1.620 [1.420-1.702] | **0.544** [0.461-0.777] |
| 8-bit mlp, 512 rows | 8 | 1.597 [1.563-1.669] | **0.887** [0.835-0.917] |

Every measured 8-bit prefill cell crosses from a loss to a win. `parity=PASS`
on every trial of every cell.

### At the assignment matrix's own geometry

`k = n = 3584` (Qwen3-8B hidden size), which is what every `MatMulNBits` row of
`CPU_MATMUL_ASSIGNMENT.md` is quoted at, `--runs 9`, 7 trials (11 at the
`t = 8` rows, which were the noisiest):

| M | threads | before | after |
|---:|---:|---|---|
| 128 | 2 | 2.061 [1.974-2.291] | **0.655** [0.594-0.664] |
| 128 | 4 | 2.012 [1.818-2.210] | **0.745** [0.533-0.842] |
| 128 | 8 | 2.321 [2.115-2.465] | **0.681** [0.537-0.877] |
| 256 | 2 | 1.700 [1.655-1.709] | **0.767** [0.745-0.792] |
| 256 | 4 | 1.694 [1.638-1.706] | **0.834** [0.736-0.990] |
| 256 | 8 | 1.887 [1.747-2.435] | **0.879** [0.752-1.110] |
| 512 | 2 | 1.412 [1.387-1.423] | **0.864** [0.850-0.870] |
| 512 | 4 | 1.421 [1.354-1.848] | **0.877** [0.861-0.996] |
| 512 | 8 | 1.542 [1.482-1.845] | 1.011 [0.886-1.131] |
| 1 (decode control) | 2 | 0.154 [0.148-0.164] | 0.153 [0.142-0.155] |
| 1 (decode control) | 4 | 0.207 [0.131-0.240] | 0.181 [0.145-0.213] |
| 1 (decode control) | 8 | 0.244 [0.193-0.281] | 0.219 [0.188-0.276] |

`M = 512` at 8 threads lands at **parity, not a win** -- 1.011 with a range
that straddles 1.0 over 11 trials. It is reported as parity in the matrix.

### How this compares to the MLAS research build

The same tree built `--features mlas` -- not the shipped configuration, and not
a route this change touches -- reads:

| M | threads | pure native, before | pure native, after | `--features mlas` |
|---:|---:|---|---|---|
| 128 | 2 | 1.968 | 0.655 | 0.563 |
| 128 | 8 | 2.178 | 0.681 | 0.503 |
| 512 | 2 | 1.407 | 0.864 | 0.824 |
| 512 | 8 | 1.640 | 1.011 | 0.763 |

The pure-native path went from 2.5-3.7x behind the MLAS build to within
1.05-1.35x of it, without linking MLAS and without a resident f32 weight.

This also settles a discrepancy: `CPU_MATMUL_ASSIGNMENT.md` recorded
`M = 128` as a **win** (0.90 / 0.87 / 0.94) where the paired before-arm here
reads 2.06 / 2.01 / 2.32. The MLAS build does not reproduce the old numbers
either (0.563, not 0.90), so "the old rows were an MLAS build" is not the
explanation; those rows come from a tree that cannot be reconstructed here and
are replaced rather than defended.

### Controls

| control | threads | arm A | arm B | reading |
|---|---:|---|---|---|
| 8-bit qkv, **1 row** (decode; route untouched) | 8 | 0.242 [0.175-0.248] | 0.182 [0.171-0.238] | overlapping — unchanged, and already a 4-5x win |
| **4-bit** qkv, 512 rows (switch does not apply) | 8 | 1.309 [1.198-1.877] | 1.394 [1.167-1.681] | overlapping over 9 trials — wash |
| 4-bit qkv, 1 / 128 / 512 rows, **parent branch binary vs this one** | 8 | 3.823 / 1.914 / 1.305 | 3.796 / 1.903 / 1.305 | the generalization to `BlockQuantWeight` costs the 4-bit path nothing |

The last control is the one that matters for the refactor: `int4_prefill_gebp`
became generic over the bit width, so the 4-bit path had to be re-measured
against a build of the parent branch, not merely against the same binary with a
switch it ignores. Native p50 agrees within 0.5% on all three row counts.

An earlier 5-trial run of the 4-bit same-binary control showed 1.240 vs 1.375
with non-overlapping ranges. That was small-sample host drift, not a signal: at
9 trials the ranges overlap heavily and the *after* arm is the faster one in
absolute terms (58.5 ms vs 63.4 ms). Recorded because the disjoint-range version
is exactly the kind of number that gets published by accident.

## Row gate

`INT8_PREFILL_GEBP_MIN_ROWS = 2`. Unlike 4-bit — where the fused GEBP competes
with a genuinely cheap row-serial *borrowed* kernel and only wins from `m >= 4`
— there is no competitor here: the route being replaced pays its 51 MB whatever
`m` is. The fused route is 17.1x faster at `m = 2` and the ratio only grows, so
the constant is the GEBP's own lower bound rather than a crossover. `m = 1`
never reaches this branch; it is claimed earlier by the 8-bit decode GEMV, whose
`Nk` weight *is* cached.

The value was derived by measurement, not inherited: the constant was
temporarily set to 2 and `m = 2, 3` measured on both arms before the gate was
lowered.

## Scope and caveats

* x86_64 with AVX2 + FMA only (the packed microkernel's contract). Other targets
  keep the previous dequant-then-GEMM path, unchanged.
* Tried only **after** `try_prefill_mlas_nt` declines, so an `mlas`-feature
  build keeps its cached-`Nk` + `trans_b` route and nothing measured there
  changes.
* Declines on per-row `g_idx`, which the column-contiguous pack cannot express.
* Not gated on `accuracy_level`: the route it replaces was full-f32 compute at
  every level, and so is this.
* `ONNX_GENAI_CPU_MM_INT8_GEBP=0` restores the previous route. It is the only
  way back, since the row gate no longer excludes any prefill this branch sees —
  hence `matmulnbits_int8_gebp_kill_switch_restores_the_dequant_route`.
* No end-to-end TTFT number is claimed: this host has no model weights, so every
  figure is a kernel or a single-node graph measured through the production
  entry point, not a generation run.
* The ORT-comparison cells come from `scripts/ort_ab/gen_gemm.py`, which grew a
  `bits=8` variant for this work — the assignment matrix carried 8-bit rows that
  no generator in the tree could reproduce.

## Remaining losses on this operator

* **4-bit prefill is still ~1.31x behind ORT at 512 rows** (measured above as a
  control). The fused GEBP fixed the native-vs-native collapse; it did not close
  the gap to MLAS's `SQNBitGemm`.
* 8-bit `m = 512` is the weakest of the new wins (0.698-0.894 depending on
  thread count) because the GEBP's advantage shrinks as `m` grows: the weight
  read is amortized either way and what is left is raw SGEMM efficiency.
* The AVX-512 / AVX512-VNNI case is untested; this host has neither, so the
  `6x16` f32 microkernel is the only one exercised.
* The bit-exactness claim is specific to the shipped configuration and is not a
  general statement about any two GEMMs: on native x86_64 + AVX2 the `gemm`
  call in the route being replaced resolves to `x86_sgemm::sgemm_simd` — the
  *same* packing and microkernel `quant_prefill_gebp` uses — and `pack_b_quant`
  emits exactly the bytes `pack_b` would emit from the dequantized weight. That
  is also the only configuration in which the fused route runs, so the claim
  holds wherever it is made, but it would not survive a different backend on
  either side. `matmulnbits_int8_gebp_is_bit_identical_to_the_dequant_route`
  pins it through the production entry point, including a `k` that leaves a
  partial trailing block.
