# Retiring the `m == 1` half-decode handover to the fused GEBP

**Date:** 2026-08-21
**Host:** AMD EPYC 9V74, 32 vCPU (16 AVX2 cores x 2 SMT), AVX2 + FMA + F16C, no
AVX-512/VNNI. L1d 512 KiB, L2 16 MiB, L3 64 MiB, 75.8 GB/s DRAM. Shared.
**Harness:** `crates/onnx-runtime-ep-cpu/benches/half_decode_gemv_ab.rs` -- the
production A/B, which drives `ExecutionProvider::get_kernel` +
`Kernel::execute`, so the dispatch decision and the output narrowing are inside
the measurement. Arms are selected by environment out of one build
(`ONNX_GENAI_CPU_MM_HALF_GEBP=0` = GEMV, `ONNX_GENAI_CPU_MM_HALF_GEMV=0` =
fused GEBP); both knobs are process-wide `OnceLock`s, so one arm per process.
**Result:** the handover is **retired**. Non-batched `m == 1` half decode now
takes the GEMV at every weight.

## What was there

`half_decode_prefers_gebp` sent every `m == 1` `f16`/`bf16` decode with
`k * n >= HALF_PREFILL_GEBP_MIN_WEIGHT` (1 048 576) to the fused widen-pack
GEBP. Its evidence table was a 32-thread sweep over `k <= 2048` reporting the
GEBP 1.15x-2.27x ahead from 1.05M elements upward.

`Gemm` never had that gate. Its `m == 1` `f16` path stayed on the GEMV at every
weight, which is the divergence #1381 recorded: the same logical operation had
two kernels and two costs depending on which op the exporter emitted.

## What the re-measurement found

Two axes the original sweep did not cover: **thread count** and **`k` above
2048**. Both reverse the result.

`ratio` is `GEMV_ms / GEBP_ms`, so `> 1` favours the GEBP. `/ctl` divides out
the `f32` control row of the same two processes -- a path neither knob can
move, so it separates a route difference from machine drift.

### 8 threads (`RAYON_NUM_THREADS=8`), `PROBE_SHAPE=full`, `f16`

| `K x N` | elements | GEMV ms | GEBP ms | ratio | /ctl |
|---|---|---|---|---|---|
| 1024x768 [^1] | 0.79M | 0.053 | 0.293 | 0.18 | **0.19** |
| 2048x2048 | 4.19M | 0.129 | 0.337 | 0.38 | **0.30** |
| 4096x1024 | 4.19M | 0.208 | 0.311 | 0.67 | **0.55** |
| 2048x4096 | 8.39M | 0.159 | 0.686 | 0.23 | **0.26** |
| 4096x2048 | 8.39M | 0.304 | 0.700 | 0.43 | **0.43** |
| 4096x4096 | 16.8M | 0.485 | 1.458 | 0.33 | **0.33** |
| 4096x8192 | 33.6M | 1.372 | 2.820 | 0.49 | **0.44** |
| 4096x11008 | 45.1M | 1.917 | 2.568 | 0.75 | **0.76** |
| 896x151936 | 136M | 4.667 | 7.868 | 0.59 | **0.49** |

[^1]: 0.79M is *below* `HALF_PREFILL_GEBP_MIN_WEIGHT`, so with `GEMV=0` the
    GEBP declines too and this column is really the row-blocked half GEMM. It
    is listed because it is the row immediately below the retired threshold,
    not as GEBP evidence; the conclusion rests on the eight rows at or above
    1M.

Every shape at or above the threshold is a loss, by 1.3x to 3.8x (`/ctl` 0.26
to 0.76).

### 8 threads, `PROBE_SHAPE=band` -- the threshold neighbourhood, twice

`band` exists because the retired gate was a `k * n` gate: it puts rows
immediately below, at and above 1 048 576 elements at three different `k`, so a
`k`-dependent effect cannot hide inside the product. Both arms were run twice,
end to end, as four separate processes.

| shape | `k` | elements | f16 run 1 | f16 run 2 | bf16 run 1 | bf16 run 2 |
|---|---|---|---|---|---|---|
| `w0.79M` [^1] | 1024 | 0.79M | 0.23 | 0.25 | 0.27 | 0.31 |
| `w1.05M_k1024` | 1024 | 1.05M | **1.14** | 0.89 | **1.29** | **1.05** |
| `w2.1M_k1024` | 1024 | 2.10M | 0.53 | 0.86 | 0.54 | **1.01** |
| `w2.1M_k2048` | 2048 | 2.10M | 0.70 | 0.49 | 0.79 | 0.58 |
| `w3.1M_k1024` | 1024 | 3.15M | 0.57 | 0.48 | 0.69 | 0.96 |
| `w3.1M_k2048` | 2048 | 3.15M | 0.96 | 0.92 | 0.98 | **1.05** |
| `w4.2M_k1024` | 1024 | 4.19M | 0.26 | 0.26 | 0.39 | 0.53 |
| `w4.2M_k2048` | 2048 | 4.19M | 0.41 | 0.30 | 0.42 | 0.52 |
| `w4.2M_k4096` | 4096 | 4.19M | 0.57 | 0.82 | 0.69 | **1.05** |
| `w6.3M_k2048` | 2048 | 6.29M | 0.41 | 0.55 | 0.42 | 0.57 |
| `w8.4M_k2048` | 2048 | 8.39M | 0.20 | 0.41 | 0.24 | 0.37 |

Read this honestly. **20 of 22 `f16` cells are losses**, the worst reproducibly
so (`w8.4M_k2048` 0.20/0.41, `w4.2M_k1024` 0.26/0.26), and the `full` set above
agrees. But two things this set shows that a one-shot sweep would not:

- The one `f16` cell above 1.00 sits **exactly at the retired threshold** and at
  `k = 1024` -- `w1.05M_k1024`, 1.14 then 0.89. It is the same non-model corner
  the 32-thread sweep found, and it does not hold across repetitions.
- The five `bf16` cells at or above 1.00 are **not usable evidence**: their
  `f32` controls drifted to 0.55-0.94 (against 0.86-1.19 elsewhere), i.e. the
  machine moved 30-45% underneath those rows. That is exactly what the control
  is there to expose, and I am reporting them rather than dropping them.

So the earlier claim that no band row exceeds 1.00 at 8 threads **does not
reproduce**, and the earlier 6.7x figure (a `/ctl` of 0.15) does not either --
the supported worst case here is 5.0x (`/ctl` 0.20). The conclusion is not
affected: everything at `k >= 2048` above the threshold loses reproducibly, and
the only cells that ever win are at `k = 1024` within a factor of two of 1M,
which is neither a shape a model issues nor a region a `k * n` gate can isolate.

The GEBP's weight bandwidth at 8 threads is pinned at **20-34 GB/s regardless
of shape**, while the GEMV reaches 24-155 GB/s. The GEBP does not degrade
gracefully as threads are removed; it is the arm that needed them.

### 32 threads, `PROBE_SHAPE=full`, `f16`

| `K x N` | elements | ratio | /ctl |
|---|---|---|---|
| 2048x2048 | 4.19M | 0.95 | **0.84** |
| 4096x1024 | 4.19M | 0.51 | **0.57** |
| 2048x4096 | 8.39M | 0.59 | **0.55** |
| 4096x2048 | 8.39M | 0.56 | **0.51** |
| 4096x4096 | 16.8M | 0.82 | **0.89** |
| 4096x8192 | 33.6M | 0.81 | **0.80** |
| 4096x11008 | 45.1M | 0.88 | **0.88** |
| 896x151936 | 136M | 0.86 | **0.86** |

`bf16` tracks it (0.41-0.88). So even at the thread count the original sweep
used, **every shape a 7B-class decode actually issues is a loss** -- the qkv and
mlp projections at `k = 4096` and the `lm_head` at 136M. The original table
stopped at `k = 2048` and at 45M/136M reported 1.18-1.37; re-run here those two
rows are 0.88 and 0.86.

### Where the GEBP did win, and why it is not enough

At 32 threads with `k = 1024` and 1.05M-4.2M elements the GEBP is genuinely
ahead -- `/ctl` 0.95-1.49 over two independent repetitions. That corner is
real, and it is what the original sweep sampled. It is not worth keeping:

- the same shapes lose 1.3x-5.0x at 8 threads, and the loss magnitudes dwarf
  the gains;
- no shape in the corner is one a 7B-class model issues;
- `k = 2048` inside the corner is not reproducible (6.3M measured `/ctl` 1.31
  then 0.70 on consecutive runs of the same binary).

## Mechanism

This is not a threshold that needs retuning. The GEBP packs `B` into
`KC x NR` `f32` panels (`KC = 256`, `NR = 16`) so that each panel is **reused
across the rows of `A`**. At `m == 1`, `m_panels = 1`: every panel is consumed
by exactly one microkernel pass, so all of the packing is unrepaid.

What that costs, stated where the cost actually lands:

- **DRAM.** Both routes read each weight element once *logically*, but not with
  the same efficiency. `pack_b_half` walks `B` column-strip-major -- for a
  strip it reads `b[(pc + p) * n + j0 .. j0 + nc]` for each of `kc` rows -- and
  at the common `panels_per_strip = 1` that is `nc = NR = 16` elements, i.e.
  **32 bytes of every 64-byte line pulled**. So the GEBP issues roughly 2x the
  DRAM read traffic of the GEMV, which reads `B` in its stored order with full
  line utilisation.
- **Cache and compute.** The widen-and-pack writes `k * n` `f32` and reads them
  back, but per strip that lives in a `bpack` of at most
  `KC * 16 * NR * 4 B = 256 KiB` -- L2-resident on this host. That traffic is
  therefore not the DRAM story; it is L2 bandwidth plus the widening work
  itself, plus a fork/join over strips that a single row of `A` cannot amortise.

The two axes fall out of the second bullet rather than the first: the
widen-pack work per unit of reuse rises with `k` (the `while pc < k` loop
re-traverses the strip once per `KC` block), and it can only be overlapped by
having workers to overlap it with -- which is why the arm collapses at 8
threads while merely losing at 32.

Because the mechanism is structural, there is no repair: a decode-specialised
GEBP that skips packing `B` **is** the GEMV.

## Disposition

- `half_decode_prefers_gebp` and `half_decode_prefers_gebp_when` deleted; the
  `!half_decode_prefers_gebp(..)` term is gone from `MatMul`'s decode arm.
- `Gemm` is unchanged in behaviour -- it was right. #1381 closes with `MatMul`
  moving to `Gemm`'s route rather than the reverse.
- `HALF_PREFILL_GEBP_MIN_WEIGHT` and `half_prefill_gebp_selected` are
  untouched: they remain the **prefill** gate, and they still serve a *batched*
  `m == 1` half MatMul, which the non-batched GEMV declines and whose only
  other destination is the row-blocked half GEMM (16x-21x slower).
- `PROBE_SHAPE=band` added to the harness: 11 shapes immediately below, at and
  above the retired threshold at three different `k`, so a `k`-dependent effect
  cannot hide inside a `k * n` gate again.

## Remaining losses

- The `k = 1024`, 1.05M-4.2M, 32-thread corner gives up 1.1x-1.5x. Recovering
  it would need a thread-count- and `k`-dependent predicate; not attempted,
  because the measurement that would justify it is the one that is least
  reproducible on this host.
- The GEMV itself is not at the roofline: 24-155 GB/s against a 75.8 GB/s DRAM
  ceiling means the large shapes (136M `lm_head`, 58 GB/s at 8 threads) are at
  ~77% and the small ones are latency- rather than bandwidth-limited. That is a
  separate line of work.
- `Gemm`'s half fast path is still `f16`-only: `bf16` operands return `None` at
  the dtype check and fall into the portable blocked half GEMM, where `MatMul`
  serves them from the same GEMV. A distinct gap, not addressed here.

## Reproducing

```bash
for arm in ONNX_GENAI_CPU_MM_HALF_GEBP=0 ONNX_GENAI_CPU_MM_HALF_GEMV=0; do
  env $arm RAYON_NUM_THREADS=8 PROBE_SHAPE=full \
    cargo bench -p onnx-runtime-ep-cpu --bench half_decode_gemv_ab
done
```

Swap `PROBE_SHAPE=band` for the threshold neighbourhood and drop
`RAYON_NUM_THREADS` for the 32-thread rows. The `Gemm`-side A/B is
`bench_gemm_half_decode_route` in `kernels::gemm`, run the same way (one arm
per process, `-- --ignored --nocapture`).
