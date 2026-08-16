# CPU EP vs ONNX Runtime CPU: attention, the surrounding transforms, and MoE

**Date:** 2026-08-15
**Host:** AMD EPYC 9V74, 16C/32T, 1 NUMA node, shared L3 = 32 MiB per 16 CPUs,
AVX2 + FMA (no AVX-512), 125 GiB RAM. Linux.
**Comparand:** a real ONNX Runtime CPU session (ort-sys pins ORT 1.27.0, API 27)
in the same process, on the same graph, with the same thread count and the same
inputs.
**Harness:** [`scripts/ort_ab/`](../../scripts/ort_ab/README.md).

> **Synthetic data.** No trained weights. Only the dimensions come from public
> architecture configs; tensor contents are the harness's deterministic
> synthetic pattern, fed identically to both runtimes. See the harness README.

> **The host is shared and contended.** Same-shape absolute timings drifted by
> more than 4× between sessions. Every number below is a *paired, interleaved*
> `native/ORT` ratio measured within one run. Lower is better; `1.0` is parity.
> Brackets are the observed min–max of the per-trial ratios.

## Why ratios against ORT and not against our own previous kernel

A kernel that gets 8× faster can still be 3× slower than the runtime a user
would otherwise run. Kernel-vs-previous-kernel speedups are not evidence of
competitiveness, and every claim below is therefore stated as a ratio against a
real ORT session including node and session overhead.

## 1. GroupQueryAttention — decode

`com.microsoft::GroupQueryAttention`, single node, static shapes.
`base` = before the direct-present change, `new` = after.

| model | past | t | base | new |
|---|--:|--:|--:|--:|
| llama3-8b b1 | 511 | 1 | 1.18 | 1.88 |
| llama3-8b b1 | 511 | 8 | 4.61 | **3.44** |
| llama3-8b b1 | 511 | 16 | 2.27 | **1.20** |
| llama3-8b b1 | 2047 | 1 | 1.02 | 1.02 |
| llama3-8b b1 | 2047 | 8 | 1.38 | **0.71** |
| llama3-8b b1 | 2047 | 16 | 1.21 | **0.61** |
| llama3-8b b1 | 8191 | 1 | 1.67 | **1.06** |
| llama3-8b b1 | 8191 | 8 | 2.72 | **1.67** |
| llama3-8b b1 | 8191 | 16 | 2.31 | **1.57** |
| llama3-8b b4 | 2047 | 1 | 1.78 | **1.15** |
| llama3-8b b4 | 2047 | 8 | 4.39 | **2.92** |
| llama3-8b b4 | 2047 | 16 | 3.50 | **2.14** |
| phi3-mini-4k b1 | 511 | 1 | 2.82 | **2.72** |
| phi3-mini-4k b1 | 511 | 8 | 7.08 | **3.84** |
| phi3-mini-4k b1 | 511 | 16 | 3.70 | 5.61 |
| phi3-mini-4k b1 | 2047 | 1 | 2.78 | **1.67** |
| phi3-mini-4k b1 | 2047 | 8 | 1.84 | **1.03** |
| phi3-mini-4k b1 | 2047 | 16 | 1.46 | **0.94** |
| phi3-mini-4k b1 | 8191 | 1 | 3.45 | **1.97** |
| phi3-mini-4k b1 | 8191 | 8 | 6.56 | **3.60** |
| phi3-mini-4k b1 | 8191 | 16 | 6.63 | **3.69** |
| phi3-mini-4k b4 | 2047 | 1 | 3.59 | **2.00** |
| phi3-mini-4k b4 | 2047 | 8 | 7.30 | **3.99** |
| phi3-mini-4k b4 | 2047 | 16 | 7.68 | **4.42** |
| qwen2.5-0.5b b1 | 511 | 1 | 1.85 | **1.68** |
| qwen2.5-0.5b b1 | 511 | 8 | 1.17 | **1.13** |
| qwen2.5-0.5b b1 | 511 | 16 | 2.17 | **1.70** |
| qwen2.5-0.5b b1 | 2047 | 1 | 1.96 | **1.40** |
| qwen2.5-0.5b b1 | 2047 | 8 | 0.83 | **0.69** |
| qwen2.5-0.5b b1 | 2047 | 16 | 1.42 | **0.95** |
| qwen2.5-0.5b b1 | 8191 | 1 | 1.59 | **1.03** |
| qwen2.5-0.5b b1 | 8191 | 8 | 0.74 | **0.37** |
| qwen2.5-0.5b b1 | 8191 | 16 | 0.85 | **0.64** |
| qwen2.5-0.5b b4 | 2047 | 1 | 1.33 | **1.21** |
| qwen2.5-0.5b b4 | 2047 | 8 | 1.39 | **0.89** |
| qwen2.5-0.5b b4 | 2047 | 16 | 1.80 | **1.06** |
| qwen3-0.6b b1 | 511 | 1 | 3.26 | **2.97** |
| qwen3-0.6b b1 | 511 | 8 | 2.70 | **2.64** |
| qwen3-0.6b b1 | 511 | 16 | 2.13 | **1.86** |
| qwen3-0.6b b1 | 2047 | 1 | 1.89 | **1.39** |
| qwen3-0.6b b1 | 2047 | 8 | 1.04 | **0.85** |
| qwen3-0.6b b1 | 2047 | 16 | 1.15 | **0.65** |
| qwen3-0.6b b1 | 8191 | 1 | 2.41 | **1.42** |
| qwen3-0.6b b1 | 8191 | 8 | 3.08 | **1.45** |
| qwen3-0.6b b1 | 8191 | 16 | 3.36 | **1.63** |
| qwen3-0.6b b4 | 2047 | 1 | 2.60 | **1.54** |
| qwen3-0.6b b4 | 2047 | 8 | 4.41 | **2.15** |
| qwen3-0.6b b4 | 2047 | 16 | 4.01 | **1.85** |

**Root cause fixed:** the present-KV tensors were materialised into a scratch
buffer and then copied into the graph outputs, so every decode step copied the
whole cache twice. Writing the appended KV straight into the `present_*`
bindings removes one full copy.

**Where we still lose:** wide-batch and many-head decode remains behind —
phi3-mini b4 at 2.00–4.42 and llama3 b4 at 1.15–2.92. Short contexts (past 511)
are dominated by fixed per-run overhead, and one of them — phi3-mini p511 at
t=16 — is an outright **regression** (3.70 → 5.61); it sits inside its own
dispersion band (base [3.57–6.22], new [5.26–6.29]) but it is not a win and is
listed as such.

All 48 measured cells are listed; **bold** marks the 45 that improved. Nothing
is omitted.

## 2. MultiHeadAttention / SDPA — encoder and cross-attention

Full grid from one interleaved session (`mha_grid.csv`), after vectorising the
x86 `sdpa_f32` path. Median [min–max] of the per-trial ratios.

| model | t=1 | t=8 | t=16 |
|---|--:|--:|--:|
| bert-base s128 | 5.75 [5.44–5.93] | 16.19 [8.51–19.96] | 14.91 [11.84–16.11] |
| bert-base b8 s128 | 5.40 [5.22–5.61] | 8.22 [7.17–9.96] | 8.10 [7.51–8.24] |
| bert-base s384 | 4.33 [4.29–4.34] | 7.95 [7.73–12.35] | 7.31 [6.92–9.90] |
| bert-large s128 | 5.50 [5.09–6.07] | 11.36 [10.81–18.26] | 10.92 [10.22–13.67] |
| clip-l14 s257 | 4.51 [4.40–4.67] | 9.98 [6.64–12.58] | 10.09 [7.22–10.80] |
| vit-b16 s197 | 4.83 [4.60–4.83] | 8.64 [5.18–10.80] | 10.11 [9.79–12.16] |
| whisper cross s1500 | 3.87 [3.79–3.98] | 5.74 [5.59–6.27] | 6.74 [6.66–7.11] |

**MHA is our worst operator against ORT: 3.9×–16× behind**, and — as with the
pre-fix transforms — the gap *widens* with thread count, which is the signature
of insufficient parallelism rather than of slow arithmetic.

> **Cross-session caveat.** The same `mha_whisper_cross_s1500` graph, measured
> as an unchanged *control* in the §3 session, gave 1.26 / 2.76 / 2.81 rather
> than 3.87 / 5.74 / 6.74. The two sessions used different `--runs`/`--warmups`,
> which changes how much of ORT's one-off packing is amortised. Both are real
> measurements of the same code; they are not interchangeable. Within-session
> comparisons are what this document relies on, and the honest overall statement
> is that MHA is somewhere between **1.3× and 16× behind** ORT depending on
> shape, thread count and measurement regime — with the well-sampled encoder
> grid above sitting at 3.9×–16×.

`InferenceSession` has **no ORT fallback**, so declining assignment is not
available as an honest escape hatch; the remaining gap is discussed in §5.

## 3. The transforms that surround attention

Requirement: prove the attention wins are not time pushed into a neighbouring
node. Isolated single-node graphs, `base` = before, `new` = after vectorising
and parallelising `Softmax` and `RotaryEmbedding`.

### Softmax

All 21 measured cells; **bold** marks the 21 that improved.

| shape | t | base | new |
|---|--:|--:|--:|
| bert-base b8 s128 (12288×128) | 1 | 13.44 | **1.50** |
| bert-base b8 s128 (12288×128) | 8 | 76.03 | **5.07** |
| bert-base b8 s128 (12288×128) | 16 | 124.01 | **9.28** |
| decode h32 kv1024 | 1 | 9.64 | **1.29** |
| decode h32 kv1024 | 8 | 17.80 | **2.23** |
| decode h32 kv1024 | 16 | 17.49 | **2.41** |
| decode h32 kv2048 | 1 | 15.68 | **1.39** |
| decode h32 kv2048 | 8 | 32.81 | **3.09** |
| decode h32 kv2048 | 16 | 34.15 | **3.35** |
| decode h32 kv4096 | 1 | 16.94 | **1.40** |
| decode h32 kv4096 | 8 | 53.05 | **4.88** |
| decode h32 kv4096 | 16 | 62.74 | **5.04** |
| decode h32 kv8192 | 1 | 17.91 | **1.42** |
| decode h32 kv8192 | 8 | 71.10 | **6.39** |
| decode h32 kv8192 | 16 | 83.66 | **7.89** |
| prefill h32 s512 | 1 | 12.92 | **2.54** |
| prefill h32 s512 | 8 | 56.37 | **5.87** |
| prefill h32 s512 | 16 | 60.13 | **6.17** |
| whisper cross (30000×1500) | 1 | 11.09 | **2.10** |
| whisper cross (30000×1500) | 8 | 41.66 | **6.05** |
| whisper cross (30000×1500) | 16 | 54.69 | **4.51** |

### RotaryEmbedding

All 12 measured cells; **bold** marks the 11 that improved. The
single regression — `llama3 b1 s1` at t=8 (8.15 → 9.02) — is a 4096-element
tensor whose time is allocation, not arithmetic; see §5.

| shape | t | base | new |
|---|--:|--:|--:|
| llama3 s128 | 1 | 22.48 | **2.53** |
| llama3 s128 | 8 | 72.41 | **10.51** |
| llama3 s128 | 16 | 83.75 | **17.38** |
| llama3 s512 | 1 | 29.43 | **6.11** |
| llama3 s512 | 8 | 55.58 | **6.38** |
| llama3 s512 | 16 | 61.36 | **9.95** |
| llama3 b8 s1 (decode) | 1 | 10.54 | **5.22** |
| llama3 b8 s1 (decode) | 8 | 9.88 | **5.12** |
| llama3 b8 s1 (decode) | 16 | 7.70 | **3.60** |
| llama3 b1 s1 (decode) | 1 | 10.08 | **8.99** |
| llama3 b1 s1 (decode) | 8 | 8.15 | 9.02 |
| llama3 b1 s1 (decode) | 16 | 8.11 | **6.93** |

**Root causes fixed:** scalar libm `exp` per element; fully serial loops (the
diagnostic is that the gap *grew* with thread count — ORT parallelizes, we did
not); and a full-tensor scratch buffer that was zeroed, filled and copied into
the output. RoPE additionally branched on tensor layout inside its innermost
loop.

Removing the scratch buffer mattered more than the vectorization for RoPE
alone. The figure that showed this — 2.247 ms → 0.571 ms on one cell — comes
from an ad-hoc bisecting run rather than from the base/new CSVs above, and is
recorded here as a diagnostic, like the figures in §5.

### Controls (graphs with no Softmax/RoPE node — must not move)

| model | t | base | new |
|---|--:|--:|--:|
| GQA decode p2047 | 1 / 8 / 16 | 1.04 / 0.72 / 0.82 | 0.97 / 0.56 / 0.67 |
| MHA whisper cross | 1 / 8 / 16 | 1.27 / 2.82 / 2.85 | 1.26 / 2.76 / 2.81 |
| GQA prefill q512 | 1 / 8 / 16 | 2.22 / 4.05 / 3.24 | 2.42 / 3.85 / 3.72 |

All within dispersion.

### Measured but not fixed

Per-model medians, min–max across the models in each group.

| op | t=1 | t=8 | t=16 |
|---|--:|--:|--:|
| Transpose BSNH→BNSH (bert b8 s128 / llama3 s512 / whisper s1500) | 13.8–24.8 | 54.9–77.9 | 68.3–120.8 |
| KV-cache `Concat` (llama3 p1023…p8191, b1 and b8) | 2.6–5.8 | 6.3–9.2 | 5.5–13.2 |

These are generic tensor ops rather than attention kernels and were deliberately
left out of scope. They are a real, quantified gap.

## 4. Mixture of Experts

`com.microsoft::MoE`, single node, top-k routing, grouped experts.
`base` = before, `fixed` = after removing the per-call expert-weight copy.

| config | tokens | t | base | fixed |
|---|--:|--:|--:|--:|
| mixtral h1024 i3584 e8 | 1 | 8 | 20.62 | **0.73** |
| mixtral h1024 i3584 e8 | 1 | 16 | 21.90 | **0.71** |
| mixtral h1024 i3584 e8 | 32 | 8 | 52.84 | **1.43** |
| mixtral h1024 i3584 e8 | 32 | 16 | 56.14 | **1.22** |
| mixtral h1024 i3584 e8 | 512 | 8 | 9.56 | **1.62** |
| mixtral h1024 i3584 e8 | 512 | 16 | 7.87 | **1.59** |
| phi3.5-moe h2048 i6400 e4 | 1 | 8 | 12.32 | **0.98** |
| phi3.5-moe h2048 i6400 e4 | 1 | 16 | 15.93 | **0.78** |
| phi3.5-moe h2048 i6400 e4 | 32 | 8 | 21.74 | **0.88** |
| phi3.5-moe h2048 i6400 e4 | 32 | 16 | 13.26 | **0.65** |
| phi3.5-moe h2048 i6400 e4 | 512 | 8 | 2.57 | **0.82** |
| phi3.5-moe h2048 i6400 e4 | 512 | 16 | 3.36 | **0.86** |
| qwen3-moe h2048 i768 e16 | 1 | 8 | 24.87 | **1.74** |
| qwen3-moe h2048 i768 e16 | 1 | 16 | 24.10 | **1.56** |
| qwen3-moe h2048 i768 e16 | 32 | 8 | 38.27 | **2.37** |
| qwen3-moe h2048 i768 e16 | 32 | 16 | 16.34 | **1.47** |
| qwen3-moe h2048 i768 e16 | 512 | 8 | 4.05 | **1.56** |
| qwen3-moe h2048 i768 e16 | 512 | 16 | 6.14 | **1.53** |

**Root cause fixed:** every forward pass transposed the full expert weight bank
and `into_owned()`-copied it, so a decode step costing microseconds of arithmetic
first copied hundreds of MiB.

**Where we win:** phi3.5-moe at every measured point (0.65–0.98), mixtral decode
(0.71–0.73).
**Where we still lose:** all qwen3-moe cells (1.47–2.37) — 16 experts × top-8
is many small GEMMs, where ORT's grouped path amortises better — and mixtral
prefill (1.22–1.62).

## 5. The residual gap is dominated by per-run output allocation

Graph outputs are freshly allocated from the system allocator every run
(`CpuExecutionProvider::allocate` → `HostAllocator`). Internal values *are*
reused. glibc `mmap`s allocations ≥128 KiB, so those pages re-fault on every
run.

Evidence (these four figures come from ad-hoc profiling runs, not from the
per-trial CSVs behind the tables above, and are recorded here as *diagnostics*
rather than as reproducible measurements):

* A `[1,1,4096]` RoPE — 16 KiB in, 16 KiB out, a few thousand multiply-adds —
  still takes **~67 µs**.
* GQA decode p2047 at t=1: **~4.9 ms** end-to-end vs **~3.6 ms** measured
  in-kernel, i.e. roughly a quarter of wall time outside the kernel.

The reproducible corroboration is in the tables: the RoPE cells whose tensors
are smallest (`llama3 b1 s1`, 8.99, and `b8 s1`, 5.22) improved *least* from a
change that made the arithmetic 5×–10× cheaper, because arithmetic is not what
they spend their time on.

**An EP-level scratch arena for graph outputs is worth more than any further
kernel work on these ops.** This is the single highest-value remaining item.

## 6. GQA fusion gate: cache-topology sweep

The KV-group-fused decode path is opt-in behind a minimum-attended-KV-bytes
gate. The gate was originally a flat 8 MiB calibrated on one host. Sweeping the
per-head working set against L3:

| per-head working set | t=1 | t=4 | t=8 | t=16 | t=32 |
|---|--:|--:|--:|--:|--:|
| 1 MiB | **0.90** / 1.35 | **1.57** / 1.75 | 1.03 / **0.98** | **0.67** / 0.75 | **1.54** / 1.61 |
| 2 MiB | 1.00 / **0.81** | **0.73** / 1.48 | 1.24 / **1.06** | **1.36** / 1.37 | **0.80** / 1.07 |
| 4 MiB | 0.98 / **0.89** | **0.87** / 0.97 | 0.78 / **0.73** | **0.61** / 0.74 | **0.84** / 0.99 |
| 8 MiB | 1.05 / **0.98** | 1.03 / **1.02** | 1.50 / **1.37** | 1.52 / **1.26** | **1.89** / 1.94 |
| 16 MiB | 1.00 / **0.91** | 1.14 / **0.86** | 1.23 / **1.06** | **1.29** / 1.41 | **1.22** / 1.32 |
| 32 MiB | 0.90 / **0.79** | 1.17 / **0.79** | 1.28 / **0.90** | **1.23** / 1.36 | 1.15 / **1.11** |

Each cell is `off / on`; **bold** is the faster arm. The fused path wins **16 of
the 30** cells — 11 of the 15 at ≥8 MiB, but only 5 of the 15 below it. It is
directionally better at and above 8 MiB *at low thread counts* — but it loses at
8 MiB t=32 (1.89 → 1.94), at 16 MiB t=16 (1.29 → 1.41) and at 32 MiB t=16 (1.23
→ 1.36), so "better above 8 MiB" is a tendency, not a rule. Below 8 MiB it is
genuinely mixed, and its single worst cell is 2 MiB t=4, where enabling fusion
costs 2.0× (0.73 → 1.48).

The gate is therefore expressed topology-relatively as
`max(last_level_cache_bytes / fused_tasks, 8 MiB)`, with the 8 MiB floor — not
the topology model — being what makes it safe, and an unreadable-sysfs sentinel
that collapses to the floor. It remains **opt-in**: no general superiority is
claimed, and the 1–4 MiB crossover is unresolved below the ±9% dispersion.

## Validation

* Parity (`parity=PASS`) recorded on **all 1194** measured cells across the six
  result files behind this document — before and after each change, with no
  failures. `ab.py` now tags any cell containing a `parity=FAIL` trial as
  `PARITY_FAIL=n/m` in the medians summary and prints a closing warning, so a
  numerically wrong arm can no longer produce a median that reads as a clean
  win.
* Bit-exactness or ≤1e-6 absolute agreement against independent scalar oracles
  for every rewritten kernel, across both layouts, both rotation modes, full and
  partial rotary, and sizes either side of each parallel threshold.
* Fully-masked (`-inf`) rows are pinned to reproduce ORT's NaN rather than being
  silently "fixed".
* Aliased output bindings (output over input buffer) are covered per kernel.

## Honest limitations

1. One host, one microarchitecture. AVX2 without AVX-512; no ARM measurements.
2. Synthetic tensor contents; production geometry only.
3. Single-node graphs isolate the operator but overweight fixed session
   overhead at small sizes.
4. The host is contended; absolutes are not comparable across sessions.
5. MHA (3.9×–16×), qwen3-moe (1.47–2.37×), mixtral prefill (1.22–1.62×),
   Softmax (1.29–9.28×), RoPE (2.53–17.38×), standalone `Transpose`
   (13.8–120.8×) and KV-cache `Concat` (2.6–13.2×) all remain behind ORT and
   are not fixed here. **`Transpose` is addressed in Phase 3 below**
   (13.2–110.7× → 0.91–9.46×); the rest are re-measured there and remain
   behind.
6. The §2 MHA grid and the §3 MHA control disagree by ~3× on the same graph
   across sessions. Only within-session comparisons are load-bearing.

---

# Phase 3 — the executor, not the kernels

Everything above measures kernels. Phase 3 started from the observation that the
worst-losing areas in the table — standalone `Transpose` at 13.8–120.8× and
`Concat` at 2.6–13.2× — were losing in code that *is not a kernel at all*, and
that no amount of kernel work would move them.

Same host as every measurement above: AMD EPYC 9V74, 16C/32T, 1 NUMA node,
32 MiB shared L3, AVX2+FMA (no AVX-512). Same harness (`scripts/ort_ab/ab.py`),
same interleaving discipline, same rule that only paired ratios from a single
driver invocation are publishable.

**Every table in this section is complete.** No row is omitted, including the
rows where the change did nothing and the rows where it lost.

## 6. Per-run `mmap`/`munmap` on graph outputs (#1075, merged `c49418452`)

`try_move_host_output` moves an output's buffer into the returned `Tensor` and
then does `buffer_shapes.remove(&vid)` — commented "force a fresh allocation next
run". `HostAllocator` is plain `std::alloc`, and glibc `mmap`s anything at or
above `M_MMAP_THRESHOLD` and `munmap`s it on free, so every run re-faulted and
the kernel re-zeroed the pages.

Measured on `sm_whisper_cross` with `strace -c` and `/usr/bin/time -v`:

| | runs=5 | runs=45 | per run |
|---|--:|--:|--:|
| `munmap` before | 24 | 64 | 1.0 |
| `munmap` after | 16 | 17 | 0.025 |
| minor faults before | 12082 | 32521 | 511 |
| minor faults after | 9016 | 9950 | 23 |

`LargeAllocCache` is a per-EP free list keyed on the exact `(bytes, align)` pair
(Rust requires the originating `Layout` at `dealloc`, so exact match keeps the
layout an invariant of the block), 8 shards, band `[256 KiB, 1 GiB]`, 2 GiB
default budget enforced *before* insertion, `0xA5` poison-fill of recycled blocks
under `debug_assertions`. `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES=0` disables it.

The poison-fill is itself the falsifier for the one thing that could have gone
wrong: no kernel in the tree depends on a freshly allocated output buffer being
zero. 1224 lib tests pass with recycled blocks filled with `0xA5`.

57 interleaved cells, `nocache` vs `cache`, same binary. Full result:

| model | t | nocache (ms) | cache (ms) | change |
|---|--:|--:|--:|--:|
| kvcat_llama3_p1023 | 1 | 4.835 | 1.841 | **−61.9%** |
| kvcat_llama3_p1023 | 8 | 14.244 | 5.553 | **−61.0%** |
| kvcat_llama3_p8191 | 16 | 8.415 | 5.056 | **−39.9%** |
| kvcat_llama3_b8_p2047 | 16 | 7.651 | 4.489 | **−41.3%** |
| sm_prefill_h32_s512 | 1 | 2.476 | 1.755 | **−29.1%** |
| sm_whisper_cross | 1 | 2.209 | 1.732 | **−21.6%** |
| tr_bert_b8_s128 | 1 | — | — | +25.0% (dispersion 51–109) |
| tr_whisper_s1500 | 1 | — | — | +12.5% (dispersion 51–109) |
| rope_llama3_s512 | 8 | — | — | +21.0% (dispersion 51–109) |
| *(remaining 48 cells)* | | | | within noise |

The three cells that moved the wrong way all sit in the `Transpose`-dominated
models whose per-trial dispersion at the time reached 51–109; the harness could
not resolve them. They are reported as neutral, not as wins.

## 7. View graph outputs: three full copies (#1076, merged `fdc5007e8`)

I rewrote the `Transpose` kernel first. It moved `tr_bert_b8_s128` from 6.87 ms
to 6.45 ms — **essentially nothing**. The reason is that
`TransposeKernel::view_outputs` returns a zero-copy strided `ViewOutput`, so for
a graph-output transpose **the kernel never runs**. All the time was in the
executor:

1. `contiguous_bytes` staged `vec![0u8; buf.len()]` and `copy_to_host`d the
   *whole source* into it before gathering a byte, even when the buffer was
   already host-resident.
2. `gather_view` walked a scalar odometer recomputing a full rank-length dot
   product **per element**, one element at a time, single-threaded.
3. The resulting `Vec` went to `Tensor::from_raw`, which allocated the same bytes
   a *second* time and memcpyd between them.

Fixed by reading host buffers in place, collapsing the view geometry (drop
size-1 axes, then fuse axis `i` into `i+1` exactly when
`strides[i] == strides[i+1] * shape[i+1]`), copying the largest contiguous run
the collapsed geometry allows, advancing an incremental odometer, fanning out
across rayon above 256 KiB, and gathering straight into the output tensor's own
allocation via `Tensor::from_host_fill`.

Interleaved, same binary pair, 5 trials × 9 runs × 3 warmups, ratio = ours/ORT
p50 (lower is better), **all 9 cells**:

| model | t | before | after | change | native before | native after |
|---|--:|--:|--:|--:|--:|--:|
| tr_bert_b8_s128 | 1 | 26.59 | **0.91** | **29.2×** | 10.53 ms | 0.249 ms |
| tr_bert_b8_s128 | 8 | 49.71 | **2.46** | **20.2×** | 7.22 ms | 0.341 ms |
| tr_bert_b8_s128 | 16 | 64.38 | **4.10** | **15.7×** | 10.34 ms | 0.547 ms |
| tr_llama3_s512 | 1 | 13.21 | **1.12** | **11.8×** | 14.00 ms | 0.966 ms |
| tr_llama3_s512 | 8 | 71.37 | **5.39** | **13.2×** | 14.48 ms | 0.695 ms |
| tr_llama3_s512 | 16 | 110.70 | **9.46** | **11.7×** | 22.16 ms | 0.930 ms |
| tr_whisper_s1500 | 1 | 20.55 | **1.46** | **14.1×** | 16.10 ms | 0.901 ms |
| tr_whisper_s1500 | 8 | 68.83 | **5.56** | **12.4×** | 16.13 ms | 0.703 ms |
| tr_whisper_s1500 | 16 | 90.09 | **9.21** | **9.8×** | 23.04 ms | 1.010 ms |

240/240 cells parity PASS. `tr_bert_b8_s128` at t=1 now **beats ORT** (0.91×).

We still lose at 8 and 16 threads. Sweeping `RAYON_NUM_THREADS` over 1/4/8/16/32
on `tr_bert_b8_s128` moves native p50 only between 0.53 and 0.63 ms, so more
workers is not the missing lever: the residue is ORT reading and writing less,
which needs a fused permute-with-consumer, not a faster gather.

## 8. KV-cache `Concat` — negative result, not shipped

`ConcatKernel` already bulk-memcpys per slab but is explicitly single-threaded
while ORT parallelizes; the arena A/B showed exactly that signature (our time
flat as ORT scaled with threads). I parallelized it across output rows.

**All 15 cells**, ratio = ours/ORT p50:

| model | t | before | after |
|---|--:|--:|--:|
| kvcat_llama3_p1023 | 1 | 1.84 | 1.93 |
| kvcat_llama3_p1023 | 8 | 6.08 | 5.64 |
| kvcat_llama3_p1023 | 16 | 6.21 | **8.85** |
| kvcat_llama3_p2047 | 1 | 1.85 | 1.85 |
| kvcat_llama3_p2047 | 8 | 4.30 | 4.31 |
| kvcat_llama3_p2047 | 16 | 5.90 | 5.55 |
| kvcat_llama3_p4095 | 1 | 1.88 | 1.93 |
| kvcat_llama3_p4095 | 8 | 5.32 | 5.17 |
| kvcat_llama3_p4095 | 16 | 5.21 | 5.11 |
| kvcat_llama3_p8191 | 1 | 1.93 | 1.91 |
| kvcat_llama3_p8191 | 8 | 4.83 | 4.37 |
| kvcat_llama3_p8191 | 16 | 5.82 | 5.06 |
| kvcat_llama3_b8_p2047 | 1 | 1.89 | 1.88 |
| kvcat_llama3_b8_p2047 | 8 | 4.36 | 4.16 |
| kvcat_llama3_b8_p2047 | 16 | 4.59 | 4.39 |

No win anywhere, and a clear regression at `p1023` t=16 (bands `[5.32–7.08]` vs
`[7.82–9.62]`, non-overlapping). The KV concat inputs are not all contiguous, so
the fan-out mostly declines to the serial gather and only adds threshold
overhead. **The change was discarded.** The real fix is an appendable/paged KV
cache that never re-copies history, which is not a `Concat` change at all.

## 9. What Phase 3 did to the rest of the table

Same-binary-pair A/B, Phase 2 head (`ab6cb0168`) vs Phase 3 head, ratio =
ours/ORT p50. **All measured cells**, including every unchanged one.

### Softmax (18 cells)

| model | t | Phase 2 | Phase 3 |
|---|--:|--:|--:|
| sm_decode_h32_kv1024 | 1 | 1.27 | 1.28 |
| sm_decode_h32_kv1024 | 8 | 2.38 | **2.05** |
| sm_decode_h32_kv1024 | 16 | 2.37 | **2.11** |
| sm_decode_h32_kv4096 | 1 | 1.40 | 1.34 |
| sm_decode_h32_kv4096 | 8 | 4.98 | 5.06 |
| sm_decode_h32_kv4096 | 16 | 4.67 | 4.60 |
| sm_decode_h32_kv8192 | 1 | 1.34 | 1.32 |
| sm_decode_h32_kv8192 | 8 | 6.61 | 6.99 |
| sm_decode_h32_kv8192 | 16 | 6.97 | 7.02 |
| sm_prefill_h32_s512 | 1 | 2.52 | **1.76** |
| sm_prefill_h32_s512 | 8 | 5.80 | **5.08** |
| sm_prefill_h32_s512 | 16 | 5.77 | **4.58** |
| sm_bert_b8_s128 | 1 | 1.53 | 1.56 |
| sm_bert_b8_s128 | 8 | 5.15 | 4.89 |
| sm_bert_b8_s128 | 16 | 9.29 | 8.83 |
| sm_whisper_cross | 1 | 2.21 | **1.73** |
| sm_whisper_cross | 8 | 5.10 | 5.17 |
| sm_whisper_cross | 16 | 5.19 | **4.67** |

The wins are exactly the allocation-bound cells the arena helps
(`sm_prefill_h32_s512` native 11.16 → 7.60 ms, `sm_whisper_cross` 59.05 →
46.10 ms at t=1). Decode-shaped softmaxes are unchanged — they allocate almost
nothing. **Every cell still loses to ORT.**

### RotaryEmbedding (12 cells)

| model | t | Phase 2 | Phase 3 |
|---|--:|--:|--:|
| rope_llama3_s1 | 1 | 9.83 | 9.70 |
| rope_llama3_s1 | 8 | 9.23 | **6.93** |
| rope_llama3_s1 | 16 | 6.97 | 6.89 |
| rope_llama3_b8_s1 | 1 | 5.82 | 5.90 |
| rope_llama3_b8_s1 | 8 | 5.75 | 5.77 |
| rope_llama3_b8_s1 | 16 | 4.24 | **3.95** |
| rope_llama3_s128 | 1 | 2.66 | 2.79 |
| rope_llama3_s128 | 8 | 12.35 | 12.59 |
| rope_llama3_s128 | 16 | 17.90 | 17.21 |
| rope_llama3_s512 | 1 | 2.05 | **1.53** |
| rope_llama3_s512 | 8 | 6.21 | **5.78** |
| rope_llama3_s512 | 16 | 8.62 | 8.83 |

**Every cell still loses to ORT**, which is why float32 `RotaryEmbedding` now
defers to the host in the plugin EP (#1078).

### MHA / SDPA (15 cells)

| model | t | Phase 2 | Phase 3 | native before | native after |
|---|--:|--:|--:|--:|--:|
| mha_bert_base_s128 | 1 | 3.15 | **2.70** | 2.62 ms | 2.26 ms |
| mha_bert_base_s128 | 8 | 10.49 | **8.89** | 7.22 ms | 6.04 ms |
| mha_bert_base_s128 | 16 | 10.58 | 9.84 | 7.66 ms | 7.34 ms |
| mha_bert_base_b8_s128 | 1 | 2.49 | **2.20** | 17.30 ms | 15.31 ms |
| mha_bert_base_b8_s128 | 8 | 4.00 | 3.94 | 17.70 ms | 14.96 ms |
| mha_bert_base_b8_s128 | 16 | 4.74 | **4.07** | 22.65 ms | 23.85 ms |
| mha_vit_b16_s197 | 1 | 2.34 | **2.02** | 4.50 ms | 3.89 ms |
| mha_vit_b16_s197 | 8 | 8.63 | 8.92 | 8.99 ms | 7.73 ms |
| mha_vit_b16_s197 | 16 | 8.38 | 7.88 | 9.33 ms | 8.87 ms |
| mha_clip_l14_s257 | 1 | 2.00 | **1.77** | 8.52 ms | 7.54 ms |
| mha_clip_l14_s257 | 8 | 5.27 | 5.21 | 10.35 ms | 9.43 ms |
| mha_clip_l14_s257 | 16 | 6.93 | **6.29** | 13.15 ms | 11.82 ms |
| mha_whisper_cross_s1500 | 1 | 1.27 | 1.26 | 37.71 ms | 37.28 ms |
| mha_whisper_cross_s1500 | 8 | 3.07 | **2.65** | 25.61 ms | 24.65 ms |
| mha_whisper_cross_s1500 | 16 | 3.35 | 3.12 | 26.29 ms | 25.89 ms |

Phase 3 bought MHA a **5–16% end-to-end improvement**, entirely from the arena
and the view-materialization path — no MHA kernel was touched. Native time falls
in 14 of 15 cells. **Every cell still loses**, best 1.26×, worst 9.84×.

### MoE (15 cells)

| model | t | Phase 2 | Phase 3 |
|---|--:|--:|--:|
| moe_qwen3moe_e16_t1 | 1 | 1.02 | 1.02 |
| moe_qwen3moe_e16_t1 | 8 | 1.78 | **1.57** |
| moe_qwen3moe_e16_t1 | 16 | 1.27 | 1.25 |
| moe_qwen3moe_e16_t32 | 1 | 1.02 | 1.02 |
| moe_qwen3moe_e16_t32 | 8 | 2.00 | 2.00 |
| moe_qwen3moe_e16_t32 | 16 | 1.51 | **1.30** |
| moe_qwen3moe_e16_t512 | 1 | 1.05 | 1.04 |
| moe_qwen3moe_e16_t512 | 8 | 2.09 | 2.32 |
| moe_qwen3moe_e16_t512 | 16 | 1.52 | **1.12** |
| moe_mixtral_e8_t32 | 1 | 1.02 | 1.02 |
| moe_mixtral_e8_t32 | 8 | 2.15 | 2.15 |
| moe_mixtral_e8_t32 | 16 | 1.27 | 1.59 |
| moe_mixtral_e8_t512 | 1 | 1.09 | 1.08 |
| moe_mixtral_e8_t512 | 8 | 2.05 | 2.42 |
| moe_mixtral_e8_t512 | 16 | 1.55 | 1.80 |

MoE is unchanged by Phase 3, as expected — it allocates once and transposes
nothing. Note these ratios are much closer to parity than §2's MoE grid
suggested (1.02–2.42 vs 0.65–2.37 there); the two grids were measured in
different sessions and, per the standing caveat, only within-session comparisons
are load-bearing. Within *this* session MoE is at parity at t=1 and loses
1.1–2.4× at 8 and 16 threads, where ORT's expert loop threads better than ours.

## 10. Assignment matrix after Phase 3

What the **plugin** EP asks ORT to hand over. This does not affect a native
`InferenceSession`, which has no host to defer to and runs every kernel.

| op | dtype | claim | evidence |
|---|---|---|---|
| `Tanh`/`Sigmoid`/`Gelu`/`Sqrt`/`Erf` | float32 | defer | 0.02–0.87× |
| same | float16 | defer | 0.59–0.96× |
| same | bfloat16 | **claim** | ORT has no kernel — capability, not perf |
| `Gelu` | non-float32 | **claim** | ORT inlines the function; deferring measured 0.014× |
| `MatMul`/`Gemm` | float32 ≥1 M weights | defer | slower at every thread count 2–16 |
| `MatMul` | float16 | defer | 2.5× @1T, 5.3–7.8× @2–32T |
| `QLinearMatMul` | any | defer | 2.2–22× |
| `MatMulNBits` int4 | decode | defer | 1.78–2.41× @8T |
| `MatMulNBits` int8 | ≥256 static rows | defer | 1.15–1.41× |
| **`RotaryEmbedding`** | **float32** | **defer (new, #1078)** | **12/12 cells lose, 1.53–17.21×** |
| `RotaryEmbedding` | non-float32 | claim | unmeasured |
| `Softmax` | any | **claim** | loses 1.28–8.83× standalone, but anchors attention fusion — see below |
| `Transpose` | any | claim | 0.91× at t=1; view path is zero-copy, deferring would force materialization |
| `Concat` | any | claim | loses 1.85–5.5×, but is the KV-cache path; deferring breaks view chaining |
| GQA / MHA / MoE | any | claim | no ORT-superior standalone alternative in the plugin path |

**Why `Softmax` is not deferred despite losing every cell.** It is the anchor of
this repo's own attention fusion: `MatMul → (Mul|Div) → [Add(mask)] → Softmax →
MatMul` collapses into a fused SDPA kernel in `onnx-runtime-optimizer`'s fusion
pass (`fusion/mod.rs:207,528`), which runs on plugin-claimed subgraphs via
`ep.custom_passes()`. Deferring the standalone node removes the anchor and
fragments the SDPA core across the EP boundary — claim QKᵀ, hand the scores to
ORT, claim the PV matmul. The 18-cell grid above was measured on **isolated
single-op graphs under `ORT_DISABLE_ALL`**, which cannot see that effect, so it
does not support deferral. A future deferral needs a fused-graph measurement.

## 11. Phase 3 limitations

1. Still one host, one microarchitecture, AVX2 without AVX-512, no ARM.
2. The 256 KiB allocation-cache floor and parallel-gather threshold are **glibc**
   numbers, chosen to sit above `M_MMAP_THRESHOLD`. Untested on musl and Windows.
3. Native p50 rises with `ONNX_GENAI_CPU_DECODE_THREADS` on the transform models
   (0.204 → 0.557 ms on `tr_bert_b8_s128`) even though the gather does not read
   that variable. Isolated to the co-resident ORT thread pool: with
   `--native-threads 1` fixed and ORT's pool swept 1/8/16, native stays at
   0.203–0.242 ms. Both arms see it identically, so before/after comparisons are
   unaffected, but the absolute t=8/16 numbers are pessimistic for both sides.
4. `Transpose` at 8/16 threads (2.5–9.5×), MHA (1.26–9.84×), Softmax
   (1.28–8.83×), RoPE (1.53–17.21×), `Concat` (1.85–5.55×) and MoE at 8/16
   threads (1.1–2.4×) all still lose to ORT and are **not** fixed here.
5. No appendable/paged KV cache. `Concat` re-copies the full history every token;
   that is the single largest remaining structural gap.
6. MoE grouped/batched GEMM was not attempted. The measured MoE gap in this
   session (1.1–2.4× at 8/16 threads) is a threading-granularity gap, not an
   arithmetic one.
