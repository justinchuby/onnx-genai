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

> **WITHDRAWN — see §23.** Every `defer` row in this table is void. The project
> rule is now that a node this EP can execute is never handed to ORT's CPU EP,
> whatever the measurement says, and a losing row is a kernel to fix rather than
> a node to give away. The table is kept because the *ratios* are real evidence
> and they are the work queue; the `claim` column is not the current behaviour.

What the **plugin** EP asked ORT to hand over, as of Phase 3. This never
affected a native `InferenceSession`, which has no host to defer to and runs
every kernel.

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

---

# Phase 4 — the appendable KV cache

## 12. Half-precision KV caches never got the O(1) append (#1083)

### What was already there

`GroupQueryAttention` has supported ORT's `past_present_share_buffer` contract
for some time. When the caller binds the graph's `present_key`/`present_value`
outputs onto the **same memory** as the `past_key`/`past_value` inputs at full
physical capacity, `detect_inplace_kv`
(`crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs`) recognises
the aliasing structurally — identical origins, exact capacity, contiguous, K and
V disjoint — and the step appends only the new token's rows. History is already
resident; nothing is recopied.

So Phase 4's requirement 1, "append the new token without recopying full
history", was **half done before Phase 4 started**. The half that was missing is
the half production uses.

### The gap

`detect_inplace_kv` accepted **contiguous f32 only**. Exported decoders ship
their KV cache in `f16` (or `bf16`), so every real model failed the gate and fell
back to the owned-scratch path, which makes two full passes over the entire
cache on every decode step:

1. `fill_present` widens the whole `f16` history into f32 scratch, then
2. the trailing output writer narrows the whole f32 scratch back into the
   graph's `present_*` outputs.

Only step 2 is avoidable — the attention core genuinely needs f32 K/V — but it is
the expensive one. It is a **write** across `2 · B · H_kv · S · D` elements that
dirties the entire cache footprint every token, fed by an equally large f32 read.

A test on the pre-#1083 base had even pinned the gap in place:
`detect_inplace_kv_gate_rejects_f16_cache` asserted the rejection, with a comment
explaining that "the append path only supports contiguous f32". It was describing
a limitation, not a requirement. (#1083 keeps the test — the *f32* gate must
still decline half-precision — and rewords the comment to point at
`detect_inplace_kv_half`.)

### What shipped

`detect_inplace_kv_half` + `append_and_widen_half`. The gate is the same
structural proof, extended with an encoding check: `f16` and `bf16` share a width
but not a bit layout, so a mismatched past/present pair is declined rather than
reinterpreted. When it fires, only the new rows are narrowed into the resident
cache and `[0, total)` is widened back out for attention. The full-history narrow
disappears.

### Results — f16 decode, `append` ÷ `copy`, median of 4 independent sweeps

Both arms are today's shipping kernel on the same geometry; the only difference
is whether the caller bound `present` onto `past`. The `copy` arm is exactly what
every half-precision model runs on `main`.

| geometry | H_kv × D | kv=2048 | kv=4096 | kv=8192 | kv=16384 |
|---|---|---|---|---|---|
| qwen2.5-0.5b | 2 × 64 | 0.870 | 0.890 | 0.983 † | 0.945 † |
| qwen3-0.6b | 8 × 128 | 0.761 | **0.504** | 0.669 | **0.600** |
| llama-3.1-8b | 8 × 128 | 0.777 † | **0.554** | 0.692 | 0.684 |
| qwen2.5-7b | 4 × 128 | 0.799 | 0.917 | 0.787 | 0.789 |

† per-run ratio range reaches or crosses 1.0 across the 4 sweeps (0.91–1.03,
0.92–1.06 and 0.77–1.03 respectively) — read these three as "no regression", not
as a gain. Every other cell's range stays strictly below 1.0. Absolute medians and full dispersion are in
PR #1083; all 16 cells are reported there, none omitted.

The pattern is what the mechanism predicts: the win tracks cache size. The three
wide-KV geometries (`H_kv · D` of 1024 or 512) gain 8–50%; `qwen2.5-0.5b`, whose
`H_kv · D = 128` is the smallest in the set, gains 11–13% at short context and
nothing measurable at long context, because the removed traffic is small next to
the attention work.

### Bytes removed — exact, not measured

Per decode step per layer the eliminated narrow is `2 · B · H_kv · C · D` half
writes plus the equally large f32 reads feeding it, replaced by
`2 · B · H_kv · q_seq · D` half writes. At `q_seq = 1` that is a factor of `C`.
For qwen3-0.6b at `C = 16384`: **≈67 MB of writes and ≈134 MB of reads removed
per step per layer**.

### What this is NOT

**This is not an "ours beats ORT" result.** ORT CPU's GQA already appends in
place under `past_present_share_buffer`; so did ours, for f32. This closes a
parity gap on the dtype production actually ships. It does not open a lead, and
it should not be quoted as one.

## 13. The structural boundary: why `Concat`-shaped KV cannot append

The GQA path above works because `GroupQueryAttention` takes the valid length as
a runtime **value** input (`seqlens_k`, `total_sequence_length`), not from a
tensor shape. That is what lets `past` and `present` both be declared at physical
capacity `[B, H_kv, C, D]` while the kernel operates on `[0, total)`.

Models that build their cache with a plain `Concat(past, new)` instead cannot do
this, and the reason is not a missing optimisation — it is the binding ABI:

* `ExternalValue` (`crates/onnx-runtime-session/src/executor/state.rs:638-646`)
  carries `dtype, shape, accepts_subshape, ptr, len, alignment, device`. The one
  knob that looks like it might help — `accepts_subshape` — is a bounded per-dim
  `valid ≤ capacity` check (`accepts_output`), not strides. **There is no strides
  field.** A device-bound value is always dense.
* `DeviceIoBinding` (`crates/onnx-runtime-session/src/tensor.rs:316`) does carry
  `physical_shape` (capacity) and `logical_shape` (valid prefix), and
  `kernel_input_shape()` exposes the logical prefix to kernels. But because the
  value is dense, a logical prefix is only *correct* when the growing axis is the
  outermost one.
* KV caches are `[B, H_kv, S, D]` — `S` is axis 2. A capacity-backed view of that
  is `shape [B,H,S,D]` with strides `[H·C·D, C·D, D, 1]`, which is **not
  expressible** as a dense `ExternalValue`. Every head's slab would need shifting.

Consequently, for a `Concat`-shaped cache the history copy is semantically
required inside a single `Run`: the caller owns `past`, `present` is a distinct
graph output, and no legal aliasing makes the old bytes already correct at their
destination. Eliding it would need one of:

1. strides on `ExternalValue`/`DeviceIoBinding` — invasive, changes the binding
   contract for every EP;
2. a KV layout whose growth axis is outermost (`[S, B, H, D]`) — no exported
   decoder uses one; or
3. the GQA route above, which is what §12 extends.

This is reported as a boundary rather than a to-do because options 1 and 2 are
model/ABI changes, not kernel work. Phase 3's attempt to attack the same cost
from the `Concat` kernel side (parallelising the copy) is recorded as a negative
result in §8: no win in any of 15 cells and a non-overlapping regression at
`kvcat_llama3_p1023` t=16.

## 14. Phase 4 limitations

* §12's gain is only reachable when the caller binds `present` onto `past`. A
  session that lets the executor allocate a fresh `present` each step keeps the
  copy path, byte-for-byte unchanged.
* The §12 measurement is kernel-level, not a full ORT session A/B. That is
  deliberate: the aliased binding is a property of *how the caller drives the
  session*, and `scripts/ort_ab/ab.py` drives plain `session.run()`, which cannot
  express it. Both arms are the shipping kernel, so this is our-old-path versus
  our-new-path under one binding contract — not a kernel-vs-scalar proxy — but it
  is also not evidence about ORT.
* Criterion runs arms sequentially within a sweep. The interleaving here comes
  from repeating the whole sweep four times under varying host load, not from
  alternating arms inside one pass. On a contended host that matters: a
  **preliminary** sweep — taken before the four reported above and not part of
  their dispersion — showed two `qwen2.5-0.5b` cells *regressing* (ratios 1.285
  at kv=4096 and 1.080 at kv=16384). The 4-sweep medians falsified both, and
  neither regression is reproducible within the reported 0.81–0.93 and 0.92–1.06
  ranges. The raw preliminary CSV no longer exists, so that pair of numbers is
  recorded as provenance for why four sweeps are used, not as a result.
* Phase 4 requirements 2–5 (grouped MoE GEMM, fused transpose-with-consumer,
  the Softmax fusion-anchor session A/B, allocator-independent cache thresholds)
  are not addressed *in this section*. Requirement 4 is answered in §15 and
  requirement 5 in §16; requirements 2 and 3 remain open (§17).

## 15. The Softmax "fusion anchor" claim, finally measured (#1094)

`assignment_policy.rs` kept its claim on `Softmax` even though the isolated node
loses to ORT by 1.28–8.83x, on the grounds that it anchors the SDPA fusion. The
module's own comment conceded the gap: *"a claim to defer it needs a fused-graph
measurement this module does not have."* Phase 4 requirement 4 was to get that
measurement.

### The graph

`scripts/ort_ab/gen_sdpa_region.py` (added in #1094) emits the full region the
matcher wants — `MatMul → Div(scalar const) → [Add(mask)] → Softmax(axis=-1) →
MatMul` — with K pre-transposed to `[B,H,D,S_kv]` so the score product is a plain
`MatMul`. `ONNX_GENAI_PROFILE_OPS=1` confirms it collapses to a **single**
`com.microsoft::FusedAttention` node at **99.8% of node time**. So this measures
the fused region, not a proxy for it.

### First answer: the region loses, badly

Against a real ORT CPU session on identical graphs, options and thread counts —
3 geometries × 3 phases × mask/nomask × {1,8,16} threads = 54 cells — the fused
region lost in **48 of 54 cells**, by up to **22.8x**. Only 5 cells were ≥5%
faster, all at 1 thread and all unmasked. (These are the `base` arm of the
same-session A/B tabulated below, so they are directly comparable to the `new`
column. An earlier single-arm session on the same graphs independently gave
50 of 54 and 21.6x; the two agree, but only the same-session figures are quoted
as results, because cross-session absolutes on this host drift >4x.)

The shape of the loss was the clue: masked cells lost **1.07–22.8x** while
unmasked cells spanned **0.91–12.6x**, and the **seven worst cells were all
masked**. The mask was costing more than the attention.

### Cause

`FusedAttention` made three full passes over the score matrix, two of them
serial:

1. `for s in &mut scores { *s *= self.scale }` — serial, scalar.
2. `broadcast_apply(mask, .., |i, v| scores[i] += v)` — serial, one indirect
   closure call per element.
3. `softmax_rows_in_place` — parallel.

The score matrix is `batch·heads·seq_q·seq_k` floats: **33 MiB** for a 32-head
512-token prefill. Two serial passes over 33 MiB is ~66 MiB of single-threaded
memory traffic that no amount of softmax parallelism can shrink. That is the
ceiling, and it is why the masked cells were so much worse — the mask pass was
overhead the unmasked path never paid.

### Fix and result (#1094)

All three stages now run as one parallel fan-out over ~32 KiB row tiles, with
each worker resolving its own mask offsets from a global row index rather than
walking the mask serially from the start.

Measured same-session and interleaved, this branch against **its own merge
base**, both arms scored against ORT in the same process. 10 warmups, 30 runs,
3 interleaved trials, medians. Cross-session absolutes on this host drift >4x,
so only paired same-session ratios are reported.

The **27 unmasked cells are a negative control**: without a mask the change only
folds a scale multiply into the parallel fan-out and adds row tiling, neither of
which should alter the cost, so their spread is this host's noise floor. (This
is slightly weaker than a true no-op control — the tiling does change cache
behaviour and parallel granularity on that path too — but empirically it did not
move the median.)

| group | cells | median | band |
|---|---:|---:|---|
| **control** (unmasked) | 27 | **1.00x** | 0.76x .. 1.10x |
| masked | 27 | **1.26x** | 0.72x .. 8.44x |

**25 of 27 masked cells improve beyond the control's best case (1.10x).** That
threshold is a deliberately conservative empirical bound, not a confidence
interval — with 27 control samples the observed maximum overstates the true
noise ceiling. The separation here (median 1.26x vs 1.00x, with individual
cells at 8x) is far too large for a more formal test to change the conclusion.

Full matrix, all 54 cells, `native ÷ ORT` (lower is better, 1.00 = parity):

| model | phase | mask | threads | base / ORT | new / ORT | new vs base |
|---|---|---|---:|---:|---:|---:|
| llama-3.1-8b | decode q1/kv1024 | mask | 1 | 1.127 | 1.009 | **1.12x** |
| llama-3.1-8b | decode q1/kv1024 | mask | 8 | 6.355 | 5.646 | **1.13x** |
| llama-3.1-8b | decode q1/kv1024 | mask | 16 | 9.343 | 8.357 | **1.12x** |
| llama-3.1-8b | decode q1/kv1024 | nomask | 1 | 0.950 | 0.971 | **0.98x** |
| llama-3.1-8b | decode q1/kv1024 | nomask | 8 | 6.337 | 6.221 | **1.02x** |
| llama-3.1-8b | decode q1/kv1024 | nomask | 16 | 12.578 | 12.698 | **0.99x** |
| llama-3.1-8b | decode q1/kv4096 | mask | 1 | 1.073 | 0.932 | **1.15x** |
| llama-3.1-8b | decode q1/kv4096 | mask | 8 | 5.236 | 4.398 | **1.19x** |
| llama-3.1-8b | decode q1/kv4096 | mask | 16 | 5.513 | 5.006 | **1.10x** |
| llama-3.1-8b | decode q1/kv4096 | nomask | 1 | 0.932 | 0.932 | **1.00x** |
| llama-3.1-8b | decode q1/kv4096 | nomask | 8 | 4.465 | 4.477 | **1.00x** |
| llama-3.1-8b | decode q1/kv4096 | nomask | 16 | 5.090 | 5.242 | **0.97x** |
| llama-3.1-8b | prefill q512/kv512 | mask | 1 | 3.171 | 1.069 | **2.97x** |
| llama-3.1-8b | prefill q512/kv512 | mask | 8 | 17.382 | 3.055 | **5.69x** |
| llama-3.1-8b | prefill q512/kv512 | mask | 16 | 17.126 | 3.279 | **5.22x** |
| llama-3.1-8b | prefill q512/kv512 | nomask | 1 | 1.117 | 1.104 | **1.01x** |
| llama-3.1-8b | prefill q512/kv512 | nomask | 8 | 3.028 | 3.091 | **0.98x** |
| llama-3.1-8b | prefill q512/kv512 | nomask | 16 | 3.548 | 3.869 | **0.92x** |
| qwen2.5-0.5b | decode q1/kv1024 | mask | 1 | 1.400 | 0.889 | **1.57x** |
| qwen2.5-0.5b | decode q1/kv1024 | mask | 8 | 9.500 | 7.517 | **1.26x** |
| qwen2.5-0.5b | decode q1/kv1024 | mask | 16 | 5.655 | 7.872 | **0.72x** |
| qwen2.5-0.5b | decode q1/kv1024 | nomask | 1 | 0.912 | 1.156 | **0.79x** |
| qwen2.5-0.5b | decode q1/kv1024 | nomask | 8 | 8.771 | 9.391 | **0.93x** |
| qwen2.5-0.5b | decode q1/kv1024 | nomask | 16 | 9.291 | 9.737 | **0.95x** |
| qwen2.5-0.5b | decode q1/kv4096 | mask | 1 | 1.348 | 0.929 | **1.45x** |
| qwen2.5-0.5b | decode q1/kv4096 | mask | 8 | 6.884 | 5.497 | **1.25x** |
| qwen2.5-0.5b | decode q1/kv4096 | mask | 16 | 7.924 | 6.692 | **1.18x** |
| qwen2.5-0.5b | decode q1/kv4096 | nomask | 1 | 0.937 | 0.927 | **1.01x** |
| qwen2.5-0.5b | decode q1/kv4096 | nomask | 8 | 5.924 | 5.580 | **1.06x** |
| qwen2.5-0.5b | decode q1/kv4096 | nomask | 16 | 8.476 | 8.115 | **1.04x** |
| qwen2.5-0.5b | prefill q512/kv512 | mask | 1 | 5.033 | 1.031 | **4.88x** |
| qwen2.5-0.5b | prefill q512/kv512 | mask | 8 | 20.182 | 2.390 | **8.44x** |
| qwen2.5-0.5b | prefill q512/kv512 | mask | 16 | 22.783 | 2.931 | **7.77x** |
| qwen2.5-0.5b | prefill q512/kv512 | nomask | 1 | 1.097 | 1.096 | **1.00x** |
| qwen2.5-0.5b | prefill q512/kv512 | nomask | 8 | 3.459 | 3.151 | **1.10x** |
| qwen2.5-0.5b | prefill q512/kv512 | nomask | 16 | 3.638 | 4.792 | **0.76x** |
| qwen3-0.6b | decode q1/kv1024 | mask | 1 | 1.107 | 0.952 | **1.16x** |
| qwen3-0.6b | decode q1/kv1024 | mask | 8 | 19.460 | 15.039 | **1.29x** |
| qwen3-0.6b | decode q1/kv1024 | mask | 16 | 10.791 | 9.691 | **1.11x** |
| qwen3-0.6b | decode q1/kv1024 | nomask | 1 | 0.959 | 0.960 | **1.00x** |
| qwen3-0.6b | decode q1/kv1024 | nomask | 8 | 7.716 | 9.431 | **0.82x** |
| qwen3-0.6b | decode q1/kv1024 | nomask | 16 | 11.438 | 10.804 | **1.06x** |
| qwen3-0.6b | decode q1/kv4096 | mask | 1 | 1.080 | 0.938 | **1.15x** |
| qwen3-0.6b | decode q1/kv4096 | mask | 8 | 4.972 | 3.529 | **1.41x** |
| qwen3-0.6b | decode q1/kv4096 | mask | 16 | 4.459 | 4.282 | **1.04x** |
| qwen3-0.6b | decode q1/kv4096 | nomask | 1 | 0.929 | 0.937 | **0.99x** |
| qwen3-0.6b | decode q1/kv4096 | nomask | 8 | 4.648 | 4.322 | **1.08x** |
| qwen3-0.6b | decode q1/kv4096 | nomask | 16 | 4.307 | 4.282 | **1.01x** |
| qwen3-0.6b | prefill q512/kv512 | mask | 1 | 3.173 | 1.018 | **3.12x** |
| qwen3-0.6b | prefill q512/kv512 | mask | 8 | 16.355 | 2.670 | **6.13x** |
| qwen3-0.6b | prefill q512/kv512 | mask | 16 | 17.325 | 2.951 | **5.87x** |
| qwen3-0.6b | prefill q512/kv512 | nomask | 1 | 1.063 | 1.076 | **0.99x** |
| qwen3-0.6b | prefill q512/kv512 | nomask | 8 | 2.929 | 2.840 | **1.03x** |
| qwen3-0.6b | prefill q512/kv512 | nomask | 16 | 2.876 | 3.149 | **0.91x** |
### Second answer: still decline it — WITHDRAWN, see §23

The fix is large — up to **8.44x** — and masked single-thread decode now beats
ORT outright (0.889–1.009x, was 1.073–1.400x). It is not enough. **44 of 54
cells are still slower than ORT**, 41 of them by ≥5%, and only 7 are ≥5% faster
(base: 5). The worst remaining cells are **15.0x at 8 threads** and **12.7x at
16 threads**, because the remaining time is in the QK and PV GEMMs and their
thread scaling, not in the scale/mask/softmax stage that #1094 fixed.

So the honest answer to requirement 4, *at the time*, was the one it
anticipated: the fused region did not clear the ≥5% bar. The conclusion drawn
from that — "decline it to ORT" — **is withdrawn (§23)**; the measurement is
not. What the data actually establishes is that **the fused region is 3–15x
short and that is the work**: Keeping `Softmax` claimed purely to preserve a fusion
that then loses is not *justified* by this data — but it is now *required* by
the rule in §23, so the only acceptable resolution is to close the gap.

The blocker is structural rather than a matter of will: `claim_preference(op,
opset, shapes, input_dtypes)` in `assignment_policy.rs` is **per-node and has no
graph context**, so "decline this whole region" is not expressible in it. Only
`Softmax` itself can be deferred, which risks fragmenting the region — claim QK,
defer Softmax, claim PV — and that fragmentation cost is *unmeasured*. Deferring
on that basis would be trading a measured 3–15x loss for an unmeasured one.

For the **native `InferenceSession`**, which has no ORT to decline to, #1094 is
strictly the best available implementation and the right outcome regardless.

## 16. Allocation-cache thresholds, calibrated rather than assumed (#1088)

`large_alloc_cache.rs` carried `MIN_CACHED_BYTES = 256 KiB`, justified by a
comment claiming glibc's dynamic `mmap` threshold adjustment *"does not apply to
a request that is freed at the same size it was taken."* That is backwards.
glibc's `_int_free`, on releasing an mmapped chunk, does exactly the opposite:

```c
if (!mp_.no_dyn_threshold
    && chunksize_nomask(p) > mp_.mmap_threshold
    && chunksize_nomask(p) <= DEFAULT_MMAP_THRESHOLD_MAX) {
    mp_.mmap_threshold = chunksize(p);
    mp_.trim_threshold = 2 * mp_.mmap_threshold;
}
```

`DEFAULT_MMAP_THRESHOLD_MAX` is `4 * 1024 * 1024 * sizeof(long)` = **32 MiB** on
64-bit. A stably-sized block *below* that is mmapped once and arena-served
thereafter, so a cache on top of it adds only a mutex; above it the adaptation
stops permanently and the cache is worth a great deal.

A standalone C probe timing alloc + first-touch-every-page + free across
256 KiB → 256 MiB found exactly that cliff: **~8 ns/page below 32 MiB, ~200
ns/page above it**.

Measured, cache ÷ system allocator (two independent sweeps):

| size | system | cached | sweep 1 | sweep 2 | effect |
|---|---:|---:|---:|---:|---|
| 256 KiB | 401.0 ns | 463.6 ns | 1.156 | 1.032 | up to **16% slower** |
| 1 MiB | 1.490 µs | 1.520 µs | 1.020 | 1.011 | neutral–2% slower |
| 4 MiB | 7.547 µs | 7.453 µs | 0.987 | 1.016 | neutral |
| 16 MiB | 29.69 µs | 30.10 µs | 1.014 | 1.000 | neutral |
| 32 MiB | 2.736 ms | 60.39 µs | 0.022 | 0.022 | **45.3x faster** |
| 64 MiB | 3.951 ms | 122.2 µs | 0.031 | 0.031 | **32.3x faster** |
| 192 MiB | 12.71 ms | 389.3 µs | 0.031 | 0.029 | **32.6x faster** |

Interleaved (4 live sizes): 1 MiB 0.997/1.013, 16 MiB 1.008/1.009, 64 MiB
0.033/0.034. Eight threads: 256 KiB 0.978/1.022, 4 MiB 0.982/1.013, 64 MiB
0.034/0.037.

The old floor was therefore ~128x too low, and the cache was doing net harm
across most of the range it was enabled for. This does not undercut #1075, whose
motivating case was a 180 MB output well above the corrected floor.

Rather than hardcode 32 MiB — which is a glibc constant, not a universal one —
the floor is now **calibrated at runtime** by a bounded geometric-ladder probe
(50 ms hard budget checked per sample, memoised per process), falling back to a
per-platform constant if no cliff is found. This is also correct under
`mallopt`/`MALLOC_MMAP_THRESHOLD_`, which disable the adaptation entirely:
the probe observes the effect rather than assuming the mechanism.

The budget is likewise topology-aware: `min(2 GiB, allowance/8)`, where the
allowance walks `/proc/self/cgroup` root→leaf taking the smallest concrete
limit before falling back to `MemAvailable`. Reading `/sys/fs/cgroup/memory.max`
alone — the previous approach — finds the limit at the *mount root*, which is
the process's own cgroup only under a private cgroup namespace; it is wrong
under a systemd slice with `MemoryMax=`, under `cgroupns=host`, and for anything
nested, and in each of those cases it hands the cache a budget the cgroup will
never allow.

## 17. Phase 4 limitations

* §15's remaining gap — up to 15.0x at 8 threads and 12.7x at 16 — is in the
  QK/PV GEMMs. #1094 does not touch them.
* §15 recommends declining the fused region in the Plugin EP but **does not
  implement it**, because `claim_preference` is per-node and the fragmentation
  cost of deferring `Softmax` alone is unmeasured. Implementing the decline
  without that measurement would substitute one unquantified loss for a
  quantified one.
* One masked cell in §15 regresses (qwen2.5-0.5b decode kv1024, 16 threads:
  5.66x → 7.87x, i.e. 0.72x). It sits inside the control band (min 0.76x) on a
  3-trial median of a sub-millisecond kernel, so it is read as noise — but it is
  reported rather than dropped.
* §16's cliff is measured on glibc/x86-64 only. The musl, macOS and Windows
  fallback floors are reasoned from allocator design, **not measured**. The
  runtime probe is what makes that acceptable: on any platform where the reasoning
  is wrong, calibration overrides the constant.
* Phase 4 requirement 3 (fused transpose-with-consumer) remains **open**; see §19 and §20.
  The §10 assignment matrix is unchanged by Phase 4.

## 18. Grouped MoE expert GEMMs (#1099)

### 18.1 Why MoE lost, and it was not arithmetic

At `--native-threads 1` the CPU MoE operator was **at parity** with a real ORT CPU
session on all nine synthetic production-shaped graphs: ratios 1.01-1.05. The entire
gap opened up as threads were added. Native time for a 512-token Mixtral-shaped
forward was 350.8 / 148.6 / 113.4 ms at 1 / 8 / 16 threads in the single-arm baseline run
(the paired run in §18.3 measured 353.2 / 148.7 / 112.2 ms for the same cell - the same
numbers inside session-to-session drift). Inverting Amdahl's law on those three points
puts the **serial fraction at ~34%**.

The serial 34% was everything except the GEMMs:

* the per-expert gather, a row-at-a-time `extend_from_slice` into a fresh `Vec`;
* the activation, which for gated SwiGLU built **three** additional `Vec`s element by
  element with `push` - roughly 56 MiB of allocation and three passes for a single
  512-token Mixtral forward;
* the weighted scatter, a scalar `+=` loop.

MLAS was already threading the expert GEMMs well. This is the same lesson as §15: the
operator's cost was in the transforms around the arithmetic, not the arithmetic.

### 18.2 What changed

`RoutingPlan` flattens top-k routing into a slot ordering - every row routed to the
lowest active expert first, then the next - which makes each expert's GEMM operand a
contiguous window and every stage between routing and the output write a flat row-wise
parallel map over one buffer. Experts whose groups hold the same number of rows are
issued to MLAS as **one batched GEMM** through a new `mlas_sys::sgemm_batch` wrapper
over `MlasGemmBatch`; that call issues a single `MlasTrySimpleParallel` fan-out of
`ThreadsPerGemm * BatchSize` work items, so batching collapses N dispatches and N
barriers into one. Both biases fold into a pass that already reads the result (FC1's
into the activation, FC2's into the scatter), removing two full read-modify-write
passes over the largest intermediates.

Because slots are laid out by ascending expert, each output row still sums its
contributions in ascending expert order, so the result is **bit-identical** to the
per-expert loop rather than merely close - the tests assert on raw `Vec<f32>` equality.

### 18.3 Full matrix, 27 cells, real ORT CPU session

> **Correction (§31).** These are **MLAS-arm** ratios. The claim below that every `t=1`
> cell sits at 1.006-1.052 is not true of the build that ships: measured on the
> production arm, the same `t=1` cells were **39-72x** ORT, because the shipping path
> materialized a transposed copy of each expert's weights on every call. Fixed in #1241.
> Current production-native numbers are in §31.5.

`ratio = native p50 / ORT p50`, **lower is better**. Interleaved arms, same host,
same session options, same thread count on both sides, 5 warmups, 5 trials x 15 runs.
Synthetic weights from `scripts/ort_ab/gen_moe.py` (public architecture dimensions,
reduced expert counts, **no weights downloaded**).

| model (h / inter / experts / top-k) | tokens | threads | ORT ratio before | ORT ratio after | native before | native after | native delta | driver |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| mixtral 1024/3584/8/top-2 | 1 | 1 | 1.033 [1.009-1.072] | **1.030** [1.005-1.033] | 3.2 ms | 3.2 ms | +1.1% | per-expert |
| mixtral 1024/3584/8/top-2 | 1 | 8 | 0.683 [0.672-0.720] | **0.695** [0.671-0.704] | 2.3 ms | 2.3 ms | +0.3% | per-expert |
| mixtral 1024/3584/8/top-2 | 1 | 16 | 0.658 [0.641-0.678] | **0.668** [0.666-0.677] | 2.2 ms | 2.2 ms | +1.3% | per-expert |
| mixtral 1024/3584/8/top-2 | 32 | 1 | 1.023 [1.021-1.029] | **1.022** [1.018-1.025] | 52.1 ms | 52.3 ms | +0.2% | per-expert |
| mixtral 1024/3584/8/top-2 | 32 | 8 | 2.074 [2.046-2.176] | **1.927** [1.585-2.099] | 17.8 ms | 18.0 ms | +1.3% | per-expert |
| mixtral 1024/3584/8/top-2 | 32 | 16 | 1.713 [1.373-2.743] | **1.410** [1.230-1.714] | 16.4 ms | 14.8 ms | -9.5% | per-expert |
| mixtral 1024/3584/8/top-2 | 512 | 1 | 1.052 [1.044-1.055] | **1.049** [1.038-1.049] | 353.2 ms | 351.4 ms | -0.5% | per-expert |
| mixtral 1024/3584/8/top-2 | 512 | 8 | 2.536 [2.405-2.544] | **1.724** [1.167-1.729] | 148.7 ms | 101.1 ms | -32.0% | grouped |
| mixtral 1024/3584/8/top-2 | 512 | 16 | 1.785 [1.727-1.889] | **1.004** [0.924-1.042] | 112.2 ms | 60.3 ms | -46.3% | grouped |
| phi35moe 2048/6400/4/top-2 | 1 | 1 | 1.006 [1.000-1.006] | **0.999** [0.970-1.011] | 10.5 ms | 10.7 ms | +1.2% | per-expert |
| phi35moe 2048/6400/4/top-2 | 1 | 8 | 0.786 [0.781-0.788] | **0.788** [0.779-0.790] | 8.8 ms | 8.9 ms | +0.7% | per-expert |
| phi35moe 2048/6400/4/top-2 | 1 | 16 | 0.784 [0.782-1.007] | **0.793** [0.783-0.807] | 8.6 ms | 8.6 ms | +0.1% | per-expert |
| phi35moe 2048/6400/4/top-2 | 32 | 1 | 1.014 [1.003-1.022] | **1.016** [1.007-1.024] | 119.9 ms | 120.0 ms | +0.1% | per-expert |
| phi35moe 2048/6400/4/top-2 | 32 | 8 | 0.997 [0.973-1.015] | **0.890** [0.845-0.938] | 37.2 ms | 33.3 ms | -10.7% | grouped |
| phi35moe 2048/6400/4/top-2 | 32 | 16 | 0.715 [0.706-0.728] | **0.671** [0.652-0.723] | 26.9 ms | 25.1 ms | -6.8% | grouped |
| phi35moe 2048/6400/4/top-2 | 512 | 1 | 1.046 [1.039-1.198] | **1.042** [1.039-1.046] | 1118.0 ms | 1107.4 ms | -0.9% | per-expert |
| phi35moe 2048/6400/4/top-2 | 512 | 8 | 1.065 [1.061-1.077] | **0.766** [0.760-0.775] | 409.2 ms | 297.6 ms | -27.3% | grouped |
| phi35moe 2048/6400/4/top-2 | 512 | 16 | 0.707 [0.642-0.724] | **0.440** [0.427-0.442] | 280.3 ms | 170.3 ms | -39.3% | grouped |
| qwen3moe 2048/768/16/top-8 | 1 | 1 | 1.012 [1.011-1.034] | **1.015** [1.004-1.027] | 5.4 ms | 5.4 ms | -0.1% | per-expert |
| qwen3moe 2048/768/16/top-8 | 1 | 8 | 1.685 [1.504-1.756] | **1.720** [1.305-1.815] | 3.9 ms | 4.2 ms | +5.9% | per-expert |
| qwen3moe 2048/768/16/top-8 | 1 | 16 | 1.304 [1.171-3.111] | **1.342** [1.157-1.368] | 4.2 ms | 4.1 ms | -1.1% | per-expert |
| qwen3moe 2048/768/16/top-8 | 32 | 1 | 1.023 [1.021-1.025] | **1.015** [1.004-1.018] | 59.9 ms | 59.3 ms | -1.0% | per-expert |
| qwen3moe 2048/768/16/top-8 | 32 | 8 | 1.944 [1.317-2.343] | **1.926** [1.910-1.966] | 20.7 ms | 20.0 ms | -3.5% | per-expert |
| qwen3moe 2048/768/16/top-8 | 32 | 16 | 1.452 [1.185-1.598] | **1.418** [1.142-1.543] | 17.3 ms | 16.7 ms | -3.6% | per-expert |
| qwen3moe 2048/768/16/top-8 | 512 | 1 | 1.035 [1.023-1.042] | **1.025** [1.022-1.029] | 532.4 ms | 527.5 ms | -0.9% | per-expert |
| qwen3moe 2048/768/16/top-8 | 512 | 8 | 2.230 [1.683-2.244] | **1.644** [1.335-1.675] | 211.3 ms | 157.8 ms | -25.3% | grouped |
| qwen3moe 2048/768/16/top-8 | 512 | 16 | 1.401 [1.086-1.520] | **0.729** [0.710-1.022] | 151.5 ms | 106.7 ms | -29.6% | grouped |

### 18.4 What this bought, and what it did not

Where the gate selects the grouped driver, native time drops **25-46%**: mixtral 512
tok/16 thr 1.785 -> 1.004, phi35moe 512/16 0.707 -> 0.440, qwen3moe 512/16 1.401 ->
0.729, and the three 8-thread 512-token cells fall 25-32%. Where the gate keeps the
per-expert loop the two arms are algorithmically equivalent and produce bit-identical
output - not literally the same code, since the after-arm still builds the routing plan -
and 16 of those 19 cells move by <=1.3%, with the three outliers inside the baseline
arm's own trial dispersion.

**We still lose to ORT** on many-expert, narrow-intermediate shapes (Qwen3-MoE: 16
experts, `inter` 768, top-8) at 1.34-1.93, on mixtral at 32 tokens (1.41-1.93), and on
mixtral/qwen3 512 tokens at 8 threads (1.64-1.72). There each expert receives only a
handful of rows and MLAS's per-GEMM threading is the limit; neither driver helps.
Closing that needs a different decomposition - splitting one expert's GEMM across
threads by `N` instead of splitting experts across threads - which was not attempted.
**No general MoE superiority over ORT is claimed.**

### 18.5 The gate, and why it is per-group work

`use_grouped_driver` compares `(slots / groups) * hidden * (fc1_size + inter)` - the
average GEMM work in *one* expert group - against `MOE_GROUPED_MIN_WORK` (3.0e8).
Per-group rather than total, because the grouped driver's win is parallelising the
gather/activation/scatter, which only matters once MLAS already saturates the machine
inside each expert's own GEMM. Measured separation was 7.6e7 (losing) to 6.3e8
(winning); the floor sits near the geometric mean of that 8x band.

This is a **one-host calibration** (AMD EPYC 9V74, 16C/32T, 32 MiB L3 shared per 16
CPUs, AVX2/FMA, no AVX-512), overridable with `ONNX_GENAI_MOE_GROUPED_MIN_WORK`. The
falsifier `the_driver_gate_matches_the_measured_crossover` pins all nine calibration
points, so changing the constant in a way that would flip a measured cell fails CI.

Expert-weight prepacking (`MlasGemmPackB`) was considered and **rejected**: decode is
bandwidth-bound on reading the expert bank, packing does not reduce bytes read, and it
would double a 352 MiB weight bank.

### 18.6 The cost: peak transient memory

The grouped driver is not free. It materialises **whole-slot** intermediates at once -
`gathered` (slots x hidden), `fc1_out` (slots x fc1_size), the optional `fc3_out` and
`activated` (slots x inter each) and `expert_out` (slots x hidden) - where the per-expert
loop held roughly one group's worth at a time. For a 512-token Phi-3.5-MoE-shaped prefill
(hidden 2048, inter 6400, fc1_size 12800, top-2, 1024 slots) that is about **78 MiB** of
transients against about **21 MiB** before, and it grows linearly with token count.

This is a deliberate trade: the transients are what make every stage a flat parallel map.
It is also the second reason the gate matters - below the floor the per-expert loop keeps
both the old latency *and* the old peak. Anyone running many concurrent large-prefill
sessions on a memory-tight host should raise `ONNX_GENAI_MOE_GROUPED_MIN_WORK`. Tiling the
grouped driver by slot blocks would cap the peak and was not attempted here.

### 18.7 Correction: the gate statistic changed after measurement

The A/B matrix above was produced by a build whose gate used the **mean** rows per expert
group. Review pointed out that a collapsed router - one expert taking most of the rows -
still has one large GEMM to hide the elementwise stages behind, and the mean would hide
it, so the shipped gate uses the **largest** group instead. Recomputing all 27 cells under
both statistics selects the **same driver in every cell** (the benchmark routers are close
to balanced, so mean and max differ by well under the 8x calibration band), which is why
the matrix stands as measured. `the_gate_reads_the_largest_group_not_the_average` pins the
difference on a deliberately skewed router so the statistic cannot silently revert.

## 19. Phase 4 requirement 3: what is actually left in Transpose

Requirement 3 asked for the remaining Transpose materialisation to be eliminated by
fusing it into its consumer. Auditing the kernel at current `main` shows the framing
needs correcting, and the correction matters for anyone picking this up:

* `TransposeKernel::view_outputs` already returns a **zero-copy strided view**, so in
  any graph where the consumer accepts strides there is no materialisation to remove.
  `execute` runs only where materialisation is *mandatory* - principally when the
  transpose result is a graph output.
* That mandatory path is no longer naive. Phase 3 gave it permutation collapsing
  (merging output axes that stay adjacent in the input and dropping unit axes), a
  contiguous-block `block_move`, a cache-blocked 2-D transpose with a 64-element tile,
  a batched 2-D variant for the `[..., S, H] <-> [..., H, S]` shapes every attention
  layout collapses to, and a `rayon` fan-out above 256 KiB.

So the residual gap on that path is **memory-bandwidth-bound**, and the only remaining
structural lever is on the *consumer* side: `matmul.rs` declares
`supports_strided_input = true` but then calls `to_dense_f32_widen`, materialising a
contiguous copy internally. For the attention layouts this is avoidable in principle -
a `BSNH -> BNSH` view has `head_dim` contiguous innermost and a uniform `num_heads *
head_dim` row pitch, which is exactly a GEMM **leading dimension**, and each `(batch,
head)` plane is one `MlasGemmBatch` item. Wiring that up would let QK and PV consume
the transposed view with zero copies.

That work is **designed but not landed**: it touches every dispatch path in a
3,800-line kernel with prepacking, half-precision and Accelerate branches, and it needs
its own paired A/B before any claim. It is recorded here rather than left implied.

One correction to how this section was originally read: the consumer-side fusion above is
worth doing, but it is **not** the lever for the Phase 3 single-node Transpose ratios.
§20 measures those graphs and finds the Transpose node costs 0.002 ms of a 0.776 ms run -
the gap is session per-run work at the graph boundary, not the kernel and not the
consumer. Read §20 before acting on this section.

## 20. Correction: the Transpose "gap" was never the Transpose kernel

The Phase 3 headline - "standalone Transpose is 2.5-9.5x behind ORT at 8/16 threads" -
was never attributed to a component. §19 reasoned about it from the code without
measuring it. Phase 4 measured it, with the per-op profiler running on the same
single-node benchmark graphs that produced those ratios. The headline was attributing to
the kernel a cost the kernel does not pay.

### 20.1 The measurement

`ONNX_GENAI_PROFILE_OPS=1 bench_generic --native-only --native-threads 16`, 30 runs,
against the merged `main` (`86a2d6eb1`, the commit that merged #1099; this branch is
based one commit later on `9b7a45860`/#1104, an int4-decode change that touches none of
these paths). Per-op figures below are single-run means over the 30 runs, not medians
across trials, and the `Concat` values in particular move by tens of percent between
runs on this contended host - they are used only to separate "negligible" from
"dominant", which is a margin they support by two orders of magnitude. The absolute
end-to-end times in this table come from that instrumented single session and are **not**
the §20.2 matrix numbers (which are ORT-interleaved medians over 5 trials x 15 runs); the
two differ by 15-22% on the same graph and thread count, and only the *ratio* of kernel
to run within this table is being used:

| graph | output bytes | total node execution | of which Transpose | end-to-end native |
|---|--:|--:|--:|--:|
| `tr_llama3_s512` (1x512x32x128, `perm=[0,2,1,3]`) | 8 MiB | 0.006 ms | **0.002 ms** | 0.776 ms |
| `tr_bert_b8_s128` (8x128x12x64) | 3 MiB | 0.007 ms | **0.002 ms** | 0.589 ms |
| `tr_whisper_s1500` (1x1500x20x64, `perm=[0,2,1,3]`) | 7.3 MiB | 0.007 ms | **0.002 ms** | 0.838 ms |
| `kvcat_llama3_p4095` (2 x `Concat`) | 2 x 16 MiB | 1.637 ms | n/a (`Concat` 1.635 ms) | 5.121 ms |
| `kvcat_llama3_p8191` (2 x `Concat`) | 2 x 32 MiB | 3.574 ms | n/a (`Concat` 3.572 ms) | 7.458 ms |

For every Transpose graph the **kernel accounts for under 0.4% of the run**. That is
`view_outputs` doing its job: `dispatch.rs` takes the view path before `execute` is ever
called, the operator returns a permuted-stride alias over the same buffer, and it moves
no bytes at all. An 8 MiB `permute_bytes` would cost 0.1-0.2 ms at any plausible
bandwidth; 0.002 ms is only consistent with installing a view, so this is not the
profiler mis-attributing a hidden copy.

Be precise about the other 99.6%. `ONNX_GENAI_PROFILE_OPS` times **only per-node
execution inside the eager plan**; everything outside that - input binding, the
boundary materialisation of a strided graph output, allocation, and fixed per-run
session work - lands in the residual undifferentiated. So what is proven is that **the
kernel is not the cost**, and that the cost is session per-run work. It is *not* proven
that the residual is all boundary copying: the end-to-end times do not scale with output
bytes the way a copy-dominated residual would (3 MiB -> 0.589 ms but 8 MiB -> 0.776 ms),
which says a fixed per-run component is folded in too. Decomposing the residual needs a
boundary-only instrument that does not exist yet.

Either way the Phase 3 headline was not measuring the Transpose kernel, and no amount of
work on `permute_bytes` could have moved those numbers - that function is not on the
path. Requirement 3's "avoid the operation, do not merely parallelise it" is satisfied
**at the transpose node**: in-graph, the node itself moves zero bytes. It is *not*
satisfied end-to-end, because the consumer can still force materialisation - `matmul.rs`
declares `supports_strided_input = true` and then calls `to_dense_f32_widen` (§19). The
zero-copy consumer is designed and unlanded; until it lands, "the copy never happens" is
true of the transpose and false of the pair.

### 20.2 The full transform matrix as it stands today

Single-arm, merged `main`, same host, interleaved with a real ORT CPU session, 5 warmups,
5 trials x 15 runs. Each cell is the **median across trials of the per-trial
`native p50 / ORT p50`**, lower is better; brackets give the min-max across the 5 trials.
Nothing omitted.

**These are pre-§22 numbers and the transform rows are now stale.** §22
parallelises the input binding that §21 showed dominates these graphs, which
moves the `tr_*` rows substantially. §22.4 gives the after-state, but in a
different harness (separate processes) whose ORT denominators are not comparable
to this table's, so the two must not be divided into each other. This table is
kept as the measured before-state of the harness it was taken in.

| graph | t=1 | t=8 | t=16 | native p50 @16 |
|---|--:|--:|--:|--:|
| `tr_bert_b8_s128` | 1.090 [1.056-1.133] | 3.188 [2.613-3.440] | 4.600 [3.846-5.314] | 0.558 ms |
| `tr_llama3_s512` | 1.107 [1.079-1.159] | 5.180 [4.974-5.947] | 8.694 [8.170-9.771] | 0.922 ms |
| `tr_whisper_s1500` | 1.451 [1.234-1.455] | 5.762 [5.041-6.279] | 9.279 [8.119-10.392] | 1.020 ms |
| `kvcat_llama3_p1023` | 1.805 [1.696-1.870] | 6.351 [5.675-7.078] | 6.109 [5.766-7.124] | 0.698 ms |
| `kvcat_llama3_p2047` | 1.831 [1.743-1.910] | 5.849 [5.233-6.887] | 8.161 [6.727-8.518] | 1.623 ms |
| `kvcat_llama3_p4095` | 2.765 [2.730-2.771] | 7.704 [7.528-8.268] | 9.137 [8.739-9.666] | 5.373 ms |
| `kvcat_llama3_p8191` | 1.961 [1.938-1.973] | 5.084 [5.009-5.502] | 5.170 [5.089-5.950] | 8.017 ms |
| `kvcat_llama3_b8_p2047` | 1.888 [1.882-1.903] | 4.450 [4.254-4.591] | 4.480 [4.398-4.587] | 14.947 ms |

Two things are visible in every row and both matter more than any kernel:

1. **At one thread we are within 1.09-2.77x**, and for the three Transpose graphs within
   1.09-1.45x. The gap opens only as threads are added.
2. **Our native time does not scale with threads, and in several graphs regresses.**
   `tr_llama3_s512` goes 0.956 / 0.660 / 0.922 ms at 1 / 8 / 16 - a 31% gain at 8 that is
   given back at 16; `tr_bert_b8_s128` goes 0.208 -> 0.558 ms and `tr_whisper_s1500`
   0.838 -> 1.020 ms from 1 to 16; `kvcat_llama3_b8_p2047` is flat at 14.870 / 14.996 /
   14.947 ms. ORT's measured medians for the same graphs fall about 8x
   (`tr_llama3_s512`: 0.870 -> 0.127 -> 0.109 ms). ORT is not winning by having a better
   transpose; it is winning by threading the part we run on one core.

Combining with §20.1: for the Transpose graphs the part we run on one core is session
per-run work, not the kernel. For the `Concat` graphs the kernel is a real **32-48%** of
the run (1.635 ms of 5.121 ms; 3.572 ms of 7.458 ms) - those two figures are the only
ones measured, so no wider range is claimed - and the remainder is again session per-run
work.

### 20.3 What this means for the remaining work

* **Transpose: closed as a kernel problem.** In-graph the transpose node itself already
  moves zero bytes. The single-node benchmark cannot show that, which is a limitation of
  the harness. It does *not* follow that attention is copy-free: §19's proposal to teach
  `matmul.rs` layout-aware GEMM addresses the consumer that still forces materialisation,
  which is a real cost these graphs simply do not exercise. That work remains open.
* **The real remaining lever is the session's per-run path** - input binding, boundary
  materialisation and fixed setup - which does not thread and is charged to every small
  graph we benchmark. It inflates every single-node ratio in this document. Attacking it
  would move all eight rows above; no operator change will. Which part of it dominates is
  not yet measured (§20.1), so that is the next instrument to build, before the next
  optimisation. **§21 builds that instrument and answers this**: for the Transpose graphs
  the per-run path is 60-74% input binding and 25-38% boundary materialisation. **§22
  then attacks the input binding**, taking the two Transpose graphs whose inputs clear
  its threshold from 6.9-14.5x ORT to 3.0-5.8x ORT. Boundary materialisation is
  untouched and remains open.
* **`Concat` remains a genuine kernel cost** and remains bounded by §13: `ExternalValue`
  carries no strides, so a `Concat`-shaped KV cache cannot append in place. The Phase 4
  in-place append (#1083) applies to the GQA-shaped cache, not this one.

These are recorded as measured facts. No fix is claimed for either.

## 21. Where a run actually goes: the boundary instrument

§20 could show that the Transpose *kernel* is not the cost, but not what is,
because `ONNX_GENAI_PROFILE_OPS` times per-node execution and nothing around it.
That instrument now exists. The executor already had an env-gated phase
profiler (`NXRT_EXEC_PHASE_PROFILE`); this adds a `bench_generic --phase-profile`
flag that turns it on programmatically, resets it after warmups so the totals
cover exactly `--runs` measured runs, and prints it - plus one new phase,
`run_scoped.bind_inputs`, around the loop that uploads graph inputs.

### 21.1 The decomposition

`bench_generic --native-only --native-threads 16 --runs 30 --warmups 10
--phase-profile`, microseconds per run, on the host described in §2. `plan_eager`
is the whole node-execution plan - everything `ONNX_GENAI_PROFILE_OPS` sees.
`bind_inputs` is a subset of `setup_total`. The column names are shortened from
the emitted span names: the `[nxrt-phase]` dump prints
`run_scoped.setup_total.top`, `run_scoped.bind_inputs`,
`run_scoped.collect_outputs.top` and `run_scoped.plan_eager.top`.

**`in` and `out` are host bytes actually moved**, not tensor size: `in` is what
`bind_inputs` copies, `out` is what `collect_outputs` materialises. The `Concat`
graphs show `out = 0` because their outputs are already owned buffers and need no
materialisation, even though §20.1 lists the same graphs with 2 x 16 MiB and
2 x 32 MiB of output *tensor*. `in` is the total over all of a graph's inputs;
the `Concat` graphs have two large past-KV tensors each, so their per-tensor
figure is half the number shown.

| graph | in | out | run | `setup_total` | of which `bind_inputs` | `collect_outputs` | `plan_eager` |
|---|--:|--:|--:|--:|--:|--:|--:|
| `tr_llama3_s512` | 8 MiB | 8 MiB | 808 | 605.7 | **601.3** | 208.0 | 7.8 |
| `tr_bert_b8_s128` | 3 MiB | 3 MiB | 570 | 346.8 | **341.1** | 218.8 | 7.6 |
| `tr_whisper_s1500` | 7.3 MiB | 7.3 MiB | 866 | 647.6 | **642.2** | 217.4 | 8.1 |
| `kvcat_llama3_p4095` | 32 MiB | 0 | 4151 | 1898.6 | **1678.5** | 2.2 | 2354.9 |
| `kvcat_llama3_p8191` | 64 MiB | 0 | 7100 | 3480.9 | **3474.3** | 1.5 | 3610.7 |

Read the first row: of a 0.808 ms run, **601 us is copying the 8 MiB input into
an EP-owned buffer**, 208 us is materialising the 8 MiB strided graph output,
and 7.8 us is every node in the graph. `bind_inputs` is 99.3% of `setup_total`
on that row and 99.8% on the last one, so "setup" is not shape work, plan
restoration or buffer sizing - those sum to about 5 us. It is one `memcpy`.
Binding plus collecting is 100.2% of the `tr_llama3_s512` run and 98.2% of
`tr_bert_b8_s128`; the excess over 100% is the phase profiler's own overhead
against a separately timed wall clock, and it is the honest error bar on these
rows.

Two caveats on this table. **These are means of one 30-run trial with no
dispersion reported** - the profiler accumulates sums, not distributions - so
§20.1's warning that `Concat` graphs move by tens of percent between runs still
applies here: `plan_eager` for `kvcat_llama3_p4095` is 2.355 ms against §20.1's
1.635 ms for the same graph and thread count, a 44% swing that is run-to-run
noise, not a change. And **`kvcat_llama3_p4095` is the one row where
`setup_total - bind_inputs` is not ~5 us**: it is 220 us, against 4.4-6.6 us on
every other row, including the strictly larger `p8191`. That residual is
unexplained and is left recorded rather than smoothed over.

The `Concat` rows say the same thing from the other side. Their outputs are
owned buffers, so `collect_outputs` is ~2 us and moves zero bytes; but they take
two large past-KV inputs, and binding those is 1.7-3.5 ms - between 40% and 49%
of the run, alongside a genuine 2.4-3.6 ms of `Concat`.

### 21.2 Why the copy exists

`Executor::prepare_run_buffers` ends with, for every graph input,
`self.ep.copy_from_host(tensor.as_bytes(), buf)`. The EP owns its buffers and
kernels write through them, so an input has to be inside one before the plan
runs. ORT does not pay this: `Value` can borrow the caller's allocation, and its
CPU EP reads the caller's memory directly.

Closing that gap properly means letting a `DeviceBuffer` borrow host memory for
the duration of a run, which is a lifetime and aliasing change to the EP ABI
(the buffer must not outlive the caller's tensor, and no kernel may write
through a borrowed input). That is not attempted here, and this section makes no
claim about how much it would be worth.

### 21.3 What this changes about earlier sections

* §20.1 said the residual was "session per-run work" and explicitly declined to
  split it further. It is now *partially* split: for the Transpose graphs it is
  **60-74% input binding, 25-38% output materialisation, ~1% nodes**. §20's
  conclusion is unchanged, but its caution is only half discharged.
  `collect_outputs` is 208.0 / 218.8 / 217.4 us for 8 / 3 / 7.3 MiB of output -
  essentially flat, not byte-proportional - so a fixed per-run cost (§6's per-run
  mmap/munmap is the obvious candidate) is still folded into it, and this
  instrument cannot separate that fixed cost from the copy inside it. Calling the
  whole 208 us "output materialisation" is a label, not a measurement.
* The single-node harness charges every graph two full tensor copies that a real
  transformer forward would pay once for the whole model. Every ratio in this
  document measured on a single-node graph is inflated by that, and - *asserted
  from the ORT C API, not measured here* - ours only: an ORT CPU `Value` can be
  constructed over the caller's buffer, so ORT has no equivalent of
  `bind_inputs`. This instrument does not time ORT, so that half of the statement
  is reasoning about an ABI rather than an observation. It is a property of the
  benchmark, not of the runtime, and it is the reason the §20.2 rows look so much
  worse than the §10 assignment evidence.
* The `Concat` rows are *not* explained away by it: 2.4-3.6 ms of real kernel
  time survives after the copies are accounted for. §13's boundary still binds.

## 22. Parallelising the input binding copy

§21 found that one single-threaded `memcpy` is 40-99% of `setup_total` on every
graph measured. The copy itself cannot be removed without changing the EP ABI
(§21.2), but on a 16-core host it does not have to run on one core.

`CpuExecutionProvider` now overrides `ExecutionProvider::copy_from_host`. Below a
threshold it is the previous `copy_from_slice`; above it the source and
destination are split into equal chunks and copied by up to 8 Rayon workers.

### 22.1 Calibrating the threshold: the micro-benchmark was wrong

The first calibration was a standalone loop copying one buffer 25 times. It said
the crossover was ~24 MiB and that an 8 MiB copy was **2.8x slower** in parallel,
with a 28.8 GB/s serial rate. Setting the threshold to 24 MiB produced a 9%
end-to-end win, which did not match the phase profile at all.

The loop was measuring an **L3-resident** copy. In a real session the graph input
was written once by the harness and evicted; the source is DRAM-cold. Sweeping
the threshold *in-session* over a 13-point input-size ladder gives a completely
different answer. `bind_inputs` microseconds, median of 3 interleaved
repetitions, 16-thread budget, one process per arm:

`input` is the **total** over a graph's inputs. The threshold is applied per
tensor, so for the `kvcat_*` rows - which take two large past-KV tensors each -
the size that actually decides the path is half the number shown.

| graph | input | serial | parallel | parallel vs serial |
|---|--:|--:|--:|---|
| `sm_decode_h32_kv1024` | 0.125 MiB | 4.8 | 108.5 | 22.6x worse |
| `sm_decode_h32_kv4096` | 0.5 MiB | 15.6 | 134.9 | 8.6x worse |
| `sm_decode_h32_kv8192` | 1 MiB | 31.5 | 130.4 | 4.1x worse |
| `rope_llama3_s1` | 2 MiB | 62.2 | 231.4 | 3.7x worse |
| `tr_bert_b8_s128` | 3 MiB | 372.6 | 120.9 | **3.1x better** |
| `rope_llama3_s128` | 4 MiB | 353.4 | 255.9 | 1.4x better |
| `sm_bert_b8_s128` | 6 MiB | 391.8 | 126.1 | 3.1x better |
| `tr_llama3_s512` | 8 MiB | 589.8 | 178.3 | 3.3x better |
| `rope_llama3_s512` | 10 MiB | 718.0 | 314.9 | 2.3x better |
| `kvcat_llama3_p2047` | 16 MiB | 698.8 | 645.6 | 1.08x better |
| `kvcat_llama3_p4095` | 32 MiB | 1714.9 | 1525.6 | 1.12x better |
| `kvcat_llama3_p8191` | 64 MiB | 3501.3 | 3032.9 | 1.15x better |
| `kvcat_llama3_b8_p2047` | 128 MiB | 7377.6 | 7188.5 | 1.03x better |

The cliff between 2 MiB and 3 MiB is **a cache-residency effect, not a size
effect**. Immediately either side of it the serial rate collapses from 33-34 GB/s
(2 MiB and below, an L3 rate) to 8-12 GB/s (3-4 MiB, a DRAM rate). That band
describes the cliff only: the 0.125 MiB row is 27.3 GB/s, and above 4 MiB the
serial rate climbs back to 14-24 GB/s as the fixed per-copy cost amortises. Byte
count is only a proxy for "is the source in L3", and it is the only proxy
available at the call site. Rayon fan-out costs about 110 us here - large, and it
reflects waking sleeping workers on a contended host.

The threshold is **4 MiB**: above every measured loss, at or below every measured
win, one step clear of the cliff rather than sitting on it.
`ONNX_GENAI_HOST_COPY_PARALLEL_MIN_BYTES` overrides it, read once per process
into a `OnceLock` - which is correct for this sweep because it runs one process
per arm. Tests move the threshold through a thread-local override instead of the
environment, so they cannot race each other.

### 22.2 The interleaved harness cannot resolve this effect

The standard `scripts/ort_ab/ab.py` methodology used everywhere else in this
document does not work here, and it is worth recording why rather than quietly
switching designs.

A 5-trial interleaved run showed large wins. A 9-trial run of the same cells
showed **the opposite signs**. The tell is the control cells - `t=1`, where
`host_copy_workers()` returns 1 and the parallel path is unreachable, and
`tr_bert_b8_s128` at 3 MiB, which is below the threshold. Both are provably
identical machine code in both arms, and both moved **+-8-13%**. That is the
noise floor of this harness on this host, and the effect on the small graphs is
inside it.

There is a second, structural reason: `bench_generic` builds a native *and* an
ORT session in one process, so ORT's intra-op pool is alive and competing for
cores exactly when the parallel copy fans out. Interleaving arms within such a
process measures the interaction, not the change.

### 22.3 Paired native-only A/B (7 interleaved repetitions)

Same binary, same process shape, one arm differing only by
`ONNX_GENAI_HOST_COPY_PARALLEL_MIN_BYTES`. 16-thread budget. Negative is faster.

| graph | per-tensor input | end-to-end | `bind_inputs` | per-rep ratio range |
|---|---|--:|--:|---|
| `tr_bert_b8_s128` (control, below threshold) | 3 MiB | +3.2% | -0.5% | 0.97-1.07 |
| `tr_llama3_s512` | 8 MiB | **-50.6%** | **-67.1%** | 0.33-0.57 |
| `tr_whisper_s1500` | 7.3 MiB | **-58.5%** | **-66.4%** | 0.36-0.56 |
| `kvcat_llama3_p1023` | 4 MiB x2 | +1.4% | -0.6% | 0.97-1.10 |
| `kvcat_llama3_p2047` | 8 MiB x2 | +3.2% | -3.3% | 0.93-1.10 |
| `kvcat_llama3_p8191` | 32 MiB x2 | **-7.6%** | **-13.0%** | 0.88-0.93 |

The two rows where `bind_inputs` was 74% of the run move by half; the rows where
it was a third of a run dominated by real `Concat` work move by 7.6% or sit in
the noise. That is the expected shape of an Amdahl bound, which is the main
reason to believe the measurement.

### 22.4 Three-arm ORT comparison (separate processes, 5 repetitions)

Three separate processes per cell - serial-native-only, parallel-native-only,
ORT-only - so no arm shares a process with another runtime's thread pool.
Medians of 5 repetitions.

| graph | t | ser ms | par ms | ort ms | ser/ort | par/ort | native gain |
|---|--:|--:|--:|--:|--:|--:|--:|
| `tr_llama3_s512` | 8 | 0.736 | 0.204 | 0.067 | 10.985 | **3.045** | -72.3% |
| `tr_llama3_s512` | 16 | 0.869 | 0.345 | 0.060 | 14.483 | **5.750** | -60.3% |
| `tr_whisper_s1500` | 8 | 0.662 | 0.336 | 0.096 | 6.896 | **3.500** | -49.2% |
| `tr_whisper_s1500` | 16 | 0.888 | 0.403 | 0.073 | 12.164 | **5.521** | -54.6% |
| `tr_bert_b8_s128` | 8 | 0.386 | 0.380 | 0.134 | 2.881 | 2.836 | -1.6% |
| `tr_bert_b8_s128` | 16 | 0.546 | 0.543 | 0.119 | 4.588 | 4.563 | -0.5% |
| `kvcat_llama3_p1023` | 8 | 0.587 | 0.585 | 0.072 | 8.153 | 8.125 | -0.3% |
| `kvcat_llama3_p1023` | 16 | 0.562 | 0.583 | 0.074 | 7.595 | 7.878 | +3.7% |
| `kvcat_llama3_p2047` | 8 | 1.452 | 1.542 | 0.087 | 16.690 | 17.724 | +6.2% |
| `kvcat_llama3_p2047` | 16 | 1.457 | 1.399 | 0.105 | 13.876 | 13.324 | -4.0% |
| `kvcat_llama3_p4095` | 8 | 4.930 | 4.822 | 0.237 | 20.802 | 20.346 | -2.2% |
| `kvcat_llama3_p4095` | 16 | 4.971 | 4.921 | 0.265 | 18.758 | 18.570 | -1.0% |
| `kvcat_llama3_p8191` | 8 | 7.292 | 6.370 | 1.110 | 6.569 | **5.739** | -12.6% |
| `kvcat_llama3_p8191` | 16 | 7.028 | 6.505 | 1.015 | 6.924 | **6.409** | -7.4% |
| `kvcat_llama3_b8_p2047` | 8 | 15.272 | 14.699 | 2.949 | 5.179 | 4.984 | -3.8% |
| `kvcat_llama3_b8_p2047` | 16 | 15.090 | 14.497 | 2.969 | 5.083 | 4.883 | -3.9% |

This is the full matrix, every cell measured. Two cells regress: `kvcat_p2047`
at 8 threads by **+6.2%** and `kvcat_p1023` at 16 threads by **+3.7%**. Both are
inside the +-8-13% noise floor established in §22.2, both are graphs whose
per-tensor input (8 MiB and 4 MiB) means only one of them can even reach the
parallel path, and both have a flat paired native-only ratio in §22.3 (+3.2% and
+1.4%, per-rep ranges spanning 1.0). They are reported as regressions rather
than explained away; no separate dispersion was collected for this table, so
they cannot be excluded on this data alone.

**These `ser/ort` ratios are not comparable to §20.2's.** §20.2 measured both
runtimes in one process; here ORT runs alone and is much faster in absolute
terms, so the denominators differ. Use §22.4 only for the `par/ort` vs `ser/ort`
comparison within a row.

### 22.5 What this does and does not claim

* The two Transpose graphs above the threshold - `tr_llama3_s512` and
  `tr_whisper_s1500` - go from 6.9-14.5x ORT to **3.0-5.8x ORT**. Still a loss;
  a large fraction of what remains is `collect_outputs` (§21) and the
  single-node harness's double-copy artefact (§21.3).
* `kvcat_llama3_p8191` improves 7-13%. The other `Concat` rows do not move,
  because their inputs are at or below the threshold or their runtime is
  dominated by real `Concat` work.
* Nothing at or below 2 MiB is touched. Decode-shaped graphs - the ones §10
  actually assigns - bind under a megabyte and take the serial path unchanged,
  and after §22.6 they do not so much as query the Rayon pool.
* No numerical change of any kind: the bytes written are identical, and a
  bit-identity falsifier plus a non-vacuity counter enforce it in CI.

### 22.6 What CI caught that no benchmark would have

The first version of the gate asked `host_copy_workers()` before it looked at
the size. `host_copy_workers()` calls `rayon::current_num_threads()`, and that
call **initialises the global Rayon pool**. So every process that bound a graph
input started a thread pool, whatever the size of the copy - including processes
that never run a parallel kernel at all.

Nothing in §22 would have found this. Every benchmark here runs a model, and any
model already has the pool up; the cost is invisible once it is paid. What found
it was the Miri job, which reported UB inside `crossbeam-epoch`'s epoch reclaim
during pool teardown in `onnx-runtime-session`'s `tensor::tests` and
`sequence::tests` - test binaries that had previously never started Rayon.

The UB is in crossbeam's provenance handling under Stacked Borrows, not in this
change, but the change is what caused those binaries to execute it. Reproduced
and bisected locally:

| gate order | `tensor::tests` | `sequence::tests` | `prefetch` | `device_binding_tests` |
|---|---|---|---|---|
| workers, then size | **UB** | **UB** | ok | ok |
| size, then workers | ok | ok | ok | ok |

The fix is to decide on size first - a cached atomic load - and only ask about
workers once a copy is known to be over the threshold. Below the threshold
nothing touches Rayon, which is what the "decode-shaped graphs are untouched"
claim was always supposed to mean.

One caveat on calling the report a false positive. During review of this fix a
full `-p onnx-runtime-ep-cpu --lib` run died with SIGSEGV once at process exit
and then passed three times, on a host under heavy contention. That is not
caused by this change - the parallel-copy tests start Rayon whatever the gate
order - but it is the same teardown path Miri objects to, so "known false
positive" may be understating it. Recorded here rather than dismissed; it is a
dependency-level question, not one this section can settle.

The lesson worth recording: a performance change can be neutral on every
benchmark and still alter process-wide behaviour. The soundness job was the only
instrument pointed at that.

### 22.7 Re-verification after the review rework

Review found that the threshold lookup ran `std::env::var` on *every* copy,
before any size check - a locked `getenv` plus a `String` allocation charged to
the small-copy path this change claims to leave alone. The environment is now
read once per process into a `OnceLock` and the cheap worker-count gate runs
first. That is a material change to the measured code, so the headline cells
were re-measured on the shipped version rather than assumed to carry over.

Paired, same binary, one process per arm, 16-thread budget, `native` p50 in ms:

Values are listed sorted, so they are marginals rather than paired samples. The
ratio range is therefore the **widest ratio any pairing could produce** -
`min(parallel) / max(serial)` to `max(parallel) / min(serial)` - which is the only
bound the data supports, applied identically to every row.

| graph | per-tensor input | reps | serial | parallel | ratio range |
|---|---|--:|---|---|---|
| `tr_llama3_s512` | 8 MiB | 5 | 0.772 / 0.806 / 0.815 / 0.860 / 0.864 | 0.343 / 0.405 / 0.419 / 0.424 / 0.439 | **0.40-0.57** |
| `tr_bert_b8_s128` (control) | 3 MiB | 3 | 0.514 / 0.531 / 0.535 | 0.520 / 0.542 / 0.546 | 0.97-1.06 |
| `sm_decode_h32_kv1024` (control) | 0.125 MiB | 3 | 0.023 / 0.023 / 0.024 | 0.023 / 0.023 / 0.023 | 0.96-1.00 |

The `tr_llama3_s512` win survives the rework: even the worst pairing is 0.57, so
every repetition is a win. The two control graphs - one just below the threshold,
one two orders of magnitude below it - are flat.

**The control rows are not a falsifier for the per-call `env::var` the review
removed.** That call cost ~72 ns, which is 0.3% of the 23 us decode run - below
this measurement's 1 us display resolution and well inside the row's own ~4%
spread. The evidence that the read is now cached is the code, not this table.
What these rows do show is that the reworked gate ordering did not disturb the
below-threshold path, which is what they are here for.

## 23. The rule changed: this EP never hands a node to ORT's CPU EP

Phases 1–4 treated "decline to ORT" as a legitimate outcome for a range where we
lose, and §10 and §15 above are written that way. That is no longer the project
rule:

> 我们的 cpu ep 不要分配到 ort cpu ep 上。用了我们的 cpu ep 就不要用 ort 的 cpu ep。但凡比 ort 慢的都要想方设法比 ort 快。
>
> *(Our CPU EP must not assign work onto ORT's CPU EP. If you are using our CPU
> EP, do not use ORT's. Anything slower than ORT must be made faster than ORT,
> by whatever means it takes.)*

So a losing cell is a kernel to fix, not a node to give away, and every ratio in
this document is now a work item rather than a justification. Nothing about the
*measurements* changes — they were paired, interleaved, same-host and
same-thread, and they still say what they said.

### 23.1 The code-level declines were already gone; the silent one was not

The perf-based decline was withdrawn before this section was written:
`claim_preference_node` returns `Claim` unconditionally and `assignment_policy.rs`
is deleted. Auditing what that actually guaranteed turned up two holes.

**Hole 1 — the guarantee was untested for the operators this EP exists for.**
`plugin_ort_e2e`'s `ASSIGNMENT_FIXTURES` covered 23 activation and normalisation
graphs and *zero* attention, MoE, KV-cache, Softmax, Transpose or RoPE graphs.
The rule was asserted where it was easy, not where it mattered.

**Hole 2 — `GetCapability`'s shape filter is a third, silent decline layer.**
Three independent gates must all pass for a node to reach our kernels:
`supports_op` (dtype/shape capability), `claim_preference_node` (was perf), and a
**fail-closed shape-inference filter** that drops any claim whose node has no
rule in `compute.rs`. That last one answers to neither of the first two, and it
was giving away `com.microsoft::Attention`, `MoE`, `QMoE`,
`PackedMultiHeadAttention`, `ScatterND`, `ScatterElements` and `Trilu`. For
`PackedMultiHeadAttention`, which **ORT has no CPU kernel for at all**, that
"fallback" bought a load failure rather than a slower run; for the rest it
simply meant not running an op we implement.

### 23.2 Two decode ops were being handed to ORT by a dtype rule

With shape rules added for all seven ops and fourteen new fixtures wired in, a real
ORT 1.27 session immediately failed the rewritten assignment test on the two ops
that matter most in decode:

```
2 of 32 fixtures were handed to ORT's CPU EP.
  rotary_assignment_f32: 'RotaryEmbedding'      — ours=[], ORT was given ["RotaryEmbedding"]
  gqa_assignment_f32:    'GroupQueryAttention'  — ours=[], ORT was given ["GroupQueryAttention"]
```

The cause is that the plugin advertises one *union* of dtypes per op and tests
every input slot against it. Attention ops map to `FLOAT_DTYPES`, but
`RotaryEmbedding`'s `position_ids` is int64 and `GroupQueryAttention`'s
`seqlens_k` / `total_sequence_length` are int32, so the integer slots failed the
float test and the claim was dropped. `input_dtype_constraints_for_op` already
existed for the opposite problem (a union too *wide* for `MatMulNBits`); it just
had no entries for the attention family. Adding the per-slot tables — with the
`com.microsoft` and `ai.onnx` RoPE slot orders kept separate, because they differ
— is what makes these two ops reachable at all.

**This means §1's and #1078's RoPE and GQA plugin-path numbers describe an EP
that was not running those ops.** Any plugin-path attention measurement taken
before this fix needs re-taking.

### 23.3 ORT stamps schema defaults on the node, and one of them is not 0

With the dtype fix in, GQA still failed — at kernel construction:

```
STAGE [CreateSession] FAILED: get_kernel failed for node '' (GroupQueryAttention):
  GroupQueryAttention: smooth_softmax is not yet supported (got -1)
```

The fixture never sets `smooth_softmax`. ORT materialises **schema defaults**
onto a node before an EP sees it, and the contrib schema's default for that
attribute is **-1**, not 0. ORT's own kernel enables the feature only for the
exact value 1, so -1 means "off" — but our `!= 0` test read it as "on" and
refused the node. Every GQA node from every real ORT session hit this.

**Correction, from review:** the same argument was originally made for `scale`,
and it is wrong. Instrumenting the GQA, MHA and `com.microsoft::Attention`
factories during a real `CreateSession` over these fixtures prints
`scale_attr=None` for all three — `scale` is `OPTIONAL_VALUE` with no schema
default, so ORT does *not* stamp it, and removing the guard leaves the numerics
in §23.5 unchanged at 1.006e-7. The guard stays as **defence, not a fix**: both
ORT's kernels and ours read an explicit `scale = 0` as "use 1/sqrt(head_size)"
rather than literally, so a zero taken at face value would multiply every score
by zero and return a silently wrong answer instead of an error. `attention.rs`,
`msft_attention.rs`, `multi_head_attention.rs`, `group_query_attention.rs` and
`packed_multi_head_attention.rs` now all treat a non-positive `scale` as absent.
`> 0` is deliberately broader than ORT's `== 0`, because a negative scale is
meaningless and would otherwise be honoured.

`smooth_softmax` is the load-bearing half of this finding and it does hold:
`smooth_softmax_attr=Some(-1)` on a node that never set it.

### 23.4 Assignment matrix, all 32 fixtures, real ORT 1.27 CPU session

`no_supported_node_is_ever_left_to_the_ort_cpu_ep` collects every failure and
asserts once, so a run reports the whole matrix instead of stopping at the first
decline. Every row is a real session with
`session.record_ep_graph_assignment_info=1`, queried through
`Session_GetEpGraphAssignmentInfo`.

| fixture | op | before | after |
|---|---|---|---|
| `softmax_assignment_f32` | `Softmax` | no fixture | **ours** |
| `transpose_assignment_f32` | `Transpose` | no fixture | **ours** |
| `kv_concat_assignment_f32` | `Concat` | no fixture | **ours** |
| `kv_scatternd_assignment_f32` | `ScatterND` | silently declined | **ours** |
| `scatter_elements_assignment_f32` | `ScatterElements` | silently declined | **ours** |
| `trilu_assignment_f32` | `Trilu` | silently declined | **ours** |
| `rotary_assignment_f32` | `RotaryEmbedding` | **ORT** (dtype union) | **ours** |
| `mha_assignment_f32` | `MultiHeadAttention` | no fixture | **ours** |
| `gqa_assignment_f32` | `GroupQueryAttention` | **ORT** (dtype union) | **ours** |
| `gqa_rotary_pos_assignment_f32` | `GroupQueryAttention` + int64 `position_ids` | **ORT** (dtype union, slot 9) | **ours** |
| `msft_attention_assignment_f32` | `com.microsoft::Attention` | silently declined | **ours** |
| `packed_mha_assignment_f32` | `PackedMultiHeadAttention` | silently declined, then **ORT** (dtype union) | **ours** |
| `moe_assignment_f32` | `MoE` | silently declined | **ours** |
| `moe_assignment_f16` | `MoE` float16 | silently declined, then **ORT** (f32-only union) | **ours** |
| `qmoe_assignment_f32` | `QMoE` | silently declined, then **ORT** (dtype union) | **ours** |
| 23 pre-existing activation/norm fixtures | — | ours | **ours** |

Six of those rows exist because review asked for them, and four of the six
found a decline that was still live after the first pass: GQA's `position_ids`
is optional input **9**, and listing only slots 5 and 6 left a `do_rotary` node
with explicit int64 positions failing the float union; `QMoE`'s uint8-packed
expert weights and `PackedMultiHeadAttention`'s int32 `token_offset` /
`cumulative_sequence_length` did the same. And `MoE` advertised **float32
only** while its kernel widens float16 and bfloat16 to f32 and narrows on the
way out — so every *realistic* MoE node, which is half precision, was declined
while the f32 fixture passed. One dtype's worth of coverage is not coverage.
The pure-Rust inventory test cannot
see any of these — it builds synthetic nodes and never opens an ORT session, so
it is blind to the dtype filter and to the kernel factory. **Only a real session
per op finds them, which is the argument for one fixture per rescued op rather
than a representative sample.**

`every_fixture_loads_with_cpu_fallback_disabled` runs the same 38 with
`session.disable_cpu_ep_fallback=1`, so ORT is *forbidden* from placing a
supported node on its own CPU EP: **38/38 load and 38/38 are assigned to us.**
**Correction, from review.** An earlier draft of this section claimed ORT has
no CPU kernel for `MoE`, `QMoE` or `com.microsoft::Attention`. It has all three;
the reviewer loaded *and ran* these very fixtures on ORT's
`CPUExecutionProvider` under both 1.27 and 1.28. Only
`PackedMultiHeadAttention` genuinely has none, which the falsifier shows
directly: reverting `PACKED_MHA_SLOTS` does not hand the node to ORT, it fails
session creation with `Could not find an implementation for
PackedMultiHeadAttention(1)`. For the other ops the decline meant we were not
running an operator we implement — bad, but not fatal.

One fixture is deliberately **not** in that table:

| fixture | op | on `main` | after | why excluded |
|---|---|---|---|---|
| `qmoe_columnwise_f32` | `QMoE`, `block_size` absent | ORT | ORT | no column-wise kernel here; see §23.6 |
| `moe_sparse_mixer_f32` | `MoE`, `use_sparse_mixer=1` | ORT | ORT | no sparse-mixer router here |
| `gqa_smooth_softmax_f32` | `GQA`, `smooth_softmax=1` | ORT | ORT | no attention-sink softmax here |

`factory_only_capability_limits_are_declined_at_claim_time` asserts all three:
the node lands on ORT, the session still loads, and — the point of the test —
the decline happens in `supports_op` rather than in the kernel factory, where it
would have been unrecoverable.

### 23.5 Assignment is not the guarantee — the arithmetic is

A claimed node that computes the wrong answer is worse than the deferral it
replaced, so the two ops that were being handed away are also checked against
the implementation they displaced. `rope_and_gqa_execute_on_our_ep_and_match_ort_numerics`
runs each fixture twice over the same model file and the same input bytes: once
with our EP appended and fallback disabled, once with our EP simply not
appended, so ORT resolves the node to its own contrib CPU kernel.

| fixture | output | values | max relative error vs ORT |
|---|---|---|---|
| `rotary_assignment_f32` | `output` | 4 096 | 1.182e-7 |
| `gqa_assignment_f32` | `output` | 4 096 | 1.006e-7 |
| `gqa_assignment_f32` | `present_key` | 1 048 576 | **0** |
| `gqa_assignment_f32` | `present_value` | 1 048 576 | **0** |

Both attention outputs are at float32 rounding, and the KV cache is written
bit-identically to ORT's.

### 23.6 Making an op reachable can break it: factory-only capability limits

`QMoE`'s fixture had to be written with `block_size = 32`, because this kernel
implements only the blocked form while ORT's schema leaves `block_size` without
a default — so an absent attribute means the *column-wise* form, one scale per
output row. The first draft of this section assumed ORT could not run that
either. **It can**, and review demonstrated it by loading and running the
fixture on ORT's `CPUExecutionProvider` under 1.27 and 1.28.

That turned a documented gap into a regression, and it is worth stating plainly
because it is a hazard of this whole change:

> Claiming an op is not free. `GetCapability` consults only the dtype and shape
> filters; a rejection raised later, in the kernel factory, arrives **after** ORT
> has compiled the node onto this EP, and **no fallback recovers from it**. So
> making `QMoE` reachable also made a previously-working column-wise model die
> at `CreateSession` with `block_size must be a power of two and at least 16,
> got 0`.

A third review round showed this was not one bug but a **class**, and that the
first fix had only patched one instance of it. Two more were live, both on
attributes real models set, and both of which ORT's CPU EP runs today:

| op | attribute | who sets it | verified on ORT CPU |
|---|---|---|---|
| `QMoE` | `block_size` absent (column-wise) | ORT-quantized MoE exports | 1.27 and 1.28 |
| `MoE` / `QMoE` | `use_sparse_mixer=1` | Phi-3.5-MoE, GRIN-MoE | 1.28 |
| `GroupQueryAttention` | `smooth_softmax=1` | Gemma-style attention sink | 1.28 |

The fix is structural rather than three more `if`s. Each kernel's attribute
parsing is now a single function, and the claim-time guard *is* that function:

```rust
pub(crate) fn unsupported_reason(node: &Node) -> Option<String> {
    attributes_from_node(node).err().map(|e| e.to_string())
}
```

so a limit cannot be added to one of these factories without appearing at claim
time too. Be precise about the scope of that guarantee: **within a wired op**
drift is impossible by construction; *wiring an op* is still discipline.
`supports_op` consults it for `MoE`, `QMoE`, `GroupQueryAttention`,
`MultiHeadAttention`, `com.microsoft::Attention`, `ai.onnx::Attention` and
`PackedMultiHeadAttention`, which sweeps up the rest of the family in one go:
GQA's quantized-KV and `qk_output` rejections, msft `Attention`'s `do_rotary`
and `past_present_share_buffer`, `ai.onnx::Attention`'s `qk_matmul_output_mode`,
and `QMoE`'s `expert_weight_bits` / `quant_type`.

Two tests pin it. `provider::tests::every_factory_attribute_rejection_is_mirrored_at_claim_time`
is pure Rust — for eleven hostile nodes it asserts *both* that the factory
rejects and that `supports_op` declines, so it fails if the two ever diverge —
and `plugin_ort_e2e::factory_only_capability_limits_are_declined_at_claim_time`
proves it against real ORT for the three fixtures above: the node lands on ORT,
the session loads. Disabling the claim-time guard turns each of the three into

```
STAGE [CreateSession] FAILED: Compile: get_kernel failed for node '' (MoE):
  MoE: use_sparse_mixer=1 is unsupported by the Phase-1 CPU reference kernel
```

Review then audited the rest of the EP for the same defect and found ten other
factories that reject something `supports_op` does not pre-check — `MatMulNBits`
(`bits`, `block_size`), `Resize`, `Pad`, `GridSample`, `LpNormalization`,
`Unique`, `BitShift`, `ConstantOfShape` and two `pkg.nxrt` internals. **None is
a live regression**: each rejects only schema-invalid values or configurations
ORT's own CPU kernel also refuses, and the reviewer could not construct a case
ORT runs and we reject. They are recorded here so the next person does not have
to redo the audit, and so the claim above is not read as EP-wide.

**These are the only deliberate declines in the suite and every one is a
capability answer, not a performance one** — we have no column-wise, sparse
mixer or smooth-softmax implementation, and the choice for each is between ORT
running the model and nobody running it. They are not exceptions to §23's rule,
which is about ranges where we are *slower*; they are an admission that we owe
three kernels. Each should stop being an exception as soon as its kernel
exists.

### 23.7 What this does not fix

It makes the losing ranges *reachable*; it does not make them fast. §15's fused
region is still 3–15x short at 8–16 threads, §19–20's Transpose gaps and §18's
MoE cells are unchanged, and #1078's own evidence has float32 RoPE losing 12/12
cells at 1.53–17.21x. Under the rule above every one of those is now a kernel to
fix with no exit, and the RoPE grid is first because it is the op that was being
given away.

---

## 24. Phase 6: the input-binding copy, removed (#1146)

§21.2 described `bind_inputs` as an open gap and explicitly declined to claim a
value for closing it. It is closed. `Executor::prepare_run_buffers` now installs
a **borrowed** `DeviceBuffer` over the caller's tensor bytes and parks the owned
buffer for the duration of the run, so a host-accessible EP reads the caller's
memory directly, exactly as an ORT CPU `Value` does.

The borrow is only taken when all of these hold: the byte lengths match, the
buffer's device is host-accessible and is this EP's device, the pointer meets the
contiguous-layout alignment (64 B), and the value is not a graph output, not a
shared buffer, and not stage-2 excluded. `unbind_borrowed_inputs` restores the
owned buffer on every normal and error path, `reset_run_state` restores again at
the top of the next run, and `Drop for Executor` is the final backstop.

**Sequence storage was the subtle half.** Guarding the *slot* is not enough when
a handle can be moved out of it: `read_seq_element` *moves* a value's buffer into
cross-run `shared_buffers`, which would have outlived the caller's tensor.
Sequence storage now copies a borrowed buffer into a fresh owned allocation.
Review confirmed `read_seq_element` is the only such choke point.

`bind_inputs` on `rope_llama3_s128` went from **~601 µs to 0.7 µs**. Across the
full 57-cell transform/KV matrix: **56/57 cells improved, 342/342 parity PASS**,
and three cells crossed to beat ORT. The one non-improvement,
`sm_decode_h32_kv1024` at t=16 (2.183 → 2.188), sits inside overlapping
dispersion and is reported as neutral, not a win.

### 24.1 What this changes about §21

* §21.2's "that is not attempted here, and this section makes no claim about how
  much it would be worth" is superseded. It was worth 60-74% of a Transpose run.
* §21.3's first bullet stands, but the arithmetic moves: with input binding at
  ~0 the residual per-run cost is now dominated by `collect_outputs`, which §21.1
  measured as flat rather than byte-proportional. The output half of the arena
  work is therefore still open, and it is still true that some fixed per-run cost
  is folded into that number.
* §21.3's second bullet is now half-obsolete: the benchmark no longer charges our
  arm an input copy that ORT does not pay. Ratios measured before #1146 are not
  comparable to ratios measured after it.

## 25. Phase 7: RoPE was scalar, and its tasks were sized by layout (#1175)

Two independent defects in `RotaryEmbeddingKernel`, both found by phase-profiling
`rope_llama3_s128` after #1146 removed the binding copy and left
`exec_kernel.compute` as 96% of the run.

### 25.1 The fan-out was net-negative at every thread count

The kernel fanned out one Rayon task per layout unit - one `[B,H,S,D]` plane or
one `[B,S,H·D]` row. On `rope_llama3_s128` that is 4096 elements, 16 KiB, per
task. Measured `exec_kernel.compute` p50, native-only, parallel path against the
same kernel forced serial:

| threads | parallel | serial |
|---|---|---|
| 1 | 162 µs | 126 µs |
| 8 | 142 µs | 126 µs |
| 16 | **195 µs** | 127 µs |

Fanning 128 tasks over a **one-worker** pool cost 36 µs of pure bookkeeping. The
parallel path was slower than the serial path everywhere, and got worse as the
pool got wider.

`rotary_units_per_task` now aggregates whole layout units until each task clears
16 Ki elements, targets ~4 tasks per worker for balance, and returns a single
serial task when the pool has one worker.

### 25.2 The inner loop was never vectorised

242 µs for 524 288 elements is ~1.4 cycles/element - scalar arithmetic. Both
rotation conventions now have explicit AVX2 arms.

They use separate `_mm256_mul_ps` + `_mm256_sub_ps`/`_mm256_add_ps` rather than
FMA. That is deliberate: FMA drops the intermediate rounding of `cos·x1`, so the
8-wide body and the scalar tail would disagree in the last ulp, and the same
graph would produce different bits depending on how a thread count happened to
split it. The kernel moves 8 bytes per 3 flops - it is bandwidth-bound, so fusing
the multiply is not measurable anyway.

The interleaved (GPT-J) arm de-interleaves with `_mm256_shuffle_ps` +
`_mm256_permutevar8x32_ps(…, (0,1,4,5,2,3,6,7))`. That index vector is its own
inverse, so the same constant straightens the operands and pre-scrambles the
results for the `unpack` pair. The A/B grid gained `rope_gptj_il_*` models
because `interleaved=1` had no coverage at all.

### 25.3 Result

`exec_kernel.compute` on `rope_llama3_s128`: **242 → 85 µs (t=1), 174 → 67 µs
(t=8), 138 → 59 µs (t=16)**.

Paired interleaved A/B against a real ORT 1.27 CPU session, 7 trials × 80 runs,
one driver invocation, base = the merge-base build:

| model | t=1 | t=8 | t=16 |
|---|---|---|---|
| rope_llama3_s1 | 1.347 → **1.278** | 1.349 → **1.273** | 1.399 → **1.320** |
| rope_llama3_b8_s1 | 1.227 → **1.032** | 1.310 → **1.145** | 1.028 → **0.881** |
| rope_llama3_s128 | 1.642 → **0.800** | 4.372 → **3.890** | 7.209 → **6.111** |
| rope_llama3_s512 | 1.258 → **0.929** | 4.485 → **3.437** | 7.036 → **5.574** |
| rope_gptj_il_s1 | 1.400 → **1.301** | 1.385 → **1.299** | 1.424 → **1.300** |
| rope_gptj_il_s128 | 1.911 → **0.874** | 3.724 → **3.628** | 7.486 → **6.146** |
| rope_gptj_il_s512 | 1.530 → **0.964** | 4.533 → **3.008** | 7.655 → **5.808** |

**21/21 cells improved, 0 regressed, 5 now beat ORT, parity 294/294 PASS.**

## 26. The real reason every kernel loses at 8 and 16 threads

§23.7 lists the remaining losses as if they were five independent problems -
MHA/SDPA, Softmax, RoPE, Transpose/Concat, MoE. They share one shape: each wins
or nearly wins at **t=1** and loses badly at **t=8/16**. That is one root cause,
and it is not in any of those kernels.

### 26.1 The measurement

An isolated 524 288-element elementwise region, 16 Rayon workers, 32 tasks, that
costs 388 µs on one thread. The only variable is how long the driver waits
between regions:

| gap between regions | region cost | effective speedup |
|---|---|---|
| none (back-to-back) | 67 µs | 5.8× |
| 20 µs | 226 µs | 1.7× |
| 100 µs | 230 µs | 1.7× |
| 500 µs | 236 µs | 1.6× |

A **20 µs** gap - less than the serial glue between two graph nodes - costs
~160 µs. Rayon parks its workers when a region ends; ORT's intra-op pool spins
first (`ALLOW_INTRA_OP_SPINNING`, on by default), so its workers are still hot
when the next op arrives and ours are not.

This is consistent with everything measured: `exec_kernel.compute` for
`rope_llama3_s128` at t=16 is 59 µs against ORT's ~30 µs, but the paired wall is
189 µs. The kernel is within 2× of ORT; the wake-up is 3× the kernel.

### 26.2 The mechanism already exists, and it is not reachable

`decode_spmd` implements the correct policy: a persistent pool whose workers hold
a core for a bounded `KMP_BLOCKTIME`-style window (500 µs) before parking, so
idle CPU still returns to ~0 between requests. It is wired only to the decode
GEMM path.

A prototype routing `RotaryEmbedding`'s fan-out through it takes
`rope_llama3_s128` at t=16 from **79 µs to 25 µs native-only - past ORT's
~30 µs**. That is the whole remaining gap, in one change.

It cannot be landed as-is, for two separate reasons.

**(a) The pool has exactly one dispatcher, and nothing enforces it.**
`SharedState::publish` writes the job into a single `UnsafeCell` slot
(`decode_spmd.rs:243`) and `dispatch` then waits on one barrier. The doc comment
on `wait` states the assumption - "the dispatcher is a single, never-idle thread"
- but it is an assumption, not an invariant. Two threads dispatching concurrently
overwrite each other's job pointer and both sets of workers run whichever closure
landed last. The failure mode is **not** a crash or a deadlock: it is silently
wrong tensors in one of the two sessions.

This was found by pointing the new `hot_parallel` unit tests at the pool and
running them under the default `cargo test` thread pool: the fan-out reported
having run 6 tasks when it dispatched 5, and separately 0 when it dispatched 5,
depending on what else in the 1350-test binary was dispatching at the time. Under
`--test-threads=1` the same tests pass. A per-module compare-exchange guard is
*not* sufficient, because the 11 existing dispatch sites in the matmul/GEMM
kernels do not take it.

Today this is latent rather than live: the decode engine runs its forward inline
on one thread, and the plugin EP calls `disable_persistent_decode_pool()` before
ORT can reach a kernel. It becomes live the moment two sessions in one process
both run decode GEMM, which is exactly the concurrency the KV work has to
preserve. Making the pool genuinely multi-dispatcher - a job slot and barrier
generation per dispatcher - is a prerequisite for any of this, and it is worth
doing on its own merits as a correctness fix.

**(b) Two spinning pools oversubscribe.** With the prototype enabled and an ORT
session alternating in the same process, the paired result becomes bimodal:
`rope_llama3_s128` at t=16 measured 0.030 ms on one run and 0.441 ms on the next,
with the same binary and the same command. The cause is visible in the affinity
log: `ONNX_GENAI_CPU_DECODE_THREADS=16` confines the process to CPUs 0-15, which
on this host is **8 physical cores** (siblings are adjacent: cpu0/cpu1 = core0).
15 pinned spinning workers on 8 cores, plus ORT's 16 unconfined intra-op threads,
is not a scheduling problem that tuning the blocktime solves - a sweep over
500/200/50/20/5/0 µs produced ratios from 0.233 to 72.3 with no monotone trend.

Note the compact mask is *correct* and should not be "fixed" to spread: pinning
one thread per physical core across 0,2,…,30 straddles two CCXs and measured
**worse** (0.133 ms vs 0.079 ms), because the 4.5 MiB working set fits in one
CCX's L3. The change that is indicated is capping *spinning* workers at one per
physical core within the compact set, which needs SMT sibling discovery
(`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`) that
`decode_affinity` does not currently do.

### 26.2.1 (a) is now fixed, and (b) alone still blocks

#1184 made the pool genuinely multi-dispatcher: `dispatch` claims the single job
slot with a compare-exchange and a losing caller runs the same shards inline on
its own thread, which is the identical computation because the shards already
partition their output by global worker index. That closed (a) as a correctness
matter, and the review confirmed the failure was reproducible - deleting the
claim gate makes the two-dispatcher test *hang* rather than fail, which is why
the hazard had gone unnoticed.

It did **not** make the routing landable. Re-wiring `RotaryEmbedding`'s fan-out
through the pool on top of #1184 and running the paired grid again reproduces
the bistability of §26.2(b) intact, and worse than before:

| model | t | rayon (base) | hot pool |
|---|---|---|---|
| rope_llama3_s128 | 8 | 5.456 | 11.313 |
| rope_llama3_s128 | 16 | 9.457 `[6.4-23.1]` | **72.704** `[20.7-86.6]` |
| rope_gptj_il_s128 | 16 | 6.247 | **55.751** `[29.0-85.7]` |
| rope_gptj_il_s512 | 16 | 5.445 | **1.232** `[1.08-35.6]` |
| rope_llama3_b8_s1 | 16 | 0.807 | **0.622** |
| rope_llama3_s512 | 16 | 30.771 | **18.385** |

Both directions in one grid, with dispersion bands wider than the medians. That
is not a tuning problem, and the bottom three rows are not a partial win - they
are the same bistable system landing on its other mode. Nothing here is
publishable as a ratio.

So the ordering is settled: **(b) is the binding constraint, not (a)**. Capping
*spinning* workers at one per physical core inside the compact mask - which
needs the SMT sibling discovery `decode_affinity` does not do - has to land
before the hot-pool routing can be evaluated at all, let alone merged. Until
then the t=8/16 column stays where it is, and the RoPE fan-out stays on Rayon.

### 26.3 What this means for the remaining §23.7 rows

Softmax, Transpose, Concat and the MoE cells should not be attacked as separate
kernel problems until §26.2(a) is fixed. Their t=1 numbers are the honest measure
of their arithmetic; their t=8/16 numbers are mostly measuring thread wake-up.
Any per-kernel tuning done against the t=16 column now is tuning against the
scheduler, and will have to be redone.

Per §26.2.1 that now reads more sharply: the scheduler work itself is blocked
behind SMT-aware spin capping, so *neither* the per-kernel tuning nor the
hot-pool routing is the next move.

The exception is §11.5's paged/appendable KV cache. `Concat` re-copying the full
history every token is a real algorithmic cost that no scheduler change touches,
and §8 already established that parallelising `ConcatKernel` is a dead end. That
remains the largest genuinely structural gap and is independent of all of the
above.

## 27. Where the CPU EP actually stands

> **Correction (§31).** The 66-cell grid in this section was produced by the MLAS-linked
> `bench_generic` and does not describe the shipping build. See §31.5 for the corrected
> production-native matrix.

Full transform + KV-concat grid on `main` at `f2e6ad97d`, paired interleaved
against a real ORT 1.27 CPU session, one driver invocation, 5 trials × 60 runs ×
15 warmups. Ratio is native/ORT p50, **lower is better**, and `< 1.000` is a win.

| model | t=1 | t=8 | t=16 |
|---|---|---|---|
| kvcat_llama3_b8_p2047 | **0.883** | 2.203 | 2.250 |
| kvcat_llama3_p1023 | 1.002 | 4.672 | 3.930 |
| kvcat_llama3_p2047 | **0.968** | 3.545 | 3.878 |
| kvcat_llama3_p4095 | 1.706 | 5.075 | 6.633 |
| kvcat_llama3_p8191 | 1.004 | 2.824 | 2.955 |
| rope_gptj_il_s1 | 1.300 | 1.290 | 1.287 |
| rope_gptj_il_s128 | **0.878** | 3.864 | 5.800 |
| rope_gptj_il_s512 | **0.962** | 3.018 | 5.773 |
| rope_llama3_b8_s1 | 1.160 | 1.261 | **0.874** |
| rope_llama3_s1 | 1.295 | 1.295 | 1.295 |
| rope_llama3_s128 | 1.022 | 4.657 | 17.557 |
| rope_llama3_s512 | 1.024 | 3.589 | 5.230 |
| sm_bert_b8_s128 | 1.246 | 3.294 | 5.179 |
| sm_decode_h32_kv1024 | 1.177 | 1.987 | 2.051 |
| sm_decode_h32_kv2048 | 1.183 | 2.648 | 3.015 |
| sm_decode_h32_kv4096 | 1.188 | 4.143 | 4.268 |
| sm_decode_h32_kv8192 | 1.214 | 5.423 | 6.776 |
| sm_prefill_h32_s512 | 1.328 | 2.539 | 2.655 |
| sm_whisper_cross | 1.374 | 2.933 | 3.059 |
| tr_bert_b8_s128 | **0.607** | 1.068 | 1.715 |
| tr_llama3_s512 | **0.820** | 2.796 | 4.292 |
| tr_whisper_s1500 | **0.984** | 2.311 | 3.413 |

**8 of 66 cells win. 58 lose. Parity 330/330 PASS.**

Read by column rather than by row, which is the whole point of §26:

| threads | wins | median ratio |
|---|---|---|
| 1 | 8 / 22 | ~1.02 |
| 8 | 0 / 22 | ~3.0 |
| 16 | 1 / 22 | ~3.6 |

At **t=1** the kernels are at parity - median 1.02, eight outright wins, and the
worst cell is 1.706. That column is a fair measure of the arithmetic, and it
says the arithmetic is broadly competitive. At **t=8 and t=16** the median is
3-3.6× with almost no wins. The kernels did not get worse; the scheduling did.

Two caveats on this table, stated rather than buried. The host is shared and
contended, so absolute numbers drift between runs - `rope_llama3_s128` at t=16
reads 17.557 here and 9.457 in the §26.2.1 grid taken later the same session.
Only ratios from a single driver invocation are comparable, which is why the
arms are interleaved. And cells whose native p50 is under ~100 µs
(`rope_*_s1`, `rope_llama3_b8_s1`) are overhead-dominated, which is why they sit
near a flat ~1.3 at every thread count instead of scaling.

### 27.1 The honest blocker list

1. **SMT-aware spin capping** (§26.2.1) - blocks the entire t=8/16 column, which
   is 58 of the 66 losing cells. Nothing else in this list can be measured
   cleanly until it lands.
2. **Paged/appendable KV cache** (§11.5) - the `kvcat_*` rows re-copy the whole
   history per token. Independent of (1); the only structural item that is
   actionable right now.
3. **Fused transpose-with-consumer** (§7, §19, §20) - `tr_*` wins at t=1 by
   reading and writing less, and that advantage should compose into QK/PV/MoE
   rather than being spent on a standalone node.
4. **Grouped/batched MoE GEMM** (§18) - not in this grid; measured separately at
   1.47-2.37× on qwen3-moe e16 top-8.
5. **Fused-attention session A/B** (§15) - the `sm_*` rows here are isolated
   single-node graphs. Softmax has to win inside a real fused SDPA region.
6. **Output/intermediate arena** (§21.3) - #1146 removed the input half;
   `collect_outputs` remains.

## 28. Phase 8: the appendable KV path, and what it actually cost

Phase 8 went at the KV cache structurally rather than through the scheduler
(which Sebastian now owns). Mapping the CPU appendable-KV path turned up two
defects that had nothing to do with thread counts, and both are now fixed.

### 28.1 A bf16 KV cache widened its entire capacity, every forward (#1199)

`PastCache::from_cache` classified a KV cache into one of three sources. It
borrowed contiguous `f32` and contiguous `f16` and sent everything else to
`to_dense_f32_widen`, which allocates a `Vec<f32>` over the cache's **whole
physical capacity** and widens every element into it — for a result the append
path then reads one token's rows out of and drops.

`bf16` landed in that catch-all. Three things made it worse than it looks:

* It is on the **general** path. `from_cache` runs on every
  `GroupQueryAttention` forward, before any aliasing gate is evaluated.
* It runs **twice per layer per token**.
* It scales with **capacity**, not with the valid prefix, so it does not get
  cheaper at short context and a generous `ONNX_GENAI_CPU_KV_MAX_LEN` makes it
  worse.

`f16` never paid any of it. The two half formats are handled identically
everywhere else in the kernel; only the classifier had the gap.

Classify + read one row at llama3-8B KV geometry (8 kv heads, head_dim 128,
capacity 4096 — 4.19M elements, a 16 MiB widen), best of 7 x 20, release:

| dtype | before | after |
|---|---|---|
| f16 | 83 ns | 83 ns |
| **bf16** | **606 147 ns** | **81 ns** |

bf16 now costs what f16 costs. On a 32-layer decoder that is ~39 ms per token
of pure widen removed; the unchanged f16 column confirms nothing else moved.

**The part worth remembering.** The first version of this fix used
`widen_bf16_slice_into`, a raw `(bits << 16)` shift. The path it replaced
reaches `half::bf16::to_f32`, which **quiets signalling NaN**. Those two
disagree on exactly 126 of the 65536 bit patterns. `dtype.rs` already carried a
warning about precisely this, on `widen_quieting`:

> A bulk path that replaces the latter has to quiet, or it would silently
> change what every bf16 kernel sees for sNaN inputs.

`cast.rs`, `elementwise.rs` and `dense_elementwise.rs` had each taken that
advice and each left a comment saying so. This kernel was the one that did not
— and the reason it was easy to get wrong is that there was no *safe* slice
wrapper for the quieting widen, only the unsafe AVX2 intrinsic. There is one
now (`widen_bf16_slice_quieting_into`), documented against its raw sibling.

The test that "verified" bit-identity listed one NaN pattern and happened to
pick a **quiet** one, so it passed against a non-quieting widen. It is now an
exhaustive sweep of all 65536 patterns across six `(start, len)` windows. Two
further instances of the same defect in the same kernel — `HalfKv::widen_into`
on the in-place append path, and `widen_rotary_prefix`, whose own fallback four
lines below is `to_dense_f32_widen` — were fixed with it.

### 28.2 The CPU KV cache could not grow at all (#1203)

`decode_cpu_inplace` hard-errored the moment a context outgrew the cache:

    bail!("CPU KV capacity exceeded: requested context length {total_len}, ...")

`DEFAULT_CPU_KV_MAX_LEN` is 4096, so **token 4097 killed the generation**. The
suggested remedy — raise the env var — means guessing a bigger number up front
and paying for it in resident memory for the whole run. The CUDA path has had
growth since it landed (`ensure_capacity`, bucketing, VMM remap); the CPU path
had none of it.

Capacity is axis 2 of `[B, H, capacity, Dh]`, not the outermost axis, so head
`i` lives at `i * capacity * Dh` and **growing relocates every head but the
first**. A flat memcpy would leave head 1 onward reading the previous head's
tail — plausible numbers, silently wrong attention, no crash. The prefix is
carried one contiguous run per `(batch, head)` block. Growth doubles, so
re-binding is amortised O(1) per token.

**The end-to-end test was worthless on the first attempt, and this is the
lesson.** It asserted that a decode with a deliberately tiny KV capacity
produced the same tokens as one with room to spare. It passed — and it *also*
passed when growth was changed to drop the entire KV history, because the
scalar-GQA fixture's q/k/v are `Constant` zeros and its output does not depend
on cache contents at all. Token equality proved nothing.

The property worth testing is the one the unit tests cannot reach:
`GroupQueryAttention` decides to append in place by comparing the past-input
and present-output pointers **at execution time**, so a reallocated buffer that
failed to re-alias would silently stop appending. Nothing observable
distinguished that from success — so the f32 in-place path now has a counter
(`present_inplace_count()`, mirroring the half-precision one that already
existed), and the test asserts a cache forced to grow repeatedly appends in
place **exactly as often** as one that never grows. Breaking the aliasing fails
it. As a side effect these counters, which had no consumer outside the kernel's
own unit tests, now have one.

### 28.3 Where the KV axis stands against ORT, context 1k–8k

Concat-shaped KV, matched paired-interleaved A/B, one driver invocation,
5 trials x 30 runs x 10 warmups. Ratio is native/ORT, lower is better.
Parity **60/60 PASS**.

| model | t=1 | t=8 | t=16 |
|---|---|---|---|
| kvcat_llama3_p1023 | **0.979** | 4.079 | 4.207 |
| kvcat_llama3_p2047 | **0.978** | 4.298 | 5.585 |
| kvcat_llama3_p4095 | 1.981 | 5.122 | 5.322 |
| kvcat_llama3_p8191 | **1.009** | 2.418 | 2.698 |

Two things to read here, and one not to.

* **t=1 is at parity across the whole 1k–8k range** (0.98–1.01), with p4095 the
  single exception.
* **t=8/16 lose 2.4–5.6×.** This is §26's worker-park effect, not a KV defect:
  the same shape of result appears on every kernel in §27 regardless of what it
  computes. It is Sebastian's area and nothing in Phase 8 touches it.
* **Do not read the p4095 row as a regression against p8191.** ORT's absolute
  time doubles from p4095 to p8191 (1.92 → 3.73 ms) as expected, while native's
  is flat (3.81 → 3.76 ms). Native is paying a roughly fixed ~3.8 ms at both
  lengths, so the *ratio* improves with context purely because ORT's cost grows
  into it. That flat cost is unexplained and is the next thing to chase on this
  axis; the honest summary is "native does not scale with past length here the
  way ORT does, in both directions".

### 28.4 What is still structurally blocked

`Concat`-shaped KV caches still pay the full O(S) history copy, and that is
still the §13 boundary: `ExternalValue` has no strides field, so a device-bound
value is always dense, and a capacity-backed dense view of `[B,H,S,D]` with the
growth axis at position 2 is not expressible. The GQA present==past route
(which Phase 8 improved) remains the only appendable path, and the engine still
declines to take it for any non-f32 cache, so the bf16 fix above is reachable
today only through the raw session API — though it applies to every bf16 GQA
forward regardless.

Beam reorder still does not exist for CPU or CUDA; `num_beams` is passthrough
metadata and the decode loop is single-sequence. Nothing in Phase 8 changed
that, and no claim about beam search should be made from this work.

## 29. Phase 9: Softmax was paying for a copy ORT never makes (#1219)

### 29.1 The defect

`softmax_rows` - the shared entry point every `Softmax` node reaches, and the one
`scale_mask_softmax_rows` and `softmax_slices` route through - began with

    dst.copy_from_slice(src);
    softmax_rows_in_place(dst, n, d);

That is a full extra read+write pass over the tensor before any arithmetic happens. On
`sm_prefill_h32_s512` the logit block is 33 MiB, so each inference moved **66 MiB** of
traffic that produced no result. ORT never pays it: it hands `MlasComputeSoftmax` the
graph input and output buffers directly.

This is the whole explanation for §9's and §27's standing Softmax losses. The kernel's
arithmetic was never the problem, and no amount of vectorising the reducer would have
found it - the cost was in the line before the reducer was called.

### 29.2 The fix

Compute out-of-place. The reducer already traverses each row twice (once for the max,
once for the exponent sum), so writing into `dst` on the final pass instead of
pre-seeding `dst` with a copy is free. `softmax_rows_serial_out(src, dst, n, d)` takes
the two buffers separately and has both a pure-Rust and an MLAS arm.

Aliasing was already handled upstream and did not need new guards:
`output_direct_write_eligible` refuses the direct-write slice when input and output
overlap, so `softmax_rows` is only ever handed disjoint buffers. The pre-existing test
`an_output_aliasing_its_input_matches_the_disjoint_result` covers that path.

### 29.3 Result

Paired interleaved A/B, one driver invocation, 3 repetitions x 40 runs x 15 warmups,
`--native-threads 1 --ort-intra-threads 1`, base arm built from `origin/main` in a
separate worktree. Median `native/ort`, lower is better:

| model | before | after |
|---|---|---|
| sm_prefill_h32_s512 | 1.318 | **0.924** |
| sm_bert_b8_s128 | 1.242 | 1.033 |
| sm_decode_h32_kv1024 | 1.174 | 1.015 |
| sm_decode_h32_kv2048 | 1.291 | 1.026 |
| sm_decode_h32_kv4096 | 1.183 | 1.028 |
| sm_decode_h32_kv8192 | 1.199 | 1.035 |

Parity 36/36 PASS, `max_abs=0` in every cell.

**What this does not claim.** One shape is won outright. The other five moved from a
17-29% loss to a 1.5-3.5% loss. They are *not* won. The remaining few percent on those
shapes is unattributed and is the next thing to chase on this axis.

### 29.4 A measurement caveat that applies to this whole document

> **Superseded by §31.** The caveat below was correct and its consequence was far larger
> than it estimated. The harness has since been fixed (#1231) and the production arm
> measured. The main table in §29.3, and every ratio in §18 and §27, is an **MLAS-arm**
> number. See §31.6 for exactly which tables are affected.

`onnx-genai-bench` declares `required-features = ["mlas"]`, so `bench_generic` can only
be built with MLAS linked in. Every ratio published in this document - including §27's
66-cell grid - was therefore produced by the **MLAS-linked build**, and where a kernel
has an MLAS arm (`softmax_rows_serial` does) that arm is what was measured.

`mlas` is **not** a default feature of `onnx-runtime-ep-cpu`. The build that ships runs
the pure-Rust arm. For this change that distinction does not affect the conclusion,
because the removed copy sat at the shared, arm-agnostic `softmax_rows` call site and
both arms lose it identically. But it does mean the *magnitude* of the win on the
production build is unmeasured, and no ratio in this document should be read as a
statement about a non-MLAS build until the harness can produce one. Fixing that -
relaxing `required-features` so `bench_generic` can build both arms - is open work.

### 29.5 Review finding: the bit-identity claim had no in-tree assertion

The adversarial review of #1219 accepted the change but noted that its central claim -
that the out-of-place form is bit-identical to the copy-then-in-place form - rested on
prose. Every existing test compared with a `1e-6` tolerance over finite synthetic
logits, and the non-finite tests exercised `softmax_rows_in_place`, not `softmax_rows`.

That gap matters here specifically because the in-place reducer is vectorised and the
out-of-place path is not, so the two `exp` implementations have to agree on the
non-finite lanes exactly. `out_of_place_softmax_is_bit_identical_on_pathological_rows`
now compares raw bits over fully masked rows, `+inf`-max rows, quiet, signalling and
negative-payload NaNs, and denormals.

Two things are worth recording about writing it:

* The first falsifier chosen - guarding the normalisation with
  `if sum == 0.0 { 0.0 }` - **did not trip the test**, and the reason was that the
  test's own doc comment was wrong. A fully masked row does not normalise `0.0/0.0`:
  the row max is `-inf`, so `exp(-inf - -inf)` is `exp(NaN)` and the sum is NaN, never
  zero. The NaN arrives by propagation, not by division. A falsifier that fails to fire
  is not evidence the test is vacuous - it can equally mean the falsifier is
  inapplicable - and the only way to tell the two apart is to work out *why*.
* The test carries an explicit non-vacuity guard asserting those rows still produce NaN
  at all, so a future change that quietly made every row finite would fail loudly
  instead of leaving a bit comparison that is trivially true. This is the third vacuous
  or near-vacuous test caught in this effort (§28.2 has the other two); assume the
  default state of a new test is vacuous until a falsifier has been run against it.

## 30. Phase 10: the profiler was charging the run for its own instrument (#1226)

### 30.1 The defect

`activation_memory_planning_enabled()` returned `phase_profile::enabled()`, so the
executor's activation-memory planner ran on every run whenever phase profiling was on.
The planner rebuilds the view map and re-plans every activation each run - work the
shipped runtime never does. `--phase-profile` therefore perturbed the run it was
measuring and reported its own cost back as a phase of that run,
`run_scoped.activation_memory_plan`, which no unprofiled run ever pays.

The planner now has its own gate. `--phase-profile` drives the profiler only;
`NXRT_EXEC_PHASE_PROFILE=1` in the environment still enables both, so the CLI memory
report keeps its stats; `NXRT_ACTIVATION_MEMORY_PLAN=1` opts in without profiling.

The CUDA step profiler switched the planner on as a side effect of enabling phase
accounting, taxing the very decode steps it exists to time. That coupling is gone.

### 30.2 Magnitude, and two withdrawn claims

The planner's own span reads **1.9-6.0 us per run**. Removing it lowers `native_min` on
`sm_decode_h32_kv4096` from **0.065 ms to 0.063 ms** - median of 15 interleaved
repetitions, 300 runs each, single thread, both arms in one loop, 13 of 15 favouring
the fixed arm. About **3%**, in line with the span.

The first version of this section, and of the PR, claimed a **35%** inflation
(0.065 -> 0.088 ms, ratio 1.023 -> 1.080) and claimed the planner also polluted the
adjacent `exec_kernel.compute` figure (84.5 vs 60.3 us/call). **Both are withdrawn.**
Review could not reproduce either, and a properly interleaved re-measurement showed
why: this host is bimodal under contention, the 0.065/0.088 ms split appears in
**both** arms, and `exec_kernel.compute` swings 75-168 us/call regardless of the
planner.

### 30.3 The methodology lesson, which is the point of this section

The withdrawn numbers were produced by running arm A three times, then arm B three
times, inside one shell loop - and reading the *mean* `native=` field. Every arm-vs-arm
number in this document is supposed to come from interleaved repetitions in one
invocation, precisely because this host drifts. I wrote the loop that way, saw a clean
35% separation, and believed it, because the separation was large and the story was
satisfying. It was the machine changing state between the two halves of the loop.

Three things that would have caught it, and are now the standing rule for this
document:

* **Interleave at the innermost level.** A B A B A B, not A A A B B B. The second form
  cannot distinguish an arm effect from a time effect, and on this host time effects
  are large.
* **Prefer `native_min` to the mean for small graphs.** The minimum over 300 runs is
  the least contaminated estimator available when the noise is one-sided, which
  contention noise is. Switching to it turned an unstable 0.065-vs-0.088 into a stable
  0.065-vs-0.063.
* **Distrust a result proportional to how much you like it.** A 35% win from deleting
  one planning call was far larger than the planner's own measured span, and that
  internal inconsistency - the span said 2-6 us, the wall clock said 23 us - was
  visible in my own output before review saw it. When the mechanism's measured cost
  and the claimed effect disagree by an order of magnitude, the effect is wrong.

The change still stands: charging a run for the instrument observing it is a defect
whatever its size, and the phantom phase made every per-phase attribution taken with
`--phase-profile` wrong by construction. Section 21's boundary instrument and every
attribution derived from it were taken under the coupled gate and should be re-read.

### 30.4 A third vacuous test, and why this one was interesting

The first regression test passed, and then still passed when the planner was
deliberately re-coupled to phase profiling. The gate read `#[cfg(test)] { true }`, so
the production wiring was the one line no unit test could reach; the test pinned the
module's parts and not the wiring between them.

Removing the `cfg(test)` override so tests and production share one gate is what made
it testable, and the two tests that genuinely need the planner now pin the production
wiring as a side effect of asking for it explicitly. Re-coupling the wiring fails both.

Counting §28.2 and §29.5, that is four vacuous or near-vacuous tests found in this
effort. Every one of them passed when written. The only thing that separated the real
tests from the decorative ones was running the falsifier - and in §29.5 even the
falsifier misfired, which is its own lesson: a falsifier that does not fire is not
evidence the test is sound until you know *why* it did not fire.


## 31. Phase 11: the benchmark was measuring an arm that does not ship (#1231, #1234, #1241)

### 31.1 The defect

`crates/onnx-genai-bench/Cargo.toml` declared:

```toml
[[bin]]
name = "bench_generic"
required-features = ["mlas"]
```

`bench_generic` contains **no MLAS references at all**. The requirement was gratuitous.
But `mlas` is not a default feature of `onnx-runtime-ep-cpu`, so the effect was that the
only binary anyone benchmarked with was the one configuration production never builds.

Every ratio in this document up to §30 was therefore an MLAS-arm measurement, read and
acted on as though it described the shipping runtime. §29.4 spotted the harness problem
and correctly flagged it as open work; it badly underestimated the consequence.

#1231 changed the requirement to a `bench-native` feature, added an `arm=native` /
`arm=mlas-reference` label to every result line, and added `mlas_is_not_a_default_feature`
so the crate fails loudly if MLAS ever becomes default.

A feature flag is not proof. The falsifier that is proof:

```
$ nm -C target/release/bench_generic | grep -ci mlas
0        # --features bench-native   (the arm that ships)
805      # --features mlas           (the research/reference arm)
```

Every performance claim from here on is taken with that `0` verified on the same binary.

### 31.2 What it hid

Two kernels turned out to have no vectorized production path at all. Both had looked
fine for months because the MLAS arm was covering for them.

**Softmax.** The `#[cfg(not(feature = "mlas"))]` arm called scalar `f32::exp()` once per
element. `scale_mask_softmax_serial` routes through the same helper, so fused attention
paid it too. The doc comment claiming an "8-lane polynomial" described only the MLAS arm.

| model | production native | mlas reference |
|---|--:|--:|
| sm_prefill_h32_s512 | **10.012** | 0.908 |
| sm_bert_b8_s128 | **9.621** | 1.032 |
| sm_decode_h32_kv1024 | **9.302** | 1.011 |
| sm_decode_h32_kv2048 | **9.944** | 1.013 |
| sm_decode_h32_kv4096 | **10.535** | 1.028 |
| sm_decode_h32_kv8192 | **10.861** | 1.000 |

**MoE.** Worse. The shipping path allocated and filled a full transposed copy of an
expert's weight matrix *on every call, for every routed expert group*, because expert
weights are stored `[out_features][in_features]` while `gemm` wants `[K][N]`. For a
Mixtral-shaped expert with `swiglu_fusion=1` that is a 29 MiB allocate-and-transpose for
fc1 plus 14.7 MB for fc2 - to multiply a **single token row**.

| model | production native, before |
|---|--:|
| moe_mixtral_h1024_i3584_e8_t1 | **71.4** |
| moe_qwen3moe_h2048_i768_e16_t1 | **42.0** |
| moe_phi35moe_h2048_i6400_e4_t1 | **48.4** |

§18.3 had recorded every `t=1` MoE cell at 1.006-1.052 and concluded the remaining MoE
problem was purely thread scheduling at `t=8/16`. On the arm that ships, `t=1` was the
worst cell in the entire benchmark suite.

### 31.3 The fixes

**#1234, softmax.** Both non-MLAS forms now route through one
`softmax_row_core(src, dst, d)` with an AVX2+FMA `exp`: Cody-Waite split of `ln2`, a
degree-6 Cephes minimax polynomial in FMA form, `2^k` built as `(k + 127) << 23`, runtime
`is_x86_feature_detected!` dispatch, scalar fallback for non-AVX2 and for sub-8 tails.
Non-finite lanes are handled by explicit masks: NaN by unordered-compare plus `blendv`,
`-inf` and deep underflow forced to exactly `0` by ordered-compare plus `andnot`. Review
independently measured the result at **max 1 ULP** against an f64 reference across the
whole domain.

**#1241, MoE.** `gemm_bt(a, bt, c, m, k, n)` computes `C = A * Bt^T` directly on the
`[out][in]` expert slice, so the transposed copy is never materialized. That layout is
also the better one: for `C[i][j] = sum_k A[i][k] * Bt[j][k]` both operands are
contiguous along `k`, making the inner loop a pure contiguous dot product instead of a
strided gather. Interior tiles are swept column-panel-first so a ~256 KiB `bt` panel is
reused across row bands - without that, an expert bank larger than the 32 MiB L3 (the
Phi experts are 52 MiB) is re-streamed from DRAM once per row band, which regressed the
largest prefill shape until it was fixed.

### 31.4 A note on what this says about the transpose question

§19 and §20 closed the Transpose investigation on the grounds that the Transpose *kernel*
moves zero bytes in-graph, and that a layout-aware consumer was "a real cost these graphs
simply do not exercise" - so a synthetic fixture would have to be built to demonstrate it.

That conclusion was wrong, and it was wrong because it only looked at `Transpose` nodes.
The most expensive transpose in the runtime was not a node at all: it was a materialized
transpose *inside* the MoE kernel, feeding straight into a GEMM. The fixture did not need
building. It was already the single largest term in the operator, on three of the models
in this document, and it went unseen because the MLAS arm did not take that path.

### 31.5 Corrected production-native matrix

All cells: production build (`bench-native`, 0 MLAS symbols), thread-matched at
`--native-threads 1 --ort-intra-threads 1`, `native_min`/`ort_min`, models interleaved
and order-alternated across reps, min/median/max of per-rep ratios. Ratio is native/ORT,
so **below 1.000 beats ORT**. Parity PASS in every cell shown.

**Transforms, after #1234:**

| group | model | min | med | max |
|---|---|--:|--:|--:|
| softmax | sm_prefill_h32_s512 | 1.455 | 1.460 | 1.470 |
| softmax | sm_bert_b8_s128 | 1.472 | 1.477 | 1.485 |
| softmax | sm_decode_h32_kv1024 | 1.529 | 1.529 | 1.625 |
| softmax | sm_decode_h32_kv2048 | 1.613 | 1.633 | 1.805 |
| softmax | sm_decode_h32_kv4096 | 1.574 | 1.574 | 1.574 |
| softmax | sm_decode_h32_kv8192 | 1.603 | 1.622 | 1.650 |
| softmax | sm_whisper_cross | 1.289 | 1.295 | 1.300 |
| rope | rope_llama3_s1 | 1.250 | 1.250 | 1.250 |
| rope | rope_llama3_s128 | **0.759** | **0.766** | 0.802 |
| rope | rope_llama3_s512 | **0.833** | **0.861** | 0.956 |
| rope | rope_llama3_b8_s1 | **0.900** | 1.000 | 1.000 |
| rope | rope_gptj_il_s1 | 1.250 | 1.250 | 1.500 |
| rope | rope_gptj_il_s128 | **0.821** | **0.855** | 0.861 |
| rope | rope_gptj_il_s512 | **0.941** | 1.002 | 1.017 |
| transpose | tr_bert_b8_s128 | **0.569** | **0.582** | 0.616 |
| transpose | tr_llama3_s512 | 1.023 | 1.046 | 1.647 |
| transpose | tr_whisper_s1500 | **0.679** | **0.846** | 0.994 |
| kvcat | kvcat_llama3_p1023 | 1.115 | 1.179 | 1.218 |
| kvcat | kvcat_llama3_p2047 | **0.981** | 1.015 | 1.273 |
| kvcat | kvcat_llama3_p4095 | **0.986** | 1.070 | 1.161 |
| kvcat | kvcat_llama3_p8191 | **0.999** | 1.052 | 1.066 |
| kvcat | kvcat_llama3_b8_p2047 | **0.947** | **0.975** | 0.984 |

**MoE, before and after #1241** (before column is the production arm, freshly measured,
not the MLAS numbers from §18.3):

| model | before (med) | after: min | med | max |
|---|--:|--:|--:|--:|
| moe_mixtral_h1024_i3584_e8_t1 | 71.366 | **0.897** | **0.926** | 0.952 |
| moe_mixtral_h1024_i3584_e8_t32 | 17.852 | **0.752** | **0.805** | 0.806 |
| moe_mixtral_h1024_i3584_e8_t512 | 3.670 | 1.217 | 1.232 | 1.240 |
| moe_qwen3moe_h2048_i768_e16_t1 | 41.958 | **0.912** | **0.918** | 0.927 |
| moe_qwen3moe_h2048_i768_e16_t32 | 8.241 | **0.975** | 1.007 | 1.057 |
| moe_qwen3moe_h2048_i768_e16_t512 | 1.877 | 1.375 | 1.383 | 1.388 |
| moe_phi35moe_h2048_i6400_e4_t1 | 48.365 | 1.094 | 1.102 | 1.136 |
| moe_phi35moe_h2048_i6400_e4_t32 | 9.008 | 1.213 | 1.225 | 1.238 |
| moe_phi35moe_h2048_i6400_e4_t512 | 1.921 | 1.553 | 1.557 | 1.571 |

### 31.6 Which earlier tables are now known to be wrong

Any ratio in this document produced before #1231 came from the MLAS-linked binary.
Concretely:

- **§18.3's 27-cell MoE matrix** - MLAS arm. Its central conclusion, that `t=1` MoE is at
  parity and only `t=8/16` scheduling remains, was false for the shipping build. Banner added.
- **§27's 66-cell grid** - MLAS arm. Banner added.
- **§29.3's softmax table** - MLAS arm. The *direction* of that change (removing a copy at
  the shared, arm-agnostic call site) still holds, and §31.5 shows the production arm
  post-fix, but the magnitudes in §29.3 are not production magnitudes. Banner added at §29.4.
- **§21's boundary instrument** - taken under the coupled profiler gate later fixed in
  #1226, *and* on the MLAS arm. Both reasons to re-read it before relying on it.

Sections that do not depend on a ratio - the structural analysis in §13, the aliasing and
capacity work in §28, the methodology in §30 - are unaffected.

### 31.7 What this adds to the methodology rules

§30.3 established: interleave innermost, prefer `native_min` on small graphs, distrust an
effect an order of magnitude larger than its mechanism. Phase 11 adds two more.

1. **Benchmark the artifact that ships, and prove it with the linker, not the flag.**
   `required-features` silently redefined what "the CPU EP" meant for every measurement in
   this document. `nm -C | grep -ci mlas` is cheap, and it is the only statement about
   linkage that a feature flag cannot make for you.

2. **A fallback arm hides the absence of the primary one.** Neither the scalar `exp` nor
   the materialized MoE transpose was subtle. Both survived because the configuration
   under measurement never executed them. When a kernel has two arms, the untested arm is
   not "probably similar" - it is unmeasured, and here it was 10x and 70x.

### 31.8 Where the CPU EP now stands, honestly

Won at `t=1`: RoPE at s128/s512, both Transpose shapes that matter, KV concat at every
context length from 2047 up, MoE decode on mixtral and qwen3moe.

Still losing at `t=1`: **softmax, 1.29-1.63x** across the whole family - a 6-7x
improvement on #1234 but not parity, and the remaining cost is the three-pass
memory-bound structure, not `exp`. **MoE prefill, 1.22-1.56x** - microkernel efficiency
against MLAS's packed panels and wider register tiles. **phi35moe decode, 1.10x** -
bandwidth-bound, the most weight bytes per token of the three. `rope_*_s1` and
`kvcat_p1023` at 1.12-1.25x are per-call overhead on graphs too small to amortize it.

Not started: paged/appendable KV behind an attention-owned handle (§13's boundary
stands), and beam reorder, which still does not exist on any EP.

Nothing here defers to ORT and nothing links MLAS.

## 32. Phase 12: erf's Estrin reassociation, and a perf number that is hardware-dependent by nature (#1218)

The change in #1218 is sound and it landed: `erf`'s two on-path polynomials (the
big-branch `R` and the `exp` it feeds) now evaluate by Estrin's scheme instead of
Horner's, trading two extra multiplies for half the dependency depth on a
latency-bound chain. The interesting part for this ledger is what happened when a
second reviewer re-measured the speedup on different hardware, and got a different
magnitude — not because either number is wrong, but because a CPU perf result is a
property of the machine it was taken on.

### 32.1 Two measurements, two machines, no conflict

The change was measured twice, on two different microarchitectures, against two
different references. Both are reported here as peers:

| source | hardware | method | result |
| --- | --- | --- | --- |
| original (#1218) | many-core **AMD server** part | vs an ORT reference on that box | **~1.36x vs ORT** (from 1.66x) |
| reviewer | **Intel Core i7-13800H** (14C/20T, laptop-class), Windows 11 | ORT-independent same-binary A/B on `erf_avx2`; single thread; min-of-200 × 9 rounds; 1,048,576 mixed-range elements | **0.6346 ms → 0.6042 ms = ~5.0% (1.05x)** |

These are **not in conflict and neither refutes the other.** They differ on every axis
that matters for a CPU kernel timing: different microarchitecture (server AMD vs client
Intel), different core count, and different reference (one is a ratio *relative to ORT*,
the other an *internal* Horner-vs-Estrin A/B on the same binary because no ORT reference
build was available on the review host). A latency-bound reassociation win is exactly
the kind of result that scales with pipeline depth, memory system, and how much of the
work is exposed to the out-of-order engine — all of which change between those two
parts. The honest reading is: **Estrin is faster on both machines; the magnitude is
hardware-dependent**, ~5% on this laptop and reported larger relative to ORT on the
server part.

### 32.2 The general lesson: CPU perf numbers here are hardware-dependent, and must carry their metadata

This is the reusable point, and it is bigger than one PR. Server-class many-core AMD
parts and consumer Intel laptops are **different measurement environments**; the same
kernel change can be ~5% on one and a different ratio on the other, with neither being
an error. It follows that **any CPU perf number recorded in these docs is only
interpretable if it carries its hardware, its thread count, and its reference baseline.**
A bare "1.36x" or "5%" with none of that attached cannot be compared to anything later,
because the reader has no way to know whether a difference is a real change or just a
different chip.

Some entries in this benchmark corpus predate that discipline and record a ratio without
naming the host, the thread count, or the ORT reference build. This section does not go
fix them — it flags that the gap exists, so that new entries name their environment and
old bare numbers are read with appropriate caution.

### 32.3 The durable fact: this reassociation is NOT bit-identical

Unlike the perf number, this part **is** hardware-independent — it is a property of the
shipped kernel, true on every machine, so it is the most important thing to write down.
Estrin reassociates floating-point FMAs, so it **does not** produce the same bits as
Horner, by construction. A two-build dump-and-diff over 404,045 points — `[-6,6]`
sampled densely, the `0.921875` split boundary, large `|x|`, denormals, and NaN/±inf —
establishes the actual envelope:

* **Estrin differs from Horner by at most exactly one ULP.** 4,401 of 404,043 finite
  points (1.09%) differ, every one of them by a single ULP; max absolute difference
  5.96e-8. Nothing anywhere in the domain moves by more than one ULP.
* **Against f64 `erf` truth:** max error Horner 5.77e-8, Estrin 6.55e-8 (worst just
  past the split, x≈0.933). Both are within ~1 ULP of the correctly-rounded f32
  result; Estrin is at most ~0.8e-8 looser than Horner and never worse than one ULP
  from it.
* **Special values are correct and identical between the two:** large `|x|`
  (7 … 1e30, `f32::MAX`) saturate to ±1.0; denormals produce bit-identical outputs;
  NaN → NaN; +inf → +1.0; −inf → −1.0.

The tripwire is the test **`erf_reassociation_costs_no_accuracy`** in
`kernels/simd_activations.rs`: it pins the worst scaled error over `|x|<=1` at `2^-24`
(half a ULP) against `erf_avx2` directly, so a future regrouping that widens the
envelope fails the build instead of being absorbed by the looser `ERF_BOUND`. A reader
who needs to know whether erf is bit-stable should start there.

## 33. Phase 13: a CPU task runtime, because the scheduler was the loss

§26 measured the thing that costs the most and is not arithmetic: Rayon parks
its workers between parallel regions, so a fan-out costs **67 µs when it follows
another fan-out immediately and 226 µs when it follows a 20 µs gap**. Decode is
a stream of small regions separated by exactly that kind of gap. Every kernel
that fans out pays the parked number, every layer, forever.

That is a scheduler problem, and it does not get better by making the kernels
faster. This phase replaces the fan-out mechanism.

### 33.1 Two backends, chosen by where we are running

`onnx_runtime_ep_cpu::task_runtime` is a thin façade over two implementations,
and the choice is made per call:

| condition | backend |
|---|---|
| `total == 0`, forced serial, or already inside a task | `Serial` |
| a `KernelContext` with `ParallelFor` is installed | `Host` |
| otherwise, and the work splits into ≥2 tasks | `Native` |
| otherwise | `Serial` |

**Inside the plugin EP we do not run our own threads at all.** ORT has already
sized an intra-op pool from the session's `intra_op_num_threads`, and starting a
second pool beside it oversubscribes every core. `Host` routes the fan-out into
`KernelContext_ParallelFor` through the existing `host_parallel` shim, so our
kernels schedule exactly like every other ORT kernel and inherit the session's
thread count, its affinity settings and its spin policy. This is not a fallback
in the §23 sense — no node is handed to ORT's CPU EP, no kernel is skipped, and
the arithmetic is still entirely ours. Only the `for` loop is the host's.

`Native` is for the standalone runtime, where there is no host pool to borrow
and Rayon is the only alternative.

Nesting is checked in both directions (`in_host_task()`, `in_task()`). A kernel
that fans out from inside another fan-out runs serially rather than exploding
the region count, which is what made the earlier hot-pool RoPE attempt bimodal.

### 33.2 The native pool

Eight job slots, not one. A single-slot pool serialises concurrent dispatchers,
which is the wrong answer when two sessions decode at once; with eight, they get
real parallelism, and a ninth concurrent dispatcher gets `false` back and runs
its own work serially rather than blocking. Slot exhaustion is a counter
(`slot_exhausted`), so the assumption is falsifiable rather than assumed —
measured **0 in 9744 dispatches** across the concurrency harness.

Workers watch a single `epoch` counter with `atomic_wait`, and the publish order
is: write the job, store `remaining`, store `claim` (Release), set the active
bit, bump `epoch`, wake. The soundness argument is the claim protocol: a worker
that successfully claims a range has proved `remaining > 0`, which proves the
dispatcher is still blocked, which proves the closure's stack frame is alive.

**Adaptive spin, 20 µs → 500 µs**, doubling when a worker catches work in its
spin window and halving when it parks. This is the part that answers §26: the
window self-sizes to the actual gap between regions instead of being tuned to a
guess.

### 33.3 Width: SMT-capped, but only above 8 hardware threads

Whether to use both SMT siblings is not obvious for memory-bound kernels — the
second sibling adds no execution resources but does add a memory-level-
parallelism generator. So it was measured, on one binary, with one arm forced to
an explicit uncapped width:

| logical CPUs | physical | capped wins | uncapped wins |
|---|---|---|---|
| 2 | 1 | 0/4 | 4/4, by 14–19% |
| 4 | 2 | 0/4 | 4/4, by 3–25% |
| 8 | 4 | 1/4 | 3/4, by 10–26% |
| 16 | 8 | **3/4, by 12–45%** | 1/4 |
| 32 | 16 | **3/4, by 12–36%** | 1/4 |

The cap only pays above 8 hardware threads, so `SMT_CAP_FLOOR = 8`: an inferred
width is never capped below 8. Below that the sibling is still worth having;
above it, the extra spinning worker costs more than the memory parallelism it
buys. The split is not uniform even above the floor — the largest
bandwidth-bound softmax prefers SMT at *every* width, the smallest RoPE prefers
the cap at every width — which is why this is a floor and not a rule.

An **explicit** budget (`set_task_thread_budget`, `ONNX_GENAI_CPU_TASK_THREADS`)
is never capped. If a caller says 32, they get 32; the cap is an inference, and
inferences do not override instructions.

### 33.4 What it costs to dispatch

16 workers, 400 rounds per sample, release build, on the §0 host:

| gap before dispatch | p50 | p90 | | concurrent sessions | p50 | p90 |
|---|---|---|---|---|---|---|
| 0 µs | 4.8 µs | 5.2 µs | | 1 | 5.3 µs | 5.7 µs |
| 5 µs | 4.9 µs | 5.2 µs | | 2 | 5.4 µs | 5.9 µs |
| 20 µs | 4.8 µs | 5.2 µs | | 4 | 5.4 µs | 6.1 µs |
| 100 µs | 4.9 µs | 5.3 µs | | 8 | 6.3 µs | 7.6 µs |
| 500 µs | 43.8 µs | 46.4 µs | | | | |
| 2000 µs | 34.7 µs | 44.8 µs | | | | |

Against Rayon's 67 µs / 226 µs that is **14× back-to-back and 47× after a
decode-shaped gap**. The 14× matters less than the flatness: 0 µs and 100 µs
cost the same, so a decode step no longer pays for having been idle. Past the
500 µs spin ceiling the workers do park and the cost returns to Rayon's order —
which is correct, because at that point the process really is idle and should
not be burning a core.

`tests/task_runtime_latency.rs` asserts the decode-shaped part of this
(gaps ≤ 100 µs stay within 3× back-to-back + 10 µs). It is `#[ignore]`d, because
a latency assertion on a shared CI runner is a flake generator.

### 33.5 Idle cost, which is the price of spinning

A spinning pool that never parks is a regression disguised as a speedup.
`tests/task_runtime_idle.rs` bounds both ends: after the spin window elapses the
workers must be parked (bounded CPU over a 750 ms idle window) and the pool's
resident set must not grow with dispatch count.

The test is deliberately **one** `#[test]` with two phases. Two test functions
in one binary run concurrently by default, and both of these measure
whole-process CPU and RSS — so as separate tests they measured each other, and
the idle bound failed with 1.31 s of CPU over a 750 ms window.

### 33.6 Test hooks, not environment variables

`task_runtime::testing` exposes `force_serial()`, `isolated_pool(n)`,
`counters()`, `pool_width()` and `planned_backend()`. Nothing in the production
path reads an environment variable for test purposes, and the tests do not race
on a global by construction: `isolated_pool` builds a private pool rather than
reconfiguring the shared one.

### 31.7 RoPE and Softmax, moved onto it

Real ORT A/B, one driver invocation per pair, 7 trials × 7 runs. Ratio is
native ÷ ORT, lower is better, `base>new`:

| cell | t=1 | t=2 | t=4 | t=8 | t=16 | t=32 |
|---|---|---|---|---|---|---|
| rope_llama3_s128 | 0.78>0.77 | 1.74>1.34 | 2.36>1.35 | 4.61>1.61 | 7.18>1.39 | 39.40>1.15 |
| rope_llama3_s512 | 0.94>0.92 | 1.52>1.32 | 1.68>1.39 | 3.49>1.92 | 4.47>1.90 | 26.72>2.34 |
| rope_gptj_il_s128 | 0.87>0.87 | 1.72>1.28 | 2.14>1.53 | 3.78>1.54 | 6.68>1.50 | 51.31>3.40 |
| rope_gptj_il_s512 | 0.98>0.98 | 1.63>1.36 | 1.83>1.35 | 3.38>2.00 | 4.83>1.76 | 24.37>2.68 |
| sm_bert_b8_s128 | 1.26>1.29 | 2.04>1.63 | 2.12>1.67 | 3.62>1.93 | 5.56>2.58 | 3.72>2.44 |
| sm_prefill_h32_s512 | 1.33>1.34 | 2.21>2.14 | 2.33>1.45 | 2.54>2.06 | 2.54>2.23 | 5.58>1.53 |
| sm_whisper_cross | 1.36>1.38 | 2.25>2.28 | 2.50>2.58 | 3.00>2.57 | 2.54>2.79 | 1.75>1.85 |

Our own native milliseconds for the same runs, which is the honest view of what
*we* changed — the ratio also moves when ORT moves, and ORT's spread on this
shared host reaches 50% on the small cells:

| cell | t=2 | t=4 | t=8 | t=16 | t=32 |
|---|---|---|---|---|---|
| rope_llama3_s128 | 0.126>0.091 | 0.103>0.067 | 0.151>0.051 | 0.233>0.043 | 2.239>0.148 |
| rope_llama3_s512 | 0.415>0.351 | 0.242>0.193 | 0.359>0.169 | 0.382>0.128 | 2.918>0.221 |
| rope_gptj_il_s128 | 0.137>0.103 | 0.104>0.077 | 0.132>0.052 | 0.248>0.046 | 3.711>0.170 |
| rope_gptj_il_s512 | 0.429>0.402 | 0.253>0.211 | 0.354>0.212 | 0.452>0.140 | 3.505>0.251 |
| sm_bert_b8_s128 | 0.908>0.883 | 0.491>0.461 | 0.476>0.334 | 0.617>0.262 | 3.536>0.328 |

Up to **15× at 32 threads** and **5.4× at 16**. The largest wins are in the
widest columns, which is where §26 said the park cost was worst — the prediction
and the measurement agree, which is the main reason to believe the mechanism.

The decode-shaped cells are deliberately unmoved: `rope_*_s1` and
`sm_decode_h32_kv*` are below the parallel threshold and run serially in both
arms, flat to within noise at every width. A one-token RoPE should not be
fanning out and still does not.

### 31.7.1 The scar: an unrelated edit de-vectorised the activations

Rewiring RoPE and Softmax made **`Tanh` and `Sigmoid` 2× slower at one thread**,
in a file the change did not touch, on inputs that never reach a parallel path.

`avx2::map_ps` takes the per-vector kernel as `impl Fn(__m256) -> __m256` and
calls it once per eight lanes. Its body is a couple of hundred bytes of
polynomial, which sits right at the inliner's threshold, so whether it inlined
came out of a cost model with the whole crate in view — and an edit anywhere
could flip it. `nm -S` is what caught it: `tanh_avx2` went from one 511-byte
function to a 5-byte shim calling an outlined `map_ps` calling an outlined
216-byte closure. Tanh `[32,4096]` 0.053 → 0.100 ms, Sigmoid 0.054 → 0.104 ms.

Two things about this are worth keeping:

1. **`codegen-units = 1` does not prevent it.** It is already pinned for this
   package (#1174) for exactly this class of bug. One codegen unit still leaves
   the inline decision to a cost model. The fix has to be `#[inline(always)]`.
2. **The A/B grid nearly published it as a scheduler result.** The first
   activation grid showed 74 regressed cells, including at t=1 where no
   scheduling happens. The t=1 column is what made it obviously not a scheduler
   effect. Any grid that only reports the widths it is trying to improve would
   have attributed this to the runtime.

`#[inline(always)]` and `#[target_feature]` cannot coexist in Rust, which is why
neither helper carries the latter; every caller is already inside an `avx2,fma`
function, so the features arrive with the inline.
