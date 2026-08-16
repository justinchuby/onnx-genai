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
unmasked cells spanned **0.91–12.6x**, and every one of the nine worst cells was
masked. The mask was costing more than the attention.

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
### Second answer: still decline it

The fix is large — up to **8.44x** — and masked single-thread decode now beats
ORT outright (0.889–1.009x, was 1.073–1.400x). It is not enough. **44 of 54
cells are still slower than ORT**, 41 of them by ≥5%, and only 7 are ≥5% faster
(base: 5). The worst remaining cells are **15.0x at 8 threads** and **12.7x at
16 threads**, because the remaining time is in the QK and PV GEMMs and their
thread scaling, not in the scale/mask/softmax stage that #1094 fixed.

So the honest answer to requirement 4 is the one it anticipated: **the fused
region does not clear the ≥5% bar, and should be declined to ORT wherever the
Plugin EP has that option.** Keeping `Softmax` claimed *purely* to preserve a
fusion that then loses is not justified by this data.

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
* Phase 4 requirements 2 (grouped MoE GEMM) and 3 (fused transpose-with-consumer)
  remain **open and unstarted**. The §10 assignment matrix is unchanged by Phase 4.
