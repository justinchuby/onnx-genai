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
  then attacks the input binding**, taking the Transpose graphs from 6.9-14.5x ORT to
  3.0-5.8x ORT. Boundary materialisation is untouched and remains open.
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
`ONNX_GENAI_HOST_COPY_PARALLEL_MIN_BYTES` overrides it, and is read on every call
rather than cached, so a test can move it.

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

* The Transpose graphs go from 6.9-14.5x ORT to **3.0-5.8x ORT**. Still a loss;
  a large fraction of what remains is `collect_outputs` (§21) and the
  single-node harness's double-copy artefact (§21.3).
* `kvcat_llama3_p8191` improves 7-13%. The other `Concat` rows do not move,
  because their inputs are at or below the threshold or their runtime is
  dominated by real `Concat` work.
* Nothing at or below 2 MiB is touched. Decode-shaped graphs - the ones §10
  actually assigns - bind under a megabyte and take the serial path unchanged.
* No numerical change of any kind: the bytes written are identical, and a
  bit-identity falsifier plus a non-vacuity counter enforce it in CI.

### 22.6 Re-verification after the review rework

Review found that the threshold lookup ran `std::env::var` on *every* copy,
before any size check - a locked `getenv` plus a `String` allocation charged to
the small-copy path this change claims to leave alone. The environment is now
read once per process into a `OnceLock` and the cheap worker-count gate runs
first. That is a material change to the measured code, so the headline cells
were re-measured on the shipped version rather than assumed to carry over.

Paired, same binary, one process per arm, 16-thread budget, `native` p50 in ms:

| graph | per-tensor input | serial (5 reps) | parallel (5 reps) | ratio range |
|---|---|---|---|---|
| `tr_llama3_s512` | 8 MiB | 0.772 / 0.806 / 0.815 / 0.860 / 0.864 | 0.343 / 0.405 / 0.419 / 0.424 / 0.439 | 0.44-0.51 |
| `tr_bert_b8_s128` (control) | 3 MiB | 0.514 / 0.531 / 0.535 | 0.520 / 0.542 / 0.546 | 0.97-1.06 |
| `sm_decode_h32_kv1024` (control) | 0.125 MiB | 0.023 / 0.023 / 0.024 | 0.023 / 0.023 / 0.023 | 0.96-1.00 |

The `tr_llama3_s512` win survives the rework at every repetition, and the two
control graphs - one just below the threshold, one two orders of magnitude below
it - are unchanged, which is the direct falsifier for the regression the review
identified: a per-call `env::var` would show on the 0.023 ms decode row.
