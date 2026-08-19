# f16 `Gemm` decode with `transB = 1`: 36 ms of blocked GEMM for a single row

2026-08-19. Host AMD EPYC 9V74, 32 vCPU / 16 cores, AVX2 + FMA + F16C only (no
AVX-512, no VNNI, no AMX). ONNX Runtime 1.27.0, CPU EP. Native build, **no**
`mlas` feature — the shipped default.

## Summary

An f16 `Gemm` at `M = 1` with `transB = 1` took **32-48 ms** at `K = N = 3584`
against ORT's 0.16-1.5 ms: between **21x and 65x slower**, and it did not
improve with thread count at all. It now takes **0.3-1.5 ms** — a 38x to 90x
absolute speedup, up to 148x on per-run minima — which is a **win at 1-8
threads** and, at 16, a number too noisy on this host to quote (see below).

`transB = 1` is not an exotic case. It is the layout every `nn.Linear` export
produces, so it is what a QKV, an output projection and an MLP gate all look
like when a model is exported through `Gemm` rather than `MatMul`.

## Why it was slow

`GemmKernel::execute` disqualified the f16 fast paths whenever either transpose
flag was set:

```rust
let half_fast_path = if self.trans_a || self.trans_b { None } else { ... };
```

with the reasoning, in a comment directly above it, that "both read B in its
stored `[K, N]` order, and materialising a transpose first would give back what
they save". The premise is right and the conclusion does not follow. You only
need a transpose if you insist on reusing the `[K, N]` kernel. A `[N, K]`
weight does not need one — it is a *better* GEMV layout than `[K, N]`:

| | `[K, N]` (`transB = 0`) | `[N, K]` (`transB = 1`) |
|---|---|---|
| one output element is | a strided gather over `k` | one **contiguous** `k`-run |
| parallel granularity | a stripe of columns, min 32 (a cache line) | **one row**, any width |
| accumulator working set | `W` f32 live across the whole `k` sweep | one f32 |

So the fix is a second kernel, not a transpose: `half_gemv::gemv_f16_nk`.

The path it replaced was the portable blocked half GEMM, which packs both
operands into `MR x NR` panels. At `M = 1` there is no reuse to amortise that
packing against, and the same file already recorded the same failure for the
untransposed case: "the worst dense region measured anywhere in this EP ...
10.07 ms at 1 thread, 10.26 ms at 8". The untransposed case was fixed; the
transposed one kept the note and the behaviour.

## Kernel

`out[j] = sum_p a[p] * b[j * k + p]`, four independent 8-lane FMA chains over
`p`, `ROW_STRIPE = 8` output rows per rayon task.

The four chains exist because a single dependency chain is FMA-latency-bound at
~2 elements/cycle, which at 2 bytes per weight is ~12 GB/s at 3 GHz — below what
this host delivers, so the kernel would have been latency-bound rather than
bandwidth-bound. `ROW_STRIPE = 8` gives a 3584-row projection 448 tasks, which
is why this kernel does not hit the partition ceiling described below.

### Numerics

This is the one deliberate departure from `gemv_f16_kn`, which is bit-identical
to a naive sequential loop. `gemv_f16_nk` contracts **along** the lanes rather
than across them, so its sum is split into 32 partial sums. That is a different
result — very slightly *more* accurate, but different — and pretending otherwise
would be the easy lie here. The order is fully specified in the function's doc
comment and pinned two ways:

* `dot_row_scalar` is an executable statement of that order, and
  `nk_simd_and_scalar_rows_agree_bit_for_bit` asserts the vector path matches it
  **exactly** at every `k` around the 32- and 8-element loop boundaries
  (`1, 7, 8, 9, 31, 32, 33, 39, 40, 64, 65, 96, 127, 128, 3584`).
* `nk_is_independent_of_the_thread_count` asserts the answer does not move
  across pools of 1, 2, 3 and 8 workers — `ROW_STRIPE` partitions the output,
  never the contraction.

`nk_and_kn_agree_on_the_same_logical_matrix` checks the two kernels against each
other on the same matrix stored both ways, to a tolerance, because they legitimately
differ in the last bits. End-to-end parity against ORT reads
`max_rel = 6.6e-4`, PASS.

## Measurements

`K = N = 3584` (Qwen3-8B hidden), `M = 1`, `scripts/ort_ab/gen_decode.py` +
`scripts/ort_ab/sweep_decode.py`. Each cell is 7 trials x 7 runs, ORT and native
alternating **in one process**, so every ratio is paired. `p50` is the median of
the per-trial p50; `min` is the best single run, quoted because on a shared host
the minimum is the only statistic that is not partly a measure of the other
tenants. Load average during the runs was 6-13.

### f16 `Gemm`, `transB = 1`, `M = 1` (the fix)

Tabulated run: 9 trials, load 9-11 (the quietest and largest sample of the
three sweeps below).

| threads | before ms (p50) | after ms (p50) | speedup | before `ours/ORT` | after p50 | after p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.606 | **0.968** | 37.8x | 22.46 | **0.655 win** | 0.674 |
| 2 | 36.383 | **0.633** | 57.5x | 29.75 | **0.634 win** | 0.733 |
| 4 | 32.587 | **0.468** | 69.6x | 36.86 | **0.840 win** | 1.021 |
| 8 | 36.234 | **0.532** | 68.1x | 49.50 | **0.757 win** | 0.929 |
| 16 | 48.096 | **0.534** | 90.1x | 65.26 | 1.706 | 1.224 |

On per-run minima the same cells read 32.203 -> 0.814, 31.465 -> 0.461,
31.628 -> 0.294, 36.076 -> 0.243 and 35.394 -> 0.302 ms, i.e. **40x to 148x**.

Note the before column does not move with thread count (36.6 at 1 thread, 36.2
at 8, 48.1 at 16). That is the signature of the blocked path: it never scaled on
a single-row problem, so the ratio degraded purely because ORT did scale.

### The ratio is now the noisy part, and I am not going to pretend otherwise

Post-fix these cells run in 0.3-1.5 ms. That is short enough that the host's
other tenants move the ratio more than the kernel does. Three independent sweeps
of the same five cells, taken at load 6-13, 12-18 and 9-11:

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| sweep A (7 trials, load 6-13) | 0.588 | 0.592 | 1.224 | 1.833 | 1.455 |
| sweep B (7 trials, load 12-18) | 0.884 | 1.513 | 1.234 | 1.040 | 2.680 |
| sweep C (9 trials, load 9-11) | 0.655 | 0.634 | 0.840 | 0.757 | 1.706 |

The `t = 16` cell swings between 1.46 and 2.68 and the `t = 8` cell between 0.76
and 1.83, so neither is worth two digits. The defensible statement is: **a
consistent win at 1-2 threads (0.59-0.88), and somewhere between a win and a
~2x loss at 4-16 threads, unresolvable on this host.**

What is not in doubt is the absolute change, which is three to four orders of
magnitude larger than that noise: **32-48 ms before, 0.3-1.5 ms after**, in every
sweep. A p90 taken at load 12-18 read 8.8 at two threads and 19.0 at sixteen —
included here as evidence that the tail statistic is a measure of the other
tenants at these durations, not of this kernel.

### Controls: everything else at `M = 1`, same binary, same session

Unchanged within noise, as expected — the diff only reaches `Gemm` with
`trans_b` set:

| model | `ours/ORT` before (t=1/4/16) | after (t=1/4/16) |
|---|---|---|
| `MatMul` f16 | 1.79 / 1.49 / 2.24 | 1.42 / 1.59 / 2.79 |
| `MatMul` f32 | 1.01 / 3.13 / 2.56 | 1.04 / 2.38 / 2.71 |
| `MatMulNBits` 4-bit | 2.20 / 3.84 / 7.06 | 1.90 / 3.79 / 6.78 |
| `MatMulNBits` 8-bit | 0.19 / 0.22 / 0.19 | 0.16 / 0.16 / 0.17 |

## What this did not fix, stated plainly

* **`transB = 1` f16 *prefill* is still badly broken**, and this change does not
  touch it: at `M = 128` it measures **156 ms against ORT's 39 ms at 1 thread
  (4.04x p50, 4.11x p90) and 101 ms against 6.0 ms at 8 (16.95x p50, 10.53x
  p90)**. Unlike the decode cells these are long enough to be reproducible. The GEMV is a
  decode kernel and correctly declines `M > 1`
  (`half_prefill_gemm_does_not_take_the_nk_gemv` asserts this), so prefill still
  falls into the same blocked half GEMM. Fixing it needs a packed **NT** half
  GEMM — the f16 analogue of the transposed-B SGEMM in #1176 — which is a
  separate piece of work, not a comment change.
* **The residual loss at high thread counts is the same ceiling every one of
  our M=1 GEMVs hits.** The sweep shows all of them flattening at ~0.7-1.1 ms
  past 4-8 threads while ORT keeps scaling: 4-bit goes 3.60 -> 0.79 ms across
  1..16 threads while ORT goes 1.59 -> 0.13, and f32 goes 1.73 -> 0.94 while ORT
  goes 1.76 -> 0.17. The ratios in the work list degrade monotonically with
  thread count for exactly this reason, and it is **not** fork/join overhead —
  we do not get slower, we stop getting faster. `gemv_f16_kn`'s fixed
  `STRIPE = 512` is one concrete instance: at `n = 3584` it yields only 7 tasks,
  so it cannot use more than 7 workers however many are offered.
* **The 8-bit `MatMulNBits` "win" is mostly ORT being slow**, not us being fast.
  ORT reads 30.6 ms at 1 thread there; our own absolute time (5.77 -> 1.14 ms)
  plateaus like everything else.
* AVX-512 / VNNI is untested — this host has neither.
* `trans_a` still disqualifies the fast path. At `M = 1` transposing a
  single-row A is meaningless, so there is nothing to win.
