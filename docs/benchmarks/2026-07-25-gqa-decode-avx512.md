# GQA decode AVX-512 investigation — negative result (2026-07-25)

**Author:** Deckard (Systems Dev)
**Branch:** `perf/gqa-decode-avx512` (off `main` @ `540c310`)
**Host:** dual-socket Intel Xeon Platinum 8480C (Sapphire Rapids), 96 cores / 2 NUMA
nodes; `avx512f avx512bw avx512cd avx512dq avx512vl avx512vbmi avx512ifma` present.
**Toolchain:** `rustc 1.97.0`.

## Summary

**Adding a wider AVX-512 reduction to the GQA decode QK-dot is *not* shippable:
it changes the greedy token IDs versus ORT on a real int4 model, and the only
token-safe part of the change (the P·V AXPY) yields no speed-up.** The AVX-512
code was implemented, validated, benchmarked, and then **reverted**; this note
records the evidence so the lever is not re-attempted without new information.

The premise for the task was "our GQA decode kernel is ~1.7× slower than ORT's
fused CPU GroupQueryAttention because the QK dot and P·V AXPY are AVX2-only,
while ORT uses AVX-512." The micro-benchmark below refutes the *wider-SIMD*
half of that hypothesis: AVX-512 buys ≈5 % on the QK-dot at long context and
**nothing** on the AXPY — nowhere near 1.7×. Whatever gap ORT enjoys comes from
its fused, flash-style, score-free kernel structure, not from 16-wide vs 8-wide
f32 vectors.

## What was implemented (and reverted)

- `has_avx512()` runtime detection (`avx512f && fma`) in `backend.rs`.
- `dot_avx512f` — 16-lane QK dot, two accumulators (32 elems/iter) + 16-wide
  remainder + scalar tail, `_mm512_reduce_add_ps`.
- `axpy_avx512f` — 16-lane P·V AXPY, per-lane FMA + scalar tail.
- Runtime dispatch in `dot_f32` / `axpy_f32` preferring AVX-512 → AVX2 → scalar.

All of this **compiled and ran on the pinned `rustc 1.97.0`**: the AVX-512F
intrinsics (`_mm512_loadu_ps`, `_mm512_fmadd_ps`, `_mm512_reduce_add_ps`,
masked loads/stores) are stable — **the toolchain was never the blocker.**

## The blocker: token-exactness vs ORT

The team bar is token-ID exactness (argmax preserved) vs the scalar reference
**and** ORT on the real model. Greedy, `qwen2.5-0.5b-int4-onnx` (121
`MatMulNBits` + 24 `GroupQueryAttention`), prompt `"The capital of France is"`,
64 decode tokens, CPU:

| path | token[9] | matches ORT? |
|------|----------|--------------|
| **ORT GenAI 0.14.1** (reference) | `3146` | — |
| native AVX2 (baseline `main`)    | `3146` | ✅ exact |
| native AVX-512 QK-dot            | `1879` | ❌ diverges |

The existing **AVX2 path is already token-exact with ORT.** The AVX-512 dot's
16-lane reduction regroups the f32 additions just enough to flip a near-tie
greedy argmax at decode token 9 (`Paris. It is the largest city in the …` —
`world`/`country` tie), so it fails the bar. Each binary is internally
deterministic (two identical runs agree); the divergence is purely the
reduction grouping.

Why the AXPY is *not* the culprit and *is* token-safe: P·V AXPY is element-wise
(`out[d] += p·v[d]`), no cross-lane reduction, so `axpy_avx512f` is bit-identical
to `axpy_avx2_fma` for any `head_dim`. Wiring **only** the AXPY to AVX-512
(dot on AVX2) reproduces the ORT token stream exactly (`token[9]=3146`).

## Micro-benchmark (why it isn't worth it)

GQA decode-row inner loop, `head_dim = 128`, median-of-3 ns/row, `taskset -c
0-47`, 1-min load < 10 at start:

| kv_len | avx2 ns | axpy512 ns | axpy spd | full512 ns | full spd |
|-------:|--------:|-----------:|---------:|-----------:|---------:|
|     64 |  1266.4 |     1476.5 |  0.858×  |     1427.3 |  0.887×  |
|    128 |  2851.4 |     2854.8 |  0.999×  |     2813.4 |  1.014×  |
|    256 |  6860.8 |     7051.7 |  0.973×  |     6787.3 |  1.011×  |
|    512 | 11227.3 |    11227.7 |  1.000×  |    10801.1 |  1.039×  |
|   1024 | 22267.4 |    23560.5 |  0.945×  |    21150.3 |  1.053×  |

- **axpy512 (token-safe): 0.86–1.00× — no gain.** The P·V AXPY streams V rows
  and is memory-bandwidth-bound; 16-wide FMA doesn't help and adds overhead.
- **full512 (dot+axpy, token-unsafe): up to 1.05× at long context** — the whole
  win is the QK dot, i.e. exactly the path that breaks token-exactness.

So the token-safe subset buys nothing, and the beneficial subset is
disqualified.

## End-to-end (inconclusive by construction)

- The correct baseline 0.6B artifact is the int4-quantized
  `Microsoft/qwen3-0.6b-generic-cpu-4/v4/model.onnx` (524 MB, weights embedded):
  **197 `MatMulNBits` + 28 `GroupQueryAttention` + 57 `SimplifiedLayerNormalization`
  + 56 `SkipSimplifiedLayerNormalization` + 28 `Sigmoid`/`Mul` (SiLU)**. The clean
  post-#154 decode profile is **MatMulNBits 51 % / GQA 24 % / norm+elementwise
  20 % / glue 1 %** (Ripley), and the ORT-vs-native per-op diff attributes
  **Attention+norm/elementwise = 61 %** of the native-vs-ORT gap versus only 34 %
  for MatMulNBits. So attention is the #1 lever, but the GQA cost is a *structural*
  kernel-shape difference, not SIMD width (see the micro-bench above: widening the
  dot buys ≈5 %, the AXPY nothing).
- Steady end-to-end tok/s varied run-to-run by ±30 % under shared-host load (other
  users' `ncu`/`clamscan`), and a ≈5 %-on-the-dot change to a component that is a
  minority of decode time is **below the measurement noise floor** — so no honest
  end-to-end delta can be claimed for this reverted change. Reported for
  transparency, not as a result. (An earlier probe against an unrelated 1.2 GB
  dense-fp16 Qwen3-0.6B export under a contaminated host produced a nonsense
  ~1.4 tok/s; that file is **not** the baseline model and its number is discarded.)

## Decision

Keep the QK-dot and P·V AXPY on the existing AVX2+FMA path. Do not re-attempt a
wider AVX-512 reduction for GQA decode purely to widen SIMD:

1. it is not token-exact with ORT (proven divergence), and
2. the token-safe portion (AXPY) shows no speed-up (bandwidth-bound).

If the ORT GQA-decode gap is revisited, target ORT's *kernel structure*
(fused/flash-style single-pass softmax without score materialization, blocked KV
streaming) rather than SIMD width — but note the online-softmax rewrite carries
**higher** argmax risk than the reduction-grouping change already shown to flip a
token here, so it must be argmax-validated before any adoption. Because the gap
is structural (attention+norm/elementwise ≈ 61 % of the native-vs-ORT gap), the
real attention lever is a flash/online-softmax GQA decode rewrite validated
against the **f64 scalar reference** (not ORT), which is tracked separately.
