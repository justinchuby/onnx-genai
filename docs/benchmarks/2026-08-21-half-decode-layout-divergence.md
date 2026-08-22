# `f16`/`bf16` decode: three operators, one backend, three different prices

**Date:** 2026-08-21 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c x 2 SMT), AVX2/FMA/F16C, no AVX-512/VNNI.
Ledger entry: §25 of [`CPU_MATMUL_ASSIGNMENT.md`](../performance/CPU_MATMUL_ASSIGNMENT.md).
Merged as `2e1cfb67c` (#1687), closing #1381. Residual split to **#1702**.

All timings `taskset`-pinned to distinct physical cores — see the
[placement record](2026-08-21-decode-worker-cpu-placement.md).

---

## 1. The enumeration

#1381's dispatch comment claimed the divergence was already closed: "both
operators, both stored orders and both 16-bit formats reach the same GEMV
backend". Same **backend**, not same **kernel** — and the operator list was
incomplete. Enumerated with route counters rather than by reading cfgs:

| operator | stored order | kernel taken |
|---|---|---|
| `Gemm` transB=1 | `[N,K]` | `gemv_half_nk` |
| `Gemm` transB=0 | `[K,N]` | `gemv_half_kn` |
| `MatMul` | `[K,N]` | `gemv_half_kn` |
| `FusedMatMulBias` | `[K,N]` | **no 16-bit GEMV at all** |

The fourth row was found empirically. `count_half_decode_gemv()` lives in
`MatMulKernel::execute_with_backend`, and `fused_matmul_bias.rs` calls the free
`matmul_dense_prepacked_into`, so a probe over a decode step reads
`matmul_gemv=1 fused_gemv=0`. Its f16 GEMV is under
`cfg(any(target_os = "macos", target_os = "ios"))`.

The MatMul-side transposed row had **no test at all**: the `decode_matmul`
fixtures only ever built B as `[k,n]`. That is the fifth unmeasured region
behind a gate this project has recorded (cf. ledger §11, §12, §14, §19).

## 2. Root cause: `[K,N]` crosses a page every `p`

Walking a single output column of a `[K,N]` weight strides `n*2` bytes between
consecutive `p` — **12 KB at n=6144**. The L2 streaming prefetcher does not
cross page boundaries, so it cannot run ahead: every few steps it restarts.

Direct kernel A/B on identical bytes, same call, only the stored order differing:

| shape | `gemv_half_kn` (us) | `gemv_half_nk` (us) | penalty |
|---|---|---|---|
| qkv 4096x6144 | 5022 | 1688 | **2.98x** |
| down 14336x4096 | 11015 | 7068 | **1.56x** |

The penalty tracks the stride, as the mechanism predicts: qkv's 12 KB stride is
the worse of the two.

## 3. The zero-memory alternative, tried first — negative

Paying `2*K*N` bytes for a transpose is a real cost, so software prefetch was
tried before accepting it. `_mm_prefetch` at distance 12 into the strided inner
loop, on qkv:

| variant | us |
|---|---|
| `kn`, no prefetch | **5022** |
| `kn`, `_mm_prefetch` distance 12 | 5580 |

**Worse.** The stride penalty is not prefetch-recoverable here — the extra
requests add pressure without arriving early enough to matter. That negative is
what justifies spending the memory.

## 4. Accuracy moves the same way, so there is no trade

`kn` carries **one serial accumulator per column** across the whole contraction.
`nk` carries four, combined pairwise. Against an f64 oracle, `nk` is
**2.7-9.3x more accurate** across the tested shapes.

The slow side was also the less accurate side. There was no tradeoff to weigh —
which is worth stating explicitly, because a layout change that reassociates a
reduction usually does have one.

**The first version of this test was worthless and looked fine.** It built
operands from `*0.125` and `*0.0625` — exactly representable, so every partial
sum was exact — and reported **zero error and 100% bit-identity**. It could not
detect the effect it existed to measure, and it reported that confidently.
Rebuilt on hostile xorshift data, bit-identity is ~3%.

This is the same class of failure as §23's instrumented baseline: **check that
the measurement can see the effect at all before believing a clean result.**

## 5. Production-path A/B

Two builds of `benches/half_decode_gemv_ab.rs` — one from an `origin/main`
worktree, one from the branch — through the shipped routing, `taskset`-pinned,
`steady_ms`:

| dtype | shape | before | after | speedup |
|---|---|---|---|---|
| **f32 (null control)** | attn_out 1024x768 | 0.068 | 0.067 | 1.01x |
| **f32 (null control)** | square 2048x2048 | 0.094 | 0.100 | 0.94x |
| **f32 (null control)** | mlp 4096x11008 | 6.605 | 6.628 | 1.00x |
| **f32 (null control)** | lm_head 896x151936 | 14.996 | 14.833 | 1.01x |
| f16 | attn_out 1024x768 | 0.063 | 0.028 | **2.25x** |
| f16 | square 2048x2048 | 0.314 | 0.058 | **5.41x** |
| f16 | mlp 4096x11008 | 2.867 | 1.791 | **1.60x** |
| f16 | lm_head 896x151936 | 8.553 | 7.089 | **1.21x** |
| bf16 | attn_out 1024x768 | 0.077 | 0.026 | **2.96x** |
| bf16 | square 2048x2048 | 0.320 | 0.058 | **5.52x** |
| bf16 | mlp 4096x11008 | 2.782 | 1.719 | **1.62x** |
| bf16 | lm_head 896x151936 | 8.635 | 7.064 | **1.22x** |

**The f32 rows are the null control** and move 0.94-1.01x, which is the evidence
that the harness and the host are quiet and that nothing outside the 16-bit
routing changed.

**Read the small numbers, not the big one.** `square 2048x2048` at 5.41x/5.52x
is a shape nobody runs. The large model-shaped rows are 1.2-1.6x. `lm_head` —
which gains **least**, at 1.21x — is also the row that pays **most**: 272 MB
resident for one transposed weight.

`max_rel` is unchanged on every row. Two bf16 rows keep a bit-identical digest:
bf16's coarser mantissa rounds the reassociation to the same bits on that data,
which is expected and is not evidence that the reassociation did not happen (see
§4).

## 6. The memory-plan coupling — the part that could have gone badly

`node_weight_transpose_cache_bytes` is the predictor `engine/load.rs` budgets
against under #1056. For `MatMul` it was `cfg(any(target_os = "macos",
target_os = "ios"))`.

On x86 this transpose would therefore have been **completely invisible to the
memory plan** — gigabytes of retained weight buffers that the plan did not know
about, on the most common operator in the graph. The predictor was rewritten so
the x86 case is predicted (`Gemm` transB=0, and `MatMul` with a constant 16-bit
B, at `2*numel`), with the Apple arm preserved.

**Rule this establishes: any change that makes a kernel retain a weight-scaled
buffer must update that predictor in the same commit.**

`FusedMatMulBias` is deliberately **excluded** from the predictor, because it
takes no x86 16-bit GEMV — budgeting it would over-reserve `2*K*N` for every
fused projection that never allocates one. That exclusion **inverts the moment
the GEMV is enabled**; #1702 carries the warning.

Two pre-existing guard tests encoded the opposite decision ("a transposed
variant would cost a permanent `2*K*N` bytes"). They were not deleted. They now
assert the stronger invariant they were reaching for: no **unbudgeted** copy,
and never an f32 widening, checked against the predictor itself.

## 7. Admission is no longer numerically neutral

Declining the transpose cache now changes **which kernel runs**, and therefore
output bits. Three contract comments still asserted neutrality — including one
that described the exact opposite of the call directly beneath it. All three
corrected. Adversarial review caught that the commit message disclosed this but
the comments a maintainer actually reads did not.

## 8. A real bug, introduced and fixed here

`WEIGHT_TRANSPOSE_F16` stores raw `u16` keyed `(addr, k, n)` with **no dtype
discriminator** — safe only while exactly one dtype used it.

Routing bf16 through the same cache let a bf16 weight hit an f16 entry left
behind at a recycled address: `-0.000021640852 != -0.8984375`. It reproduced
**only in company** — never when the test ran alone — because it needs a prior
allocation to recycle.

Guarding the *view* dtype is not sufficient; the **key** needs the
discriminator. Fixed by adding `tag: u16` to `WeightTransposeKey`, with
`the_two_16_bit_formats_do_not_share_a_cache_entry` as the regression.

Related standing trap: tests depending on the cache verdict must use
`weight_transpose::CacheEnabledScope` (thread-local RAII), never
`set_cache_enabled` (process-global) — the documented #983/#1033/#1056 "passes
alone, fails in company" failure.

## 9. Still divergent, deliberately

`FusedMatMulBias` takes no 16-bit GEMV on x86: **2845 us** on qkv against
`MatMul`'s **1830 us** after this change, same weights and shape. It is a
separate mechanism with its own memory-plan consequence and a real bias-epilogue
difference, so it is not folded in here. Split to **#1702**.
