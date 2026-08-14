# Decision: prompt-lookup non-losslessness is near-tie fp noise in the M=K verify (root-caused)

**Author:** Deckard (Systems Dev)
**Date:** 2026-08-14
**Branch:** squad/verify-rootcause
**Status:** proposed
**Follows:** PR #932 (finding 2: M=K verify argmax != M=1 greedy argmax)

## Question
Is the speculative-decode divergence a real bug, or a fundamental numerical limit?
This gates the whole speculative lever — the eager M=K `decode_verify`
(`native_decode/cuda.rs:1243` `decode_verify` -> `decode_cuda_eager` ->
`run_cuda_eager_rows`) is the foundation any future method (EAGLE-3 / MTP) reuses.

## Method (controlled, on already-loaded qwen2.5-0.5b-int4)
GPU: **CUDA_VISIBLE_DEVICES=6** (verified idle H200; re-checked before runs).
New diagnostic bin `verify_logits_probe`:
1. Greedy M=1 decode on the native CUDA captured hot path, recording full logits
   per output index.
2. Re-run the SAME greedy continuation as the M=K draft through eager
   `decode_verify`, so every verify row has **mathematically identical causal
   inputs** to the corresponding M=1 forward. Any logit difference is therefore
   pure kernel numerics (batched M=K eager vs sequential M=1 captured).
3. Compare per-position argmax + top-k + gap + diff stats; classify.

## Evidence
Prose prompt, first divergence at **output index 27**, verify block P=24 **row 3**:
```
M=1 argmax=1817 (22.01366)   M=K argmax=6894 (21.85298)
M=1 top1-top2 gap=0.1697     M=K top1-top2 gap=0.0433
logit diff: max|Δ|=0.5066 (at one outlier token)  mean Δ=-0.0376  mean|Δ|=0.0827
M=1 top5: 1817 6894 279 264 1376
M=K top5: 6894 1817 279 264 1376   <- same set/order, only the top-2 swap
```
Repetitive prompt, first divergence output index 39, block P=36 **row 3**:
tokens 1334 vs 4718, M=1 gap 0.1024, mean|Δ|=0.0653 — same picture.

**Discriminating K-sweep (prose, 64 tokens):**
| K | flips | first divergence | note |
|---|-------|------------------|------|
| 1 | **0** | none | width-1 eager verify == M=1 exactly |
| 2 | 4 | P=26, **row 1** | |
| 4 | 8 | P=24, **row 3** | |
| 8 | 16 | P=20, **row 7** | |

Two hard facts:
- **Row 0 of every block NEVER flips** (K=1 → 0 flips). The eager-vs-captured
  kernel path alone introduces no argmax-affecting difference.
- Flips appear **only at in-block rows ≥ 1**, the first is always the **last row
  of the block**, and the count scales ~linearly with K. Across all flips the
  greedy top1-top2 gap was ≤ ~0.17 (min **0.014**); confidently-decided tokens
  never flip.

## Classification: (1) near-tie fp noise — NOT a bug, NOT a systematic offset
- Not **systematic offset**: mean Δ ≈ -0.01..-0.04 (negligible), and K=1 (which
  also uses the eager kernel) has zero flips — a real scale/bias would flip row 0
  too. Rejected.
- Not **wild / wrong**: top-5 sets are essentially identical, differences are
  0.01–0.5 logit, only the near-tie top-2 swap. No indexing/masking bug (a mask or
  position-offset bug would corrupt row 0 or produce large/wild deltas). Rejected.
- **Near-tie fp noise**: confirmed. The M=K forward computes the in-block draft
  K/V inside the batched GEMM and runs a batched attention over the current block,
  whereas the M=1 reference reads those same positions from the persistent KV
  cache written by prior single-token GEMMs. Floating-point non-associativity
  between these two paths yields ~0.01–0.5 logit differences; where top-1 and
  top-2 are within that band the argmax flips. Root cause localized to **in-block
  attention/KV numerics** in the eager M=K verify (`run_cuda_eager_rows`), not the
  captured/eager dispatch difference.

## Implication: is lossless prompt-lookup achievable?
- Byte-losslessness requires the accept/bonus decision for every committed
  position to match the M=1 greedy argmax. Position `base` already does
  (`base_logits` is M=1). Every position `> base` is decided by an M=K in-block
  row, which is not bit-exact — so the stream can diverge whenever a near-tie
  lands on an in-block row.
- The task's hypothesis ("route the last verify position through M=1") does **not**
  suffice: divergences occur on in-block rows generally (any row ≥ 1), not just
  the last, and the bonus token at any mismatch also comes from an M=K row.
- Two viable designs:
  1. **Accept it as approximate.** Speculation gives greedy-*quality*, not
     byte-identical output. Drop the "lossless" claim; document it.
  2. **Near-tie guard (cheap, restores exactness).** When an M=K verify row's
     top1-top2 gap < τ (e.g. 0.5), re-decode that single position with the M=1
     captured kernel and use its argmax as authoritative. Near-ties are rare
     (~4–9% of rows here; 8/≈180 at K=4), so the extra M=1 passes are negligible,
     and confidently-decided tokens (the vast majority) keep the batched speedup.
     This is the only path to *exact* greedy identity without serializing verify.

This is **not** a one-line fix, so no fix PR — it's a design decision (approximate
vs near-tie-guard). Flagging for the speculative owner.

## Does this break EAGLE-3 / MTP?
They reuse the **same** eager M=K `decode_verify` as the target-model verification
primitive, so they inherit the identical near-tie in-block-row divergence. For
those methods it is not a correctness bug (they are approximate speculation judged
on quality/throughput, not byte-identity), but **any "exact/lossless" guarantee
would not hold** for them either, and the same near-tie guard would be required to
make them byte-exact.

## Tooling
Added `crates/onnx-genai-bench/src/bin/verify_logits_probe.rs` (bench-only, no
engine change) — reusable to validate any future numeric-reproducibility fix.
Note: the bare `NativeDecodeSession::load_with_resolved_io` needs
`ONNX_GENAI_CUDA_KV_MAX_LEN` set for models whose dir has genai_config.json but no
inference_metadata.yaml (else "full CUDA mask reservation size overflow").
