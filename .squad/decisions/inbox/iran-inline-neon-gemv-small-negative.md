# Decision: Inline NEON GEMV for small shapes — negative result

**Date:** 2026-07-28T10:27:08Z
**Author:** Iran (Mac CPU Optimization Engineer)
**Status:** Concluded — change NOT shipped

## Hypothesis

Sebastian attributed ~half of the small-model performance gap (0.445× ORT)
to `cblas_sgemm` dispatch overhead (~205 ns call floor) on 64×64 GEMVs.
The proposed fix: bypass Accelerate/`cblas_sgemm` for small shapes and use
an inline NEON GEMV kernel instead.

## Investigation

### Existing dispatch analysis

All three M=1 GEMV paths in `matmul.rs` were audited:

| Path | Data type | B layout | Small-shape handler | Uses cblas? |
|------|-----------|----------|---------------------|-------------|
| `neon_gemv_f16_col_parallel` | f16 | transposed (N×K) | `neon_gemv_f16_batch` (inline NEON) | **No** |
| `neon_gemv_col_parallel` | f32 | transposed (N×K) | `neon_gemv_batch` (inline NEON) | **No** |
| `neon_gemv_parallel` | f32 | row-major (K×N) | `sgemm` → `cblas_sgemm` | **Yes** |

The f16 and transposed-f32 paths — which cover all constant-weight decode
GEMVs — **already use inline NEON** for small shapes. The only path still
going through `cblas_sgemm` at M=1 is `neon_gemv_parallel` (row-major B
without pre-transpose), which is hit only for non-constant B matrices.

### Microbenchmark (interleaved A/B, 200K iters, M1 Max, load 3–12)

Two inline NEON approaches were benchmarked against `cblas_sgemm`:

| K | N | K×N | cblas (ns) | NEON dot-T (ns) | NEON outer (ns) | dot/cblas | outer/cblas |
|---|---|-----|------------|-----------------|-----------------|-----------|-------------|
| 64 | 64 | 4096 | 158–236 | 191–356 | 197–203 | 0.66–0.83× | 0.80–1.17× |
| 64 | 128 | 8192 | 301–306 | 370–379 | 351–361 | 0.81× | 0.85–0.86× |
| 128 | 64 | 8192 | 333–340 | 330–338 | 382–390 | 1.00–1.01× | 0.87× |
| 128 | 128 | 16384 | 644–646 | 652–775 | 690–732 | 0.83–0.99× | 0.88–0.93× |
| 256 | 256 | 65536 | 1670–1700 | 4003–4274 | 4243–4518 | 0.40–0.42× | 0.38–0.39× |

**Result: cblas_sgemm is competitive or faster across all tested shapes.**
The inline NEON approaches do not beat cblas at any small shape consistently.

At 64×64, cblas achieves 158–236 ns total (not 205 ns overhead + compute),
indicating the BLAS dispatch is well-optimized on Apple Silicon for these sizes.

### Why inline NEON loses

1. **Row-major B access pattern**: For y = x @ B with row-major B[K,N],
   the outer-product approach writes N partial sums K times; the dot-product
   approach reads K elements with stride N per output. Both suffer cache
   pressure that cblas avoids with internal micro-tiling.

2. **cblas M=1 specialization**: Accelerate's `cblas_sgemm` likely detects
   M=1 and routes to an internal GEMV micro-kernel that is already NEON-tuned
   with L1/L2-optimal blocking — making the "dispatch overhead" argument moot
   at these sizes.

## Conclusion

The inline NEON GEMV optimization does not close the small-model gap because:

1. The **relevant** small-shape paths (f16 and transposed-f32) already use
   inline NEON — this was implemented in PR #227.
2. The **remaining** cblas path (row-major B) is actually faster than inline
   NEON due to Accelerate's internal M=1 specialization.
3. The small-model gap (0.445× ORT) is likely not in MatMul dispatch but in
   other ops (SDPA per-head cblas calls, elementwise ops, or session overhead).

**Recommendation**: Investigate SDPA per-head `cblas_sgemm` calls next.
During decode (q_seq=1) with head_size=64, SDPA makes 2×num_heads cblas calls
per step. For a 12-head model that's 24 BLAS calls. An inline NEON SDPA
decode path for small head sizes may be the real lever.
