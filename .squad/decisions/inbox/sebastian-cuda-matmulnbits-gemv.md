### 2026-08-13: MatMulNBits bf16 decode — cache the Float16-staged constant scales (40.21 → 47.31 tok/s)

**By:** Sebastian (Performance/Systems)

**What:** On the native CUDA decode path, `MatMulNBitsKernel::run_bf16` stages every
BFloat16 input to Float16 into an ephemeral arena on *every* decode step, then runs
the tuned fp16 GEMV. The int4 block scales are immutable weights (N × ceil(K/32)
elements per node) yet were re-cast bf16→f16 each step. For Muse-Glimmer-30B that is
~3.3 GB/token of pure-copy traffic (read+write) across 417 matmuls — ≈25% of the
13.2 GB/token int4 weight traffic — plus 417 redundant cast launches. Added a
persistent per-kernel `Bf16ConstCache` that stages the constant scale slots once
(general path: input 2; gate/up SwiGLU fusion: inputs 2 and 4) and reuses the f16
copies across steps. The per-step activation (input 0) and any per-token residual
bound into the bias slot stay on the ephemeral arena — they are genuinely dynamic,
and caching never keys on pointer identity for them (a reused activation buffer has a
stable pointer but changing contents).

**Numbers (H200, CUDA_VISIBLE_DEVICES=0, ONNX_GENAI_CUDA_GRAPH=1, --pipeline
--backend native --steady --warmups 1 --runs 3 --tokens 128):**
- tok/s: **40.21 → 47.31 median** (47.35 / 47.31 / 47.19), **+17.7%**, a clear win over ORT's ~40.
- MatMulNBits share of eager per-op decode: **~44% → ~31%**.
- Capture: **1 segment / 0 seams** (unchanged). First-16 greedy ids match the
  reference `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768,
  328, 2885, 262]` exactly; full 128-token sequence unchanged.

**Why byte-exact (no Chew gate):** bf16→f16 conversion yields identical f16 bits
whether done once (cached) or per step, and the downstream fp16 GEMV reads identical
scales, so decode output is bit-for-bit identical. Added a dedicated unit test
`bf16_scale_cache_is_bit_exact_to_inline_staging` proving the cached path is
bit-identical to inline per-call staging (and deterministic across steps). No change
to accumulation precision — the fp16 GEMV internals are untouched.

**Capture-safety:** the persistent buffer is allocated + populated on the pre-capture
warmup call (cache miss → alloc + one cast per constant); captured replays hit only
cache lookups (no alloc, no cast), so the graph stays capture-stable. Same lifecycle
as the existing `Bf16Scratch` arena.

**Follow-ups (not done here):** GroupQueryAttention is now the largest eager share
(~41%). A fuller bf16-native GEMV (bf16 in/out, no f16 round-trip at all) or the
accuracy_level=4 DP4A int8-activation path remain open levers, but both carry a
numerics cost (Chew gate) and were deferred — Muse-Glimmer's nodes declare no
accuracy_level, so DP4A is not currently exercised.
