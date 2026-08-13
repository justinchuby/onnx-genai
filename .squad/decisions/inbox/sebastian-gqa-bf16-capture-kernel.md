### 2026-08-13: bf16 capture-safe GQA device-length decode kernel (Muse-Glimmer 54→2 segments)

**By:** Sebastian

**What:**
Added `gqa_decode_bf16` — a bfloat16 device-length split-K GQA flash-decode kernel —
so native CUDA-graph capture admits Muse-Glimmer's 52 bf16 GQA nodes at the CUDA-EP
kernel gate. Standalone kernel PR (branch `squad/gqa-bf16-capture-kernel` off origin/main).

- `crates/onnx-runtime-ep-cuda/src/kernels/gqa_decode_bf16.rs`: mirrors `gqa_decode_fp16`
  (`q_seq==1`, `k_seq<=1`, fixed-capacity aliased device-KV, Phase2a backend) with bf16
  types/intrinsics (`__nv_bfloat16`/`__nv_bfloat162`), **fp32 accumulation preserved**
  (matmul + softmax in fp32; bf16 only at load/store boundaries), and a distinct NVRTC
  module key to avoid fp16 cache collision.
- Wired into `group_query_attention.rs`: bf16 branch in the `capture_candidate` dtype gate
  (gated on `gqa_decode_bf16::supported(q_seq, dim)`), read-path selection, and compute
  dispatch; `capture_support()` decline message updated to f32/fp16/bf16.
- `kv_stride.rs`: `KvCachePath::Bf16DecodeRead` + `decode_module_key_bf16()`.

**Numerics (accuracy gate — Chew is the standing precision reviewer):**
The parity test `bf16_decode_kernel_matches_reference_softmax_at_short_and_long_context`
compares the kernel against an **f64-accumulated softmax oracle** fed the same bf16-rounded
inputs, so the only residual is the kernel's bf16 output rounding + fp32-vs-f64 reduction.
Measured **max_abs=1.953e-3, max_rel=3.888e-3**, within the justified bounds (abs<2e-2,
rel<1e-1; bf16's 8-bit mantissa is ~8x coarser than fp16, ~2^-8·|out|). Byte-exact greedy
parity preserved (ids `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921,
768, 328, 2885, 262, ...]`, capture-ON == capture-OFF).

**Measured (Muse-Glimmer-30B int4, H200, --pipeline --backend native, steady, tokens 128):**
- Captured segments: **54 → 2** (1 residual eager seam: the bf16
  `SkipSimplifiedLayerNormalization` warmup-signature node 2452).
- Throughput: **22.52 tok/s** (44.4 ms/token), up from the ~14.5 baseline.

**Why / follow-up:**
This is the kernel prerequisite. The 1 residual `SkipSimplifiedLayerNormalization` seam is
removed by a stacked follow-up (skip-norm bf16-via-f32 capture-safety fix) that takes it to
2→1 segment / 0 seams and 23.13 tok/s (+33% vs capture OFF). Decode remains kernel-bound
(per-op: Cast 40%, MatMulNBits 21%, GQA 14%), so closing the gap to ORT's 40 tok/s needs the
Cast-round-trip elimination tracked separately.

**Dependencies:** builds on Deckard #848, Batty #850, Leon #852 (GQA fixed-capacity KV
seq-symbol pin) — all in main.
