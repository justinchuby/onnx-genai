### 2026-08-12: bf16 GQA capture-safe decode kernel + skip-norm capture-safety fix engage CUDA-graph capture for Muse-Glimmer native decode

**By:** Sebastian

**What:**
Added the missing bf16 capture-safe path so native CUDA-graph capture finally engages
for Muse-Glimmer (bfloat16 decoder) pipeline decode, and fixed the last eager seam.

1. **`gqa_decode_bf16` kernel** (`crates/onnx-runtime-ep-cuda/src/kernels/gqa_decode_bf16.rs`):
   a bf16 device-length split-K GQA flash-decode kernel mirroring `gqa_decode_fp16`
   (bf16 types/intrinsics, fp32 accumulation preserved, distinct NVRTC module key to
   avoid fp16 cache collision). Wired into `group_query_attention.rs` capture-candidate
   dtype gate, read-path selection, and compute dispatch, plus a `Bf16DecodeRead`
   `KvCachePath` + `decode_module_key_bf16()` in `kv_stride.rs`. This lets
   `capture_support()` admit the 52 bf16 GQA nodes that Leon's #852 pin had already
   cleared at the classifier level (they previously declined at the CUDA-EP kernel gate
   because only f32/f16 decode kernels existed). Collapsed GQA seams 54 -> 2.

2. **Skip-norm capture-safety fix** (`normalization.rs`, `SkipSimplifiedLayerNormKernel`):
   the bf16-via-f32 path uses a persistent grow-only f32 staging arena (`NormBf16Scratch`,
   mirroring `matmul_nbits::Bf16Scratch`) instead of per-call `cudaMalloc`/`cudaFree`
   (a `cuMemFree` forces a per-token stream sync -> capture-unsafe eager seam). The
   remaining bug: the first warm call *grows* the arena and the pre-capture audit reads
   `capture_support()` right after that warm call, so the `grew` demotion set the flag
   false at exactly the moment the audit sampled it. Fix: only demote on `grew` when
   `is_capturing()` (a grow racing an in-progress capture is unsafe; the first warm-time
   grow outside capture sizes the arena once and leaves the base fixed for every steady
   replay step). This eliminated the final `SkipSimplifiedLayerNormalization` seam:
   segments 2 -> 1, **0 eager seams** (whole decode step captures as one graph).

**Measured (Muse-Glimmer-30B int4, H200, --pipeline --backend native, steady, tokens 128):**
- Capture OFF: **17.35 tok/s** (58.5 ms/token)
- Capture ON:  **23.13 tok/s** (43.4 ms/token) -> **+33%** from graph capture
- Segments: 54 -> 1 captured segment, 0 eager seams
- Greedy parity preserved (ids `[24, 372, 1045, 10016, 328, 2885, 262, 5091, ...]`,
  capture-ON == capture-OFF).

**Why / next lever (important correction to the original diagnosis):**
Graph capture now engages cleanly and delivers a real +33%, but decode does **not**
reach ORT's 40 tok/s because it is **not purely launch/dispatch-bound** — with the whole
step captured, replay ~= eager would be true if it were pure dispatch, but capture still
helps +33%, and the per-op profile (`ONNX_GENAI_PROFILE_OPS=1`) shows decode is also
**kernel-bound**:
- **Cast: 40.1%** (626 calls/token) <- new dominant cost, mostly bf16<->f32 round-trips
- MatMulNBits: 21.1% (417 calls)
- GroupQueryAttention: 14.1% (52 calls)
- SkipSimplifiedLayerNorm: 9.2%

The next dominant lever is the **Cast overhead (40%)**: eliminate the bf16<->f32
round-trips on the decode hot path (native bf16 data path / fuse Cast into the
MatMulNBits + norm consumers, extending the existing `CudaFoldConstantCast` runtime-cast
fusion). That is a substantial EP graph-rewrite / kernel-io-dtype effort and is the
recommended follow-up to close the remaining 23 -> 40 gap. Flagged for coordination
(EP kernel-io + graph-rewrite; overlaps Batty's decode-graph domain).

**Dependencies:** builds on Deckard #848 (sliding_window classification), Batty #850
(native CUDA embedding), Leon #852 (GQA fixed-capacity KV seq-symbol pin) — all in main.
