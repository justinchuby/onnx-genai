### 2026-08-19: Route well-parallelised f32 ReduceSum/ReduceMean off cuDNN to the NVRTC block reduction (gated-delta L2-norm decode)

**By:** Gaff (CUDA-kernel / SSM-fusion)

**What:**
`ReduceKernel` (`crates/onnx-runtime-ep-cuda/src/kernels/reduce.rs`) previously
sent *all* f32 `ReduceSum`/`ReduceMean` through `cudnnReduceTensor`. It now
prefers the existing typed NVRTC block reduction (one capture-safe kernel, f32
register accumulation) for f32 whenever the reduction is well-parallelised —
`out_count >= multiprocessor_count` **or** `reduce_count <= REDUCE_BLOCK (256)` —
and falls back to cuDNN only for the low-parallelism "few outputs, huge group"
regime (e.g. a global reduce-all), where a single serialising block would lose
to cuDNN's multi-block reduce. This is the fp32 analogue of the f16/bf16 routing
#1486 already installed. The decision is a general parallelism property (SM count
+ group size), never a per-model shape/head-dim/layer hardcode.

On the qwen3.5-0.8b-hybrid **fp16io** export the gated-delta q/k L2-norm SumSq is
an f32 `ReduceSum` over `d_k` (16 outputs × 128 elements, 36×/decode step) and
was the single **largest** decode op (~11-13% of forward op time — Deckard's #1
lever). It now takes the block reduction; the retained cuDNN f32 capture path
keeps unit coverage via two new low-parallelism tests in `reduce_capture_gpu.rs`.

**Result (H200, GPU 0, ORT 1.28 CUDA build, profile_native --steady):**
- Decode: **200.95 -> 209.71 tok/s (+4.36%)**, byte-identical generated token ids.
- Golden lock `qwen35_0_8b_text_decode_lock`: **PASS** (byte-identical greedy).
- Reduce GPU tests (`reduce_capture_gpu`, `nvrtc_reduce_capture_gpu`,
  `reduce_comptype_fp16_gpu`, incl. new cuDNN low-parallelism cases): all pass.
- Matches Deckard's ReduceSum-only prediction (~252µs recovered ≈ +4.4%).

**Why:** cuDNN's generic reduce carries per-launch overhead that dominates at the
tiny decode reduce shape; the NVRTC block reduction runs it as one capture-safe
kernel with no fp32 temporary. Clean, general, byte-identical.

**Honest ceiling — NOT landed (deferred):** Deckard's sub-levers #2 (fold the
linear-attn transpose/split/concat layout shuffle into kernel addressing) and #3
(fuse the recip/sqrt/sigmoid/softplus/mul gating chain into the LinearAttention
epilogue) require a graph-level fusion pass that rewrites the surrounding ONNX
nodes into the `LinearAttention` kernel — a large change to graph optimisation
plus the kernel signature, with real byte-identity/generality risk. Out of clean
surgical scope for this PR; the ReduceSum lever alone delivered its full
predicted ROI. Transpose (~2-3%), Sigmoid (~3%), Split (~2.4%) remain as the next
targets for whoever picks up the graph-fusion work.
