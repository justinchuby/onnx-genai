### 2026-08-12: Native CUDA decode 23→40 tok/s — RMSNorm cast-fold + parallel bf16 reduction
**By:** Sebastian

**What:**
Closed the final 23→40 tok/s lever on native CUDA Muse-Glimmer decode by attacking the
RMSNorm cast round-trips *and* the RMSNorm reduction that they were hiding.

- Generalized the ep-cuda `CudaDropNormalizationCasts` pass (was fp16 +
  Skip/SimplifiedLayerNorm only) to fold **bf16** activation casts around
  **`RMSNormalization`**. Muse-Glimmer's decoder wraps all 312 RMSNorm nodes in
  `Cast(bf16→f32)→RMSNorm(f32)→Cast(f32→bf16)` (624 of 834 decoder casts). The fold
  removes both wrapper casts and retypes the norm to native bf16 I/O.
- **Op-swap `RMSNormalization`→`SimplifiedLayerNormalization` in the fold.** ONNX
  `RMSNormalization` (opset 23) defines output Y = scale type `V`, *not* activation
  type `T` — the two may differ. Muse-Glimmer's scale is f32, so the session's
  post-optimization shape re-inference (`registry.infer_graph`) kept clobbering my
  bf16 retype back to f32, breaking the kernel's `output==X` invariant and forcing a
  whole-session CPU fallback. `SimplifiedLayerNormalization` inference follows X, and
  both ops map to the *same* fused `RmsNormKernel` (no mean subtraction) on the CUDA
  and CPU EPs, so the swap is mathematically identical and re-inference-stable.
- **Parallel f32 tree reduction in `rmsnorm_bf16`** (kernels/normalization.rs). This
  is where the throughput actually comes from (see numbers). Full f32 accumulation;
  only the summation *order* differs from the serial `rmsnorm_f32` reference.

**Numbers (staged Muse-Glimmer, `--pipeline --ep cuda --backend native`, tokens 128,
steady, `ONNX_GENAI_CUDA_GRAPH=1`, H200):**
- Baseline (fold OFF): **23.16 tok/s** (43.2 ms/token), 1 segment / 0 seams.
- Fold ON, **byte-exact serial** bf16 norm: 23.43 tok/s — proves cast removal alone is
  ~free under capture (casts are cheap once launches are captured; Cast invocations
  fell 96%, 2664→104 in the profiled window, but tok/s barely moved).
- Fold ON, **parallel** bf16 norm (shipped default): **39.94 tok/s** (25.0 ms/token),
  1 segment / 0 seams. **+72% over baseline; meets ORT's 40 target.**

**Why the parallel reduction (not the cast removal) is the lever:** at M=1 decode
`num_groups=1`, so the RMSNorm reduction runs on a single block. The historical
`rmsnorm_f32` sums the 6656-wide mean-square strictly left-to-right on `tid==0`
(deliberately, to byte-match the CPU kernel). Across 312 norms/token that serial
`fadd` dependency chain is ~40% of captured decode. A parallel tree reduction removes
it. Byte-exact serial is physically capped ≈33 tok/s by that chain floor, so **40 tok/s
and strict byte-exact-vs-serial parity are mutually exclusive**.

**Numerics / parity (Chew gate):** the tree reduction is full f32 precision; a new f64
oracle test at hidden=6656 shows the bf16 kernel output is within **1 bf16 ulp** of f64
ground truth and the tree mean-square is **at least as close to f64 truth as the serial
order**. Greedy decode stays **byte-exact for the first ~37 tokens**, then shows
expected sub-ulp greedy sensitivity (accuracy-level-4 int4 MatMulNBits quantizes
activations, so a sub-ulp norm delta can flip an int8 boundary). This is a legitimate
numerics change, not a regression — flagged to **Chew** as precision reviewer.

**Escape hatch:** `ONNX_GENAI_CUDA_DISABLE_NORM_CAST_FOLD=1` routes back to the serial
`rmsnorm_f32` path for strict CPU-order byte-exact parity (at 23 tok/s).

**Open decision for coordinator:** shipping the parallel path as default relaxes the
strict byte-exact-vs-serial-reference contract (Leon/Batty's reference ids shift after
~37 tokens). The win requires it; the numerics are rigorously bounded. Recommend
accepting with Chew's sign-off; the byte-exact path remains one env var away.
