### 2026-08-13: Cheap-node graph fusion is NOT a decode lever under CUDA-graph capture — removing 208 byte-exact constant `Add` nodes/token does not help (confirms & extends Sebastian #870)

**By:** Batty (Decode-graph / pipeline)

**What:**
Coordinator-assigned to reduce the captured CUDA-graph node count via graph-rewrite fusion (Sebastian's #870 handed me this lever after finding decode is node-dispatch-bound at 2568 nodes/token). I measured the captured node histogram, then implemented and A/B-tested the single largest *byte-exact, no-kernel* node reduction available: folding the per-token constant `Add(Cast(weight_bf16→f32), 1.0)` RMS-norm "weight+1" scale expression into a resident f32 initializer (`CudaFoldConstantAdd`). It removes **208 Add nodes/token** (4 of the 6 norms/layer × 52 layers). It is byte-exact (full 128-token sequence identical) and preserves capture at 1 segment / 0 seams.

**Conclusion: it provides zero throughput benefit and in fact slightly, reproducibly regresses.** This is the same signature Sebastian found with `split_fill=1` (−52 nodes → *worse*): reducing the count of *cheap, off-critical-path* nodes does not realize the ~8.17 µs/node average, because that average is dominated by the expensive serial nodes (417 MatMulNBits int4 GEMVs + 52 GQA), not the tiny elementwise glue. **I am not shipping the pass** (a default-on regression, or default-off dead code, is churn that risks parity for zero gain). This drop carries the finding.

**Evidence (all H200, CUDA_VISIBLE_DEVICES=0, staged Muse-Glimmer int4, `--pipeline --backend native --ep cuda`, capture 1seg/0seams, full-sequence greedy ids byte-identical to reference `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262]`):**

- **Captured node histogram (per token, post-optimization, `ONNX_GENAI_PROFILE_OPS=1`):** MatMulNBits **417**, SimplifiedLayerNormalization **312**, Add **311**, Mul **210**, Reshape **208**, Sigmoid **104**, GroupQueryAttention **52**, Cast **2** (already folded from 834 by #860). Of the 311 Adds, ~208 are the constant `weight+1` scale expressions and ~103 are the true residual adds (2/layer).
- **A/B on the SAME binary** via `ONNX_GENAI_CUDA_DISABLE_CONST_ADD_FOLD=1` (isolates only the fold; GPU 0 idle, interleaved trials):
  - **Captured (`ONNX_GENAI_CUDA_GRAPH=1`)** — DISABLED/baseline: **47.13 / 47.28 / 47.17** tok/s (median 47.17); ENABLED/fold: **45.96 / 45.80 / 45.85** (median 45.85). → **−2.8%, reproducibly WORSE**, with Add 311→103 confirmed and byte-exact output.
  - **Eager (`ONNX_GENAI_CUDA_GRAPH=0`)** — DISABLED/baseline: **36.72** tok/s; ENABLED/fold: **36.14**. → also flat/slightly worse. The fold does not even help the launch-bound eager path here.
- **Signature:** removing 208 byte-exact cheap nodes/token moves tok/s the *wrong* way on both paths. Combined with Sebastian's `split_fill=1` (−52 nodes → worse) and his loop-cheapening being flat, the fingerprint is clear: **decode is bound by the expensive serial GEMV/GQA critical path, not by the count of cheap elementwise/norm glue nodes.** The 8.17 µs/node figure is an *average* over a heavily skewed distribution, not a savable per-node cost for tiny ops that overlap or sit off the critical path.

**Suspected mechanism (unconfirmed):** before the fold, each norm's f32 gamma is produced by the `Add` immediately preceding the norm's read (warm in L2); after the fold, gamma is a cold resident f32 initializer. Resident footprint is ~unchanged (the f32 weight already existed post cast-fold), so the small regression is most likely cold-read / scheduling perturbation rather than added traffic. Not worth chasing — the point is the *absence of any win*.

**Not pursued (and why):**
- **Target B — residual `Add` + `SimplifiedLayerNormalization` → `SkipSimplifiedLayerNormalization` (~104 nodes):** the existing `SkipSimplifiedLayerNormKernel` (`run_bf16_via_f32`) is capture-safe, but the fusion is **NOT byte-exact** — the standalone path rounds the residual sum to bf16 *before* the norm reads it, whereas the skip kernel normalizes the un-rounded f32 sum. That requires a **Chew** numerics gate + oracle coverage. Given the direct evidence above that cheap-node removal yields no win, spending a parity-risk budget on another ~104 cheap nodes is not justified. Deferred unless the QKV result below proves the count of *expensive* nodes is the lever.
- **Targets (b)/(c) — mask/position/rotary elementwise chains and redundant Cast/reshape glue:** these are also cheap off-critical-path nodes; same expected null result. Casts are already down to 2/token (#860).

**Recommendation — the one UNTESTED graph-side lever worth a kernel-coordinated experiment:**
Reduce the count of *expensive* nodes, not cheap ones. Concretely: **fuse the 3 per-layer QKV projections (`q_proj`/`k_proj`/`v_proj`, all reading the same input-norm output) into a single `MatMulNBits`** by column-concatenating their int4 packed weights / scales / zero-points along N, then splitting the output. This removes **~104 expensive GEMV launches/token** (2/layer × 52) and is **byte-exact** (each output column is computed identically). It is the decisive **dispatch-vs-bandwidth** test:
- if decode is truly dispatch-bound → ~104 fewer 8µs launches ≈ 0.85 ms/token ≈ 47→~49.7 tok/s (real win);
- if it is int4-weight-bandwidth-bound → flat (the 3 GEMVs read disjoint weights; fusing does not cut bytes).

This is graph-side (my domain: weight concatenation + output split in `optimizer.rs`, reusing the existing `MatMulNBits` kernel — **no new kernel**). **@Sebastian** — if you want the fused QKV output consumed directly by a GQA/qk-norm epilogue to avoid the split nodes, that's a kernel-side coordination; otherwise I can do it purely graph-side. I'll pick this up next unless the coordinator redirects.

**PR:** `perf(cuda-ep): constant weight+1 Add fold removes 208 nodes/token but is not a decode lever (byte-exact, 47.2→45.9) — doc-only`. No code shipped (mirrors #870).

**Note on profiling:** hardware profilers still blocked in-sandbox (ncu absent; nsys "Creating threads in this process is forbidden by design"; RmProfilingAdminOnly=1). All numbers from the built-in op timer + `ONNX_GENAI_PROFILE_OPS` node counts + capture-safe env-gated A/B on a single binary.
