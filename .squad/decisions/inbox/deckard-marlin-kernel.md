### 2026-08-14: Marlin int4 tensor-core GEMM — design + Stage-2 measured numbers

**By:** Deckard (Systems Dev — CUDA/decode-performance)
**Branch:** squad/marlin-kernel (draft PR, refs #957)
**Status:** Stage 2 in progress — kernel correct + validated + capture-safe grid; performance tuning ongoing.

**What:**
Landed a from-scratch SM80 `mma.sync.m16n8k16` fused fp16×int4 tensor-core GEMM in a
new module `crates/onnx-runtime-ep-cuda/src/kernels/marlin_gemm.rs` (kernel + repack
isolated from the 475 KB `matmul_nbits.rs`). It targets the M>1 path of
`com.microsoft::MatMulNBits` (today's portable 16×16 CUDA-core tiled GEMM) and,
conditionally, the M=1 decode GEMV.

Design decisions:
- **Adapt, not vendor.** Original kernel adapting Marlin's core ideas (repacked
  per-lane weight layout; **per-group scale applied AFTER the tensor-core
  accumulate** so the fp32 accumulator never carries a K-varying scale) to our
  ONNX-native format: N-major nibble packing, even-K low nibble, **asymmetric
  nibble zero-points**, group sizes 16/32/64/128. Upstream IST-DASLab/vLLM marlin
  assume symmetric GPTQ layouts and depend on `<cuda_pipeline.h>`/`crt/` headers
  unavailable to our NVRTC-string path, so a native kernel is both correct and
  simpler than translating our weights into their format. No upstream source
  copied ⇒ no third-party LICENSE vendoring required; lineage credited in the
  module header.
- **Raw `mma.sync` inline PTX** (no `<mma.h>`), consistent with the file's existing
  LOP3 asm — needs only `cuda_fp16.h` (present); `crt/mma.h` is absent here.
- **Repack** (`repack_int4_weights`): bijective reorder of packed int4 into an
  8-column-interleaved per-lane tensor-core layout so a 32-lane warp reads one
  contiguous 64-byte weight chunk per K slice. Same byte count as source packed
  (a reordering, not an expansion). It is an added packaging step; the current
  layout is untouched → two layouts coexist per Rule 11.
- **Rule 11 portability:** `device_supports_marlin()` SM80 arch guard; callers fall
  back byte-for-byte to the current CUDA-core tiled GEMM on <SM80/CPU. Marlin is
  opt-in and tier-scoped, never the default.
- **Capture-safety:** launch grid is a pure function of (M,N) with no alloc / sync /
  host-readback → capture-safe by construction (the property that unlocks
  speculative-decode capture, #957).

**Numerics (Chew's gate):** the relayout reorders partial sums ⇒ not byte-exact.
Validated against an **f64 dequant→GEMM oracle** to tolerance across M∈{1,2,7,16,33,64},
group∈{16,32,64,128}, fp16/fp32 scales, and symmetric+asymmetric zero-points
(`marlin_parity_vs_f64_oracle`, SM80+ GPU test). PASSES at abs ≤ 2e-2·max|out|
(worst rel well under 2%). Coordinated with Pris's f64 oracle harness on
squad/marlin-numerics.

**Measured (H200, `marlin_bandwidth_microbench`, group=128, fp16 scales):**
Achieved weight-DRAM bandwidth (peak assumed 4.8 TB/s HBM3e):

| shape (K×N) | M=1 | M=2 | M=8 | M=32 | M=128 |
|---|---|---|---|---|---|
| 5120×5120  | 89 µs / 3.1% | 90 µs / 3.0% | 106 µs / 2.6% | 167 µs | 491 µs |
| 5120×13824 | 100 µs / 7.4% | 101 µs / 7.3% | 148 µs / 5.0% | 358 µs | 1326 µs |

**Honest assessment:** the kernel is **correct and capture-safe** but **not yet
performance-competitive**. It is grid-starved / latency-bound at small M and narrow N
(M=1 5120² fills ~1.2 waves on 132 SMs). The M=1 lever precondition
(feasibility §3: ≥~55% weight-DRAM to beat the existing GEMV, ≥40% to land at all)
is **NOT yet met** — do not switch M=1 to Marlin yet. Next levers (in progress):
split-K to fill SMs at small M, cp.async multistage pipelining, and shared-memory A
reuse. The M>1 win vs the tiled GEMM will be measured apples-to-apples through the
op after wiring (Stage 3), coordinating with Sebastian (squad/marlin-bench).

**Why:** Primary decode-perf lever (collapse the ~67 ms M=1→M=2 cliff) and the
enabler of capture-stable speculative decoding. Building standalone-validated
(correctness vs f64 + measured bandwidth) before wiring into the op, per the staging
plan.
