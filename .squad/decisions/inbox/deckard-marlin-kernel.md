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

---

### 2026-08-14 (update): Stage 3–4 landed — M>1 wired, capture support advertised

**Status:** Stage 3 (op wiring) + Stage 4 (capture support) COMPLETE and validated on
H200. M=1 stays on the existing GEMV (precondition still unmet). Head: d722705c.

**What landed:**
- **Wired Marlin into the `MatMulNBits` M>1 dispatch** (`matmul_nbits.rs`, the
  `if m > 1` seam only + the `capture_support` doc). Opt-in via
  `ONNX_GENAI_MARLIN_M_GT_1=1` (default OFF, Rule-11 tier-scoped). Eligibility:
  SM80+, int4, no fused SwiGLU/RMSNorm epilogue yet, K divisible by 16 and group.
  Any ineligibility or launch error falls through **byte-compatibly** to the
  portable tiled GEMM.
- **Module-level repack cache** (`ensure_repacked` in `marlin_gemm.rs`) keyed by
  (packed_ptr, ordinal, N, K, group). Immutable initializer weights are repacked
  on-device **once** during warmup; captured replays hit the warm cache with no
  allocation. A cold miss during capture is rejected (safety valve → caller falls
  back). Dispatch stores the cache-warm flag into `last_call_capture_safe`, so
  **cold call = not-capture-safe, warm replay = capture-safe** — this is what
  unlocks capture-stable speculative verify at M>1.
- Op-level test `marlin_m_gt_1_op_parity_and_capture_safety`: real op, flag on,
  vs f64 oracle (asymmetric zp) — **worst_abs=0.013 ≪ tol=0.67**; asserts
  cold=not-safe / warm=safe + byte-identical replay. PASSES on GPU.

**Measured — apples-to-apples wall vs tiled through the op**
(`marlin_m_gt_1_op_wall_vs_tiled`, H200, K=5120 N=13824 block=128, median µs):

| M | tiled µs | marlin µs | speedup |
|---|---|---|---|
| 1 | 69.5 | 69.0 | 1.01× (M=1 stays on GEMV) |
| 2 | 478.0 | 140.0 | **3.41×** |
| 4 | 494.1 | 138.5 | 3.57× |
| 8 | 536.4 | 176.9 | 3.03× |
| 16 | 658.1 | 251.2 | 2.62× |
| 32 | 1076.7 | 385.0 | 2.80× |
| 64 | 2010.6 | 821.6 | 2.45× |
| 128 | 3851.5 | 1357.9 | 2.84× |

**Cliff collapse (the mission payoff):** the tiled path jumps 69.5 µs (M=1 GEMV)
→ 478 µs (M=2 tiled) = a **6.9× cliff**. Marlin M=2 = 140 µs, collapsing the
M=1→M=2 cliff to **~2×** (69.5→140 µs). Marlin is **2.4–3.6× faster than the
tiled GEMM across every M>1**.

**M=1 decision (unchanged, honest):** standalone M=1 5120² is ~3.1% HBM peak and
through-op M=1 Marlin ties the GEMV (69 vs 69.5 µs) — it does NOT clear the
feasibility §3 ≥40% precondition, so **M=1 remains on the existing GEMV**. Not
silently dropped: reported.

**Next:** extend fused epilogues (rmsnorm-prologue first, then swiglu/gate_up/down)
so those M>1 entry points can also use Marlin; small-M perf levers (split-K /
cp.async) to push M=1 toward the precondition; Stage 5 e2e on glm-4-9b-int4 +
qwen2.5-14b (coordinate with Sebastian on squad/marlin-bench).

---

### 2026-08-14 (update 2): fused RMS-norm epilogue + Stage-5 e2e PASS

**Status:** Fused rmsnorm-prologue path landed; e2e parity on both target models
PASSES. Head: a3ece463.

**What landed:**
- **Fused RMS-norm prologue for Marlin M>1** (first of the fused epilogues that
  MatMulNBits must preserve). Stages the per-token normalized activation into
  scratch via the existing `launch_rmsnorm_prefill` (byte-identical to the
  standalone prologue the tiled path uses), then runs Marlin over the normalized
  rows. Scratch alloc keeps it off the capture contract (like the tiled rmsnorm
  prefill; prefill is outside the decode graph). Falls through to tiled on
  ineligibility/error. Test `marlin_m_gt_1_rmsnorm_op_parity` vs the trusted
  fused-tiled path: **worst_abs=0.0156 ≪ tol=0.90**.
- **Stage-5 e2e** (`crates/onnx-genai-engine/tests/marlin_m_gt_1_e2e.rs`, native
  CUDA): runs each real model twice through the engine — tiled (flag off) then
  Marlin (flag on) — and diffs the greedy token stream. Prefill runs MatMulNBits
  at M=prompt-length (Marlin path); decode stays on the M=1 GEMV.
  - **glm-4-9b-int4:** 24 greedy tokens **byte-identical** tiled vs Marlin.
  - **qwen2.5-14b-instruct-int4 (asymmetric zp):** 24 greedy tokens
    **byte-identical** tiled vs Marlin.

**Remaining (perf-focused, mission = full production kernel):** swiglu/gate_up/down
fused epilogues; small-M levers (split-K to fill SMs, cp.async multistage) to lift
M=1 DRAM% toward the feasibility precondition and further widen the M>1 win.
Coordinate with Sebastian (squad/marlin-bench) on the ORT e2e + capture re-probe.

---

### 2026-08-14 (update 3): P1 COMPLETE — full M>1 coverage (gate_up SwiGLU fused), zero tiled fallbacks

**Status:** Every hot MatMulNBits node in glm-4-9b and qwen2.5-14b now runs
through Marlin at M>1. Zero tiled M>1 fallbacks. Head: 1154c977.

**What landed:**
- **gate_up SwiGLU MLP fusion at M>1** (`try_launch_marlin_gate_up_prefill`,
  wired into `run_f16_gate_up_swiglu`): paired gate/up Marlin int4 GEMMs +
  the identical `launch_silu_mul_f16_raw` epilogue the tiled path uses, with an
  optional RMS-norm prologue (into pooled scratch). This is the MLP — the bulk
  of prefill/verify cost. down_proj is a plain tall-skinny MatMulNBits already
  covered by the plain `gemm_marlin_int4` path.
- **Capture-safe scratch pool** (`ensure_scratch` in marlin_gemm.rs): size- and
  slot-keyed module-global pool (FIFO cap 256, rejects cold-miss during capture)
  so warm replays of the gate_up + rmsnorm paths allocate NOTHING → capture-safe.
  `try_launch_marlin_gemm_rmsnorm` was migrated onto it (now returns the warm flag
  instead of always allocating), so the rmsnorm prefill is now capture-safe on
  warm replay too.
- **Tracer-driven per-node coverage audit** (e2e test): runs prefill with the flag
  on, tallies per-op `kernel_variant`, asserts zero `gemm_f16_tiled{,_rmsnorm}` /
  `gate_up_swiglu{,_rmsnorm}_prefill` at M>1.
- **Unit tests:** gate_up SwiGLU M>1 Marlin-vs-tiled parity + capture-safety
  (plain / rmsnorm / decomposed-silu): worst_abs 0.0001–0.008 vs tiled, warm
  replays capture-safe and byte-identical.

**Measured coverage audit (H200, GPU 7, Marlin M>1 enabled):**
- **glm-4-9b-int4:** 240 `gemm_marlin_int4`, **0 tiled M>1**. (glm's MLP is
  separate plain MatMulNBits nodes → covered by the plain Marlin path.)
- **qwen2.5-14b-int4 (asym zp):** 240 `gemm_marlin_int4` + 48
  `gate_up_swiglu_marlin_prefill` + 1 `gemm_marlin_int4_rmsnorm` = **289 Marlin
  M>1, 0 tiled**.
- Greedy token streams remain **byte-identical** tiled-vs-Marlin on both (24 tokens).

**Nodes that still can't go through Marlin at M>1:** NONE for the hot path.
Remaining non-Marlin variants observed are all legitimately M=1 decode
(`gemv_f16_*`, `gate_up_swiglu_rmsnorm_fused`) or non-MatMulNBits (attention,
rmsnorm). M=1 stays on the GEMV by design (has not cleared the ≥40% DRAM
precondition).

**→ Unblocks Sebastian's Increment-0 capture re-probe** with
`ONNX_GENAI_MARLIN_M_GT_1=1` for the decisive segments→1 and captured-M=8/M=1 B*.

**Next (P2):** split-K (fill SMs at small M) + cp.async multistage to lift the
M>1 ratio; honest re-run at the M=1 DRAM precondition (keep M=1 on GEMV unless
it truly clears ≥40% DRAM).
