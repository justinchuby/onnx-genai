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

---

## Update 4 — P2: split-K (widen the win) + M=1 re-run + cp.async finding

**cp.async multistage — REJECTED (measured, not assumed).** Implemented a
2-slice/128B cp.async.cg pipelined loader (all 32 lanes issue, ring-slot indexed
by `fetch % STAGES`). Passed f64-oracle parity but measured **SLOWER** across all
M (M=1 N=5120 89→114µs; N=13824 100→130µs). Root cause: the small-M bottleneck is
**occupancy** (too few blocks in flight to saturate DRAM), not per-load latency,
so hiding load latency added barrier overhead without touching the real limiter.
Reverted the whole experiment.

**split-K — LANDED (opt-in `ONNX_GENAI_MARLIN_SPLITK=1`, default OFF).** Partition
the K/group range across `grid.z`; each z accumulates its groups' `frag*scale`
into fp32 `partials[split_k,M,N]` (no bias); a reduce kernel sums the z-slices in
**fixed order** (deterministic ⇒ capture-stable) then applies fold_bias → fp16.
`choose_split_k` is conservative: only splits when the base grid is under ~2 waves
(so it NEVER regresses the already-occupied M≥32 / large-N cases), each split owns
≥2 groups, factor ≤8. New CUDA in `marlin_gemm.rs` (module untouched elsewhere);
wired into the plain + rmsnorm M>1 paths via `maybe_launch_marlin_splitk`;
partials from the capture-safe scratch pool (slot 4).

**Measured (H200 GPU 3, median µs, split-K vs direct kernel):**
| shape | direct | best split-K | speedup | weight-DRAM |
|---|---|---|---|---|
| N=5120  M=1 | 90.0 | 32.0 (k=8) | **2.81×** | 3.0% → 8.5% |
| N=5120  M=2 | 90.2 | 32.1 (k=8) | **2.81×** | 3.0% → 8.5% |
| N=5120  M=8 | 105.5 | 42.9 (k=8) | **2.46×** | 2.6% → 6.4% |
| N=13824 M=1 | 210.3 | 88.2 (k=2) | **2.38×** | 3.5% → 8.4% |
| N=13824 M=2 | 127.3 | 81.3 (k=8) | **1.57×** | 5.8% → 9.1% |
| M≥32 / large-N M≥8 | — | (heuristic returns 1) | stays direct | no regression |

split-K wins big exactly in the occupancy-bound small-M/small-N regime
(qkv/o_proj-shaped nodes during speculative verify at M=2–8), and correctly
declines where the base grid already fills the SMs.

**Gates — all intact:**
- f64-oracle parity + determinism: `marlin_splitk_parity_vs_f64_oracle`,
  `marlin_splitk_is_deterministic` (tol 2e-2, incl. asym-zp) pass.
- e2e greedy tokens **BYTE-IDENTICAL** vs tiled with `SPLITK=1` on both
  glm-4-9b-int4 and qwen2.5-14b-int4 (zp) M>1 prefill.
- SM80 guard + tiled fallback unchanged; split-K is itself SM80-gated and opt-in.
- Static-grid gemm+reduce pair, pooled partials ⇒ capture-safe warm replays.

**M=1 DRAM precondition — HONEST re-run: still fails, keep M=1 on the GEMV.**
Even the best split-K lifts the M=1 Marlin GEMM weight-DRAM only to **~8.5% peak**
(N=5120) / ~8.4% (N=13824) — far below the ≥40% floor (feasibility §3). The
full-tile MMA wastes 15/16 rows at M=1; split-K helps time but the weight-load
efficiency stays low. **Decision: M=1 remains on the existing dedicated GEMV.**

**Remaining work:** gate_up SwiGLU still uses the direct kernel at M>1 (its fused
SiluMul epilogue would need a split-K reduce variant); left as follow-up since the
plain path covers the bulk of split-K-eligible small-M nodes.

---

## Update 5 — the last two M>1 capture barriers: GQA + SkipSimplifiedLayerNorm

Sebastian's decisive A/B confirmed Marlin is a clean GEMM-level win (capture
B* 8.76×→4.99×, all 240 MatMulNBits capture seams gone, prefill ~2×,
byte-identical). The sole remaining blocker to the speculative-capture GO gate
was the last two ops still declaring `KernelCaptureUnsupported` at M>1:
**GroupQueryAttention (×40)** and **SkipSimplifiedLayerNormalization (×80)**.
Both were conservative gates, not real capture hazards. Fixed (commit
`18d00f90`), both now advertise capture support for a warmed M=K signature.

**SkipSimplifiedLayerNormalization** — *signature relaxation.* The launch is
already a static grid `(num_groups,1,1)` with no mid-kernel sync / host
read-back / alloc-free on the hot path, and the shape-keyed
`SkipBroadcastMetadataCache` already rejects any shape change (including
`num_groups`) mid-capture (the same pre-warm cold-miss valve as the Marlin
repack cache). The `num_groups == 1` capture latch was purely conservative →
relaxed to `true`. bf16 staging path still demotes on arena growth. **What made
it unsafe at M>1:** nothing — a deliberately conservative latch. **Fix:** latch
capture-safe for any pre-warmed shape.

**GroupQueryAttention** — *pre-warmed fixed-M flash workspace.* The capture
signature hard-required `q_seq == 1` (Phase2a split-K decode). At q_seq>1 the op
fell to the eager path that **host-reads-back** `seqlens_k` /
`total_sequence_length` (and rotary `position_ids`) — the actual capture killer.
**Fix:** admit a fp16/bf16 `q_seq>1` **fused-flash** signature. `flash_attention::run`
is already capture-safe by construction (static grid `batch·heads·q_tiles`,
device `total_lengths`/`past_lengths`, no host read-back, skips `synchronize`
during capture). The on-device `gqa_prepare_metadata` kernel + `capture_error_ptr`
latch replace host validation; the capture-safe branch uses `present_capacity`
(masking the KV tail on-device) exactly like the M=1 decode path. Backend is
resolved against the fixed `present_capacity` so warm and captured runs agree
deterministically. f32 M>1 (reference-scores, host-sized) stays on the eager
fallback (documented, not a hot decode/verify shape).

**Gates:** greedy tokens byte-identical on glm-4-9b-int4 (heavy
SkipSimplifiedLayerNorm) M>1 prefill; GQA + normalization lib unit tests pass;
fmt + lib clippy + native-backend clippy clean. Capture support still declared
only for a **warmed** signature — the probe must pre-warm the M=K shape before
the captured attempt (same safety valve as the Marlin repack/scratch pools).

**→ Hands off to Sebastian** for the decisive Increment-0 re-probe with
`ONNX_GENAI_MARLIN_M_GT_1=1` at M=8: expect `segments → ~1` and `B* ≤ ~2`
(speculative-capture GO). Validate byte-identical captured M=8 tokens + warm
replays. One assumption to confirm on the probe: the KV cache tail beyond the
valid length must not contain NaN (zero-init cache holds; the M=1 capture path
already relies on the same invariant).

**Per-kernel M>1 capture census (post-fix, by construction):**
- MatMulNBits ×240 (+ gate_up/rmsnorm): Marlin, capture-safe. ✅ (update 3)
- SkipSimplifiedLayerNormalization ×80: static-grid, pre-warm valve. ✅
- GroupQueryAttention ×40: fused-flash, on-device metadata, pre-warm valve. ✅
- Residual eager seams at M>1 for the hot path: NONE expected — Sebastian's
  probe is the authoritative segment census.

---

## Update 6 — Sebastian's segment census confirmed; split-K default + last dense seam closed

**Sebastian's Increment-0 re-probe of c842b759 (glm-4-9b, M=8, MARLIN_M_GT_1=1)
CONFIRMED the capture-safety landing:** `segments 120 → 1`; all 40 GQA + 80
SkipSimplifiedLayerNorm + 240 MatMulNBits M>1 capture seams GONE. Byte-identical
tokens, no NaN under capture (the KV-tail zero-init / on-device tail-mask
assumption holds). ✅

**Key measured finding — B\* is COMPUTE-bound, not segmentation-bound.**
Collapsing 120→1 segments moved the captured M=8 wall by ~0 (50.7→51.8ms, B*
4.99→5.10×). Segmentation overhead was already negligible. **Split-K is the
lever that moves B\*:** exploratory `ONNX_GENAI_MARLIN_SPLITK=1` (also
capture-safe, segments=1, byte-identical tokens) drops **B\* 5.10× → 2.69×**
(M=8 wall 51.8→27.3ms, cliff 40.5→16.5ms) — at the ≤2 GO line.

**Actions this update (commit 29714037):**
1. **Split-K default-ON within the opt-in Marlin M>1 path.** `marlin_splitk_enabled()`
   now defaults true (opt out with `ONNX_GENAI_MARLIN_SPLITK=0`). It lives
   *inside* `ONNX_GENAI_MARLIN_M_GT_1=1` (default OFF), so no default / consumer
   / edge tier is affected; `choose_split_k` still elects a split only for
   small-M / low-wave shapes (large-M prefill stays on the byte-identical direct
   kernel). This makes the canonical GO probe reach B*≈2.69 without a second flag.
2. **Closed the last M>1 capture seam — the dense `lm_head` logits projection.**
   Sebastian's census left exactly one residual node: `lm_head/MatMul_node_734`
   → `logits`, a plain fp16 dense `MatMul` (NOT MatMulNBits), which fell to
   cuBLASLt's per-call heuristic path at M>1 (M=K verify). It does not fragment
   the body (whole stack captures as 1 segment) but was the last node declaring
   `KernelCaptureUnsupported`. **Fix:** a cached-plan fast path
   (`blas::CaptureGemmPlan` + `DenseGemmPlan` in `kernels/matmul.rs`) for the
   plain 2-D (`batch==1`) M>1 GEMM — select algorithm + persistent workspace
   once at warmup, replay with no heuristic query / allocation / synchronization
   (its own workspace is never shared, so no post-GEMM sync). M>1 analogue of
   `F32GemvPlan`; reproduces `governed_gemm`'s arithmetic bit-for-bit at a fixed
   shape (same heuristic-selected algo). Batched/broadcast GEMMs keep the
   per-call path. Advertises capture after a warmed call.

**Gates:** greedy tokens byte-identical on glm-4-9b-int4 AND qwen2.5-14b-int4-zp
(`marlin_m_gt_1_matches_tiled_on_*`, GPU 5); fmt + lib clippy + native-backend
clippy clean. SM80 guard + fallback unchanged.

**→ Sebastian:** re-probe 29714037 with `ONNX_GENAI_MARLIN_M_GT_1=1` (split-K now
default; no SPLITK flag needed). Expect segments=1 (now incl. the lm_head node)
and B*≈2.69 out of the box. The lm_head being inside the captured graph removes
its per-step eager launch + sync; measure whether that nudges B* toward ≤2. The
final small increment to break-even is small-M GEMM throughput (split-K tuning),
not segmentation.

---

## Update 7 — small-M split-K retune for the M=8 verify shape (crack toward B*≤2)

Coordinator asked for "one more small-M GEMM increment" tuned to the M=K verify
shape to crack B*≤2. Profiled the glm/qwen decode-GEMM dims at M=8 on H200
(`marlin_bandwidth_microbench`, extended with glm dims + sk=16 candidate) and
found **`choose_split_k` was systematically UNDER-splitting at M=8.**

**Root cause:** the old rule ("aim ~2 waves of blocks, else don't split") uses
block count as the fill proxy. At M=8 each block is latency-bound (weight DRAM
2-6%), so the optimum is deeper K-splitting (more concurrent memory requests),
not just ~2 waves. It picked sk=2-3 and, worse, sk=1 (no split) for large-N
gate_up — leaving the biggest MLP GEMM on the slow path.

**Fix (commit 4abe4e57):** split for `m <= 32`; oversubscribe ~8 waves; floor
toward the measured optimum (8 for m<=16, 4 above) even when the base grid
already spans many waves; cap at 8 (16 regresses medium-N). Measured M=8 auto
factor before→after (weight-DRAM %):

| GEMM (glm/qwen) | K,N | old auto | new auto | gain |
|---|---|---|---|---|
| o/q proj | 4096,4096 | sk3 2.34× | **sk8 2.93×** (6.0%) | +25% |
| down_proj | 13696,4096 | sk3 2.45× | **sk8 3.18×** (7.0%) | +30% |
| gate_up (glm) | 4096,27392 | sk1 none | **sk8 1.11×** (7.2%) | new |
| gate_up (qwen) | 5120,13824 | sk1 none | **sk8 1.35×** (6.8%) | new |
| kv proj | 4096,256 | sk8 5.30× | sk8 5.30× | — |

The dominant verify cost (MLP gate_up + down_proj) now splits. Prefill (m>32)
still returns sk=1 → byte-identical direct kernel. **Caveat (honest):** qwen
square attn proj (K=5120,N=5120) is bimodal — cold-clock 0.9× / clocked-up 2.5×
(a cold-clock microbench artifact; sustained decode clocks up). It's a minor
fraction of verify cost vs the stably-winning MLP.

**Gates:** split-K f64-oracle parity + determinism pass at sk=8; greedy tokens
byte-identical on glm-4-9b-int4; fmt + lib clippy clean. `choose_split_k` only
affects m≤32, so prefill/decode are unchanged.

**→ Sebastian:** this is the "one more small-M increment." Re-probe 4abe4e57 with
`ONNX_GENAI_MARLIN_M_GT_1=1` (split-K default; sk now auto=8 for the M=8 verify
GEMMs incl. gate_up/down). Expect the M=8 verify wall below the 27.3ms that gave
B*=2.71 → B* should drop toward/through ≤2. Report the number; if it plateaus a
bit above 2, that's an honest landing (still GO for strong drafters).

---

## Update 8 — gate_up (SwiGLU MLP) routed through split-K (head 3735d57e)

**Root-cause of the residual B\* gap.** Sebastian's B\*=2.63 (on 29714037) still
ran the two LARGEST MLP GEMMs on the slow path: `try_launch_marlin_gate_up_prefill`
launched gate→gate_buf and up→output as **direct `launch_marlin_gemm`** calls,
*bypassing split-K entirely*. So the `choose_split_k` retune (update 7) only
reached o/q/kv/down — gate_up (glm K=4096 N=13696 ×2 ×40 layers; the single
biggest MLP projection) never got sk=8. That is the untapped lever.

**Fix.** Route both gate and up through `maybe_launch_marlin_splitk` so
`choose_split_k` elects the 8-way split at the M=8 verify width. gate/up reuse
the slot-4 partials scratch **sequentially** — gate's reduce fully writes
`gate_buf` before up's GEMM overwrites the partials, so single-stream ordering
keeps it correct. Both split-K `warm` flags are AND-ed into the returned warmth,
so capture-safety now accounts for split-K scratch (pre-warmed at M=K by Part-D).

**Gates (H200 GPU6):**
- `marlin_gate_up_swiglu_matches_tiled_{plain,rmsnorm,decomposed}` PASS — these
  run at M=8 so they now exercise split-K gate_up: worst_abs 0.00012 / 0.00049 /
  0.00781 (≪ tol 0.96 / 4.73 / 4.73), **warm replay byte-identical**, cold-miss
  rejected (safety valve intact).
- glm-4-9b-int4 **and** qwen2.5-14b-int4-zp e2e greedy tokens **byte-identical**.
- fmt + lib clippy + native-backend clippy clean.

**→ Sebastian:** your B\*=2.63 predates BOTH update-7 (`4abe4e57`, o/q/kv/down
retune) and this update-8 (`3735d57e`, gate_up split-K) — they crossed in flight.
Please re-probe **head 3735d57e** with `ONNX_GENAI_MARLIN_M_GT_1=1` (split-K
default). The M=8 verify wall should drop below 26.8ms since the biggest MLP GEMM
now splits. Report the decisive B\*; if it plateaus just above 2 that's an honest
GO for strong drafters — do not force it.

**Honest ceiling.** M=8 uses `mma.m16n8k16` (16-row tiles) at half occupancy →
~50% MMA waste is intrinsic to tensor-core int4 at M=8 (GPTQ-Marlin pads the
same). Split-K + occupancy is the only lever short of a non-tensor-core path
(cp.async already measured occupancy-bound/slower). Beyond this increment, the
remaining path to a universal B\*≤2 is drafting-depth amortization (Sebastian's
domain), not a capture blocker.

---

## Update 9 — FINAL: gate_up split-K measured; frozen for merge review (head dffabf0d / code 3735d57e)

Sebastian re-probed **3735d57e** (glm M=8, GPU7, ×2): **B\* = 2.15–2.19×**,
captured M=8 verify wall **stable 21.9ms**, segments=1 whole-graph, **zero
unsupported nodes**, byte-identical greedy tokens, no NaN, split-K slot-4 covered
by Part-D M=8 pre-warm (capture_alloc=(0,0), no cold-miss). The run-to-run ratio
jitter (2.15–2.19) is entirely the M=1 captured baseline (9.98–10.18ms); the M=8
wall itself is rock-stable. So the frozen head is now the *measured* commit.

**FROZEN for merge review.** Perf/capture side is DONE + GO (Sebastian concurs).
Full arc: segments **41→1** whole-graph zero-seams; capture **B\* 8.76 → 4.99
(Marlin) → 2.71 (split-K) → 2.63 (lm_head dense-plan) → 2.16× (small-M retune +
gate_up split-K)**, byte-identical throughout. B\*≈2.16 is the intrinsic small-M
GEMM floor (M=8 `mma.m16n8k16` ~50% MMA waste); universal ≤2 is a drafting-depth
story, not a GEMM-tuning one. Practical GO: any 8-wide draft accepting >2.16
tok/verify wins. Next: Chew (numerics) + Gaff (quality) review of PR #960.

---

## Update 10 — CORRECTION + qwen breadth: gate_up fusion is block-32-only (glm is block-128)

Sebastian probed qwen at 3735d57e and flagged a per-model gap; investigating the
code resolves the attribution:

**Fused gate_up SwiGLU requires `block_size==32`** (hard gate in
`run_f16_gate_up_swiglu`, matmul_nbits.rs:5782). Model block sizes:
- **glm-4-9b = block-128** → the ≥5-input fused gate_up node is NEVER formed;
  glm runs gate/up as *separate* MatMulNBits nodes through the **general** Marlin
  split-K dispatch (`try_launch_marlin_gemm`→`maybe_launch_marlin_splitk`). Those
  were already covered by the update-7 `choose_split_k` retune. **⇒ 3735d57e
  (fused gate_up split-K) is a NO-OP for glm.** glm's 2.63→2.16× was ENTIRELY the
  4abe4e57 retune; my update-8 claim that gate_up split-K drove the glm number was
  WRONG — corrected here.
- **qwen2.5-14b = block-32** → the fused gate_up node forms; `try_launch_marlin_
  gate_up_prefill` fires at M>1 (eligibility gates block≠0 / k%16 / k%block all
  pass for K=5120), and 3735d57e routes its gate/up through split-K. **Confirmed
  firing:** qwen's Increment-0 census is segments=1 / zero-unsupported — a
  None/Err fall-through would go to the NON-capture-safe tiled prefill and
  fragment the graph, so segments=1 *proves* `gate_up_swiglu_marlin_prefill` is
  the path taken. The 35.5ms M=8 wall is WITH gate_up Marlin split-K active.

**qwen M=8 verify @ 3735d57e (Sebastian, GPU7 ×2):** B\* = 4.62/4.72× (captured
M=8 35.5ms, M=1 7.5ms); segments=1, whole-graph, zero unsupported (all 48 GQA +
96 LN capture-safe — capture-safety generalizes to qwen ✅).

**Diagnosis (honest, not a bug):** qwen's higher B\* is a *denominator* effect,
not a coverage gap. Capture accelerates qwen's launch-bound block-32 M=1 GEMV 34%
(11.4→7.5ms) but the heavier 14B/48-layer M=8 verify only 18% (43.1→35.5ms), so
the RATIO inflates even though both walls drop. qwen's tuned block-32 M=1 is
genuinely fast (7.5ms < glm's 10.2ms), which raises B\* (more accepted tokens
needed to break even). caveat #1 (qwen square-attn K=5120 cold-clock bimodal) is
NOT biting — the probe warms 1000 decode steps so clocks are up; the bimodal was
a cold-clock microbench artifact.

**Net:** glm-4-9b (canonical gate, block-128) is a clean practical GO at 2.16×.
qwen (block-32) at 4.7× is an honest second-model characteristic — capture-safety
is fully solved (whole-graph, zero seams, byte-identical eager parity), the M=8
wall is compute-bound at the intrinsic small-M `mma.m16n8k16` floor, and closing
qwen's B\* is a drafting-depth story (deeper/stronger drafter amortizes the higher
ratio), NOT a frozen-kernel fix. No further kernel change; freeze stands.

---

## Update 11 — test hardening: marlin M>1 e2e parity robust to greedy near-ties (head 4803b4fc, test-only)

Sebastian found `marlin_m_gt_1_matches_tiled_on_qwen2_5_14b_int4` ~25% flaky: it
asserted exact greedy-token equality against the portable **tiled** GEMM, whose
fp32 atomic reduction order is nondeterministic. At a near-degenerate argmax (a
near-tie) the TILED reference flips run-to-run (Marlin is the deterministic
side), and greedy decode is autoregressive so one early flip cascades. glm's
prompt has no such tie → its test was already clean. **Not a Marlin bug — a test
oracle bug.**

**Fix (test-only; kernel unchanged, frozen at 3735d57e):** replaced exact-equality
with a cascade-robust classifier. Compare one tiled vs one Marlin stream; if
identical → full-strength pass (glm). If they first diverge at position `d`
(sharing the identical prefix `tokens[0..d]`), re-run the tiled config: a
prefix-matching tiled flip at `d` proves `d` is a near-tie (nondeterministic in
the tiled reference itself) → not a regression; only a divergence where the tiled
reference stays deterministic across all probes fails. The decision is a pure,
GPU-free `classify_parity()` with deterministic unit tests for every outcome
(identical / length-mismatch / near-tie-from-probe / prefix-unstable-near-tie /
real-regression), so the hard-to-reproduce branch is verified in CI without a
live tie.

**Gates:** 5 classify_parity unit tests PASS; glm + qwen e2e PASS (18+ consecutive
qwen runs, zero flakiness on idle GPU6); fmt + lib + native-backend + cuda
all-targets clippy clean.
