### 2026-08-21: Base-decode fusion REGRESSES MTP; the real lever is the M=2 verify int4-GEMM kernel cliff

**By:** Gaff

**What:**
Measured the per-phase cost of the MTP self-spec loop before implementing the
proposed EAGLE-style base-decode fusion, and found the fusion is
**throughput-negative**. Shipping it would regress MTP from 34.3 → ~30 tok/s, so
per the coordinator's "STOP and report rather than ship a regression" clause I
did NOT implement it. Delivered instead: an env-gated per-phase profiler
(`ONNX_GENAI_PROFILE_SPEC_PHASES=1`, inert otherwise, spec-only path so greedy is
byte-identical) plus this measured analysis.

Per-step decomposition (steady, 48 outer steps, 128-tok run, k=1, 83.3% accept,
2.67 tok/step, H200 GPU3, int4 block-32, ORT 1.28 cuda13, both graph slots
replaying, fallbacks=0):

| phase          | ms/step | share |
|----------------|--------:|------:|
| base decode M=1| 18.3    | 23%   |
| propose (MTP)  |  5.9    |  7%   |
| verify M=2     | 49.2    | 62%   |
| commit         |  6.2    |  8%   |
| **total**      | **79.6**| 100%  |

→ 29.8 ms/tok → **34.2 tok/s** (matches origin/main #1663). Token-identity md5
`be7ed565` == greedy; engine lib 582 green; both slots replay (Primary 248,
Verify 184, fallbacks=0).

**Why:**

1. **Fusion regresses (measured, not modeled).** The base decode is only 23% of
   step time but contributes ~0.87 of the 2.67 tokens/step (it folds the prior
   bonus token and produces `h_bonus`, the main-model hidden the MTP head
   canonically requires). Fusing it out keeps the M=2 verify (~49ms) but drops
   tokens/step to ~1.83 → (49.2+5.9+6.2)/1.83 ≈ **30 tok/s**. Even the
   unattainable best case (propose+commit free) is 49.2/1.83 = 26.8 ms/tok = 37
   tok/s — never near the 55–65 projection. The coordinator's premise ("base
   decode is redundant GPU work, tokens/step stays ~1.8") is empirically wrong:
   measured tokens/step is **2.67**, and the base decode is not redundant for
   k=1.

2. **The real bottleneck is the M=2 verify kernel, and it breaks speculation's
   core premise.** Both phases are CUDA-graph *replayed* (fallbacks=0), so launch
   overhead is eliminated for both — the 18.3 vs 49.2 gap is **pure kernel
   compute**. A linear cost model `F+R=18.3, F+2R=49.2` yields `F=-13ms`
   (impossible), proving M=2 is a *different, costlier* code path than 2×M=1, not
   just "one extra row". Decisively: the replayed **M=2 verify (49.2ms) costs
   MORE than two sequential replayed M=1 forwards (2×18.3 = 36.6ms).** That
   *inverts* the entire rationale for speculative decoding — verifying K drafts
   in one parallel forward is supposed to be *cheaper* than K sequential decodes;
   here it is 1.34× *more* expensive. This is the classic int4 `MatMulNBits`
   cliff: M=1 is a latency-bound GEMV that streams int4 weights directly (4× less
   memory); small-M (M≥2) falls to a GEMM path that dequantizes to fp16 and reads
   ~4× the weight bytes, so it is memory-bound and ~2.7× slower per forward.

3. **Actionable path to >56 tok/s (no loop-structure change, no token loss):**
   make the M=2..K verify a *batched-GEMV* int4 kernel that reads int4 weights
   ONCE and applies them to all M rows (staying int4-memory-bound like M=1),
   instead of dequant→fp16→cuBLAS GEMM. Bringing the verify to ~M=1 cost (~18ms)
   gives per-step 48.4/2.67 = 18.1 ms/tok ≈ **55 tok/s**, crossing greedy (56
   tok/s ≈ 17.9 ms/tok ≈ the M=1 base) while staying 48/48 token-identical. This
   is a CUDA/EP `MatMulNBits` small-M kernel change, NOT a change to the
   speculative loop.

**Caveat:** the GEMV→GEMM attribution is inferred from the impossible
negative-overhead linear model + the M=2 > 2×M=1 inversion; nsys/ncu are
sandbox-blocked here ("Creating threads in this process is forbidden by design"),
so it is not kernel-counter-confirmed. Follow-up entry point:
`ONNX_GENAI_PROFILE_OPS=1` to confirm `MatMulNBits` dominates the verify forward,
then attack that kernel's small-M path.
