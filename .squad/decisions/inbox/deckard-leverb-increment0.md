### 2026-08-14 — Lever B Increment-0: DECISIVE NO-GO — promote Lever A (Marlin) to primary

**By:** Deckard (CUDA/decode performance engineer) · branch `squad/leverb-increment0-capture`

**What:** Built the cheap Lever B **Increment-0** capture-enablement overlay (test-only, `#[ignore]`d, UN-WIRED) — the three fixes the Phase-0 (a)-FAIL identified — and re-ran the probe on glm-4-9b-int4 (H200) to get the ONE decisive number: does a *captured* M=K verify replay cost ≈ a *captured* M=1 replay (cliff was dispatch → Lever B GO), or does the ~80 ms eager floor persist under capture (cliff is compute → Lever B NO-GO)?

Increment-0 fixes (all validated to work):
1. Persistent padded `[1, K_max, vocab]` logits device binding (absorbs the M=K logits output).
2. Alloc-free captured region via a pre-capture warm forward at the M=K shape (grows the scratch arena before `BeginCapture`).
3. KV-symbol pin — inherited from the constructor.

Measured (reproduced 3×, deterministic):
- **(a) instantiates capture-safe — now PASS.** INC0 M=8: `captured=true`, `capture_alloc=(0,0)`, `decline=None` (Phase-0's raw M=8 was NotCapturable). The build recipe works.
- **(b) stable replay across bucket growth — PASS** (unchanged): 994/1000 replays, 3 invalidations, 90 tok/s captured M=1.
- **(c) captured M=K wall ≈ captured M=1 wall — FAIL. THE DECISIVE NUMBER = 8.58×.** Captured M=8 replay = **87.2 ms** vs captured M=1 replay = **10.2 ms**. Capture removed M=1 dispatch (13.4→10.2 ms) but did essentially nothing to M=K: captured M=8 (87.2 ms) ≈ eager M=8 (90.9 ms). The ~80 ms floor persists under capture.

**Root cause (structural, the smoking gun):** `segments=41` at M=8 (vs 1 at M=1). The M=K forward does NOT whole-graph capture — every hot kernel opts out at query-width > 1: `GroupQueryAttention[KernelCaptureUnsupported]×40`, `MatMulNBits[KernelCaptureUnsupported]×240`, `SkipSimplifiedLayerNormalization[KernelCaptureUnsupported]×80`. These kernels advertise CUDA-graph capture support ONLY at the M=1 decode shape; at M=K each is a forced eager seam, so the "captured" M=K forward degrades to ~361 eager per-op relaunches — the exact dispatch-bound regime the lever was meant to escape.

**Why:** The Lever B premise — "K× compute is free in one captured replay (GPU idle, weights read once)" — is falsified at K=8 for this int4 decode-bound model. Worse, fixing (a) revealed that (c) fails for a **deeper, more expensive** reason than dispatch: the three hottest op families (GQA, MatMulNBits, SkipSimplifiedLayerNorm) have no M>1 capture support. A real Lever B therefore requires adding M>1 CUDA-graph capture support to those three kernel families (static launch grid, no sync/host-readback/alloc at query-width K) — a deep multi-family kernel program, NOT the "~3–5 eng-weeks reuse existing machinery" the design assumed, and still only for the conditional floor≈1.0×/ceiling~2–3× payoff.

**Decision:** **NO-GO for Lever B.** Promote **Lever A (Marlin int4 relayout, unconditional ~1.3–1.6×, ~4–6 eng-weeks, no capture-support prerequisite) to the primary decode lever.** Lever B is not dead but is **gated behind a kernel-capture-support program** (make GroupQueryAttention / MatMulNBits / SkipSimplifiedLayerNormalization capture-safe at M>1); once that lands, re-run this exact probe to get the true captured-M=K wall before funding the verify-graph build.

Deliverables on branch: Increment-0 overlay (`leverb_increment0_capture_attempt` in `cuda.rs`, PART D in `leverb_phase0_probe.rs`, both `#[cfg(test)]`/`#[ignore]`d, UN-WIRED) and the findings note `docs/research/leverb-phase0-capture-probe.md` (Increment-0 section). Coordinator opens/admin-merges the PR.
