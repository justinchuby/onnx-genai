# Decision drop — Speculative decoding under CUDA-graph capture: CONDITIONAL-GO gated behind Marlin

**Author:** Deckard (CUDA/decode-performance engineer)
**Date:** 2026-08-14
**Branch/PR:** `squad/spec-capture-design` (see PR referencing #948/#949 + the Marlin scoping doc)
**Deliverable:** `docs/research/speculative-capture-feasibility.md` (DESIGN-ONLY; no kernels touched)

## Verdict
**CONDITIONAL-GO — and the single condition is Marlin (Lever A), which is already the funded PRIMARY decode lever.** Speculative-capture is a NO-GO on any path that does not first fix the M>1 int4 GEMM, but it is NOT a separate multi-week kernel program: the one expensive prerequisite IS Marlin. This refines (does not overturn) the settled Lever B NO-GO (#948/#949): Lever B's kernel-capture-support program is largely *subsumed by* Marlin plus two cheap fixes.

## The number that decides it — acceptance break-even B* = C_verify(M=K) / C_decode(M=1), C_decode=10.2 ms
- **Today (measured, #949):** C_verify(M=8) = 87.2 ms → **B\* = 8.5** ⇒ needs >K max-accept ⇒ **guaranteed loss** (matches spec KILL 0.47–0.74× @96%).
- **Post-Marlin (ESTIMATE, ~12–20 ms):** **B\* ≈ 1.2–2.0** ⇒ cleared by prompt-lookup (B≈2–4) and EAGLE-3/MTP ⇒ lands the settled ~2–3× ceiling.

## Root-cause attribution (code-grounded, no new GPU run)
The ~80 ms M=8 floor / 67 ms cliff is **dominated by the 240 `MatMulNBits` generic GEMMs**, NOT GQA/norm:
- MatMulNBits at M>1 leaves its tuned fused int4 GEMV for a slow "portable 16×16 CUDA-core fp32-accumulate tiled GEMM" (`matmul_nbits.rs:4803-4869`) — the only family that switches to a fundamentally un-tuned kernel; streams the same ~4.1 GB int4 weights but ~an order of magnitude slower.
- GQA switches to the *efficient* flash-prefill kernel (~168 MB KV read); norm keeps the *same* kernel (num_groups=M). Both live in the cheap 1.85 ms/row tail. Marlin (static grid, weights reused across M rows) is capture-safe by construction and collapses the cliff.

## Residual capture blockers after Marlin (ranked; both cheap, both structurally capturable)
1. **SkipSimplifiedLayerNorm ×80** — VERY LOW cost: relax the deliberate `last_call_capture_safe=(num_groups==1)` gate to a warmed fixed `num_groups=K` signature (`normalization.rs:2306-2308`, "left conservative by design"). No kernel rewrite.
2. **GroupQueryAttention ×40** — LOW–MEDIUM: pre-warm the fixed-M flash workspace + admit a warmed `q_seq=K` capture signature (`group_query_attention.rs:842-1063,3099-3110`). Flash path has no host read-back. No new attention kernel.

## Staged plan with hard gates (do NOT skip ahead)
- **Stage 1 — Marlin (already primary; ref `decode-remaining-levers-feasibility.md`).** Add ONE contract item: Marlin must advertise **capture-safe at fixed M=K**. Gate 1 = Marlin's own DRAM≥55% gate + instantiates capture-safe at M=K (no seam).
- **Stage 2 — GQA + norm M>1 capture-support (the two cheap fixes).** Gate 2 = seam census at M=8 collapses to ~1 segment (0 MatMulNBits/GQA/norm KernelCaptureUnsupported).
- **Stage 3 — RE-RUN THIS EXACT Increment-0 probe (PART D)** for the true post-Marlin captured M=K wall. **Gate 3 (fund/kill):** measured B\* ≤ ~2 ⇒ GO fund verify-graph; B\* ≳ ~4 ⇒ NO-GO (Marlin already banked its win).
- **Stage 4 — the actual speculative build (only if Gate 3 passes):** verify sub-graph + capture-stable selective KV-commit (write K slots, move logical length host-side) + exact-greedy near-tie guard (#935) + draft sources. Gate 4 = end-to-end >1.0× at measured acceptance with greedy identity.

## Portability (Rule 11)
Marlin is SM-version-specific (SM80+ cp.async/LOP3; relayout differs SM80 vs SM90) ⇒ runtime arch guard + byte-identical split-K fallback on <SM80/CPU EP ⇒ two maintained layouts + SM-scoped repack. Speculative-capture inherits this: on non-Marlin tiers it must **degrade to disabled** (CPU EP already runs M=K eagerly), never regress. All perf here is H200/glm-4-9b-int4-scoped.

## One-line for the roadmap
Do NOT fund the speculative verify-graph/KV-commit/near-tie-guard build ahead of Marlin — it would be fighting the 8.58× wall Marlin exists to remove. Build Marlin (with an M=K capture-safe contract) + the two cheap fixes, then re-run the Increment-0 probe; fund speculative only if measured B\* ≤ ~2.
