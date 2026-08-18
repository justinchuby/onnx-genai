# Deckard — History (compacted 2026-08-14T09-18-56Z)

**Role:** Systems developer on the Rust runtime, ORT2 loader/shape/IR/EPContext tracks, and CPU/CUDA execution performance. Preserve model-agnostic dispatch, fail-closed claims, checked arithmetic, byte-exact serialization, precision-sensitive tests, and reviewer-lockout ownership transfers.

## Durable lessons
- Repeated invariants: model-agnostic dispatch, fail closed at claim time, checked arithmetic, byte-exact serialization, and precision-sensitive tests.
- Parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.
- **ORT plugin-EP ABI:** `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice` — ORT stores the raw pointer; do not call `ReleaseMemoryInfo` on success. Use `CreateMemoryInfo_V2`. Release only on failure.
- **Shape inference fail-closed:** `Declined` is the correct return for any unmodelled op; never fall back to `SameAsInput(0)`.
- **`validate_dims` must be wired** in the actual read path, not just implemented.
- `OrtGraph*` / `OrtNode*` handles must NOT be stored beyond callback return.

## Historical context

Pre-2026-08-10 entries in `history-archive.md` (shape inference overhaul, EP device lifetime UAF, clippy lint cleanup).

2026-08-10 ep-plugin-export wave archived in `history-archive.md` under "Archive batch 2026-08-10": GetKernelRegistry + NEW-2, f16/bf16 registry, dtype-aware GetCapability, needless_borrow fix.

2026-08-11 ep-plugin-parity-cuda wave and the 2026-08-11/12 upstream-CI correction chronicle (PRs #31985/#31988/#32003/#31973/#31974, CUDA plugin wiring, #762 clippy) archived in `history-archive.md` under "Archive batch 2026-08-14 (Scribe decode-levers)".

## 2026-08-12 — CUDA capture arc COMPLETE (shared: 11.4 → 23.13 tok/s)
Blocker 2 (CLASSIFY) landed as **#848** (`a32900bf`): graph-truth SWA detection
(`graph_enforces_sliding_window`/`effective_sliding_window`) — a vestigial
`sliding_window` in generated metadata was force-routing Muse-Glimmer off the
capture-stable shared-buffer path; Gemma/Mistral real-SWA preserved. My fix was
prerequisite #1 of the 5-blocker chain (#848 classify → #850 load [Batty] → #852
pin [Leon] → #855 bf16 kernel [Sebastian] → #854 skip-norm [Sebastian]). Team
result: native decode **11.4 → 23.13 tok/s**, capture fully engaged (1 segment /
0 seams). Next lever = Cast bf16↔f32 round-trip elimination (now kernel-bound).

## 2026-08-14 — native speculative decode: benchmarked, root-caused, KILLED (#932/#935)
Three-part investigation of native prompt-lookup speculation. **(a) Benchmark (#932, MERGED):**
unreachable on the PipelineEngine path (no verify/rewind hook), NOT byte-lossless, and a large net loss on
qwen2.5-0.5b (best 0.18×/0.27×). **(b) Root-cause (#935, MERGED):** the M=K-verify ≠ M=1 divergence is
**near-tie FP noise, not a bug** — `verify_logits_probe` shows K=1 = 0 flips, row 0 never flips, flips only
at in-block rows ≥1 where greedy top1-top2 gap ≤ ~0.17 (in-block draft K/V batched-GEMM vs M=1 persistent
KV; FP non-associativity). A cheap near-tie guard restores exact greedy identity. **(c) Decisive test
(deckard-spec-14b-verdict) — KILL:** on decode-bound glm-4-9b-int4, greedy 96.4 tok/s vs best speculative
0.74× and **0.47× at 96% acceptance**, because the eager M=K `decode_verify` **abandons CUDA-graph
capture** (replays 1267→25, invalidations 6→280). Native decode is DISPATCH-bound; acceptance is not the
bottleneck. Do NOT wire speculative into the pipeline; same gate blocks EAGLE-3/MTP. Single-stream wins
must come from Marlin relayout / capture-preserving changes.

## 2026-08-14 — Lever B (capture-stable M=K verify) CLOSED as NO-GO; Marlin (Lever A) promoted primary (#948/#949)
Followed the speculative KILL into the #938 "big build" levers. **Phase-0 (#948):** capture machinery is
stable (994/1000 replays, 90.3 tok/s captured M=1) but the raw M=K forward is `NotCapturable`; eager proxy
= 6.77× step function (~80 ms floor, composition unknown) → NO-GO to an unconditional commit, re-test gated
on Increment-0. **Increment-0 (#949) — DECISIVE:** built the capture-enablement overlay (persistent
`[1,K_max,vocab]` logits binding + alloc-free M=K workspace + KV-symbol pin); M=8 now captures but
**captured M=8 = 87.2 ms = 8.58× captured M=1 (10.2 ms)** — the ~80 ms floor PERSISTS. Root cause:
`segments=41` at M=8 because GroupQueryAttention/MatMulNBits/SkipSimplifiedLayerNormalization declare
`KernelCaptureUnsupported` at M>1 → ~361 eager relaunches. **Verdict: NO-GO for Lever B** (gated behind a
deep kernel-capture-support program, not "reuse existing machinery"); **Lever A (Marlin int4 relayout,
unconditional ~1.3–1.6×) promoted to the primary decode lever.** Supersedes #938's Lever-B-first call.

- 2026-08-14 (#957, MERGED 2f0b62b3): Spec-capture feasibility (design-only) → **CONDITIONAL-GO gated behind Marlin** — refines (not overturns) the Lever B NO-GO. Break-even B\*=8.5 today → ~1.2–2.0 post-Marlin; ~80 ms M=8 floor is the 240 MatMulNBits generic GEMMs (Marlin fixes it) + two cheap GQA/norm M>1 capture fixes. Do NOT fund the speculative build ahead of Marlin.

## 2026-08-14 — Marlin fp16×int4 tensor-core GEMM (Lever A) LANDED & MERGED (#960, 7774ec5b)
Built + shipped the from-scratch SM80 `mma.sync.m16n8k16` fused fp16×int4 tensor-core GEMM
(`crates/onnx-runtime-ep-cuda/src/kernels/marlin_gemm.rs`) — adapt-not-vendor (ONNX N-major nibble
packing, asymmetric nibble zero-points, groups 16/32/64/128; per-group scale applied AFTER the
tensor-core accumulate; raw inline `mma.sync` PTX; no upstream source ⇒ no LICENSE vendoring). Wired into
`MatMulNBits` M>1 (plain + rmsnorm-prologue + gate_up SwiGLU fused) + split-K + the GQA/SkipLN M>1
capture-safety valves + lm_head cached dense-GEMM capture plan. Opt-in `ONNX_GENAI_MARLIN_M_GT_1` (default
OFF, SM80 guard, byte-identical tiled fallback — Rule 11); split-K default-ON (`ONNX_GENAI_MARLIN_SPLITK=0`
opts out). **Result (glm-4-9b canonical gate):** M=8 verify graph **41 fragmented segments / B\*=8.76×
(hard NO-GO) → SINGLE whole-graph capture, ZERO unsupported nodes, B\*=2.16×** (arc 8.76→4.99→2.71→2.63→
2.16×), byte-identical greedy tokens, prefill ~2×. B\*≈2.16 is the intrinsic small-M `mma.m16n8k16` floor,
not a tuning gap. **Attribution correction (my update-10 fixes update-8):** fused gate_up split-K needs
`block_size==32`; glm is block-128 so its fused gate_up node never forms — glm's 2.63→2.16× was ENTIRELY
the `4abe4e57` general small-M split-K retune, NOT gate_up fusion. qwen (block-32) forms/fires the fused
node; its capture is fully solved (segments=1, zero unsupported) but B\*≈4.7× is a denominator effect (fast
tuned block-32 M=1), a drafting-depth follow-up not a kernel bug. Gates the #957 spec-capture
CONDITIONAL-GO — Lever A delivered. Reviews: Chew 🟡 APPROVE-WITH-NOTES, Gaff 🟢 APPROVE. Lesson: keep the
flag opt-in until broader prompt/model parity coverage; add a fallback logging/counter (Chew N2) before any
default flip.

## 2026-08-15 — glm int4 decode second act in flight
After #978 split-K and #981 block SkipRMSNorm landed, the post-#981 base-floor read was overturned by the ORT GEMV head-to-head: ORT streams the same gate_up int4 GEMV at 2.42 TB/s vs native 0.92 TB/s with similar tiling/occupancy. Current program: `squad/int4-gemv-wideload` on GPU6, byte-identical 128-bit synchronous wide-load GEMV preserving per-lane accumulation order; target ~180-236 tok/s base decode and ORT base parity. cp.async M=1 remains a measured NO-GO.
## 2026-08-18T01:35Z — Assigned V2-Lite graph-capture unlock

- New implementation target: topology-gated capacity-policy fix for V2-Lite's additive attention-mask builder so CUDA graph capture can engage on MoE without regressing GLM-5.2 logical-width masks.
- Expected surface is executor geometry/build tests plus capture eligibility, not CUDA kernels; Rachael review and Wallace byte-identity/perf validation are required.
