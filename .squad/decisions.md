# Decisions

> Current decision ledger. Detailed historical and source records are archived under `.squad/decisions-archive/`.

## Index

- `2026-07-23T00-00-00Z-pre-reconciliation-ledger.md`: pre-reconciliation ledger plus complete processed inbox source notes.

## 2026-07-23 — CUDA fusion reconciliation

The full source notes for this reconciliation are preserved in `2026-07-23T00-00-00Z-pre-reconciliation-ledger.md`.

### Establish 4372f1b as the pre-fusion CUDA baseline
**By:** Marsten
**What:** GPU5 @128 medians were Qwen2.5 0.5B 821.35 tok/s, 1.5B 586.82, 7B 288.64, and Phi-4-mini 136.49; Qwen had one captured segment, Phi three, and diagnostics reported zero fallbacks.
**Why:** This is the clean end-to-end baseline for evaluating the Phi fusion stack and Qwen-versus-ORT behavior.

### Prioritize Phi int8 fused norm seams after zero-point fusion
**By:** Marsten
**What:** Post-fusion Phi reached 166.12 tok/s (+6.32% ON/OFF; +21.71% vs 4372f1b). Profiling assigns 28.0% of decode to int8 GEMV and 15.0% to standalone skip-RMSNorm; Qwen7 regressed to ~253 tok/s pending separate follow-up.
**Why:** The combined 43.0% cost is the largest actionable Phi decode target.

### Enable Phi int4 SwiGLU-RMS zero-point fusion
**By:** Deckard
**What:** The model-agnostic fp32-gamma and asymmetric-int4-zp fusion admitted Phi gate/up projections while retaining Qwen symmetric behavior; rebased validation reported 190/0 CUDA lib tests and coherent, byte-identical Phi/Qwen output.
**Why:** Phi had been excluded by fp32 gamma and explicit asymmetric zero points, leaving a major fusion opportunity unused.

### Approve Phi zero-point fusion with non-blocking nits
**By:** Chew
**What:** Review found asymmetric dequant bit-exact, symmetric Qwen behavior byte-identical, block-128 independent and correct, and steady replay capture-safe; 190/0 CUDA tests, clippy, and real-model checks passed. The ignored parity helper parameter and a blank line are non-blocking.
**Why:** The fusion is numerically sound and generic, while documenting minor follow-up hygiene.

### Fix Qwen symmetric int4 fused-GEMV regression (12efc92)
**By:** Deckard
**What:** The runtime zero-point branch retained an unnecessary per-block global-load path for null-zp Qwen weights, causing 7B -12.3% and 1.5B -7.41% regressions. A compile-time HasZp split restores constant-subtrahend kernels for symmetric weights while retaining asymmetric Phi dequant. GPU4 A/B restored Qwen 7B to 289.9 tok/s (base 291.3) and 1.5B to 595 (base 602).
**Why:** The regression was real code, not CPU noise; the model-agnostic specialization restores occupancy and performance without changing correctness.

### Approve the Qwen int4 regression fix (12efc92)
**By:** Chew
**What:** Chew verified the HasZp=false kernels never touch zero points, launch routing selects _zp only when needed, and both symmetric and asymmetric paths are covered. CUDA lib tests passed 190/0 and clippy was clean.
**Why:** The review confirms the recovery does not compromise Phi asymmetric dequant, block-128, int8, or fp32-gamma paths.

### Keep Phi int8 skip-RMSNorm MatMulNBits fusion in flight (c644b0f)
**By:** Deckard
**What:** The model-agnostic bits-{4,8} fusion adds an int8 RMSNorm-prologue GEMV and prefill zero-point threading. GPU4 reported Phi 160.65→181.62 tok/s (+13.0%), byte-identical/coherent output, 192/0 CUDA tests, and clean clippy; Qwen remained coherent and inert.
**Why:** Phi qkv/down int8 projections and their standalone input norm are a high-cost remaining seam.

### Approve Phi graph-seams control-flow shape seeding (4372f1b)
**By:** Roy
**What:** Review found seeding affects segmentation only; control-flow seams execute eagerly before consumers and invalidate safely on branch-shape changes. Qwen is inert. CUDA/session/engine tests, clippy, long-RoPE identity, and coherent Phi/Qwen runs passed.
**Why:** The capture improvement is model-agnostic and preserves shape/capture correctness.
