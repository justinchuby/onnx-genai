# Speculative decoding under CUDA-graph capture — feasibility & the minimal staged path (DESIGN-ONLY)

**Author:** Deckard (CUDA / decode-performance engineer)
**Date:** 2026-08-14
**Branch:** `squad/spec-capture-design` (worktree `deckard-specdesign`, based on `main` 4d4c2273)
**Model of record:** glm-4-9b-int4-cuda (`/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda`) — the decode-bound reference from the speculative KILL test and the Lever B probes. 40 GQA layers, `num_attention_heads=32`, `num_kv_heads=2`, `head_dim=128` ⇒ q_hidden `4096`, kv_hidden `256`; vocab `151552`; FFN intermediate ≈ `13696` (standard GLM-4-9B; used only for the order-of-magnitude byte census in §2).
**HW of record:** 1× H200 SXM (HBM3e ≈ 4.8 TB/s). All cited numbers are M=1 captured decode, `ONNX_GENAI_CUDA_GRAPH=1`.
**Scope:** This is a **design/feasibility** doc grounded in kernel code + the already-measured Lever B probes. It does **NOT** build Marlin or any kernel-capture fix and does not modify production kernels. No new GPU measurement was required or run — the decisive numbers are already merged (#948/#949) and the floor attribution is settled by reading the M>1 kernel dispatch paths.

> **Bottom line up front.** Speculative decoding *can* be made a net win under capture, but **not on its own terms** — it is a **CONDITIONAL-GO whose single condition is Marlin (Lever A), which we are already building as the primary decode lever.** Today a captured K=8 verify costs **8.58× a single-token decode**, so speculative needs a physically-impossible **B* ≈ 8.5** mean-accepted-tokens to break even (guaranteed loss, matching the settled KILL). The 8.58× is dominated by one thing — the int4 `MatMulNBits` GEMM falling off its tuned M=1 fused GEMV onto a slow generic 16×16 CUDA-core tiled GEMM at M>1 — which is **exactly the kernel Marlin replaces.** With a Marlin int4 GEMM (static launch grid, weights read once across the M query rows) the estimated captured verify drops to order **~12–20 ms**, moving break-even to **B* ≈ 1.2–2.0**, which prompt-lookup and EAGLE-3/MTP clear comfortably. Speculative-capture is therefore **not a separate multi-week kernel bet** — it falls out of Marlin plus two *cheap* capture-support fixes (SkipSimplifiedLayerNorm = a one-line signature relaxation; GQA = pre-warmed fixed-M workspace). **Do not fund the verify-graph/KV-commit/near-tie-guard build until Stage-3 re-measures the post-Marlin captured verify wall.**

---

## 1. The settled root cause (self-contained restatement — do NOT re-measure)

From the Lever B Phase-0 + Increment-0 probes (`docs/research/leverb-phase0-capture-probe.md`; `.squad/decisions.md` "Decode perf REOPENED — RESOLVED"; PRs #948/#949), all measured on glm-4-9b-int4 / H200, reproduced 3×:

- **Capture machinery works.** After the three Increment-0 fixes (persistent padded `[1, K_max, vocab]` logits binding · alloc-free M=K warm forward · KV-symbol pin) the M=8 forward **instantiates capture-safe** (`captured=true`, `capture_alloc=(0,0)`) and **replays stably** across bucket growth (994/1000 replays, 3 invalidations = growth count, 90 tok/s). Criteria (a) and (b) **PASS**. The capture state machine is *not* the risk.
- **But the payoff premise is falsified.** The **decisive number: captured M=8 replay = 87.2 ms = 8.58× captured M=1 (10.2 ms)**, and captured M=8 (87.2 ms) ≈ eager M=8 (90.9 ms). **Capture removed almost nothing at M>1** — it killed the M=1 dispatch (eager 13.4 → captured 10.2 ms) but left the ~80 ms M=K floor intact. Criterion (c) **FAILS**.
- **Two-fold root cause, both tracing to one missing capability — no fast, capture-safe batched int4 GEMM at M>1:**
  1. **Segmentation (`segments=41`, not 1).** The "captured" M=8 graph is 41 glue segments stitched by ~361 **eager seam nodes**. Seam census (`capture_segmentation()`, #949):
     `GroupQueryAttention[KernelCaptureUnsupported]×40 · MatMulNBits[KernelCaptureUnsupported]×240 · MatMul[KernelCaptureUnsupported]×1 · SkipSimplifiedLayerNormalization[KernelCaptureUnsupported]×80`.
     Every hot decode kernel advertises capture support **only at the M=1 decode shape**; at query-width K each is a forced eager relaunch → the dispatch-bound regime the whole lever was meant to escape.
  2. **The 66.9 ms M=1→M=2 CLIFF persists under capture.** Capture kills dispatch; the cliff survived; therefore the cliff is **not** dispatch. It is the int4 `MatMulNBits` leaving its fused GEMV fast path for a slow generic multi-row GEMM. The M=2→M=8 **tail is only 1.76–1.85 ms/row** — extra verify rows are *cheap*; they sit on an ~80 ms floor established the instant M crosses 1.

The premise "K× compute is free because the GPU is idle and weights are read once" is true for the *tail* (1.85 ms/row) but false for the *floor*, because the floor is not idle-GPU arithmetic — it is a genuinely slow kernel doing the weight stream inefficiently.

---

## 2. Decomposing the ~80 ms M=8 floor by op family (code-grounded attribution)

**Claim to confirm/refute:** the 67 ms cliff (hence the ~80 ms floor) is dominated by the generic M>1 int4 `MatMulNBits` GEMM, **not** by GroupQueryAttention or the norms. **Confirmed** from the M>1 dispatch code. I use the cheapest credible evidence — the kernel *path selection* at M>1 plus a weight-byte census — rather than fabricating a per-op millisecond split.

### 2.1 Which kernel each op family runs at M>1 (read from the dispatch)

| Op family | M=1 path (fast, captured today) | **M>1 path actually taken** | Why it opts out of capture |
|---|---|---|---|
| **MatMulNBits ×240** | Tuned fused int4 **GEMV** (`GEMV_ACCURACY4*` / `GEMV_F16_*`), memory-bound at ~29% peak DRAM, fused dequant, wide loads, fused rmsnorm/swiglu/down epilogues | **`launch_f16_gemm` / `launch_f16_gemm_rmsnorm_prefill`** — a *"portable 16×16 CUDA-core tiled GEMM with fp32 accumulation"* (`matmul_nbits.rs:4803-4869`). A **different, un-tuned kernel**: CUDA cores (no tensor cores), fp32 accumulate over dequantized int4, 16×16 tiles, no `cp.async` wide loads, no fused-dequant pipeline. | `last_call_capture_safe=false` forced at `m>1` (`matmul_nbits.rs:4803-4808`). Comment: kernel itself has fixed pointers / no alloc / no sync, but is *"outside the persistent M=1 decode graph … no replay coverage."* Conservative flag **on top of** a slow kernel. |
| **GroupQueryAttention ×40** | Fused single-token split-K decode (`gqa_decode*`, `q_seq==1`), streams softmax through registers | **Flash-prefill path** (`flash_attention`, `q_seq>1`) or f32 **reference-scores** path — needs a `q_seq`-sized transpose/scores **scratch** (`gqa_transpose_scratch`, `gqa_reference_scores_bytes`, `group_query_attention.rs:842-1063`) | `last_capture_safe_signature` warmed only for the "one-token, fixed-capacity, in-place device-KV decode signature" (`group_query_attention.rs:3099-3110`); the q_seq>1 workspace was never warmed. |
| **SkipSimplifiedLayerNorm ×80** | Fused residual RMS-norm, **single group** | **Same kernel**, just `num_groups = M` (one group per token row). Static grid `groups_u`, `block_dim (32,1,1)`, **no alloc, no sync** (`normalization.rs:2280-2308`). | `last_call_capture_safe.store(num_groups == 1)` — **a deliberate policy** (`normalization.rs:2306-2308`), explicitly *"Left conservative by design"*, **not** a structural veto. |
| **MatMul ×1** (lm_head) | fp16 GEMM `[1,1]×[hidden,vocab]` | fp16 GEMM `[1,K]×[hidden,vocab]` — a *single* tensor-core GEMM, ≈ 8·4096·151552 ≈ 5 GMAC at K=8 (~sub-ms on H200) | outside the advertised decode-capture contract |

### 2.2 Weight-byte census — why MatMulNBits must dominate

Per-token, `MatMulNBits` streams the model's **entire int4 weight matrix once** (weight DRAM is independent of M). Order-of-magnitude per layer (int4 ≈ 0.5 B/weight): QKV `4096×4608` + attn-out `4096×4096` + MLP gate_up `4096×27392` + MLP down `13696×4096` ≈ **204M weights ≈ 102 MB**; ×40 layers ≈ **~4.1 GB** streamed every forward (consistent with a ~9B-param int4 model).

- At **M=1** that 4.1 GB stream through the tuned GEMV is ~29% of 4.8 TB/s ≈ effective ~1.4 TB/s → the GEMV portion is **~61% of the 10.2 ms token** (≈ 6 ms), 39% non-GEMV tail (`.squad/decisions.md`, #928).
- At **M>1** the *same* 4.1 GB streams through the **generic 16×16 CUDA-core fp32-accumulate** kernel — far lower DRAM efficiency plus per-tile int4 dequant on CUDA cores. Same bytes, ~an order of magnitude slower kernel ⇒ the stream now dominates the ~80 ms floor.
- **GQA reads only the KV cache:** 2 kv-heads · 128 · past_len · 40 · 2(K,V) · 2 B ≈ **~168 MB at 2048 ctx** — ~25× less than the 4.1 GB weight stream, through an *efficient* flash kernel. **Norm** touches only hidden-sized activations (`M·4096` elements) through a fast reduction — negligible DRAM.

**Conclusion (confirmed):** the ~80 ms floor is **dominated by the 240 `MatMulNBits` generic GEMMs**, corroborated three ways: (i) it is the *only* family that switches to a fundamentally different, un-tuned kernel at M>1 (GQA switches to an *efficient* flash kernel; norm keeps the *same* kernel); (ii) it carries ~25× more DRAM traffic than the next family; (iii) the near-flat 1.85 ms/row tail proves the floor is a **fixed** cost incurred at M=2 (the switch to the slow kernel), not a per-row cost from attention/logits. GQA and norm live inside the cheap tail. *(This is a code + byte-census attribution; a per-op `ONNX_GENAI_PROFILE_OPS` split at M=8 would quantify it further but requires wiring the un-wired M=K probe and is not needed to establish dominance.)*

---

## 3. The Marlin payoff for the verify path, and the acceptance break-even

### 3.1 Why Marlin is the enabler, not just a speedup

A Marlin-style fused int4 GEMM is **capture-safe by construction and M-scalable by construction**:
- **Static launch grid + no alloc/sync/D2H** at any fixed M=K ⇒ it can advertise capture support at M=K, **eliminating the 240 `MatMulNBits` seams** (the largest seam family) — it does not merely go faster, it removes the segmentation.
- **Weights reused across the M query rows** via tensor-core `mma` tiles (Marlin's published wins are precisely the M=16–64 batched regime) ⇒ the 4.1 GB weight stream is read **once** and amortized over K rows, collapsing the 67 ms generic-GEMM cliff toward the M=1 weight-stream cost.

### 3.2 Estimated captured verify wall after Marlin  *(clearly labeled ESTIMATE)*

Lower bound is hard: a captured M=K verify can never be cheaper than the captured M=1 wall (**10.2 ms**) — it does at least the same weight DRAM plus more attention/activation. Building up:

- **MatMulNBits (GEMM) portion:** post-Marlin ≈ the M=1 weight-stream cost, since weight DRAM is M-independent and the GPU is idle for the extra K× ALU. Estimate **~6–10 ms** (≈ the current M=1 GEMV share, possibly *better* since Marlin targets the 29%-DRAM inefficiency directly).
- **Attention (GQA flash at M=K) + norms + activations + lm_head:** the cheap tail, ~1.8 ms/row measured, i.e. **~a few ms** additional at K=8, most of it the idle-GPU-free arithmetic.
- **Estimated captured M=8 verify ≈ 12–20 ms** (central **~15 ms**), vs **87.2 ms today** — a ~5–7× reduction on the verify wall. Labeled an **order-of-magnitude estimate**; Stage-3 (§5) measures the true number.

### 3.3 The acceptance break-even (the number that decides "worth it")

Per speculative iteration: a draft proposes K tokens, **one** verify forward at width M=K commits **B** tokens (accepted prefix + bonus), `1 ≤ B ≤ K+1`. Speculative is a net win iff the verify wall beats producing those B tokens autoregressively:

> **C_verify(M=K) < B × C_decode(M=1)**  ⇒ break-even  **B\* = C_verify(M=K) / C_decode(M=1)**, with `C_decode(M=1) = 10.2 ms`.

| Regime | C_verify(M=8) | **Break-even B\*** | Achievable? |
|---|---|---|---|
| **Today (measured)** | 87.2 ms | **8.5** | **No** — exceeds K=8 max accept ⇒ guaranteed loss (matches settled KILL: 0.47–0.74× even at 96% accept, #932/#935/spec-14b) |
| **Post-Marlin — optimistic** | ~12 ms *(est.)* | **~1.2** | Yes — trivially |
| **Post-Marlin — central** | ~15 ms *(est.)* | **~1.5** | Yes |
| **Post-Marlin — conservative** | ~20 ms *(est.)* | **~2.0** | Yes |

Typical mean-accepted-tokens: prompt-lookup on repetitive/structured text B≈2–4; EAGLE-3/MTP B≈2.5–4 workload-general. Against a post-Marlin **B\* ≈ 1.2–2.0**, those clear break-even and land in the settled **~2–3× ceiling**. **The single fact that flips speculative from guaranteed-loss to conditional-win is Marlin dropping the verify wall — i.e. exactly Lever A.**

---

## 4. Residual capture blockers after Marlin (ranked by cost)

Marlin removes the 240 `MatMulNBits` seams and the 1 lm_head `MatMul` follows the same tensor-core-GEMM treatment. Two seam families remain; both are already known to be **structurally capturable** at fixed M=K — the blocks are *policy/warm-up*, not kernel vetoes:

| Rank | Blocker | Kernel reality at M=K | What it needs to be capture-safe at fixed M=K | Cost |
|---|---|---|---|---|
| **1 (cheapest)** | **SkipSimplifiedLayerNorm ×80** | Static grid `groups_u=M`, `block_dim (32,1,1)`, **no alloc / no sync / no D2H** (`normalization.rs:2280-2308`). Already structurally capturable. | Relax the deliberate `last_call_capture_safe = (num_groups == 1)` gate to admit a **warmed fixed `num_groups=K`** signature, with the existing `SkipBroadcastMetadataCache` drift guard extended to key on the fixed M. Essentially a **signature relaxation**. | **Very low** — hours/days; no kernel rewrite. |
| **2** | **GroupQueryAttention ×40** | q_seq>1 uses the efficient **flash-prefill** kernel (streams softmax through registers — no host read-back). The block is the **`q_seq`-sized transpose/scores workspace** not being warmed for capture (`gqa_transpose_scratch`, `gqa_reference_scores_bytes`), and the capture signature admitting only `q_seq==1`. | **Pre-allocate/warm the fixed M=K attention workspace** before capture (same trick Increment-0 used for the arena) and **admit a warmed fixed-`q_seq=K` signature**. Prefer the flash path (no reference-scores buffer, no read-back). Must guarantee: static grid at fixed M=K, no mid-kernel alloc/free/sync. | **Low–medium** — the flash kernel exists; work is workspace pre-warm + signature admission + a capture-drift guard. Days–~1 week. |

Neither requires a new attention or norm kernel. Contrast the *pre-Marlin* Lever B framing ("make three kernel families capture-safe from scratch") — **once Marlin lands, only these two cheap fixes remain, and both are warm-up/signature changes, not `mma`-kernel builds.**

---

## 5. Staged plan with hard go/no-go gates

Each stage has a measurable gate; **do not start the next stage until the prior gate passes.** Stages 1–2 are prerequisites already justified by decode work; Stage 3 is the cheap re-measurement that must pass **before any verify-graph build is funded.**

### Stage 1 — Marlin int4 GEMM (Lever A). *Already the PRIMARY decode lever — do NOT re-scope here.*
Reference `docs/research/decode-remaining-levers-feasibility.md` for full scoping (41 GEMV entry points w/ fused epilogues, offline repack, Chew numerics gate, SM80+ arch guard). This design adds **one requirement to Marlin's contract:** the M>1 kernel must be **advertised capture-safe at a fixed M=K** (static grid, no alloc/sync/D2H) — the same bar the M=1 variants meet — so it doubles as the speculative enabler at no extra kernel cost.
- **Gate 1 (Marlin's own gate, unchanged):** M=1 relayout GEMV lifts achieved DRAM from ~29% → ≥~55% ⇒ ~1.3–1.6× decode win. Plus the **added check**: the same kernel instantiates capture-safe at a fixed M=K probe shape (no seam). Fail ⇒ Marlin still ships for M=1 decode, but speculative-capture is dead (no other path to a fast capturable M>1 int4 GEMM) → re-open only with a bespoke batched-GEMV effort.

### Stage 2 — GQA + SkipSimplifiedLayerNorm M>1 capture-support (the two cheap fixes from §4).
- **Gate 2:** re-run the seam census (`capture_segmentation()`) from the #949 probe at M=8 **with Marlin + these fixes**. Pass = `segments` collapses to ~1 (seam census shows **0** `MatMulNBits` / GQA / norm `KernelCaptureUnsupported` nodes). Fail on GQA ⇒ fall back to a per-verify capture that seams only the 40 GQA nodes and re-price at Gate 3 (40 seams ≪ 361; may still win).

### Stage 3 — RE-RUN THIS EXACT INCREMENT-0 PROBE for the true post-Marlin captured-M=K wall. *(The decisive, cheap gate.)*
Re-run `leverb_phase0_capture_probe` PART D (`#[ignore]`d, un-wired) on Marlin + Stage-2 fixes to measure **C_verify(M=K)** and compute **B\* = C_verify / 10.2 ms** directly (no estimate).
- **Gate 3 (the fund/kill decision):**
  - **B\* ≤ ~2** (i.e. C_verify(M=8) ≲ ~20 ms as estimated) ⇒ **GO** — fund the verify-graph build (Stage 4).
  - **B\* ≳ ~4** (floor only partially collapses) ⇒ **NO-GO / re-scope** — speculative can't clear realistic acceptance; Marlin already banked its unconditional win regardless.

### Stage 4 — the actual speculative build (ONLY if Gate 3 passes).
Verify sub-graph + **capture-stable selective KV-commit** (always write K physical slots, move logical length host-side — a pointer move, no graph change; per `decode-remaining-levers-feasibility.md` §4) + **exact-greedy near-tie guard** (re-decode ~4–9% of rows at M=1, #935) + draft sources (prompt-lookup ≈ free; EAGLE-3/MTP = separate trained-head build).
- **Gate 4:** end-to-end tok/s on a repetitive workload (prompt-lookup) is **> 1.0×** baseline at measured acceptance, and exact-greedy output identity holds through the near-tie guard. Fail ⇒ ship Marlin-only.

---

## 6. Portability note (Rule 11)

Marlin is **SM-version-specific**: `cp.async`/`LOP3` dequant is SM80+, and the tensor-core `mma` tile shapes and the **weight relayout differ by arch** (Ampere SM80 vs Hopper SM90). Per Rule 11 (`RULES.md` §11) this means: a runtime arch guard, a **byte-identical fallback to the current split-K layout on <SM80 and the CPU EP**, and an offline repack keyed to the target SM — i.e. **two maintained weight layouts** and an SM-scoped repack step. The speculative feature inherits this: on tiers without Marlin, the M=K verify keeps seaming on the generic GEMM (today's 8.58×) and speculative-capture must **degrade to disabled**, never regress — the CPU EP already runs M=K eagerly, so it degrades to today's autoregressive decode gracefully. Perf claims here are **H200-scoped** (glm-4-9b-int4); a bandwidth-starved or VRAM-limited tier may shift both the floor and the break-even.

---

## 7. Honest bottom line

**CONDITIONAL-GO, and the condition is Marlin — which we are already building.**

- Speculative-capture is a **NO-GO today** and will remain one on any path that does not first fix the M>1 int4 GEMM: the captured verify is 8.58× a decode, requiring an impossible B\* ≈ 8.5.
- It is **not a standalone multi-week kernel program** as the pre-Marlin Lever B framing implied. The single expensive prerequisite — a fast, capture-safe batched int4 GEMM at M>1 — **is Marlin (Lever A), already the primary decode lever.** The two residual blockers (GQA, SkipSimplifiedLayerNorm) are cheap warm-up/signature fixes, not kernel rewrites.
- Post-Marlin, the **estimated** captured verify (~12–20 ms) moves break-even to **B\* ≈ 1.2–2.0**, which prompt-lookup and EAGLE-3/MTP clear, unlocking the settled ~2–3× ceiling.
- **Therefore: build Marlin (already funded), add the M=K capture-safe contract + the two cheap fixes, then re-run the Increment-0 probe (Stage 3). Fund the verify-graph build only if that measured B\* ≤ ~2.** No speculative kernel work should start ahead of Marlin — it would be measuring the 8.58× wall that Marlin exists to remove.

---

## 8. Sources
- Settled root cause & decisive numbers: `docs/research/leverb-phase0-capture-probe.md` (Phase-0 + Increment-0); `.squad/decisions.md` "Decode perf REOPENED — RESOLVED" (#928/#932/#933/#935/**#948**/**#949**).
- Lever A (Marlin) scoping & the dispatch-bound diagnosis this doc must respect: `docs/research/decode-remaining-levers-feasibility.md`.
- MatMulNBits M>1 = generic 16×16 CUDA-core tiled GEMM, capture flag forced at m>1: `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs:4803-4869, 6847-6866`.
- GQA M>1 flash-prefill / reference-scores workspace + one-token capture signature: `crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs:842-1063, 3099-3110`.
- SkipSimplifiedLayerNorm structurally capturable, `num_groups==1` policy "left conservative by design": `crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs:2280-2331`.
- Selective KV-commit / near-tie guard / EAGLE-3-MTP share `decode_verify`: `decode-remaining-levers-feasibility.md` §4; #935.
- Portability (Rule 11): `RULES.md` §11; `docs/portability/2026-07-25-cuda-consumer-gpu-audit.md`.
