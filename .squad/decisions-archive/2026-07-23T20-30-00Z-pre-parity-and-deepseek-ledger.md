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

## 2026-07-23 — CUDA fusion, model enablement, and capture follow-up

### Approve Phi int8 skip-RMSNorm MatMulNBits fusion (c34f813)
**By:** Chew
**What:** The model-agnostic int8 RMSNorm-prologue fusion preserves the HasZp compile-time specialization: asymmetric Phi uses `_zp`, while symmetric paths fold to constant 128 without a zero-point load. Review verified bit-exact dequantization, correct dispatch and prefill threading, 192/0 CUDA tests, and clean clippy; Phi gains about 10–13% with Qwen unaffected.
**Why:** It removes Phi's standalone skip-RMSNorm/int8-GEMV seam without reintroducing the Qwen symmetric int4 regression.

### Treat int4 zero-point GEMV as a dedicated latency-hiding spike
**By:** Deckard
**What:** Phi's fused and standalone int4 zero-point GEMVs are latency/issue-bound (about 17% DRAM peak), not bandwidth-bound; forcing higher occupancy produced no speedup and was reverted. The viable lever is a separately reviewed cp.async double-buffer or split-K pipeline that preserves Qwen symmetric byte identity.
**Why:** The zero-point machinery is already register- and traffic-efficient, so incremental occupancy changes cannot improve decode.

### Record the post-int8-fused milestone documentation as verified
**By:** Fact Checker
**What:** The post-int8-fused benchmark documentation recomputes all five native-versus-ORT deltas within 0.005pp, its Nsight allocation sums to 99.2%, and its dense-DeepSeek/MoE status matches the enablement document.
**Why:** The benchmark narrative is accurate, including Phi remaining 25.46% behind ORT and practical MoE packing blockers.

### Prioritize GLM-4 Split-seam capture defragmentation
**By:** Marsten
**What:** DeepSeek-R1-Distill-Qwen-1.5B reaches 576.41 tok/s, 17.83% above ORT. GLM-4-9B is runnable at 110.34 tok/s with zero fallbacks but has 41 captured segments; ORT GenAI 0.14.1 cannot load its partial-RoPE GQA attribute.
**Why:** Removing GLM-4's forty eager, synchronizing activation-Split seams is the clearest native-only performance lever.

### Establish dense DeepSeek-Coder and R1-Distill as runnable int4 CUDA targets
**By:** Batty
**What:** DeepSeek-Coder-1.3B is a coherent, zero-fallback block-32 symmetric int4 smoke target, and R1-Distill-Qwen-1.5B captures as one segment. Defer DeepSeek V2/V3 and GLM-5.2/DeepSeek-V4 until fused-QMoE packing and sparse-attention/MTP integration are available.
**Why:** Dense Llama-style artifacts use existing GQA, RMSNorm, and MatMulNBits coverage, unlike the remaining MoE and sparse-model blockers.

### Make GLM-4-9B native CUDA support a Mobius preprocessing path
**By:** Batty
**What:** Mobius remapping plus fused GPTQ QKV and gate/up splitting makes the 6.3 GB GLM-4-9B graph runnable on native CUDA; ORT's current remote config remains unloadable. The resulting graph has 240 MatMulNBits and 40 GQA nodes.
**Why:** Checkpoint hierarchy and fused quantized projections, rather than runtime CUDA support, were the GLM-4 execution blockers.

### Prefer Mobius gate/up pre-splitting before a capture-safe CUDA Split path
**By:** Batty
**What:** GLM-4's 40 Split seams arise from CUDA Split's trailing stream synchronization despite resolved output shapes. First split packed gate/up weights and nodes during Mobius preprocessing; if that is insufficient, implement a static capture-safe CUDA Split path rather than extending executor shape seeding.
**Why:** Pre-splitting can collapse 41 capture segments with a smaller runtime blast radius.

## 2026-07-23 — CUDA decode and GLM-4 follow-up

### Keep GLM-4 packed gate/up pre-splitting
**By:** Batty
**What:** Import fused GPTQ gate/up tensors as separate Mobius gate_proj and up_proj MatMulNBits weights, eliminating runtime Split nodes; a fresh 9B export has zero Split nodes and one capture segment.
**Why:** Release A/B improved steady decode 110.39→118.24 tok/s (+7.1%) with identical 128-token greedy output and zero capture fallbacks.

### Decline cp.async for Phi asymmetric int4 zero-point GEMV
**By:** Deckard
**What:** The latency-bound `_zp` GEMVs reach only 13–18% DRAM peak; occupancy forcing was a no-op and a two-stage cp.async buffer improved issue activity but added registers and lost occupancy.
**Why:** The best end-to-end effect was about 1% and within noise, so no kernel change ships; pursue targeted grid-starvation remedies instead.

### Adopt split-K=2 for grid-starved asymmetric int4 zero-point GEMV
**By:** Deckard
**What:** Route eligible block-32 fp16-scale asymmetric-zp GEMVs with K divisible by 256 to a separate within-block split-K=2 entry, retaining byte-identical existing symmetric and fallback paths.
**Why:** It doubles the sparse grid (0.36→0.73 waves/SM), reduces the target kernel 8.5→7.1µs, and improves Phi decode 173.6→176.9 tok/s (+1.9%); tolerance parity, 192 CUDA tests, and clippy pass.

### Record GLM-4 static Split capture as generic EP support
**By:** Marsten
**What:** Generic single-input static Split capture removes GLM-4's 40 fused-MLP gate/up activation Split seams (`Split(axis=-1, num_outputs=2)`), giving one capture segment, zero fallbacks, and 110.34→118.85 tok/s (+7.71%).
**Why:** This fixes a model-agnostic capture barrier without a graph rewrite; ORT's partial-RoPE GQA schema incompatibility is separate and unrelated.

### Prioritize Phi fused gate-up int4 efficiency after split-K and GQA
**By:** Marsten
**What:** With split-K and occupancy-aware GQA, Phi-4-mini reaches 184.27 tok/s (19.75% behind ORT); fused gate-up/SwiGLU/RMSNorm zero-point int4 is now 33.3% of captured decode.
**Why:** Standalone zero-point int4 is down to 12.1% and GQA core is 5.89µs, so next work should improve fused gate-up dequant/GEMV efficiency; GQA prep/merge is secondary.

### Guard static Split capture with concrete-shape replay parity
**By:** Sebastian
**What:** The static even-Split integration test now supplies concrete shapes, executes the static kernel, captures and replays with changed input, and byte-compares replay output with eager output.
**Why:** The generic helper provides empty shapes and only exercises dynamic Split; capture/replay verifies the static no-synchronization path.

## 2026-07-23 — QMoE, GLM-5.2, and Phi decode follow-up

### Make QMoE CUDA graph capture safe after eager warmup
**By:** Batty
**What:** QMoE reuses pooled scratch, avoids synchronization while capturing, and advertises capture only after fixed-shape eager warmup; all 21 GPU cases capture, replay, and retain CPU parity.
**Why:** Per-call allocation/free and synchronization previously split MoE decode out of CUDA graphs; stable storage and precompiled modules make fixed-shape replay viable.

### Serialize QMoE capture tests and prove routing stays live
**By:** Sebastian
**What:** QMoE GPU integration tests now serialize each test with a process-wide GPU mutex and replay changed `router_probs` against an uncaptured eager reference.
**Why:** Serialization eliminates concurrent-allocation capture invalidation, while changed-routing parity verifies replay recomputes expert selection instead of baking capture-time routes.

### Record QMoE capture review and test-gate resolution
**By:** Holden
**What:** Review approved the production fixed-geometry, device-routed QMoE capture design, but initially rejected its flaky parallel test gate because it lacked the repository GPU serial guard; the subsequently landed Sebastian test fix addresses that blocker.
**Why:** Capture must avoid allocation, synchronization, and host routing reads, and its test gate must be stable under default parallel execution.

### Enable strict native CUDA GLM-5.2 decode with logical-prefix bindings
**By:** Batty
**What:** Prefix-sensitive consumers receive logical binding prefixes while capacity-oriented mask consumers keep padded geometry; int64 Min/Max arithmetic is supported and replay is disabled when logical geometry grows. Dense, int4, and QMoE tiny GLM-5.2 decode paths now run with zero fallbacks.
**Why:** DSA attention indexing requires the logical `[1,4]` mask rather than padded `[1,4096]`; the generalized contract preserves fixed-capacity GLM-4 capture.

### Establish GLM-5.2 QMoE native CUDA capability
**By:** Marsten
**What:** At `bd05b75`, tiny GLM-5.2 dense, q4, and fused-QMoE models complete DSA decode on native CUDA at 70.63, 148.58, and 174.41 tok/s respectively, with zero fallbacks.
**Why:** This validates the architecture and native QMoE execution path, while growing logical prefixes correctly keep whole-model graph replay disabled.

### Verify the comprehensive GLM-5.2 benchmark refresh
**By:** Fact Checker
**What:** All six documented deltas recompute within 0.05pp; capability/capture, GLM-4 Split, Phi split-K, and progress claims are supported. A caveat identifies refreshed Qwen 0.5B/1.5B native highs as host-load variance.
**Why:** The report remains accurate without attributing measurement variance to kernel gains.

### Ship Phi int4 zero-point gate/up weight prefetch
**By:** Deckard
**What:** The asymmetric (`HasZp`) fused gate/up SwiGLU-RMSNorm GEMV prefetches only the next packed gate/up weight words; Phi kernel time improves 34.7→31.2µs and decode improves 180.7→184.0 tok/s (+1.8%), with bit-exact output. Symmetric Qwen retains its original loop and stays flat.
**Why:** The kernel is global-load-latency-bound; weight-only scheduling overlaps loads without the register cost of scale/zero-point prefetch or a TMA staging rewrite.

### Approve Phi gate/up prefetch boundary and identity properties
**By:** Chew
**What:** Review confirmed the guarded next-iteration prefetch has no terminal over-read, accumulation order is unchanged, and compile-time isolation leaves the symmetric Qwen PTX path unchanged; CUDA tests and clippy passed.
**Why:** The optimization is a low-risk scheduling-only improvement under the gate/up kernel's `k >= 256` production geometry.

### Decline fused Phi int8 prefetch and split-K variants
**By:** Deckard
**What:** Int8-zp prefetch reduced scoreboard stalls but left the ~20µs kernel and end-to-end decode flat. Split-K increased waves/SM but also washed or regressed because the serial full-vector RMSNorm reduction/staging prologue dominates; both spikes were reverted.
**Why:** Grid-starvation remedies help pure GEMVs, not this fused kernel's Amdahl-limited prologue; investigate the standalone grid-starved int8 down-projection instead.


## 2026-07-23 — Domain normalization, Mobius QMoE emitter, and Qwen-7B roofline

### Normalize default ONNX domains at IR load boundaries
**By:** Sapper
**What:** `ai.onnx` is normalized to the empty default domain at load time, with the new IR domain helper collapsing roughly fifteen call sites to `Node::is_default_domain()` and removing duplicate helper logic; merged to `origin/main` as `06d71ba` and `1073404` after Luv green reviews.
**Why:** Canonical domain representation prevents repeated ad-hoc normalization and keeps operator dispatch/default-domain checks consistent.

### Record Qwen-7B down-projection column-split as a no-go
**By:** Deckard
**What:** A bounded Qwen-7B decode spike found the down-projection GEMV grid-starved on paper, but column-splitting doubled activation staging and washed end-to-end; the gate/up shared-prefetch attempt regressed. No code shipped.
**Why:** The clean bounded levers did not convert into throughput, and true K-slice split-K required a separate scratch/reduction design before being attempted.

### Record Qwen-7B true K-slice split-K as a no-go
**By:** Deckard
**What:** A correct two-kernel K-slice split-K implementation filled waves/SM for the int4 down-projection, but the partial kernel stayed flat and the reduction node regressed Qwen-7B about 2.3%; the prototype was reverted.
**Why:** The kernel is limited by int4 weight-read efficiency/shared-memory latency rather than available grid parallelism.

### Close Qwen-7B int4 vectorized-load investigation as roofline-limited
**By:** Deckard
**What:** Profile-only analysis showed int4 weight loads already use contiguous 128-byte warp regions with about 94–100% global-load sector efficiency; dominant stalls are short-scoreboard shared-memory dependencies. No load-pattern code changed.
**Why:** 128-bit load widening or coalescing changes cannot reduce bytes moved or the binding shared-memory dependency; further 7B micro-optimization should stop absent a larger GEMV redesign.

### Keep standalone Phi int8 zero-point split-K as a validated win
**By:** Deckard
**What:** The standalone asymmetric int8 GEMV split-K path improved Phi end-to-end throughput about 2.1% while leaving Qwen unchanged by construction; CUDA tests and clippy were clean on the feature branch.
**Why:** Unlike fused int8 kernels, this standalone GEMV has no serial RMSNorm prologue, so K_SPLIT=2 converts grid fill into useful wall-time.

## 2026-07-23 — DeepSeek QMoE validation and Phi capture priority

### Establish DeepSeek-V2-Lite as the real-scale native CUDA MoE target
**By:** Batty
**What:** Mobius PR #404's expert-major int4 emitter produces one QMoE per routed layer with fused/interleaved FC1, FC2, f32 scales, asymmetric zero-points on inputs 11/12, explicit routing/aggregation weights, and separate shared experts. The first DeepSeek MLA export must retain standard Attention when K/V head dimensions differ (192 versus 128), because GQA requires equal widths.
**Why:** DeepSeek-V2-Lite (64 routed experts, top-6) is the smallest practical target covering MLA and real QMoE breadth. The native QMoE ABI generalizes to that geometry; exporter packing and an MLA-aware capture-safe latent-cache boundary, not a kernel expert-count limit, remain the production work.

### Verify native DeepSeek QMoE structural and f16 execution smoke
**By:** Fact Checker and Batty
**What:** A random-weight two-layer 64-expert/top-6 asymmetric-int4 DeepSeek artifact matches the QMoE ABI and graph contract (one QMoE, two standard Attention, no GQA, 16 MatMulNBits) and completed strict-CUDA decode with zero fallbacks and finite output. The f16 root cause was router MatMul mixing f32 hidden states with f16 gate weights, not Attention-mask or RoPE dtypes; shared `_router_logits()` now casts both operands to f32. The corrected f16 smoke completed 32 finite tokens at 2.534 ms/token (394.64 tok/s), with 107 targeted tests and Ruff clean.
**Why:** This proves emitter-to-native wiring and low-precision router correctness, but random weights make it structural rather than semantic validation. ORT GenAI 0.14.1 rejects `model_type=deepseek_v2`; full-weight export, semantic comparison, and native smoke remain in flight.

### Prioritize fixed-decode Greater/If capture specialization for Phi
**By:** Marsten
**What:** At 193.90 tok/s (5.157 ms/token), Phi has only 2.899 ms/token of GPU kernels; 2.258 ms/token (43.8% of wall time) is dispatch/capture-seam overhead. Eager `Greater` metadata handling and host-side `If` control flow split decode into three CUDA graphs. A fixed-predicate, recapture-on-change specialization has an approximately 0.91 ms/token recoverable budget versus ORT.
**Why:** GEMVs consume 86.02% of GPU time and are already at their established roofline, while capture-safe control flow directly targets the demonstrated wall-time gap.

### Close warp-cooperative Phi gate/up GEMV as a no-go
**By:** Deckard
**What:** Removing the staged shared activation path regressed the asymmetric Phi gate/up kernel from about 31.5 to 50.7 µs (+61%). Each of eight warps redundantly recomputed RMS normalization; the prototype also forced a local-memory round-trip.
**Why:** Shared staging is load-bearing cross-warp reuse, not removable overhead. Further fused gate/up micro-optimization should yield to capture seams or other kernels.

## 2026-07-23 — Phi on-device LongRoPE and real-weight scoreboard

### Complete Phi capture-seam groundwork
**By:** Deckard
**What:** Capture-safe scalar `Greater` metadata caching and invariant-`If` memoization reduced Phi LongRoPE decode from three captured regions to two. The predicate remains live, changing-capture branches are never memoized, and branch flips invalidate capture safely; 192 CUDA tests and session control-flow tests passed.
**Why:** The groundwork removed repeated constant-cache materialization while preserving the shape and control-flow guards required for a subsequent fully on-device select.

### Adopt the on-device constant-select lowering for LongRoPE
**By:** Deckard; reviewed by Gaff
**What:** Merged as `97c1a56`, `CudaOnDeviceConstantSelect` lowers a loop-invariant scalar-predicate `If` whose branches are pure constants to capture-safe CUDA `Where` nodes. Equal shapes lower directly; unequal leading dimensions lower only when a scalar integer `Greater`/`GreaterOrEqual` threshold exactly matches the smaller false table, the true table is larger, and trailing dimensions agree. The false table is zero-padded to a fixed large output. Phi reduced two captured regions to one, eliminated eager rejections, and improved 203.50→322.15 tok/s (+58.3%) in idle-GPU interleaving; 160-token and 4,200-token boundary output were byte-identical, with 201/0 CUDA tests.
**Why:** Keeping selection and the live predicate on device removes the per-token LongRoPE host `If` seam without stale-branch risk. The deliberately narrow unequal-shape guard makes appended zero rows unreachable while the short table is selected.

### Record the post-fix Phi profile and independent benchmark
**By:** Marsten
**What:** Before the device-select landing, fixed main at `719d2fe` still spent 1.935 ms/token in the eager LongRoPE `If`, versus 2.948 ms in GPU kernels, making the host branch the primary remaining lever. After `97c1a56`, four accepted nine-run measurements on idle GPU 3 established a 321.98 tok/s Phi median: +40.22% versus canonical ORT and +57.2% versus the session's `origin/main` baseline; two nondeterministic shared-host launches were excluded by the harness.
**Why:** The profile correctly prioritized de-hosting control flow over GEMV tuning, while the independent result establishes the real-weight score with an explicit shared-host caveat.

### Retain the pre-de-hosting cumulative Phi frontier as historical context
**By:** Marsten
**What:** At `4e774ee`, after fused gate/up int4 prefetch and standalone int8 zero-point split-K, Phi reached a 193.32 tok/s seven-run median with a 121.21–194.67 tok/s shared-host spread and zero fallbacks. Qwen2.5-1.5B and DeepSeek-R1-Distill-Qwen-1.5B remained within noise at 617.90 and 622.66 tok/s.
**Why:** The result records the validated GEMV frontier and its contention caveat; it must not be confused with the subsequent, much larger LongRoPE seam elimination.

### Declare native CUDA faster than ORT on all available real-weight models
**By:** Marsten and Deckard
**What:** The authoritative real-weight scoreboard is Qwen2.5 0.5B +62.7%, 1.5B +36.8%, 7B +10.8%, and Phi-4-mini +40.2% versus ORT after LongRoPE de-hosting. The previous Phi reference was 193.89 versus 229.62 tok/s; the on-device select removes that deficit.
**Why:** The faster-than-ORT mandate is achieved for every available real-weight model, so future optimization work should use this scoreboard rather than the obsolete pre-fix Phi shortfall.

### Bound DeepSeek-V2-Lite full-depth validation honestly
**By:** Batty
**What:** No local or Foundry DeepSeek-V2-Lite source checkpoint weights are available. A deterministic synthetic 27-layer artifact at real configuration geometry (`26` QMoE, `27` Attention, FC1 `[64,2816,1024]`, FC2 `[64,2048,704]`) completed strict native CUDA with zero fallbacks at 26.66 tok/s; ORT GenAI rejects the `deepseek_v2` architecture.
**Why:** This validates full-depth native int4-QMoE wiring and the f16 router fix, but it is structural synthetic-weight evidence, not a real-weight semantic or ORT comparison.
