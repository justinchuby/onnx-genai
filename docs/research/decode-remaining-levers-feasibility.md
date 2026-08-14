# Decode's two remaining big-build levers — feasibility & sequencing (DESIGN-ONLY)

**Author:** Roper (Architect; feasibility scoping — read-only codebase analysis, NO code/kernels/build)
**Date:** 2026-08-14
**Model:** Muse-Glimmer-30B, `cuda/int4` Olive package (52 layers, hidden 6656, intermediate 19968, heads 32, kv_heads 2, head_dim 128, vocab 202048, `tie_word_embeddings=false`). Config source: `docs/research/lowbit-quant-feasibility.md:13`.
**HW/regime:** H200 SXM (HBM3e ≈ 4.8 TB/s). M=1 captured decode, `ONNX_GENAI_CUDA_GRAPH=1`. Baseline ~47 tok/s (Muse-Glimmer) / 96.4 tok/s (glm-4-9b-int4, decode-bound reference).

> **Headline (read first).** Both remaining levers are real multi-week builds, but they attack **different binding constraints and are not symmetric.**
> **Lever A (Marlin int4 relayout)** improves the GEMV *kernel*, which is only ~61% of the token and runs at ~29% peak DRAM; because it is Amdahl-capped inside the GEMV and the weights are still read **once per token**, its end-to-end ceiling is **~1.3–1.6× and unconditional.**
> **Lever B (capture-stable padded M=K verify graph)** attacks the actual binding constraint — **dispatch.** A fixed-shape verify sub-graph that *replays* instead of re-launching commits up to K+1 tokens per replay, and — critically — a captured M=K replay reads the weights **once, exactly like M=1** (weight DRAM is independent of M), so it amortizes **both dispatch and weight-DRAM over K tokens.** Its payoff is **asymmetric: floor ≈1.0×, ceiling ~2–3×** on a decode-bound model at good acceptance, and it is the single prerequisite that unlocks prompt-lookup **and** EAGLE-3/MTP.
> **Recommendation: pursue B first**, gated on a cheap Phase-0 capture-stability probe. A is the unconditional fallback / parallel lever if B's Phase-0 fails. This ordering follows directly from the dispatch-bound diagnosis (`.squad/decisions.md`): a lever that removes dispatches per committed token strictly dominates one Amdahl-capped inside a single-token GEMV.

---

## 0. The settled diagnosis these levers must respect

All measured and merged this week (`.squad/decisions.md`):

- Native int4 batch-1 decode is **DISPATCH-bound.** CUDA-graph **capture is load-bearing**: greedy runs `captures=6, replays=1267, invalidations=6` and hits 96.4 tok/s on glm-4-9b **only** because it replays. Anything that abandons capture collapses.
- The dominant int4 GEMV sustains only **~29% peak DRAM**, co-bound **40.7% Long-Scoreboard load-latency AND 64.8% dequant-ALU** (#928). It is **~61% of the token** (39% non-GEMV tail).
- Three cheap levers are **closed**: kernel micro-opt = **+2.7%, Amdahl-capped, no hidden serial-dispatch floor** (fold-scale #928; `cp.async` **regressed −13%** at 4 B/lane because the current packed layout defeats wide loads; higher-way split-K flat). TP = **NO-GO for tok/s** (+104 all-reduces/token, `docs/research/tensor-parallelism-feasibility.md`). Eager-M=K speculative = **KILL** — verify abandons capture (replays 1267→25), **0.47–0.74× even at 96% acceptance** (#932/#935/spec-14b).
- Weight-byte reduction barely helps: byte-fold **−75% bytes → +2.8%** (`docs/research/lowbit-quant-feasibility.md:7`). The binding cost is latency/dispatch, not byte count.

Two facts from this diagnosis drive everything below:
1. **The GPU is ~99% idle during decode** (dispatch-bound). ⇒ Doing **K× the arithmetic inside one captured replay is nearly free.**
2. **Weight DRAM is per-*token*, not per-*M*.** A MatMulNBits at M=K reads its weight matrix **once** (only the tiny activation grows). ⇒ Verifying K tokens in one replay costs ≈ the DRAM + dispatch of **one** M=1 step.

---

## 1. Side-by-side comparison

| Axis | **Lever A — Marlin int4 relayout** | **Lever B — capture-stable padded M=K verify** |
|---|---|---|
| **What it attacks** | GEMV kernel efficiency (40.7% Long-Scoreboard + 64.8% dequant-ALU) via 16 B `cp.async.cg` loads + LOP3 dequant | The **dispatch binding itself** — replays a fixed-shape verify graph instead of re-launching K uncaptured kernels |
| **Mechanism ceiling** | Amdahl over the ~61% GEMV fraction. Even a 2.7× kernel → `1/(0.39 + 0.61/2.7)` ≈ **1.6×** | Commits up to **K+1 tokens per replay**; replay cost ≈ one M=1 step (weights read once, GPU idle) |
| **End-to-end token uplift** | **~1.3–1.6×**, **unconditional** (every token, every workload) | **Floor ≈1.0×; ceiling ~2–3×**, **conditional on acceptance** (draft quality / workload) |
| **Realized where** | Inside the captured replay (does NOT change dispatch structure — capture already handles that) | Removes replays-per-committed-token AND amortizes weight-DRAM over K |
| **Build cost** | **~4–6 eng-weeks.** From-scratch relayout kernel; **41 GEMV entry points** in `matmul_nbits.rs` consume the current layout and many carry fused epilogues (rmsnorm/swiglu/gate_up/down) that must be preserved; offline repack tooling; numerics gate | **~3–5 eng-weeks** for the verify graph (reuses existing bucket/mask-freeze capture machinery). +multi-week each for draft sources (prompt-lookup ~free; EAGLE-3/MTP = trained head) |
| **Numerical risk** | **Not byte-exact** — relayout reorders partial sums ⇒ **Chew gate + f64 oracle** required (mirrors #928 fold-scale failing the asymmetric-zp parity guard) | **Not byte-exact** either — near-tie FP divergence already root-caused (#935): M=K in-block K/V vs M=1 KV-cache reads flip argmax only where top1–top2 gap ≤~0.17. Fixable with a **near-tie guard** (re-decode ~4–9% of rows M=1) for exact greedy identity |
| **Structural risk** | Medium — well-understood kernel technique, but M=1 (pure GEMV, zero arithmetic reuse) is a **harder Marlin regime** than its published M=16–64 batched wins | **Higher** — must prove padded M=K stays capture-stable through KV commit + mask + variable past; batched GQA/GEMM M=K kernels must be capture-safe (no alloc/sync). But the machinery pattern already exists (§3) |
| **Portability (Rule 11)** | **SM80+ only** (`cp.async`/LOP3). Needs runtime arch guard + **fallback to current split-K layout** on <SM80 and CPU EP ⇒ **two weight layouts maintained**; repack adds a packaging step | **Arch-agnostic** core (any CUDA-graph-capable device; machinery already runs on all CUDA tiers). CPU EP already runs M=K eagerly ⇒ graceful degrade to today. Cleaner posture than A |
| **Unlocks** | Nothing beyond faster GEMV (also raises TP's Regime-B precondition toward bandwidth-bound) | **Prompt-lookup** (finally net-positive on repetitive/structured text) **AND EAGLE-3/MTP** (they reuse the same `decode_verify`, #935) |

### Why B's mechanism strictly dominates A's

At **M=1** the GEMV streams every weight once for **one** output token — it is pure DRAM streaming with **zero arithmetic reuse**. Lever A raises the *efficiency* of that stream (29% → maybe 55–70% DRAM), a bounded ~1.3–1.6× over 61% of the token.

At **M=K** the *same* weight stream produces **K** output rows — weight DRAM is unchanged, only the negligible activation matrix and K× ALU grow, and the GPU is idle so that ALU is ~free. Lever B therefore turns **one weight stream + one dispatch into up-to-K+1 committed tokens.** It multiplies throughput on the exact axis (dispatch, weight-read-per-token) that is binding, rather than shaving the per-stream efficiency. This is the concrete form of "a lever that attacks dispatch should beat one Amdahl-capped inside the GEMV."

---

## 2. RECOMMENDATION — build B first (capture-stable verify), keep A as the unconditional fallback

**Pursue Lever B first**, sequenced behind a Phase-0 capture-stability probe (§3). Rationale, grounded in the dispatch-bound diagnosis:

1. **B attacks the binding constraint; A does not.** Decode is dispatch-bound and capture is load-bearing (`.squad/decisions.md`). B removes dispatches (and weight-reads) per committed token; A leaves the dispatch structure untouched and only improves work the capture already amortizes.
2. **Asymmetric payoff vs Amdahl cap.** B's floor ≈1.0× (a capture-stable padded verify that accepts nothing still commits ~1 token per ~1 replay — never the 0.2–0.74× *loss* of today's capture-breaking verify) with a **2–3× ceiling** at high acceptance. A is a hard ~1.6× ceiling. The spec-14b result is explicit: at **96.1% acceptance speculative would win big IF verify didn't break capture** — B is exactly that fix.
3. **One build, two unlocks.** The capture-stable verify graph is the shared prerequisite for prompt-lookup **and** EAGLE-3/MTP (all route through `decode_verify`, #935). A unlocks nothing further.
4. **Cleaner portability.** B's core is arch-agnostic and degrades gracefully on CPU EP; A is SM80+-gated and forces two maintained weight layouts (Rule 11, `RULES.md:107-117`).

**Keep Lever A funded as the unconditional fallback**, and optionally in parallel: it is the only lever that helps *every* token on *every* workload regardless of acceptance, and it is the sole remaining single-GPU kernel win #928 identified ("bigger wins need a from-scratch Marlin kernel"). It also incrementally raises the bandwidth-bound fraction that TP's Regime-B GO is gated on (`docs/research/tensor-parallelism-feasibility.md`). If B's Phase-0 shows the padded verify cannot stay capture-stable, A becomes the primary.

---

## 3. Phase-0 microbench for the recommended lever (B) — the cheapest de-risking experiment

**Mirrors the TP doc's Phase-0 probe** (`tensor-parallelism-feasibility.md` §6/§7): one throwaway, `#[ignore]`, un-wired probe that answers the single load-bearing question before any multi-week commit — **can a fixed-shape padded M=K verify graph capture, replay stably across steps, and cost ≈ one M=1 replay?** Acceptance logic, draft models, and KV-commit correctness are explicitly **out of scope** for Phase-0.

**Why this is the right probe.** The existing single-token capture machinery already proves the pattern is viable: it **freezes the mask to the physical bucket** to stay capture-eligible and **re-captures only on bucket growth** (`native_decode/cuda.rs:912-919, 1077-1079, 2825-2848`); per-step token/position/mask are written to **device bindings**, not baked into the graph, so variable `past_len` needs no re-capture. Phase-0 tests whether that same discipline extends from M=1 to a fixed, padded M=K.

**Design (measure, do not ship):**
1. Pick **K=4**, pad to a fixed `K_max`. Build a device-resident `[1,K_max]` token/position buffer and a `[K_max × bucket]` causal+padding mask (constant node count).
2. `cudaStreamBeginCapture` → run the M=K forward against the existing persistent KV/mask bindings → end/instantiate. Confirm it **instantiates** (batched GQA/GEMM M=K kernels must contain **no alloc/free/sync** — the #854/#867 capture rules).
3. **Replay it across ~1000 simulated steps**, rewriting only the device token/position/mask each step and advancing `past_len` through ≥1 bucket-growth boundary. Assert **replay count stays ~1/verify** (not K) and invalidations stay ~= bucket-growth count — i.e. it does **not** thrash like today's eager verify (invalidations 6→280, `native_decode/cuda.rs:1229-1312`).
4. **Time per-verify replay wall vs an M=1 replay.** The decisive number: it must be **≈1×** (validating "K× compute is free / weights read once"), not ~K×. Also record the K×vocab host logit-readback cost (K·202048·4 B ≈ 3.2 MB at K=4 — expected small).

**Pass criteria (all three):** instantiates capture-safe · replays ~1 dispatch/verify across bucket growth · per-verify wall ≈ M=1 wall. **~S effort, very high information** — it converts the entire "is capture-stable verify feasible?" question from argument to measurement before a single eng-week of the real build.

**(Phase-0 for A, if run in parallel):** an M=1 Marlin-relayout GEMV microbench on one MLP shape (6656→19968) measuring **achieved DRAM %** vs the current 29%. Pass = ≥~55% at M=1; that is the precondition that separates a ~1.5× token win from mere fold-scale (+2.7%) territory.

---

## 4. Explicit go/no-go per lever, with the precondition that flips each verdict

### Lever B — capture-stable padded M=K verify graph
**🟢 GO to Phase-0; GO to build conditioned on Phase-0 passing.**
- **Feasibility is architecturally supported, not assumed:** the capture machinery already runs fixed-bucket, mask-frozen, device-input-driven graphs and re-captures only on growth (`native_decode/cuda.rs:912-919, 2825-2848`). Variable accepted length → **selective KV commit is capture-stable**: always write K physical KV slots (fixed shape), then set the **logical length** host-side to `past+accepted+1` (a pointer move, no graph change); unaccepted slots are overwritten next step. Variable K → **pad to `K_max`, mask the tail**. Exact-greedy identity → the **near-tie guard** from #935 (re-decode ~4–9% of rows M=1).
- **Precondition to flip to NO-GO:** Phase-0 shows either (a) the batched M=K GQA/GEMM kernels **cannot be made capture-safe** (hidden alloc/sync), or (b) per-verify replay cost scales **~K×** rather than ≈1× (meaning it is *not* amortizing dispatch/DRAM — the whole thesis). Either result kills B and promotes A to primary.
- **Payoff caveat (state honestly):** the tok/s *ceiling* still requires an acceptance source. Prompt-lookup gets B to net-positive only on repetitive/structured workloads; **workload-general** gains need EAGLE-3/MTP (a separate multi-week trained-head build). B's floor of ≈1.0× means it is **never a regression** once capture-stable — unlike today.

### Lever A — Marlin-style int4 weight relayout
**🟡 CONDITIONAL GO as the unconditional fallback / parallel lever.**
- **Precondition to flip to full 🟢 GO:** an M=1 Marlin GEMV microbench must lift achieved DRAM from ~29% to **≥~55% at batch-1** (§3). Marlin's published wins are at M=16–64; at M=1 the GEMV has **zero arithmetic reuse** and is Long-Scoreboard-bound, so the relayout's value rests entirely on wide `cp.async.cg` loads improving memory-level parallelism — plausible (it directly targets the 40.7% Long-Scoreboard + 64.8% dequant-ALU) but unproven at M=1.
- **Precondition to flip to NO-GO:** the M=1 microbench stays **<~40% DRAM** — then Marlin lands in fold-scale territory (~+2.7%, already shipped opt-in #928) and does not justify rewriting **41 GEMV entry points** + repack tooling + the Chew numerics gate.
- **Portability precondition (Rule 11):** must ship with a runtime SM80 arch guard and a **byte-identical fallback to the current split-K layout** on <SM80 / CPU EP; the relayout is opt-in and tier-scoped, never the default that could regress a consumer/edge tier.

---

## 5. Sources
- Diagnosis & measured levers: `.squad/decisions.md` (Decode perf REOPENED — dispatch-bound, capture load-bearing, #928/#932/#935/#933/spec-14b).
- GEMV efficiency (29% DRAM, 40.7% Long-Scoreboard, 64.8% dequant-ALU, fold-scale +2.7%, cp.async −13%, ~61% GEMV fraction): #928, `.squad/decisions.md`.
- Speculative KILL + verify-abandons-capture (replays 1267→25, 0.47–0.74× @96% accept) + near-tie FP root-cause + EAGLE-3/MTP gate: #932/#935/spec-14b, `crates/onnx-genai-ort/src/native_decode/cuda.rs:1229-1312` (`decode_verify`, "invalidates any captured graph").
- Capture machinery (frozen-bucket mask, device-input writes, re-capture on growth): `crates/onnx-genai-ort/src/native_decode/cuda.rs:849-980, 912-919, 1054-1079, 2825-2848, 3164-3364`.
- GEMV entry-point surface (41 `_ENTRY` symbols incl. fused rmsnorm/swiglu/gate_up/down epilogues): `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs:21-186`.
- Byte-fold / roofline (−75% bytes → +2.8%, 15% roofline): `docs/research/lowbit-quant-feasibility.md:7,215`.
- Latency/dispatch-bound + graph-replay-amortizes-launches + cooperative-capture cleared: `docs/research/dense-decode-megakernel-feasibility.md`.
- TP NO-GO-for-speed + Phase-0-probe template: `docs/research/tensor-parallelism-feasibility.md`.
- Portability: `RULES.md:107-117` (Rule 11).
