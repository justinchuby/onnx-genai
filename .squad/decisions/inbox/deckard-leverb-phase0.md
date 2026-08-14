### 2026-08-14 — Lever B Phase-0 capture-stability probe: NO-GO (gated re-test on cheap Increment-0)

**By:** Deckard (CUDA/decode performance engineer) · branch `squad/leverb-phase0-capture-probe`

**What:** Ran the cheap Phase-0 capture-stability probe on glm-4-9b-int4 (H200) to answer the one load-bearing Lever B question — *can a fixed-shape padded M=K forward be captured, replay ~1 dispatch/verify across bucket growth, and cost ≈ one M=1 replay?* Result by the strict "PASS = all three" rule: **(b) PASS, (a) FAIL, (c) UNMEASURED → NO-GO** to an unconditional multi-week Lever B commit.

Measured (reproduced GPU 6 & 7):
- **(b) capture stability — PASS.** 1000 real M=1 steps → captures=3, replays=994, invalidations=3, kv_growth_events=2; **90.3 tok/s** captured; no thrash (contrast eager verify 6→280). The capture state machine is stable across KV bucket growth on the exact kernels an M=K graph inherits.
- **(a) instantiates capture-safe — FAIL, non-fundamental.** Real M=K `try_capture` returns `NotCapturable`: (1) `[1,K,vocab]` logits are materialized (persistent binding is `[1,1,vocab]`); (2) ~722 transient workspace allocs inside the M=8 forward. Both are the missing **Increment-0** work (persistent `[1,K_max,vocab]` logits binding + pre-allocated alloc-free M=K workspace + KV-symbol pin), not a kernel veto.
- **(c) captured M=K wall ≈ M=1 wall — UNMEASURED (blocked by a).** Eager proxy: M=1 13.5 ms, M=2 80.4 ms, M=8 91.5 ms → **6.77×**, a STEP FUNCTION: **M=1→M=2 cliff = 66.9 ms**, near-flat **M=2→M=8 tail = 1.85 ms/row**. The cliff's composition (dispatch, which capture removes, vs generic-GEMM + alloc, which it may not) is the pivotal unknown Phase-0 could not resolve. M=8 logit readback 4.62 MB (small, as predicted).

**Why:** Lever B's central premise — "K× compute is free (GPU idle, weights read once) so a captured M=K replay ≈ one M=1 replay" — is exactly what a wall-clock capture measurement would confirm, and it **cannot be measured today** because the M=K forward is not capturable. The only proxy (eager) is a 6.8× upper bound dominated by an ~80 ms floor 7× the 11 ms captured M=1 cost, of unproven composition. Do not fund the multi-week build on an unconfirmed premise.

**Recommendation (gate, days not weeks):** Fund **Increment-0** (persistent `[1,K_max,vocab]` logits binding + fixed alloc-free M=K workspace + pin KV seq-symbol), then re-run this probe. Decision rule:
- captured M=8 replay ≈ M=1 replay (~11–15 ms) → **GO** on full Lever B build.
- ~80 ms floor persists under capture → **NO-GO**; promote **Lever A (Marlin int4 relayout, unconditional ~1.3–1.6×)** to primary.
(b) passing means the capture machinery is not the risk; the risk is solely whether captured M=K compute stays flat — Increment-0 measures that directly.

Deliverables on branch: `#[ignore]`d un-wired probe (`crates/onnx-genai-engine/src/native_decode/leverb_phase0_probe.rs` + `#[cfg(test)]` helpers in `cuda.rs`/`load.rs`) and findings note `docs/research/leverb-phase0-capture-probe.md`. Coordinator opens/admin-merges the PR.
