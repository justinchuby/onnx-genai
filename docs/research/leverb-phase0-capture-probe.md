# Lever B Phase-0 — capture-stability probe (MEASURED)

**Author:** Deckard (CUDA/decode performance engineer)
**Date:** 2026-08-14
**Branch:** `squad/leverb-phase0-capture-probe` (worktree `deckard-captureverify`, based on `main` 2689e8e8)
**Model:** glm-4-9b-int4-cuda (`/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda`) — the decode-bound reference from the speculative KILL test. vocab 151552, 40 GQA layers, `num_kv_heads=2`, `head_dim=128`.
**HW:** 1× H200 (verified idle, pinned via `CUDA_VISIBLE_DEVICES`). Numbers reproduced on GPUs 7 and 6.
**Probe:** `crates/onnx-genai-engine/src/native_decode/leverb_phase0_probe.rs` (`#[ignore]`d, UN-WIRED) + throwaway `NativeDecodeSession::leverb_phase0_capture_attempt` (cuda.rs, `#[cfg(test)]`).

---

## 0. The one question Phase-0 had to answer

Before committing multi-week eng to Lever B (capture-stable padded M=K verify), answer the single load-bearing question:

> Can a fixed-shape, padded **M=K** forward graph be **captured**, **replay stably** across ~1000 steps (including bucket-growth boundaries) at **~1 dispatch/verify**, and cost **≈ ONE M=1 replay** (not ~K×)?

**PASS = all three:** (a) instantiates capture-safe · (b) replays ~1 dispatch/verify across bucket growth · (c) per-verify replay wall ≈ M=1 wall.

Acceptance logic, draft models, KV-commit correctness, and the exact-greedy near-tie guard are **out of Phase-0 scope**. The probe deliberately dirties device KV and discards each session.

---

## 1. Exact commands

```bash
cd /home/justinchu/onnx-genai/.worktrees/deckard-captureverify
source /home/justinchu/onnx-genai/.cudaenv.sh
# ALWAYS re-check idle + pin a high-index idle GPU before every run:
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

cargo test -p onnx-genai-engine --features native-cuda --release --no-run

CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_RUN_CUDA_SMOKE=1 \
  ONNX_GENAI_LEVERB_MODEL=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda \
  cargo test -p onnx-genai-engine --features native-cuda --release \
  --lib leverb_phase0_capture_probe -- --ignored --nocapture
```

## 2. Probe design (measure, do NOT ship)

The probe uses only the real decoder, real persistent KV/mask bindings, and the real batched GQA/GEMM (`MatMulNBits`) kernels — no hand-rolled toy graph.

- **Part B — capture stability (criterion b).** 1000 real M=1 greedy decode steps with `graph_capture=on`, `kv_max_len=2048` so the run crosses the 256→512→1024 KV bucket-growth boundaries. Reads `cuda_kv_debug_stats().graph.{captures,replays,invalidations}` and `kv_growth_events`. This is the exact "freeze mask to the physical bucket, re-capture only on bucket growth" state machine (`native_decode/cuda.rs` `run_one_token`) on the exact kernels the M=K graph would inherit.
- **Part C — M-scaling wall sweep (criterion c, eager proxy).** For M ∈ {1,2,4,8}, times a single eager `decode_verify` forward (real M=K path), `rewind`ing between iterations. Each is ONE forward (same op/dispatch count), so the wall-vs-M curve isolates how compute + activation + logit-readback + workspace-alloc scale with M.
- **Part A — real M=K capture ATTEMPT (criterion a).** `leverb_phase0_capture_attempt` pads to a fixed `[1, K_max]`, builds the token/position inputs, `extend_mask`es to the bucket, and calls `try_capture_with_device_bindings` against the existing persistent bindings — reporting `Captured`/`NotCapturable{reason}`, segment count, and the device (alloc,free) delta across the capture run. If captured, it replays and times each replay (synchronized by a logits-binding read).

**What the probe could NOT exercise (stated up front, per the honesty mandate):** it could not instantiate an actual captured M=K graph — see §3(a). Building that is itself the first Lever B increment. So criterion (c) is reported via the eager proxy + curve decomposition, not a captured M=K replay number.

## 3. Measurements (medians; reproduced on 2 GPUs)

### (a) Instantiates capture-safe? — **NO (blocked, with a concrete fix)**

```
M=8 capture: captured=false segments=0 bucket=256 alloc_delta=(724 alloc, 722 free)
             decline="CUDA graph capture rejected: <graph> — every graph output
                      must use a persistent device binding during capture"
M=1 capture: captured=false ... (same decline reason)
```

Two concrete, **non-fundamental** blockers, both in the un-built M=K path:
1. **Materialized logits output.** The persistent logits binding is single-token-shaped `[1,1,vocab]`; an M=K forward emits `[1,K,vocab]`, which does not fit, so the runtime materializes it to host — and capture requires every graph output to be a persistent device binding. **Fix: add a persistent padded `[1, K_max, vocab]` logits device binding.**
2. **~722 transient workspace alloc/free** inside the M=8 forward (capture-mode allocator count). A captured region must contain no alloc/free (#854/#867). **Fix: pre-allocate a fixed M=K scratch workspace so the forward is alloc-free.**

Both are exactly what the M=1 captured path already does for its shapes; neither is a kernel-level veto.

### (b) Replays ~1 dispatch/verify across bucket growth? — **YES (clean pass)**

```
steps=1000  captures=3  replays=994  invalidations=3  growth_keeps=0  kv_growth_events=2
median M=1 captured step wall = 11.08 ms (90.3 tok/s)
```

994/1000 steps are single-graph replays; `captures=3` = initial + 2 bucket growths; `invalidations=3` ≈ growth count. **No thrash** (contrast today's eager verify 6→280). The real capture machine is stable across bucket growth on these exact kernels — the property the M=K graph inherits structurally.

### (c) Per-verify replay wall ≈ M=1 wall? — **NOT DEMONSTRATED (blocked by (a); eager proxy is a red-flagged upper bound)**

Eager `decode_verify` forward wall vs M (single forward each):

| M | eager forward wall | note |
|---|---|---|
| 1 | **13.5 ms** | fast single-token path (captured M=1 = 11.1 ms) |
| 2 | **80.4 ms** | falls off the M=1 fast path |
| 4 | 83.3 ms | |
| 8 | **91.5 ms** | |

**Curve decomposition:**
- **M=1→M=2 CLIFF = 66.9 ms** — the cost of leaving the single-token fast path (per-op host dispatch across 40 layers + generic multi-row GEMM kernels + workspace alloc). This is the un-captured, capture-abandoning regime — the same regime that made eager-M=K speculative a KILL.
- **M=2..M=8 TAIL slope = 1.85 ms/row** — the marginal cost of each *extra* verify row is small: the M=2→M=8 wall grows only 80.4→91.5 ms (+14%) for 4× the rows.

The eager M=8/M=1 ratio is **6.77×** — but this is an **upper bound**, not the captured cost: it is dominated by the 66.9 ms cliff (per-op dispatch + alloc + generic-GEMM kernel selection) that CUDA-graph capture is designed to remove. Whether capture actually collapses that cliff (as it does for M=1: eager 13.5 → captured 11.1 ms) is **the pivotal unknown that Phase-0 could not measure**, because the M=K forward cannot be captured today (§3a).

**Logit readback:** M=8 host logit readback = 4.62 MB (K·vocab·4B). Small, as predicted — not a bottleneck.

## 4. Verdict — **NO-GO to an unconditional multi-week commit; GATED-GO on a cheap Increment-0**

By the strict "PASS = all three" rule: **(b) PASS, (a) FAIL, (c) UNMEASURED → NO-GO.**

The decisive point: **Lever B's load-bearing number (captured M=K replay ≈ M=1 replay) is unmeasurable without a build step**, and the only available proxy (eager) shows a 6.8× ratio whose dominant component (the 66.9 ms M=1→M=2 cliff) has an **unresolved composition** — part is per-op dispatch that capture removes, part may be generic-GEMM kernel cost + alloc that it does **not**. The near-flat M=2→M=8 tail (1.85 ms/row) is encouraging (extra rows are cheap), but it sits on an ~80 ms floor that is 7× the 11 ms M=1 captured cost, and nothing in this probe proves capture removes that floor.

### Recommended gate (days, not weeks) — do this before funding the multi-week build

**Increment-0 (capture-enablement only):**
1. Persistent padded `[1, K_max, vocab]` logits device binding.
2. Pre-allocated fixed M=K decode scratch workspace (zero the ~722 transient allocs).
3. Pin the KV seq-symbol for the M=K shape (reuse `pin_fixed_capacity_kv_capture_symbols`).

Then re-run this exact probe — the M=K `try_capture` will now instantiate, and Part A's captured-replay timing yields the **decisive captured-M=8-replay-vs-M=1 number**:
- If captured M=8 replay **≈ M=1 replay (~11–15 ms)** → the cliff was dispatch/alloc, K× is effectively free → **GO** on the full multi-week Lever B build (verify graph + KV commit + near-tie guard + draft sources).
- If the **~80 ms floor persists under capture** → K× is NOT free on this model → **NO-GO**, and **Lever A (Marlin int4 relayout, unconditional ~1.3–1.6×) is promoted to primary.** Per the settled diagnosis, Lever A is the unconditional fallback.

### Why this is the right call
Increment-0 is cheap (it is a strict prerequisite of Lever B anyway) and converts the pivotal unknown into a measured number, protecting a multi-week bet whose central premise ("K× compute is free because the GPU is idle and weights are read once") this probe could **not** confirm at the wall-clock level and actively flagged with a 66.9 ms cliff of unproven composition. (b) passing means the capture machinery itself is not the risk; the risk is entirely whether captured M=K compute stays flat — which Increment-0 measures directly.

## 5. Reproducibility / artifacts

- Probe test: `crates/onnx-genai-engine/src/native_decode/leverb_phase0_probe.rs` (`#[ignore]`, gated on `ONNX_GENAI_RUN_CUDA_SMOKE=1`).
- Throwaway capture-attempt entry point: `NativeDecodeSession::leverb_phase0_capture_attempt` (`native_decode/cuda.rs`, `#[cfg(test)]`) + test-support loader `load_with_cuda_options_and_io_spec` (`native_decode/load.rs`, `#[cfg(all(test, feature = "cuda"))]`).
- None of the above is wired into any decode path.
- Raw run (GPU 6), verbatim:

```
[leverb-phase0][B] steps=1000 captures=3 replays=994 invalidations=3 growth_keeps=0 kv_growth_events=2 enabled=true decline=None
[leverb-phase0][B] median M=1 captured step wall = 11.077 ms (90.3 tok/s); logical_len=1013
[leverb-phase0][C] eager M=1: wall = 13.512 ms | device alloc/free per forward = 0/0
[leverb-phase0][C] eager M=2: wall = 80.368 ms | device alloc/free per forward = 0/0
[leverb-phase0][C] eager M=4: wall = 83.324 ms | device alloc/free per forward = 0/0
[leverb-phase0][C] eager M=8: wall = 91.462 ms | device alloc/free per forward = 0/0
[leverb-phase0][C] curve decomposition: M=1->M=2 CLIFF = 66.856 ms | M=2..8 TAIL slope = 1.849 ms/row
[leverb-phase0][A] M=8 capture: captured=false ... alloc_delta=Some((724, 722)) decline="... every graph output must use a persistent device binding during capture"
[leverb-phase0][VERDICT] GO/NO-GO = NO-GO
```

---

# Increment-0 — capture-enablement build + the DECISIVE captured number (MEASURED)

**Date:** 2026-08-14 (follow-up, Justin greenlit proceeding from the Phase-0 NO-GO)
**Branch:** `squad/leverb-increment0-capture` (off `squad/leverb-phase0-capture-probe`)
**Probe additions:** `NativeDecodeSession::leverb_increment0_capture_attempt` (`native_decode/cuda.rs`, `#[cfg(test)]`) + PART D of `leverb_phase0_probe.rs`. Still `#[ignore]`d and UN-WIRED.
**HW:** 1× H200, verified idle, pinned. Numbers reproduced 3× (GPU 7 twice, GPU 6 once) — deterministic.

## I0.0 What Increment-0 built (the three Phase-0 (a)-FAIL fixes, as a test-only overlay)

1. **Persistent padded `[1, K_max, vocab]` logits device binding** for the logits output name — so an M=K forward's `[1,K,vocab]` logits land in device memory instead of materializing to the host (the exact reason Phase-0's raw M=K `try_capture` was rejected).
2. **Alloc-free captured region via a pre-capture warm forward at the M=K shape** — a normal `run_with_device_bindings` at `[1,K]` grows the capture-safe scratch arena *before* `BeginCapture`, so the captured region itself does zero device alloc/free.
3. **KV-symbol pin** — inherited from the constructor (already pins fixed-capacity KV seq symbols for a graph-enabled session); no new work.

These are applied by swapping padded token/position/logits bindings into the persistent binding vector for the duration of the attempt (then restored), and driving the **real** batched GQA/GEMM M=K kernels through the **real** persistent KV/mask bindings — no toy graph.

## I0.1 Result — criterion (a) now PASSES

```
INC0 M=8 capture: captured=TRUE  segments=41 warm_alloc=(922,722) capture_alloc=(0,0) decline=None
INC0 M=1 capture: captured=TRUE  segments=1  warm_alloc=(0,0)     capture_alloc=(0,0) decline=None
```

The three fixes work: the padded logits binding absorbs the output (`decline=None`), and the warm forward moves **all** arena growth out of the captured region (`capture_alloc=(0,0)`). **The M=K forward now instantiates capture-safe.** The Increment-0 build recipe is validated.

## I0.2 The DECISIVE number — criterion (c): captured M=K replay wall vs captured M=1 replay wall

```
DECISIVE captured replay wall: M=8 = 87.2 ms | M=1 = 10.2 ms | ratio = 8.58×   (reproduced 8.58–8.59×)
```

**The ~80 ms floor PERSISTS under capture.** Capture removed the M=1 dispatch overhead (eager M=1 13.4 ms → captured M=1 10.2 ms, ~3 ms) but did essentially **nothing** to the M=K cost: captured M=8 (87.2 ms) ≈ eager M=8 (90.9 ms). Verifying 8 tokens in one "captured" replay costs **8.6× a single-token replay, not ≈1×.** The Phase-0 M=1→M=2 cliff (66.9 ms) was **not** dispatch — capture, which kills dispatch, left it intact. The Lever B premise "K× compute is free because the GPU is idle and weights are read once" is **falsified** at K=8 for this int4 decode-bound model.

## I0.3 Root cause — the M=K forward does NOT whole-graph capture; every hot kernel opts out at query-width > 1

`segments=41` (not 1) at M=8 is the tell. The captured M=8 "graph" is 41 tiny captured glue segments stitched by hundreds of **eager seam nodes**. The seam census (`capture_segmentation()`):

```
INC0 M=8 seam nodes:
  GroupQueryAttention[KernelCaptureUnsupported]            × 40   (one per layer)
  MatMulNBits[KernelCaptureUnsupported]                    × 240  (the int4 GEMV/GEMM — 6 per layer)
  MatMul[KernelCaptureUnsupported]                         × 1
  SkipSimplifiedLayerNormalization[KernelCaptureUnsupported] × 80  (two per layer)
```

**Every hot decode kernel — all 40 GroupQueryAttention, all 240 MatMulNBits, all 80 SkipSimplifiedLayerNormalization — returns `KernelCaptureUnsupported` at M>1.** They advertise capture support **only** at the M=1 decode shape; at query-width K they are each a forced eager seam. So the "captured" M=K forward degrades to ~361 eager per-op relaunches — precisely the dispatch-bound eager regime the entire lever was meant to escape. This is a **structural** result (deterministic, not a timing artifact): it is a property of the CUDA kernels' `capture_support()` at M>1, independent of workspace, logits binding, or KV pin — all of which Increment-0 already solved.

## I0.4 VERDICT — **NO-GO for Lever B. Promote Lever A (Marlin int4 relayout) to primary.**

| Criterion | Phase-0 | Increment-0 | 
|---|---|---|
| (a) instantiates capture-safe | FAIL (materialized logits + allocs) | **PASS** (padded logits + alloc-free warm) |
| (b) replays ~1 dispatch/verify across growth | PASS (994/1000, 3 invalidations, 90 tok/s) | PASS (unchanged) |
| (c) captured M=K wall ≈ captured M=1 wall | unmeasured | **FAIL — 8.58× (floor persists)** |

Increment-0 did its job: it converted the pivotal unknown into a hard, reproduced number, and the number is a clean NO-GO. Fixing (a) exposed that (c) fails for a **deeper, more expensive** reason than dispatch — the three hottest op families do not support CUDA-graph capture at query-width > 1.

### What a real Lever B would now require (why it is no longer "3–5 eng-weeks reuse existing machinery")
To make an M=K verify replay actually cost ≈ one M=1 replay, someone must **add M>1 CUDA-graph capture support to three kernel families**: `GroupQueryAttention` (×40), `MatMulNBits` (×240, the int4 GEMV/GEMM), and `SkipSimplifiedLayerNormalization` (×80). Each must guarantee, at query-width K: a static launch grid, no mid-kernel `cudaStreamSynchronize`, no host read-back, and no device alloc/free — the same bar the M=1 variants already meet, re-established for the batched shape. That is a deep, multi-family kernel-capture-support effort, not a "verify sub-graph" build, and it must clear the same numerical near-tie guard (#935) on top. The payoff remains only the **conditional** floor≈1.0×/ceiling~2–3×, and even that is contingent on the K× GEMM/attention arithmetic being genuinely free once captured — which the eager M=2..8 tail (1.76–1.85 ms/row) suggests but this probe could not confirm under capture (no hot kernel captures at M>1 to measure it).

Against that, **Lever A (Marlin int4 relayout) is unconditional ~1.3–1.6× on every token of every workload, at ~4–6 eng-weeks with no capture-support prerequisite.** With Lever B's cost/risk now materially higher and its central premise falsified at the wall-clock level, **Lever A is the primary lever.** Lever B is not dead — it is **gated behind a kernel-capture-support program** (make GQA/MatMulNBits/SkipSimplifiedLayerNorm capture-safe at M>1), after which this exact probe can be re-run to get the true captured-M=K wall. Until then, do not fund the Lever B verify-graph build.

## I0.5 Increment-0 reproducibility

Build/run identical to the Phase-0 command (the probe test is the same `leverb_phase0_capture_probe`, now with PART D). Raw run (GPU 7), verbatim:

```
[leverb-phase0][D] INC0 M=8 capture: captured=true segments=41 rows=8 bucket=256 warm_alloc=Some((922, 722)) capture_alloc=Some((0, 0)) decline=None
[leverb-phase0][D] INC0 M=8 seam nodes (root cause of segmented capture): GroupQueryAttention[KernelCaptureUnsupported]×40, MatMulNBits[KernelCaptureUnsupported]×240, MatMul[KernelCaptureUnsupported]×1, SkipSimplifiedLayerNormalization[KernelCaptureUnsupported]×80
[leverb-phase0][D] INC0 M=1  capture: captured=true segments=1 rows=1 bucket=256 warm_alloc=Some((0, 0)) capture_alloc=Some((0, 0)) decline=None
[leverb-phase0][D] DECISIVE captured replay wall: M=8 = 87.185 ms | M=1 = 10.164 ms | ratio = 8.58x
[leverb-phase0][VERDICT] (a) capture-safe: inc0_capture=true capture_alloc_clean=true (phase0_raw_capture=false)
[leverb-phase0][VERDICT] (b) stable replay: replays/step=0.994 invalidations=3 growths=2 pass=true
[leverb-phase0][VERDICT] (c) wall≈M=1: DECISIVE captured_ratio=Some(8.578) captured_pass=false | eager_ratio(upper bound)=6.77 eager_pass=false
[leverb-phase0][VERDICT] GO/NO-GO = NO-GO
```
