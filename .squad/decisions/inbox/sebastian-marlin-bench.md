### 2026-08-14 — qwen numerics: parity "failure" is a FLAKY TILED ORACLE at a near-tie, NOT a Marlin regression

**By:** Sebastian (Perf/CUDA). Ran the qwen byte-identical parity test
(`marlin_m_gt_1_matches_tiled_on_qwen2_5_14b_int4`) at `3735d57e` to close qwen breadth. **First run FAILED**
— Marlin M>1 greedy stream diverged from the tiled reference at token index 19 (identical first 19 tokens,
then tiled `320,94409,701,15576,13` vs marlin `5692,374,264,1140,315`). Investigated rigorously before
raising a regression:

| Run | split-K | tiled reference @ tok19+ | marlin @ tok19+ | result |
|---|---|---|---|---|
| 1 | ON (default) | `320, 94409, …` | `5692, 374, 264, 1140, 315` | FAIL |
| 2 | OFF | `5692, 374, …` | `5692, 374, 264, 1140, 315` | pass |
| 3 | ON | `5692, 374, …` | `5692, 374, 264, 1140, 315` | pass |
| 4 | ON | `5692, 374, …` | `5692, 374, 264, 1140, 315` | pass |

- **Marlin output is DETERMINISTIC** — `5692,374,264,1140,315` in all 4 runs, split-K on *and* off. **The
  tiled reference is the flaky one** (divergent on 1 of 4 runs). Token index 19 is a **near-degenerate
  argmax**; the tiled GEMM's own run-to-run nondeterminism (atomic/reduction-order accumulation) occasionally
  flips it. Marlin (fixed split-K reduction) is actually *more* stable here.
- **Conclusion: NOT a Marlin numerics regression.** My initial alarm was wrong; rigorous A/B + repeat showed
  Marlin is the deterministic path and the tiled oracle flakes ~1-in-4 at this near-tie. Deckard's
  "qwen byte-identical" gate passed for him because it landed on the majority (matching) coin-flip.
- **Action item for Chew/Gaff (merge review of #960):** the qwen exact-token-stream parity test is
  **flaky (~25%)** because it asserts equality against a nondeterministic tiled reference at a near-tie. Not a
  blocker, but the assertion should be hardened — e.g. compare logits within tolerance, use a
  deterministic tiled reduction for the oracle, or assert Marlin's own run-to-run determinism instead of
  exact-match to a flaky tiled stream. glm's test prompt doesn't hit such a near-tie, so glm passes cleanly.

#### Reproduce

```bash
QWEN=/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx  # QWEN2_5_14B_CUDA_E2E_DIR default
# Run several times: marlin stays fixed, tiled reference occasionally flips tok19 (near-tie)
CUDA_VISIBLE_DEVICES=7 cargo test -p onnx-genai-engine --features cuda,native-backend --release \
  --test marlin_m_gt_1_e2e marlin_m_gt_1_matches_tiled_on_qwen2_5_14b_int4 -- --ignored --nocapture
# split-K off (also byte-identical to marlin): ONNX_GENAI_MARLIN_SPLITK=0 prefix
```

---

### 2026-08-14 — Cross-model capture re-probe @ `3735d57e` — glm block_128 (gate_up N/A) vs qwen block_32 — **qwen B\* ≈ 4.7× (NOT GO)**

**By:** Sebastian (Perf/CUDA). Follow-up to the glm re-probe below. Deckard's `3735d57e` routes the **fused
gate_up SwiGLU** MLP node through split-K — but that fused node is **only produced for `block_size==32`
models** (hard requirement in `run_f16_gate_up_swiglu`, `matmul_nbits.rs:5782`). Verified the two models'
ONNX (`onnx` 1.22): **glm-4-9b `block_size=128`, qwen2.5-14b `block_size=32`.**

- **glm is NOT eligible for the gate_up fusion (block_128)** → `3735d57e` is a **no-op for glm**. glm's 240
  MatMulNBits run as separate gate/up ops through the *general* Marlin M>1 + split-K path (the `4abe4e57`
  retune). This is exactly why the glm captured wall stayed flat (22.0→21.9 ms) after `3735d57e`. **glm
  B\*≈2.16× is final and correct; it comes from the general split-K retune, not gate_up fusion.**
- **qwen IS eligible (block_32).** qwen M=8 verify capture re-probe @ `3735d57e` (reproduced ×2, GPU 7):

  | Model | segments | seam census | captured M=8 | captured M=1 | **B\*** | eager M8/M1 |
  |---|---|---|---|---|---|---|
  | **glm-4-9b** (block_128) | 1 | `<none: whole-graph>` | 21.9 ms | 10.2 ms | **2.16×** | 2.13× |
  | **qwen2.5-14b** (block_32) | 1 | `<none: whole-graph>` | 35.5 ms | 7.5 ms | **4.62–4.72×** | 3.73–3.80× |

- **Capture-safety generalizes to qwen** — segments=1, whole-graph, **zero unsupported nodes** (all 48 GQA +
  96 SkipSimplifiedLayerNorm captured, same as glm). The capture arc is model-independent. ✅
- **But qwen is NOT at GO (B\*≈4.7×).** Two honest contributors: (a) **denominator effect** — qwen's *tuned*
  block-32 M=1 decode is fast (captured 7.5 ms vs glm 10.2 ms; capture speeds M=1 by 34% vs M=8 by 18%
  because M=1 is more launch-bound), which *inflates* the ratio; (b) qwen's 14B/48-layer M=8 verify is
  genuinely heavier (35.5 ms). The `captured_ratio > eager_ratio` (4.7 > 3.8) is the denominator effect, not
  a capture inefficiency. Deckard's caveat #1 (qwen square attn-proj K=5120 cold-clock bimodal) may add to
  the M=8 weight.
- **Takeaway:** the small-M GEMM floor is **model-dependent**. glm reaches a practical GO (2.16×); qwen at
  4.7× needs either more small-M GEMM work (verify whether the gate_up fusion is actually firing on qwen and
  whether the attn-proj bimodal is real under sustained clocks) **or** deeper drafting to amortize. glm
  remains the canonical gate and is GO; qwen is a genuine second-model gap flagged for Deckard.

#### Reproduce (qwen, `3735d57e`)

```bash
QWEN=/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$QWEN \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
# block_size probe: python3 -c "import onnx;m=onnx.load(P,load_external_data=False);..."  (see log)
```

---

### 2026-08-14 — Marlin capture RE-PROBE @ `3735d57e` (gate_up SwiGLU split-K routing) — **B\* ≈ 2.16× (GEMM PLATEAU)**

**By:** Sebastian (Perf/CUDA). Merged Deckard's `3735d57e` / `dffabf0d`: `try_launch_marlin_gate_up_prefill`
(gate+up SwiGLU MLP — glm K=4096 N=13696 ×40, the single biggest MLP GEMM) was still DIRECT-launching and
bypassing split-K even after the `4abe4e57` factor retune; now both route through split-K (sk=8). H200
**GPU 7** (Deckard off 6), all idle. Canonical path = `ONNX_GENAI_MARLIN_M_GT_1=1` (split-K default).

#### Result (leverb-phase0 Part D INC0, glm-4-9b, M=8) — reproduced ×2

| Config (head) | segments | seam census | captured M=8 wall | **B\*** |
|---|---|---|---|---|
| M=8 split-K factor retune (`4abe4e57`) | 1 | `<none: whole-graph>` | 22.0 ms | 2.16× |
| **+ gate_up split-K routing (`3735d57e`)** | **1** | `<none: whole-graph>` ✅ | **21.9 ms** | **2.15× / 2.19×** |

- **B\* ≈ 2.16× — the GEMM plateau.** The captured M=8 verify wall is now stable at **~21.9 ms** (run-to-run
  ratio jitter 2.15–2.19× comes from the M=1 captured baseline 9.98–10.18 ms, not the M=8 wall). Routing the
  dominant gate_up MLP GEMM through split-K shaved the last easy small-M cost; segments=1, whole-graph, zero
  unsupported nodes; byte-identical greedy tokens (parity PASS, no NaN).
- **This is the small-M GEMM floor.** Deckard's honest read (concur): M=8 `mma.m16n8k16` is intrinsically
  ~half-occupancy (~50% MMA lane waste), so split-K + occupancy is the **last GEMM lever**. Beyond ~2.1×,
  the path to a *universal* ≤2 is **drafting depth** (more accepted tokens amortize the verify), not more
  GEMM tuning.
- **Speculative-capture verdict = GO in practice.** B\*≈2.16× vs the conservative break-even-at-2 line: any
  8-wide draft accepting **>2.16 tokens/verify on average** wins. Capture-safety is fully solved.

#### Full progression (the number that gates Marlin)

```
segments:  41  → 120 →  1  →  1  →  1  →  1   (BEFORE→…→whole-graph, zero seams)
B*:       8.76 → 4.99 → 5.09 → 2.71 → 2.63 → 2.16×
heads:     OFF   c692   c842  c842+sk 29714  4abe/3735   (byte-identical throughout)
```

#### Reproduce (`3735d57e`)

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh && git fetch origin && git merge 3735d57e
GLM=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
CUDA_VISIBLE_DEVICES=7 GLM_4_9B_CUDA_E2E_DIR=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release --test marlin_m_gt_1_e2e \
  marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4 -- --ignored --nocapture
```

---

### 2026-08-14 — Marlin capture RE-PROBE @ `4abe4e57` (M=8 verify split-K retune) — **B\* = 2.16×** (at the GO line)

**By:** Sebastian (Perf/CUDA). Merged Deckard's `4abe4e57` / `0143f235`: retuned `choose_split_k` for the M=8
verify shape (the old heuristic under-split — gate_up MLP GEMM was on the no-split slow path). New M=8 auto
factors split gate_up + down + o/q deeper (sk=8). Prefill (M>32) still returns sk=1 → byte-identical direct
kernel (prefill/decode measurements unchanged). H200 **GPU 7** (Deckard on 4/6), all idle. Canonical path =
`ONNX_GENAI_MARLIN_M_GT_1=1` (split-K default, sk auto=8 for the verify GEMMs).

#### Result (leverb-phase0 Part D INC0, glm-4-9b, M=8) — reproduced ×2

| Config (head) | segments | seam census | captured M=8 wall | **B\*** |
|---|---|---|---|---|
| lm_head plan, split-K default (`29714037`) | 1 | `<none: whole-graph>` | 26.8 ms | 2.63× |
| **+ M=8 split-K retune (`4abe4e57`)** | **1** | `<none: whole-graph>` ✅ | **22.0 / 22.1 ms** | **2.16× / 2.17×** |

- **B\* 2.63 → 2.16×** — Deckard's gate_up/down/o-q M=8 split-K retune dropped the captured verify wall
  26.8→22.0 ms. Reproduced across two runs (2.16×, 2.17×); Part-C eager upper-bound 2.13×, CLIFF 10.8 ms,
  TAIL slope 0.935 ms/row. Still segments=1, whole-graph, **zero unsupported nodes**.
- **Byte-identical** greedy tokens on glm-4-9b-int4 with the retuned path (parity PASS, no NaN).
- **This is the plateau: B\*=2.16× is an honest hair above the conservative ≤2 line.** But ≤2 is the
  break-even-at-2-accepted-tokens threshold; at B\*=2.16 speculative decode is a **practical GO** the moment
  the draft accepts **>2.16 tokens/verify on average** (trivially met by any decent 8-wide draft). The full
  arc: **segments 41→1 (whole-graph, zero seams); B\* 8.76→4.99→5.09→2.71→2.63→2.16×**, byte-identical
  throughout. Capture-safety is DONE; the residual 0.16× is pure small-M GEMM compute, not a capture blocker.

#### Reproduce (`4abe4e57`)

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh && git fetch origin && git merge 4abe4e57
GLM=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
CUDA_VISIBLE_DEVICES=7 GLM_4_9B_CUDA_E2E_DIR=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release --test marlin_m_gt_1_e2e \
  marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4 -- --ignored --nocapture
```

---

### 2026-08-14 — Marlin capture RE-PROBE @ `29714037` (lm_head dense-plan capture + split-K default) — **ZERO unsupported nodes, whole-graph capture**

**By:** Sebastian (Perf/CUDA). Merged Deckard's `29714037` / `1a72c865` onto `squad/marlin-bench`: (1) lm_head
`MatMul_node_734 → logits` now uses a cached dense-GEMM plan (`blas::CaptureGemmPlan` + `DenseGemmPlan` in
`kernels/matmul.rs`) — select algo + persistent workspace once at warmup, replay with no heuristic/alloc/sync,
so it **folds into the captured graph**; (2) **split-K is now default-on inside `ONNX_GENAI_MARLIN_M_GT_1=1`**
(no second flag; opt out with `ONNX_GENAI_MARLIN_SPLITK=0`). H200 **GPU 7** (Deckard on GPU 5, no contention),
all 8 verified idle. Canonical probe path = just `ONNX_GENAI_MARLIN_M_GT_1=1`.

#### Result (leverb-phase0 Part D INC0, glm-4-9b, M=8)

| Config (head) | **segments** | residual seam nodes | captured M=8 wall | **B\*** (M8/M1) |
|---|---|---|---|---|
| Marlin M>1 + GQA/LN safe (`c842b759`) | 1 | **MatMul×1** (lm_head) | 51.9 ms | 5.09× (split-K off) / 2.71× (split-K on) |
| **+ lm_head plan, split-K default (`29714037`)** | **1** | **`<none: whole-graph>`** ✅ | **26.8 ms** | **2.63×** |

- **Census is now TRUE ZERO unsupported nodes.** The probe reports `seam nodes: <none: whole-graph>` — the
  entire M=8 verify forward (including lm_head logits projection) captures as **one whole-subgraph replay**.
  Deckard's cached dense-plan closed the last `MatMul[KernelCaptureUnsupported]×1`. It needs the same M=8
  pre-warm (dense plan rejects an unwarmed shape during capture) — Part-D INC0's full-M=K warm covers it
  (`warm_alloc=(923,722)`, `capture_alloc=(0,0)`).
- **B\* 2.71→2.63×** (captured M=8 wall 27.5→26.8 ms): folding lm_head out of its per-step eager launch+sync
  nudged B\* down as Deckard predicted (~one small-M GEMM increment), but the strict **≤2 GO line is not yet
  crossed** — the residual is genuine small-M GEMM compute, not dispatch/segmentation. Eager upper-bound
  ratio 2.56×; Part-C CLIFF 16.98 ms, TAIL slope **0.769 ms/row**.
- **Numerics:** greedy tokens **byte-identical** to tiled on glm-4-9b-int4 with the new default path
  (split-K default + lm_head plan) — `marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4` PASS. No NaN.
- **This is the cleanest capture config to date:** zero unsupported nodes, whole-graph replay,
  byte-identical, B\*=2.63×. Speculative-capture verdict remains **structurally GO / numerically break-even
  pending** — 2.63 is just above ≤2; deeper drafting or one more small-M GEMM step reaches it.

#### Reproduce (`29714037`)

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh && git fetch origin && git merge 29714037
GLM=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
# Canonical probe — split-K is now DEFAULT inside MARLIN_M_GT_1=1 (no SPLITK flag)
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
# Byte-identical parity (new default path)
CUDA_VISIBLE_DEVICES=7 GLM_4_9B_CUDA_E2E_DIR=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release --test marlin_m_gt_1_e2e \
  marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4 -- --ignored --nocapture
```

---

### 2026-08-14 — Marlin capture RE-PROBE @ `c842b759` (GQA + LayerNorm M>1 capture landed) — **segments → 1**

**By:** Sebastian (Perf/CUDA). Merged Deckard's `c842b759` (commit `18d00f90`: `GroupQueryAttention`×40 +
`SkipSimplifiedLayerNormalization`×80 now advertise M>1 capture support) onto `squad/marlin-bench`. H200
GPU 7 (Deckard on GPU 4, no contention). `ONNX_GENAI_MARLIN_M_GT_1=1`, Part-D INC0 pre-warms the M=8 shape
(no cold-miss). Canonical = split-K OFF (per Deckard). Exploratory row adds `ONNX_GENAI_MARLIN_SPLITK=1`.

#### The capture-safety progression (leverb-phase0 Part D INC0, glm-4-9b, M=8)

| Config (head) | **segments** | residual seam nodes | captured M=8 wall | **B\*** (M8/M1) |
|---|---|---|---|---|
| BEFORE — Marlin OFF | 41 | GQA×40, **NBits×240**, MatMul×1, LN×80 | 87.3 ms | 8.76× |
| Marlin M>1 (`c692bffe`) | 120 | GQA×40, MatMul×1, LN×80 | 50.7 ms | 4.99× |
| **+ GQA/LN capture-safe (`c842b759`, CANONICAL)** | **1** ✅ | **MatMul×1** | 51.8 ms | **5.10×** |
| + split-K (exploratory, also capture-safe) | **1** | MatMul×1 | **27.3 ms** | **2.69×** |

- **`segments → 1` ACHIEVED.** Deckard's GQA+LN capture-safety cleared the last M>1 seams; the whole M=8
  verify forward now captures as a **single replayable graph**. Only residual seam: a lone
  **`MatMul[KernelCaptureUnsupported]×1`** (one dense MatMul — not NBits; Deckard to identify/relax). This
  is the operational win for speculative decode: **one graph replay, not 120 segment dispatches.**
- **But capture-safety alone did NOT move B\*** (5.10× vs the marlin-only 4.99× — within contention noise):
  collapsing 120→1 segments changed the captured wall by ~0 (50.7→51.8 ms). **The captured M=8 wall is
  compute-bound, not segmentation-bound** — segmentation overhead was already negligible, so the
  expectation "segments→1 ⇒ B\*≤2" does not hold on its own.
- **Split-K is the lever that actually moves B\*.** Exploratory `ONNX_GENAI_MARLIN_SPLITK=1` (which is **also
  capture-safe → segments=1, and byte-identical**, verified) nearly halves the captured M=8 wall again
  (51.8→**27.3 ms**), dropping **B\* 5.10×→2.69×** — right at the ≤~2 GO line (cliff 40.5→16.5 ms; eager
  M8/M1 2.51×). **Path to speculative-capture GO = capture-safety (done, segments=1) + split-K on the M=K
  verify.** B\*=2.69 is a hair above ≤2; one more small-M GEMM increment (or drafting slightly deeper)
  reaches break-even.
- **Numerics correct (Deckard's NaN check):** Marlin-ON greedy tokens are **byte-identical** to tiled-OFF,
  both **with and without split-K** (`marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4` PASS, 24/24 tokens);
  no NaN/garbage under capture → the zero-init KV-tail + on-device tail-masking assumption holds.
- **Warmup coverage confirmed (Deckard's must-handle #1):** Part-D INC0 (`leverb_increment0_capture_attempt`,
  `cuda.rs:1157` FIX 2) warms via a **full M=K forward** (`run_with_device_bindings` on the exact M=8
  bindings later captured) — one pass populates **every** op's fixed-shape cache (Marlin repack + scratch,
  GQA metadata, LN metadata). Empirically `capture_alloc=(0,0)` and `segments=1` confirm no cold-miss valve
  fired. All new shapes covered.
- **Fresh reconfirm @ HEAD (measurement discipline):** re-ran both probes clean — canonical
  segments=1/seam MatMul×1/**B\*=5.09×** (M8=51.9, M1=10.2 ms); split-K segments=1/**B\*=2.71×** (M8=27.5,
  M1=10.2 ms). Reproduces the 5.10/2.69 within noise.

#### Final end-to-end table @ `c842b759`, split-K ON (M-scaling + prefill)

Marginal-compute **TAIL slope** (ms/row, the pure GEMM signal Marlin targets) across the whole arc:

| Model | BEFORE (OFF) | Marlin M>1 (`c692bffe`) | **+ split-K (`c842b759`)** | cliff (split-K) | prefill L=128/512/1024 tok/s (split-K) |
|---|---|---|---|---|---|
| **glm-4-9b** | 2.21 | 0.93 | **0.822** (−63%) | 16.6 ms | 382 / 378 / 419 |
| **qwen2.5-14b** | 3.26 | 1.10 | **1.147** (−65%) | 48.0 ms | 267 / 271 / 277 |

- **Prefill nuance:** for **large-M prefill**, canonical Marlin edges split-K (qwen 291–300 vs split-K
  267–277 tok/s) — split-K's win is concentrated at **small-M** (the M=K verify, where B\* lives).
  **Recommendation: canonical Marlin for prefill; split-K auto-on for the M=K speculative verify.**
- Both vs BEFORE prefill (glm 218/213/217, qwen 147/152/154 tok/s) the arc is **~2×** end-to-end.
- qwen's cliff stays large (48 ms) because Axis-2 is the *eager, capture-breaking* wall (per-M penalty for
  leaving graph replay); the **tail slope** is the Marlin signal and it collapses 3.26→1.15.

#### Reproduce (`c842b759`)

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh && git fetch origin && git merge c842b759
GLM=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
# CANONICAL capture re-probe (split-K off)
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
# EXPLORATORY (adds split-K) — B* 2.69x
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_MARLIN_SPLITK=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
# Byte-identical token parity (add ONNX_GENAI_MARLIN_SPLITK=1 to check split-K too)
CUDA_VISIBLE_DEVICES=7 GLM_4_9B_CUDA_E2E_DIR=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release --test marlin_m_gt_1_e2e \
  marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4 -- --ignored --nocapture
# Final end-to-end M-scaling + prefill @ split-K ON (qwen: swap --model)
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin marlin_bench
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_MARLIN_SPLITK=1 ./target/release/marlin_bench \
  --model $GLM --label glm-4-9b-int4 --m-list 1,2,4,8,16 --past 32 --iters 20 --warmups 3 \
  --prefill-lens 128,512,1024 --prefill-iters 4
```

---

### 2026-08-14 — Marlin int4 GEMM **AFTER** measurements (Lever A landed, #957)

**By:** Sebastian (Perf/CUDA). Measured on the merged snapshot **`c692bffe`** (Deckard's
`squad/marlin-kernel` P1 = full M>1 Marlin coverage, zero tiled fallbacks, byte-identical tokens) merged
into `squad/marlin-bench`. H200 SXM (SM90), `CUDA_VISIBLE_DEVICES=7` (verified idle; Deckard's P2 work on
GPU 0), `nvidia-smi` re-checked each run. Toggle: **`ONNX_GENAI_MARLIN_M_GT_1=1`** (default OFF = portable
tiled fallback = the BEFORE path). Every OFF row below reconfirms the BEFORE baseline on the *same* binary,
so OFF↔ON is a clean A/B on one snapshot. Contention-invariant **min** reported (a co-tenant spike
inflated one qwen L=1024 *median* to 23 611 ms; its min stayed 6 668 ms — exactly why min is the statistic).

#### HEADLINE — Increment-0 capture re-probe (leverb-phase0 Part D INC0, glm-4-9b, GPU 7)

The decisive gate. Part D warms the M=K(=8) shape *before* the captured attempt (FIX 2), so Marlin's
repack cache + scratch pools are populated → no cold-miss safety-valve refusal. Target: `segments`→~1,
captured M=8/M=1 ratio (break-even **B\***) →≤~2 (flips speculative-capture to GO).

| Metric | BEFORE (Marlin OFF) | AFTER (Marlin ON) | Δ | target | 
|---|---|---|---|---|
| INC0 M=8 **captured replay ratio B\*** | **8.76×** (M8 87.3 / M1 10.0 ms) | **4.99×** (M8 50.7 / M1 10.2 ms) | **−43%** | ≤~2 |
| INC0 M=8 **segments** | **41** | **120** (see note) | — | ~1 |
| INC0 M=8 seam nodes | GQA×40, **MatMulNBits×240**, MatMul×1, LN×80 | GQA×40, MatMul×1, LN×80 (**MatMulNBits×240 GONE**) | −240 seams | — |
| eager M=8/M=1 (Part C) | 6.78× | 4.10× | −40% | — |
| eager cliff (M1→M2) | 66.9 ms | 37.5 ms | −44% | — |
| eager **tail slope** (M2..8) | 1.814 ms/row | **0.744 ms/row** | **−59%** | — |
| VERDICT | NO-GO | NO-GO (much closer) | — | GO@B\*≤2 |

- **Marlin did its job decisively:** it made **all 240 MatMulNBits M>1 nodes capture-safe** (they vanish
  from the seam list) and **nearly halved the captured M=8 wall** (87.3→50.7 ms), dropping **B\* 8.76×→4.99×**.
- **Why `segments` rose 41→120 (not a regression — the opposite):** with MatMulNBits now *inside* captured
  regions, the graph's capturable runs are no longer swallowed by all-seam stretches; they surface as
  distinct segments bounded by the **still-uncapturable `GroupQueryAttention`×40 + `SkipSimplifiedLayerNormalization`×80**. Those seams are **not Marlin's scope** — they are the complementary
  **capture-safety lever** (make GQA/LN capture-safe). Marlin removed the GEMM barrier; the residual barrier
  to `segments→1` / `B*≤2` is now cleanly isolated to attention+layernorm capture-safety.
- **Bottom line for the program:** Lever A (Marlin) is a **success at the GEMM level** and moved B\* 43% of
  the way toward the GO line. Full speculative-capture GO (B\*≤2) additionally needs GQA+LN capture-safety;
  that is the next lever, and this probe is already wired to re-measure it (same command).

#### Prefill sweep — Marlin OFF vs ON (native), and the native-vs-ORT gap

`marlin_bench --prefill-lens 128,512,1024` (batched M=len forward; contention-invariant min). ORT-genai
column = the re-exported stock chatglm int4 (unchanged, Marlin is native-only).

| Model / L | native OFF | native ON | Marlin speedup | ORT-genai | gap OFF→ON |
|---|---|---|---|---|---|
| **glm** 128 | 209.8 | **415.4** | **1.98×** | 8 002 | 38× → 19× |
| **glm** 512 | 216.4 | **417.3** | 1.93× | 21 044 | 97× → 50× |
| **glm** 1024 | 218.4 | **425.6** | 1.95× | 26 331 | 121× → **62×** |
| **qwen** 128 | 147.3 | **291.3** | 1.98× | — | — |
| **qwen** 512 | 151.8 | **295.3** | 1.95× | — | — |
| **qwen** 1024 | 153.6 | **300.4** | 1.96× | — | — |

- **Marlin ~doubles native prefill throughput** on both models (tok/s flat vs length still, but the whole
  curve lifts ~2×), **halving the glm-vs-ORT prefill gap** (121×→62× at L=1024). The residual gap is the
  same non-GEMM surface: native prefill is eager (no CUDA graph for M>1) and materializes full-sequence
  logits (D2H), while og-genai runs a CUDA-graph-fused pipeline — i.e. closing it further is again the
  **capture-safety lever**, not more GEMM. Marlin delivered the GEMM half of the prefill win.

#### M-scaling wall — Marlin OFF vs ON (`marlin_bench`, past=32, min ms)

| Model | M=1 | M=2 | M=4 | M=8 | M=16 | cliff (M1→M2) | tail (ms/row) | M16/M1 |
|---|---|---|---|---|---|---|---|---|
| glm **OFF** | 13.27 | 80.30 | 82.92 | 90.59 | 111.34 | 67.3 | 2.214 | 8.37× |
| glm **ON** | 13.40 | 51.02 | 51.79 | 54.90 | 63.90 | **37.7** | **0.925** | **4.77×** |
| qwen **OFF** | 11.16 | 105.09 | 111.25 | 121.70 | 150.75 | 94.2 | 3.258 | 13.45× |
| qwen **ON** | 11.31 | 89.38 | 90.15 | 92.85 | 104.61 | **78.1** | **1.103** | **9.24×** |

- **Confirms Deckard's kernel-level 6.9×→~2× at the forward/op level:** the **tail slope** (marginal
  compute per extra verify row = the pure GEMM signal) collapses **glm 2.21→0.93 (−58%)**, **qwen
  3.26→1.10 (−66%)** — the M>1 int4 GEMM is ~2.4–3× cheaper per row, matching the kernel result once the
  fixed non-GEMM cliff is separated out. The **cliff** (per-op host dispatch + attention/LN, NOT GEMM)
  drops less (glm −44%, qwen −17%) — again pointing at the capture-safety lever for the remainder.

#### Reproduce (AFTER, head `c692bffe` = merge of `squad/marlin-kernel` into `squad/marlin-bench`)

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
git fetch origin && git merge c692bffe        # pin Deckard's P1 snapshot
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin marlin_bench
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # pin an idle HIGH index (7)
GLM=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda

# HEADLINE — capture re-probe, OFF then ON (Part D INC0 = warmed M=K -> no cold-miss)
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_MARLIN_M_GT_1=1 ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_MODEL=$GLM \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0_capture_probe -- --ignored --nocapture

# M-scaling wall + prefill sweep, OFF then ON (drop ONNX_GENAI_MARLIN_M_GT_1 for OFF)
CUDA_VISIBLE_DEVICES=7 [ONNX_GENAI_MARLIN_M_GT_1=1] ./target/release/marlin_bench \
  --model $GLM --label glm-4-9b-int4 --m-list 1,2,4,8,16 --past 32 --iters 20 --warmups 3 \
  --prefill-lens 128,512,1024 --prefill-iters 4
# qwen: --model /home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx
```

#### Portability (Rule 11) — <SM80 does NOT regress

Marlin is SM80+ and **opt-in** (`marlin_m_gt_1_enabled()` defaults OFF). The **Marlin-OFF rows above are
byte-for-byte the portable tiled-fallback path** and reconfirm the BEFORE baseline on the same binary →
a <SM80 device (which can only take the OFF path) is provably unchanged. `marlin_bench` prints device
`compute_cap` so every number is arch-attributable (host = SM90 H200).

---

### 2026-08-14 — Marlin int4 GEMM BEFORE baseline + reproducible PERF/CAPTURE harness (gates #957 Lever A)

**By:** Sebastian (Performance/CUDA & Perf). Measured on H200 SXM (SM90, HBM3e ≈ 4.8 TB/s),
`CUDA_VISIBLE_DEVICES` pinned to a verified-idle high index (GPU 6/7), `nvidia-smi` re-checked before
runs. Branch `squad/marlin-bench`, base **`94c69fdc`** (main). New harness: `marlin_bench`
(`crates/onnx-genai-bench/src/bin/marlin_bench.rs`, `--features bench-native,cuda`) — new file, zero
overlap with Deckard's kernel branch.

**Why:** Justin greenlit the full Marlin int4 GEMM build (Lever A). This is the BEFORE baseline on
today's code so the moment Marlin lands the *same commands* produce a decisive delta, across the four
gate axes in `docs/research/decode-remaining-levers-feasibility.md` §3/§4.

> **SHARED-BOX MEASUREMENT NOTE.** The 8× H200 host is shared with other squad agents; foreign compute
> apps appeared and vanished mid-run (e.g. a 45 GB job on another GPU). Contention only ever *adds* wall
> time, so for the M-scaling wall the **min across iters is the contention-invariant estimator** and is
> what the table reports; the harness also prints median + p10/p90/max so a contaminated run is visible
> (a p90≫min flags it). The e2e tok/s figures below were captured in windows with tight p10/p90 spread.

---

#### Axis 1 — M=1 int4 GEMV achieved DRAM % (the Marlin precondition)

`ncu --graph-profiling node -k regex:matmul_nbits_gemv` on the captured native-CUDA decode
(`profile_native --ep cuda --backend native --steady`), metric
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`. Marlin M=1 pass = ≥~55%; <~40% = M=1 Marlin
NO-GO (thresholds apply to Marlin's *achieved* %, measured AFTER it lands — these are the BEFORE numbers).

| Model | GEMV kernel (shape) | grid | DRAM % | SM % | check (bytes/time) |
|---|---|---|---|---|---|
| **qwen2.5-14b** | `gemv_f16_gate_up_decomposed_swiglu_rmsnorm` (MLP gate_up) | 1728 | **29.4–29.7%** | 49% | 79.7 MB / 57.5 µs = 1386 GB/s = 28.9% ✓ |
| qwen2.5-14b | `gemv_f16_scales_f16_down_c2` (down_proj) | 2560 | 39.7% | 47% | 39.8 MB / 21.6 µs ✓ |
| qwen2.5-14b | `gemv_f16_scales_f16` (qkv/o proj) | 640 | 24.0–24.4% | 41% | 14.8 MB / 12.8 µs ✓ |
| **glm-4-9b** | `gemv_f16_general_bs` (MLP) | 3424 | **17.2%** | 77% | 57.9 MB / 72.7 µs = 797 GB/s = 16.6% ✓ |
| glm-4-9b | `gemv_f16_general_bs` (attn proj) | 512 | 10.8–12.4% | 50–56% | 8.7 MB / 16.7 µs = 519 GB/s ✓ |

- **The qwen MLP gate_up GEMV = ~29.5% DRAM reproduces the doc §3 precondition ("~29% DRAM on an MLP
  shape") to the point.** Its down_proj tops out at ~40%, attn projections ~24%.
- **glm-4-9b runs the *untuned* `gemv_f16_general_bs` path** (no fused-epilogue tuned variant selected for
  its shapes) → only 11–17% DRAM. Both models sit far below the ≥55% Marlin pass line and the MLP shapes
  straddle the 40% NO-GO line — i.e. the M=1 GEMV is **latency/dispatch-bound, not DRAM-bound**, exactly
  the diagnosis Marlin must overturn. Marlin's decisive number is the *delta* it puts on these rows.

#### Axis 2 — M-scaling wall (the cliff Marlin + Lever B must collapse)

`marlin_bench` times a single real `decode_verify` (eager, capture-breaking) over M∈{1,2,4,8,16} on each
model's own MatMulNBits stack; **min ms** (contention-invariant), 30 iters, 5 warmups.

| Model / past_len | M=1 | M=2 | M=4 | M=8 | M=16 | **cliff (M1→M2)** | tail (ms/row) |
|---|---|---|---|---|---|---|---|
| glm-4-9b / past=32 (Increment-0 repro) | 13.4 | 81.1 | 83.9 | **91.0** | 111.7 | **67.7 ms** | ~1.65 |
| glm-4-9b / past=512 (realistic ctx) | 14.0 | 102.7 | 103.3 | 112.3 | 130.7 | 88.7 ms | ~1.6–2.0 |
| qwen2.5-14b / past=512 | 11.9 | 132.9 | 139.2 | 149.5 | 182.1 | 120.9 ms | ~2.8–3.5 |

- **Reproduces Deckard's Increment-0** at matched past (~32): cliff **67.7 ms** (≈ the ~67 ms M=1→M=2
  cliff), M=8 **91 ms** (≈ 87 ms), tail **~1.65 ms/row** (≈ 1.85). The cliff is the fixed penalty for
  leaving the captured M=1 fast path (per-op host dispatch + generic multi-row GEMM + scratch alloc); the
  tail is true marginal compute per verify row.
- Reference captured M=1 decode step (non-verify, graph replay): glm **10.1–10.4 ms**, qwen **7.8 ms** —
  the eager verify M=1 (13–14 ms) is already ~1.3× the captured M=1, before the cliff. **This whole wall
  is what Lever B (capture-stable M=K verify) collapses toward ~1×; Marlin attacks the per-GEMV cost
  underneath it.**

#### Axis 3 — end-to-end tok/s, native CUDA EP vs ORT

`profile_native --ep cuda --steady --warmups 1 --runs 3 --tokens 128 --decode-skip 8` (tight spread).

| Model | native CUDA EP | ORT CUDA EP | native / ORT |
|---|---|---|---|
| **qwen2.5-14b-int4** | **127.8 tok/s** | **89.2 tok/s** (ORT 1.27 EP, same ONNX) | **1.43×** |
| **glm-4-9b-int4** | **93.4–98.6 tok/s** | **233.0 tok/s** (ORT-genai, see note) | **0.40× (native SLOWER)** |

- qwen2.5-14b native **127.8 tok/s** vs ORT-EP **89.2** on the *same* native-authored ONNX → clean
  **1.43× native win**. glm-4-9b native **93–99 tok/s** matches the doc's 96.4 tok/s reference.
- **glm-ORT head-to-head — RESOLVED (was blocked).** No stock ORT can load the native-authored glm ONNX:
  ORT 1.27 EP, pip ORT 1.28, *and* onnxruntime-genai 0.14.1's bundled ORT all reject its
  `GroupQueryAttention` with `Unrecognized attribute: rotary_embedding_dim` (a native-EP-exclusive GQA
  schema). No foundry-local glm package exists. The doc-specified path is therefore a **re-export**: I ran
  the onnxruntime-genai model builder on the local `glm-4-9b-gptq-source` (`ChatGLMModel`, gptq→int4,
  `-p int4 -e cuda`) to produce a clean stock ORT-genai `chatglm` int4 package (`./ort-glm-build`,
  `type=chatglm`, 40L/32H/2KV, matches native dims). It loads in og's ORT and runs.
- **FINDING (truthful negative):** on glm-4-9b int4, stock **onnxruntime-genai BEATS the native CUDA EP**
  — decode **233 vs ~95 tok/s (≈2.5× faster)** and prefill catastrophically (see Axis 3b, ~100–120×). The
  "native beats ORT" claim currently holds for **qwen** (1.43×) but **NOT for glm**. Two contributors:
  (a) native glm runs the *untuned* `gemv_f16_general_bs` GEMV (Axis 1: 17% DRAM vs qwen's tuned 29%);
  (b) native has **no efficient M>1 int4 GEMM**, so its prefill is ~100× slower — *exactly the Marlin
  gate*. og uses CUDA graphs + fused attention + its own int4 GEMM.
- **Caveats (not perfectly apples-to-apples):** the ORT side is builder-*requantized* (gptq→int4 via the
  og builder), not the identical weights/quant as the native file; accuracy parity is **not** verified.
  The methodology also differs from qwen (qwen = same ONNX under ORT-EP; glm = og-genai rebuild, a
  *stronger* stock baseline). The **prefill M>1 gap is architectural** (native lacks the M>1 GEMM) and
  holds regardless of quant. Recommend an og-builder qwen rebuild for a symmetric prefill comparison, and
  an accuracy check on the rebuilt glm before quoting the decode number externally.

#### Axis 3b — long-prompt prefill sweep (prefill IS a Marlin M>1 gate)

`marlin_bench --prefill-lens 128,512,1024` times a single batched `decode(prompt,0)` (M=len forward, the
MatMulNBits M>1 path), KV reset via `rewind(0)` between iters, **min across 4 iters** (+1 warmup),
contention-invariant. Prefill tok/s = len / wall. ORT-genai glm via `bench-scripts/ort_glm_bench.py`.

| Prompt len | glm native | glm ORT-genai | qwen native | native/ORT (glm) |
|---|---|---|---|---|
| **128** | 199.7 tok/s (641 ms) | **8 002 tok/s** (16.0 ms) | 147.1 tok/s (870 ms) | **0.025× (40× slower)** |
| **512** | 214.2 tok/s (2 390 ms) | **21 044 tok/s** (24.3 ms) | 152.2 tok/s (3 363 ms) | **0.010× (98× slower)** |
| **1024** | 217.1 tok/s (4 716 ms) | **26 331 tok/s** (38.9 ms) | 153.7 tok/s (6 664 ms) | **0.008× (121× slower)** |

- **This is the decisive Marlin BEFORE number.** Native prefill throughput is **flat ~150–217 tok/s** and
  scales *linearly with wall* (L=1024 wall ≈ 8× L=128 wall) — i.e. the M=len forward gets **no** batching
  efficiency: each extra prompt token costs ~a full decode step because MatMulNBits has no efficient M>1
  int4 GEMM. ORT-genai's prefill *rises* with length (8k→26k tok/s) as a real GEMM amortizes launch cost.
  The **~100–120× gap** is precisely the wall Marlin (efficient M>1 int4 GEMM) must collapse; after Marlin
  the *same* `--prefill-lens` command must show native prefill rising toward the ORT curve.
- Prefill wall includes full-sequence logits materialization (D2H of all-position logits + full lm_head);
  it is a prefill-to-first-token-ready wall, consistent BEFORE/AFTER since AFTER uses the same path.

#### Axis 4 — capture-safety re-probe (wired to re-run against Marlin)

`leverb_phase0` probe (`onnx-genai-engine`, `#[cfg(all(test, feature="cuda"))]`,
`leverb_{phase0,increment0}_capture_attempt`) on glm-4-9b. Target AFTER Marlin/Lever-B: `segments`→1 (or
near) at M>1 and captured-M=8 ≈ M=1 (break-even B\* ≤ ~2). **Baseline reconfirmed on `94c69fdc`
(GPU 7):**

- **INC0 M=8 capture: `captured=true`, `segments=41`** (matches Deckard's baseline exactly).
- **DECISIVE captured replay wall: M=8 = 87.7 ms, M=1 = 10.0 ms → ratio = `8.77×`** (Deckard prior 8.58×;
  the +0.2× is co-tenant contention on the M=8 wall). VERDICT = **NO-GO** for Lever B today, as expected.
- **Segment root cause (41 seams):** `GroupQueryAttention[KernelCaptureUnsupported]×40`,
  `MatMulNBits×240`, `MatMul×1`, `SkipSimplifiedLayerNormalization×80` — the M=K kernels aren't
  capture-safe yet. This is the surface Lever B must make persistent-binding-clean.
- **Cross-validation:** the probe's Part-C eager sweep independently reproduces `marlin_bench` Axis 2 —
  M=1 14.5 ms, M=8 92.9 ms, **cliff 67.2 ms, tail 1.88 ms/row** — two independent harnesses agree on the
  cliff/tail to within noise, and captured M=1 = 11.1 ms / 90.3 tok/s. The probe is already wired to
  re-run against Marlin (command below).

---

#### Reproduce (exact commands, head `94c69fdc`)

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # verify idle; pin a clean high index

# Build
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin marlin_bench
cargo build --release -p onnx-genai-bench --features bench-native,bench-ort,cuda --bin profile_native

# Axis 2 (M-scaling wall) + Axis 3 (native e2e)
CUDA_VISIBLE_DEVICES=7 ./target/release/marlin_bench \
  --model /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda --label glm-4-9b-int4 \
  --past 512 --iters 30 --warmups 5 --m-list 1,2,4,8,16 --e2e --e2e-tokens 128
CUDA_VISIBLE_DEVICES=7 ./target/release/marlin_bench --model .../glm-4-9b-int4-cuda --past 32 ...  # Increment-0 repro
CUDA_VISIBLE_DEVICES=7 ./target/release/marlin_bench \
  --model /home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx --label qwen2.5-14b-int4 \
  --past 512 --iters 30 --warmups 5 --m-list 1,2,4,8,16 --e2e --e2e-tokens 128

# Axis 1 (M=1 DRAM%) — ncu on the int4 GEMV (needs sudo; RmProfilingAdminOnly=1)
NCU=/usr/local/cuda-13.3/bin/ncu
sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=7 \
  $NCU --graph-profiling node -k regex:matmul_nbits_gemv --launch-skip 400 --launch-count 8 --csv \
  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,gpu__time_duration.sum,launch__grid_size \
  ./target/release/profile_native --model <model> --ep cuda --backend native --steady --warmups 1 --runs 1 --tokens 96

# Axis 3 (ORT head-to-head) — force the CUDA-enabled ORT 1.27
ONNX_GENAI_ORT_LIB_DIR=$ORT_ROOT/lib CUDA_VISIBLE_DEVICES=7 \
  ./target/release/profile_native --model <model> --ep cuda --backend ort --steady --warmups 1 --runs 3 --tokens 128 --decode-skip 8

# Axis 3b (long-prompt prefill sweep) — native, both models
CUDA_VISIBLE_DEVICES=7 ./target/release/marlin_bench \
  --model /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda --label glm-4-9b-int4-native \
  --m-list 1 --past 32 --iters 3 --warmups 1 --e2e --e2e-tokens 96 --prefill-lens 128,512,1024 --prefill-iters 4
CUDA_VISIBLE_DEVICES=7 ./target/release/marlin_bench \
  --model /home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx --label qwen2.5-14b-int4-native \
  --m-list 1 --past 32 --iters 2 --warmups 1 --prefill-lens 128,512,1024 --prefill-iters 4

# glm ORT side: re-export via onnxruntime-genai model builder (native glm ONNX is rejected by all stock ORT),
# then run the og decode+prefill bench. CUDA-12 runtime libs needed for og's bundled ORT.
CU12=$(ls -d /usr/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH="$CU12$LD_LIBRARY_PATH" \
  /home/justinchu/.conda/envs/onnx/bin/python -m onnxruntime_genai.models.builder \
  -i /home/justinchu/glm-e2e-artifacts/glm-4-9b-gptq-source -o ./ort-glm-build -p int4 -e cuda -c ./ort-glm-cache
cp /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda/tokenizer.json ./ort-glm-build/tokenizer.json
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH="$CU12$LD_LIBRARY_PATH" \
  /home/justinchu/.conda/envs/onnx/bin/python bench-scripts/ort_glm_bench.py ./ort-glm-build 128,512,1024 128 3

# Axis 4 (capture re-probe)
CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_RUN_CUDA_SMOKE=1 \
  ONNX_GENAI_LEVERB_MODEL=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda \
  cargo test -p onnx-genai-engine --features cuda,native-backend --release leverb_phase0 -- --ignored --nocapture
```

#### Portability (Rule 11)

Every number above is the **current split-K / `gemv_f16_general_bs` fallback layout** (pre-Marlin).
Marlin is SM80+; because these are the fallback-path numbers, re-running this *identical* harness after
Marlin lands — on a <SM80 device, or with the fallback deliberately selected — proves the fallback did
not regress. Host here is SM90 (H200). `marlin_bench` prints device `compute_cap` so every number is
attributable to the arch that produced it.
