# Sebastian — History (compacted 2026-08-12T06:00:00Z)

**Role:** Owns DESIGN §26 batched serving, runtime/server performance, and cross-runtime benchmark analysis for `onnx-genai`. Preserve `submit`/`step`/`poll` batching semantics, force single-thread ORT for exact-equality real-model tests, and use canonical benchmark/observability harnesses for runtime comparisons.

## Durable lessons
- §26 Stage A/B: `Engine::generate_batched_static` and `ContinuousBatchManager`; byte-denominated VRAM/RAM limits and transactional lowering.
- CPU decode profiling showed ORT `session.run` dominates (~98.9%); fp32 `lm_head` quantization and op fusion are major levers.
- `filter_map` is wrong wherever position or rank is load-bearing; use `map → Vec<Option<usize>>`.
- A reviewer's "SAFE" is not proof; verify the load-bearing claim independently.
- `cargo test --workspace` silently truncates on failure — always use `--no-fail-fast`.
- Never commit `.squad/` files to external repos.

## Historical context (2026-08-12→13 decode arc — detail in `history-archive.md`)

Owned the native-CUDA capture + kernel work that took Muse-Glimmer-30B int4 decode **11.4 → 47.25 tok/s (now beats ORT ~40)**: capture arc to 23.13 (#854/#855), norm-cast-fold + parallel tree reduction to 40.21 (#860), cached Float16-staged int4 scales to 47.25 (#867), bf16 SwiGLU kernels (#871). Then FOUR independent confirmations that batch-1 decode is at its latency/dispatch floor, NOT bandwidth-bound: lower-bit-quant NO-GO (#885), multi-CTA megakernel NO-GO (#898), skip-RMSNorm standalone fold −1.5% NO-SHIP (kernel byte-exact SHIP, #903), norm→GEMV-prologue fusion NO-GO (#916). Standing lesson: at M=1, folding parallel work into a single-CTA reduction serializes what per-op spreads across 132 SMs; CUDA-graph replay already banks per-launch overhead, so node/launch fusion is not a decode lever. Detailed dated entries archived `.squad/agents/sebastian/history-archive.md`.

## 2026-08-14 — bf16 norm-into-GEMV-prologue fusion NO-GO (#916, MERGED)
- **2026-08-14 (#916, MERGED):** bf16 norm-into-GEMV-prologue fusion measured NO-GO — −4.6% regression AND numeric divergence (≈token 38) under CUDA-graph replay; fp16 prologue reduction is single-warp-serial on the critical GEMV path. Finding-only (docs §8.7), nothing landed. **Fourth** independent confirmation of the batch-1 decode latency floor; norm→GEMV-prologue kill-gate CLOSED.

## 2026-08-14 — int4 decode GEMV bandwidth pass (#928, MERGED, main 8fe56961)
`ncu` on the dominant Muse-Glimmer-30B int4 GEMV: sustains only **~29% peak DRAM** — kernel-efficiency
floor, not hardware floor. Three phased levers, H200 GPU 7: **split-K (2→4→8) NO-GO** (K4 +1% noise,
DRAM fell, K8 regressed; occupancy already ~91%); **cp.async double-buffered loads NO-GO −13%** (4 B/lane
too small — needs 16 B `.cg` over a Marlin tiled relayout); **fold per-block scale into LOP3 dequant
(`fma(code,scale,-zp·scale)`) = the only win, +2.7%** (~47.6 → 48.9–49.0 tok/s). Kernel −4.6% over ~61%
GEMV fraction predicts +2.9%, measured +2.7% — **fully Amdahl-explained, no hidden serial-dispatch floor**.
GEMV is **co-bound** (40.7% Long-Scoreboard + 64.8% dequant-ALU); ~39% non-GEMV Amdahl-caps end-to-end.
Reframes the "launch-amortized floor" as HBM-bandwidth/dispatch-co-bound. Numerics: greedy 128-token stream
byte-identical but not byte-exact per element (fails synthetic asymmetric-zp guard, near-zero cols only).
**Ship opt-in default OFF** (`ONNX_GENAI_GEMV_FOLDSCALE=1`). Bigger single-GPU wins need a from-scratch
Marlin int4 kernel (multi-week).

## 2026-08-14 — Marlin int4 GEMM perf/capture harness + BEFORE/AFTER B* arc (#962 MERGED df6d3afb)
Built `marlin_bench` (`crates/onnx-genai-bench/src/bin/marlin_bench.rs`, `--features bench-native,cuda`): times a real `decode_verify` over M∈{1,2,4,8,16} + prefill L∈{128,512,1024}, prints median/p10/p90/max + device `compute_cap` so every number is arch-attributable. Ran all BEFORE→AFTER capture re-probes for Deckard's Marlin (Lever A): the glm-4-9b M=8 speculative-verify graph went **41 fragmented segments / capture B\*=8.76× → SINGLE whole-graph capture, ZERO unsupported nodes, B\*=2.16×** (arc 8.76→4.99→2.71→2.63→2.16×), prefill ~2× (glm 218→426 tok/s @L=1024, halving the vs-ORT gap 121×→62×), byte-identical greedy tokens throughout. Per-model honesty: glm (block-128) clean GO at 2.16×; qwen (block-32) capture fully solved but B\*≈4.7× — a denominator effect (fast tuned block-32 M=1), not a kernel bug. Also caught that the qwen exact-token parity test is **flaky ~25%** (asserts equality vs a nondeterministic tiled reference at a near-tie tok19); Marlin's fixed split-K reduction is the DETERMINISTIC path — harden the assertion, not a regression. Gates the #957 spec-capture CONDITIONAL-GO. Reviews: Chew 🟡 / Gaff 🟢.
