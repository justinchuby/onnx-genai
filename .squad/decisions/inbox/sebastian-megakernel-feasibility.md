### 2026-08-13: Dense-decode megakernel — Phase A gate PASSED, Phase B measured, P2 is a GO (staged)

**By:** Sebastian (Performance/Systems). Brief: `docs/research/dense-decode-megakernel-feasibility.md`. Branch `squad/dense-decode-megakernel`. Prototype harness: `crates/onnx-runtime-ep-cuda/tests/megakernel_headroom_gpu.rs` (`#[ignore]`, throwaway, not wired into any pipeline).

**Context:** the bandwidth probe (#885) confirmed native Muse-Glimmer-30B decode is **latency-bound on the ~2568-node serial launch chain** (21.4 ms/token, 46.7 tok/s H200), NOT bandwidth-bound. A whole-step/persistent megakernel is the reopened lever. The only prior megakernel experiment (#769 P0) was a per-op QMoE kernel on the 35B-A3B **MoE** path (~13% Amdahl cap, NO-SHIP) — it does **not** bind the dense path (different op chain). This ran the untested dense P1.

**Phase A — headroom gate (measured, no kernel writing): PASS.**
- Captured 21.4 ms/token; eager 27.6 ms (capture already recovered ~6.1 ms CPU launch).
- Op mix (eager): MatMulNBits 28%, GQA 26%, Mul 22%, Add 9%, Norm 5%, Sigmoid 5%, Reshape 3%. The elementwise/norm **"glue" = 70% of node count, ~44% of eager time** — tiny launches round-tripping the 6656-bf16 hidden vector through global memory.
- Essential floor a megakernel must keep = weight+KV DRAM = 15.325 GB / 4.8 TB/s = **3.2 ms/token (~313 tok/s ceiling, 6.7×)**. Byte-fold showed only ~0.7 ms of that is currently *exposed* (weights are latency-hidden, not throughput-bound).
- **Recoverable overhead ≈ 21.4 − 3.2 ≈ 18.2 ms ≈ 85% of the token** (launch + round-trip latency + fill/drain) — far above the 25% gate.

**Phase B — fusion-mechanism micro-bench (CUDA events, H200, median 200 iters): STRONG.**
- Per-launch GPU floor (trivial kernel × 2568 sequential on one stream): **~1.5–2.0 µs/launch** → 2568 launches = ~4.4 ms = ~20% of the token in pure launch alone.
- Realistic glue op (H=6656 bf16 round-trip): ~2.1 µs — i.e. glue ops are **launch/latency-bound, not compute-bound**.
- **22 glue ops fused into 1 register-resident launch recovers 85.6% of the chain's GPU time** (0.045 → 0.007 ms; 2 runs 85.5/85.7). Numerics: fused fp32 chain vs per-op bf16 = 0 ulp on this input (but the real megakernel's RMSNorm/softmax reductions reorder fp32 → **Chew gate mandatory**, not cleared here).

**Projection (measured-anchored):** conservative glue-only fusion → **~63 tok/s (+35%)**; full one-layer persistent megakernel (pipelined int4 weight loads, activations resident) → **~95–140 tok/s (2–3×)**; ceiling ~313 tok/s. All > 47.25 — the lever is real.

**VERDICT: GO to prototype P2, STAGED.**
1. **P1.5 (do this before funding full P2):** build the *real* one-layer dense megakernel with actual int4 weights (RMSNorm→QKV→RoPE→GQA→O→residual→RMSNorm→gate/up→SiLU·Mul→down→residual), intermediates in registers/shared, chunked at the GQA-softmax dependency, behind an env flag, measured vs the current per-op layer. Phase B measured the **glue** recovery + launch floor directly but did NOT yet build the fused **int4 GEMV** path — that per-layer number gates full P2.
2. **P2 (whole-step integration):** large, multi-week. Risks to budget: capture-safety (no internal alloc/free/sync — pre-stage all scratch per #854/#867); numerics (fused reductions → **Chew** + f64 oracle); decode-loop/graph-structure changes to emit one fused op/layer instead of ~49 nodes → **coordinate with Batty** (optimizer/graph side). Keep byte-exact greedy parity (first-16 ref ids `[24,372,1045,10016,328,2885,262,5091,8811,511,917,4921,768,328,2885,262]`).

**File boundary:** kernel/harness in `kernels/`+`tests/` are mine; the graph-pass side that collapses ~49 nodes/layer to one fused op is Batty's (`optimizer.rs`). Two coordinated PRs when P2 starts.
