# Decision — Dense-decode megakernel P2-prototype (multi-CTA cooperative, MEASURED)

**Author:** Sebastian (Performance/Systems) · **Date:** 2026-08-13 · **Branch:** `squad/dense-megakernel-multicta`

## Context
Final bounded micro-bench before committing agents to the multi-week P2 whole-step
megakernel. P1.5 pinned the architecture (persistent multi-CTA cooperative, grid.sync
capturable, single-CTA ruled out). This step **built** that kernel for the MLP
triple-GEMV block and measured per-layer time vs the production-style per-op baseline.
Throwaway `#[ignore]` GPU probe `megakernel_multicta_mlp_probe` added to
`crates/onnx-runtime-ep-cuda/tests/megakernel_headroom_gpu.rs` (not pipeline-wired).

## Measurement (H200, median 200 iters, 3 repeats, `CUDA_VISIBLE_DEVICES=0`)
- Per-op baseline MLP (gate/up/SiLU-mul/down, 4 launches, grid=N): **0.656 ms/layer-MLP**.
- **Multi-CTA cooperative megakernel** (1 cooperative launch, grid=1056 CTAs = 8/SM ×
  132 SMs, grid.sync seams, activations in L2-resident global scratch): **0.676–0.680
  ms/layer-MLP → recovered fraction −3.2% (reproducible −2.9%…−3.5%). Megakernel is
  ~3% SLOWER.**
- grid.sync barrier cost = **2.23 µs/barrier** (full 1056-CTA grid); a full layer would
  pay ~6–8 → ~0.7–0.9 ms/token of pure barrier tax across 52 layers.
- Numerics **byte-exact (0 ulp)** — identical dequant + block_sum order, no reorder.
- Cooperative launch supported + occupancy 8 blocks/SM (co-residency headroom fine).

## Why (mechanism)
1. **CUDA-graph replay already removes the per-launch overhead** the megakernel exists
   to recover (Phase A: eager 27.6 ms → captured 21.4 ms, ~6.1 ms already banked).
2. **The multi-CTA design must pay a grid.sync tax** (2.23 µs × seams) the per-op path
   never pays; it roughly cancels / slightly exceeds the launch+round-trip savings.
3. The GEMVs are genuine **full-device weight-read work** (per-op already fans across all
   132 SMs — that's why P1.5 single-CTA was 926× worse). A megakernel does the *same*
   reads and cannot accelerate them; removed activation round-trips are already
   L2-resident (~80 KB), saving ~nothing.

## Decision / Recommendation
**NO-GO on the whole-layer GEMV megakernel (P2).** Architecture is sound and capturable,
but the measured per-layer payoff is **negative** on the GEMV-dominated path. Do NOT
fund the multi-week cooperative-megakernel integration + capture-safety/numerics gating.

**Redirect the lever to graph-side glue node-collapse (Batty, `optimizer.rs`):** the only
decode component with recoverable overhead is the elementwise/norm glue (Phase B: 85.6%
of *glue* GPU time fusible). Collapsing glue nodes in the graph shrinks the captured
graph's replay overhead with **no cooperative kernel, no grid.sync tax, no numerics
reorder**. Sebastian's kernel-side role stays limited to the already-landed fused
epilogues (#867 SwiGLU-mul, #854 skip-RMSNorm) that let Batty delete standalone nodes.

**Projected tok/s from a GEMV megakernel: ~0% (decode stays ~47 tok/s).** The earlier
§4 "~62 → ~100+ tok/s" projection assumed ~85% chain recovery; §7 shows that recovery
does not apply to the GEMVs. Realistic upside now lives in glue node-count reduction
(bounded, graph-side) + GQA-decode kernel tuning.

## Caveats (honest)
- Representative f32 int4 GEMV, not the production f16 dp4a split-K kernel → the *ratio*
  transfers, absolute ms does not. Mechanism (§7.2) is kernel-speed-independent.
- Eager-timed baseline; under graph replay the megakernel's edge only worsens.
- MLP subset; attention adds more GEMVs + more barriers → megakernel falls further behind.

## The one un-excluded future path
A software-pipelined design that **overlaps next-layer int4 weight prefetch with current
compute** (Hazy-style) attacks the *GEMV time itself* (the real floor), not launch
overhead. Harder than node-collapse; scope only if graph-side glue collapse + GQA tuning
are exhausted and still short of target.

## Ownership
- Graph-side glue node-collapse (the live lever): **Batty** (`optimizer.rs`).
- Fused kernel epilogues that enable node deletion: **Sebastian** (already landed).
- Numerics gate (only if any future fused reduction reorders): **Chew**.

Docs: `docs/research/dense-decode-megakernel-feasibility.md` §7 (+ §5/§6 marked superseded).
