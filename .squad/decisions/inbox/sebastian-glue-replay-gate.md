# Decision: glue node-collapse is a GO — validated UNDER graph replay (not just eager)

**Author:** Sebastian (Performance/Systems) · **Date:** 2026-08-13 · **Branch:** `squad/glue-collapse-replay-gate`

## Context / the gate

#898 killed the GEMV megakernel (NO-GO, −3.2%) and §7.4 redirected the decode lever to
**graph-side glue node-collapse** (Batty, `optimizer.rs`). But #898's own mechanism was
"CUDA-graph replay already removes per-launch overhead" — and node-collapse targets that
*same* overhead. The Phase B 85.6% glue recovery was measured **eager**. So before staffing
a multi-week `optimizer.rs` pass we measured whether glue collapse recovers anything **under
graph replay** (the real production decode path), not eager.

## Measurement (H200, `glue_collapse_replay_gate_probe`, median ≥200 iters, 4 repeats)

- Per-op glue chain (22 nodes) vs fused (1 node), **each captured into a CUDA graph, timed under replay**:
  - Eager (reference): 84–85% recovered.
  - **Under replay: per-op 0.0280 ms → fused 0.0069–0.0073 ms = 74.0–75.5% recovered.**
- **~0.90 µs/node residual dispatch cost SURVIVES replay** (22 trivial graph nodes vs 1).
- **Byte-exact, 0 ulp.**
- Whole-model projection (52 layers, 21.4 ms/token baseline): saves ~1.08 ms/token →
  **46.7 → ~49.2 tok/s (+5.3%)** *(projection — ceiling, glue nodes interleave with GEMVs)*.

## Decision

**GO on graph-side glue node-collapse.** The #898 concern is empirically disproven for glue:
graph replay amortizes per-node dispatch by only ~2.5× (eager ~2.3 µs/op → ~0.9 µs/node),
**not to zero**, and collapsing ~22 nodes/layer recovers the residual. This is the *opposite*
of the megakernel because glue ops are tiny L2-resident dispatch-bound work (not irreducible
GEMV work) and collapse uses an ordinary fused launch — **no grid.sync tax, no reduction
reorder, no Chew gate**.

## Ownership / next steps

- **Batty (`optimizer.rs`):** collapse the elementwise+norm chain in the captured graph
  (target ~49 nodes/layer → a handful), reusing the landed fused epilogues (#867 SwiGLU-mul,
  #854 skip-RMSNorm) so standalone nodes can be deleted.
- **Sebastian:** kernel-side fused epilogues already landed; available to add more if Batty
  needs a node deleted.
- **Chew:** not gated — collapse is byte-exact (no fused reduction).
- **Bounded upside ~+5% decode** (46.7 → ~49 tok/s). Low risk, small surface.
- **Validation gate for Batty's first candidate:** measure node-collapse on the **real**
  captured decode graph to convert the +5.3% ceiling into a realized number.

Docs: `docs/research/dense-decode-megakernel-feasibility.md` §8.
