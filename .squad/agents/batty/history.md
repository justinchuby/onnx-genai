# Batty — History (compacted 2026-08-27T17:00:00Z)

**Role:** Engine/EP implementer for the Rust ONNX runtime. Owns generation policy, logical KV, scheduler/default semantics, CLI maintainer harness wiring, and CPU/native EP correctness while preserving ORT ownership of physical forward execution/KV.

## Durable lessons
- Canonical ownership: ORT owns forward execution and physical KV; engine owns generation policy and logical KV.
- CPU kernels rely on session-side `strided::view_in_bounds` before dispatch.
- Optimizer fusions live under `com.microsoft` and must fail closed with strict decline-to-fuse guards.
- Batty remains locked out of H-D1 storage sizing, earlier fusion follow-ups, EPContext writer, `test/tiny-reasoning-fixture`, and any artifact explicitly reassigned by reviewers.
- `validate_model()` is the shared load-time validation path; empty graphs remain valid.
- CUDA EP work must remain capture-safe and correct across supported SM architectures.
- Sampling flags disable greedy when temperature/top-p/top-k imply stochastic decoding unless `--temperature 0` or explicit `--greedy`.
- Tiny reasoning fixture trap: statistical token-stream replacement was rejected (15/15 failures). Batty locked out.
- Empty assistant turns poison context; closed paths must drop whitespace-only answers.
- Never infer output dtypes from inputs; read graph-declared value info.
- Multi-output ops must not assume input[0]'s shape; reduction outputs follow keepdims semantics.
- A mutation test for one CUDA wait operation must not change registry semantics or another causal edge.

## Historical context

Engine/KV, ORT2 EP/C API, optimizer, EPContext, load validation, upstream CI, CUDA capture, and the 2026-08-11–13 decode-fusion chronicle are archived. Full older history is in `history-archive.md`; the exact hot file before this compaction is in `history-archive-2026-08-27T17-00-00Z.md`.

## 2026-08-20T05:50:19+00:00 — Phase-4 q38 int4 GEMV/argmax wins merged

Scribe recorded Batty's Phase-4 contributions after merge to `origin/main`: #1557 added bf16 device-argmax and dtype-aware greedy routing, moving q38 **52.6→54.6 tok/s** (~+3.8%) while proving device token-loop/host-argmax was not the dominant remaining lever. #1561 added the asymmetric int4 block-32 split-K occupancy gate, removing the large-N zero-point split-K mis-route and lifting q38 to ~**59.5 tok/s** standalone; integration later measured q38 **61.32 tok/s** with #1562 stacked. Standing lesson: for Qwen3.8-27B, keep chasing int4 M=1 GEMV occupancy/arithmetic intensity; split-K GEMV nondeterminism still blocks a stable q38 golden oracle.

## 2026-08-20T13:46Z — GEMV latency-hiding floor; projection/GLU fusion ranked #2

Scribe recorded Holden's survey and Batty's GEMV floor result. The current block-32 asymmetric int4 M=1 GEMV has shipped PF=2 at the optimum; deeper prefetch regresses and wide 128-bit loads are not applicable to q38 block-32. External-engine survey ranks **fuse adjacent projections + inline SwiGLU** as lever #2 after the GDN recurrence megakernel, because launch-bound M=1 decode wins by reducing kernel count rather than tuning each kernel.

## 2026-08-26 — #1896 causal-gate revision rejected

Batty's causal-gate revision was rejected because the mutation also changed event-registry behavior. Durable lesson: a mutation test for one CUDA wait operation must not alter registry lookup/removal semantics or other causal edges.
