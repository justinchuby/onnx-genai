# Batty — History (compacted 2026-08-12T06:00:00Z)

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
- Never infer output dtypes from inputs — always read from graph's declared value info.
- Multi-output ops must not assume all outputs share input[0]'s shape; reduction outputs (Mean, InvStdDev) follow keepdims semantics.
- The upstream CUDA EP is mature and actively staffed; competitive advantages are runtime-level and not portable upstream.

## Recent work (current wave, ~2026-08-11/12)

### 2026-08-11 — B2 Fix: Device-ID comparison for D2D copies (PR #762, commit fb9d757b3)
`is_same_device()` via `MemoryDevice_GetDeviceId` (verified at `bindings.rs:6309`). Fast path: pointer equality. Null guard: fail-closed. 6 unit tests; 161+9 pass.

### 2026-08-12 — PR #31988 Build Fix (sm_count parameter mismatch)
`TryMatMulNBits` gained `sm_count` but `fpA_intB_gemm_kernel_test.cc` not updated. Fixed by passing `device_prop_.multiProcessorCount`. Commit `55e438ca6f`.

### 2026-08-12 — #762 Opus memory-safety wave corrective completion (commit b906ab2bb)
(1) EP assignment assertion added — `add_skip_layer_norm_mul_routed` proves Add/SkipLayerNormalization/Mul assigned to `cpu_ep`. (2) `end_version: since` → `i32::MAX`. (3) `struct_size` loader validation. (4) `NXRT_REQUIRE_ORT_TESTS=1` gate. (5) `matmul_initializer_weights` fixture. (6) 5 `.gitignore` negations. 278 passed / 0 failed.

Full pre-compaction history in `history-archive.md`.

## 2026-08-12 — PR #32003 narrative fix + PTX evidence (confirmed ready for review)

- Corrected three overstated claims in PR body (unused-param not reproduced,
  aliasing scope narrowed, template-instantiations wording removed).
- Produced PTX codegen-neutral evidence: 12/12 pairs byte-identical across
  sm_53/70/75/80/86/90 × {-O0, -O3}. No normalisation required.
- Updated PR body via GitHub REST API.
- clang-format passes; leak check clean.
- Coordinator independently reproduced PTX result: byte-identical confirms
  codegen-neutral status. PR marked ready for review.

## 2026-08-12 — PR #31973 N1: HasCenteredTwoPassKernel()

Closed the final blocker on #31973. Added `HasCenteredTwoPassKernel()` predicate
(x86-64 compile-time guard) so the six precision suites skip on RISC-V/ARM where
the centered two-pass algorithm isn't used. Fixed `mlas.h` wording (AMD64/IX86 →
x86-64). Fresh build: 41 passed / 2 disabled, 43/43 with disabled. Produced
benchmark numbers on AMD EPYC 9V74 (AVX2/FMA): LayerNorm 6.8–11.9× vs scalar
at N≥128, RMSNorm 2.3–3.5× at same sizes, 1000 iters, p50 median.

## 2026-08-12 — PR #31973 N1 wording fix (lockout, Deckard)

Deckard corrected "x86-64" → "x86 (32-bit and 64-bit)" in `mlas.h`, `layernorm_kernel_avx2.cpp`,
and six `GTEST_SKIP` messages after Challenger's delta review flagged the inaccuracy. Batty's
implementation was correct; only the comment wording was stale. Head `4a16925a88`. PR #31973
marked ready for review.

## 2026-08-12 — Assigned Blocker 1 (LOAD) of the CUDA-capture escalation
Branch `squad/native-pipeline-embedding`. Fix the native pipeline embedding / load
path so Muse-Glimmer loads resident on the engine native decode path — precondition
for CUDA-graph capture engaging. Part of Sebastian's 3-blocker escalation (LOAD +
CLASSIFY [Deckard] + CAPTURE [Sebastian]). Shared team goal: **beat ORT 40 tok/s via
CUDA-graph capture**. In progress.

## 2026-08-12 — CUDA capture arc COMPLETE (shared: 11.4 → 23.13 tok/s)
Blocker 1 (LOAD) landed as **#850** (`29bd8a35`): `PipelineEngine` now runs
Muse-Glimmer's embedding component on the native CUDA EP (embeds-producer promoted
to every_step, ORT-skips on native backend, bf16 gates relaxed, empty image-features
seed, KV context ceiling threaded). End-to-end load + byte-exact parity. Prerequisite
#2 of the 5-blocker chain (#848 → #850 → #852 → #855 → #854). Team result: native
decode **11.4 → 23.13 tok/s**, capture fully engaged (1 segment / 0 seams). Next lever
= Cast round-trip elimination (overlaps my decode-graph domain).

## 2026-08-13 — Fusion arc: 47.25 tok/s is the CEILING (#872 doc-only, #873 opt-in)
**#872** (doc-only): `CudaFoldConstantAdd` removed 208 cheap constant `Add` nodes/token but
**REGRESSED −2.8%** (47.17→45.85) — not shipped. **#873** (MERGED): `CudaQkvProjectionFusion`
(`optimizer.rs`) fused 3 per-layer q/k/v int4 projections into 1 wider `MatMulNBits` + `Split`,
removing **104 EXPENSIVE GEMV launches/token** (417→313); byte-exact; tok/s **FLAT** (47.33→47.26).
No new kernel. **Retained DISABLED-BY-DEFAULT** behind `ONNX_GENAI_CUDA_ENABLE_QKV_FUSION=1`;
preserved for future dispatch-bound architectures. **CONCLUSION (with #870 GQA + #871/#872):
native int4 decode of Muse-Glimmer-30B is weight-bandwidth/compute-floor bound at ~47.25 tok/s
(H200), NOT dispatch-bound — the 3 projections read disjoint int4 weights so fusing cannot cut
bytes. 47.25 is the architectural ceiling. A fused-launch QKV epilogue kernel is NOT worth
building (still bandwidth-bound).**

## 2026-08-13 — Graph-side glue node-collapse (`optimizer.rs`) is now the PRIMARY decode-overhead lever
The persistent multi-CTA cooperative GEMV megakernel was built and measured a 🟥 NO-GO (~3% slower,
#898) — so the kernel-family alternative to node fusion is off the table on H200. Combined with the
#885 finding (decode is LATENCY-bound on the ~2568-node serial chain, ~8.2 µs/node), **graph-side
glue node-collapse in `optimizer.rs` is now the primary named recoverable-overhead lever for native
decode.** Target: the elementwise/norm "glue" (Phase B measured ~85.6% of *glue* GPU time fusible) —
collapse glue nodes in the graph to shrink the captured graph's replay overhead, with no cooperative
kernel, no grid.sync tax, no numerics reorder. Sebastian's landed fused epilogues (#867 SwiGLU-mul,
#854 skip-RMSNorm) are the kernel-side enablers that let standalone nodes be deleted. NOTE: my #872
`CudaFoldConstantAdd` (208 cheap Adds) REGRESSED and #873 QKV fusion was FLAT — so target the glue
round-trips, not marginal disjoint-weight GEMV fusion.
