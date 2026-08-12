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
