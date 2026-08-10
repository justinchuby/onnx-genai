# Roy — History (compacted 2026-07-29)

**Role:** Architecture/planning and implementation reviewer spanning engine phases, ORT2 shape/optimizer work, EPContext, packaging, router design, CLI contracts, and stress-test design. Honor reviewer lockouts, keep documented contracts aligned with executable behavior, and preserve model/vendor/EP-agnostic interfaces.

## Durable lessons
- Engine work should keep moving away from monoliths toward explicit backend/sampler/proposer seams; runtime owns KV and public contracts must match executable behavior.
- Router work established affinity/load policies, persisted session mapping, SSE/polling semantics, metrics, drain/rebalance, response caps, overload guards, and 73-test coverage as the baseline.
- Original ORT2 shape-inference artifact ignored transpose/overflow traps; Roy is locked out and Deckard's fixes are canonical.
- Session optimization remains opt-in/default-off; byte invariance, strict fusion decline guards, and separate fused-vs-unfused drift tolerances are invariants.
- EPContext generic encoding must stay model-agnostic and byte-exact; Roy's EP-literal encoder v1 was rejected and Deckard v2 is canonical.
- Crate-reservation publication cycles are forbidden; Leon's path-only dev-dependency fix is canonical after Deckard's rejected runbook.
- CUDA M2 packed-GQA host-seeded cache/PTX artifact was rejected; Wallace's repair is canonical. CPU/CUDA token-10 mismatch was classified as pre-existing M2 numerical drift.
- Native CUDA serving must fail closed when real CUDA-only models cannot run; Roy's rejected safety-gate artifact is locked out and Deckard's `fa30410` is canonical.
- QMoE supports 1/2/4/8-bit expert weights; 3-bit remains rejected because packed values cross byte boundaries, and sparse mixer/IQ1-IQ2 work stayed follow-up.
- CSA ratio-4 and ratio-128 paths must be separately keyed; five-output ratio-4 misrouting was the B5 trap, and graph capture is valid only for the explicitly supported ratio-4 fp8 6-output configuration.
- Performance claims must be profile-backed across supported architectures, not just sm_90; stale CUDA tests must assert the current failure path, not old phase wording.
- `onnx-genai` CLI is a maintainer/debug harness, not a consumer local-inference product; `docs/research/cli/00-backlog.md` is the source of truth before CLI backlog work.
- Project rule 2 must explicitly forbid hardcoded architecture/vendor/EP assumptions; condensation must not weaken review-blocking identity rules.
- Integration stress tests should assert invariants: termination, non-empty committed turns, reasoning progress, repetition bounds, token/history/KV consistency, scheduler liveness, sampling observability, feature-state coherence, and reproducible failure packets.

## Recent work

### 2026-08-10T21:15:32+0000 — EP provider readiness verification and doc consolidation

**Mission:** Verify provider inventory, prove CUDA blocker, define ORT compatibility boundary, consolidate EP_PLUGIN_EXPORT docs, roadmap to full compatibility.

**Findings (all evidence-backed):**

- **Inventory re-verified:** 2 production EPs: `CpuExecutionProvider` (`provider.rs:118`, NEAR) and `CudaExecutionProvider` (`provider.rs:513`, BLOCKED). 2 inbound adapters (not candidates). 7 test/mock impls (excluded). Complete and correct.

- **`../onnxruntime-mlx` contradiction resolved:** `ls /workspace/dev/` → only `onnx-genai` present. The MLX sibling repo does not exist in this workspace. `.squad/team.md` references an external repo not checked out here. No Metal EP in scope.

- **CUDA blocker — dual:**
  1. *Adapter compile error* (hardware-independent): `onnx-runtime-ep-plugin/src/ep.rs:34` initializes `OrtEp` struct missing 9 optional fields added in ORT 1.23–1.27. Affects both CPU and CUDA plugin. Mechanical fix for Nabil.
  2. *Runtime hardware requirement* (CUDA-only): `nvidia-smi` absent, `nvcc` absent, `/usr/local/cuda*` absent, `/dev/nvidia*` absent. CUDA EP uses `cudarc` with `dynamic-loading`; builds cleanly (`cargo check -p onnx-runtime-ep-cuda` → `Finished in 10.02s`) but fails at runtime without `libcuda.so`. Additional design work needed: CUDA context/stream sharing, allocator callbacks, data transfer.

- **CPU EP plugin compile error confirmed:** `cargo check -p onnx-runtime-ep-cpu-plugin` → `error[E0063]: missing fields CreateProfiler, GetAvailableResource, GetDefaultMemoryDevice and 8 other fields in initializer of OrtEp`. NOT a clean compile, contrary to the status claim in `EP_PLUGIN_EXPORT.md`.

- **ORT compatibility boundary:** ORT 1.27.0 (`ORT_API_VERSION = 27`). `OrtEp` has 24 fields; `OrtEpFactory` has 19 fields. Required: `CreateEpFactories` + `ReleaseEpFactory`. Fail-closed version check required in `CreateEpFactories`. `ValidateCompiledModelCompatibilityInfo` is on `OrtEpFactory`; `GetCompiledModelCompatibilityInfo` is on `OrtEp` — bindings confirmed.

**Docs changed:**
- `docs/EP_PLUGIN_EXPORT.md`: Corrected stale "v1 implemented" status; replaced §6 with true compile state; replaced dependency order with accurate roadmap; added §8 roadmap.
- `docs/EP_PLUGIN_EXPORT_ABI_TRUTH.md`: Added §6 with accurate `OrtEp` (24 fields) and `OrtEpFactory` (19 fields) field inventories from `bindings.rs`; identified 9 missing fields by name; added fix guidance.
- `docs/EP_PLUGIN_EXPORT_INVENTORY.md`: Updated summary table with correct readiness; added §6 verification note.
- `docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`: Added remediation status note (do not mark findings resolved; Holden re-audits at merge).

**Decision record:** `.squad/decisions/inbox/roy-ep-provider-readiness.md`

### 2026-08-10T22:42:00Z — EP plugin export: final docs + PR

Verified end-to-end state of `squad/ep-plugin-export` branch. Both adapter crates
compile cleanly. `cargo test -p onnx-runtime-ep-plugin --lib`: 82 passed / 0 failed.
10 ORT integration tests individually confirmed passing (ORT 1.27.0 loads, registers,
and runs the Rust CPU EP; numerically correct outputs). `conformance_two_sessions`
carries `#[ignore]` — known OrtEpDevice corruption bug (Nabil, factory.rs).

Updated `docs/EP_PLUGIN_EXPORT.md`: removed stale "does not compile" status block,
rewrote §6 (What Executes Now) to reflect actual state, replaced stale milestone plan
with accurate done/planned table, added §8 ABI compatibility boundary and §9 hard-won
ABI contracts.

Written `docs/EP_PLUGIN_EXPORT_PR.md`: PR description with verified validation numbers,
security finding dispositions, process note, known limitations, and §524 compliance statement.

Written `.squad/decisions/inbox/roy-ep-export-milestone.md`: milestone record and
ORT compatibility boundary.

Security: Holden's re-audit (2026-08-10T21:30Z) recorded 3 findings. N2/M1 resolved on
this branch. N1 (`compute_execute` unguarded) remains open — Deckard's responsibility.
Final verdict pending.

**Durable lesson:** Test harness mutex poisoning from `#[ignore]`d tests can cascade
failures across unrelated tests when running with `--include-ignored`. Always confirm
by running failing tests individually before reporting them as broken.



### 2026-08-10T20:12:35+0000 — EP plugin export architecture

Produced `docs/EP_PLUGIN_EXPORT.md`: architecture for exporting nxrt EPs as ORT plugin dylibs. Key decisions: single shared adapter crate (`onnx-runtime-ep-plugin`) owns all unsafe FFI; per-EP thin `cdylib` shim crates are mechanical (`export_ep_factories!` macro). Reuses inbound `UnionFind`/`SubgraphClaim`/`OrtGraphView::query_capabilities`. New outbound code: `OutboundGraphReader` (reads ORT's `OrtGraph*`) and `OutboundKernelContext` (bridges `OrtKernelContext` ↔ `TensorView`). CPU EP is the v1 candidate (no device dependency). CUDA EP blocked on allocator/stream/transfer callback design. Decision record at `.squad/decisions/inbox/roy-ep-plugin-export.md`.

### 2026-07-28/29 wave
### 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- Approved PR #313 decode-garble triage and byte-identity/fires regression guard; repeated-sentence output classified as natural greedy behavior.

### 2026-07-27T23:31:47-07:00 — Integration stress fixture audit

- Corrected `docs/research/testing/00-integration-stress-design.md` after PR #330 review: replaced stale fixture tier claims with an on-disk inventory of every `tests/fixtures/` directory, including `.onnx` vs `.onnx.textproto` formats and stress-harness suitability.
- Verified concrete references for `repl_e2e.rs`, `cli-ort`, bench binaries, and the Qwen/GLM env-var tests; clarified that current Qwen real-model env vars are native-CUDA locks, not CPU ORT fixtures.

### 2026-07-28T17:40:00+0000
Large-model recon documented 27B runtime and GPU-capacity blockers; Granite unfused MoE does not engage offload.

Full pre-compaction history in `history-archive.md`.