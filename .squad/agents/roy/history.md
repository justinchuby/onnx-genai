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
### 2026-08-10T22:56Z — EP plugin PR doc accuracy sweep

**Task:** Correct stale claims in `docs/EP_PLUGIN_EXPORT_PR.md` and related docs after all agents finished.

**Validation I personally ran (commit `bad3682`, 2026-08-10T22:56Z):**

- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → `Finished dev profile in 1.59s` — zero errors, zero warnings.
- `cargo test -p onnx-runtime-ep-plugin --lib` → `82 passed; 0 failed; 0 ignored` in 0.00s.
- `cargo test -p onnx-runtime-ep-cpu-plugin` → `6 passed` (lib) + `15 passed` (integration) = **21 total, 0 failed, 0 ignored** in 0.73s. Full parallel run.
- `cargo check --workspace` → `Finished dev profile in 0.26s` — clean.

**Stale claims corrected in `docs/EP_PLUGIN_EXPORT_PR.md`:**

1. Removed the "mutex poison cascade / 4-test parallel failure" section entirely — full suite now passes in parallel, zero ignored.
2. `conformance_two_sessions` is no longer `#[ignore]`d and no longer fails — was a test-assertion bug (Pris fixed); stated plainly with root-cause.
3. `stress_register_run_unregister_cycles` (25 cycles) now cited as the regression test for the UAF fix (Deckard, `c92838d`).
4. Holden's verdict updated from 🔴 RED (pending) to 🟡 YELLOW — May ship (2026-08-10T22:42Z) with all N1/N2/N3/UAF findings resolved.
5. Security table rewritten: all findings marked RESOLVED; two LOW post-merge advisories (NEW-1/NEW-2) recorded honestly.
6. `validate_dims` safety fix called out as genuine safety fix (not cleanup) — was never called before Leon's commit `2fb7150`.
7. f16/bf16 coverage gap (Pris) documented: `GetKernelRegistry` not wired, no fake test written.
8. CUDA EP limitation: stated both halves — hardware blocked AND design work remaining.
9. Process section updated: Leon (N1, N2), Isidore (N3), Deckard (UAF), Pris (two_sessions) all credited.
10. Follow-ups list stripped of already-done items; only genuine open items remain (NEW-1/NEW-2 LOW, f16/bf16, CUDA design, §524 trait).

**Also swept `docs/EP_PLUGIN_EXPORT.md`:**
- "PLANNED (not yet true)" section replaced with "CURRENT STATUS" reflecting passing tests and YELLOW verdict.
- Known-gaps table: `conformance_two_sessions` marked Closed; Holden sign-off marked Closed.

**Durable lesson:** Pre-final docs written mid-work will always be stale. Always re-run validation commands at head and quote actual output — do not copy numbers from coordinator memo on faith.

---

## 2026-08-10T23:30Z — EP inventory, M2 state, §524 update (branch `squad/ep-plugin-parity-cuda`)

**Task:** Complete EP inventory, update PR doc for both milestones, state §524 compliance honestly.

**EP inventory result (re-run at HEAD `5fa8cb2a8`):**
- Production EPs: exactly 2 — `CpuExecutionProvider` and `CudaExecutionProvider`.
- Excluded non-EPs confirmed: `LegacyOrtEp` (inbound adapter), `PluginExecutionProvider` (inbound bridge), `onnx-runtime-eager` (orchestrator), `mlas-sys` (BLAS lib), all test/mock impls.
- Metal EP (`../onnxruntime-mlx`): **OUT OF SCOPE** — `ls /workspace/dev/` confirms only `onnx-genai` present. Sibling repo must be cloned separately.
- QNN NPU EP: **ASPIRATIONAL** — no crate, no stub, no design. Luba's domain; not in scope this wave.

**M2 state at HEAD `5fa8cb2a8`:** `squad/ep-plugin-parity-cuda` = `squad/ep-plugin-export` (zero new M2 commits).
- Leon NEW-1 (`catch_unwind` on `compute_release_state`): **DONE** — code at `compute.rs:1563` confirms it landed before M1 hand-off, not post-merge.
- Pris trait↔C-ABI parity tests: **NOT YET COMMITTED** — §524 trait-half proof pending.
- Deckard NEW-2 + `GetKernelRegistry`: **NOT YET COMMITTED**.
- Nabil CUDA shim (`onnx-runtime-ep-cuda-plugin`): **NOT YET COMMITTED** — crate does not exist.

**Validation observed (`cargo test -p onnx-runtime-ep-plugin --lib` + `cargo test -p onnx-runtime-ep-cpu-plugin`):**
- 82 lib unit tests, ok.
- 6 lib integration tests + 15 ORT e2e tests = 21 total, all ok, 0 ignored.

**§524:** C ABI half complete. Rust trait half structurally wired but §524 proof (Pris's parity tests) not yet committed. Native nxrt dynamic ABI unimplemented in both milestones.

**Recommendation:** Two stacked PRs (not one). M1 is independently mergeable; M2 is not.

**Deliverables written:**
- `.squad/decisions/inbox/roy-ep-inventory-complete.md`
- `docs/EP_PLUGIN_EXPORT_PR.md` updated (both milestones, branch structure, M2 state, §524 table, corrected NEW-1 status, Follow-Ups updated).
- `.squad/agents/roy/history.md` (this entry).

---

## 2026-08-10T23:52Z — Milestone 2 documentation pass

**Branch:** `squad/ep-plugin-parity-cuda` at `5a5b40877`

**Task:** Re-verify M2 state from scratch; update all EP plugin export docs accurately.

**Validation observed (Roy, personal):**

| Command | Result |
|---------|--------|
| `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` | **FAILS** — 2 errors `ep.rs:1041,1047` (`needless_borrows_for_generic_args`) |
| `cargo test -p onnx-runtime-ep-plugin` | **141 pass** (132 lib + 9 parity), 0 fail |
| `cargo test -p onnx-runtime-ep-cpu-plugin` | **23 pass** (6 + 17 e2e incl. f16/bf16), 0 fail |
| `cargo check --workspace` | **pass** |

**Key findings:**

1. M2 largely complete: parity proven (9 tests), f16/bf16 end-to-end proven (bit-exact), device surfaces exist, CUDA shim crate scaffolded.
2. M2-1 (EP leak in `stream_release`) — OPEN, Leon not yet fixed.
3. M2-2 (misleading doc `device.rs:86`) — OPEN, Leon not yet fixed.
4. Clippy regression: 2 trivial errors at `ep.rs:1041,1047` — pre-merge blocker for M2.
5. Declined-set correction: Squeeze/ReduceMean/Conv now resolve; only opset-13 data-dependent Unsqueeze and NonZero are truly Declined.
6. Native nxrt dynamic ABI: still unimplemented in both milestones — §524 gap documented plainly.

**Deliverables written:**
- `docs/EP_PLUGIN_EXPORT_PR.md` — full M2 update (branch table, Validation, Milestone 2 section, Security, §524, Known Limitations, Follow-Ups).
- `docs/EP_PLUGIN_EXPORT_INVENTORY.md` — summary table and Roy verification note updated for M2 CUDA scaffold state.
- `.squad/decisions/inbox/roy-milestone2-status.md` — final status of both milestones.
- `.squad/agents/roy/history.md` (this entry).

---

## 2026-08-11T00:00Z — Final doc correction pass (commit `3ab0ded68`)

**Branch:** `squad/ep-plugin-parity-cuda` — HEAD confirmed at `3ab0ded68`.

**Task:** Third and final correctness pass on `docs/EP_PLUGIN_EXPORT_PR.md` after Leon's M2-1/M2-2 fixes and Deckard's clippy fix landed post `5a5b40877`.

**Validation observed (Roy, personal, `3ab0ded68`):**

| Command | Result |
|---------|--------|
| `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` | **CLEAN** — 0 errors, 0 warnings (Finished in 3.56s) |
| `cargo test -p onnx-runtime-ep-plugin` | **142 pass** (133 lib + 9 parity), 0 fail (lib +1 from Leon's `stream_release_reclaims_owned_ep_no_leak` regression test) |
| `cargo test -p onnx-runtime-ep-cpu-plugin` | **23 pass** (6 lib + 17 integration), 0 fail, 0 ignored |
| `cargo check --workspace` | **CLEAN** — Finished in 0.25s |

**Stale claims removed from `docs/EP_PLUGIN_EXPORT_PR.md`:**

1. M2 status row: removed "one MEDIUM resource-leak and one LOW doc advisory open; clippy regression must be fixed" — replaced with "all findings resolved; Holden's re-verification was not separately run."
2. Pre-merge blocker paragraph: removed the clippy blocker warning; replaced with "Both M1 and M2 are now green and mergeable."
3. Validation section: removed stale `5a5b40877` clippy failure block with 2 error listings; replaced with clean output at `3ab0ded68`.
4. Test counts updated: 132 → 133 lib; 141 → 142 total for `onnx-runtime-ep-plugin`.
5. Security table: M2-1 (MEDIUM) and M2-2 (LOW) moved from "Open findings" to "Resolved findings" with Leon as fixer, commit `3ab0ded68`, and evidence (double-free analysis, Drop counter regression test, comment correction).
6. Holden's verdict: 🟡 YELLOW retained; added honest note that M2-1/M2-2 are resolved but Holden did not re-verify at `3ab0ded68`.
7. Engineer status table: Leon row updated from 🔴 NOT YET DONE → ✅ DONE.
8. Known Limitations: removed "M2 clippy regression" bullet.
9. Follow-Ups 5/6/7: marked as DONE (strikethrough).
10. Process section: added M2-1 Reviewer Rejection Protocol note (Nabil locked out; Leon fixed) and Deckard clippy fix.
11. M2 commits paragraph updated to include `3ab0ded68`.

**Things confirmed unchanged (per instructions — must stay honest):**
- Native nxrt dynamic ABI: still 🔴 Not implemented in §524 table.
- CUDA EP: still blocked (no toolkit/GPU + design work remaining); mock-tested surfaces are genuine progress but not a working CUDA EP.
- f16/bf16: distinction preserved — our EP claims and executes, ORT does not fall back.
