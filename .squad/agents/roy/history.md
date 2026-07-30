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

## Recent work (current wave, ~2026-07-28/29)
### 2026-07-27T09:15:14-07:00 — CLI architecture triage

- Reviewed `crates/onnx-genai-cli` command surface and module boundaries for Justin's CLI-improvements research track.
- Filed `docs/research/cli/01-architecture-triage.md`: current command surface is useful, but `lib.rs` is a 1,134-line parser/dispatch/test god module; top priorities are parser refactor, JSON/output contracts, model-source grammar normalization, and exit-code taxonomy.
- Dropped decision inbox note `roy-cli-improvements.md` recommending the next CLI wave be treated as an interface-contract project before adding model-management/config features.

### 2026-07-27T09:27:53-07:00 — CLI dev-tool backlog consolidation

- Reconciled Justin's directive that `onnx-genai` CLI is a maintainer/debug harness, not a consumer local-inference product.
- Wrote `docs/research/cli/00-backlog.md` as the research entry point: P0 now prioritizes hidden live stats, structured output, local bench/batch harness, reachable engine features, and help snapshots.
- Explicitly rejected remote-client mode, model registry/pull, conversion/fine-tune product loops, and a standalone lib.rs split; Wierzbowski's earlier split makes more refactor only worthwhile when tied to P0/P1 work.

### 2026-07-27T13:10:00-07:00 — CLI backlog now on main
Scribe note: the CLI dev-tool charter and prioritized backlog from the merged CLI improvement track are now on main at `docs/research/cli/00-backlog.md`. Use that file as the source of truth before picking up queued CLI backlog work.

### 2026-07-27T16:44:54Z — Wave 9 update
Approved PR #282 after mutation-proven equivalence for tree speculative decoding core.

### 2026-07-27T14:34:22-07:00 — Rules type-safety standard

- Added a project rule requiring Rust types to enforce invariants: newtypes for transposable primitives, capability values over late hot-path checks, and ownership/borrowing for aliasing.

### 2026-07-27T14:38:08-07:00 — Rules condensation pass

- `RULES.md` was 8,923 bytes on `main`; the new rule initially pushed it to 10,167 bytes, and the condensation pass brought it to 7,149 bytes, smaller than the starting file despite adding a rule.
- Condensed the new Rust type-safety rule to the core invariants: invalid states, newtypes, capability values, and ownership/borrowing; dropped repeated unsafe/Miri, property-testing, and TLA+ detail.
- Folded graph-fusion detail into the model/vendor/EP-agnostic rule without changing its force.

### 2026-07-27T14:42:22-07:00 — Rule 2 force restored

- Restored explicit no-hardcoded-architecture enumeration, fusion generalization, and review-blocking consequence in `RULES.md` rule 2 after Justin flagged the condensation as weakening the project identity rule.
- Removed the stable-ABI wheel rule's stale inbox link; the corresponding abi3 details were found in archived decisions, not active `.squad/decisions.md`.

### 2026-07-27T20:42:16-07:00 — Integration stress design

- Authored `docs/research/testing/00-integration-stress-design.md` for Justin's request to stress-test real multi-turn backend mechanisms.
- Centered the design on invariants rather than exact stochastic output: termination, non-empty committed turns, reasoning progress, repetition bounds, token/history/KV consistency, scheduler liveness, sampling observability, feature-state coherence, and reproducible failure packets.
- Recommended phase 1 as a committed tiny reasoning fixture plus per-PR CPU ORT REPL stress; CUDA DeepSeek/native and ORT CUDA shared-GQA failures require self-hosted/manual GPU lanes.
- Dropped decision inbox note `.squad/decisions/inbox/roy-stress-test-design.md`.

### 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- Approved PR #313 decode-garble triage and byte-identity/fires regression guard; repeated-sentence output classified as natural greedy behavior.

### 2026-07-27T23:31:47-07:00 — Integration stress fixture audit

- Corrected `docs/research/testing/00-integration-stress-design.md` after PR #330 review: replaced stale fixture tier claims with an on-disk inventory of every `tests/fixtures/` directory, including `.onnx` vs `.onnx.textproto` formats and stress-harness suitability.
- Verified concrete references for `repl_e2e.rs`, `cli-ort`, bench binaries, and the Qwen/GLM env-var tests; clarified that current Qwen real-model env vars are native-CUDA locks, not CPU ORT fixtures.

### 2026-07-28T17:40:00+0000
Large-model recon documented 27B runtime and GPU-capacity blockers; Granite unfused MoE does not engage offload.

Full pre-compaction history in `history-archive.md`.