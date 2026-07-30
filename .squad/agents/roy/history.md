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
### 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- Approved PR #313 decode-garble triage and byte-identity/fires regression guard; repeated-sentence output classified as natural greedy behavior.

### 2026-07-27T23:31:47-07:00 — Integration stress fixture audit

- Corrected `docs/research/testing/00-integration-stress-design.md` after PR #330 review: replaced stale fixture tier claims with an on-disk inventory of every `tests/fixtures/` directory, including `.onnx` vs `.onnx.textproto` formats and stress-harness suitability.
- Verified concrete references for `repl_e2e.rs`, `cli-ort`, bench binaries, and the Qwen/GLM env-var tests; clarified that current Qwen real-model env vars are native-CUDA locks, not CPU ORT fixtures.

### 2026-07-28T17:40:00+0000
Large-model recon documented 27B runtime and GPU-capacity blockers; Granite unfused MoE does not engage offload.

Full pre-compaction history in `history-archive.md`.