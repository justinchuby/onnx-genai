# Zhora — History (compacted 2026-07-29)

**Role:** Server developer for the OpenAI-compatible HTTP surface and related model/KV lifecycle plumbing. Keep handlers thin over the batched driver, preserve deterministic model routing/admission contracts, and honor reviewer lockouts and CLI maintainer-tool boundaries.

## Durable lessons
- Server handlers stay thin over the batched engine driver; modular routes/driver/SSE/types/state/session/metrics/image/audio boundaries are part of the surface.
- Model lifecycle M1/M2/M3 is complete: deterministic default, no silent fallback for unknown model, runtime load/unload/admin endpoints gated by `--enable-admin-endpoints`, LRU keeps at least one model and prefers non-default victims.
- Registry lock discipline is canonical: `std::sync::RwLock` only, never held across `spawn_blocking` or `.await`; per-id async load guards prevent double-builds.
- Embeddings routing bug was found by review and fixed by Rachael; Zhora is locked out of that follow-up.
- KV connector invariant: equal `KvCacheKey` proves identical token sequence from position 0 through chunk end; prefix-dependent chunk hashing defuses K4 materialization traps.
- `cpu_load_ms_per_page` must scale by pages needing upload, not act as a constant; the regression is locked by `cpu_load_ms_scales_by_configured_rate`.
- Full-spec onnx-rs serde attempt was rejected because stale IR10 proto plus base64 sidecar did not meet current ONNX IR13/native-text requirements; Batty owns revision under lockout.
- CLI is a development/maintainer harness, not a consumer product; source of truth for queued CLI work is `docs/research/cli/00-backlog.md`, and remote-client mode is out of scope.
- REPL/plain-mode compatibility matters: `Plain` preserves piped `main` behavior for `//...` and `/help <arg>`; newline decisions depend on whether the current turn used live rendering.
- Generate stats contract: compact token stats default only when stdin/stdout are terminals, are stderr-only, are suppressed by `--profile`, and piped stdout remains byte-stable generated text.
- Cache prefetch is valid only in eviction-neutral cache regimes; #364 merged only after that correction.

## Recent work (current wave, ~2026-07-28/29)

### 2026-07-27T13:10:00-07:00 — CLI backlog now on main
Scribe note: the CLI dev-tool charter and prioritized backlog from the merged CLI improvement track are now on main at `docs/research/cli/00-backlog.md`. Use that file as the source of truth before picking up queued CLI backlog work.

### 2026-07-27T14:55:00-07:00 — REPL Phase 1 rejection revision
- Took over PR #289 revision after Gaff's rejection under reviewer lockout.
- Made REPL command parsing mode-aware: `Plain` preserves `main` piped behavior for `//...` and `/help <arg>`, while `Tty` keeps the new rich affordances.
- Fixed the post-generation newline decision to depend on whether the current turn actually used live rendering, not on the reusable renderer lifecycle state.
- Added lib and e2e regressions; local ONNX Runtime mismatch still prevents model-loading e2e verification.

### 2026-07-27T18:46:09-07:00 — generate default stats stdout/stderr contract
- Text `onnx-genai generate` now follows the REPL default: compact stats are on only when the shared REPL input-mode detector sees stdin and stdout as terminals, and `--no-stats` opts out.
- The compact stats line is stderr-only and suppressed by `--profile`; piped stdout remains byte-stable generated text.
- Image/audio `generate` and `transcribe` keep compact token stats off by default; their non-token throughput belongs in `--profile`.

### 2026-07-28T17:40:00+0000
#364 merged after the implementation was corrected to use prefetch only in eviction-neutral cache regimes.

Full pre-compaction history in `history-archive.md`.
