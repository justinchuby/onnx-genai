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

### 2026-08-11T19:30:00Z — PR #762 documentation accuracy pass
- Rewrote the PR body for #762 (`squad/ep-plugin-parity-cuda`): replaced stale M1/M2 description with accurate account of 6 review rounds, 3 late-breaking blockers fixed, strengthened test story (EP assignment proof), exact test numbers (269/0 EP crates).
- Swept docs: updated 8 stale SHA refs in `NXRT_ABI.md`, stale commit in `EP_PLUGIN_EXPORT.md`, rewrote header in `EP_PLUGIN_EXPORT_PR.md`, updated `EP_PLUGIN_EXPORT_INVENTORY.md`.
- `CUDA_EP_STATUS.md` was already accurate — no changes needed.
- No code modified. PR remains draft. Decision record at `.squad/decisions/inbox/zhora-final-docs-762.md`.

## 2026-08-11 — PR #762 documentation accuracy pass

**Task:** Docs and PR-body accuracy pass before undrafting.

**Commit:** `bb280c0ea`

- PR body completely rewritten: all three late-breaking blockers documented, test story upgraded (14 assignment assertions, non-vacuity proof), correct numbers (269 EP tests, 4598/20/436 workspace), explicit "What Is NOT Proven" section for CUDA.
- `docs/NXRT_ABI.md`, `docs/EP_PLUGIN_EXPORT.md`, `docs/EP_PLUGIN_EXPORT_PR.md`, `docs/EP_PLUGIN_EXPORT_INVENTORY.md`: 8 stale SHA refs (`fb9d757b3`, `62f23440f`) → `c1d2556b5`.
- `docs/CUDA_EP_STATUS.md`: already accurate, no change.
- No CUDA hardware claims made anywhere.

**Outcome:** Gaff's final review confirmed doc accuracy. PR marked ready for review.

### 2026-08-12 — PR #31973: Fix stale algorithm references in LayerNorm test comments

**Task:** Rewrite the ReferenceLayerNorm oracle comment block which described the kernel as Welford (it is centered two-pass). Also fix ScalarFp32Baseline comment claiming to match current kernel.

**Changes:** Comments only in `onnxruntime/test/mlas/unittest/test_layernorm.cpp`. Two sites updated:
1. Lines 58-85: oracle block now names the kernel's actual algorithm, explains uncentered vs centered distinction, preserves independent-oracle argument.
2. Line 436: ScalarFp32Baseline clarified as historical baseline.

**Other stale references found:** ~30 other Welford mentions exist in the file (in disabled accuracy-comparison tests, WelfordFp64Reference helper, error-reporting labels). These are accurate in their own context — they describe the *historical* Welford baseline or the fp64 reference function, not the current kernel. No changes needed.

**Validation:** Fresh build; 41 passed, 2 disabled; 43/43 with disabled. clang-format clean. No leaks. Pushed as `9a4fcaeaa4`.

**Outcome:** PR remains draft, pending fresh Opus approval.

## 2026-08-12 — PR #31973 lockout revision: oracle block and ScalarFp32Baseline comments

- Coordinator found a fourth stale Welford comment site missed by Luv's review: oracle block at `test_layernorm.cpp:58-85`.
- Rewrote oracle block to name centered two-pass, explain uncentered `E[x²] − mean²` reference form (fp64, safe only due to precision), preserve independent-oracle argument, replace "Do NOT fix this to Welford" with clearer directive.
- Clarified `ScalarFp32Baseline` comment as historical baseline, not a mirror of current kernel.
- Assessed ~30 other Welford mentions as accurate in context (Welford baseline history, fp64 helper).
- Fresh build: 41 passed + 2 disabled; 43/43 with disabled; clang-format clean. Head `9a4fcaeaa4`. PR remains draft.
