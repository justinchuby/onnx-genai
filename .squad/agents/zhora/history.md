# Zhora — History (compacted 2026-08-12)

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

## Historical context (2026-07-27 through 2026-08-11)

Jul-27/28 wave: CLI backlog on main; REPL Phase 1 rejection revision (PR #289); generate stats stdout/stderr contract; #364 merged. Aug-11 wave: PR #762 documentation accuracy pass (commit `bb280c0ea`; 8 stale SHA refs updated; PR marked ready for review). Full entries in `history-archive.md`.

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

## 2026-08-12 — PR #32001: Deduplicate Apple Accelerate validation

- Removed dead-code copy of `--use_apple_accelerate` validation from `build.py:881-897` (BuildError path never reached because `build_args.py` parser.error() exits first).
- Kept single source of truth in `build_args.py` — matches existing pattern for argument validation.
- Improved non-macOS error message to distinguish "wrong OS" from "wrong arch".
- 13/13 tests pass unchanged (all assert SystemExit, matching parser.error() path).
- cmake_args assertion not added — needs heavier harness (generate_build_tree mocking). Flagged for future.
- ruff format/check clean. Leak check clean. Head `3a0bd75aa3`. PR remains draft.

## 2026-08-12 — PR #32001: Deduplicate validation to single site (lockout revision)

- Holden (locked out after authoring the review) found dead-code duplication in `build.py`.
- Removed `build.py:881-897` BuildError block. Left only cmake_args append + comment pointing to `build_args.py`.
- Improved non-macOS message: "requires Apple Silicon host" (distinguishes wrong OS from wrong arch).
- 13/13 tests pass unchanged. ruff format/check clean. Head `3a0bd75aa3`.
- PR #32001 marked **ready for review**.
