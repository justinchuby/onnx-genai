# Rachael — History (compacted 2026-08-12)

**Role:** CLI/server/API implementer for onnx-genai. Owns OpenAI-compatible server behavior, REPL/maintainer-tool UX, endpoint routing, streaming/session invariants, and user-visible runtime controls while preserving non-TTY byte stability.

## Durable lessons
- Server surface includes `/health`, `/v1/models`, `/v1/chat/completions`, SSE streaming, `X-Session-Id`, session lifecycle routes, tools/tool_choice/tool-role handling, JSON response constraints, FIM, image parts, and audio routing.
- Server DoS/session hardening is canonical: `max_output_tokens=4096`, `max_sessions=256` LRU, 128-bit CSPRNG session ids, context-token caps, loopback/no-auth deployment notes.
- Static-cache HTTP concurrency uses a single engine driver thread and channels; do not reintroduce shared Engine locking.
- Batched-driver admission is bounded by `max_pending` with HTTP 429; output delivery must be non-blocking.
- `/v1/debug/*` must be default-off and redact session identifiers.
- Zero-copy mmap initializer borrowing landed; producer-aliasing soundness restrictions added by Zhora.
- Qwen Sigmoid fusion must recognize `Mul(x, Sigmoid(x))`, allocation-free, with multi-consumer negative coverage.
- REPL Phase 1: non-TTY stdin/stdout byte-stable; TTY uses `reedline`; slash parser from declarative registry; `/fork`/`rewind` out of Phase 1.
- TTY output owns exactly one separator newline when generated text lacks a trailing newline.
- EP test hardening: `disable_cpu_ep_fallback=1` + EP assignment assertions; non-vacuity proven by forced wrong-EP assertions.
- `NXRT_REQUIRE_ORT_TESTS` gate must route all skip paths (fixture-missing, dlopen failure) through the gate; `.ok()?` silent skips are forbidden.
- `find_ort_lib_dir` must honour `NXRT_ORT_LIB_DIR` → `CARGO_TARGET_DIR`/debug/build → workspace default with platform-aware library names.

## Historical context (pre-2026-08-11)
Wave coverage through 2026-07-28: CLI UX research, REPL redesign, Phase 1 TTY/plain split with `reedline`, PR #300/#346 lockout revisions, REPL stats two-line block. Full detail in `history-archive.md`.

## Recent entries

## 2026-08-11 — PR #762 final test hardening

**Commit:** `c1d2556b5`

- `layernorm_dynamic_axis.rs`: `disable_cpu_ep_fallback=1` + EP assignment assertion for `LayerNormalization`.
- New conformance tests: `conformance_add_float16`, `conformance_add_bfloat16`, `conformance_layer_norm_multi_output`, `conformance_layer_norm_neg_axis`, `conformance_rms_norm` — all with assignment assertions.
- 14 total EP assignment assertions across two test files.
- BL1 shape assertions (mean/inv_std shape `[2, 3, 1]`) fully preserved. Non-vacuity proved.

## 2026-08-12 — PR #762: Test-integrity gate hardening

- Fixed `find_ort_lib_dir` to honour `CARGO_TARGET_DIR` (matching `cdylib_resolve.rs` logic).
- Routed fixture-missing and dlopen-failure skip paths through the `NXRT_REQUIRE_ORT_TESTS` gate.
- Added `NXRT_REQUIRE_ORT_TESTS=1` to CI `CLI ORT (Linux x86_64)` lane + cpu-plugin test step.
- Test counts: 40 passed, 0 failed. Commit: `88f9de8df`.

## 2026-08-12 — PR #762 ready for review (gate hardening complete)

All five red CI jobs trace to pre-existing `onnx-genai-server` failure on `main`; branch does not touch that crate. 283 passed / 0 failed. Gaff confirmed gate coverage genuinely closed. Freysa completed `find_ort_lib_dir` consolidation. PR #762 marked ready for review.

*Full pre-2026-08-11 history in `history-archive.md`.*

## 2026-08-12 — PR #32001 (microsoft/onnxruntime): Cross-target Accelerate blocker

- Added `parser.error()` rejections for `--android`, `--build_wasm`, `--rv64` alongside `--use_apple_accelerate`.
- Refactored all Apple Accelerate rejection tests to assert diagnostic message content (not bare SystemExit).
- 4 new tests (android, build_wasm, build_wasm_static_lib, rv64). Test count: 13→17.
- Documented Catalyst/macabi CMake limitation: `PLATFORM_NAME` comes from external toolchain, no reliable direct-CMake detection.
- Commit: `184f76a00e`. PR left in draft for Opus review.
