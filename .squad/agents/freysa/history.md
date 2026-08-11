# Freysa — History

## 2026-07-12: Joined
Hired as MPS Perf & Testing engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns per-kernel correctness (vs ORT CPU reference), benchmarking, Metal profiling, and E2E testing through the onnx-genai runtime (`ONNX_GENAI_EP=metal`). Targets: beat llama.cpp Metal / LM Studio / Foundry Local on Apple Silicon. Reuses the onnx-genai benchmark harness (`scripts/compare_runtimes.sh`, `compare.rs`). Pairs with Sebastian. Correctness (coherent output) gates every perf claim.

- 2026-07-14T19:05:00Z — Pipeline API seams (`ChatTemplate::builtin_default`, `Engine::tokenize`, `embed_text*`) recorded in decisions; Holden review GREEN for commit `ecba2c1`.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Added missing C1 shape handlers and initial DLPack import support; both were consolidated in the July 15 coverage/interoperability work.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed the unified string-serde surface as approve-with-notes.

### 2026-07-16T00:00:00Z — onnx-rs Python binding review cycle
Rejected Batty's initial `onnx_rs` binding for lossy paths, an `exists()` preflight, and swallowed `__fspath__` exceptions. Cleared Deckard's `5b348b5` revision after targeted Rust tests and six Python regressions verified lossless paths and native filesystem errors.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Freysa's raw-protobuf admission rejection was resolved in the final WP-B3 path.

## 2026-07-28T09-10-28+00-00 — PR #338 review
- Approved Luv's #67 CUDA `Pad`/`Range` batch after H200 GPU 2 passed 174/174 parity cases, the coverage gate passed, a content-corrupting mutation probe failed as expected, and default-target warnings-denied Clippy was clean. PR #338 merged as `c59383db`.

## 2026-08-11 — EP assignment proof (PR #762)

- Added `session.disable_cpu_ep_fallback=1` to `conformance_setup()` in `plugin_ort_e2e.rs`
- All 22 conformance tests (plus stress test) now prove our EP actually claimed nodes
- `conformance_mixed_partition` exempted (intentionally tests partition with fallback)
- Proved non-vacuity: with flag applied universally, mixed_partition correctly fails with ORT's "fallback disabled" error
- Profiling assertion not practical: ORT 1.27 plugin-EP API has no post-session per-node provider query
- 23 tests pass, 0 fail. No test count decrease.
