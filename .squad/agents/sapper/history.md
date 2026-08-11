# Sapper — History (compacted 2026-07-29)

**Role:** Systems/model-building implementer for onnx-genai and Mobius export/preprocess work. Owns native CUDA/CPU EP correctness and model-package metadata details; must preserve real-model parity, capture safety, Mobius lintrunner hygiene, and reviewer lockouts.

## Durable lessons
- onnx-genai uses its own `InferenceMetadata` (`inference_metadata.yaml`), not ORT-GenAI `genai_config`; Mobius PRs must pass lintrunner (RUFF + RUFF-FORMAT).
- Gemma4 VLM export is not solved by adding metadata fields: it needs rank-3 pre-patchified vision ingestion, embedding→decoder orchestration (`inputs_embeds`, not token IDs), and extended Mobius topology; E2B text runs needed explicit BOS.
- Input-embedding export must read the post-embed `Mul` scale from the graph, not hardcode `sqrt(hidden)`; the real f16 scale was `39.1875`.
- CUDA RMSNorm/SkipRMSNorm parity requires separately rounded f32 multiply/add; CUDA SiLU and acc4 scale boundaries need CPU-matching operation order/rounding.
- The token-16 K=4864 reduction-order delta (`1.9073486e-5`) is accepted because exact GPU emulation costs 8.4%; do not chase it as a correctness bug without new evidence.
- Loop v1 was rejected because scan accumulation reserved from untrusted `M` and carried shapes were not validated; Sapper was locked out and Leon owned the remediation.
- Control-flow graph attributes retain ordered typed formal I/O and scoped inline initializers; `ChildExecutor` must preserve lexical captures and branch-specific `If` caches.
- onnx-rs `SimpleShardedDimProto.dim` is optional; preserve IR13 checker/codec round trips.
- CPU op traps: OneHot out-of-range indices are all-off; BitShift direction is required; GridSample opset 16 rejects rank-5 while opset 20 keeps 2-D/3-D support; unsafe Unique String execution was removed.
- CUDA graph/kernel work must stay capture-safe and portable across supported SM architectures, not only sm_90; explicit int4 zero-points must preserve symmetric zp=8 fast paths.
- WP-B optional fallback validation treats raw `GraphProto.input` as authoritative.
- Rewind policy split is canonical: public `restore_session`/`rewind_session_to` reject unsupported rewinds before mutating tokens/KV, while internal speculative runner rewind may use the allow-runner path; `RewindRequest` replaced raw `(len, policy)` tails.
- If local engine tests need the pinned ORT DLL, place it beside the test binary.
- Reviewer lockouts remain binding, especially the Loop remediation handed to Leon and any artifact a reviewer explicitly reassigns.

## Recent work (current wave, ~2026-07-28/29)

## 2026-07-26T22:38:02+00:00 — Mobius PR triage handoff

- Prepared Mobius PRs #404/#423/#430 for Justin review without merging. #404 replacement branch `sapper/404-rebase` at `fa30534` resolves conflicts and review comments; #423 `squad/hythe-deepseek-moe-phase1` at `40846bb` and #430 `test/l4-l5-golden-new-models` at `d1d235e` have current review fixes, Ruff clean, and focused tests passing.

## 2026-07-27T21:45:00-07:00 — Runtime fork rewind policy split

- Revised PR #291 runner-backed rewind handling so the unsupported-runner policy is explicit and limited to the public session rewind API.
- Internal speculative target rejection and draft realignment validation now use the allow-runner policy; public `restore_session`/`rewind_session_to` keep fail-closed ordering before session removal or token/KV mutation.
- Updated the stale tiny PastPresent checkpoint test to expect the clean unsupported error and added model-free regression coverage for the speculative runner rewind policy boundary.
- 2026-07-28T00:55-07:00 follow-up: replaced the raw `(len, policy)` tail with `RewindRequest` to avoid an 8-argument helper and fixed remaining kv_bridge test call sites. Full engine lib tests now pass locally after staging the pinned ORT DLL beside the test binary.

## 2026-07-28T07:46:01+00:00 — Wave 5
- PR #331 (`52b1fc59`) merged: added CUDA GatherND, SpaceToDepth, and EyeLike; #67 remains open for later coverage batches. Hallett independently approved the GPU parity and mutation-probe evidence.

## 2026-08-11 — B2: ReleaseEpFactory ABI fix

- Fixed `export_ep_factories!` macro: `ReleaseEpFactory` now returns `*mut OrtStatus` per `onnxruntime_ep_c_api.h:2669`, not `void`.
- Caught panics now surface as error `OrtStatus` instead of being silently swallowed.
- Verified `CreateEpFactories` matches header — no second mismatch.
- CPU and CUDA hand-written shims still return `void` — owners must update (not my files).
- Told Chew to update ABI test type alias to `-> *mut ort::OrtStatus`.

## 2026-08-11 — B2 follow-up: CPU shim ReleaseEpFactory fixed

- Updated hand-written `ReleaseEpFactory` in `crates/onnx-runtime-ep-cpu-plugin/src/lib.rs` to return `*mut OrtStatus` (was `void`).
- Panic path now surfaces as `panic_to_fail_status(...)` rather than being silently swallowed.
- Shim stays hand-written because `CreateEpFactories` calls `create_ep_factories_with_registry` (not exposed via the macro); annotated with keep-in-sync comment mirroring the macro arm.
- Audited `CreateEpFactories` — already returned `*mut OrtStatus`, no drift found.
- CUDA shim is Iran's file; not touched.
- Build: `cargo build -p onnx-runtime-ep-cpu-plugin` ✓. Tests: 154 lib + 9 parity ✓. Clippy clean ✓. fmt clean ✓.

Full pre-compaction history in `history-archive.md`.
