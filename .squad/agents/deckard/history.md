# Deckard — History (compacted 2026-07-29)

**Role:** Systems developer on the Rust runtime, ORT2 loader/shape/IR/EPContext tracks, and CPU/CUDA execution performance. Preserve model-agnostic dispatch, fail-closed claims, checked arithmetic, byte-exact serialization, precision-sensitive tests, and reviewer-lockout ownership transfers.

## Durable lessons
- Repeated invariants: model-agnostic dispatch, fail closed at claim time, checked arithmetic, byte-exact serialization, and precision-sensitive tests.
- Parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.
- Deckard owns canonical revisions after lockouts for shape inference, IR dtype, EPContext writer, and 2026-07-19 CPU reduction/activation dtype waves.
- CSA B5 initial five-output ratio-4 assembly misrouted to the ratio-128 kernel; Roy's ratio-keyed fix is canonical.
- CUDA token-index-10 drift root cause was SkipSimplifiedLayerNorm RMS FMA contraction; fix landed in `de3c556`/`ccf994c`.
- `cudarc` CUDA feature unification: ORT keeps CUDA 12.6 weak default, engine disables ORT defaults and selects CUDA 13.0 with `onnx-runtime-ep-cuda`.
- GridSample opset-16 rank-5 acceptance was rejected; Sapper's correction is canonical.
- Replay binding metadata caching gained only +0.23%; do not reattempt raw-address correctness-sensitive hot-path caching without stronger evidence.
- CUDA graph capture fixes require exact warmed signatures, persisted GQA scratch, handle ownership correctness, and replay metadata bounds.
- `ONNX_GENAI_EP=cuda` without ORT CUDA and runtime CUDA-provider unavailability are hard session errors; `ONNX_GENAI_REQUIRE_CUDA=1` only gates native CUDA node-level fallback after CUDA is compiled and selected.
- Public rewind/checkpoint APIs may use existing speculative helpers; public `fork_session` stays capability-gated and fail-closed until backend runner state can be safely cloned/imported.
- Backend reporting must show the loaded engine's resolved backend; `auto` is only a requested backend when it differs.
- Runtime ORT selection order is machine-independent: explicit env vars, active conda/venv, target-cache fallback, pathful API-mismatch diagnostics; host paths are validation evidence, not docs.
- Fitted performance constants are acceptable only when labelled as fitted and bracketed by measured data; a false rationale is worse than no rationale.

## Recent work (current wave, ~2026-07-28/29)
## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Under Moss's lockout, repaired PR #266 ReduceLogSumExp numerical stability with a dedicated two-pass reduction; Ferro approved and the PR merged.

## 2026-07-27T16:44:54Z — Wave 8 update
- Took ownership of PR #276 / #87 fix cycle after Ferro REQUEST-CHANGES; must address GPU test build break and WAR-safe driver semantics/docs.

## 2026-07-27T12:30:00-07:00 — PR #279 CUDA fallback semantics

- Adjudicated Copilot's CUDA documentation findings against source. `ONNX_GENAI_EP=cuda` without `onnx-genai-ort/cuda` is rejected at ORT session creation, runtime CUDA-provider unavailability is also a hard session error, and `ONNX_GENAI_REQUIRE_CUDA=1` only gates native CUDA node-level fallback to CPU after CUDA is compiled and selected. Corrected CLI/cargo/CAPI/Python docs and comments; verified `cargo tree`, `cargo fmt --check`, and `cargo build -p onnx-genai-cli`.

### 2026-07-27 — Runtime capability inventory for REPL design

Authored `docs/research/cli/04-runtime-capability-inventory.md` as Deckard. Key finding: the runtime has strong low-level primitives (paged KV CoW fork, rewind/checkpoint, prefix reuse, speculative stats, continuous batching), but the REPL currently drives mostly CLI-side chat history rather than engine persistent sessions. Session fork and real undo/rewind need new engine APIs; many other high-value REPL surfaces are CLI wiring over existing APIs.

## 2026-07-27T16:44:54Z — Wave 9 update
Fixed PR #276 after Ferro rejection: build break plus driver-enforced WAR fence/neutering proof; re-review approved and merged as 9ab24fa5.

## 2026-07-27T14:15:00-07:00 — Runtime fork/rewind API

- Added public engine APIs for persistent-session checkpoints and KV rewind (`checkpoint_session`, `restore_session`, `rewind_session_by`, `rewind_session_to`) using the existing target/draft speculative rewind helpers.
- Reserved `fork_session` behind an unconstructible public `SessionForkCapability`; current backends return no capability, and the internal path still fail-closes until decoder runner state can be safely cloned/imported without deep-copying or aliasing KV.
- Documented the API/cost model/invariants/backend matrix in `docs/research/cli/06-fork-rewind-api.md` and added model-free engine unit coverage plus `proptest` randomized fork/rewind/append/remove refcount coverage for paged KV.

## 2026-07-27T14:56:47-07:00 — PR #287 backend flag lockout revision

- Fixed Batty's rejected CLI backend flag revision: profile output, REPL /session, bare /backend, and /stats now use the loaded engine's resolved backend; auto is only shown as a requested backend when it differs.
- Kept --backend on transcribe deliberately because speech transcription drives the same autoregressive pipeline decoder; added parser and invalid-backend coverage plus README documentation.
- Verified cargo build -p onnx-genai-cli, cargo test -p onnx-genai-cli --lib, cargo fmt -p onnx-genai-cli -- --check, cargo clippy -p onnx-genai-cli --all-targets -- -D warnings, and cargo build -p onnx-genai-server.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Fixed PR #288 tests after Moss lockout: LogSoftmax overflow-stability is value-falsifiable; BitShift width guard is locked by source-contract test.

## 2026-07-27T17:25:00-07:00 — ORT runtime library selection

- Implemented explicit runtime ORT selection for native binaries: `ONNX_GENAI_ORT_LIB`, `ONNX_GENAI_ORT_LIB_DIR`, active conda/venv probing, target-cache fallback, and pathful API-mismatch diagnostics. The approach is valid because `ort-sys` now resolves `OrtGetApiBase` with `libloading` instead of import-library linking the final binary.
- Found Justin's ORT 1.27.0 at `C:\Users\justinchu\AppData\Local\anaconda3\Lib\site-packages\onnxruntime\capi\onnxruntime.dll`; `onnxwin` exists but has no `onnxruntime` package installed.
- Corrected README after Justin's feedback: document machine-independent build/runtime resolution order and inspection via `onnx-genai version`; keep host-specific conda paths as validation evidence, not as the documented answer.

## 2026-07-27T02:00:00Z — Roadmap wave update
- Fixed PR #301 / #85 after author lockout: executor liveness now treats If/Loop/Scan free-variable captures as use sites; merged after Roy approval.

## 2026-07-27T20:15:00Z — Kernel pre-binding (Stage 3)
- Implemented per-plan-node kernel pre-binding to eliminate the 2.15 µs/op dispatch tax (Vec<Vec<usize>> allocation per op per token).
- Added `kernel_bindings: Vec<Option<KernelKey>>` on Executor, `get_prebound` zero-alloc fast path on KernelCache.
- Static-shape graphs pre-populate bindings at build; symbolic graphs populate on first dispatch.
- Shape changes (prefill→decode) detected via `matches_shapes` (slice comparison, no alloc) and fall through to `get_or_create`.
- Reachability: PREBIND_FAST_PATH_TEST_HITS + PREBIND_FALLBACK_TEST_HITS counters with paired tests.
- All session tests pass (211+), both clippy targets clean, format clean, dispatch/platform lints pass.
- Decision: `.squad/decisions/inbox/deckard-kernel-prebinding.md`.

- 2026-07-28: 1x1 Conv routing PR #347 merged after replacing a magic threshold with spatial-size-dependent evidence and measuring EfficientNet-B0 (-8.9%). Fitted constants are acceptable only when labelled as fitted and bracketed by measured data; a false rationale is worse than no rationale.

Full pre-compaction history in `history-archive.md`.