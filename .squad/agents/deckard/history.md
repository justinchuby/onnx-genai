# Deckard — History (compacted 2026-07-29)

## Condensed history through 2026-07-21 (summarized 2026-07-30 by Scribe)

- Systems developer on the Rust runtime, ORT2, CPU EP, CUDA, shape-inference, loader, IR dtype, EPContext, encoder, and external-data safety tracks.
- Durable review habits: preserve model-agnostic dispatch, fail closed at claim time, use checked arithmetic, keep byte-exact serialization, and require precision-sensitive tests.
- Repaired or owned lockout follow-ups across shape inference, IR dtype, EPContext writer, CPU reductions/activations, Hardmax/bitwise, SpaceToDepth/pooling, EyeLike, GridSample, LogSoftmax, executor liveness, and WAR-safe CUDA driver semantics.
- CUDA milestones through 2026-07-21 included CSA B2/B5 work, SkipSimplifiedLayerNorm RMS drift root cause, cudarc CUDA-version unification, structured CUDA-decline/required-CUDA reporting, graph handle/scratch/replay hardening, and native fp16 decode wave-2 parity guidance.
- DeepSeek/MLA/MoE and CPU EP milestones included tiny DeepSeek-V2 E2E validation, ConvTranspose coverage, QMoE grouped top-k recovery, vendored MLAS CPU-GEMM parity, PackedB reuse, SQNBitGemm wiring, and M-based MLAS int4 routing.
- Performance lessons: coarse CPU decode fork-join and replay binding metadata caching were measured regressions/dead ends; GQA was the larger post-residency cost at that time. Parallel commit-producing work requires separate worktrees, and reviewer rejection transfers ownership.

## 2026-07-22T12:00:00Z — Luv Phase 0 review
- Independently reviewed Luv's partial-CUDA-graph capture path-kind change at `3c94a57`; approved 🟢 GREEN after confirming additive behavior, correct structural seam mapping, model-agnostic dispatch, exhaustive matches, and clean fmt/clippy/tests.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Deckard's intermediate WP-B3 revision fixed raw membership/default classification but was superseded by Sapper's v3 raw signature fix.
- 2026-07-24T16:04:31Z: Freshly reviewed Keaton's GLM-4-9B native decode-coherence lock: two deterministic GPU-4 golden runs, perturbed-output rejection, native-only comparison semantics, and documentation all verified. 🟢 APPROVE; landed as `13af95d7`.

## 2026-07-26T19:45:52Z — Scribe update

- Spawned as deckard-22 on `perf/cuda-next-wave` for profile-first portable CUDA decode-performance work; outcome pending.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:02Z — Delivered portable device-gated symmetric int4 GEMV split-K; after Sebastian fixed the ineffective coverage shape, PR #203 merged to main at `b80a8c83`.

## 2026-07-27T03:45:00-07:00 — PR #227 FP16 GEMV review follow-ups

- Took Chew's approved non-blocking C1/C2 follow-ups from Iran's FP16 CPU EP work: documented the `fcvtl`/`fcvtn` inline-asm rationale and future intrinsic replacement path, tightened FP16 GEMV tolerances to 1e-4 relative / 1e-5 absolute, and exercised model-scale parity at 1/3/7/11 workers.
- Guard proof: perturbing the FP16 batched accumulator by +0.001 made `f16_col_parallel_gemv_matches_reference` fail at max error 0.0010000467 against the new 0.00001 limit; restored code passed the repeated targeted tests and full CPU EP suite.

## 2026-07-27T07:35:00-07:00 — PR #227 review fixes (cache test + SiLU doc)

- Fixed tautological cache-reuse assertion in `matmul.rs`: moved pointer capture before the second `execute()` so the comparison actually spans the call. Guard-break proof: fresh-kernel substitution made the assertion fail with distinct pointers (0x1030b68d0 ≠ 0x1030b6860).
- Fixed conflicting SiLU accuracy claim in `activations.rs`: slice-level doc now reads "~28 ULP worst-case" matching the implementation comment. Grep-confirmed no other "1 ULP" exp-accuracy claim survives in the crate.
- Full verification: fmt ✅, clippy aarch64 ✅, clippy x86_64 ✅, 906 tests passed ✅, NEON SDPA dispatch confirmed ✅.
## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Under Moss's lockout, repaired PR #266 ReduceLogSumExp numerical stability with a dedicated two-pass reduction; Ferro approved and the PR merged.

## 2026-07-27T16:44:54Z — Wave 8 update
- Took ownership of PR #276 / #87 fix cycle after Ferro REQUEST-CHANGES; must address GPU test build break and WAR-safe driver semantics/docs.
## 2026-07-27T12:30:00-07:00 — PR #279 CUDA fallback semantics

- Adjudicated Copilot's CUDA documentation findings against source. `ONNX_GENAI_EP=cuda` without `onnx-genai-ort/cuda` is rejected at ORT session creation, runtime CUDA-provider unavailability is also a hard session error, and `ONNX_GENAI_REQUIRE_CUDA=1` only gates native CUDA node-level fallback to CPU after CUDA is compiled and selected. Corrected CLI/cargo/CAPI/Python docs and comments; verified `cargo tree`, `cargo fmt --check`, and `cargo build -p onnx-genai-cli`.
### 2026-07-27 — Runtime capability inventory for REPL design

Authored `docs/research/cli/04-runtime-capability-inventory.md` as Deckard. Key finding: the runtime has strong low-level primitives (paged KV CoW fork, rewind/checkpoint, prefix reuse, speculative stats, continuous batching), but the REPL currently drives mostly CLI-side chat history rather than engine persistent sessions. Session fork and real undo/rewind need new engine APIs; many other high-value REPL surfaces are CLI wiring over existing APIs.
## 2026-07-27T16:44:54Z — Wave 8 update
- Took ownership of PR #276 / #87 fix cycle after Ferro REQUEST-CHANGES; must address GPU test build break and WAR-safe driver semantics/docs.

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
## 2026-07-27T20:15:00Z — Kernel pre-binding (Stage 3)
- Implemented per-plan-node kernel pre-binding to eliminate the 2.15 µs/op dispatch tax (Vec<Vec<usize>> allocation per op per token).
- Added `kernel_bindings: Vec<Option<KernelKey>>` on Executor, `get_prebound` zero-alloc fast path on KernelCache.
- Static-shape graphs pre-populate bindings at build; symbolic graphs populate on first dispatch.
- Shape changes (prefill→decode) detected via `matches_shapes` (slice comparison, no alloc) and fall through to `get_or_create`.
- Reachability: PREBIND_FAST_PATH_TEST_HITS + PREBIND_FALLBACK_TEST_HITS counters with paired tests.
- All session tests pass (211+), both clippy targets clean, format clean, dispatch/platform lints pass.
- Decision: `.squad/decisions/inbox/deckard-kernel-prebinding.md`.

- 2026-07-28: 1x1 Conv routing PR #347 merged after replacing a magic threshold with spatial-size-dependent evidence and measuring EfficientNet-B0 (-8.9%). Fitted constants are acceptable only when labelled as fitted and bracketed by measured data; a false rationale is worse than no rationale.

## 2026-07-29T21:00:00-07:00 — PR #398
- Pool prototype (`8fad4915`) won dispatch microbench; real decode kept fixed-SPMD. Admin blocker (`ec062ebb`) was UAC filename heuristic; avoid `patch`/`dispatch`/`setup`/`install`/`update` exe names.
## 2026-07-29T22:00:00-07:00 — MLAS pool backend
- Routed MLAS standalone threading to the persistent WorkStealingThreadPool (`aef9dfd7`); useful infrastructure, but full-width MLAS still regressed versus static-SPMD.

## 2026-07-30T08:20:00-07:00 — ORT-costmodel tuning verdict

- Native static-SPMD CPU EP now matches/slightly beats ORT on Qwen3 best-case and p90 throughput (110.3/109.8 tok/s vs ORT 106.2/106.0), but still trails median on the contended host because variance is higher.
- ORT-style dynamic block claiming helps the isolated full-width QNBit kernel, but loses end-to-end: full-width dynamic 91.72 tok/s best vs static-SPMD 110.32 and ORT 106.16, with stalls from pool park/wake variance across many small ops.
- Full-width path is abandoned as a live toggle. If pushed further, the next candidate lever is vendoring Eigen `NonBlockingThreadPool` for lower wakeup variance, with uncertain payoff.
Full pre-compaction history in `history-archive.md`.