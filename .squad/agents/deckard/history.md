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