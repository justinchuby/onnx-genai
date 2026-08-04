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

## 2026-08-02T10:05:00+0000 — #594 lockout revisions

- Took over #594 after Harry's reviewer lockout, ran `cargo fmt --all`, and produced a formatting-only fix for the new E2E test code.
- After rebasing #594 onto main, fixed pinned shape-inference registry counts to account for standard-domain LinearAttention (operator_count 217→218, entry_count 262→263).

## 2026-08-02T11:40:00+0000 — #595 profile_native bench fix

- Authored and merged #595, restoring `reset_exec_phase_profile` so `profile_native --steady` bench binaries compile on main.
- Fix stayed scoped to reset plumbing/re-export/test coverage and preserved the disabled hot path.

## 2026-08-02T19:00:00+0000 — PR #602 lockout revision

- Took over #602 after Harry's rejection under author-lockout and pushed `788dc609`.
- Added conservative `function_has_attribute_parameters` fail-closed behavior for formal attrs, body `ref_attr_name`, and call-site attrs; preserved `ModelFunction.attributes`/`has_attribute_refs` in the loader; added mutation-proven `ParamLeakyRelu` regression coverage.

## 2026-08-02T19:50:00+0000 — PR #604 phase-profile flake fix

- Authored and merged #604, a test-only fix for the pre-existing #595 `phase_profile_gating_and_accumulation` parallel-runner flake seen after #602.
- Replaced racing global `all_stats().is_empty()` reset assertions with phase-scoped `snapshot(enabled_phase).is_some()` / `.is_none()` checks on the test's unique phase name; validated 30/30 full-parallel lib runs plus fmt/clippy clean.

## 2026-08-03T03:10:00+0000 — mobius PR #449 PackedMHA bias slot

- Authored mobius PR #449, adding `bias` as the 4th formal to the `PackedMultiHeadAttention` fallback while keeping it inert in the body.
- Added positional/admissibility tests covering full input order, bias at index 3, and 6-input calls; ruff clean, 3/3 new tests passed, and 30/30 `ep_optimization_test` regression passed. Awaiting Justin merge.
