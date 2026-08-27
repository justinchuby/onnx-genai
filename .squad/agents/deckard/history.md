# Deckard — History (compacted 2026-08-27T17:00:00Z)

**Role:** Systems developer on the Rust runtime, ORT2 loader/shape/IR/EPContext tracks, and CPU/CUDA execution performance. Preserve model-agnostic dispatch, fail-closed claims, checked arithmetic, byte-exact serialization, precision-sensitive tests, and reviewer-lockout ownership transfers.

## Durable lessons
- Model-agnostic dispatch, fail-closed claim time, checked arithmetic, byte-exact serialization, and precision-sensitive tests are recurring invariants.
- Parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership.
- `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive `OrtEpDevice`; use `CreateMemoryInfo_V2` and release only on failure.
- Unmodelled shape inference returns `Declined`, never an assumed `SameAsInput(0)`.
- `validate_dims` must be wired in the actual read path, not merely implemented.
- `OrtGraph*`/`OrtNode*` handles do not outlive their callback.
- Profile each caller of shared CUDA helpers before applying cache or occupancy hints uniformly.
- CUDA ordering proofs preserve the production operation and never allow panic/unwind through teardown or FFI-sensitive paths.

## Historical context

Shape inference, EP lifetime, plugin export, CUDA capture/speculation/Marlin, DeepSeek-V2 capture, qwen3.5 attribution, QMoE optimization, and no-go chronicles are archived. Full older history is in `history-archive.md`; the exact hot file before this compaction is in `history-archive-2026-08-27T17-00-00Z.md`.

## 2026-08-20T05:50:19+00:00 — Phase-4 Gated-DeltaNet L2-normalize glue fusion merged

Scribe recorded Deckard's #1562 after merge to `origin/main`: `CudaL2NormalizeFusion` collapses Q/K L2-normalize chains in Qwen3.8 Gated-DeltaNet from ReduceSumSquare→Sqrt→Div into a byte-faithful fused route, reducing roughly **288→96 launches/token**. Sebastian's integrated validation measured the stacked #1561+#1562 result at q38 **61.32 tok/s** (+12.4% over the #1557 base) and mary **60.59 tok/s** (+3.0%), with mary byte-identical. Standing lesson: SSM glue fusion is useful but secondary; q38 is still forward int4 M=1 GEMV latency/occupancy-bound.

## 2026-08-20T13:46Z — #1569 merged; next GDN megakernel lever active

Scribe recorded PR #1569 after Sebastian's independent re-validation and merge to `origin/main` (`b693f2bb2`): q38 improved **61.27 → 62.76 tok/s (+2.43%)** under the relaxed dtype-tolerance bar; mary control stayed byte-identical and q38 clear prompts were coherent/byte-identical. The unsupported q38 determinism claim should be dropped because split-K GEMV nondeterminism persists. Deckard's next assigned lever is the GDN recurrence megakernel: fold β-sigmoid + softplus/dt_bias + conv1d/state into the fused recurrence.

## 2026-08-26 — #1896 rejected initial revision

Deckard's initial #1896 revision was rejected for classifying the problem away from the production defect, using a non-equivalent mutation, and unsafe unwind behavior. Durable lesson: CUDA ordering proofs must preserve the production operation and must never permit panic/unwind across teardown or FFI-sensitive paths.
