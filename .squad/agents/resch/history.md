# Resch — History (compacted 2026-08-12)

**Role:** Intel CPU Optimization Engineer for x86-64 CPU-EP performance, AVX2/AVX-512/VNNI, MLAS interplay, and int8 DP4A GEMV. Optimizations must be portable beyond one machine, benchmark-backed, and numerically matched to scalar/f64 references with regression tests.

## Durable lessons
- SIMD/NPU paths must be opt-in early returns over an always-present scalar baseline; avoid architecture forks and keep every target compilable.
- Runtime feature detection/dispatch is expected for SIMD paths; tune for consumer/edge hardware, not only flagship machines.
- Untested SIMD paths are as risky as placeholders: every AVX/NEON/SVE/QNN path needs guard-break tests and paired scalar/reference checks.
- Apple Silicon FP16 NEON can widen f16 loads directly while ORT CPU widens before GEMM; do not assume the same architecture distinction on other platforms.
- Cross-platform compile failures can hide in cfg-gated code that only builds on one OS/arch; `x86_64-apple-darwin` alone is insufficient because it changes arch but not enough OS surface.
- Kernel layout should group by role in dispatch, not pure platform, to avoid recreating the banned architecture fork.
- Platform-naming lint catches files with single-arch cfg and no platform marker, but does not catch within-file missing implementations.
- Cross-compile check catches cfg-gating errors in `--all-targets`; known gaps: ep-cpu from macOS, runtime dispatch, and Windows cfgs outside portable matrix.
- Dispatch manifest is CI-only and cross-EP ready; every optimization counter should have a manifest row unless deliberately excluded.
- Inverse manifest check closes the "human must remember to add a row" gap; AtomicUsize/AtomicU64 and `pub static` counters must be recognized.
- BatchNorm fusion elimination does not fit dispatch-tier claim schema; it is an optimizer/opset-registration failure mode needing optimizer-level counters.

## Recent work (current wave, ~2026-07-28/29)

## Historical context (2026-07-27 through 2026-08-11 early)

Jul-27 wave: platform-naming lint + x86 GEMM renames (PR #278); cross-target compilation check (PR #319); dispatch manifest lint; manifest backfill + inverse check. Aug-11 early wave: AVX2 LayerNorm kernel pilot + hardening (PR #31973); BFloat16 CPU LayerNorm/RMSNorm registration; RMSNorm mean-null skip (Gaff S1). Full entries in `history-archive.md`.

## 2026-08-11 — Direct EP assignment assertions via Session_GetEpGraphAssignmentInfo

- **Deleted false claim** that ORT 1.27 lacked per-node provider attribution. The API
  (`Session_GetEpGraphAssignmentInfo`) has existed since ORT 1.24 — confirmed in bindings.
- Added `query_ep_assignment` helper and `assert_ops_assigned_to_our_ep` to plugin_ort_e2e.rs.
- Enabled `session.record_ep_graph_assignment_info=1` in `conformance_setup`.
- 8 conformance tests now directly assert node→EP assignment (Add, Mul, MatMul, Cast, Where).
- `conformance_mixed_partition`: asserts NonZero is never on our EP (negative invariant).
- `conformance_shape_f32`: soft-check (Shape may be constant-folded by ORT).
- **Non-vacuity proved:** Forcing assertion on "Relu" (not our op) panics as expected.
- **Task 2:** Replaced `unwrap_or(0)` with named `DIM_UNKNOWN` constant + loud invariant
  documenting that kernels must not pre-allocate from compile-time shapes.
- 269 passed, 0 failed across all 5 EP crates.

## 2026-08-11 — PR #762: Session_GetEpGraphAssignmentInfo wiring

**Task:** Wire `Session_GetEpGraphAssignmentInfo` (present since ORT 1.24; corrected by Fact Checker).

**Commit:** `e0ef1f0a8`

- Enable `session.record_ep_graph_assignment_info=1` in `conformance_setup`.
- Use `Session_GetEpGraphAssignmentInfo` → `EpAssignedSubgraph_GetEpName` → `EpAssignedNode_GetOperatorType` to query per-node EP assignment.
- 8 conformance tests now assert specific ops assigned to `"cpu_ep"` (distinct from ORT's `"CPUExecutionProvider"` by exact string).
- `DIM_UNKNOWN = 0` constant documented with invariant comment.
- Non-vacuity: forced `"Relu"` assertion produces expected failure message.

**Outcome:** Pris's sixth review found BL1 regression test still lacked fallback guard. Rachael hardened.

## 2026-08-12 — PR #32001: Strict CLI validation for --use_apple_accelerate

**Commit:** ec73d63a0e on `nxrt/mlas-apple-framework-option`

Fixed the blocker where `build.py --use_apple_accelerate` only checked `is_macOS()`, allowing Intel Macs, x86_64 cross-compiles, iOS/tvOS/visionOS, and Mac Catalyst to silently pass through to CMake warn-and-disable — contradicting the PR body's "fails loudly" promise.

Added validation in `build_args.py` (parser.error) and `build.py` (BuildError) rejecting all non-arm64 arches, iOS, tvOS, visionOS, and Catalyst. All Apple-target attrs accessed via `getattr()` to avoid `AttributeError` on non-macOS. 9 new test cases, all passing. macOS arm64 CI deferred to first kernel PR.

## 2026-08-12 — PR #32001: Strict CLI validation consolidated (Zhora S1 resolution)

- Holden's focused review found `build.py:881-897` BuildError block was dead code (never reached since `build_args.py` parser.error() exits first).
- Zhora removed the dead copy; `build.py` now only appends the cmake arg with a comment pointing to `build_args.py` for validation.
- Improved non-macOS error message: now reads "requires Apple Silicon host" to correctly distinguish wrong-OS from wrong-arch.
- 13/13 tests pass unchanged. ruff clean. Head `3a0bd75aa3`. PR #32001 **ready for review**.
