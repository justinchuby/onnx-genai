# Resch — History (compacted 2026-07-29)

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

### 2026-07-27T10:40:00-07:00 — Platform-naming lint + x86 GEMM renames (PR #278)
Renamed `simd_gemm.rs` → `x86_sgemm.rs` and `bf16_gemm.rs` → `x86_bf16.rs` with zero behavior change. Added `scripts/check_platform_naming.py`, a CI lint for single-arch cfg files lacking platform markers; it uses a portable-item check to avoid false positives, but cannot catch within-file missing implementations like the sdpa `dot_f32` case. Guard-break restored old names and failed; 945 tests passed, clippy green on aarch64/x86_64.

### 2026-07-27 — Cross-Target Compilation Check (PR #319)
Added `scripts/check_cross_compile.sh` for Roy's structural fix plan. It targets `x86_64-unknown-linux-gnu`, uses `--all-targets`, runs full offline crates on Ubuntu and an FFI-free subset on macOS, and teaches why `x86_64-apple-darwin` is insufficient. Guard-break synthetic `is_undilated` in onnx-runtime-ir failed under the new script. Known gaps: ep-cpu from macOS, runtime dispatch, Windows cfg via portable matrix.

### 2026-07-27T22:05:00-07:00 — Dispatch manifest lint
Created `dispatch_manifest.toml` plus `scripts/check_dispatch_manifest.py`: declarative (op, variant, platform) → minimum tier + proving counter, CI-only and cross-EP ready. Seeded MatMul/Conv/SDPA/KernelDispatch claims, documented depthwise Conv and bf16 M≥2 exclusions, and proved guard-break by renaming `CONV_BNNS_TEST_HITS`. It would have caught 7/9 historical instances; misses compilation errors and unclaimed ops.

### 2026-07-27T23:13:00-07:00 — Manifest backfill + inverse check
Backfilled PR #324 claims (MaxPool→BNNS, Add→vDSP, MatMul f16 colmaj→NEON GEMV), added dilated MaxPool and BatchNorm fusion exclusions, and implemented inverse checking so any optimization counter without a manifest row fails CI. Fixed the AtomicU64 blind spot in both reachability and manifest lints. BatchNorm was documented as graph-level fusion elimination, not a dispatch-tier claim. Guard-breaks failed as intended; 973 CPU EP tests, 4 lints, and fmt were green.

Full pre-compaction history in `history-archive.md`.
