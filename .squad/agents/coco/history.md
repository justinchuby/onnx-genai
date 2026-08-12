# Coco — History (compacted 2026-08-12)

**Role:** Systems engineer — compute kernels, test integrity, data flow correctness. Owns scratch sizing, routing, EP partitioning, and kernel contract validation.

## Durable lessons
- `scratch_alloc_bytes(numel, dtype)` is the single source of truth for scratch buffer sizing. Both production sites and all canary tests must call it directly. Duplicate copies cause formula drift → heap overflow.
- `validate_write_dtype` is a test-only contract helper. It is not a runtime guard — the kernel writes via raw pointers through `Kernel::execute`. Document as such; don't imply runtime enforcement.
- Unroutable graphs (dual-role slots, `NodeOutputSink` cannot represent graph output + downstream consumer) must fail at **Compile** via `fail_status`. Deferring to Run loses the ORT fallback opportunity.
- `OutboundGraphReader` uses `absent_outputs: HashSet<ValueId>` (arena indices) not string sentinels — arena indices are unforgeable from model content.
- `ShapeInference::for_node` takes `&[Vec<Option<usize>>]` preserving rank at claim time; `filter_map(|d| d.as_static())` silently destroys rank and is forbidden.
- `TARGET_OS_OSX` / `CMAKE_SYSTEM_NAME STREQUAL "Darwin"` for macOS-only gating; `TARGET_OS_OSX` excludes iOS/tvOS/visionOS.
- NEON `FCVTL`/`FCVTN` quiet signalling NaNs; MLAS software reference does not. NaN comparison: `isnan` + sign + payload modulo quiet bit.
- `-march=armv8.2-a+fp16` not needed for vcvt_f32_f16/vcvt_f16_f32; guarded by `__ARM_FP & 2` in arm_neon.h — AArch64 baseline.
- `onnxruntime_target_platform STREQUAL "arm64"` is the canonical Apple arch variable (cmake/CMakeLists.txt:567/575/589).

## Historical context (pre-2026-08-12)
Wave coverage through 2026-08-11: Metal/MPS kernel engineer role, tracer AutoDiagnosis module, oneDNN wheels, onnx-rs text-format review, shape-inference catalog PR #339, B1/B2 blocker fixes (forgeable sentinel, filter_map rank destruction), PR #32001 macOS arm64 scope, PR #31993 NaN fix. Full detail in `history-archive.md`.

## Recent entries

## 2026-08-12 — PR #762: Three test-integrity gaps closed

**Commit:** `2106ac0f7`

- **validate_write_dtype:** Wired into two new tests; documented as test-only contract helper.
- **scratch_alloc_bytes:** Extracted as `pub fn` at `compute.rs:601`; both production sites + all canaries call it directly; old test copy deleted.
- **routing None → Compile failure:** `build_subgraph_routing` returning `None` now fails at Compile with explicit message.

Validation: 222 passed, 0 failed; clippy clean; fmt clean; Miri clean (173 lib tests).

## 2026-08-12 — PR #762 ready for review (scratch/routing/dtype gaps closed)

Three test-integrity gaps closed. 283 passed / 0 failed; Miri clean. Gaff confirmed scratch_alloc_bytes and routing genuinely closed. PR #762 marked ready.

*Full pre-2026-08-12 history in `history-archive.md`.*
