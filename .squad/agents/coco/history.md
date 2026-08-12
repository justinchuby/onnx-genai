# Coco — History

## 2026-07-12: Joined
Hired as a Metal/MPS kernel engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns data/quantization/elementwise kernels: GatherBlockQuantized (int4 embedding), quantize/dequantize (int4/int8/fp8), KV ops, elementwise/activations, reshape/transpose/cast. Reference ExecuTorch + PyTorch MPS. Op set must match exactly what onnx-genai/Mobius models emit (MatMulNBits, GQA, GatherBlockQuantized, RoPE, RMSNorm). Tested via onnx-genai runtime.

- 2026-07-14T19:05:00Z — Tracer AutoDiagnosis and roofline module merged in `8607687`; Hodge review GREEN. Follow-up decision requires first-class missed-fast-path diagnosis from executor selection metadata.

- 2026-07-15 — Bundled oneDNN in Linux and macOS Python wheels (merged `ef89a95`).

### 2026-07-16T00:00:00Z — Performance-and-design wave
Applied CUDA coverage documentation correction for the merged kernel slice.

## 2026-07-16T00:00:00Z — onnx-rs upstream text-format port review
- Cleared merged commit `23e4995` 🟢: 10 added tests make 16 upstream-derived text-format cases and assert real parser/IR/codec behavior.
- Confirmed 89 passing onnx-rs tests with no ignored or vacuous cases; documented unsupported functions, non-tensor type forms, complex/2-bit dtypes, and literal tensor/sparse payload syntax.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Coco's initial WP-B3 admission work was superseded by raw-protobuf authority fixes that completed the epic.

## 2026-07-28T09-10-28+00-00 — Shape-inference catalog batch 2 merged
- PR #339 (`b1f9d3bb`) added Det, LpPool, GlobalLpPool, MaxUnpool, Col2Im, and CenterCropPad; registry 181→187 operators and 219→226 versioned entries. Chew approved after specification review and mutation probing. #75 remains open; signal/loss operators and SSA container types are deferred.

## 2026-08-11 — B1/B2 blocker fixes (PR #762)

- **B1:** Replaced in-band `__absent_output_*` string sentinel with `HashSet<ValueId>` on `OutboundGraphReader`. Unforgeable: ValueIds are arena indices not derivable from model content.
- **B2:** Replaced `filter_map(|d| d.as_static())` with `map(|d| d.as_static())` at both call sites (ep.rs:238, ep.rs:493). `ShapeInference::for_node` now takes `&[Vec<Option<usize>>]`. `build_conv` fails closed on unknown dims.
- **mixed_partition:** Added compiled-node counter + C symbol. Soft diagnostic (ORT 1.27 lacks per-node provider attribution API).
- Tests: 269 passed (+3 new: `forgeable_name_not_treated_as_absent`, `symbolic_dims_preserve_rank`, `conv_declines_with_symbolic_spatial_dims`), 0 failed.

## 2026-08-11 — PR #762: B1/B2 correction (Challenger's blockers)

**Task:** Fix Challenger's B1 (forgeable sentinel) and B2 (filter_map rank destruction).

**Commit:** `38625fb38`

- **B1:** `OutboundGraphReader` now maintains `absent_outputs: HashSet<ValueId>`. ValueIds are graph-internal arena indices — model content can influence names and shapes but never the arena index a value receives. String prefix `__absent_output_*` removed entirely.
- **B2:** `filter_map(|d| d.as_static())` → `map(|d| d.as_static())` → `Vec<Option<usize>>` preserving rank at claim time. `get_kernel` trait receives `unwrap_or(0)` with `DIM_UNKNOWN` constant. `build_conv` fails closed (`return None` → `Declined`) on `None` dims.
- `conformance_mixed_partition` assertion: added compiled-node counter via `dlsym` as best effort under ORT 1.27 API constraints.

## 2026-08-12 — PR #32001 scope correction (macOS arm64 gate)

**Commit:** `52db6351b5`

- Added `onnxruntime_target_platform STREQUAL "arm64"` gate in `cmake/CMakeLists.txt:616` and updated comments in `cmake/onnxruntime_mlas.cmake:1172`.
- Non-arm64 Apple → warn-and-disable (not FATAL_ERROR), matching SVE/KleidiAI idiom.
- Removed universal2/iOS claims from MLAS cmake comments and `build_args.py` help text.
- Validated: default Linux configure has zero Accelerate references; option ON on Linux warns and disables; `build.py --help` shows updated text.
- Only macOS arm64 CI can confirm: `find_library(Accelerate)` succeeds and `target_link_libraries` actually links.

---

### 2026-08-12T02:30:00Z — PR #32001 lockout revision: rescoped to macOS arm64 only

Added `onnxruntime_target_platform STREQUAL "arm64"` condition to `cmake/CMakeLists.txt` as `elseif` after the `if(NOT APPLE)` check, using warn-and-disable (matching SVE/KleidiAI idiom). Rescoped option description, MLAS comment, and `build_args.py` help text. Verified no-behaviour-change-when-disabled on Linux x86-64. Head: `52db6351b5`. PR remains draft.

---

### 2026-08-12T01:35:00Z — PR #31993 revision: NaN assertions, RNE tie, macOS gating, -march removal

Fixed four issues in the MLAS f16↔f32 cast PR:
1. **NaN assertions**: replaced bit-exact `ASSERT_EQ` with sign+payload-modulo-quiet-bit check, tolerating NEON sNaN quieting while still pinning sign and payload.
2. **RNE tie**: replaced 1.0009765625f (not a tie) with 1.00048828125f (genuine halfway value between fp16 0x3C00 and 0x3C01).
3. **macOS-only gating**: `TARGET_OS_OSX` in C++, `CMAKE_SYSTEM_NAME STREQUAL "Darwin"` in CMake — excludes iOS/tvOS/visionOS.
4. **Dropped `-march=armv8.2-a+fp16`**: vcvt_f32_f16/vcvt_f16_f32 are baseline ARMv8-A conversions (guarded by `__ARM_FP & 2` in arm_neon.h), no +fp16 needed.

Evidence for -march removal: clang's `arm_neon.h` guards vcvt_f32_f16/vcvt_f16_f32 under `#if (__ARM_FP & 2)`, not `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`. Could not empirically cross-compile (no AArch64 sysroot on host), but the header evidence is conclusive.

PR body updated. Pushed to `fork`, head `02a9f34`. PR remains draft — awaiting Opus review.

## 2026-08-12 — PR #31993 NaN fix (hardware sNaN quieting)

Fixed NaN blocker on Apple Silicon f16 cast test. NEON `FCVTL`/`FCVTN` quiet signalling NaNs; MLAS software reference does not → bit-exact comparison fails. New assertion: `isnan` + sign match + payload equality modulo quiet bit (bit 22 f32, bit 9 f16). Non-NaN: raw-bit equality preserved. Also: RNE tie input corrected to 1 + 2⁻¹¹ (genuine tie). Removed `-march=armv8.2-a+fp16` — conversion intrinsics are AArch64 baseline (`__ARM_FP & 2`), not guarded by `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`. macOS gate via `TARGET_OS_OSX`. Head `02a9f34`.

## 2026-08-12 — PR #762: Three test-integrity gaps closed

**Commit:** `2106ac0f7`

- **ITEM 1 (validate_write_dtype):** Wired into two new tests exercising absent and present TensorMut validation. Cannot enforce at runtime without restructuring the raw-pointer Kernel trait; tests prove the mechanism and document the contract.
- **ITEM 2 (scratch_alloc_bytes):** Extracted `pub fn scratch_alloc_bytes(numel, dtype)` as single source of truth. Both production sites (single-node, multi-node) and all canary tests call it directly. Former test copy (`production_scratch_alloc`) deleted.
- **ITEM 3 (routing None → Compile failure):** `build_subgraph_routing` returning `None` now fails at Compile with explicit message. Dual-role slots (graph output consumed by downstream node) cannot be expressed in current `NodeOutputSink` enum; fail closed at Compile is the correct behavior.

**Validation:** 222 passed, 0 failed; clippy clean; fmt clean; Miri clean on lib tests (173 passed). Pre-existing Miri UB in `trait_cabi_parity.rs` (not my code) unaffected.
