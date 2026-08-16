# Iran — History Archive (2026-08-12T10-15-00Z compaction)

Archived from live `history.md` during final-review-wave Scribe run. Contains dated entries 2026-07-27 through 2026-08-11 (upstream CI correction wave).

## 2026-07-27: Conv Three-Tier Dispatch (#317)
- Diagnosed 643× ResNet-18 gap: `conv_ref.rs` scalar loop was only path on macOS due to `mlas` feature gate being x86-64-Linux-only.
- Assessed BNNSGraph: requires `.mlmodelc`, cannot do per-op dispatch. No migration target exists.
- Implemented Tier 1 (BNNS Filter Conv, AMX, 877–1458 GFLOPS), Tier 2 (im2col + cblas_sgemm, ~300 GFLOPS), Tier 3 (scalar ref).
- Result: ResNet-18 8792ms → 93ms (94× faster), now 0.15× ORT (from 0.0016×). Whisper-tiny unchanged (MatMul-bound). Decode unregressed.
- Remaining ResNet-18 gap (6.7×) is non-Conv ops (BatchNorm, Pool, Add on scalar paths).
- BNNS Filter API deprecated but no replacement for per-op use. `cblas_sgemm` is durable fallback.
- 2026-07-28: Small-shape GEMV investigation produced a valid negative result: existing inline paths and cblas already cover the remaining cases. SDPA decode PR #349 merged after attribution and after correcting the headline from 1.9x to 1.37x by naming the model (TinyStories-1M vs -33M). Always state which model each ratio refers to.

## 2026-08-11: AVX2 LayerNorm kernel revision (PR #31973 blockers B1/B2/N2/N3/N4)

- **B1 reproduced:** AVX2 lane-parallel Welford hits 28.2% rel err at base=1e5/spread=1e-2/N=512 vs fp64 oracle (scalar Welford: 0.47%). Root cause: per-lane fp32 mean accumulation rounds before merge.
- **B2 evaluated:** Centered two-pass with double-precision first-pass sum is 1.8× faster than AVX2 Welford (427ns vs 751ns, N=1024) AND more accurate (worst 5.95e-3 vs 2.82e-1). fp32 sum is catastrophic at base=1e7 (100% error); double sum is essential.
- **Algorithm replaced:** Welford → centered two-pass with `_mm256_cvtps_pd`+`_mm256_add_pd` for mean, fp32 for centered variance.
- **N2 fixed:** NormSize<8 gate moved to `#if x86` so RVV is unaffected.
- **N3 verified:** RMSNorm mean-skip for null MeanOut was already correct; upgraded RMSNorm MeanOut path to double-sum too.
- **N4 fixed:** Added explicit `set_source_files_properties(/arch:AVX2)` for layernorm_kernel_avx2.cpp on MSVC.
- **Tests:** 39/41 pass. 2 Pris-owned precision tests fail because they check parity vs scalar Welford (wrong reference now). All 32 functional tests pass.
- Lesson: centered two-pass ≠ uncentered two-pass. The prior team rejection of "two-pass" conflated E[x²]-mean² (catastrophic) with sum((x-mean)²) (numerically standard). Always specify the formulation.

### 2026-08-11 — B4: CUDA Plugin Fail-Closed (PR #762 Reviewer Rejection)
- **Owned B4:** CUDA cdylib advertised GPU EP it could not honour.
- **Root cause:** Implementation-blocked, NOT hardware-blocked. Four defects: (1) separate CUDA runtime/context per EP/allocator/stream, (2) non-functional data transfer (no OrtApi, no shared stream), (3) NULL stream handle, (4) `device_free` passes `size=0`.
- **Fix:** `CreateEpFactories` returns 0 factories + actionable status in BOTH feature configs. Crate docs specify all 4 defects as a roadmap for future implementation.
- **CanCopy fail-closed:** Both `transfer_can_copy` and `transfer_full_can_copy` now return `false` for device EPs (were returning `true` unconditionally — fail-open).
- **device_free defect #4:** Documented the `size=0` contract violation with fix specification (allocation size tracking side-table).
- **Validation:** `cargo check --workspace` ✓, `cargo check -p onnx-runtime-ep-cuda-plugin --features cuda` ✓, ep-plugin 154+9 tests ✓, clippy clean ✓, fmt clean ✓.
- **Key correction:** CUDA is implementation-blocked, not hardware-blocked. Even with a GPU, this code cannot work as written.

### 2026-08-11 — ReleaseEpFactory ABI fix (follow-up to B4, reported by Sapper)
- **Bug:** CUDA shim's hand-written `ReleaseEpFactory` returned `void`; correct ABI is `OrtStatus*` per `onnxruntime_ep_c_api.h:2669`.
- **Fix:** Updated return type to `*mut OrtStatus`. On normal release returns null (success); on panic catches unwind and returns actionable status via `panic_to_fail_status`.
- **Macro not used:** `export_ep_factories!` emits both symbols as one expansion; CUDA shim needs custom `CreateEpFactories`, so macro would conflict. Commented in file explaining why it is hand-written and must stay in sync with the macro's `ReleaseEpFactory` arm.
- **CreateEpFactories drift check:** Signature matches `CreateEpApiFactoriesFn` at header line 2654 — no drift.
- **Fail-closed unchanged:** Both configs still return 0 factories + error status.
- **Validation:** `cargo check --workspace` ✓, `cargo check -p onnx-runtime-ep-cuda-plugin --features cuda` ✓, ep-plugin 9 tests ✓, clippy clean ✓, fmt clean ✓.

## 2026-08-11 — B5: Fix stats round-trip through BFloat16 (PR #31974)

**Task:** Reviewer rejection B5 — Mean/InvStdDev outputs were narrowed through BFloat16/MLFloat16 instead of being written as float, losing ~0.4% precision.

**Changes:**
- `layer_norm_impl.cc`: BFloat16 and MLFloat16 `ComputeJob` overloads now use `WriteStat<U>()` (U=float) instead of `BFloat16(mean)` / `MLFloat16(mean)`.
- `layer_norm_impl.h`: `SrcDispatcher` changed from runtime `if` to `if constexpr` to avoid instantiating `ComputeImpl<NarrowType, NarrowType>`, eliminating dead template paths.
- `WriteStat`: Removed dead `MLFloat16`/`BFloat16` branches now that they can never be instantiated.
- `layer_norm_impl.cc`, `skip_layer_norm.cc`: Updated `NarrowToFloat`/`FloatToNarrow` comments to honestly describe scalar BF16 conversions (no hardware bf16 instructions on AVX2).

**N1 decision:** Keep MLFloat16 `U=float` registration (matches schema, matches CUDA, declaration-only change, adjacent to BFloat16 registration).

**Deduplication:** Not done — `NarrowToFloat`/`FloatToNarrow` duplication between files requires a shared header, which is scope creep. Follow-up.

**Validation:** Build clean with `-Werror`. 17/17 BFloat16 tests ✓, 96/96 full LayerNorm suite ✓.

## 2026-08-11 — Deduplicate NarrowToFloat/FloatToNarrow (PR #31974)

Reviewer flagged duplicated conversion helpers. Confirmed `ConvertMLFloat16ToFloatIfNeeded` was pre-existing (left alone). `NarrowToFloat<T>` / `FloatToNarrow<T>` were ours — identical copies in `layer_norm_impl.cc` and `skip_layer_norm.cc`.

No upstream template helper for dispatching across both narrow types existed. Created `onnxruntime/core/util/narrow_float_utils.h` in `namespace onnxruntime`, included from both sites, removed local defs.

Build clean, 17/17 BFloat16 LayerNorm tests pass, 96/96 `*LayerNorm*` pass, clang-format clean. Pushed as `6dd19a6f56`.

## 2026-08-11 — B4 fail-closed, BF16 stats fix, LayerNorm kernel, NarrowToFloat dedup

**B4 fail-closed gate (#762):** CUDA plugin returns zero factories both feature configs. `CanCopy` returns false for device EPs. Four defects documented for future hardware-gated work.

**B5 stat precision (#31974):** `ComputeJob<BFloat16/MLFloat16>` overloads call `WriteStat<U=float>` directly. `if constexpr` prevents `ComputeImpl<T,T>` for narrow types. 17/17 BFloat16 + 96/96 LayerNorm tests pass.

**LayerNorm B1/B2 kernel fix (#31973):** Welford → centered two-pass + double-precision first-pass sum. Worst-case: 5.95e-03 vs 28.2% for old Welford. 14.3× faster than scalar. `NormSize < 8` gate moved inside x86-only guard (was blocking RVV).

**NarrowToFloat dedup (commit `6dd19a6f56`, #31974):** `onnxruntime/core/util/narrow_float_utils.h` created. Helpers removed from both source files.

**Rebase PR #31973 (2026-08-11):** Rebased 7 commits onto upstream/main (`86d38813a8`). Zero conflicts. Build clean (no warning suppression). 42/42 MLAS LayerNorm tests pass (43 with disabled). New HEAD `6ef1f61f88`. All 5 preserved properties confirmed intact. ContribOperators.md CI failure expected (not ours, covered by #31985).

## 2026-08-11 (upstream CI correction wave) — Rebase PR #31973

Rebased `nxrt/mlas-avx2-layernorm` (7 commits) from `16b486a2` onto `upstream/main` at `86d38813a8`. Zero conflicts. Build clean (warnings-as-errors). 42 MLAS LayerNorm tests pass. All five preserved properties intact. Force-pushed. New HEAD: `6ef1f61f88`. PR remains draft per user instruction (correct posture: draft until CI board is green).
