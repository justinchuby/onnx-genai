# Mariette — History (compacted 2026-07-29)

**Role:** Metal/MPS kernel engineer for the Apple Metal EP, owning heavy kernels such as MatMulNBits, GQA, softmax, RoPE, and RMSNorm. Correctness against CPU reference comes first, then simdgroup/threadgroup optimization, using ExecuTorch/PyTorch MPS references and onnx-genai end-to-end tests.

## Durable lessons
- Offline per-EP ONNX conformance harness and `docs/execution/EP_CONFORMANCE.md` merged in `1dfab0d`; process-bridge design is recorded in decisions.
- Vendored `cpuinfo` beneath its crate so cargo publish succeeds.
- Mobius native block reviews require exact `BlockQuantizedMatMul` format/dimension/byte-preservation contracts, 4-bit/block-32 mixed-native scaffold, genai opset v1, and unchanged pure-Q8 behavior.
- Attention, CUDA CSA, and MTP reviews needed rejection/fix cycles before approval; keep reviewer-lockout corrections canonical.
- Omitted-optional dtype trap: reject CUDA standard-Attention optional past-KV claim regressions; Nabil's `8eb23f1` fix passed CUDA/session/CPU gates.
- CUDA claim-gate hardening must avoid GLM over-rejection, handle omitted optionals, scope standard-domain checks, and preserve CPU/GLM/CUDA parity.
- Perf campaign inbox decisions were consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.
- Wave-3 SwiGLU fusion halved activation launches from 48 to 24/token, merged as `12e48b8`, and measured about 673→689 tok/s at 256 tokens with zero fallbacks.
- WP-B2 engine runtime is the accepted presence/fallback/gating implementation feeding the completed epic.

## Recent work (current wave, ~2026-07-28/29)
- Latest live item is 2026-07-22: WP-B landed; Mariette's WP-B2 engine runtime remains the accepted implementation.

Full pre-compaction history in `history-archive.md`.

## 2026-08-11 — S1/S2/S3: Optional-Slot Liveness, Axis Bounds, Scratch Buffer

**Commit:** fbd565160
**Verdict:** Our EP WAS declining optional-slot nodes. With fallback disabled, 2/4 tests failed at session creation.

**Fixes applied:**
- Claim filter + dtype filter: carve-out for `__absent_output_*` sentinels
- Single-kernel fast path: inject absent sentinels using `input_slots` mapping
- Clip added to shape inference (SameAsInput(0))
- axis >= rank rejection (was >)
- Scratch buffer sized from primary output dtype

**Result:** 266 passed, 0 failed. All optional_slots tests pass with fallback disabled.

## 2026-08-11 — PR #762: S1–S3 optional-slot liveness proof

**Task:** Confirm and fix S1–S3 from Luv's review.

**Commits:** `fbd565160`, `4757e25b6`

**Findings:** With `disable_cpu_ep_fallback=1`, optional-slot tests failed at `CreateSession` — EP was declining nodes. Three root causes:
1. Claim filter (`ep.rs:275`) rejects `DataType::Undefined` outputs.
2. Dtype filter same rejection.
3. `Clip` missing from shape inference op lists.
4. Single-kernel fast path passed ORT inputs directly without injecting absent sentinels.

**Fixes:** Claim filter carve-out for absent outputs; Clip added to `SameAsInput(0)`; `input_slots` mapping in fast path; axis bounds: `>= rank`; scratch buffer: `numel * primary_output.byte_size()`.

**Outcome:** Challenger's review found the `__absent_output_*` sentinel was forgeable from model content. Locked out; Coco fixed.

## 2026-08-12 — Scope correction on PR #31993

**Task:** Rescope MLAS f16↔f32 cast kernel PR to macOS arm64 only.

**Changes:**
- Removed `#else` branch in `TestKernelIsDispatched()` (x86_64 Apple null-pointer assertions)
- Rewrote first commit message: removed universal2/iOS/Intel references
- Updated dispatch test comment to specify macOS arm64
- Verified clang-format clean; ran leak check

**Preserved:** Compile-time gate (`__APPLE__ && MLAS_TARGET_ARM64`), positive non-null dispatch assertion (non-vacuous), portable scalar fallback, numeric coverage (normals, denormals, ±0, ±Inf, qNaN, sNaN, RTE, non-aligned lengths).

**New head:** `68ee0de`. Needs macOS arm64 CI to validate.

---

### 2026-08-12T02:30:00Z — PR #31993 lockout revision: rescoped to macOS arm64 only

Removed the `#else` branch in `test_cast_fp16.cpp` that asserted null dispatch pointers on non-ARM64 Apple (x86_64 slice test). Rescoped commit messages and PR body from universal2/iOS/Intel to macOS arm64 only. Compile-time gate `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` unchanged. Positive dispatch assertions (`ASSERT_NE`) survive. Head: `68ee0de`. PR remains draft.

## 2026-08-12 — PR #31988 B1/B2/B3 fixes

- **B1 fixed:** Separated admission (cols=8 shared-mem gate) from launch (selected cols_per_block). Accepted-shape set is provably identical to upstream. Regression test added.
- **B2 fixed:** Replaced `kTargetCtasPerSm = 12` with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` per-instantiation queries. Falls back to conservative host heuristic. Cannot validate without GPU.
- **B3 fixed:** Added acceptance-set regression test, wide-N coverage, forcing hook test, structured GPU parity/occupancy tests (GTEST_SKIP'd).
- **Reverted** memcpy/strict-aliasing changes from common.cuh (separate concern).
- **Simplified** no-op else-if nesting in TryMatMul4Bits.
- **Added** explicit else-if dispatch with ORT_THROW for unexpected cols_per_block.
- **#29469 overlap:** Mechanical signature conflict (both add params to TryMatMul4Bits), no semantic overlap.
- **Recommendation:** Park until GPU access. Head: `dc1e173e4b`.

## 2026-08-12 — PR #31988 B1/B2/B3 fixes (admission separated, occupancy model)

Fixed admission bug B1: shared-memory gate scaled with `cols_per_block` (2/4) instead of fixed 8, silently admitting large-K shapes upstream declines → fp16 GEMV on shapes intended for cuBLAS fp32. Admission now always uses `kColsPerThreadBlock`=8. Proved correct by sweeping 20,800 combinations. B2: replaced `kTargetCtasPerSm=12` with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` per instantiation. B3: 6 new tests (2 GPU tests GTEST_SKIP'd). PR parked pending GPU access. Head `dc1e173e4b`.

## 2026-08-12 — PR #31973 evidence-accuracy fix

**Commit:** `fbf322f76b` on `nxrt/mlas-avx2-layernorm`

Fixed evidence-accuracy rejection: replaced unreproducible B1 figures
with independently measured numbers (scalar Welford 0.9357 vs kernel
0.03298, 28.4× better at base=1e5/spread=1e-2/N=1024/eps=1e-6). Fixed
RMSNorm benchmark to pass nullptr MeanOut matching production (speedups
up ~15-30%). Added dispatch and non-zero-case assertions. Fixed stale
labels and comments. Tests: 41 passed, 2 disabled, 43/43 total.

## 2026-08-12 — PR #31973 evidence-accuracy fix (B1 + B2)

Fixed both evidence-accuracy blockers. B1: replaced unreproducible accuracy headline (compared against deleted implementation) with figures printed by committed test — scalar Welford fp32 = 9.3573e-01 vs kernel = 3.2976e-02 (28.4× better), sweep 180 cases / 0 failures / worst 2.2318e-02. B2: fixed RMSNorm benchmark to pass `nullptr` MeanOut matching production (`layer_norm_impl.cc:507`); speedups rose ~15-30% at larger sizes. Added dispatch assertion on first warmup, non-zero case-count assertion, fixed stale label (`avx2_welford` → `avx2_centered`), corrected SCENARIO 3 comment, documented division disclosure. Head: `fbf322f76b`. Tests: 41 passed, 2 disabled, 43/43 total.
