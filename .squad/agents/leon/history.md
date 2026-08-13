# Leon — History (compacted 2026-08-12)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, reviewer lockouts.

**Historical summary through 2026-08-12:** Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization. Hardened loaders/fusion (unsupported dtypes fail-closed, LayerNorm operand-order guarded, opset validation recursive, `nxrt_*` C ABI). CUDA graph/capture correctness. PR #291 rewind policy split. Unified native CUDA/ORT KV capacity policy. EP plugin compute hardening (BL2/BL3 slot fidelity wave 1). Clippy dead_code cleanup. NEW-1 fix + f16/bf16 marshaling. Stream EP memory leak fix. Device data-transfer contract. TensorRT build fix (#31988). Apple Accelerate arm64 detection (#32001). BF16 LayerNorm PrePack counter + MLFloat16 stats coverage (#31974 — introduced regression, fixed by Coco).

Older detailed work archived in `history-archive.md`.

## 2026-08-12 — PR #31974 final cleanup: PrePack counter, MLFloat16 stats, centralised trait

- Threaded `number_of_pre_packed_weights_counter` through `RunBF16CpuOnly`; PrePack A/B tests now assert counter=0 (non-initializer) and counter=2 (initializer).
- Added `LayerNorm17_MLFloat16_MeanInvStdDev_FloatPrecision` test for fp16 stat precision.
- Moved `is_narrow_float_v` to `narrow_float_utils.h`.
- Verified counter non-vacuity by breaking PrePack and observing test failure.
- Test counts: 21 BF16 (was 20), 107 LayerNorm suite (was 106).
- Head SHA: 59b84aca7a
- ⚠️ This commit introduced a regression — see entry below.

## 2026-08-12 — PR #31974 regression: is_packed default flip caused float LayerNorm breakage

Commit `59b84aca7a` (Leon) introduced a regression: flipped `is_packed` default from `false` to `true` in `LayerNormImpl::PrePack`. `ConvertMLFloat16ToFloatIfNeeded` only sets `is_packed` inside narrow-float branches; for float inputs it is a no-op, so float dispatch incorrectly believed Scale/Bias were prepacked and failed with "Missing Input: Scale". Nine float `LayerNormTest` cases broke. Coco root-caused and fixed in `e036e53d31` (one-line restore of `false` default). Full-suite results: BF16 21/21, LayerNorm 107/107, SkipLayerNorm 26/26. The `narrow_float_utils.h` centralisation was sound and kept.

**Lesson reinforced:** A flag set only on some code paths must default to the conservative value. Set it explicitly where the work happens.

## 2026-08-12 — CUDA-graph capture arc: PR #852 pin GQA fixed-capacity KV seq symbol (MERGED)

Link 3 (**PIN**) of the 5-blocker capture chain. Pinned the GQA fixed-capacity KV seq symbol so
the capture classifier admits **52 GQA nodes** (53 → 0 disqualifying symbols), keeping the
two-gate-AND capture safety and growth-invalidation intact. Merged after #848 → #850, before
#855 → #854. Shared arc result: native CUDA decode **11.4 → 23.13 tok/s**, capture fully engaged
(1 segment, 0 seams).
