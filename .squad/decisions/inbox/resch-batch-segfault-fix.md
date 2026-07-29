# Decision: Fix BNNS batch>1 SIGSEGV in Conv/Pool kernels

**By:** Resch (2026-07-28)
**PR:** squad/fix-batch-segfault

## Root cause

`BNNSFilterApplyBatch` with `batch_size > 1` causes a SIGSEGV inside
`libBNNS.dylib` for convolution filters created via
`BNNSFilterCreateLayerConvolution`. The crash is inside Apple's framework
code (frame #0 in libBNNS.dylib, confirmed via AddressSanitizer). The single-
image `BNNSFilterApply` works correctly for the same filter.

The bug was introduced when BNNS Conv was added (PR #324 / #317) — it
exercised only batch=1 shapes. The batch dimension was correctly threaded
through buffer allocation and stride calculations, but the BNNS framework
itself crashes when the deprecated `BNNSFilterCreateLayerConvolution` +
`BNNSFilterApplyBatch` combination receives batch>1.

## Fix

Replace `BNNSFilterApplyBatch` with a per-image loop using `BNNSFilterApply`:
- Conv: `bnns_conv_execute` in `conv_ref.rs`
- Pool: `bnns_pool_execute` in `pooling.rs` (same class, prophylactic fix)

BNNS still uses its internal thread pool per `BNNSFilterApply` call, so the
AMX compute advantage is preserved. The overhead is one extra function call
per image — negligible relative to the convolution work.

## Why this is correct (not merely non-crashing)

- Batch invariance test proves element-wise equality between single-image and
  batched outputs for batch=2 and batch=4 on MobileNetV2.
- No buffer sizing or offset logic changed — only the BNNS dispatch strategy.
- The im2col+GEMM fallback (Tier 2) already processes per-image correctly and
  produces identical numerics; this fix aligns the BNNS tier to the same model.

## Audit: other kernels in this campaign

| Kernel | PR | Batch-safe? | Notes |
|--------|-----|------------|-------|
| Clip NEON (#359) | ✓ | Uses `numel(input.shape)` — shape-agnostic |
| Relu NEON (#361) | ✓ | Uses `numel(input.shape)` — shape-agnostic |
| Depthwise NEON (#342) | ✓ | Explicit `for b in 0..batch` loop |
| 1×1 Conv GEMM (#347) | ✓ | Routes through im2col_gemm_execute which has per-batch loop |
| BNNS pooling (#324) | ✓ (fixed) | Same BNNSFilterApplyBatch pattern — fixed prophylactically |
| vDSP Add (#324) | ✓ | No BNNS filter; uses vDSP on flat element spans |
| BatchNorm fusion | ✓ | Graph-level elimination, not a runtime kernel |
| BNNS MatMul (accelerate_gemm) | ✓ | Uses `BNNSFilterApplyTwoInput` (not batch) |

Only the two `BNNSFilterApplyBatch` call sites (Conv and Pool) shared this
defect class.

## Batch>1 performance (MobileNetV2, native)

Measured at load 2.7–3.0 (contended — Iran holds bench lock):

| batch | median ms | throughput (samples/s) | scaling |
|------:|----------:|----------------------:|--------:|
| 1 | 11.5 | 86.7 | 1.00× |
| 2 | 22.7 | 88.1 | 1.02× |
| 4 | 45.1 | 88.7 | 1.02× |
| 8 | 89.2 | 89.7 | 1.04× |
| 16 | 178.1 | 89.8 | 1.04× |

Native batch scaling is ~1.0× (linear cost, no amortization). ORT gets 1.9×
because its internal NCHWc path and thread pool amortize overhead across
images. Our per-image BNNS dispatch can't achieve that without the newer
`BNNSGraph` API (which supports batch natively). This is a future optimization
axis, not a regression — before this fix, batch>1 simply crashed.

## Standing lesson

`BNNSFilterApplyBatch` is unreliable for `BNNSFilterCreateLayerConvolution`
filters. Use per-image `BNNSFilterApply` until the crate migrates to
`BNNSGraph` (which supersedes the deprecated per-layer API and supports batch
natively via its graph-level batch dimension).
