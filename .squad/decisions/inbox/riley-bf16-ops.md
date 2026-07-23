### 2026-07-23: Extend CPU kernels across bfloat16 gaps
**By:** Riley
**What:** Added or verified bfloat16 execution and oracle coverage for the requested CPU kernels.
**Why:** Bfloat16 models must remain on the native CPU EP without dtype-conversion islands around decode-critical and general-purpose operators.

| op | bfloat16 before? | bfloat16 after? | test added? | notes |
|---|---:|---:|---:|---|
| RMSNormalization | Yes, unverified | Yes | Yes | Existing widen/compute-in-f32/narrow path; tested decode `[1,8]` and prefill `[2,3,8]`. |
| RotaryEmbedding (standard + contrib math) | Yes, unverified | Yes | Yes | Existing generic widen/narrow path; tested decode and batched prefill. |
| Clip | Yes, unverified | Yes | Yes | Existing arithmetic dispatch already included `half::bf16`. |
| ArgMax | No | Yes | Yes | Bfloat16 widens exactly to f32 before comparisons. |
| ArgMin | No | Yes | Yes | Bfloat16 widens exactly to f32 before comparisons. |
| TopK | No | Yes | Yes | Values widen for ordering and narrow to the input dtype; indices remain Int64. |
| NonZero | No | Yes | Yes | Bfloat16 values widen before exact zero comparison. |
| QuantizeLinear | No | Yes | Yes | Bfloat16 input and scale are read as f32; integer output semantics are unchanged. |
| DequantizeLinear | No | Yes | Yes | Integer values compute with widened bfloat16 scale and narrow to bfloat16 output. |
| DynamicQuantizeLinear | No | No | Yes, rejection | Intentionally Float32-only: ONNX schema 11 constrains input to `tensor(float)`. |
| BlockQuantizedMatMul | No | Yes | Yes | Float32/Float16/BFloat16 activations, bias, and output now share f32 accumulation. |
| MatMul | Yes | Yes | Existing | Bfloat16 unit and golden oracle tests verified. |
| Add | Yes | Yes | Existing | Bfloat16 unit and golden oracle tests verified. |
| LayerNormalization | Yes | Yes | Existing | Existing widened-input f32 oracle test verified. |

All added arithmetic paths widen `half::bf16` storage to f32, compute in f32, then narrow once with round-to-nearest-even. Tests derive references from the same already-rounded bfloat16 inputs. Normalization, rotary embedding, selection, and dequantization use `2e-3`–`5e-3` absolute floors plus `1e-2` relative tolerance; block-quantized matmul uses `5e-3 + 1e-2 * |reference|` for accumulated output. Exact discrete results use equality.

The CPU provider claims ordinary registered operators by registry key rather than dtype-specific metadata. Therefore bfloat16 reaches these kernels as `TensorView { dtype: BFloat16 }`; correctness depends on execute-time dtype dispatch and widening.

Full CPU EP library test count: **798 before, 808 after**. `cargo test -p onnx-runtime-ep-cpu --features mlas` passed, including the 808 library tests and integration regressions.

Intentionally skipped:

- `DynamicQuantizeLinear` bfloat16, because ONNX defines only Float32 input for this operator. The kernel now rejects bfloat16 explicitly with a corrective message.

Remaining candidates for a future bfloat16 pass:

- `causal_conv.rs` explicitly mentions Float16 without BFloat16.
- F32-only helper users needing individual schema/contract review: `affine_grid.rs`, `block_quantized_moe.rs`, `col2im.rs`, `compressed_sparse_attention.rs`, `fused_gemm.rs`, `fused_matmul_bias.rs`, `grid_sample.rs`, `index_share.rs`, `moe.rs`, `qmoe.rs`, and `sparse_kv_gather.rs`.
- Selection integer coverage remains incomplete for ArgMax/ArgMin/TopK/NonZero, but that is separate from the completed bfloat16 gap.
