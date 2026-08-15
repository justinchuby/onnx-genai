# CUDA Execution Provider — Op Coverage & Library Mapping

**Crate:** `onnx-runtime-ep-cuda` · **Target:** NVIDIA Hopper (SM90, H100/H200) ·
**Backend stack:** `cudarc` (dynamic-loading: driver + cuBLASLt + NVRTC).

This is the **roadmap and source of truth** for which ops the CUDA EP covers,
which off-the-shelf library backs each one, and which ops justify a custom fused
kernel. It follows the governing directive
(`.squad/decisions/inbox/coordinator-cuda-kernel-strategy.md`) and RULES.md #4:

> **Library-first.** Use cuBLAS/cuBLASLt (GEMM), cuDNN (conv/pool/softmax/norm/
> activations), CUTLASS (fused-epilogue GEMM), thrust/cub (reductions, scan,
> sort, topk). Write a **custom kernel only** when nothing off-the-shelf covers
> the op *or* we can measurably beat the library via fusion. **PyTorch-class
> fast. Coverage must be full.**

The **coverage reference set** is the CPU EP registry
(`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs::build_cpu_registry`): the CUDA
EP should ultimately cover the same ops the runtime emits. This matrix is
model-agnostic (RULES.md #2) — every op is shape-/dtype-/attribute-driven.

---

## Running the GPU tests at all

The suite is 45 `*_gpu.rs` files, and every one of them returns early when there
is no usable GPU. That is correct, but it means a machine with no CUDA and a
machine where everything passed report the same thing: `ok`. Printing a warning
does not help either — `cargo test` captures the output of a *passing* test, so
a `SKIPPED` line is invisible in exactly the runs where it matters.

This is not hypothetical. All 44 files then in the tree skipped on a developer
machine with a working RTX 4060, because nothing on the pure-Rust path
discovered NVIDIA's pip wheels. Nothing was red. Two real defects were sitting
behind it: a tensor-core kernel that could not compile, and an allocation
counter that had silently stopped counting.

### Point the loader at the wheels

No system CUDA install is needed.

```
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
            nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cudnn-cu12

NXRT_CUDA_WHEEL_ROOTS=<the site-packages directory containing `nvidia/`>
```

`nvidia-cuda-nvcc` is easy to miss and separately required: `mma.h` includes
`crt/mma.h`, which ships there rather than in `nvidia-cuda-runtime`, and without
it the tensor-core kernels fail inside NVRTC with a message naming a header
rather than a package.

The component `bin` (Windows) or `lib` directories on the loader path work too;
the environment variable is just the explicit form.

### Make a skip visible

```
NXRT_REQUIRE_CUDA=1
```

Set it wherever a GPU is supposed to exist. `suite_canary_gpu.rs` then fails —
with the likely cause and the fix — instead of the whole suite quietly passing.
Left unset it is a no-op, so CPU-only machines are unaffected.

There is no GPU runner in CI today; the CUDA lanes compile only. So this is a
developer-machine guard, and the reason the flag exists rather than the check
being unconditional.

---

## Backend legend


| Tag | Backend | When it is the right choice |
|-----|---------|-----------------------------|
| **cuBLASLt** | `cudarc::cublaslt` (`blas.rs`) | GEMM / batched GEMM, incl. fused bias/act epilogues (`CUBLASLT_EPILOGUE_*`). |
| **cuDNN** | `cudarc` `cudnn` feature | conv, pooling, softmax, activations, batch/instance/layer norm, LRN. Vendor-tuned, PyTorch's own backend. |
| **CUTLASS/CuTe** | NVRTC-compiled device templates | GEMM epilogue fusions cuBLASLt can't express; flash-attention-class kernels. |
| **thrust/cub** | `cudarc` (device primitives) | reductions, cumsum/scan, sort, topk, argmax. |
| **NVRTC-custom** | runtime-compiled `extern "C"` kernel (`nvrtc_function`) | pointwise elementwise / activation chains, fused norm+residual, RoPE — cases with **no library** or a **fusion win**. |
| **memcpy** | driver D2D copy / view rewrite | pure data-movement ops (no arithmetic). |

Custom kernels are compiled via **NVRTC at runtime** (cudarc dynamic-loading) —
there is **no `nvcc` / `build.rs`** in this crate, so `cargo build` needs no CUDA
toolkit (the driver, cuBLASLt, and NVRTC are `dlopen`'d at run time).

---

## Coverage matrix (reference set = CPU EP registry)

Status: **✅ implemented** on CUDA today · **⏳ next** (clear library mapping,
not yet wired) · **🔬 custom** (needs a fused NVRTC/CUTLASS kernel).

### GEMM family

| Op | Domain | Status | Backend | Notes / justification |
|----|--------|--------|---------|-----------------------|
| `MatMul` | `` | ✅ | **cuBLASLt** | Dense rank ≥ 2 with N-D batch broadcasting, f32/f16/bf16, true-fp32 accum (`matmul.rs`); rank-1 promotion pending. |
| `Gemm` | `` | ✅ | **cuBLASLt** + NVRTC bias | `Y=α·A'·B'+β·C`, transA/transB, α/β; fused NVRTC `β·C` broadcast-bias epilogue (`gemm.rs`). f32. |
| `FusedMatMulBias` | `com.microsoft` | ✅ | **cuBLASLt** `CUBLASLT_EPILOGUE_BIAS` | Dense rank-2 f32/f16/bf16 with an exact per-N bias vector; bias add is fused into GEMM with no elementwise pass. |
| `FusedGemm` | `com.microsoft` | ✅ | **cuBLASLt** `CUBLASLT_EPILOGUE_{BIAS,RELU_BIAS,GELU_BIAS}` | Dense rank-2 f32/f16/bf16; transA/transB and α. CUDA 13's `GELU_BIAS` is the tanh/0.044715 GELU approximation (H200 output differs from exact-erf at the expected ~2.2e-4 for x=1.5); cuBLASLt exposes no exact-erf selector, so this deliberately follows the vendor epilogue rather than adding a separate pass. Bias must be per-N and `beta=1` because `BIAS_POINTER` is unscaled; other β values fail explicitly. Missing `activation` defaults to Relu for the repository optimizer's existing `FusedGemm` contract; empty/Identity selects plain BIAS. |
| `MatMulNBits` | `com.microsoft` | ✅ | **NVRTC-custom + cuBLASLt** | Standard packed INT4 `[N, ceil(K/block_size), block_size/2]` weights are block-wise dequantized to f32 on-device, then multiplied by f32 activations with full-f32 accumulation. Supports optional packed zero points, group indices, and fused per-N bias. |
| `QMoE` | `com.microsoft` | ✅ | **NVRTC grouped block-dequant expert GEMV/GEMM** | Single-GPU resident: exact ORT expert-major affine INT1/INT2/INT4/INT8 layouts, optional packed zero points and expert biases, top-k routing/normalization, Relu/Gelu/Silu/Identity and fused or separate-gate SwiGLU. Decode uses per-route GEMV; prefill counts and prefix-groups routes by expert, gathers contiguous rows, runs a tiled dequant-GEMM for experts with at least two tokens, then deterministically combines weighted route outputs. f32/f16/bf16 activations use f32 accumulation; tile width, shared-memory fallback, launch width, and saturation grid derive from detected device limits. Native IQ/MXFP4 blocks remain explicitly unsupported because the ORT QMoE affine inputs cannot encode those layouts; they require a block-quantized MoE operator. Weight paging/prefetch and expert-parallel sharding are deferred. |

### Convolution

| Op | Domain | Status | Backend | Notes / justification |
|----|--------|--------|---------|-----------------------|
| `Conv` | `` | ✅ | **cuDNN** | 2-D dense NCHW f32/f16/bf16; strides, dilation, groups, symmetric explicit padding, `VALID`, symmetric `SAME_UPPER`/`SAME_LOWER`, and optional fused channel bias. Asymmetric padding is rejected explicitly. Fused bias+identity forces `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`; other paths use v7 heuristics and queried workspace. |

### Pooling

| Op | Domain | Status | Backend | Notes / justification |
|----|--------|--------|---------|-----------------------|
| `MaxPool` | `` | ✅ | **cuDNN** `cudnnPoolingForward` | 2-D NCHW f32/f16/bf16; kernel, strides, symmetric explicit padding, `VALID`, and symmetric `SAME_UPPER`/`SAME_LOWER`. `ceil_mode=1`, dilated pooling, `storage_order`, asymmetric padding, and the optional ONNX Indices output are rejected explicitly. |
| `AveragePool` | `` | ✅ | **cuDNN** `cudnnPoolingForward` | Same geometry/dtypes; `count_include_pad` maps to cuDNN include/exclude-padding modes. `ceil_mode=1`, dilation, `storage_order`, and asymmetric padding are rejected explicitly. |
| `LpPool` | `` | ✅ | **NVRTC-custom** | Arbitrary-rank NCHW-style f32/f16/bf16 Lp window reduction with positive integer `p`, strides, dilation, explicit/automatic padding, and `ceil_mode`; accumulation widens to f32 (`pooling.rs`). |
| `GlobalAveragePool`, `GlobalMaxPool`, `GlobalLpPool` | `` | ✅ | **NVRTC block reduction** | Arbitrary-rank NCHW-style f32/f16/bf16 global spatial reduction. Average, max, and integer-`p` Lp semantics match the CPU EP (`global_reduction.rs`). |

### Elementwise — unary / activations

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Relu`, `Sqrt`, `Erf`, `Tanh`, `Sigmoid`, `Gelu` | standard / `com.microsoft` | ✅ | **NVRTC-custom** | f32/f16/bf16; half storage widens to f32 compute and narrows once on store (`elementwise.rs`). |
| `Abs`, `Neg`, `Reciprocal`, `Exp`, `Log`, `Sign`, `Floor`, `Ceil`, `Round`, `Sin`, `Cos`, `Softplus` | `` | ✅ | **NVRTC-custom** | f32/f16/bf16 with CPU-matched formulas (`pointwise.rs`); `Round` uses ties-to-even and `Sign` preserves NaN. |
| `Tan`, `Sinh`, `Cosh`, `Asin`, `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh` | `` | ✅ | **NVRTC-custom** | Trigonometric/hyperbolic family (`pointwise.rs`); f32/f16/bf16 with half storage widened to f32 compute, matching the CPU EP's f32-widened reference. |
| `LeakyRelu`, `Elu`, `HardSigmoid`, `Clip`, `Softsign`, `Selu` | `` | ✅ | **NVRTC-custom** | Attribute/input-driven f32/f16/bf16 activations (`activations.rs`), computed in f32 for half storage. |
| `Swish` (v24), `ThresholdedRelu` (v10) | `` | ✅ | **NVRTC-custom** | Attribute-driven f32/f16/bf16 activations (`activations.rs`); `Swish` is `x·sigmoid(alpha·x)` and `ThresholdedRelu` is `x>alpha ? x : 0` (both `alpha` default `1.0`), computed in f32 for half storage. |

### Elementwise — logical / comparison

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Not` | `` | ✅ | **NVRTC-custom** | bool→bool, non-zero byte = true, canonical `1`/`0` out (matches CPU `logical.rs`). |
| `And`, `Or`, `Xor` | `` | ✅ | **NVRTC-custom** | bool operands → bool with NumPy right-aligned broadcasting and canonical `1`/`0` output. |
| `Equal`, `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual` | `` | ✅ | **NVRTC-custom** | f32/i32/i64 operands → bool with NumPy right-aligned broadcasting and ONNX comparison semantics; `Equal` also supports bool operands. |

### Elementwise — binary

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max` | `` | ✅ | **NVRTC-custom** | f32/f16/bf16 with NumPy right-aligned broadcasting. Host-computed output shape plus zero-stride metadata drives one generic device index walk; half arithmetic computes in f32. |
| `Sum`, `Mean` | `` | ✅ | **NVRTC-custom** | Variadic f32/f16/bf16 with NumPy right-aligned broadcasting (`nary.rs`); accumulates each input into an f32 scratch buffer, then `Mean` divides by the input count before narrowing once on store. |
| `Mod` (v10) | `` | ✅ | **NVRTC-custom** | Broadcasting modulo (`mod_op.rs`); f32 requires `fmod=1` (`fmodf`), i32/i64 support both C-truncated (`fmod=1`) and Python floor (`fmod=0`) modes; a zero divisor yields `0`, matching the CPU EP. |

### Normalization & softmax

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Softmax` (v1 & v13) | `` | ✅ | **cuDNN** `cudnnSoftmaxForward` | `ACCURATE` algorithm, f32/f16/bf16. Legacy coerce-to-2D uses INSTANCE mode; opset ≥ 13 uses a 4-D channel view for exact single-axis semantics. Falls back to the prior NVRTC kernel for f32 when cuDNN is unavailable. |
| `LayerNormalization` | `` / `com.microsoft` | ✅ | **NVRTC-custom** (fused) | Mean/var + normalize + affine in **one** pass over one HBM read — beats a cuDNN reduce + separate pointwise affine. Population stats, optional `Mean`/`InvStdDev` outputs, arbitrary `axis` (`normalization.rs`). f32. |
| `SkipLayerNormalization` | `com.microsoft` | ✅ | **NVRTC-custom** (fused) | `LayerNorm(input + skip + bias)·γ + β` — the residual add is fused into the norm, saving a whole tensor round-trip. Optional `beta`/`bias` inputs, optional `mean`/`inv_std`/`input_skip_bias_sum` outputs (`normalization.rs`). f32. |
| `SkipSimplifiedLayerNormalization` | `com.microsoft` | ✅ | **NVRTC-custom** (fused) | `RMSNorm(input + skip + bias)·γ` with no mean subtraction. Right-aligned broadcast `skip`, optional `bias`, and optional mean/inverse-RMS/residual-sum outputs (`normalization.rs`). f32. |
| `RMSNormalization` / `SimplifiedLayerNormalization` | `` / `com.microsoft` | ✅ | **NVRTC-custom** (fused) | Root-mean-square scale, no mean subtraction (LLaMA-family norm). Optional `InvStdDev` output, arbitrary `axis` (`normalization.rs`). f32. |
| `BatchNormalization` | `` | ✅ | **NVRTC-custom** | Inference-mode channel-wise normalization for contiguous f32/f16/bf16 NCHW-style tensors; custom epsilon and per-channel scale/bias/mean/variance (`batch_normalization.rs`). |
| `InstanceNormalization`, `GroupNormalization` | `` | ✅ | **NVRTC block reduction** | Arbitrary-rank contiguous NCHW-style f32/f16/bf16 normalization. Instance normalization reduces per `(N,C)` slice; group normalization supports opset-18 per-group and opset-21 per-channel affine parameters with float stash semantics (`group_normalization.rs`). |
| `LpNormalization` | `` | ✅ | **NVRTC block reduction** | Axis-wise p=1/p=2 normalization for f32/f16/bf16, including negative and interior axes with CPU-matched tiny-norm clamping (`global_reduction.rs`). |
| `ReduceMean` | `` | ✅ | **cuDNN** `cudnnReduceTensor` | See reductions below. |

### Reductions

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `ReduceSum` | `` | ✅ | **cuDNN** `cudnnReduceTensor` (ADD) | Arbitrary axes (attribute or opset-13+ input), `keepdims`, `noop_with_empty_axes`, negative axes; f32/f16/bf16. RAII workspace, no indices. Falls back to the prior NVRTC f32 kernel when cuDNN is unavailable. |
| `ReduceMean` | `` | ✅ | **cuDNN** `cudnnReduceTensor` (AVG) | Same shape/axis handling and fallback as `ReduceSum`. |
| `ReduceMax` | `` | ✅ | **NVRTC block reduction** (cub-class) | As above; NaN-propagating (numpy / CPU-EP semantics). |
| `ReduceMin` | `` | ✅ | **NVRTC block reduction** (cub-class) | As above; NaN-propagating. |
| `ReduceProd` | `` | ✅ | **NVRTC block reduction** (f32) | Product over the reduced group; shares the offset-table plumbing (`reduce.rs`). |
| `ReduceSumSquare` | `` | ✅ | **NVRTC block reduction** (f32) | Sum of squares (`x²`) over the group. |
| `ReduceL1` | `` | ✅ | **NVRTC block reduction** (f32) | Sum of `abs(x)` over the group. |
| `ReduceL2` | `` | ✅ | **NVRTC block reduction** (f32) | `sqrt` of the sum of squares over the group. |
| `ReduceLogSum` | `` | ✅ | **NVRTC block reduction** (f32) | `log` of the sum over the group. |
| `ReduceLogSumExp` | `` | ✅ | **NVRTC block reduction** (f32) | `log(sum(exp(x)))` over the group, matching the CPU EP's naive (no max-shift) accumulation; opset-1 attribute and opset-18 axes-input forms. |

> **Why NVRTC block reduction, not cub?** cub's `DeviceSegmentedReduce` is the
> vendor primitive, and our kernel matches its shape (one block per output
> element, cooperative shared-memory tree reduce over that element's group). We
> keep it as an NVRTC kernel so the crate stays toolkit-free (no `nvcc`/`build.rs`);
> the offset tables let it handle any axis set / rank without special-casing.

### Attention

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Attention` | `com.microsoft` | ✅ | **NVRTC tiled online-softmax + Phase-2a fallback** | Phase-2b fused f32/f16/bf16 prefill, including GQA and additive masks; measured auto-gate retains the cuBLAS baseline where faster. See `CUDA_FLASH_ATTENTION.md`. |
| `Attention` (opset 23/24) | `` | ✅ | **deterministic CUDA EP fallback** | Standard ONNX SDPA with 3D/4D layouts, GQA, bool/additive masks, and in-op KV cache. f32. |
| `RotaryEmbedding` (opset 23) | `` | ✅ | **deterministic CUDA EP fallback** | f32 interleaved/non-interleaved RoPE, partial rotary dimensions, and optional position-id gathering. |
| `FusedAttention` | `com.microsoft` | 🔬 | **fusion rewrite to `Attention`** | The fused kernel exists behind `AttentionKernel`; registering/lowering this op name remains. |

### Shape / data-movement / misc

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Cast` | `` | ✅ | **NVRTC-custom** | Element-wise dtype conversion; f32/f64/f16/bf16/int8-64/uint8-64/bool, ONNX saturating float→int. Two NVRTC modules keep f16/bf16 (which need NVRTC's built-in `cuda_fp16.h`/`cuda_bf16.h`) out of the common integer/f32 path (`cast.rs`). |
| `CastLike` | `` | ✅ | **NVRTC-custom** | Same kernel as `Cast`; target dtype taken from the output tensor. |
| `Identity` | `` | ✅ | **memcpy** (D2D) | Straight dtype-agnostic device copy into the pre-shaped output (`movement.rs`). |
| `Flatten` | `` | ✅ | **memcpy** (D2D) | Dtype-agnostic D2D copy; the 2-D output shape is metadata-only, so the bytes are copied unchanged (`movement.rs`). |
| `Concat` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic byte copy, arbitrary/negative axis, multiple inputs. |
| `Reshape`, `Squeeze`, `Unsqueeze` | `` | ✅ | **memcpy** | Dtype-agnostic D2D copy into the executor's pre-shaped output; modern axes inputs and legacy attributes are accepted. |
| `Transpose` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic indexed byte copy; explicit permutation or default axis reversal. |
| `Gather` | `` | ✅ | **NVRTC-custom** | Axis-parametric contiguous indexed copy; Int32/Int64 indices, negative index wrap, arbitrary index rank. |
| `GatherElements` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic contiguous element gather with Int64 indices, negative axes/indices, and CPU-matched rank/dimension validation. |
| `ScatterElements` (v11 & v16) | `` | ✅ | **NVRTC-custom** | Deterministic f32/Int64 updates in row-major input order. Opset 16 supports `none`, `add`, `mul`, `min`, and `max`; duplicate indices therefore preserve CPU last-write/sequential-reduction semantics. |
| `TopK` (v10) | `` | ✅ | **NVRTC-custom** | Deterministic f32 per-slice selection with input `K`, arbitrary/negative axis, largest/smallest modes, and CPU tie-breaking by lower source index. |
| `CumSum` (v11) | `` | ✅ | **NVRTC-custom** | Deterministic per-lane f32/Int64 scan with scalar axis input, negative axes, and all exclusive/reverse combinations. |
| `Shape` | `` | ✅ | **host + H2D** | Computes the metadata-only Int64 shape vector on host, including opset-15 `start`/`end`, then uploads it. |
| `Size` | `` | ✅ | **host + H2D** | Computes the metadata-only Int64 element count on host and uploads the scalar (`size.rs`). |
| `Trilu` (v14) | `` | ✅ | **NVRTC-custom** | Dtype-agnostic byte copy that zeroes the elements outside the retained triangle over the trailing two dimensions (`trilu.rs`); `upper` attribute plus optional Int64 `k` diagonal input, matching the CPU EP. |
| `Expand` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic broadcast copy sharing the binary-elementwise zero-stride indexing infrastructure. |
| `Slice` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic strided/stepped copy with opset-10 input-driven ranges, negative axes, and negative steps. |
| `Split` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic byte copy; split input, legacy attribute, even split, negative axis, and opset-18 `num_outputs`. |
| `Tile` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic repeated indexed copy across arbitrary axes. |
| `Where` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic branch selection with right-aligned broadcasting across condition, x, and y. |
| `Constant` | `` | ✅ | **host + H2D** | Uploads `value` tensors and numeric `value_*` attribute forms to the device. |
| `BiasGelu` | `com.microsoft` | ✅ | **NVRTC-custom** | Exact (erf) GELU of `X + bias` with the bias broadcast over the last dimension; f32/f16/bf16, GELU evaluated in `double` and the `-inf → 0` guard matched to the CPU EP (`fused_gelu.rs`). |
| `FastGelu` | `com.microsoft` | ✅ | **NVRTC-custom** | Tanh-approximation GELU of `X` plus an optional last-dim bias (1- or 2-input arity); f32/f16/bf16, matched to the CPU EP. |
| `QuickGelu` | `com.microsoft` | ✅ | **NVRTC-custom** | `X · sigmoid(alpha · X)` with the `com.microsoft` default `alpha=1.702`; f32/f16/bf16, numerically stable sigmoid matched to the CPU EP. |
| `CumProd` (v26) | `` | ✅ | **NVRTC-custom** | Deterministic per-lane cumulative product (f32/i64) honouring `exclusive`/`reverse`, mirroring the `CumSum` scan (`cumprod.rs`). |
| `ArgMax` | `` | ✅ | **NVRTC-custom** | Per-lane axis reduction to Int64 indices (f32/f16/bf16 widened to f32), honouring `keepdims` and `select_last_index`; first-index tie-break matched to the CPU EP (`argreduce.rs`). |
| `ArgMin` | `` | ✅ | **NVRTC-custom** | Per-lane axis reduction to Int64 indices (f32/f16/bf16 widened to f32), honouring `keepdims` and `select_last_index`; first-index tie-break matched to the CPU EP (`argreduce.rs`). |
| `GatherND` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic indexed copy with Int32/Int64 indices, negative index wrapping, arbitrary tuple depth, and `batch_dims`; eager execution validates indices before launch and graph capture uses the device error latch (`structural.rs`). |
| `SpaceToDepth` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic NCHW spatial-block rearrangement with runtime `blocksize`, including multi-channel and empty-batch tensors (`structural.rs`). |
| `EyeLike` | `` | ✅ | **NVRTC-custom** | Rank-2 identity-like construction with positive/negative diagonal offsets and the full CPU numeric/bool dtype set, including dtype override (`structural.rs`). |
| `AffineGrid` | `` | ✅ | **NVRTC-custom** | Two- and three-dimensional sampling-grid construction for f32/f16/bf16 theta, including `align_corners` and singleton spatial dimensions (`data_transform.rs`). |
| `Compress` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic fixed-width selection with omitted-axis flattening, positive/negative axes, and short conditions (`data_transform.rs`). |
| `DynamicQuantizeLinear` | `` | ✅ | **NVRTC-custom** | Float32 per-tensor dynamic uint8 quantization; one block derives min/max, scale, and ties-to-even zero point before quantizing (`quantization.rs`). |
| `Pad` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic constant/reflect/edge/wrap padding, negative-pad cropping, and opset-18 subset/negative axes (`pad.rs`). |
| `Range` | `` | ✅ | **NVRTC-custom** | Scalar-driven f32/f16/bf16/Int64 sequence construction with positive/negative steps and CPU-matched output-count validation (`range.rs`). |
| `ScatterND` (v11/v16/v18) | `` | ✅ | **NVRTC-custom** | Deterministic slice updates in row-major tuple order for f32/f16/bf16/Int64 data and Int64 indices; negative indices and `none`/`add`/`mul`/`min`/`max` reductions match the CPU EP (`indexing.rs`). |
| `HannWindow`, `HammingWindow`, `BlackmanWindow` (v17) | `` | ✅ | **NVRTC-custom** | Periodic or symmetric signal windows generated directly on device in f16/bf16/f32/f64, with the scalar size and `output_datatype` contract matched to the CPU EP (`window.rs`). |
| `CenterCropPad` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic fixed-width centered crop/zero-pad over all or selected axes, including negative axes and CPU-matched odd-difference placement (`index_transform.rs`). |
| `Col2Im` | `` | ✅ | **NVRTC-custom** | Arbitrary spatial-rank f32/f16/bf16 inverse image-column transform with overlap accumulation, dilation, strides, and padding; accumulation is widened to f32 (`index_transform.rs`). |

## Source-derived coverage audit (2026-07-29)

This snapshot is derived directly from `build_cpu_registry`,
`build_cuda_registry`, and `CUDA_COVERED_OPS`. It supersedes the stale
pre-batch counts retained in the historical wave notes below.

| Measure | Count |
|---------|------:|
| CPU registry `(domain, op_type)` pairs | **173** |
| CPU standard-domain (`ai.onnx`) op types | **145** |
| CUDA registry `(domain, op_type)` pairs | **169** |
| CUDA advertised op names (`CUDA_COVERED_OPS`) | **164** |
| CPU pairs implemented by CUDA in the same domain | **166 / 173** |
| CPU standard-domain op types implemented by CUDA | **143 / 145** |

The **2 remaining CPU `ai.onnx` gaps** are `NonMaxSuppression` and `Unique`.

Issue #67 batch (2026-07-30): a data-driven placement audit over the real target
decode models (Qwen2.5 0.5b/1.5b/7b, Phi-4-mini, Qwen3.6-27B, Qwen3.5-35B-A3B)
found the classic transformer decode path already places 100% of EP-placeable
nodes on CUDA (the only "uncovered" types are executor-handled control-flow
`If`/`Loop`/`Scan`). The genuine remaining gaps are the **Qwen3.5 hybrid**
(Mamba + linear-attention) family. This batch landed **`CausalConvWithState`**
(new NVRTC fp32/fp16/bf16 kernel — depthwise causal short-conv with rolling
state) and declared the already-registered **`GatherBlockQuantized`** in
`CUDA_COVERED_OPS` with a dedicated GPU parity suite. Honest follow-ups:
`com.microsoft::LinearAttention`, registering `RotaryEmbedding` for the
`com.microsoft` domain, and a `Bool`-input `NonZero` path.

The decode/transformer-oriented priority set from issue #67 is already covered:
`LogSoftmax`, `Hardmax`, `PRelu`, `IsInf`, the five bitwise/shift operators,
`ArgMax`, `ArgMin`, and `EyeLike`. `BiasGelu`, `FastGelu`, and `QuickGelu` were
also already registered and GPU-tested; they require no kernel change in this
batch.

For `com.microsoft`, the remaining CPU-only gap is `FusedAttention`;
`BiasGelu`, `FastGelu`, and `QuickGelu` are covered by `fused_gelu.rs`. CUDA
additionally exposes `com.microsoft::Attention`. CUDA standard-domain extras not
currently registered by the CPU EP include `Conv` (cuDNN).


### Library mapping for the remaining CPU gaps

| Backend | CPU-covered gaps mapped here | Rationale |
|---------|------------------------------|-----------|
| **CUTLASS / cuDNN SDPA** | `FusedAttention` | Flash/SDPA implementation avoids materialising the O(S²) score tensor. |
| **NVRTC-custom** | `Unique` | Data-dependent output construction with no suitable runtime library. |
| **deferred heavy operators** | `NonMaxSuppression` | Data-dependent selection deserves a dedicated follow-up wave and focused review. |

Wave 4 raises the advertised CUDA set from **48 to 54** op names. Its six
activations are GPU-validated against independent CPU formulas on the local
CUDA 13.0 host; broader performance validation remains separate.

The cuDNN Conv pass raises the advertised set to **55** op names and is
GPU-validated for padded f32, grouped/strided f32, padded f16, and numerically checked
dilated convolution on H200.

The cuDNN pooling pass raises the advertised set to **57** op names and is
GPU-validated on H200 for 2×2 stride-2 MaxPool in f32/f16 and padded AveragePool
with both include-padding and exclude-padding divisor modes.

The cuBLASLt fused-epilogue pass adds two advertised op names.
`FusedMatMulBias` uses `CUBLASLT_EPILOGUE_BIAS`; `FusedGemm` uses
`CUBLASLT_EPILOGUE_BIAS`, `CUBLASLT_EPILOGUE_RELU_BIAS`, or
`CUBLASLT_EPILOGUE_GELU_BIAS`. All three keep bias/activation inside GEMM.

The pointwise dtype/broadcast pass is GPU-validated on H200 for f16 and bf16
`Add`/`Sub`/`Mul`/`Div`, `[4,1,3]` with `[1,5,3]` NumPy broadcasting, and
representative unary/activation kernels. Half storage is widened to f32 for
compute and rounded once on output, matching the CPU EP convention.

The issue #67 operator-coverage batch adds thirteen advertised op names — the
trigonometric/hyperbolic unary family (`Tan`, `Sinh`, `Cosh`, `Asin`, `Acos`,
`Atan`, `Asinh`, `Acosh`, `Atanh`) plus `Identity`, `Flatten`, `Size`, and
`Trilu`. Each is GPU-validated against the CPU EP on the local CUDA host across
the dtypes/attributes it claims (trig ops in f32/f16/bf16; `Trilu` upper/lower
with positive/negative `k` in f32/Int64), raising CPU standard-domain op-type
coverage to **88 / 141**.

The issue #67 operator-coverage batch 2 adds eleven more advertised op names —
the extended f32 reductions (`ReduceProd`, `ReduceSumSquare`, `ReduceL1`,
`ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`), the attribute-driven activations
`Swish` (opset 24) and `ThresholdedRelu` (opset 10) in f32/f16/bf16, the variadic
broadcasting ops `Sum` and `Mean` in f32/f16/bf16, and `Mod` (f32 with `fmod=1`
plus i32/i64 truncated and floor modulo). Each is GPU-validated against the CPU
EP on the local CUDA host across the dtypes/attributes it claims (reductions over
several axes/keepdims combinations, including the opset-18 axes-input form for
`ReduceLogSumExp`; `Mod` with negative operands and a zero divisor). This raises
`CUDA_COVERED_OPS` to **113** advertised op names and CPU standard-domain op-type
coverage to **99 / 141**.

The issue #67 operator-coverage batch 4 adds seven more advertised op names — the
integer **bitwise** family (`BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`
over all integer dtypes, broadcasting) plus unsigned `BitShift` (LEFT/RIGHT), and
the softmax-family axis reductions `LogSoftmax` (f32/f16/bf16, opset-13 per-axis
and legacy opset-≤12 coerce-to-2D) and `Hardmax` (f32/f16/bf16, first-argmax
one-hot). All are **NVRTC-custom** kernels matched to the CPU EP: `LogSoftmax`
uses the stable shifted-`logsumexp` formulation (`max + log(sum(exp(x - max)))`,
the #266 overflow lesson), and `BitShift` mirrors the CPU `checked_shl`/
`checked_shr` contract (an amount `>=` the operand width yields `0`). Each is
GPU-validated against the CPU oracle on the local CUDA host across the
dtypes/attributes it claims (signed + unsigned bitwise with broadcasting,
over-shift, `LogSoftmax` large-magnitude rows and interior axes, `Hardmax` ties
and negative interior axes). This brings the machine-verified
`CUDA_COVERED_OPS` list length from **118** to **125** op names. (The narrative
"113" figure above is a stale pre-batch-3 snapshot; the authoritative count is the
`CUDA_COVERED_OPS` slice length, which the `covered_ops_have_no_duplicates` test
guards.)

The issue #67 operator-coverage batch 5 adds six more advertised op names — the
`com.microsoft` fused GELU activations `BiasGelu` (exact GELU of `X + bias`),
`FastGelu` (tanh GELU of `X` + optional broadcast bias), and `QuickGelu`
(`X · sigmoid(alpha · X)`) in f32/f16/bf16; the cumulative product `CumProd`
(f32/i64, `exclusive`/`reverse`); and the index reductions `ArgMax`/`ArgMin`
(f32/f16/bf16 → Int64, honouring `keepdims` and `select_last_index`). All are
**NVRTC-custom** kernels matched to the CPU EP: the fused GELUs evaluate the
error/tanh GELU in `double` to stay bit-close to the CPU oracle
(`contrib_fused.rs`) and guard the `-inf → 0` case; `CumProd` mirrors the
deterministic per-lane `CumSum` scan; and `ArgMax`/`ArgMin` reproduce the CPU
first-index (or, with `select_last_index`, last-index) tie-break. Each is
GPU-validated against the CPU oracle on the local CUDA host across the
dtypes/attributes it claims (bias broadcasting, the `FastGelu` no-bias arity and
`-inf` guard, `CumProd` exclusive/reverse/negative-axis and i64 byte-exact, and
`ArgMax`/`ArgMin` keepdims/axis variants plus a falsifiable tie-break probe).
This brings the machine-verified `CUDA_COVERED_OPS` list length from **125** to
**131** op names. `ArgMax`, `ArgMin`, and `CumProd` move out of the CPU `ai.onnx`
gap list, and `BiasGelu`/`FastGelu`/`QuickGelu` are no longer `com.microsoft`
gaps.

The issue #67 operator-coverage batch 6 adds three standard-domain structural
operators: `GatherND`, `SpaceToDepth`, and `EyeLike`. All three use
dtype-agnostic NVRTC byte kernels rather than a vendor library:
`GatherND` supports Int32/Int64 indices, negative wrapping, tuple tails, and
`batch_dims`; `SpaceToDepth` rearranges arbitrary fixed-width NCHW tensors; and
`EyeLike` emits exact zero/one storage for every numeric/bool dtype supported by
the CPU EP. GPU parity covers batch dimensions, negative indices, multiple
channels, dtype overrides, diagonal offsets, and empty tensors. This raises the
machine-verified `CUDA_COVERED_OPS` count from **131** to **134** and the current
CPU standard-domain parity count from **102 / 141** to **105 / 141**. These ops
move out of the CPU `ai.onnx` gap list.

The issue #67 operator-coverage batch 7 adds `Pad` and `Range`. `Pad` supports
all four ONNX modes, negative cropping, subset/negative axes, and arbitrary
fixed-width element storage. `Range` supports f32/f16/bf16 and Int64 with
positive or negative deltas. GPU parity covers nonzero constant fill, subset
axes, cropping, reflect/wrap modes, fractional float steps, narrow float
storage, and descending Int64 sequences. This raises the machine-verified
`CUDA_COVERED_OPS` count from **134** to **136** and current CPU standard-domain
parity from **105 / 141** to **107 / 141**.

The issue #67 operator-coverage batch 8 adds `ScatterND`, `HannWindow`,
`HammingWindow`, and `BlackmanWindow`. `ScatterND` performs deterministic,
ordered slice updates with negative-index wrapping and all opset-18 reductions
for f32/f16/bf16/Int64 data. The three window operators share one generalized
NVRTC implementation and support periodic/symmetric generation in
f16/bf16/f32/f64. GPU parity covers slice updates, duplicate indices,
`add`/`mul`/`max`, negative indices, every supported window dtype, and both
periodic modes. This raises `CUDA_COVERED_OPS` from **136** to **140** and CPU
standard-domain parity from **107 / 141** to **111 / 141**.

The issue #67 operator-coverage batch 9 adds per-tensor `QuantizeLinear` and
`DequantizeLinear` (f32 with scalar scale and optional scalar i8/u8 zero point),
deterministic inference-mode `Dropout` (identity data plus optional all-true
mask), and `NonZero` coordinate extraction for f32/f16/bf16. Inline GPU parity
covers signed and unsigned quantization, saturation and ties-to-even rounding,
multi-output Dropout, multiple ranks, narrow float storage, signed zero, and
NaN. This raises `CUDA_COVERED_OPS` from **140** to **144** and CPU
standard-domain parity from **111 / 141** to **115 / 141**.

The issue #67 operator-coverage batch 10 adds `AffineGrid`,
`BatchNormalization`, `Compress`, `DynamicQuantizeLinear`,
`GlobalAveragePool`, `GlobalLpPool`, `GlobalMaxPool`, and `LpNormalization`.
The implementations share three small NVRTC families: fixed-width data
transforms, channel-wise normalization, and block reductions. GPU parity covers
two- and three-dimensional affine grids, `align_corners`, negative and omitted
Compress axes, f32/f16/bf16 normalization and pooling, non-default Lp powers,
negative/interior normalization axes, and dynamic quantization for mixed-sign,
constant, and empty inputs. Current source-derived coverage is **152**
advertised CUDA op names, **157** CUDA `(domain, op_type)` pairs, and
**134 / 145** CPU standard-domain op types.

The issue #67 operator-coverage batch 11 adds `InstanceNormalization` and
`GroupNormalization`; batch 12 adds `LpPool`, `CenterCropPad`, and `Col2Im`.
The latter shares a general N-D pooling geometry path and a fixed-width index
transform module. GPU parity covers p=1/p=2, asymmetric padding, strides,
dilation, `ceil_mode`, mixed crop/pad with odd differences and selected axes,
and overlapping Col2Im accumulation with both stride/padding and dilation.
Current source-derived coverage is **157** advertised CUDA op names, **162**
CUDA `(domain, op_type)` pairs, and **139 / 145** CPU standard-domain op types.

The issue #67 operator-coverage batch 13 adds `QLinearMatMul` and `Resize`.
`QLinearMatMul` preserves the CPU reference's wrapping i32 accumulation,
ties-to-even requantization, and per-tensor/per-row/per-column quantization for
Int8 and Uint8. `Resize` supports nearest and N-D linear interpolation with
`half_pixel`, `align_corners`, and `asymmetric` coordinates, scales or sizes,
selected axes, and all four standard nearest rounding modes. Cubic,
`pytorch_half_pixel`, `tf_crop_and_resize`, `half_pixel_symmetric`, antialiasing,
and non-stretch aspect policies remain fail-closed at the claim gate. GPU parity
covers quantized signed/unsigned and batched per-axis cases plus nearest/linear
upsampling and downsampling across the supported coordinate and control modes.
Current source-derived coverage is **159** advertised CUDA op names, **164**
CUDA `(domain, op_type)` pairs, and **141 / 145** CPU standard-domain op types.

The issue #67 operator-coverage batch 14 adds `ConvTranspose` and `GridSample`.
`ConvTranspose` covers 1-D/2-D f32/f16/bf16 overlap-add with strides, asymmetric
pads, dilations, output padding, bias, groups, and depthwise geometry.
`GridSample` covers 4-D bilinear/nearest sampling for zeros, border, and
reflection padding with either `align_corners` setting. ConvTranspose
`SAME_UPPER`/`SAME_LOWER` and output-shape-driven padding, plus cubic/bicubic and
volumetric GridSample, remain explicitly fail-closed. GPU parity covers the
supported narrow storage types and out-of-bounds sampling. Current source-derived
coverage is **163** advertised CUDA op names, **168** CUDA `(domain, op_type)`
pairs, and **143 / 145** CPU standard-domain op types.

The issue #67 operator-coverage batch 15 adds `com.microsoft::LinearAttention`
(Gated DeltaNet / gated delta-rule linear attention), the recurrent attention of
the Qwen3.5 / Qwen3-Next **hybrid** family. A single NVRTC kernel (f32/f16/bf16
entry points) exploits the fact that each column of the per-head state matrix
`S[d_k, d_v]` evolves independently, mapping one thread to each `(batch, kv_head,
d_v-column)` and running the whole recurrent scan in **f32** (state kept in a
per-thread f32 register array so the arithmetic matches ORT's `float` CPU kernel
regardless of I/O dtype). It covers all four `update_rule` variants (linear,
gated, delta, gated_delta), standard and inverse GQA, key-head sharing
(`n_k < H_kv`), per-head and per-key-dim decay, per-head and shared beta, and
step-to-step state carry via `past_state`/`present_state`. The claim gate
fail-closes on unsupported dtypes and `d_k > 256`. GPU parity validates output
and present_state against the CPU EP oracle across every variant plus a
chained-vs-full state-carry proof; the placement probe confirms all
18 / 18 / 24 LinearAttention nodes in qwen3.5-0.8b/2b/9b now place on CUDA (0
before). This pairs with `CausalConvWithState` to land the hybrid decode path.
This raises the machine-verified `CUDA_COVERED_OPS` count to **162** and CUDA
`(domain, op_type)` pairs to **168**.

---

## Custom-kernel candidates (with WHY)

Ops that justify a **custom fused NVRTC / CUTLASS kernel** — either no library
covers them, or fusion measurably beats calling a library op-by-op. Ordered by
expected impact for transformer inference.

1. **`FusedAttention` → fused `AttentionKernel` lowering** *(Phase-2b kernel
   implemented)* — tiled online-softmax now removes the `[B,H,Sq,Sk]` HBM tensor
   for f32/f16/bf16 prefill and is 1.53x faster for H200 f16 S=512. Long-context
   tuning and the `FusedAttention` graph rewrite remain; automatic dispatch keeps
   Phase-2a at measured slower shapes. See `CUDA_FLASH_ATTENTION.md`.
2. **`LayerNormalization` / RMSNorm (fused)** — mean+variance reduction, the
   normalize, and the affine (`γ·x̂+β`) in **one** kernel over one HBM read.
   A library path is a reduction + several pointwise passes; the fused kernel
   removes the intermediate traffic. Add the residual add (`x+sublayer`) to make
   it **residual+norm** — a further fusion that saves a whole tensor round-trip.
3. **`FusedGemm` / `FusedMatMulBias` (cuBLASLt epilogue) — implemented.**
   `CUBLASLT_EPILOGUE_GELU_BIAS` / `RELU_BIAS` / `BIAS` run activation+bias
   inside GEMM, eliminating the separate elementwise pass.
4. **Elementwise chain fusion** — the unary/binary NVRTC kernels are deliberately
   *ours* (not cuDNN OpTensor) precisely so a producer→activation→add chain can
   be fused into a single pointwise kernel (one HBM read/write instead of N).
   This is why activations are NVRTC-custom in the matrix above.
5. **RoPE (rotary position embedding)** — no library op; a small fused kernel
   applying the sin/cos rotation in place over Q/K. Pure win, transformer-
   ubiquitous.
6. **Elementwise chain fusion** remains the next pointwise optimization;
   dtype-generic NumPy broadcasting is now implemented for arithmetic binaries.

Everything else in the matrix (`ReduceMean`→cub, `Softmax`→cuDNN, `Cast`,
data-movement) is a **straight library/primitive mapping**, not a custom-kernel
candidate.

---

## Runtime / build notes

- **Build is toolkit-free.** `cargo build -p onnx-runtime-ep-cuda` compiles with
  no CUDA toolkit because `cudarc` uses `dynamic-loading`; the driver, cuBLASLt,
  and NVRTC are `dlopen`'d at run time. Adding the `cudnn` feature for the ⏳
  norm/softmax/conv rows preserves this (cuDNN is dlopen'd too).
- **cuDNN is enabled** through cudarc's `cudnn` feature and a lazy, stream-bound
  backend in `CudaRuntime`; softmax, reductions, and Conv share that handle.
- **Runtime execution requires the libraries on the loader path.** A host with
  only `libcuda` (driver) but **without** `libcublasLt` / `libcudnn` can *build*
  and can run *pure-driver* code, but cuBLASLt/cuDNN ops error/skip until those
  libs are installed. Every such failure is an actionable `EpError` (RULES.md #1)
  naming the missing library and how to fix it.

`SkipSimplifiedLayerNormalization` raised the advertised CUDA set to **61** op names; `MatMulNBits` raised it to **62**. `Gather`, `Shape`, and `Constant` raise it to **65**; Gather uses an NVRTC device indexed-copy kernel, while Shape and Constant correctly use host metadata construction plus H2D upload.

---

## Conformance profile & GPU parity sweep

The CUDA EP is validated against the **CPU EP as the reference oracle** through a
declarative *conformance profile* rather than ad-hoc per-op tests. The profile
lives in `crates/onnx-runtime-ep-cuda/tests/cuda_conformance_gpu.rs` and its
shared parity harness in `crates/onnx-runtime-ep-cuda/tests/common/mod.rs`.

### What the profile is

A **conformance profile** is a table with exactly one entry per op in
`CUDA_COVERED_OPS` (`kernels/mod.rs`). Each op is classified as either:

- **`Sweep`** — inline `(op, dtype, shapes, attrs)` parity cases that a single
  generic harness runs on the real GPU and compares to the CPU EP running the
  identical node. New parity coverage is concentrated here on ops that
  previously had *no* dedicated parity test at all (e.g. `Sqrt`, `Erf`, `Tanh`,
  `Sigmoid`, `Abs`, `Neg`, `Reciprocal`, `Log`, `Sign`, `Floor`, `Ceil`,
  `Round`, `Sin`, `Cos`, `Softplus`, `Pow`, `Min`, `Max`, `ReduceMean/Max/Min`,
  `Cast`, `CastLike`, `Not`, `Gemm`, `SkipLayerNormalization`).
- **`Dedicated`** — covered by a named dedicated GPU suite (another
  `tests/*_gpu.rs` file). The profile records the suite file and a one-line note.

### Coverage-of-coverage guards (no GPU required — run in CI)

Three audits run on any host, including GPU-less CI runners, and are the
highest-value deliverable:

- `every_covered_op_has_a_conformance_entry` — **fails the moment an op is added
  to `CUDA_COVERED_OPS` without a parity test entry** (the "claimed but
  untested" defect class — exactly the miss that let `ReduceLogSumExp` and bf16
  coverage gaps slip through). It also flags stale entries for ops no longer
  covered.
- `dedicated_suites_exist_and_name_their_op` — reads each referenced suite file
  and asserts it actually names its op, so a deleted, renamed, or gutted suite
  cannot silently leave an op unverified.
- `profile_has_no_duplicate_entries`.

### How to run

The parity sweep is a normal `#[test]` that **graceful-skips when no CUDA device
is present**, so it is safe on CPU-only CI. Run the whole conformance suite on a
GPU box with:

```bash
CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-runtime-ep-cuda --features cuda \
    --test cuda_conformance_gpu -- --nocapture
```

The no-GPU audits alone (fast, deterministic, CI-friendly):

```bash
cargo test -p onnx-runtime-ep-cuda --features cuda --test cuda_conformance_gpu \
    -- --skip conformance_sweep_matches_cpu
```

There is intentionally **no new GPU GitHub Actions workflow**: hosted CI runners
have no GPU, so the sweep would only ever skip there. The no-GPU audits provide
the enforcement in CI; the on-device parity sweep is run on a GPU host (e.g. an
H100/H200 box) as documented above.
