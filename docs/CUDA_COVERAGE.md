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

### Elementwise — unary / activations

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Relu`, `Sqrt`, `Erf`, `Tanh`, `Sigmoid`, `Gelu` | standard / `com.microsoft` | ✅ | **NVRTC-custom** | f32/f16/bf16; half storage widens to f32 compute and narrows once on store (`elementwise.rs`). |
| `Abs`, `Neg`, `Reciprocal`, `Exp`, `Log`, `Sign`, `Floor`, `Ceil`, `Round`, `Sin`, `Cos`, `Softplus` | `` | ✅ | **NVRTC-custom** | f32/f16/bf16 with CPU-matched formulas (`pointwise.rs`); `Round` uses ties-to-even and `Sign` preserves NaN. |
| `LeakyRelu`, `Elu`, `HardSigmoid`, `Clip`, `Softsign`, `Selu` | `` | ✅ | **NVRTC-custom** | Attribute/input-driven f32/f16/bf16 activations (`activations.rs`), computed in f32 for half storage. |

### Elementwise — logical / comparison

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Not` | `` | ✅ | **NVRTC-custom** | bool→bool, non-zero byte = true, canonical `1`/`0` out (matches CPU `logical.rs`). |
| `And`, `Or`, `Xor` | `` | ✅ | **NVRTC-custom** | bool operands → bool with NumPy right-aligned broadcasting and canonical `1`/`0` output. |
| `Equal`, `Greater`, `Less`, `GreaterOrEqual`, `LessOrEqual` | `` | ✅ | **NVRTC-custom** | f32 operands → bool with NumPy right-aligned broadcasting and ONNX comparison semantics. |

### Elementwise — binary

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max` | `` | ✅ | **NVRTC-custom** | f32/f16/bf16 with NumPy right-aligned broadcasting. Host-computed output shape plus zero-stride metadata drives one generic device index walk; half arithmetic computes in f32. |

### Normalization & softmax

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Softmax` (v1 & v13) | `` | ✅ | **cuDNN** `cudnnSoftmaxForward` | `ACCURATE` algorithm, f32/f16/bf16. Legacy coerce-to-2D uses INSTANCE mode; opset ≥ 13 uses a 4-D channel view for exact single-axis semantics. Falls back to the prior NVRTC kernel for f32 when cuDNN is unavailable. |
| `LayerNormalization` | `` / `com.microsoft` | ✅ | **NVRTC-custom** (fused) | Mean/var + normalize + affine in **one** pass over one HBM read — beats a cuDNN reduce + separate pointwise affine. Population stats, optional `Mean`/`InvStdDev` outputs, arbitrary `axis` (`normalization.rs`). f32. |
| `SkipLayerNormalization` | `com.microsoft` | ✅ | **NVRTC-custom** (fused) | `LayerNorm(input + skip + bias)·γ + β` — the residual add is fused into the norm, saving a whole tensor round-trip. Optional `beta`/`bias` inputs, optional `mean`/`inv_std`/`input_skip_bias_sum` outputs (`normalization.rs`). f32. |
| `SkipSimplifiedLayerNormalization` | `com.microsoft` | ✅ | **NVRTC-custom** (fused) | `RMSNorm(input + skip + bias)·γ` with no mean subtraction. Right-aligned broadcast `skip`, optional `bias`, and optional mean/inverse-RMS/residual-sum outputs (`normalization.rs`). f32. |
| `RMSNormalization` / `SimplifiedLayerNormalization` | `` / `com.microsoft` | ✅ | **NVRTC-custom** (fused) | Root-mean-square scale, no mean subtraction (LLaMA-family norm). Optional `InvStdDev` output, arbitrary `axis` (`normalization.rs`). f32. |
| `ReduceMean` | `` | ✅ | **cuDNN** `cudnnReduceTensor` | See reductions below. |

### Reductions

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `ReduceSum` | `` | ✅ | **cuDNN** `cudnnReduceTensor` (ADD) | Arbitrary axes (attribute or opset-13+ input), `keepdims`, `noop_with_empty_axes`, negative axes; f32/f16/bf16. RAII workspace, no indices. Falls back to the prior NVRTC f32 kernel when cuDNN is unavailable. |
| `ReduceMean` | `` | ✅ | **cuDNN** `cudnnReduceTensor` (AVG) | Same shape/axis handling and fallback as `ReduceSum`. |
| `ReduceMax` | `` | ✅ | **NVRTC block reduction** (cub-class) | As above; NaN-propagating (numpy / CPU-EP semantics). |
| `ReduceMin` | `` | ✅ | **NVRTC block reduction** (cub-class) | As above; NaN-propagating. |

> **Why NVRTC block reduction, not cub?** cub's `DeviceSegmentedReduce` is the
> vendor primitive, and our kernel matches its shape (one block per output
> element, cooperative shared-memory tree reduce over that element's group). We
> keep it as an NVRTC kernel so the crate stays toolkit-free (no `nvcc`/`build.rs`);
> the offset tables let it handle any axis set / rank without special-casing.

### Attention

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Attention` | `com.microsoft` | ✅ | **cuBLAS GEMM + NVRTC softmax** | SDPA/GQA baseline (`attention.rs`); §13.3 binding. |
| `FusedAttention` | `com.microsoft` | 🔬 | **cuDNN SDPA / FlashAttention-3** | Fused flash-attention behind the same binding — the top perf item. |

### Shape / data-movement / misc

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Cast` | `` | ✅ | **NVRTC-custom** | Element-wise dtype conversion; f32/f64/f16/bf16/int8-64/uint8-64/bool, ONNX saturating float→int. Two NVRTC modules keep f16/bf16 (which need NVRTC's built-in `cuda_fp16.h`/`cuda_bf16.h`) out of the common integer/f32 path (`cast.rs`). |
| `CastLike` | `` | ✅ | **NVRTC-custom** | Same kernel as `Cast`; target dtype taken from the output tensor. |
| `Identity` | `` | ⏳ | **memcpy** (D2D) | Straight device copy; dtype-agnostic. |
| `Concat` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic byte copy, arbitrary/negative axis, multiple inputs. |
| `Reshape`, `Squeeze`, `Unsqueeze` | `` | ✅ | **memcpy** | Dtype-agnostic D2D copy into the executor's pre-shaped output; modern axes inputs and legacy attributes are accepted. |
| `Transpose` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic indexed byte copy; explicit permutation or default axis reversal. |
| `Gather` | `` | ✅ | **NVRTC-custom** | Axis-parametric contiguous indexed copy; Int32/Int64 indices, negative index wrap, arbitrary index rank. |
| `Shape` | `` | ✅ | **host + H2D** | Computes the metadata-only Int64 shape vector on host, including opset-15 `start`/`end`, then uploads it. |
| `Expand` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic broadcast copy sharing the binary-elementwise zero-stride indexing infrastructure. |
| `Slice` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic strided/stepped copy with opset-10 input-driven ranges, negative axes, and negative steps. |
| `Split` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic byte copy; split input, legacy attribute, even split, negative axis, and opset-18 `num_outputs`. |
| `Tile` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic repeated indexed copy across arbitrary axes. |
| `Where` | `` | ✅ | **NVRTC-custom** | Dtype-agnostic branch selection with right-aligned broadcasting across condition, x, and y. |
| `Constant` | `` | ✅ | **host + H2D** | Uploads `value` tensors and numeric `value_*` attribute forms to the device. |

## Source-derived coverage audit (2026-07-15)

This snapshot is derived directly from `build_cpu_registry`,
`build_cuda_registry`, and `CUDA_COVERED_OPS`, rather than the historical wave
counts:

| Measure | Count |
|---------|------:|
| CPU registry `(domain, op_type)` pairs | **103** |
| CPU standard-domain (`ai.onnx`) op types | **93** |
| CUDA registry `(domain, op_type)` pairs | **66** |
| CUDA advertised op names | **65** |
| CPU pairs implemented by CUDA in the same domain | **54 / 103** |
| CPU standard-domain op types implemented by CUDA | **46 / 93** |

The **46 shared `ai.onnx` ops** are: `Abs`, `Add`, `AveragePool`, `Cast`, `CastLike`,
`Ceil`, `Clip`, `Constant`, `Cos`, `Div`, `Elu`, `Equal`, `Erf`, `Exp`, `Floor`,
`Gather`, `Gemm`, `HardSigmoid`, `LayerNormalization`, `LeakyRelu`, `Log`, `MatMul`,
`Max`, `Min`, `MaxPool`, `Mul`, `Neg`, `Not`, `Pow`, `RMSNormalization`, `Reciprocal`,
`ReduceMax`, `ReduceMean`, `ReduceMin`, `ReduceSum`, `Relu`, `Round`, `Shape`,
`Sigmoid`, `Sign`, `Sin`, `Softmax`, `Softplus`, `Sqrt`, `Sub`, and `Tanh`.

The **47 CPU `ai.onnx` gaps** are: `Acos`, `Acosh`, `ArgMax`, `ArgMin`, `Asin`,
`Asinh`, `Atan`, `Atanh`, `Attention`, `Concat`, `ConstantOfShape`, `Cosh`, `CumSum`,
`DequantizeLinear`, `DynamicQuantizeLinear`, `Expand`, `Flatten`, `GatherElements`,
`GatherND`, `Gelu`, `GlobalAveragePool`, `GlobalMaxPool`, `Identity`,
`LogSoftmax`, `Mean`, `NonZero`, `Pad`, `QuantizeLinear`, `Range`,
`ReduceL2`, `ReduceProd`, `ReduceSumSquare`, `Reshape`, `RotaryEmbedding`,
`Sinh`, `Size`, `Slice`, `Split`, `Squeeze`, `Sum`, `Swish`, `Tan`, `Tile`, `TopK`,
`Transpose`, `Unsqueeze`, and `Where`.

For `com.microsoft`, CUDA matches eight CPU pairs (`FusedGemm`,
`FusedMatMulBias`, `Gelu`, `LayerNormalization`, `MatMulNBits`,
`SimplifiedLayerNormalization`, `SkipLayerNormalization`,
`SkipSimplifiedLayerNormalization`); CPU-only gaps are `BiasGelu`, `FastGelu`,
`FusedAttention`, and `QuickGelu`. CUDA additionally exposes
`com.microsoft::Attention`. CUDA standard-domain extras not currently registered
by the CPU EP are `And`, `Conv`, `Greater`, `GreaterOrEqual`, `Less`,
`LessOrEqual`, `Or`, `Selu`, `Softsign`, and `Xor`.

### Library mapping for the remaining CPU gaps

| Backend | CPU-covered gaps mapped here | Rationale |
|---------|------------------------------|-----------|
| **cuBLASLt** | `BiasGelu`/`FastGelu`/`QuickGelu` where expressible as an epilogue | GEMM+bias/activation belongs in the matrix multiply epilogue. |
| **cuDNN** | `GlobalAveragePool`, `GlobalMaxPool`, `LogSoftmax`, `ReduceL2`, `ReduceProd`, `ReduceSumSquare` | Vendor-tuned pooling, normalization/softmax, and reduction primitives. |
| **CUTLASS / cuDNN SDPA** | standard `Attention`, `FusedAttention` | Flash/SDPA implementation avoids materialising the O(S²) score tensor. |
| **cub/thrust via NVRTC (CCCL headers)** | `ArgMax`, `ArgMin`, `TopK`, `CumSum`, `NonZero` | Scan/select/sort/reduction primitives; cudarc has no dlopen-able cub/thrust API. |
| **NVRTC-custom** | remaining unary math (`Acos`…`Tan`, `Swish`), quantize/dequantize, `Where`, `RotaryEmbedding`, indexed/strided movement (`Gather*`, `Slice`, `Tile`, `Expand`, `Transpose`, `Concat`, `Pad`, `Split`, `Range`) | Pointwise or index-transform work with no suitable runtime library; RoPE is a justified fusion kernel. |
| **view / memcpy / host** | `Identity`, `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Shape`, `Size`, `Constant`, `ConstantOfShape` | Metadata-only views, raw D2D copies, or small host-generated tensors. |

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

---

## Custom-kernel candidates (with WHY)

Ops that justify a **custom fused NVRTC / CUTLASS kernel** — either no library
covers them, or fusion measurably beats calling a library op-by-op. Ordered by
expected impact for transformer inference.

1. **`FusedAttention` → FlashAttention-3 / cuDNN SDPA** *(highest impact)* —
   the current baseline materialises the full `[B,H,Sq,Sk]` score matrix
   (O(S²) memory + two GEMM round-trips through HBM). Flash-attention keeps
   scores in SRAM and is the single biggest latency/throughput win. Drop in
   behind the existing §13.3 `AttentionKernel` binding (`supports_strided_input`
   / `cuda_graph_compatible` already advertise the target shape).
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
