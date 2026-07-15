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
| `FusedMatMulBias` | `com.microsoft` | ⏳ | **cuBLASLt** epilogue | `CUBLASLT_EPILOGUE_BIAS` — bias add fused into the GEMM (no extra pass). |
| `FusedGemm` | `com.microsoft` | ⏳ | **cuBLASLt** epilogue | `EPILOGUE_RELU_BIAS`/`GELU_BIAS` — activation+bias fused in-GEMM. |

### Convolution

| Op | Domain | Status | Backend | Notes / justification |
|----|--------|--------|---------|-----------------------|
| `Conv` | `` | ✅ | **cuDNN** | 2-D dense NCHW f32/f16/bf16; strides, dilation, groups, symmetric explicit padding, `VALID`, symmetric `SAME_UPPER`/`SAME_LOWER`, and optional fused channel bias. Asymmetric padding is rejected explicitly. Uses cuDNN v7 forward-algorithm heuristics and queried workspace. |

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
| `And` | `` | ✅ | **NVRTC-custom** | bool operands → bool, **equal-shape**. Broadcasting ⏳. |
| `Or` | `` | ✅ | **NVRTC-custom** | bool operands → bool, equal-shape. |
| `Xor` | `` | ✅ | **NVRTC-custom** | bool operands → bool, equal-shape. |
| `Equal` | `` | ✅ | **NVRTC-custom** | f32 operands → **bool**, equal-shape. ONNX comparison semantics. |
| `Greater` | `` | ✅ | **NVRTC-custom** | f32 operands → bool, equal-shape. |
| `Less` | `` | ✅ | **NVRTC-custom** | f32 operands → bool, equal-shape. |
| `GreaterOrEqual` | `` | ✅ | **NVRTC-custom** | f32 operands → bool, equal-shape. |
| `LessOrEqual` | `` | ✅ | **NVRTC-custom** | f32 operands → bool, equal-shape. |

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
| `Reshape` | `` | ⏳ | **view rewrite** | Metadata-only when contiguous; else materialise. |
| `Transpose` | `` | ⏳ | **NVRTC-custom** / cuBLAS | Tiled-transpose kernel (or fold into a consumer's GEMM `op`). |
| `Gather` | `` | ⏳ | **NVRTC-custom** | Indexed copy (axis-parametric gather kernel). |
| `Shape` | `` | ⏳ | **host** | Emits a shape tensor; no device compute. |
| `Unsqueeze` | `` | ⏳ | **view rewrite** | Metadata-only. |
| `Expand` | `` | ⏳ | **NVRTC-custom** | Broadcast copy (shares the broadcasting index math with binary-elementwise-broadcast). |
| `Slice` | `` | ⏳ | **NVRTC-custom** | Strided/stepped copy (opset-10 input-driven ranges). |
| `Constant` | `` | ⏳ | **host + H2D** | Upload the constant to device once. |

## Source-derived coverage audit (2026-07-15)

This snapshot is derived directly from `build_cpu_registry`,
`build_cuda_registry`, and `CUDA_COVERED_OPS`, rather than the historical wave
counts:

| Measure | Count |
|---------|------:|
| CPU registry `(domain, op_type)` pairs | **103** |
| CPU standard-domain (`ai.onnx`) op types | **93** |
| CUDA registry `(domain, op_type)` pairs | **56** |
| CUDA advertised op names | **55** |
| CPU pairs implemented by CUDA in the same domain | **45 / 103** |
| CPU standard-domain op types implemented by CUDA | **41 / 93** |

The **41 shared `ai.onnx` ops** are: `Abs`, `Add`, `Cast`, `CastLike`, `Ceil`,
`Clip`, `Cos`, `Div`, `Elu`, `Equal`, `Erf`, `Exp`, `Floor`, `Gemm`,
`HardSigmoid`, `LayerNormalization`, `LeakyRelu`, `Log`, `MatMul`, `Max`, `Min`,
`Mul`, `Neg`, `Not`, `Pow`, `RMSNormalization`, `Reciprocal`, `ReduceMax`,
`ReduceMean`, `ReduceMin`, `ReduceSum`, `Relu`, `Round`, `Sigmoid`, `Sign`,
`Sin`, `Softmax`, `Softplus`, `Sqrt`, `Sub`, and `Tanh`.

The **52 CPU `ai.onnx` gaps** are: `Acos`, `Acosh`, `ArgMax`, `ArgMin`, `Asin`,
`Asinh`, `Atan`, `Atanh`, `Attention`, `AveragePool`, `Concat`, `Constant`,
`ConstantOfShape`, `Cosh`, `CumSum`, `DequantizeLinear`,
`DynamicQuantizeLinear`, `Expand`, `Flatten`, `Gather`, `GatherElements`,
`GatherND`, `Gelu`, `GlobalAveragePool`, `GlobalMaxPool`, `Identity`,
`LogSoftmax`, `MaxPool`, `Mean`, `NonZero`, `Pad`, `QuantizeLinear`, `Range`,
`ReduceL2`, `ReduceProd`, `ReduceSumSquare`, `Reshape`, `RotaryEmbedding`,
`Shape`, `Sinh`, `Size`, `Slice`, `Split`, `Squeeze`, `Sum`, `Swish`, `Tan`,
`Tile`, `TopK`, `Transpose`, `Unsqueeze`, and `Where`.

For `com.microsoft`, CUDA matches four CPU pairs (`Gelu`,
`LayerNormalization`, `SimplifiedLayerNormalization`, `SkipLayerNormalization`);
CPU-only gaps are `BiasGelu`, `FastGelu`, `FusedAttention`, `FusedGemm`,
`FusedMatMulBias`, and `QuickGelu`. CUDA additionally exposes
`com.microsoft::Attention`. CUDA standard-domain extras not currently registered
by the CPU EP are `And`, `Conv`, `Greater`, `GreaterOrEqual`, `Less`,
`LessOrEqual`, `Or`, `Selu`, `Softsign`, and `Xor`.

### Library mapping for the remaining CPU gaps

| Backend | CPU-covered gaps mapped here | Rationale |
|---------|------------------------------|-----------|
| **cuBLASLt** | `FusedMatMulBias`, `FusedGemm`; `BiasGelu`/`FastGelu`/`QuickGelu` where expressible as an epilogue | GEMM+bias/activation belongs in the matrix multiply epilogue. |
| **cuDNN** | `AveragePool`, `MaxPool`, `GlobalAveragePool`, `GlobalMaxPool`, `LogSoftmax`, `ReduceL2`, `ReduceProd`, `ReduceSumSquare` | Vendor-tuned pooling, normalization/softmax, and reduction primitives. |
| **CUTLASS / cuDNN SDPA** | standard `Attention`, `FusedAttention` | Flash/SDPA implementation avoids materialising the O(S²) score tensor. |
| **cub/thrust via NVRTC (CCCL headers)** | `ArgMax`, `ArgMin`, `TopK`, `CumSum`, `NonZero` | Scan/select/sort/reduction primitives; cudarc has no dlopen-able cub/thrust API. |
| **NVRTC-custom** | remaining unary math (`Acos`…`Tan`, `Swish`), quantize/dequantize, `Where`, `RotaryEmbedding`, indexed/strided movement (`Gather*`, `Slice`, `Tile`, `Expand`, `Transpose`, `Concat`, `Pad`, `Split`, `Range`) | Pointwise or index-transform work with no suitable runtime library; RoPE is a justified fusion kernel. |
| **view / memcpy / host** | `Identity`, `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Shape`, `Size`, `Constant`, `ConstantOfShape` | Metadata-only views, raw D2D copies, or small host-generated tensors. |

Wave 4 raises the advertised CUDA set from **48 to 54** op names. Its six
activations are GPU-validated against independent CPU formulas on the local
CUDA 13.0 host; broader performance validation remains separate.

The cuDNN Conv pass raises the advertised set to **55** op names and is
GPU-validated for padded f32, grouped/strided f32, and padded f16 convolution.

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
3. **`FusedGemm` / `FusedMatMulBias` (cuBLASLt epilogue)** — *not a hand-written
   kernel*, but a library-fusion win: use `CUBLASLT_EPILOGUE_GELU_BIAS` /
   `RELU_BIAS` / `BIAS` so the activation+bias run inside the GEMM, eliminating
   the separate elementwise pass our current `Gemm`+`Gelu` chain does.
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
