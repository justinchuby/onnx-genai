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
| **cuDNN** | `cudarc` `cudnn` feature (to add) | conv, pooling, softmax, activations, batch/instance/layer norm, LRN. Vendor-tuned, PyTorch's own backend. |
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
| `MatMul` | `` | ✅ | **cuBLASLt** | 2-D + equal-batch 3-D, f32/f16/bf16, true-fp32 accum (`matmul.rs`). |
| `Gemm` | `` | ✅ | **cuBLASLt** + NVRTC bias | `Y=α·A'·B'+β·C`, transA/transB, α/β; fused NVRTC `β·C` broadcast-bias epilogue (`gemm.rs`). f32. |
| `FusedMatMulBias` | `com.microsoft` | ⏳ | **cuBLASLt** epilogue | `CUBLASLT_EPILOGUE_BIAS` — bias add fused into the GEMM (no extra pass). |
| `FusedGemm` | `com.microsoft` | ⏳ | **cuBLASLt** epilogue | `EPILOGUE_RELU_BIAS`/`GELU_BIAS` — activation+bias fused in-GEMM. |

### Elementwise — unary / activations

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Relu` | `` | ✅ | **NVRTC-custom** | f32 pointwise (`elementwise.rs`). NVRTC (not cuDNN) so it can later fuse into a GEMM epilogue. |
| `Sqrt` | `` | ✅ | **NVRTC-custom** | f32 pointwise. |
| `Erf` | `` | ✅ | **NVRTC-custom** | f32 pointwise (`erff` intrinsic). |
| `Tanh` | `` | ✅ | **NVRTC-custom** | f32 pointwise. |
| `Sigmoid` | `` | ✅ | **NVRTC-custom** | f32 pointwise (bonus; not in CPU set yet). |
| `Gelu` | `com.microsoft` | ✅ | **NVRTC-custom** | exact (erf) GELU, f32. Prime fusion target (GELU-bias GEMM epilogue). |

### Elementwise — binary

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Add` | `` | ✅ | **NVRTC-custom** | f32, **equal-shape**. Broadcasting ⏳ (actionable error today). |
| `Sub` | `` | ✅ | **NVRTC-custom** | f32, equal-shape. |
| `Mul` | `` | ✅ | **NVRTC-custom** | f32, equal-shape. |
| `Div` | `` | ✅ | **NVRTC-custom** | f32, equal-shape. |
| `Pow` | `` | ✅ | **NVRTC-custom** | f32, equal-shape (`powf`). |
| `Min` | `` | ✅ | **NVRTC-custom** | f32, equal-shape. |
| `Max` | `` | ✅ | **NVRTC-custom** | f32, equal-shape. |

### Normalization & softmax

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Softmax` (v1 & v13) | `` | ⏳ | **cuDNN** (`cudnnSoftmaxForward`) *or* NVRTC | Per-axis, numerically stable. A reusable NVRTC softmax already exists inside `attention.rs` — extract it as a standalone `Softmax` kernel; cuDNN for large rows. |
| `LayerNormalization` | `` / `com.microsoft` | 🔬 | **NVRTC-custom** (fused) | Mean/var + affine in one pass beats a cuDNN call + separate affine; classic fusion win. |
| `ReduceMean` | `` | ⏳ | **cub** (`DeviceReduce`) / NVRTC | Multi-axis, keepdims. cub segmented reduce for the reduced axis. |

### Attention

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Attention` | `com.microsoft` | ✅ | **cuBLAS GEMM + NVRTC softmax** | SDPA/GQA baseline (`attention.rs`); §13.3 binding. |
| `FusedAttention` | `com.microsoft` | 🔬 | **cuDNN SDPA / FlashAttention-3** | Fused flash-attention behind the same binding — the top perf item. |

### Shape / data-movement / misc

| Op | Domain | Status | Backend | Notes |
|----|--------|--------|---------|-------|
| `Cast` | `` | ⏳ | **NVRTC-custom** | dtype conversion pointwise kernel (fixed-width numeric + bf16/f16). |
| `Identity` | `` | ⏳ | **memcpy** (D2D) | Straight device copy; dtype-agnostic. |
| `Reshape` | `` | ⏳ | **view rewrite** | Metadata-only when contiguous; else materialise. |
| `Transpose` | `` | ⏳ | **NVRTC-custom** / cuBLAS | Tiled-transpose kernel (or fold into a consumer's GEMM `op`). |
| `Gather` | `` | ⏳ | **NVRTC-custom** | Indexed copy (axis-parametric gather kernel). |
| `Shape` | `` | ⏳ | **host** | Emits a shape tensor; no device compute. |
| `Unsqueeze` | `` | ⏳ | **view rewrite** | Metadata-only. |
| `Expand` | `` | ⏳ | **NVRTC-custom** | Broadcast copy (shares the broadcasting index math with binary-elementwise-broadcast). |
| `Slice` | `` | ⏳ | **NVRTC-custom** | Strided/stepped copy (opset-10 input-driven ranges). |
| `Constant` | `` | ⏳ | **host + H2D** | Upload the constant to device once. |

**Score:** reference set (unique op types) = **31**. CUDA **before** this slice =
**2** (`MatMul`, `Attention`). CUDA **after** = **16**
(`MatMul`, `Gemm`, `Relu`, `Sqrt`, `Erf`, `Tanh`, `Sigmoid`, `Gelu`, `Add`,
`Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`, `Attention`).

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
6. **Broadcasting elementwise** — extend the binary kernels with NumPy
   broadcast index math (shared with `Expand`). Removes today's "materialise the
   smaller operand" restriction. Library alternative (cuDNN OpTensor) is
   clunkier and less fusable.

Everything else in the matrix (`ReduceMean`→cub, `Softmax`→cuDNN, `Cast`,
data-movement) is a **straight library/primitive mapping**, not a custom-kernel
candidate.

---

## Runtime / build notes

- **Build is toolkit-free.** `cargo build -p onnx-runtime-ep-cuda` compiles with
  no CUDA toolkit because `cudarc` uses `dynamic-loading`; the driver, cuBLASLt,
  and NVRTC are `dlopen`'d at run time. Adding the `cudnn` feature for the ⏳
  norm/softmax/conv rows preserves this (cuDNN is dlopen'd too).
- **Adding cuDNN:** enable cudarc's `cudnn` feature in
  `crates/onnx-runtime-ep-cuda/Cargo.toml` and add a `cudnn` handle to
  `CudaRuntime` alongside the cuBLASLt handle. Confirm the offline build still
  completes (it will — dynamic-loading).
- **Runtime execution requires the libraries on the loader path.** A host with
  only `libcuda` (driver) but **without** `libcublasLt` / `libcudnn` can *build*
  and can run *pure-driver* code, but cuBLASLt/cuDNN ops error/skip until those
  libs are installed. Every such failure is an actionable `EpError` (RULES.md #1)
  naming the missing library and how to fix it.
