//! # `onnx-runtime-ep-cuda`
//!
//! The CUDA execution provider for the ORT 2.0 runtime (`docs/ORT2.md` §15 and
//! §56 Phase 2). It implements [`onnx_runtime_ep_api::ExecutionProvider`] on top
//! of [`cudarc`] (driver + cuBLASLt), mirroring the structure of the CPU EP.
//!
//! ## Scope — cuBLASLt GEMM family + NVRTC elementwise + SDPA/GQA attention
//!
//! This EP wires the foundation (device context, stream, allocator, H2D/D2H/
//! D2D copies) and covers, keyed on `(op_type, domain)` via the shared
//! [`onnx_runtime_ep_api::OpRegistry`]:
//!
//! * **GEMM family** — `MatMul` and `Gemm` via `cudarc::cublaslt` (`Gemm` adds a
//!   fused NVRTC `β·C` broadcast-bias epilogue).
//! * **Elementwise** — unary activations (`Relu`, `Sqrt`, `Erf`, `Tanh`,
//!   `Sigmoid`, and `com.microsoft` `Gelu`) and equal-shape binary ops (`Add`,
//!   `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`) via runtime-compiled (NVRTC) f32
//!   pointwise kernels — kept as our own kernels so they can later fuse into a
//!   GEMM epilogue or an elementwise chain (RULES.md #4).
//! * **Attention** — the scaled-dot-product / grouped-query attention baseline
//!   (`Attention`, `com.microsoft`) built from cuBLAS batched GEMMs around a
//!   runtime-compiled fused softmax — the §13.3 `Kernel` binding a cuDNN-fused
//!   SDPA / FlashAttention-3 shim drops in behind later.
//!
//! The full op → backend mapping matrix, remaining coverage, and the
//! prioritised custom-kernel candidate list live in `docs/CUDA_COVERAGE.md`.
//! Roadmap ops not yet wired (cuDNN softmax/norm, cub reductions, data-movement,
//! FP8, FlashAttention-3) return an actionable [`onnx_runtime_ep_api::EpError`].
//!
//! No `.cu` sources and no `nvcc`/`build.rs` compile step exist in this crate:
//! `cudarc` is used in its **dynamic-loading** configuration, so `cargo build`
//! needs no CUDA toolkit — the driver, cuBLASLt, and NVRTC are `dlopen`'d at
//! runtime (the attention softmax is compiled from a CUDA-C string at runtime).
//!
//! ## Model-agnostic hard rule (§15.1)
//!
//! Kernels are shape-driven and dtype-parameterized; attention dims
//! (`num_heads`, `num_kv_heads`, `head_dim`, `causal`, `scale`) are runtime data
//! / node attributes. There are **no** hardcoded model constants anywhere.
//!
//! ## Error discipline (KEY PROJECT RULE)
//!
//! Every unsupported op / dtype / rank / device condition returns an actionable
//! [`onnx_runtime_ep_api::EpError`] stating *what* is unsupported and that it is
//! *CUDA-EP Phase-2a* scope. NVRTC compile failures surface the compiler log.
//! There are no bare panics on the dispatch path.
//!
//! ## `unsafe`
//!
//! `unsafe` is confined to the FFI boundary: raw device alloc/free/copy in
//! [`runtime`], the cuBLASLt descriptor/matmul calls in [`blas`], and the
//! per-head GEMM / NVRTC softmax launches in [`kernels::attention`]. Each site
//! is isolated and `SAFETY`-documented. CUDA device pointers travel as opaque
//! addresses in [`onnx_runtime_ep_api::DeviceBuffer`] and are never
//! dereferenced on the host.

pub mod blas;
pub mod cudnn;
pub mod error;
pub mod kernels;
pub mod provider;
pub mod runtime;

pub use kernels::attention::AttentionKernel;
pub use kernels::{build_cuda_registry, CUDA_COVERED_OPS};
pub use provider::CudaExecutionProvider;
pub use runtime::CudaRuntime;
