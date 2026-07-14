//! # `onnx-runtime-ep-cuda`
//!
//! The CUDA execution provider for the ORT 2.0 runtime (`docs/ORT2.md` §15 and
//! §56 Phase 2). It implements [`onnx_runtime_ep_api::ExecutionProvider`] on top
//! of [`cudarc`] (driver + cuBLASLt), mirroring the structure of the CPU EP.
//!
//! ## Phase 2a scope — cuBLASLt GEMM + SDPA/GQA attention baseline
//!
//! This slice wires the foundation (device context, stream, allocator, H2D/D2H/
//! D2D copies), **standard GEMM** (`MatMul`) via `cudarc::cublaslt`, and the
//! **scaled-dot-product / grouped-query attention** baseline (`Attention`) built
//! from cuBLAS batched GEMMs around a runtime-compiled (NVRTC) fused softmax —
//! the §13.3 `Kernel` binding a cuDNN-fused SDPA / FlashAttention-3 shim drops in
//! behind later. Other design kernels are staged into subsequent slices:
//!
//! * **Phase 2b:** cuDNN-fused SDPA / FlashAttention-3 (behind the same
//!   attention binding), paged-KV (§13.4), custom fused kernels (LayerNorm/
//!   RMSNorm, residual+norm, RoPE) via CuTe/`extern "C"`, and FP8 GEMM.
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
pub mod error;
pub mod kernels;
pub mod provider;
pub mod runtime;

pub use kernels::attention::AttentionKernel;
pub use kernels::{build_cuda_registry, CUDA_PHASE2A_OPS};
pub use provider::CudaExecutionProvider;
pub use runtime::CudaRuntime;
