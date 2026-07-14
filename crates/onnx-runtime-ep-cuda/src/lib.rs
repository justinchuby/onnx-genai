//! # `onnx-runtime-ep-cuda`
//!
//! The CUDA execution provider for the ORT 2.0 runtime (`docs/ORT2.md` §15 and
//! §56 Phase 2). It implements [`onnx_runtime_ep_api::ExecutionProvider`] on top
//! of [`cudarc`] (driver + cuBLASLt), mirroring the structure of the CPU EP.
//!
//! ## Phase 2a scope — cuBLASLt GEMM only
//!
//! This is the **first CUDA slice**: it wires the foundation (device context,
//! stream, allocator, H2D/D2H/D2D copies) and **standard GEMM** (`MatMul`) via
//! `cudarc::cublaslt`. That is deliberately all — the design's custom fused
//! kernels are staged into later slices:
//!
//! * **Phase 2b:** custom fused kernels (LayerNorm/RMSNorm, residual+norm,
//!   RoPE, softmax) via CuTe/`extern "C"`, NVRTC elementwise, attention
//!   (cuDNN SDPA → FlashAttention-3), and FP8 GEMM.
//!
//! No `.cu` sources and no `nvcc`/`build.rs` compile step exist in this crate:
//! `cudarc` is used in its **dynamic-loading** configuration, so `cargo build`
//! needs no CUDA toolkit — the driver and cuBLASLt are `dlopen`'d at runtime.
//!
//! ## Model-agnostic hard rule (§15.1)
//!
//! The GEMM kernel is shape-driven and dtype-parameterized; there are **no**
//! hardcoded `num_heads` / `head_dim` / model constants anywhere in the crate.
//!
//! ## Error discipline (KEY PROJECT RULE)
//!
//! Every unsupported op / dtype / rank / device condition returns an actionable
//! [`onnx_runtime_ep_api::EpError`] stating *what* is unsupported and that it is
//! *CUDA-EP Phase-2a* scope. There are no bare panics on the dispatch path.
//!
//! ## `unsafe`
//!
//! `unsafe` is confined to the FFI boundary: raw device alloc/free/copy in
//! [`runtime`] and the cuBLASLt descriptor/matmul calls in [`blas`]. Each site
//! is isolated and `SAFETY`-documented. CUDA device pointers travel as opaque
//! addresses in [`onnx_runtime_ep_api::DeviceBuffer`] and are never
//! dereferenced on the host.

pub mod blas;
pub mod error;
pub mod kernels;
pub mod provider;
pub mod runtime;

pub use kernels::{build_cuda_registry, CUDA_PHASE2A_OPS};
pub use provider::CudaExecutionProvider;
pub use runtime::CudaRuntime;
