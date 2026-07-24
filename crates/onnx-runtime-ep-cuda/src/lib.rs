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
//! * **GEMM family** — `MatMul`, `Gemm`, `com.microsoft::FusedMatMulBias`, and
//!   `com.microsoft::FusedGemm` via cuBLASLt. The fused ops use native
//!   `BIAS`, `RELU_BIAS`, and `GELU_BIAS` epilogues.
//! * **Elementwise** — unary activations (`Relu`, `Sqrt`, `Erf`, `Tanh`,
//!   `Sigmoid`, and `com.microsoft` `Gelu`) and equal-shape binary ops (`Add`,
//!   `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`) via runtime-compiled (NVRTC) f32
//!   pointwise kernels — kept as our own kernels so they can later fuse into a
//!   GEMM epilogue or an elementwise chain (RULES.md #4).
//! * **Attention** — tiled online-softmax prefill (`Attention` and
//!   `GroupQueryAttention`, `com.microsoft`) compiled by NVRTC, with an f16
//!   tensor-core specialization and retained decode/unsupported-shape baselines.
//!
//! The full op → backend mapping matrix, remaining coverage, and the
//! prioritised custom-kernel candidate list live in `docs/CUDA_COVERAGE.md`.
//! Roadmap ops not yet wired (cuDNN softmax/norm, cub reductions, data-movement,
//! FP8 and remaining fusion-node lowering return an actionable
//! [`onnx_runtime_ep_api::EpError`].
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
pub mod capture;
pub mod cudnn;
mod dynamic_library;
pub mod error;
mod graph;
pub mod kernels;
mod optimizer;
pub mod provider;
pub mod runtime;

pub use capture::{require_subgraph_graph_capturable, subgraph_graph_capturable};
pub use kernels::attention::AttentionKernel;
pub use kernels::csa_checkpoint::{
    CsaAttentionMode, CsaCheckpoint, CsaCheckpointJournal, CsaCursors, CsaLayerMetrics, CsaMetrics,
};
pub use kernels::gather::GATHER_CAPTURE_ERROR_INDEX;
pub use kernels::gather_block_quantized::GATHER_BLOCK_QUANTIZED_CAPTURE_ERROR_INDEX;
pub use kernels::group_query_attention::{
    GQA_CAPTURE_ERROR_PAST_CAPACITY, GQA_CAPTURE_ERROR_PAST_NEGATIVE, GQA_CAPTURE_ERROR_POSITION,
    GQA_CAPTURE_ERROR_PRESENT_CAPACITY, GQA_CAPTURE_ERROR_QUERY_NEGATIVE,
    GQA_CAPTURE_ERROR_TOTAL_OVERFLOW, GroupQueryAttentionBackend, GroupQueryAttentionKernel,
    gqa_capture_error_description,
};
pub use kernels::index_share::INDEX_SHARE_CAPTURE_ERROR_INDEX;
pub use kernels::indexing::SCATTER_CAPTURE_ERROR_INDEX;
pub use kernels::reduce::REDUCE_CAPTURE_ERROR_AXES;
pub use kernels::{CUDA_COVERED_OPS, build_cuda_registry, build_cuda_registry_with_metrics};
pub use provider::CudaExecutionProvider;

/// Number of additional u32 words required by the CUDA device argmax result buffer.
pub fn device_argmax_scratch_words(elements: usize) -> usize {
    kernels::device_argmax::scratch_words(elements)
}
pub use runtime::{CudaAllocationCounts, CudaRuntime};
