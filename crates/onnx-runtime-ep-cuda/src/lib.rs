//! # `onnx-runtime-ep-cuda`
//!
//! The CUDA execution provider for the ORT 2.0 runtime (`docs/architecture/ORT2.md` §15 and
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
//! prioritised custom-kernel candidate list live in `docs/execution/CUDA_COVERAGE.md`.
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

#[cfg(not(any(
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
)))]
compile_error!(
    "onnx-genai CUDA build: no CUDA version selected. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000."
);

#[cfg(any(
    all(feature = "cuda-12060", feature = "cuda-12080"),
    all(feature = "cuda-12060", feature = "cuda-12090"),
    all(feature = "cuda-12060", feature = "cuda-13000"),
    all(feature = "cuda-12080", feature = "cuda-12090"),
    all(feature = "cuda-12080", feature = "cuda-13000"),
    all(feature = "cuda-12090", feature = "cuda-13000")
))]
compile_error!(
    "onnx-genai CUDA build: multiple CUDA versions selected; cudarc bindings cannot compile with more than one. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000 (and set default-features = false on inter-crate deps if you override the default)."
);

pub mod blas;
pub mod capture;
pub mod cudnn;
pub mod deferred_release;
mod dynamic_library;
pub mod error;
mod graph;
pub mod kernels;
mod optimizer;
pub mod pinned_pool;
pub mod provider;
pub mod runtime;
mod trace;
// Device memory moved to `onnx-runtime-cuda-memory`: an execution provider is
// about operators, and where the memory came from is a separate question that
// a caller with no interest in kernels may need to answer. Re-exported so the
// move is not a breaking change for anyone already reaching for these paths.
pub use onnx_runtime_cuda_memory::{virtual_memory, vmm_allocator};
pub mod weight_paging;

pub use capture::{require_subgraph_graph_capturable, subgraph_graph_capturable};
pub use dynamic_library::set_wheel_search_paths;
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
pub use kernels::{
    CUDA_COVERED_OPS, CudaOpDescriptor, build_cuda_registry, build_cuda_registry_descriptors,
    build_cuda_registry_with_metrics, cuda_supported_dtypes_for_op,
};
pub use pinned_pool::{PinnedStagingPool, global_pinned_alloc_calls, global_pinned_reuses};
pub use provider::{
    CudaExecutionProvider, DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES, dynamic_kv_weight_lending_enabled,
};
pub use weight_paging::{
    CudaResidencyStats, CudaWeightPage, CudaWeightPager, CudaWeightResidency, DeviceOffloadPolicy,
    EvictOrderProbe, GlobalOffloadStats, WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV,
    WEIGHT_OFFLOAD_BYTE_AWARE_ENV, WEIGHT_OFFLOAD_DEVICE_BYTES_ENV, WEIGHT_OFFLOAD_ENV,
    WEIGHT_OFFLOAD_EVICT_ORDER_ENV, WEIGHT_OFFLOAD_SCAN_RESISTANT_ENV,
    WEIGHT_OFFLOAD_ZERO_COPY_HYBRID_ENV, WeightKeyTrace, byte_aware_residency_from_env,
    evict_order_probe_from_env, global_offload_stats, pinned_hot_set, reset_global_offload_stats,
    weight_paging_key_trace, zero_copy_hybrid_from_env,
};

/// Number of additional u32 words the CUDA device argmax result buffer needs
/// beyond its `2 × batch` header words, for `batch` sequences of `elements`
/// logits each.
pub fn device_argmax_scratch_words(elements: usize, batch: usize) -> usize {
    kernels::device_argmax::scratch_words(elements, batch)
}
pub use runtime::{CudaAllocationCounts, CudaRuntime};
