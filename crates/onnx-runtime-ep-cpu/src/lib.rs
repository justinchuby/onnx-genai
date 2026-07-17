//! # `onnx-runtime-ep-cpu`
//!
//! The CPU execution provider for the ORT 2.0 runtime (see `docs/ORT2.md` §4.4
//! and §54 Phase 1). It implements [`onnx_runtime_ep_api::ExecutionProvider`]
//! and hosts pure-Rust reference kernels for the Phase-1 op set (`MatMul`,
//! `Add`, `Relu`, `Reshape`, `Transpose`, `Gather`, `LayerNormalization`).
//!
//! ## Backends: correctness baseline + optional oneDNN
//!
//! The GEMM hot spot is served through [`backend::CpuBackend`] (`docs/ORT2.md`
//! §25.2). The **default** backend is a pure-Rust blocked, register-tiled,
//! rayon-parallelized f32 GEMM — the portable, offline correctness baseline that
//! compiles anywhere with no C++/FFI. The non-default `onednn` cargo feature
//! statically links oneDNN and routes the 2-D tile GEMM through `dnnl_sgemm`
//! ([`kernels::onednn`]). Every backend lives behind the
//! [`onnx_runtime_ep_api::Kernel`] trait, so neither the EP contract nor the
//! session observes which one ran. See [`kernels::matmul`] for the hot spot.
//!
//! ## `unsafe`
//!
//! The default (Generic) path is `unsafe`-minimal: the only `unsafe` is the raw
//! device-buffer access the ep-api contract forces (aligned host
//! `alloc`/`dealloc`, `memcpy`, and strided element reads/writes), each isolated
//! and `SAFETY`-documented, plus — only under the `onednn` feature — the
//! `dnnl_sgemm` FFI call, confined to [`kernels::onednn`]. The blocked rayon GEMM
//! itself contains no `unsafe`; all kernel arithmetic is safe Rust operating on
//! dense `Vec<f32>` buffers produced by the two audited accessors in [`kernels`].

pub mod backend;
pub mod dtype;
pub mod kernels;
pub mod optimizer;
pub mod provider;
pub mod strided;
pub mod weight_offload;

pub use backend::{CpuBackend, has_onednn};
pub use kernels::qmoe::WeightOffloadHostCache;
pub use optimizer::{ProjectionFusion, cpu_optimization_passes};
pub use provider::CpuExecutionProvider;
pub use weight_offload::{
    LinuxProcessMemoryStats, WEIGHT_OFFLOAD_ENV, WEIGHT_OFFLOAD_HOST_BYTES_ENV,
    WeightOffloadLayerStats, WeightOffloadStats, set_weight_offload_host_budget,
    weight_offload_stats,
};

pub use kernels::slice::{SliceAxisPlan, slice_axes_steps, slice_plan};
