//! # `onnx-runtime-ep-cpu`
//!
//! The CPU execution provider for the ORT 2.0 runtime (see `docs/architecture/ORT2.md` §4.4
//! and §54 Phase 1). It implements [`onnx_runtime_ep_api::ExecutionProvider`]
//! and hosts pure-Rust reference kernels for the Phase-1 op set (`MatMul`,
//! `Add`, `Relu`, `Reshape`, `Transpose`, `Gather`, `LayerNormalization`).
//!
//! ## Backends: correctness baseline + SIMD fast path
//!
//! The GEMM hot spot is served through [`backend::CpuBackend`] (`docs/architecture/ORT2.md`
//! §25.2). The **default** backend is a pure-Rust blocked, register-tiled,
//! rayon-parallelized f32 GEMM — the portable, offline correctness baseline that
//! compiles anywhere with no C++/FFI. On supported x86 hosts, the built-in
//! `SimdX86` implementation provides the default fast path. Every backend lives behind the
//! [`onnx_runtime_ep_api::Kernel`] trait, so neither the EP contract nor the
//! session observes which one ran. See [`kernels::matmul`] for the hot spot.
//!
//! ## `unsafe`
//!
//! The default (Generic) path is `unsafe`-minimal: the only `unsafe` is the raw
//! device-buffer access the ep-api contract forces (aligned host
//! `alloc`/`dealloc`, `memcpy`, and strided element reads/writes), each isolated
//! and `SAFETY`-documented. The blocked rayon GEMM itself contains no `unsafe`;
//! all kernel arithmetic is safe Rust operating on
//! dense `Vec<f32>` buffers produced by the two audited accessors in [`kernels`].

// Kernel entry points mirror ONNX operator schemas, whose independent tensors and
// dimensions often exceed Clippy's generic argument-count threshold.
#![allow(clippy::too_many_arguments)]

pub mod backend;
pub mod core_topology;
pub mod decode_affinity;
pub mod decode_numa;
pub mod decode_spmd;
pub mod dtype;
pub mod kernels;
#[cfg(all(feature = "mlas", feature = "ops-cnn"))]
pub mod nchwc_layout;
pub mod optimizer;
pub mod provider;
pub mod strided;
pub mod task_runtime;
mod trace;
pub mod weight_offload;

pub use backend::CpuBackend;
pub use kernels::qmoe::WeightOffloadHostCache;
pub use kernels::{CpuOpDescriptor, build_cpu_registry_with_descriptors, supported_dtypes_for_op};
pub use optimizer::{
    ConvBatchNormActivationFusion, MatMulNBitsBiasFusion, ProjectionFusion, SiblingProjectionMerge,
    cpu_optimization_passes,
};
pub use provider::CpuExecutionProvider;
pub use weight_offload::placement::{
    GpuLayersOverrideReport, HostFallbackReason, IqFormat, LayerPlacement, LayerWeightRegions,
    Placement, PlacementError, PlacementPlan, QuantTileFormat, RegionPlacement, SnappedTileSize,
    TileSizeError, plan_placement, snap_transfer_tile_bytes,
};
pub use weight_offload::weight_handle::{
    ExecutionProviderCapabilities, LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary,
    NXRT_WEIGHT_PAGING_CAPABILITY, NegotiatedWeight, Phase3aHostOnlyBinder, ResidentWeight,
    ResidentWeightMaterializer, WeightHandle, WeightHandleError,
};
pub use weight_offload::{
    LinuxProcessMemoryStats, WEIGHT_OFFLOAD_ENV, WEIGHT_OFFLOAD_HOST_BYTES_ENV,
    WeightOffloadLayerStats, WeightOffloadStats, weight_offload_stats,
};

pub use kernels::selection::non_max_suppression;
pub use kernels::slice::{SliceAxisPlan, slice_axes_steps, slice_plan};

pub use kernels::matmul_nbits::bound_process_to_decode_budget;
pub use kernels::matmul_nbits::set_decode_thread_budget;
pub use kernels::matmul_nbits::with_decode_pool_scope;
pub use kernels::matmul_nbits::{
    matmul_nbits_decode_caches_dequant_f32, matmul_nbits_resident_side_cache_bytes,
    mlas_sqnbit_packed_live_bytes, resident_dequant_f32_cache_bytes,
    set_mlas_sqnbit_packing_enabled, set_resident_dequant_f32_cache_enabled,
};
// #1056: a resident, session-lifetime, weight-scaled buffer must be reportable
// in bytes. The entry-count accessor stays for the benchmarks that assert reuse.
pub use kernels::matmul::weight_transpose_cache_bytes;
// #1056: the transpose cache is now *governed*, not just reported: the plan
// budgets `weight_transpose_cache_predicted_bytes` and, when it does not fit,
// declines it via `set_weight_transpose_cache_enabled` (kernels then transpose
// per call and retain nothing).
pub use kernels::matmul::{
    set_weight_transpose_cache_enabled, weight_transpose_cache_predicted_bytes,
};
// #1056: the per-kernel `MatMulPrepack::dense` widened-f32 cache is the fourth
// resident, weight-scaled buffer brought under the plan. The plan budgets
// `matmul_dense_cache_predicted_bytes` and, when it does not fit, declines it via
// `set_matmul_dense_cache_enabled` (kernels then widen per call and retain
// nothing, byte-identical output).
pub use kernels::matmul::{matmul_dense_cache_predicted_bytes, set_matmul_dense_cache_enabled};
// #1133: the QLinearMatMul per-thread `i32` accumulator scratch was bounded by a
// per-buffer constant but parked on every worker thread, so the real ceiling was
// `32 MiB x threads`. It is now a process-wide, declinable budget: the plan
// budgets `qlinear_accumulator_budget_predicted_bytes` and declines it via
// `set_qlinear_accumulator_budget_admitted` (kernels then reallocate per call,
// byte-identical). The constant-`B` MLAS pre-pack (`packed_b`, ungoverned since
// `ac394fd6`) is the second buffer in the same kernel brought under the plan.
pub use kernels::qlinear_matmul::{
    qlinear_accumulator_budget_predicted_bytes, qlinear_accumulator_live_bytes,
    qlinear_accumulator_process_cap_bytes, qlinear_packed_b_live_bytes,
    qlinear_packed_b_predicted_bytes, set_qlinear_accumulator_budget_admitted,
    set_qlinear_accumulator_process_cap_bytes, set_qlinear_packed_b_enabled,
};
