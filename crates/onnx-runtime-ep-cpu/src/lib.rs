//! # `onnx-runtime-ep-cpu`
//!
//! The CPU execution provider for the ORT 2.0 runtime (see `docs/ORT2.md` §4.4
//! and §54 Phase 1). It implements [`onnx_runtime_ep_api::ExecutionProvider`]
//! and hosts pure-Rust reference kernels for the Phase-1 op set (`MatMul`,
//! `Add`, `Relu`, `Reshape`, `Transpose`, `Gather`, `LayerNormalization`).
//!
//! ## Backends: correctness baseline + SIMD fast path
//!
//! The GEMM hot spot is served through [`backend::CpuBackend`] (`docs/ORT2.md`
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
pub mod decode_affinity;
pub mod decode_numa;
pub mod decode_spmd;
pub mod dtype;
pub mod kernels;
#[cfg(feature = "mlas")]
pub mod nchwc_layout;
pub mod optimizer;
pub mod provider;
pub mod strided;
mod trace;
pub mod weight_offload;

pub use backend::CpuBackend;
pub use kernels::qmoe::WeightOffloadHostCache;
pub use optimizer::{
    ConvBatchNormActivationFusion, MatMulNBitsBiasFusion, ProjectionFusion, cpu_optimization_passes,
};
pub use provider::CpuExecutionProvider;
pub use weight_offload::placement::{
    ArbitrationAction, GpuLayersOverrideReport, HostFallbackReason, IqFormat, KvAdmissionDecision,
    KvAdmissionLimitingFactor, LayerPlacement, LayerWeightRegions, Placement, PlacementError,
    PlacementPlan, QuantTileFormat, RegionPlacement, SnappedTileSize, TileSizeError,
    VramArbitrationConfig, VramArbitrationError, VramArbitrationOutcome, VramArbitrationState,
    VramDemand, VramSubBudgets, arbitrate_vram, decide_kv_admission, plan_placement,
    snap_transfer_tile_bytes,
};
pub use weight_offload::weight_handle::{
    ExecutionProviderCapabilities, LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary,
    NXRT_WEIGHT_PAGING_CAPABILITY, NegotiatedWeight, Phase3aHostOnlyBinder, ResidentWeight,
    ResidentWeightMaterializer, WeightHandle, WeightHandleError,
};
pub use weight_offload::{
    LinuxProcessMemoryStats, WEIGHT_OFFLOAD_ENV, WEIGHT_OFFLOAD_HOST_BYTES_ENV,
    WeightOffloadLayerStats, WeightOffloadStats, set_weight_offload_host_budget,
    weight_offload_stats,
};

pub use kernels::selection::non_max_suppression;
pub use kernels::slice::{SliceAxisPlan, slice_axes_steps, slice_plan};

pub use kernels::matmul_nbits::with_decode_pool_scope;
