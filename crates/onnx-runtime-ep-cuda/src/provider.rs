//! The [`CudaExecutionProvider`]: a GPU execution provider backed by cudarc +
//! cuBLASLt (`docs/ORT2.md` §15). Phase 2a wires standard GEMM (`MatMul`) only;
//! everything else returns an actionable "not implemented in CUDA EP Phase 2a"
//! error rather than silently falling back or panicking.
//!
//! # Memory & safety model
//!
//! Mirrors the ep-api safety contract used by the CPU EP, but the buffers live
//! in **device** memory:
//!
//! 1. **Owner-frees** — every [`allocate`](CudaExecutionProvider::allocate)
//!    (`cuMemAlloc`) pairs with exactly one
//!    [`deallocate`](CudaExecutionProvider::deallocate) (`cuMemFree`).
//!    [`onnx_runtime_ep_api::DeviceBuffer`] has no `Drop`, so a dropped handle
//!    leaks but never double-frees.
//! 2. **No cross-EP free** — `deallocate`/`copy` assert the buffer's device
//!    matches this EP's `CUDA:ordinal`.
//! 3. **Bounds** — `copy` rejects a `size` larger than either endpoint.
//! 4. **Opaque device pointers** — a CUDA device pointer is *not* host-
//!    dereferenceable; it only travels between `allocate`, `copy`, and kernels,
//!    exactly as [`onnx_runtime_ep_api::DeviceBuffer`] documents for CUDA.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, ExecutionProviderCapabilities, Fence,
    Kernel, KernelMatch, LazyWeight, OpRegistry, PagedWeight, Result, deny,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::kernels::build_cuda_registry_with_metrics;
use crate::kernels::csa_checkpoint::CsaMetrics;
use crate::optimizer::cuda_optimization_passes;
use crate::runtime::{CudaRuntime, cuptr};
use crate::weight_paging::{CudaWeightResidency, DeviceOffloadPolicy};

/// Default VRAM budget for the device weight-offload residency cache when
/// `ONNX_GENAI_WEIGHT_OFFLOAD` is enabled without an explicit
/// `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES` override (4 GiB).
pub const DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES: u64 = 4 << 30;

/// CUDA execution provider (Phase 2a: cudarc + cuBLASLt GEMM).
///
/// Unlike the always-available CPU EP, this provider needs a real device, so
/// [`CudaExecutionProvider::new`] is **fallible** — it returns an error when no
/// CUDA device is present or the driver / cuBLASLt cannot be loaded. Callers on
/// a machine without a GPU should treat that error as "CUDA EP unavailable".
pub struct CudaExecutionProvider {
    device: DeviceId,
    runtime: Arc<CudaRuntime>,
    /// Where this EP's device buffers come from.
    ///
    /// The same DeviceAllocator contract the CPU EP and the ONNX Runtime
    /// allocator use, so an allocator a caller writes serves every backend.
    /// Defaults to CudaDeviceAllocator, which is the cuMemAlloc call this
    /// EP used to make directly.
    memory: Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
    /// Installed by `adopt_memory_governor` when VMM is enabled, and read in
    /// place of `memory` from then on.
    ///
    /// A `OnceLock` rather than a lock because this is set once, before any
    /// allocation, and read on every one: `get` is a relaxed atomic load, so
    /// the hot path pays nothing for the option to swap.
    vmm: std::sync::OnceLock<Arc<crate::vmm_allocator::CudaVmmAllocator>>,
    /// Allocations and frees this EP made through `memory`.
    ///
    /// Kept here rather than asked of the allocator, because the allocator is
    /// the part a caller replaces and counting is not something the shared
    /// contract should require of them. The EP knows every call it makes.
    ///
    /// These exist because roughly twenty-five tests assert that a warmed,
    /// capture-safe path performs no further allocations, and they assert it by
    /// reading a counter. Before the allocator seam those went through
    /// `CudaRuntime::alloc_raw`, which counted them; afterwards they did not,
    /// and every one of those assertions silently became "0 == 0".
    ep_allocations: Arc<AtomicU64>,
    ep_frees: Arc<AtomicU64>,
    initialized: bool,
    registry: OpRegistry,
    csa_metrics: Arc<CsaMetrics>,
    /// Device weight-offload policy resolved from the environment. When enabled,
    /// the EP advertises the `nxrt` weight-paging capability and pages lazy
    /// weights host↔device on demand during dispatch.
    offload_policy: DeviceOffloadPolicy,
    /// LRU device residency cache. `Some` iff `offload_policy.enabled`.
    residency: Option<Arc<CudaWeightResidency>>,
}

impl std::fmt::Debug for CudaExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaExecutionProvider")
            .field("device", &self.device)
            .field("initialized", &self.initialized)
            .field("registered_ops", &self.registry.len())
            .finish()
    }
}

impl CudaExecutionProvider {
    /// Construct a CUDA EP bound to `CUDA:ordinal` with the Phase-2a kernels
    /// registered. Fails if the device or CUDA libraries are unavailable.
    pub fn new(ordinal: u32) -> Result<Self> {
        Self::new_with_offload_policy(ordinal, DeviceOffloadPolicy::from_env())
    }

    /// Construct a CUDA EP with an already-resolved weight-offload policy.
    ///
    /// The engine uses this when `--vram-limit` is the authority that enables
    /// offload. Reading only the process environment here would recreate #712:
    /// the limit would be parsed by the CLI while weights still loaded under an
    /// unrelated residency policy.
    pub fn new_with_offload_policy(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_and_governor(ordinal, offload_policy, None)
    }

    /// Construct a CUDA EP with the device authority available before the
    /// allocator reserves or commits memory.
    pub fn new_with_offload_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<Self> {
        Self::new_with_policy_and_governor(ordinal, offload_policy, Some(governor))
    }

    fn new_with_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Option<Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>>,
    ) -> Result<Self> {
        let runtime = Arc::new(CudaRuntime::new(ordinal)?);
        let csa_metrics = Arc::new(CsaMetrics::default());
        let registry = build_cuda_registry_with_metrics(runtime.clone(), csa_metrics.clone());
        let residency = offload_policy.enabled.then(|| {
            let budget = offload_policy
                .device_budget_bytes
                .unwrap_or(DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES);
            Arc::new(
                CudaWeightResidency::new(runtime.clone(), budget)
                    .with_async_pagein(offload_policy.async_pagein)
                    .with_scan_resistant_dense(offload_policy.scan_resistant_dense),
            )
        });
        Ok(Self {
            device: DeviceId::cuda(ordinal),
            memory: Arc::new(crate::device_allocator::CudaDeviceAllocator::new(
                runtime.cuda_context(),
            )),
            // Built here rather than at governor adoption, because on the
            // native path the session allocates every tensor it will use while
            // loading -- which is before any governor reaches this provider.
            // An arena installed at adoption is installed at the one moment
            // after which nothing will ask it for memory (#659).
            //
            // Address space is free, so the reservation is generous rather than
            // fitted: 64 GiB comfortably exceeds any single accelerator we
            // target, and running out of *reservation* is a hard failure while
            // leaving it unmapped costs nothing.
            vmm: {
                let cell = std::sync::OnceLock::new();
                if crate::vmm_allocator::vmm_enabled() {
                    const RESERVATION_BYTES: usize = 64 << 30;
                    let arena = match governor.as_deref() {
                        Some(governor) => crate::vmm_allocator::CudaVmmAllocator::new(
                            runtime.cuda_context(),
                            onnx_runtime_memory_governor::DeviceKey::device(ordinal),
                            ordinal as i32,
                            RESERVATION_BYTES,
                            governor,
                            onnx_runtime_memory_governor::HolderId::new(64),
                            onnx_runtime_memory_governor::MemoryRole::Workspace {
                                step_scoped: false,
                            },
                        ),
                        None => crate::vmm_allocator::CudaVmmAllocator::standalone(
                            runtime.cuda_context(),
                            onnx_runtime_memory_governor::DeviceKey::device(ordinal),
                            ordinal as i32,
                            RESERVATION_BYTES,
                            onnx_runtime_memory_governor::HolderId::new(64),
                            onnx_runtime_memory_governor::MemoryRole::Workspace {
                                step_scoped: false,
                            },
                        ),
                    };
                    match arena {
                        Ok(arena) => {
                            eprintln!(
                                "cuda_ep: device allocations go through a VMM arena over \
                                 {RESERVATION_BYTES} bytes of reserved address space; physical \
                                 granules are mapped on demand and join the memory ledger when \
                                 one arrives"
                            );
                            let _ = cell.set(Arc::new(arena));
                        }
                        // Falling back to cuMemAlloc keeps the model running,
                        // which matters more than the accounting -- but say so,
                        // because a silent fallback is how a feature reads as
                        // enabled while doing nothing.
                        Err(error) => eprintln!(
                            "cuda_ep: WARNING: could not build the VMM arena, falling back to \
                             cuMemAlloc; device allocations will not be charged to the ledger: \
                             {error}"
                        ),
                    }
                }
                cell
            },
            ep_allocations: Arc::new(AtomicU64::new(0)),
            ep_frees: Arc::new(AtomicU64::new(0)),
            runtime,
            initialized: false,
            registry,
            csa_metrics,
            offload_policy,
            residency,
        })
    }

    /// The allocator in force: the VMM arena once installed, otherwise the one
    /// this provider was built with.
    ///
    /// `OnceLock::get` is a relaxed atomic load, so the allocation path pays
    /// nothing for the option to swap.
    fn memory(&self) -> &dyn onnx_runtime_memory_governor::DeviceAllocator {
        match self.vmm.get() {
            Some(arena) => arena.as_ref(),
            None => self.memory.as_ref(),
        }
    }

    fn synchronize_before_pooled_unmap(&self) -> Result<()> {
        if self
            .vmm
            .get()
            .is_some_and(|arena| arena.physical_pool_stats().is_some())
        {
            self.runtime.synchronize().map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not synchronize before returning pooled physical memory: \
                     {error}"
                ))
            })?;
            self.runtime.copy_stream().synchronize().map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not synchronize the copy stream before returning pooled \
                     physical memory: {error}"
                ))
            })?;
        }
        Ok(())
    }

    /// Take device buffers from `memory` instead of calling `cuMemAlloc`
    /// directly.
    /// The same `DeviceAllocator` a caller installs on the CPU EP or the ONNX
    /// Runtime side. This is where a CUDA arena belongs: `cudaMalloc` is a
    /// synchronising driver call in the microseconds, so unlike host memory,
    /// device memory genuinely needs one — and behind this seam it is written
    /// once rather than once per backend.
    ///
    /// # Errors
    ///
    /// If `memory` does not serve this EP's device. Pointers from it are handed
    /// to kernels as this device's addresses, so a host allocator or another
    /// device's allocator would produce an address that is invalid where it is
    /// used. That fails inside a kernel launch, far from the substitution that
    /// caused it, so it is rejected here instead.
    pub fn with_memory(
        mut self,
        memory: Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
    ) -> Result<Self> {
        let key = memory.device();
        let expected = onnx_runtime_memory_governor::DeviceKey::device(self.device.index);
        if key != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: this execution provider serves CUDA device {}, but the allocator \
                 offered serves {:?} {}; its pointers would not be valid where this EP uses \
                 them. Supply an allocator for CUDA device {}.",
                expected.index, key.tier, key.index, expected.index
            )));
        }
        self.memory = memory;
        Ok(self)
    }

    /// Construct and initialize a CUDA execution provider with default settings.
    pub fn initialized(ordinal: u32) -> Result<Self> {
        let mut provider = Self::new(ordinal)?;
        <Self as ExecutionProvider>::initialize(&mut provider, &EpConfig::default())?;
        Ok(provider)
    }

    /// Construct and initialize a CUDA execution provider with an already
    /// resolved weight-offload policy.
    pub fn initialized_with_offload_policy(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
    ) -> Result<Self> {
        let mut provider = Self::new_with_offload_policy(ordinal, offload_policy)?;
        <Self as ExecutionProvider>::initialize(&mut provider, &EpConfig::default())?;
        Ok(provider)
    }

    /// Construct and initialize with a device authority available to allocator
    /// construction.
    pub fn initialized_with_offload_policy_and_governor(
        ordinal: u32,
        offload_policy: DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> Result<Self> {
        let mut provider =
            Self::new_with_offload_policy_and_governor(ordinal, offload_policy, governor)?;
        <Self as ExecutionProvider>::initialize(&mut provider, &EpConfig::default())?;
        Ok(provider)
    }

    /// Construct a CUDA EP on the default device (`CUDA:0`).
    pub fn new_default() -> Result<Self> {
        Self::new(0)
    }

    /// Return whether a fully initialized CUDA EP can be constructed for this
    /// device right now. This checks the driver, wheel/system libraries, device,
    /// and thread binding rather than reporting a compile-time feature.
    pub fn is_available(ordinal: u32) -> bool {
        Self::initialized(ordinal).is_ok()
    }

    /// Borrow the CUDA op registry (shared with the session layer).
    pub fn registry(&self) -> &OpRegistry {
        &self.registry
    }

    /// Borrow the shared CUDA runtime (context + stream + cuBLASLt handle).
    pub fn runtime(&self) -> &Arc<CudaRuntime> {
        &self.runtime
    }

    /// Borrow the shared CSA observability surface (§8). Every CSA kernel this
    /// EP builds records per-layer attention mode, bytes avoided, cursor
    /// lengths, sink mass, and host/device byte counts here; speculative
    /// rollbacks accumulate via the checkpoint journal.
    pub fn csa_metrics(&self) -> &Arc<CsaMetrics> {
        &self.csa_metrics
    }

    /// Build a live GPU weight pager (WEIGHT_OFFLOAD Phase 3b) that binds an
    /// offloaded `pkg.nxrt::BlockQuantizedMoE` weight into a VRAM page, copying
    /// its canonical bytes from `source` host→device. The returned binding is
    /// byte-identical to a resident upload of the same weight.
    pub fn weight_pager<'a, S: onnx_runtime_ep_api::MmapRegionSource>(
        &self,
        source: &'a S,
    ) -> crate::weight_paging::CudaWeightPager<'a, S> {
        crate::weight_paging::CudaWeightPager::new(Arc::clone(&self.runtime), source)
    }

    /// Build a bounded-VRAM [`CudaWeightResidency`] (WEIGHT_OFFLOAD Phase 3b
    /// page-in + eviction) sized by `budget_bytes`, sharing this EP's runtime.
    pub fn weight_residency(&self, budget_bytes: u64) -> crate::weight_paging::CudaWeightResidency {
        crate::weight_paging::CudaWeightResidency::new(Arc::clone(&self.runtime), budget_bytes)
    }

    /// Borrow the live device residency cache used to page lazy weights during
    /// dispatch, or `None` when weight offload is disabled. Tests use this to
    /// assert page-in / eviction counters after a decode.
    pub fn residency(&self) -> Option<&Arc<CudaWeightResidency>> {
        self.residency.as_ref()
    }

    /// The resolved device weight-offload policy for this EP.
    pub fn offload_policy(&self) -> &DeviceOffloadPolicy {
        &self.offload_policy
    }
}

impl ExecutionProvider for CudaExecutionProvider {
    fn name(&self) -> &str {
        "cuda_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda
    }

    fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Advertise the `nxrt` weight-paging capability only when device weight
    /// offload is enabled. This is what makes the session build real lazy weight
    /// handles for boundary-matched quantized weights; the default (offload off)
    /// path keeps stock capabilities so the fast resident path is unchanged.
    fn capabilities(&self) -> ExecutionProviderCapabilities {
        if self.residency.is_some() {
            ExecutionProviderCapabilities::nxrt_weight_paging()
        } else {
            ExecutionProviderCapabilities::stock()
        }
    }

    /// Page a lazy weight into VRAM (or reuse a resident page) through the LRU
    /// residency cache, returning a [`PagedWeight`] whose keep-alive pins the
    /// device allocation for the kernel's lifetime. Returns `Ok(None)` when
    /// offload is disabled so dispatch falls back to the resident path.
    fn page_lazy_weight(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn onnx_runtime_ep_api::MmapRegionSource,
    ) -> Result<Option<PagedWeight>> {
        let Some(residency) = self.residency.as_ref() else {
            return Ok(None);
        };
        let page = residency
            .resident_mapped(key, weight, source)
            .map_err(|error| EpError::KernelFailed(format!("weight offload page-in: {error}")))?;
        let device_ptr = page.device_ptr();
        let len = page.len();
        Ok(Some(PagedWeight::new(
            device_ptr,
            self.device,
            len,
            page as Arc<dyn std::any::Any + Send + Sync>,
        )))
    }

    fn prefetch_lazy_weight(
        &self,
        key: u64,
        weight: &LazyWeight,
        source: &dyn onnx_runtime_ep_api::MmapRegionSource,
    ) -> Result<bool> {
        let _ = (self, key, weight, source);
        Ok(false)
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        // The context, stream, and cuBLASLt handle are created eagerly in
        // `new`; binding here confirms the device is reachable on this thread.
        self.runtime.bind()?;
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        // Keyed on (op_type, domain, opset) via the registry, the same single
        // source of truth the CPU EP uses.
        if !self.registry.supports(&op.op_type, &op.domain, opset) {
            let domain = if op.domain.is_empty() {
                "ai.onnx"
            } else {
                &op.domain
            };
            if let Some(since) = self
                .registry
                .earliest_since_version(&op.op_type, &op.domain)
            {
                deny!(
                    "no handler for {}::{} at opset {} — this EP registers {} since opset {} (or: add a claim+handler)",
                    domain,
                    op.op_type,
                    opset,
                    op.op_type,
                    since
                );
            }
            deny!(
                "no handler for {}::{} at opset {} — add a claim+handler",
                domain,
                op.op_type,
                opset
            );
        }
        if matches!(op.op_type.as_str(), "FusedMatMulBias" | "FusedGemm")
            && op.domain == "com.microsoft"
            && let Some(reason) = crate::kernels::fused_gemm::unsupported_reason(op, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "BlockQuantizedMatMul"
            && op.domain == "pkg.nxrt"
            && let Some(reason) = crate::kernels::block_quantized_matmul::unsupported_reason(op)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "BlockQuantizedMoE"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::block_quantized_moe::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "CompressedSparseAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) = crate::kernels::compressed_sparse_attention::unsupported_reason(
                op,
                shapes,
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "IndexShare"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::index_share::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "PackedVarlenAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::packed_varlen_attention::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "VarlenAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::varlen_attention::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "QMoE"
            && op.domain == "com.microsoft"
            && let Some(reason) = crate::kernels::qmoe::unsupported_reason(op)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "GatherBlockQuantized"
            && op.domain == "com.microsoft"
            && let Some(reason) =
                crate::kernels::gather_block_quantized::unsupported_reason(op, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "CausalConvWithState"
            && op.domain == "com.microsoft"
            && let Some(reason) =
                crate::kernels::causal_conv_with_state::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "LinearAttention"
            && matches!(op.domain.as_str(), "" | "ai.onnx" | "com.microsoft")
            && let Some(reason) =
                crate::kernels::linear_attention::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "Attention"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::standard_attention::unsupported_reason(opset, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "RotaryEmbedding"
            && matches!(op.domain.as_str(), "" | "ai.onnx" | "com.microsoft")
            && let Some(reason) = crate::kernels::rotary_embedding::unsupported_reason(
                op.domain == "com.microsoft",
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::standard_claims::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(op.op_type.as_str(), "QuantizeLinear" | "DequantizeLinear")
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) = crate::kernels::quantization::unsupported_reason(op, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(
            op.op_type.as_str(),
            "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual"
        ) && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::pointwise::comparison_unsupported_reason(&op.op_type, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(op.op_type.as_str(), "IsInf" | "IsNaN")
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::unary_predicate::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "PRelu"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) = crate::kernels::prelu::unsupported_reason(op, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if matches!(
            op.op_type.as_str(),
            "BitwiseAnd" | "BitwiseOr" | "BitwiseXor" | "BitwiseNot" | "BitShift"
        ) && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::bitwise::unsupported_reason(&op.op_type, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        let output_layouts = vec![TensorLayout::contiguous(); op.outputs.len()];
        let elems: u64 = shapes
            .iter()
            .map(|s| {
                s.iter()
                    .map(|d| d.as_static().unwrap_or(1) as u64)
                    .product::<u64>()
            })
            .sum();
        // GPU compute is cheap per element but launch latency is high; bias the
        // rough estimate accordingly so tiny ops still prefer the CPU EP. The
        // real cost model lands in Phase 2.
        let cost = Cost::new(elems as f64 * 0.01, elems as f64 * 0.01, 0.0)
            .with_launch_us(10.0)
            .with_bytes_moved(elems.saturating_mul(4));
        KernelMatch::Supported {
            cost,
            required_input_layouts: None,
            output_layouts,
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>> {
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
            .ok_or_else(|| EpError::NoEpForOp {
                domain: if op.domain.is_empty() {
                    "ai.onnx".to_string()
                } else {
                    op.domain.clone()
                },
                op_type: op.op_type.clone(),
                opset,
            })?;
        factory.create(op, shapes)
    }

    fn custom_passes(&self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        cuda_optimization_passes()
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        let full = 0..size;
        self.allocate_committed(size, alignment, std::slice::from_ref(&full))
    }

    fn allocate_committed(
        &self,
        size: usize,
        alignment: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<DeviceBuffer> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        // One allocator for the whole project: the same `DeviceAllocator` a
        // caller can install on the CPU EP or the ONNX Runtime side backs this
        // one. The default is `CudaDeviceAllocator`, which is the `cuMemAlloc`
        // this used to call directly.
        //
        // `size` is passed through unchanged so that `deallocate` can pass the
        // same value; normalising a zero-byte request is the allocator's job,
        // because the contract lets an implementation rely on the two sizes
        // agreeing.
        let ptr = self
            .memory()
            .allocate_committed(size, alignment, committed_ranges)
            .map_err(|error| {
            // Keep what the allocator said. Reporting every failure as "out of
            // memory" would describe an alignment rejection or a dead context
            // as exhausted VRAM and send the reader looking in the wrong place.
            EpError::KernelFailed(format!(
                "cuda_ep: could not allocate {size} bytes aligned to {alignment} on CUDA device \
                 {}: {error}",
                self.device.index
            ))
        })?;
        self.ep_allocations.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` is a fresh, unique, non-null device allocation of
        // >= `size` bytes owned by this EP and freed exactly once in
        // `deallocate`. It is a device address, never dereferenced on the host.
        Ok(unsafe {
            DeviceBuffer::from_raw_parts(ptr.as_ptr().cast(), self.device, size, alignment)
        })
    }

    fn commit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        assert_eq!(
            buffer.device(),
            self.device,
            "cuda_ep: refusing to commit a buffer from device {:?}",
            buffer.device()
        );
        let Some(ptr) = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8) else {
            return Ok(());
        };
        self.memory()
            .commit_allocation_range(ptr, buffer.len(), buffer.alignment(), offset, bytes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not commit range {offset}..{} of {} byte allocation on CUDA device {}: {error}",
                    offset.saturating_add(bytes),
                    buffer.len(),
                    self.device.index
                ))
            })
    }

    fn decommit_allocation_range(
        &self,
        buffer: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        assert_eq!(
            buffer.device(),
            self.device,
            "cuda_ep: refusing to decommit a buffer from device {:?}",
            buffer.device()
        );
        let Some(ptr) = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8) else {
            return Ok(());
        };
        self.synchronize_before_pooled_unmap()?;
        self.memory()
            .decommit_allocation_range(ptr, buffer.len(), buffer.alignment(), offset, bytes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: could not decommit range {offset}..{} of {} byte allocation on CUDA device {}: {error}",
                    offset.saturating_add(bytes),
                    buffer.len(),
                    self.device.index
                ))
            })
    }

    fn allocation_committed_bytes(&self, buffer: &DeviceBuffer) -> usize {
        let Some(ptr) = std::ptr::NonNull::new(buffer.as_ptr().cast::<u8>() as *mut u8) else {
            return 0;
        };
        self.memory()
            .allocation_committed_bytes(ptr, buffer.len(), buffer.alignment())
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        assert_eq!(
            buffer.device(),
            self.device,
            "cuda_ep: refusing to deallocate a buffer from device {:?}",
            buffer.device()
        );
        // Borrowed buffers alias memory owned elsewhere and must never be
        // cuMemFree'd. CUDA does not yet produce borrowed buffers, but keep the
        // invariant sound so one can never be freed here.
        if buffer.is_borrowed() {
            return Ok(());
        }
        self.synchronize_before_pooled_unmap()?;
        let size = buffer.len();
        let align = buffer.alignment();
        let ptr = buffer.into_raw();
        let Some(ptr) = std::ptr::NonNull::new(ptr.cast::<u8>()) else {
            return Ok(());
        };
        // SAFETY: `ptr`, `size` and `align` are the triple this EP obtained
        // from `self.memory` in `allocate`; `into_raw` consumed the owning
        // handle so no alias remains, and this is its single free.
        unsafe { self.memory().deallocate(ptr, size, align) };
        self.ep_frees.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()> {
        assert_eq!(
            src.device(),
            self.device,
            "cuda_ep::copy: foreign src buffer"
        );
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy: foreign dst buffer"
        );
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy: size {size} exceeds src {} or dst {}",
                src.len(),
                dst.len()
            )));
        }
        if size == 0 {
            return Ok(());
        }
        let src_p = cuptr(src.as_ptr());
        let dst_p = cuptr(dst.as_mut_ptr());
        // SAFETY: both endpoints are live device allocations of >= `size` bytes
        // (checked) on this EP's device; `dst` is `&mut` so it cannot alias `src`.
        unsafe { self.runtime.dtod(src_p, dst_p, size) }
    }

    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence> {
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_async: foreign dst buffer"
        );
        if size > dst.len() || size > src.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_async: size {size} exceeds src {} or dst {}",
                src.len(),
                dst.len()
            )));
        }
        if size == 0 {
            return Ok(Fence::signalled());
        }
        let dst_p = cuptr(dst.as_mut_ptr());
        if src.device().is_host_accessible() {
            // Host → device weight prefetch: the real Phase-4 overlap path. The
            // copy is enqueued on the dedicated transfer stream and the returned
            // fence names its completion event.
            // SAFETY: a host-accessible src buffer exposes a dereferenceable host
            // pointer to at least `size` bytes (checked); the async copy keeps
            // reading `src` until the transfer stream completes, which the caller
            // orders via `wait_fence` before mutating or freeing `src`.
            let host = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), size) };
            // SAFETY: `dst` is a live device allocation of >= `size` bytes.
            unsafe { self.runtime.htod_async(host, dst_p) }?;
        } else {
            assert_eq!(
                src.device(),
                self.device,
                "cuda_ep::copy_async: foreign device src buffer"
            );
            let src_p = cuptr(src.as_ptr());
            // SAFETY: both endpoints are live device allocations of >= `size`
            // bytes (checked) on this EP's device; `dst` is `&mut` so it cannot
            // alias `src`. The transfer-stream copy is ordered via the fence.
            unsafe { self.runtime.dtod_async_on_copy_stream(src_p, dst_p, size) }?;
        }
        let fence_id = self.runtime.record_copy_fence()?;
        Ok(Fence::new(fence_id))
    }

    fn wait_fence(&self, fence: &Fence) -> Result<()> {
        // Order the compute stream after the prefetch transfer: a stream-ordered,
        // non host-blocking cross-stream wait so the next kernel reads the fully
        // transferred bytes. An already-signalled fence is a no-op.
        self.runtime.compute_wait_fence(fence.id)
    }

    fn record_compute_fence(&self) -> Result<Fence> {
        // Record a completion event over the compute stream so a later reuse
        // prefetch (via `copy_wait_fence`) waits for this consumer to finish
        // reading a double-buffer slot before overwriting it (WAR ordering).
        let fence_id = self.runtime.record_compute_fence()?;
        Ok(Fence::new(fence_id))
    }

    fn copy_wait_fence(&self, fence: &Fence) -> Result<()> {
        // Order the transfer stream after the prior consumer's compute: a
        // stream-ordered, non host-blocking cross-stream wait so a reuse prefetch
        // never clobbers a staging buffer mid-read. Already-signalled is a no-op.
        self.runtime.copy_wait_fence(fence.id)
    }

    fn device_argmax_supported(&self) -> bool {
        true
    }

    fn device_argmax(
        &self,
        logits: &DeviceBuffer,
        elements: usize,
        dtype: DataType,
        result: &mut DeviceBuffer,
    ) -> Result<()> {
        crate::kernels::device_argmax::launch(&self.runtime, logits, elements, dtype, result)
    }

    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> Result<()> {
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_from_host: foreign dst buffer"
        );
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_from_host: source {} bytes exceeds dst {}",
                src.len(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `dst` is a live allocation on this CUDA device with enough
        // capacity (checked above), and the synchronous copy completes here.
        unsafe { self.runtime.htod(src, cuptr(dst.as_mut_ptr())) }
    }

    fn copy_from_host_at(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
        byte_offset: usize,
    ) -> Result<()> {
        assert_eq!(
            dst.device(),
            self.device,
            "cuda_ep::copy_from_host_at: foreign dst buffer"
        );
        let end = byte_offset.checked_add(src.len()).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep::copy_from_host_at: upload range overflows".into())
        })?;
        if end > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_from_host_at: range {byte_offset}..{end} exceeds dst {}",
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        let ptr = cuptr(dst.as_mut_ptr())
            .checked_add(byte_offset as u64)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep::copy_from_host_at: device pointer offset overflows".into(),
                )
            })?;
        // SAFETY: `ptr` names the checked byte range within `dst`, and the
        // synchronous copy completes before this method returns.
        unsafe { self.runtime.htod(src, ptr) }
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        assert_eq!(
            src.device(),
            self.device,
            "cuda_ep::copy_to_host: foreign src buffer"
        );
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep::copy_to_host: destination {} bytes exceeds src {}",
                dst.len(),
                src.len()
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: `src` is a live allocation on this CUDA device with enough
        // readable bytes (checked above); `dtoh` synchronizes before returning.
        unsafe { self.runtime.dtoh(dst, cuptr(src.as_ptr())) }
    }

    fn begin_device_graph_capture(&self, kernels: &[&dyn Kernel]) -> Result<()> {
        self.runtime.begin_graph_capture(kernels)
    }

    fn end_device_graph_capture(&self) -> Result<()> {
        self.runtime.end_graph_capture()
    }

    fn abort_device_graph_capture(&self) -> Result<()> {
        self.runtime.abort_graph_capture()
    }

    fn replay_device_graph(&self) -> Result<()> {
        self.runtime.replay_graph()
    }

    fn replay_device_graph_segment(&self, index: usize) -> Result<()> {
        self.runtime.replay_graph_segment(index)
    }

    fn reset_device_graph(&self) -> Result<bool> {
        // Graph invalidation (reset / rewind / KV-capacity or shape change /
        // re-capture) is the explicit host reset point for the capture-error
        // latch, so a fresh generation always starts un-poisoned.
        let invalidated = self.runtime.reset_graph()?;
        self.runtime.reset_capture_error()?;
        Ok(invalidated)
    }

    fn check_device_capture_error(&self) -> Result<u32> {
        self.runtime.check_capture_error()
    }

    fn device_allocation_counts(&self) -> Option<(u64, u64)> {
        // The sum of both paths. Kernels still reach the driver through
        // `CudaRuntime::alloc_raw` for their own workspaces, while buffers this
        // EP hands out go through the replaceable allocator. Reporting only one
        // of them is how the capture-safety assertions stopped observing
        // anything without ever going red.
        let counts = self.runtime.allocation_counts();
        Some((
            counts.allocations + self.ep_allocations.load(Ordering::Relaxed),
            counts.frees + self.ep_frees.load(Ordering::Relaxed),
        ))
    }

    /// True when the VMM arena is in use: it maps 2 MiB granules as spans are
    /// handed out and leases each one before mapping it, so committed memory
    /// tracks real use rather than the largest request anyone might make.
    ///
    /// False on the `cuMemAlloc` path, which takes physical memory at the
    /// moment it is asked for.
    fn commits_on_demand(&self) -> bool {
        self.memory().commits_on_demand()
    }

    fn set_weight_residency_budget(&self, budget_bytes: u64) -> Result<Option<u64>> {
        let Some(residency) = self.residency.as_ref() else {
            return Ok(None);
        };
        residency
            .set_ungoverned_budget(budget_bytes)
            .map(Some)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: cannot set the device weight-residency budget to \
                     {budget_bytes} bytes before governor adoption: {error}"
                ))
            })
    }

    fn adopt_memory_governor(
        &self,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<u64> {
        if crate::vmm_allocator::vmm_enabled()
            && let Some(arena) = self.vmm.get()
        {
            if let Some(authority) = arena.physical_pool_authority()
                && authority != governor.authority_id()
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep: physical-handle pool uses {authority}, but adoption supplied {}; \
                     both must use the same memory authority",
                    governor.authority_id()
                )));
            }
            // The arena has been serving allocations against its own ledger
            // since construction. Move the claim to the real one now that it
            // exists.
            let adoption = arena.adopt_governor(governor, holder);
            if adoption.recorded_bytes > 0 {
                let bytes = adoption.recorded_bytes;
                eprintln!(
                    "cuda_ep: VMM arena joined the memory ledger holding {bytes} bytes already \
                     committed"
                );
            }
            if adoption.unaccounted_bytes > 0 {
                let bytes = adoption.unaccounted_bytes;
                eprintln!(
                    "cuda_ep: WARNING: {bytes} committed VMM arena byte(s) were not recorded in \
                     the memory ledger; profile output will report the accounting fault"
                );
            }
        }

        // The weight-residency cache is the standing pool this EP keeps. With
        // offload disabled there is none, and zero is the honest answer rather
        // than a failure.
        let Some(residency) = self.residency.as_ref() else {
            return Ok(0);
        };
        residency
            .adopt_governed_budget(governor, tier, holder)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "cuda_ep: the device weight-residency cache holds a budget the governor \
                     cannot grant on {tier:?}: {error}"
                ))
            })
    }

    fn sync(&self) -> Result<()> {
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn public_constructor_installs_configured_physical_pool() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        unsafe {
            std::env::set_var(crate::vmm_allocator::CUDA_VMM_ENV, "1");
            std::env::set_var(
                crate::vmm_allocator::CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
                (64usize << 20).to_string(),
            );
        }
        let provider = CudaExecutionProvider::new(0).expect("public CUDA provider");
        assert!(
            provider
                .vmm
                .get()
                .is_some_and(|arena| arena.physical_pool_stats().is_some()),
            "public constructor must use the configured physical pool"
        );
        let stats = provider
            .vmm
            .get()
            .and_then(|arena| arena.physical_pool_stats())
            .expect("pool stats");

        let runtime = provider.runtime().clone();
        let write_after_delay = runtime
            .nvrtc_function(
                "cuda_ep_pool_reuse_sync_test",
                r#"
extern "C" __global__ void write_after_delay(unsigned int* out, long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
    *out = 0x736u;
}
"#,
                "write_after_delay",
            )
            .expect("compile delayed writer");
        let first = provider.allocate(4, 256).expect("first allocation");
        let first_ptr = cuptr(first.as_ptr());
        let spin = 8_000_000_i64;
        let mut launch = runtime.stream().launch_builder(&write_after_delay);
        launch.arg(&first_ptr).arg(&spin);
        unsafe {
            launch
                .launch(LaunchConfig::for_num_elems(1))
                .expect("enqueue delayed write")
        };

        provider
            .deallocate(first)
            .expect("deallocation synchronizes before pooled return");
        let second = provider.allocate(4, 256).expect("reused allocation");
        assert_eq!(stats.snapshot().pool_hits, 1);
        let mut value = [0_u8; 4];
        unsafe { runtime.dtoh(&mut value, cuptr(second.as_ptr())) }.expect("read reused mapping");
        assert_eq!(u32::from_ne_bytes(value), 0x736);
        provider.deallocate(second).expect("final deallocation");
    }

    #[test]
    fn runtime_availability_matches_constructability() {
        let available = CudaExecutionProvider::is_available(0);
        let constructible = CudaExecutionProvider::initialized(0).is_ok();
        assert_eq!(available, constructible);
    }

    // Phase-4 overlap through the public `ExecutionProvider` surface: a host→
    // device `copy_async` returns an awaitable `Fence`, and `wait_fence` orders
    // the compute stream after the transfer. The async copy is delayed behind a
    // spin kernel on the transfer stream, so a consumer launched on the compute
    // stream reads the correct payload only because `wait_fence` established the
    // cross-stream dependency — an already-signalled placeholder fence would let
    // it race ahead and read the pre-transfer poison.
    #[test]
    fn copy_async_fence_orders_h2d_prefetch_through_ep_api() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use std::ffi::c_void;

        const MODULE: &str = "cuda_ep_copy_async_api_test";
        const SOURCE: &str = r#"
extern "C" __global__ void spin_delay(long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
}
extern "C" __global__ void copy_out(const float* in, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i];
}
"#;
        let Ok(ep) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping copy_async API test: CUDA EP unavailable");
            return;
        };
        let runtime = ep.runtime().clone();
        let spin_delay = runtime
            .nvrtc_function(MODULE, SOURCE, "spin_delay")
            .unwrap();
        let copy_out = runtime.nvrtc_function(MODULE, SOURCE, "copy_out").unwrap();

        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let n_u64 = n as u64;

        // Pinned host staging holds the payload; wrap it as a borrowed,
        // host-accessible source buffer for `copy_async`.
        let mut staging = runtime.alloc_pinned(bytes).unwrap();
        let payload: Vec<f32> = (0..n).map(|i| 2.0 + (i % 11) as f32).collect();
        staging.as_mut_slice().copy_from_slice(unsafe {
            std::slice::from_raw_parts(payload.as_ptr().cast::<u8>(), bytes)
        });
        // SAFETY: the pinned staging outlives `src` and every use of it, and it
        // is only read (never written) through the borrowed handle.
        let src = unsafe {
            DeviceBuffer::from_borrowed_parts(
                staging.as_slice().as_ptr() as *mut c_void,
                DeviceId::cpu(),
                bytes,
                1,
            )
        };

        let mut dst = ep.allocate(bytes, 256).unwrap();
        let out = ep.allocate(bytes, 256).unwrap();
        let out_p = cuptr(out.as_ptr());

        for _ in 0..8 {
            // Poison the device destination so a premature read is detectable.
            let poison = vec![-321.0f32; n];
            let poison_bytes =
                unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.htod(poison_bytes, cuptr(dst.as_ptr())) }.unwrap();
            runtime.synchronize().unwrap();

            // Occupy the transfer stream so the async copy cannot finish at once.
            let spin: i64 = 8_000_000;
            let mut delay = runtime.copy_stream().launch_builder(&spin_delay);
            delay.arg(&spin);
            unsafe { delay.launch(LaunchConfig::for_num_elems(1)).unwrap() };

            // Public EP surface: async prefetch, then await its fence.
            let fence = ep.copy_async(&src, &mut dst, bytes).unwrap();
            assert!(
                !fence.is_signalled(),
                "a real transfer must return an unsignalled fence"
            );
            ep.wait_fence(&fence).unwrap();

            // Consume the prefetched buffer on the compute stream.
            let dst_p = cuptr(dst.as_ptr());
            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&dst_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };

            let mut host = vec![0.0f32; n];
            let host_bytes =
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(host_bytes, out_p) }.unwrap();
            assert_eq!(
                host, payload,
                "copy_async consumer read poison — the fence did not order the \
                 transfer before the compute-stream read"
            );
        }

        ep.deallocate(dst).unwrap();
        ep.deallocate(out).unwrap();
    }

    // Anti-regression lock for the async, fence-ordered weight page-in (#87 first
    // increment). Both arms drive the transfer and compute streams through the
    // *same* primitive chain `CudaWeightPage::upload_async` composes internally
    // (`htod_async` + `record_copy_fence`), differing only in how the
    // compute-stream consumer is ordered relative to the transfer:
    //
    //   * Positive (real page-in ordering): a spin-delay holds the H2D copy
    //     pending on the transfer stream, then `compute_wait_fence` orders the
    //     compute-stream consumer after it, so the consumer reads the fully
    //     paged-in bytes. Deleting `compute_wait_fence` leaves the consumer to
    //     read the pre-copy POISON, so the lock is non-vacuous.
    //   * Negative (deterministic poison control): the transfer is event-ordered
    //     strictly *after* the consumer (`record_compute_fence` + `copy_wait_fence`),
    //     so with no `compute_wait_fence` the consumer provably reads pre-transfer
    //     POISON. This proves the compute-side wait is load-bearing without a
    //     wall-clock race — an earlier revision raced the consumer against a
    //     spin-delayed copy, which the parallel, captured `cargo test` invocation
    //     flaked whenever GPU contention delayed the consumer kernel past the copy.
    //
    // Every device/pinned allocation is hoisted out of the timing window, so no
    // synchronizing `cuMemAlloc`/`cuMemHostAlloc` can drain the copy-stream
    // spin-delay: the delay→async-copy→fence→consume window is the only thing the
    // positive arm's ordering depends on. A trailing `upload_async` byte-parity
    // check keeps the real allocate+stage+copy+fence entry point under test.
    #[test]
    fn async_pagein_fence_orders_weight_page_in_consumer() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        const MODULE: &str = "cuda_ep_async_pagein_test";
        const SOURCE: &str = r#"
extern "C" __global__ void spin_delay(long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
}
extern "C" __global__ void copy_out(const float* in, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i];
}
"#;
        let Ok(ep) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping async page-in fence test: CUDA EP unavailable");
            return;
        };
        let runtime = ep.runtime().clone();
        let spin_delay = runtime
            .nvrtc_function(MODULE, SOURCE, "spin_delay")
            .unwrap();
        let copy_out = runtime.nvrtc_function(MODULE, SOURCE, "copy_out").unwrap();

        let n = 4096usize;
        let bytes = n * std::mem::size_of::<f32>();
        let n_u64 = n as u64;
        let payload: Vec<f32> = (0..n).map(|i| 5.0 + (i % 13) as f32).collect();
        let payload_bytes =
            unsafe { std::slice::from_raw_parts(payload.as_ptr().cast::<u8>(), bytes) };

        // Hoist every device/pinned allocation OUT of the per-iteration timing
        // window: a synchronizing `cuMemAlloc`/`cuMemHostAlloc` between the
        // spin-delay and the consumer would drain the delay and let ordering lean
        // on the alloc instead of the fence. All buffers are reused each iteration.
        let poison = vec![-777.0f32; n];
        let poison_bytes =
            unsafe { std::slice::from_raw_parts(poison.as_ptr().cast::<u8>(), bytes) };
        let pos_dst = ep.allocate(bytes, 256).unwrap();
        let neg_dst = ep.allocate(bytes, 256).unwrap();
        let out = ep.allocate(bytes, 256).unwrap();
        let pos_dst_p = cuptr(pos_dst.as_ptr());
        let neg_dst_p = cuptr(neg_dst.as_ptr());
        let out_p = cuptr(out.as_ptr());
        let mut staging = runtime.alloc_pinned(bytes).unwrap();
        staging.as_mut_slice().copy_from_slice(payload_bytes);
        let spin: i64 = 8_000_000;

        for _ in 0..8 {
            // ── Positive: the real page-in ordering. Poison the destination, hold
            // the H2D copy pending behind a spin-delay, then order the
            // compute-stream consumer after the transfer with `compute_wait_fence`
            // (the exact `htod_async` + `record_copy_fence` chain `upload_async`
            // composes). With the fence the consumer reads the paged-in payload;
            // delete the fence and it reads the poison below.
            unsafe { runtime.htod(poison_bytes, pos_dst_p) }.unwrap();
            runtime.synchronize().unwrap();

            let mut delay = runtime.copy_stream().launch_builder(&spin_delay);
            delay.arg(&spin);
            unsafe { delay.launch(LaunchConfig::for_num_elems(1)).unwrap() };

            unsafe { runtime.htod_async(staging.as_slice(), pos_dst_p) }.unwrap();
            let fence = runtime.record_copy_fence().unwrap();
            runtime.compute_wait_fence(fence).unwrap();

            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&pos_dst_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            let mut got = vec![0.0f32; n];
            let got_bytes =
                unsafe { std::slice::from_raw_parts_mut(got.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(got_bytes, out_p) }.unwrap();
            runtime.sync_copy_stream().unwrap();
            assert_eq!(
                got, payload,
                "async page-in consumer read stale bytes — compute_wait_fence did \
                 not order the transfer before the compute-stream read"
            );

            // ── Negative (deterministic poison control): event-order the transfer
            // strictly AFTER the consumer, so with NO `compute_wait_fence` the
            // consumer provably reads pre-transfer poison. The `copy_wait_fence`
            // on a compute-stream fence removes all wall-clock racing — the
            // outcome never depends on the consumer winning against a delayed copy.
            unsafe { runtime.htod(poison_bytes, neg_dst_p) }.unwrap();
            runtime.synchronize().unwrap();

            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&neg_dst_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            // Hold the transfer until the consumer above has read `neg_dst`.
            let consumer_fence = runtime.record_compute_fence().unwrap();
            runtime.copy_wait_fence(consumer_fence).unwrap();
            unsafe { runtime.htod_async(staging.as_slice(), neg_dst_p) }.unwrap();
            let _unused_fence = runtime.record_copy_fence().unwrap();

            let mut raced = vec![0.0f32; n];
            let raced_bytes =
                unsafe { std::slice::from_raw_parts_mut(raced.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(raced_bytes, out_p) }.unwrap();
            // Drain the transfer (which lands after the consumer) before the next
            // iteration reuses `neg_dst` / `staging`.
            runtime.sync_copy_stream().unwrap();
            assert_eq!(
                raced, poison,
                "un-ordered async page-in consumer did NOT read poison — the \
                 compute-stream wait is not load-bearing, so this lock proves nothing"
            );

            // ── Real `upload_async` entry point: allocate + stage + async-copy +
            // fence, then a fenced consumer must observe the byte-identical
            // payload. Keeps the production API (not just its primitive chain)
            // under regression cover.
            let staging = runtime.alloc_pinned(payload_bytes.len()).unwrap();
            let (page, page_fence, staging) = crate::weight_paging::CudaWeightPage::upload_async(
                &runtime,
                DataType::Float32,
                vec![n],
                payload_bytes,
                staging,
            )
            .unwrap();
            runtime.compute_wait_fence(page_fence).unwrap();
            drop(staging);
            let page_p = cuptr(page.device_ptr());
            let mut consume = runtime.stream().launch_builder(&copy_out);
            consume.arg(&page_p).arg(&out_p).arg(&n_u64);
            unsafe {
                consume
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .unwrap()
            };
            let mut paged = vec![0.0f32; n];
            let paged_bytes =
                unsafe { std::slice::from_raw_parts_mut(paged.as_mut_ptr().cast::<u8>(), bytes) };
            unsafe { runtime.dtoh(paged_bytes, out_p) }.unwrap();
            assert_eq!(
                paged, payload,
                "upload_async page-in read stale bytes — the returned copy fence \
                 did not order the transfer before the compute-stream read"
            );
            drop(page);
        }

        ep.deallocate(pos_dst).unwrap();
        ep.deallocate(neg_dst).unwrap();
        ep.deallocate(out).unwrap();
    }
}
