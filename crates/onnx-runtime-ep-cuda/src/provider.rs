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

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch,
    OpRegistry, Result, deny,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::kernels::build_cuda_registry_with_metrics;
use crate::kernels::csa_checkpoint::CsaMetrics;
use crate::optimizer::cuda_optimization_passes;
use crate::runtime::{CudaRuntime, cuptr, raw_ptr};

/// CUDA execution provider (Phase 2a: cudarc + cuBLASLt GEMM).
///
/// Unlike the always-available CPU EP, this provider needs a real device, so
/// [`CudaExecutionProvider::new`] is **fallible** — it returns an error when no
/// CUDA device is present or the driver / cuBLASLt cannot be loaded. Callers on
/// a machine without a GPU should treat that error as "CUDA EP unavailable".
pub struct CudaExecutionProvider {
    device: DeviceId,
    runtime: Arc<CudaRuntime>,
    initialized: bool,
    registry: OpRegistry,
    csa_metrics: Arc<CsaMetrics>,
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
        let runtime = Arc::new(CudaRuntime::new(ordinal)?);
        let csa_metrics = Arc::new(CsaMetrics::default());
        let registry = build_cuda_registry_with_metrics(runtime.clone(), csa_metrics.clone());
        Ok(Self {
            device: DeviceId::cuda(ordinal),
            runtime,
            initialized: false,
            registry,
            csa_metrics,
        })
    }

    /// Construct and initialize a CUDA execution provider with default settings.
    pub fn initialized(ordinal: u32) -> Result<Self> {
        let mut provider = Self::new(ordinal)?;
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
        if op.op_type == "Attention"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::standard_attention::unsupported_reason(opset, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "RotaryEmbedding"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) = crate::kernels::rotary_embedding::unsupported_reason(input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::standard_claims::unsupported_reason(op, input_dtypes)
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
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        // cuMemAlloc returns at least 256-byte-aligned device pointers, which
        // satisfies any realistic tensor alignment; we still record the
        // requested `alignment` on the handle for symmetry with the CPU EP.
        let dptr = self.runtime.alloc_raw(size)?;
        // SAFETY: `dptr` is a fresh, unique, non-null device allocation of
        // >= `size` bytes owned by this EP and freed exactly once in
        // `deallocate`. It is a device address, never dereferenced on the host.
        Ok(unsafe { DeviceBuffer::from_raw_parts(raw_ptr(dptr), self.device, size, alignment) })
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
        let dptr = cuptr(buffer.into_raw());
        // SAFETY: `dptr` came from this EP's `alloc_raw`; `into_raw` consumed the
        // owning handle so no alias remains, and this is its single free.
        unsafe { self.runtime.free_raw(dptr) }
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
        let counts = self.runtime.allocation_counts();
        Some((counts.allocations, counts.frees))
    }

    fn sync(&self) -> Result<()> {
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
