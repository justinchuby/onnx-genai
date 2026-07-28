//! Live GPU weight paging (WEIGHT_OFFLOAD Phase 3b): allocate a bounded VRAM
//! page for one offloaded `pkg.nxrt::BlockQuantizedMoE` weight tensor, copy its
//! canonical compressed bytes host→device, and hand back an owned device
//! binding a kernel can read.
//!
//! This is the CUDA implementation of the [`LazyDeviceWeightBinder`] seam that
//! Phase 3a stubbed out with [`Phase3aHostOnlyBinder`]. It copies only the
//! selected external-data region bytes (resolved through [`MmapRegionSource`]),
//! never a full host expansion, so it upholds the WEIGHT_OFFLOAD §9 invariant
//! that residency never allocates an unbudgeted full expert/model expansion and
//! keeps compressed blocks in VRAM (§7.3). The copied bytes are the exact
//! canonical backing bytes, so an offloaded weight is byte-identical to the
//! resident-weight path — offload is an optimization, never an output change.
//!
//! Deferred (clean seams left in place): wiring this binder into the executor's
//! `pkg.nxrt::BlockQuantizedMoE` dispatch so the fused MoE kernel consumes the
//! device page, multi-page LRU eviction, and prefetch overlap (issues #82/#87).

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    LazyDeviceWeightBinder, LazyWeight, MmapRegionSource, WeightHandleError,
};
use onnx_runtime_ir::DataType;

use crate::runtime::{CudaRuntime, raw_ptr};

/// A live VRAM residency page for one offloaded weight tensor.
///
/// Owns a single device allocation holding the tensor's canonical compressed
/// bytes and frees it exactly once on drop. The address is a CUDA device
/// pointer — never dereferenced on the host — exposed through [`Self::device_ptr`]
/// for a consuming kernel's `TensorView`.
pub struct CudaWeightPage {
    runtime: Arc<CudaRuntime>,
    ptr: CUdeviceptr,
    len: usize,
    dtype: DataType,
    shape: Vec<usize>,
}

impl CudaWeightPage {
    /// Opaque device pointer to the paged bytes, for a kernel `TensorView`.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        raw_ptr(self.ptr)
    }

    /// Number of canonical bytes resident in this VRAM page.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the page is empty (never true for a validated binding).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Canonical element type of the paged tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Canonical shape of the paged tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl std::fmt::Debug for CudaWeightPage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaWeightPage")
            .field("len", &self.len)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl Drop for CudaWeightPage {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from this runtime's `alloc_raw` in `bind_block_quantized_moe`
        // and is freed exactly once here; no alias to it escapes `CudaWeightPage`.
        let _ = unsafe { self.runtime.free_raw(self.ptr) };
    }
}

/// CUDA Phase-3b device binder: pages one offloaded weight tensor into VRAM.
///
/// Copies the canonical compressed region bytes host→device, so the device page
/// is byte-identical to the resident tensor a stock EP would upload.
pub struct CudaWeightPager<'a, S: MmapRegionSource> {
    runtime: Arc<CudaRuntime>,
    source: &'a S,
}

impl<'a, S: MmapRegionSource> CudaWeightPager<'a, S> {
    pub fn new(runtime: Arc<CudaRuntime>, source: &'a S) -> Self {
        Self { runtime, source }
    }
}

impl<S: MmapRegionSource> LazyDeviceWeightBinder for CudaWeightPager<'_, S> {
    type Binding = CudaWeightPage;

    fn bind_block_quantized_moe(
        &self,
        weight: &LazyWeight,
    ) -> Result<Self::Binding, WeightHandleError> {
        let total = weight.region_bytes_len();
        if total == 0 {
            return Err(WeightHandleError::MissingRegions);
        }

        // Allocate the VRAM page up front, then stream each selected region into
        // its contiguous slot. The page is owned by `CudaWeightPage` on success;
        // on any copy failure we free it before returning so no VRAM leaks.
        let ptr = self
            .runtime
            .alloc_raw(total)
            .map_err(|error| WeightHandleError::DeviceBinding(format!("VRAM alloc: {error}")))?;
        let page = CudaWeightPage {
            runtime: Arc::clone(&self.runtime),
            ptr,
            len: total,
            dtype: weight.dtype,
            shape: weight.shape.clone(),
        };

        let mut offset: usize = 0;
        for region in &weight.regions {
            let bytes = self.source.region_bytes(region)?;
            if bytes.len() != region.len {
                return Err(WeightHandleError::DeviceBinding(format!(
                    "region source returned {} bytes for a {}-byte region",
                    bytes.len(),
                    region.len
                )));
            }
            let dst = ptr + offset as CUdeviceptr;
            // SAFETY: `dst` lies within the `total`-byte allocation `page` owns
            // ([offset, offset + region.len) with the running sum bounded by
            // `total`); `bytes` covers exactly `region.len` bytes.
            unsafe { self.runtime.htod(bytes, dst) }
                .map_err(|error| WeightHandleError::DeviceBinding(format!("H2D copy: {error}")))?;
            offset += region.len;
        }

        Ok(page)
    }
}
