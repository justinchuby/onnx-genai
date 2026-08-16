//! Capability-negotiated lazy weight handles for executor-to-EP delivery.

use std::collections::BTreeSet;
use std::sync::Arc;

use onnx_runtime_ir::{DataType, Graph, NodeId, ValueId};

use crate::ExternalMmapRegion;

/// Capability flag advertised by paging-aware execution providers.
pub const NXRT_WEIGHT_PAGING_CAPABILITY: &str = "nxrt";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionProviderCapabilities {
    flags: BTreeSet<String>,
}

impl ExecutionProviderCapabilities {
    pub fn stock() -> Self {
        Self::default()
    }

    pub fn nxrt_weight_paging() -> Self {
        Self::from_flags([NXRT_WEIGHT_PAGING_CAPABILITY])
    }

    pub fn from_flags(flags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            flags: flags.into_iter().map(Into::into).collect(),
        }
    }

    pub fn advertises(&self, capability: &str) -> bool {
        self.flags.contains(capability)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentWeight {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    bytes: Arc<[u8]>,
}

impl ResidentWeight {
    pub fn new(
        dtype: DataType,
        shape: Vec<usize>,
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<Self, WeightHandleError> {
        let elements = checked_shape_product(&shape)?;
        let expected = dtype.checked_storage_bytes(elements).ok_or_else(|| {
            WeightHandleError::InvalidResident("resident weight byte count overflow".into())
        })?;
        if expected > isize::MAX as usize {
            return Err(WeightHandleError::InvalidResident(
                "resident weight byte count exceeds isize::MAX".into(),
            ));
        }
        let bytes = bytes.into();
        if bytes.len() != expected {
            return Err(WeightHandleError::InvalidResident(format!(
                "resident weight has {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

fn checked_shape_product(shape: &[usize]) -> Result<usize, WeightHandleError> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &dimension in shape {
        if dimension == 0 {
            has_zero = true;
        } else {
            product = product.checked_mul(dimension).ok_or_else(|| {
                WeightHandleError::InvalidResident("resident weight element count overflow".into())
            })?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LazyWeightBoundary {
    /// `ai.onnx::MatMul`, the ordinary dense GEMM boundary used by unquantized
    /// decoder exports.
    MatMul,
    /// `pkg.nxrt::BlockQuantizedMoE`, the Phase-3 offload binding boundary.
    BlockQuantizedMoe,
    /// `com.microsoft::QMoE`, the boundary real DeepSeek/GLM/Qwen exports use.
    QMoe,
    /// `com.microsoft::MatMulNBits`, the packed INT4/INT8 GEMV boundary that
    /// dominates dense-model (e.g. Qwen2.5) VRAM.
    MatMulNBits,
}

impl LazyWeightBoundary {
    /// Every op boundary at which a lazy weight may be device-paged.
    pub const ALL: [LazyWeightBoundary; 4] = [
        Self::MatMul,
        Self::BlockQuantizedMoe,
        Self::QMoe,
        Self::MatMulNBits,
    ];

    /// Canonical (domain, op_type) this boundary binds at.
    fn identity(self) -> (&'static str, &'static str) {
        match self {
            Self::MatMul => ("", "MatMul"),
            Self::BlockQuantizedMoe => ("pkg.nxrt", "BlockQuantizedMoE"),
            Self::QMoe => ("com.microsoft", "QMoE"),
            Self::MatMulNBits => ("com.microsoft", "MatMulNBits"),
        }
    }

    pub fn matches(self, domain: &str, op_type: &str) -> bool {
        let (want_domain, want_op) = self.identity();
        domain == want_domain && op_type == want_op
    }

    /// The offload boundary that binds `(domain, op_type)`, if any.
    pub fn for_op(domain: &str, op_type: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|boundary| boundary.matches(domain, op_type))
    }

    /// Whether any offload boundary binds `(domain, op_type)`.
    pub fn matches_any(domain: &str, op_type: &str) -> bool {
        Self::for_op(domain, op_type).is_some()
    }
}

/// An initializer the executor may expose as a lazy weight handle.
///
/// Strategy inference and executor construction share this classifier so the
/// reported pageable geometry cannot drift from the weights the runtime pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LazyWeightCandidate {
    pub value: ValueId,
    pub boundary: LazyWeightBoundary,
    pub first_consumer: NodeId,
}

pub fn lazy_weight_candidates(graph: &Graph) -> Vec<LazyWeightCandidate> {
    let mut candidates = Vec::new();
    for &value in graph.initializers.keys() {
        let graph_value = graph.value(value);
        let consumers = graph.consumers(value);
        let Some(&first_consumer) = consumers.first() else {
            continue;
        };
        let mut boundary = None;
        let lazy_only = graph_value.producer.is_none()
            && !graph.outputs.contains(&value)
            && consumers.into_iter().all(|consumer| {
                let node = graph.node(consumer);
                match LazyWeightBoundary::for_op(&node.domain, &node.op_type) {
                    Some(found) => {
                        boundary.get_or_insert(found);
                        true
                    }
                    None => false,
                }
            });
        if let Some(boundary) = boundary.filter(|_| lazy_only) {
            candidates.push(LazyWeightCandidate {
                value,
                boundary,
                first_consumer,
            });
        }
    }
    candidates
}

pub trait ResidentWeightMaterializer: Send + Sync {
    fn materialize(&self) -> Result<ResidentWeight, WeightHandleError>;
}

impl<F> ResidentWeightMaterializer for F
where
    F: Fn() -> Result<ResidentWeight, WeightHandleError> + Send + Sync,
{
    fn materialize(&self) -> Result<ResidentWeight, WeightHandleError> {
        self()
    }
}

#[derive(Clone)]
pub struct LazyWeight {
    pub boundary: LazyWeightBoundary,
    /// Canonical element type of the backing tensor.
    pub dtype: DataType,
    /// Canonical shape of the backing tensor.
    pub shape: Vec<usize>,
    /// Validated external mmap ranges that back this initializer, in binding
    /// order. Their lengths sum to the canonical byte size of the tensor.
    pub regions: Vec<ExternalMmapRegion>,
    resident_materializer: Arc<dyn ResidentWeightMaterializer>,
}

impl std::fmt::Debug for LazyWeight {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LazyWeight")
            .field("boundary", &self.boundary)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("regions", &self.regions)
            .field("resident_materializer", &"<deferred>")
            .finish()
    }
}

impl LazyWeight {
    /// Build a lazy weight bound at an arbitrary offload boundary.
    pub fn new<M>(
        boundary: LazyWeightBoundary,
        dtype: DataType,
        shape: Vec<usize>,
        regions: Vec<ExternalMmapRegion>,
        resident_materializer: M,
    ) -> Result<Self, WeightHandleError>
    where
        M: ResidentWeightMaterializer + 'static,
    {
        if regions.is_empty() {
            return Err(WeightHandleError::MissingRegions);
        }
        Ok(Self {
            boundary,
            dtype,
            shape,
            regions,
            resident_materializer: Arc::new(resident_materializer),
        })
    }

    pub fn block_quantized_moe<M>(
        dtype: DataType,
        shape: Vec<usize>,
        regions: Vec<ExternalMmapRegion>,
        resident_materializer: M,
    ) -> Result<Self, WeightHandleError>
    where
        M: ResidentWeightMaterializer + 'static,
    {
        Self::new(
            LazyWeightBoundary::BlockQuantizedMoe,
            dtype,
            shape,
            regions,
            resident_materializer,
        )
    }

    /// Total canonical byte size of the backing tensor, summed across regions.
    pub fn region_bytes_len(&self) -> usize {
        self.regions.iter().map(|region| region.len).sum()
    }

    /// Materialize the unchanged stock-EP resident behavior.
    pub fn materialize(&self) -> Result<ResidentWeight, WeightHandleError> {
        self.resident_materializer.materialize()
    }
}

/// General executor weight input: resident today, lazy when an EP opts in.
#[derive(Clone, Debug)]
pub enum WeightHandle {
    Resident(ResidentWeight),
    Lazy(LazyWeight),
}

impl WeightHandle {
    pub fn negotiate(
        &self,
        capabilities: &ExecutionProviderCapabilities,
    ) -> Result<NegotiatedWeight, WeightHandleError> {
        match self {
            Self::Resident(weight) => Ok(NegotiatedWeight::Resident(weight.clone())),
            Self::Lazy(weight) if capabilities.advertises(NXRT_WEIGHT_PAGING_CAPABILITY) => {
                Ok(NegotiatedWeight::Lazy(weight.clone()))
            }
            Self::Lazy(weight) => Ok(NegotiatedWeight::Resident(weight.materialize()?)),
        }
    }

    pub fn is_lazy_for(&self, capabilities: &ExecutionProviderCapabilities) -> bool {
        matches!(self, Self::Lazy(_)) && capabilities.advertises(NXRT_WEIGHT_PAGING_CAPABILITY)
    }

    /// Borrow the inner [`LazyWeight`] when this handle is lazy.
    pub fn as_lazy(&self) -> Option<&LazyWeight> {
        match self {
            Self::Lazy(weight) => Some(weight),
            Self::Resident(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NegotiatedWeight {
    Resident(ResidentWeight),
    Lazy(LazyWeight),
}

impl NegotiatedWeight {
    /// Explicit host route available to every executor and EP.
    pub fn materialize_host_fallback(&self) -> Result<ResidentWeight, WeightHandleError> {
        match self {
            Self::Resident(weight) => Ok(weight.clone()),
            Self::Lazy(weight) => weight.materialize(),
        }
    }

    /// Phase 3b will call this at `pkg.nxrt::BlockQuantizedMoE` binding time.
    pub fn try_bind_device<B: LazyDeviceWeightBinder>(
        &self,
        binder: &B,
    ) -> Result<B::Binding, WeightHandleError> {
        match self {
            Self::Resident(_) => Err(WeightHandleError::Unsupported(
                "resident weights do not require lazy device binding".into(),
            )),
            Self::Lazy(weight) => binder.bind_block_quantized_moe(weight),
        }
    }
}

/// EP seam for Phase-3b live device paging.
pub trait LazyDeviceWeightBinder {
    type Binding;

    fn bind_block_quantized_moe(
        &self,
        weight: &LazyWeight,
    ) -> Result<Self::Binding, WeightHandleError>;
}

/// Resolves the live host bytes backing a validated external mmap region.
///
/// A paging-capable device binder uses this to copy only the selected region
/// bytes host→device, rather than materializing the whole resident tensor on the
/// host first (WEIGHT_OFFLOAD §9 invariant 5: never allocate an unbudgeted full
/// expansion). The executor's weight store owns the live mappings and implements
/// this; the returned slice must stay valid for the duration of the copy.
pub trait MmapRegionSource {
    fn region_bytes(&self, region: &ExternalMmapRegion) -> Result<&[u8], WeightHandleError>;

    /// Return the whole live mapping identified by `mapping_id`, if available.
    ///
    /// Used by the zero-copy hybrid weight path to page-lock and device-map an
    /// entire mapping in a single `cuMemHostRegister`, guaranteeing that every
    /// weight's device pointer is contiguous over its full length. Returning
    /// `None` disables zero-copy for that mapping (the caller falls back to a
    /// copy). The returned slice must stay valid for the mapping's lifetime.
    fn full_mapping_bytes(&self, _mapping_id: usize) -> Option<&[u8]> {
        None
    }
}

/// A lazy weight paged into device memory by an EP, ready for a kernel to read.
///
/// The executor substitutes its [`device_ptr`](Self::device_ptr) into the input
/// `TensorView` for the weight and holds this value for the kernel's lifetime.
/// `keep_alive` owns whatever device residency the EP allocated (e.g. a VRAM
/// page), so the memory stays resident until the executor drops the binding —
/// after the kernel has run.
pub struct PagedWeight {
    device_ptr: *const std::ffi::c_void,
    device: onnx_runtime_ir::DeviceId,
    len: usize,
    keep_alive: Arc<dyn std::any::Any + Send + Sync>,
}

impl PagedWeight {
    pub fn new(
        device_ptr: *const std::ffi::c_void,
        device: onnx_runtime_ir::DeviceId,
        len: usize,
        keep_alive: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            device_ptr,
            device,
            len,
            keep_alive,
        }
    }

    /// Opaque device pointer to the paged weight bytes.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        self.device_ptr
    }

    /// Device the paged bytes live on.
    pub fn device(&self) -> onnx_runtime_ir::DeviceId {
        self.device
    }

    /// Number of bytes resident in the paged allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the paged allocation is empty (never true for a valid page).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the residency keep-alive (for downcasting / observability).
    pub fn keep_alive(&self) -> &Arc<dyn std::any::Any + Send + Sync> {
        &self.keep_alive
    }
}

impl std::fmt::Debug for PagedWeight {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PagedWeight")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// CPU-only Phase-3a binder: callers must use the host materialization route.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phase3aHostOnlyBinder;

impl LazyDeviceWeightBinder for Phase3aHostOnlyBinder {
    type Binding = ();

    fn bind_block_quantized_moe(
        &self,
        _weight: &LazyWeight,
    ) -> Result<Self::Binding, WeightHandleError> {
        Err(WeightHandleError::Unsupported(
            "live device weight paging is deferred to WEIGHT_OFFLOAD Phase 3b".into(),
        ))
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum WeightHandleError {
    #[error("invalid resident weight: {0}")]
    InvalidResident(String),
    #[error("lazy weight requires at least one external mmap region")]
    MissingRegions,
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("device weight binding failed: {0}")]
    DeviceBinding(String),
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn resident() -> ResidentWeight {
        ResidentWeight::new(DataType::Uint8, vec![4], vec![1, 2, 3, 4]).unwrap()
    }

    fn region() -> ExternalMmapRegion {
        ExternalMmapRegion {
            mapping_id: 7,
            offset: 100,
            len: 4,
        }
    }

    fn lazy() -> WeightHandle {
        WeightHandle::Lazy(
            LazyWeight::block_quantized_moe(DataType::Uint8, vec![4], vec![region()], || {
                Ok(resident())
            })
            .unwrap(),
        )
    }

    #[test]
    fn stock_ep_materializes_the_resident_fallback() {
        let NegotiatedWeight::Resident(weight) = lazy()
            .negotiate(&ExecutionProviderCapabilities::stock())
            .unwrap()
        else {
            panic!("stock EP must receive resident materialization");
        };
        assert_eq!(weight.bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn nxrt_capability_preserves_lazy_block_quantized_moe_handle() {
        let materializations = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&materializations);
        let lazy = WeightHandle::Lazy(
            LazyWeight::block_quantized_moe(DataType::Uint8, vec![4], vec![region()], move || {
                counter.fetch_add(1, Ordering::Relaxed);
                Ok(resident())
            })
            .unwrap(),
        );
        let NegotiatedWeight::Lazy(weight) = lazy
            .negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())
            .unwrap()
        else {
            panic!("nxrt EP must receive lazy weight handle");
        };
        assert_eq!(weight.boundary, LazyWeightBoundary::BlockQuantizedMoe);
        assert_eq!(weight.regions, vec![region()]);
        assert_eq!(materializations.load(Ordering::Relaxed), 0);
        assert_eq!(weight.materialize().unwrap().bytes(), &[1, 2, 3, 4]);
        assert_eq!(materializations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn offload_boundary_recognizes_dense_and_moe_boundaries() {
        assert_eq!(
            LazyWeightBoundary::for_op("", "MatMul"),
            Some(LazyWeightBoundary::MatMul)
        );
        assert_eq!(
            LazyWeightBoundary::for_op("pkg.nxrt", "BlockQuantizedMoE"),
            Some(LazyWeightBoundary::BlockQuantizedMoe)
        );
        assert_eq!(
            LazyWeightBoundary::for_op("com.microsoft", "QMoE"),
            Some(LazyWeightBoundary::QMoe)
        );
        assert!(LazyWeightBoundary::matches_any("com.microsoft", "QMoE"));
        assert!(LazyWeightBoundary::matches_any(
            "pkg.nxrt",
            "BlockQuantizedMoE"
        ));
        assert!(LazyWeightBoundary::matches_any("", "MatMul"));
        // Wrong domain/op pairings and unrelated ops are not offload boundaries.
        assert_eq!(LazyWeightBoundary::for_op("pkg.nxrt", "QMoE"), None);
        assert_eq!(
            LazyWeightBoundary::for_op("com.microsoft", "BlockQuantizedMoE"),
            None
        );
        assert!(!LazyWeightBoundary::matches_any("ai.onnx", "MatMul"));
    }

    #[test]
    fn phase3a_device_binding_is_explicitly_unsupported_with_host_route() {
        let negotiated = lazy()
            .negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())
            .unwrap();
        assert_eq!(
            negotiated.try_bind_device(&Phase3aHostOnlyBinder),
            Err(WeightHandleError::Unsupported(
                "live device weight paging is deferred to WEIGHT_OFFLOAD Phase 3b".into()
            ))
        );
        assert_eq!(
            negotiated.materialize_host_fallback().unwrap().bytes(),
            &[1, 2, 3, 4]
        );
    }
}
