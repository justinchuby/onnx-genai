//! Capability-negotiated lazy weight handles for executor-to-EP delivery.

use std::collections::BTreeSet;
use std::sync::Arc;

use onnx_runtime_ir::DataType;

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
    /// `pkg.nxrt::BlockQuantizedMoE`, the Phase-3 offload binding boundary.
    BlockQuantizedMoe,
}

impl LazyWeightBoundary {
    pub fn matches(self, domain: &str, op_type: &str) -> bool {
        matches!(self, Self::BlockQuantizedMoe)
            && domain == "pkg.nxrt"
            && op_type == "BlockQuantizedMoE"
    }
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
    /// Validated external mmap ranges that back this initializer.
    pub regions: Vec<ExternalMmapRegion>,
    resident_materializer: Arc<dyn ResidentWeightMaterializer>,
}

impl std::fmt::Debug for LazyWeight {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LazyWeight")
            .field("boundary", &self.boundary)
            .field("regions", &self.regions)
            .field("resident_materializer", &"<deferred>")
            .finish()
    }
}

impl LazyWeight {
    pub fn block_quantized_moe<M>(
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
            boundary: LazyWeightBoundary::BlockQuantizedMoe,
            regions,
            resident_materializer: Arc::new(resident_materializer),
        })
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
            LazyWeight::block_quantized_moe(vec![region()], || Ok(resident())).unwrap(),
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
            LazyWeight::block_quantized_moe(vec![region()], move || {
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
