//! Production capability probe for CUDA VMM HOST_NUMA physical backing.
//!
//! # Why a probe rather than a query
//!
//! Driver attributes (`CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`)
//! advertise capability but do not guarantee it: a platform that claims support
//! and then returns `CUDA_ERROR_NOT_SUPPORTED` on the first real allocation is
//! indistinguishable from an unsupported one unless we attempt an allocation.
//! This module does that: it gates on attributes first (fast), then confirms
//! with one real `cuMemCreate` + `cuMemGetAllocationGranularity` (slow, once,
//! cached).
//!
//! # Fail-closed contract
//!
//! Every function here returns `Err(CapabilityGateFailure)` rather than any
//! fallback or `None` if any step does not succeed. Callers that require
//! HOST_NUMA must treat `Err` as terminal; this crate never silently degrades
//! to a different backing.
//!
//! # Caching
//!
//! The probe is expensive (one allocation round-trip per device). Results are
//! cached process-wide, keyed by device ordinal, so repeated calls do not
//! re-probe.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

use cudarc::driver::sys as cu;

/// The result of a successful HOST_NUMA capability probe for one device.
#[derive(Clone, Debug)]
pub struct HostNumaCapability {
    /// The device ordinal this capability was probed on.
    pub device_ordinal: i32,
    /// Whether CUDA VMM is supported at all on this device.
    pub vmm_supported: bool,
    /// The host NUMA node the driver associates with this device.
    pub host_numa_id: i32,
    /// Whether HOST_NUMA VMM-mapped physical handles are supported.
    pub host_numa_vmm_supported: bool,
    /// Recommended granularity for HOST_NUMA allocations (bytes).
    pub granularity: usize,
}

/// Why a capability probe failed.
#[derive(Clone, Debug)]
pub enum CapabilityGateFailure {
    /// The probe ran but the platform does not support this capability, for the
    /// stated reason.
    Unsupported(String),
}

impl std::fmt::Display for CapabilityGateFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilityGateFailure::Unsupported(reason) => write!(formatter, "{reason}"),
        }
    }
}

// Process-wide cache keyed by device ordinal. `Ok` is cached after a
// successful probe; `Err` is NOT cached (a transient failure should be
// retriable). Each entry is `Arc`-free: the `HostNumaCapability` is `Clone`
// and cheap to copy.
static CAPABILITY_CACHE: OnceLock<Mutex<HashMap<i32, HostNumaCapability>>> = OnceLock::new();

fn capability_cache() -> &'static Mutex<HashMap<i32, HostNumaCapability>> {
    CAPABILITY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Query and confirm CUDA VMM HOST_NUMA physical-backing support for
/// `device_ordinal`, with the result cached process-wide.
///
/// Performs (once per device per process):
/// 1. `cuInit` — driver must be live.
/// 2. `CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED` — VMM at all.
/// 3. `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` — device's preferred NUMA node.
/// 4. `CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` —
///    HOST_NUMA VMM handles specifically.
/// 5. `cuMemGetAllocationGranularity` with `CU_MEM_LOCATION_TYPE_HOST_NUMA` —
///    confirm the driver reports a usable granularity.
/// 6. One real `cuMemCreate` + `cuMemRelease` with `CU_MEM_LOCATION_TYPE_HOST_NUMA` —
///    confirm the platform actually honors what it advertises (not just the
///    attribute query).
///
/// Returns `Err(CapabilityGateFailure::Unsupported)` with an explicit reason
/// string on any failure. Never falls back to device-only backing silently.
pub fn host_numa_capability(
    device_ordinal: i32,
) -> Result<HostNumaCapability, CapabilityGateFailure> {
    {
        let cache = capability_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&device_ordinal) {
            return Ok(cached.clone());
        }
    }

    let result = probe_host_numa_capability(device_ordinal)?;

    {
        let mut cache = capability_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.insert(device_ordinal, result.clone());
    }
    Ok(result)
}

fn probe_host_numa_capability(
    device_ordinal: i32,
) -> Result<HostNumaCapability, CapabilityGateFailure> {
    // SAFETY: all raw driver calls use valid out-parameters and are checked
    // immediately for CUDA_SUCCESS before any further use of the result.
    unsafe {
        let init = cu::cuInit(0);
        if init != cu::CUresult::CUDA_SUCCESS {
            return Err(CapabilityGateFailure::Unsupported(format!(
                "cuInit failed: {init:?}"
            )));
        }

        let mut dev: cu::CUdevice = 0;
        let r = cu::cuDeviceGet(&mut dev, device_ordinal);
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(CapabilityGateFailure::Unsupported(format!(
                "cuDeviceGet({device_ordinal}) failed: {r:?}"
            )));
        }

        let mut vmm_supported = 0i32;
        cu::cuDeviceGetAttribute(
            &mut vmm_supported,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            dev,
        );
        if vmm_supported == 0 {
            return Err(CapabilityGateFailure::Unsupported(
                "CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED=0: this device does \
                 not support CUDA VMM"
                    .into(),
            ));
        }

        let mut host_numa_id = -1i32;
        let r2 = cu::cuDeviceGetAttribute(
            &mut host_numa_id,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            dev,
        );
        if r2 != cu::CUresult::CUDA_SUCCESS || host_numa_id < 0 {
            return Err(CapabilityGateFailure::Unsupported(format!(
                "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID query failed or returned no NUMA node \
                 (result={r2:?}, value={host_numa_id})"
            )));
        }

        let mut host_numa_vmm = 0i32;
        let r3 = cu::cuDeviceGetAttribute(
            &mut host_numa_vmm,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            dev,
        );
        if r3 != cu::CUresult::CUDA_SUCCESS || host_numa_vmm == 0 {
            return Err(CapabilityGateFailure::Unsupported(format!(
                "CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=0 \
                 (result={r3:?}): this driver/device does not support VMM-mapped HOST_NUMA \
                 physical handles"
            )));
        }

        let mut prop: cu::CUmemAllocationProp = std::mem::zeroed();
        prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = host_numa_id;

        let mut granularity = 0usize;
        let rg = cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        );
        if rg != cu::CUresult::CUDA_SUCCESS || granularity == 0 {
            return Err(CapabilityGateFailure::Unsupported(format!(
                "cuMemGetAllocationGranularity(HOST_NUMA) failed or returned 0 \
                 (result={rg:?}, value={granularity})"
            )));
        }

        // Confirm with a real allocation: advertised support that fails here
        // means the platform is not actually capable.
        let mut handle: cu::CUmemGenericAllocationHandle = 0;
        let rc = cu::cuMemCreate(&mut handle, granularity, &prop, 0);
        if rc != cu::CUresult::CUDA_SUCCESS {
            return Err(CapabilityGateFailure::Unsupported(format!(
                "cuMemCreate(HOST_NUMA, node={host_numa_id}) failed despite advertised support: \
                 {rc:?}"
            )));
        }
        cu::cuMemRelease(handle);

        Ok(HostNumaCapability {
            device_ordinal,
            vmm_supported: true,
            host_numa_id,
            host_numa_vmm_supported: true,
            granularity,
        })
    }
}
