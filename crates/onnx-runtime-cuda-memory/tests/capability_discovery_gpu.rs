//! Headline guarantee of the discovery-only capability split (#1186).
//!
//! The contract makes exactly three checkable claims, and no more. Each test
//! here fails when its claim is violated — that is the point. The previous
//! design (PR #1192, reverted by #1247) asserted a mechanism identity it could
//! not prove; this phase asserts only what a test can break:
//!
//! 1. Capability presence tracks a **construction-time** input, not a runtime
//!    guess. A CUDA VMM allocator built without a shared physical-handle pool
//!    advertises virtual backing but **not** shared mapping; built with one, it
//!    advertises both. Falsify by making `as_shared_mapping` ignore the pool
//!    and this file's `..._without_a_pool_does_not_advertise_shared_mapping`
//!    goes red.
//! 2. A discovered capability reports the **same device** as the allocator that
//!    vended it — the only identity claim the contract makes.
//! 3. An eager allocator advertises **neither** capability, so a caller can
//!    distinguish "does not implement commit" from "committed successfully".
//!
//! Notably absent: any assertion about release, refund, decommit, or an
//! unforgeable mechanism handle. Those were the five reverted hazards and this
//! contract exposes no surface that could exhibit them.

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::device_allocator::CudaDeviceAllocator;
use onnx_runtime_cuda_memory::vmm_allocator::{CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};

const HOLDER: HolderId = HolderId::new(1186);

/// Build a VMM allocator on device 0. `pool_bytes` is set through the
/// production environment switch before construction; `None` clears it so the
/// allocator comes up with no shared physical-handle pool. `--test-threads=1`
/// (mandated for these tests) keeps this process-global switch deterministic.
fn vmm(pool_bytes: Option<usize>) -> (CudaVmmAllocator, LedgerGovernor) {
    // SAFETY: this integration test owns its process and sets the option before
    // constructing any allocator or spawning any thread.
    unsafe {
        match pool_bytes {
            Some(bytes) => {
                std::env::set_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, bytes.to_string())
            }
            None => std::env::remove_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV),
        }
    }
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "capability-discovery test requires a CUDA driver; CPU-only runs must leave it ignored: {error}"
        ),
    };
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = CudaVmmAllocator::new(
        context,
        DeviceKey::device(0),
        0,
        64 << 20,
        &governor,
        HOLDER,
        MemoryRole::Weights,
    )
    .expect("reserving device address space");
    (allocator, governor)
}

/// GUARANTEE 1a + 2: a VMM allocator built without a shared physical-handle
/// pool advertises virtual backing (it genuinely commits on demand) but **not**
/// shared mapping (it has no pool to share), and the virtual-backing capability
/// reports the same device as the allocator.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_vmm_allocator_without_a_pool_does_not_advertise_shared_mapping() {
    let (allocator, _governor) = vmm(None);
    let handle: &dyn DeviceAllocator = &allocator;

    let backing = handle
        .as_virtual_backing()
        .expect("a VMM allocator commits on demand and must advertise virtual backing");
    assert_eq!(
        backing.device(),
        handle.device(),
        "the virtual-backing capability must report the allocator's own device"
    );

    assert!(
        handle.as_shared_mapping().is_none(),
        "an allocator built with no shared physical-handle pool must not advertise shared \
         mapping: doing so would promise a zero-cost shared prefix it cannot create"
    );
    // The construction input the discovery answer must track.
    assert!(
        allocator.physical_pool_authority().is_none(),
        "test premise: this allocator was built without a pool"
    );
}

/// GUARANTEE 1b + 2: a VMM allocator built with a shared physical-handle pool
/// advertises **both** capabilities, and each reports the allocator's device.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_vmm_allocator_with_a_pool_advertises_both_capabilities() {
    let (allocator, _governor) = vmm(Some(64 << 20));
    // SAFETY: restore the process default now that construction has read it.
    unsafe { std::env::remove_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV) };

    // Skip cleanly if this machine's driver refused to install the pool, so the
    // test asserts the split — not the presence of a production pool.
    if allocator.physical_pool_authority().is_none() {
        eprintln!("SKIPPED: driver did not install a shared physical-handle pool");
        return;
    }

    let handle: &dyn DeviceAllocator = &allocator;
    let backing = handle
        .as_virtual_backing()
        .expect("a pooled VMM allocator must advertise virtual backing");
    assert_eq!(backing.device(), handle.device());

    let shared = handle
        .as_shared_mapping()
        .expect("an allocator with a shared physical-handle pool must advertise shared mapping");
    assert_eq!(
        shared.device(),
        handle.device(),
        "the shared-mapping capability must report the allocator's own device"
    );
}

/// GUARANTEE 3: an eager allocator advertises neither capability, so a caller
/// gets an unambiguous "this capability does not exist" rather than a
/// successful-looking no-op.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn an_eager_cuda_allocator_advertises_neither_capability() {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "capability-discovery test requires a CUDA driver; CPU-only runs must leave it ignored: {error}"
        ),
    };
    let allocator = CudaDeviceAllocator::new(context);
    let handle: &dyn DeviceAllocator = &allocator;

    assert!(
        handle.as_virtual_backing().is_none(),
        "an eager cuMemAlloc allocator has no lazy-commit mechanism and must not advertise one"
    );
    assert!(
        handle.as_shared_mapping().is_none(),
        "an eager cuMemAlloc allocator has no shared physical-handle pool and must not advertise \
         shared mapping"
    );
}
