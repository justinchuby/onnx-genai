//! CUDA correctness and submission-count proof for fused indexed multi-bank
//! expert page-in. Run on an idle pinned GPU with `--features gpu-tests`.

use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use cudarc::driver::{result, sys};
use onnx_runtime_ep_api::{CaptureSupport, Kernel, TensorMut, TensorView};
use onnx_runtime_ep_cuda::deferred_release::{
    CudaDeferredReleaseQueue, CudaStreamFences, DEFAULT_DEFERRED_RELEASE_CAPACITY,
};
use onnx_runtime_ep_cuda::runtime::{CudaRuntime, FailedHtodCompletion};
use onnx_runtime_ep_cuda::weight_paging::CudaWeightResidency;
use onnx_runtime_ep_cuda::{
    ExpertSlot, IndexedBankOwner, IndexedBankVisibility, IndexedExpertBank, IndexedExpertBankSpec,
    IndexedMultiBankPageInPlan, IndexedPageInAttribution, IndexedPageInFailureDisposition,
    IndexedPageInPhase, ProjectionBankPair, execute_indexed_multi_bank_page_in,
    execute_indexed_multi_bank_page_in_with_partial_write_fault, global_offload_stats,
    reset_global_offload_stats,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

const EXPERTS: usize = 4;
const SLOTS: usize = 4;
const ROW_BYTES: usize = 256 * 1024;
static GPU_SERIAL: Mutex<()> = Mutex::new(());

struct CapturableTestKernel;

impl Kernel for CapturableTestKernel {
    fn execute(
        &self,
        _inputs: &[TensorView],
        _outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::Supported
    }
}

struct MappedHost {
    host: *mut u8,
    device: u64,
    bytes: usize,
}

// SAFETY: CUDA owns this page-locked allocation until `Drop`; tests only mutate
// it during construction and retain it immutably for DMA afterwards.
unsafe impl Send for MappedHost {}
unsafe impl Sync for MappedHost {}

impl MappedHost {
    fn new(runtime: &CudaRuntime, bytes: usize, seed: u8) -> Self {
        runtime.bind().unwrap();
        const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
        let host = unsafe { result::malloc_host(bytes, CU_MEMHOSTALLOC_DEVICEMAP) }
            .unwrap()
            .cast::<u8>();
        let slice = unsafe { std::slice::from_raw_parts_mut(host, bytes) };
        for expert in 0..EXPERTS {
            slice[expert * ROW_BYTES..(expert + 1) * ROW_BYTES]
                .fill(seed.wrapping_add(expert as u8));
        }
        let mut device = 0;
        unsafe {
            sys::cuMemHostGetDevicePointer_v2(&mut device, host.cast::<c_void>(), 0)
                .result()
                .unwrap();
        }
        Self {
            host,
            device,
            bytes,
        }
    }

    fn expert(&self, expert: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host.add(expert * ROW_BYTES), ROW_BYTES) }
    }
}

impl Drop for MappedHost {
    fn drop(&mut self) {
        let _ = self.bytes;
        unsafe {
            result::free_host(self.host.cast::<c_void>()).unwrap();
        }
    }
}

struct VmmDestinationOwner {
    allocator: Arc<onnx_runtime_ep_cuda::vmm_allocator::CudaVmmAllocator>,
    address: usize,
}

unsafe impl Send for VmmDestinationOwner {}
unsafe impl Sync for VmmDestinationOwner {}

impl Drop for VmmDestinationOwner {
    fn drop(&mut self) {
        let ptr = NonNull::new(self.address as *mut u8).expect("VMM destination address");
        self.allocator.deallocate_span(ptr);
    }
}

fn bank(
    residency: &Arc<CudaWeightResidency>,
    source: &Arc<MappedHost>,
    destination_owner: &Arc<VmmDestinationOwner>,
    destination: u64,
) -> IndexedExpertBank {
    let source_owner: IndexedBankOwner = Arc::clone(source) as Arc<dyn Any + Send + Sync>;
    let destination_owner: IndexedBankOwner =
        Arc::clone(destination_owner) as Arc<dyn Any + Send + Sync>;
    // SAFETY: the two owners exactly retain the mapped host source and VMM
    // destination allocation described by this geometry.
    unsafe {
        residency
            .seal_indexed_expert_bank(
                IndexedExpertBankSpec {
                    source_base: source.device,
                    destination_base: destination,
                    source_expert_stride: ROW_BYTES,
                    destination_slot_stride: ROW_BYTES,
                    bytes_per_expert: ROW_BYTES,
                    experts: EXPERTS,
                    slots: SLOTS,
                },
                source_owner,
                destination_owner,
            )
            .unwrap()
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore)]
fn fused_pagein_matches_unfused_bytes_and_reduces_real_submissions() {
    let _serial = GPU_SERIAL.lock().unwrap();
    let runtime = Arc::new(CudaRuntime::new(0).expect("CUDA runtime"));
    let bank_count = 6usize;
    let bytes_per_bank = SLOTS * ROW_BYTES;
    let destination_bytes = bank_count * bytes_per_bank * 2;
    let queue = CudaDeferredReleaseQueue::new(
        Box::new(CudaStreamFences::new(Arc::clone(&runtime))),
        DEFAULT_DEFERRED_RELEASE_CAPACITY,
    );
    let residency = Arc::new(
        CudaWeightResidency::new(Arc::clone(&runtime), (destination_bytes * 4) as u64)
            .with_deferred_release_queue(queue),
    );
    let sources: Vec<Arc<MappedHost>> = (0..bank_count)
        .map(|bank| {
            Arc::new(MappedHost::new(
                &runtime,
                EXPERTS * ROW_BYTES,
                17 + bank as u8 * 19,
            ))
        })
        .collect();
    let governor = Box::leak(Box::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::device(runtime.ordinal()),
        (destination_bytes * 4) as u64,
        0,
        0,
    ))));
    let allocator = Arc::new(
        onnx_runtime_ep_cuda::vmm_allocator::CudaVmmAllocator::new(
            runtime.cuda_context(),
            DeviceKey::device(runtime.ordinal()),
            runtime.ordinal() as i32,
            destination_bytes * 2,
            governor,
            HolderId::new(2323),
            MemoryRole::Weights,
        )
        .expect("VMM allocator"),
    );
    let destination_span = allocator
        .allocate(destination_bytes, 256)
        .expect("stable-VA destination span");
    let destination_base = destination_span.as_ptr() as u64;
    let destination_owner = Arc::new(VmmDestinationOwner {
        allocator: Arc::clone(&allocator),
        address: destination_span.as_ptr() as usize,
    });
    let fused_destinations: Vec<u64> = (0..bank_count)
        .map(|bank| destination_base + (bank * bytes_per_bank) as u64)
        .collect();
    let reference_destinations: Vec<u64> = (0..bank_count)
        .map(|bank| destination_base + ((bank_count + bank) * bytes_per_bank) as u64)
        .collect();

    let poison = vec![0xA5; SLOTS * ROW_BYTES];
    for &destination in fused_destinations
        .iter()
        .chain(reference_destinations.iter())
    {
        unsafe { runtime.htod(&poison, destination).unwrap() };
    }

    let selections = [
        ExpertSlot {
            expert: 3,
            slot: 0,
            resident: false,
        },
        ExpertSlot {
            expert: 1,
            slot: 2,
            resident: true,
        },
        ExpertSlot {
            expert: 0,
            slot: 3,
            resident: false,
        },
    ];
    // Construct the hit non-vacuously in both arms.
    for bank_index in 0..bank_count {
        unsafe {
            runtime
                .htod(
                    sources[bank_index].expert(1),
                    fused_destinations[bank_index] + (2 * ROW_BYTES) as u64,
                )
                .unwrap();
            runtime
                .htod(
                    sources[bank_index].expert(1),
                    reference_destinations[bank_index] + (2 * ROW_BYTES) as u64,
                )
                .unwrap();
        }
    }

    // Unfused reference: one measured H2D submission per missing expert/bank.
    let reference_before = runtime.transfer_counts();
    for selection in selections.iter().filter(|selection| !selection.resident) {
        for bank_index in 0..bank_count {
            unsafe {
                let (_elapsed, completed) = runtime
                    .htod_async_elapsed_ms(
                        sources[bank_index].expert(selection.expert),
                        reference_destinations[bank_index] + (selection.slot * ROW_BYTES) as u64,
                    )
                    .unwrap();
                drop(completed);
            }
        }
    }
    let reference_submissions = runtime
        .transfer_counts()
        .async_host_to_device
        .saturating_sub(reference_before.async_host_to_device);

    let projections = [
        ProjectionBankPair {
            projection: 0,
            packed: bank(
                &residency,
                &sources[0],
                &destination_owner,
                fused_destinations[0],
            ),
            auxiliary_scale: bank(
                &residency,
                &sources[1],
                &destination_owner,
                fused_destinations[1],
            ),
        },
        ProjectionBankPair {
            projection: 1,
            packed: bank(
                &residency,
                &sources[2],
                &destination_owner,
                fused_destinations[2],
            ),
            auxiliary_scale: bank(
                &residency,
                &sources[3],
                &destination_owner,
                fused_destinations[3],
            ),
        },
        ProjectionBankPair {
            projection: 2,
            packed: bank(
                &residency,
                &sources[4],
                &destination_owner,
                fused_destinations[4],
            ),
            auxiliary_scale: bank(
                &residency,
                &sources[5],
                &destination_owner,
                fused_destinations[5],
            ),
        },
    ];
    let plan = IndexedMultiBankPageInPlan::build(
        IndexedPageInAttribution {
            layer: 11,
            phase: IndexedPageInPhase::Decode,
        },
        &selections,
        &projections,
    )
    .unwrap();

    reset_global_offload_stats();
    let physical_before = global_offload_stats();
    let vmm_before = onnx_runtime_ep_cuda::vmm_allocator::global_vmm_stats();
    let batches_before = runtime.batch_copy_counts();
    let boundary = residency.complete_indexed_page_in_boundary(1).unwrap();
    let receipt = execute_indexed_multi_bank_page_in(&residency, &boundary, &plan).unwrap();
    let batches_after = runtime.batch_copy_counts();
    let physical_after = global_offload_stats();
    let vmm_after = onnx_runtime_ep_cuda::vmm_allocator::global_vmm_stats();

    assert_eq!((receipt.hits, receipt.misses), (1, 2));
    assert_eq!(receipt.copy_entries, 12);
    assert_eq!(reference_submissions, 12);
    assert_eq!(
        batches_after.submissions - batches_before.submissions,
        1,
        "counter brackets the accepted cuMemcpyBatchAsync_v2 call"
    );
    assert_eq!(batches_after.entries - batches_before.entries, 12);
    assert_eq!(physical_after.htod_bytes, receipt.payload_bytes);
    assert_eq!(physical_after.indexed_expert_page_ins, 2);
    assert_eq!(
        physical_after.physical_owned_bytes,
        physical_before.physical_owned_bytes
    );
    assert_eq!(
        physical_after.mapped_physical_bytes,
        physical_before.mapped_physical_bytes
    );
    assert_eq!(
        (
            vmm_after.committed_bytes,
            vmm_after.reserved_bytes,
            vmm_after.allocations,
        ),
        (
            vmm_before.committed_bytes,
            vmm_before.reserved_bytes,
            vmm_before.allocations,
        ),
        "page-in must consume existing coarse stable mappings without remapping"
    );
    assert_eq!(vmm_after.ref_underflows, 0);
    assert_eq!(vmm_after.byte_underflows, 0);
    assert_eq!(vmm_after.unaccounted_committed_bytes, 0);
    let managed_limit_bytes = (destination_bytes * 4) as u64;
    assert!(
        vmm_after.peak_committed_bytes < managed_limit_bytes,
        "peak committed {} must remain below managed limit {managed_limit_bytes}",
        vmm_after.peak_committed_bytes
    );
    assert_eq!(
        MemoryGovernor::oversubscribed_bytes(governor, Tier::Device),
        0
    );
    println!(
        "indexed_pagein_evidence gpu=A100 row_bytes={ROW_BYTES} banks={bank_count} \
         hits={} misses={} payload_bytes={} reference_submissions={} fused_submissions={} \
         fused_entries={} payload_invariant=true \
         peak_committed_bytes={} managed_limit_bytes={} oversubscribed_bytes=0 \
         ref_underflows=0 byte_underflows=0 unaccounted_committed_bytes=0",
        receipt.hits,
        receipt.misses,
        receipt.payload_bytes,
        reference_submissions,
        batches_after.submissions - batches_before.submissions,
        batches_after.entries - batches_before.entries,
        vmm_after.peak_committed_bytes,
        managed_limit_bytes,
    );

    for bank_index in 0..bank_count {
        let mut fused = vec![0u8; SLOTS * ROW_BYTES];
        let mut reference = vec![0u8; SLOTS * ROW_BYTES];
        unsafe {
            runtime
                .dtoh(&mut fused, fused_destinations[bank_index])
                .unwrap();
            runtime
                .dtoh(&mut reference, reference_destinations[bank_index])
                .unwrap();
        }
        assert_eq!(fused, reference, "bank {bank_index} final bytes");
    }

    let transfer_before = runtime.batch_copy_counts();
    let accounting_before = global_offload_stats();
    runtime
        .begin_graph_capture(&[&CapturableTestKernel])
        .expect("begin capture");
    let error = residency.try_indexed_page_in_boundary(1).unwrap_err();
    assert!(
        error.to_string().contains("graph capture is active"),
        "{error}"
    );
    assert_eq!(runtime.batch_copy_counts(), transfer_before);
    assert_eq!(
        global_offload_stats().htod_bytes,
        accounting_before.htod_bytes
    );
    runtime.abort_graph_capture().expect("abort empty capture");

    let graph_src = runtime.alloc_raw(4096).unwrap();
    let graph_dst = runtime.alloc_raw(4096).unwrap();
    unsafe {
        runtime.htod(&vec![0x6c; 4096], graph_src).unwrap();
        runtime
            .begin_graph_capture(&[&CapturableTestKernel])
            .unwrap();
        runtime.dtod_async(graph_src, graph_dst, 4096).unwrap();
        runtime.end_graph_capture().unwrap();
    }
    let stale_boundary = residency.complete_indexed_page_in_boundary(1).unwrap();
    runtime.replay_graph().unwrap();
    let replay_error = residency.try_indexed_page_in_boundary(1).unwrap_err();
    assert!(
        replay_error
            .to_string()
            .contains("replay completion is not established"),
        "{replay_error}"
    );
    let before_stale = runtime.batch_copy_counts();
    let stale_error =
        execute_indexed_multi_bank_page_in(&residency, &stale_boundary, &plan).unwrap_err();
    assert!(
        stale_error
            .to_string()
            .contains("stale replay-completion witness"),
        "{stale_error}"
    );
    assert_eq!(runtime.batch_copy_counts(), before_stale);
    residency.complete_indexed_page_in_boundary(1).unwrap();
    runtime.reset_graph().unwrap();
    unsafe {
        runtime.free_raw(graph_src).unwrap();
        runtime.free_raw(graph_dst).unwrap();
    }

    let fault_bytes = bank_count * bytes_per_bank;
    let fault_span = allocator
        .allocate(fault_bytes, 256)
        .expect("fault destination span");
    let fault_owner = Arc::new(VmmDestinationOwner {
        allocator: Arc::clone(&allocator),
        address: fault_span.as_ptr() as usize,
    });
    let fault_base = fault_span.as_ptr() as u64;
    let fault_destinations: Vec<u64> = (0..bank_count)
        .map(|bank| fault_base + (bank * bytes_per_bank) as u64)
        .collect();
    for (bank_index, &destination) in fault_destinations.iter().enumerate() {
        unsafe {
            runtime.htod(&poison, destination).unwrap();
            runtime
                .htod(
                    sources[bank_index].expert(1),
                    destination + (2 * ROW_BYTES) as u64,
                )
                .unwrap();
        }
    }
    let fault_projections = [
        ProjectionBankPair {
            projection: 0,
            packed: bank(&residency, &sources[0], &fault_owner, fault_destinations[0]),
            auxiliary_scale: bank(&residency, &sources[1], &fault_owner, fault_destinations[1]),
        },
        ProjectionBankPair {
            projection: 1,
            packed: bank(&residency, &sources[2], &fault_owner, fault_destinations[2]),
            auxiliary_scale: bank(&residency, &sources[3], &fault_owner, fault_destinations[3]),
        },
        ProjectionBankPair {
            projection: 2,
            packed: bank(&residency, &sources[4], &fault_owner, fault_destinations[4]),
            auxiliary_scale: bank(&residency, &sources[5], &fault_owner, fault_destinations[5]),
        },
    ];
    let fault_plan = IndexedMultiBankPageInPlan::build(
        IndexedPageInAttribution {
            layer: 12,
            phase: IndexedPageInPhase::Decode,
        },
        &selections,
        &fault_projections,
    )
    .unwrap();
    let fault_accounting_before = global_offload_stats();
    let fault_boundary = residency.complete_indexed_page_in_boundary(1).unwrap();
    let fault = execute_indexed_multi_bank_page_in_with_partial_write_fault(
        &residency,
        &fault_boundary,
        &fault_plan,
        3,
    )
    .unwrap_err();
    assert!(matches!(
        fault.completion(),
        FailedHtodCompletion::MayBeInFlight
    ));
    assert_eq!(
        fault.disposition(),
        IndexedPageInFailureDisposition::QuarantinedMayBeInFlight
    );
    assert!(fault.to_string().contains("after 3 partial destination"));
    assert!(
        fault_projections.iter().all(|pair| {
            pair.packed.visibility() == IndexedBankVisibility::Poisoned
                && pair.auxiliary_scale.visibility() == IndexedBankVisibility::Poisoned
        }),
        "every packed and scale destination must be poisoned atomically"
    );
    assert_eq!(
        global_offload_stats().htod_bytes,
        fault_accounting_before.htod_bytes,
        "partial writes must not publish successful payload accounting"
    );

    let retry_span = allocator
        .allocate(fault_bytes, 256)
        .expect("retry destination span");
    let retry_owner = Arc::new(VmmDestinationOwner {
        allocator: Arc::clone(&allocator),
        address: retry_span.as_ptr() as usize,
    });
    let retry_base = retry_span.as_ptr() as u64;
    let retry_destinations: Vec<u64> = (0..bank_count)
        .map(|bank| retry_base + (bank * bytes_per_bank) as u64)
        .collect();
    for (bank_index, &destination) in retry_destinations.iter().enumerate() {
        unsafe {
            runtime.htod(&poison, destination).unwrap();
            runtime
                .htod(
                    sources[bank_index].expert(1),
                    destination + (2 * ROW_BYTES) as u64,
                )
                .unwrap();
        }
    }
    let retry_projections = [
        ProjectionBankPair {
            projection: 0,
            packed: bank(&residency, &sources[0], &retry_owner, retry_destinations[0]),
            auxiliary_scale: bank(&residency, &sources[1], &retry_owner, retry_destinations[1]),
        },
        ProjectionBankPair {
            projection: 1,
            packed: bank(&residency, &sources[2], &retry_owner, retry_destinations[2]),
            auxiliary_scale: bank(&residency, &sources[3], &retry_owner, retry_destinations[3]),
        },
        ProjectionBankPair {
            projection: 2,
            packed: bank(&residency, &sources[4], &retry_owner, retry_destinations[4]),
            auxiliary_scale: bank(&residency, &sources[5], &retry_owner, retry_destinations[5]),
        },
    ];
    let retry_plan = IndexedMultiBankPageInPlan::build(
        IndexedPageInAttribution {
            layer: 12,
            phase: IndexedPageInPhase::Decode,
        },
        &selections,
        &retry_projections,
    )
    .unwrap();
    let retry_boundary = residency.complete_indexed_page_in_boundary(1).unwrap();
    execute_indexed_multi_bank_page_in(&residency, &retry_boundary, &retry_plan).unwrap();
    for bank_index in 0..bank_count {
        let mut retry = vec![0u8; SLOTS * ROW_BYTES];
        let mut reference = vec![0u8; SLOTS * ROW_BYTES];
        unsafe {
            runtime
                .dtoh(&mut retry, retry_destinations[bank_index])
                .unwrap();
            runtime
                .dtoh(&mut reference, reference_destinations[bank_index])
                .unwrap();
        }
        assert_eq!(
            retry, reference,
            "retry must restore every byte of bank {bank_index}"
        );
    }
    let final_vmm = onnx_runtime_ep_cuda::vmm_allocator::global_vmm_stats();
    assert_eq!(final_vmm.ref_underflows, 0);
    assert_eq!(final_vmm.byte_underflows, 0);
    assert_eq!(final_vmm.unaccounted_committed_bytes, 0);
    assert!(final_vmm.peak_committed_bytes < managed_limit_bytes);
    assert_eq!(
        MemoryGovernor::oversubscribed_bytes(governor, Tier::Device),
        0
    );
    println!(
        "indexed_pagein_fault_evidence replay_requires_completion=true stale_witness_rejected=true \
         injected_partial_writes=3 disposition=quarantined_may_be_in_flight \
         poisoned_banks=6 retry_restored_banks=6 oversubscribed_bytes=0 \
         ref_underflows=0 byte_underflows=0 unaccounted_committed_bytes=0"
    );
}
