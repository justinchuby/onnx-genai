//! CUDA correctness and submission-count proof for fused indexed multi-bank
//! expert page-in. Run on an idle pinned GPU with `--features gpu-tests`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{result, sys};
use onnx_runtime_ep_api::{CaptureSupport, Kernel, TensorMut, TensorView};
use onnx_runtime_ep_cuda::runtime::CudaRuntime;
use onnx_runtime_ep_cuda::{
    ExpertSlot, IndexedExpertBank, IndexedMultiBankPageInPlan, IndexedPageInAttribution,
    IndexedPageInPhase, ProjectionBankPair, execute_indexed_multi_bank_page_in,
    global_offload_stats, reset_global_offload_stats,
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

fn bank(source: &MappedHost, destination: u64) -> IndexedExpertBank {
    IndexedExpertBank {
        source_base: source.device,
        destination_base: destination,
        source_expert_stride: ROW_BYTES,
        destination_slot_stride: ROW_BYTES,
        bytes_per_expert: ROW_BYTES,
        experts: EXPERTS,
        slots: SLOTS,
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore)]
fn fused_pagein_matches_unfused_bytes_and_reduces_real_submissions() {
    let _serial = GPU_SERIAL.lock().unwrap();
    let runtime = Arc::new(CudaRuntime::new(0).expect("CUDA runtime"));
    let bank_count = 6usize;
    let sources: Vec<MappedHost> = (0..bank_count)
        .map(|bank| MappedHost::new(&runtime, EXPERTS * ROW_BYTES, 17 + bank as u8 * 19))
        .collect();
    let bytes_per_bank = SLOTS * ROW_BYTES;
    let destination_bytes = bank_count * bytes_per_bank * 2;
    let governor = Box::leak(Box::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::device(runtime.ordinal()),
        (destination_bytes * 4) as u64,
        0,
        0,
    ))));
    let allocator = onnx_runtime_ep_cuda::vmm_allocator::CudaVmmAllocator::new(
        runtime.cuda_context(),
        DeviceKey::device(runtime.ordinal()),
        runtime.ordinal() as i32,
        destination_bytes * 2,
        governor,
        HolderId::new(2323),
        MemoryRole::Weights,
    )
    .expect("VMM allocator");
    let destination_span = allocator
        .allocate(destination_bytes, 256)
        .expect("stable-VA destination span");
    let destination_base = destination_span.as_ptr() as u64;
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
            packed: bank(&sources[0], fused_destinations[0]),
            auxiliary_scale: bank(&sources[1], fused_destinations[1]),
        },
        ProjectionBankPair {
            projection: 1,
            packed: bank(&sources[2], fused_destinations[2]),
            auxiliary_scale: bank(&sources[3], fused_destinations[3]),
        },
        ProjectionBankPair {
            projection: 2,
            packed: bank(&sources[4], fused_destinations[4]),
            auxiliary_scale: bank(&sources[5], fused_destinations[5]),
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
    let receipt = execute_indexed_multi_bank_page_in(&runtime, &plan).unwrap();
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
         fused_entries={} fused_device_ms={:.3} payload_invariant=true \
         peak_committed_bytes={} managed_limit_bytes={} oversubscribed_bytes=0 \
         ref_underflows=0 byte_underflows=0 unaccounted_committed_bytes=0",
        receipt.hits,
        receipt.misses,
        receipt.payload_bytes,
        reference_submissions,
        batches_after.submissions - batches_before.submissions,
        batches_after.entries - batches_before.entries,
        receipt.elapsed.as_secs_f64() * 1_000.0,
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
    let error = execute_indexed_multi_bank_page_in(&runtime, &plan).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("illegal during CUDA graph capture"),
        "{error}"
    );
    assert_eq!(runtime.batch_copy_counts(), transfer_before);
    assert_eq!(
        global_offload_stats().htod_bytes,
        accounting_before.htod_bytes
    );
    runtime.abort_graph_capture().expect("abort empty capture");

    allocator.deallocate_span(destination_span);
}
