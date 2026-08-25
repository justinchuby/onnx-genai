//! #1810 Slice 7C — GPU tests for the *production boundary wiring* of the
//! route-telemetry consumer.
//!
//! Slice 7B (`route_residency_consume_gpu.rs`) proved the raw consumer
//! (`consume_route_window_at_boundary`) drives a real VMM transition. Slice 7C
//! wires that consumer into the real request lifecycle: the session executor
//! calls `ExecutionProvider::consume_route_residency_at_boundary` once per
//! top-level request at the single coarse safe boundary
//! (`Executor::finish_device_validation`, after `sync()`), and the CUDA EP
//! override drives snapshot → consume → reset through an installed
//! `RouteResidencyBoundary`.
//!
//! These tests therefore never call the raw consumer: they install a boundary
//! binding on a real `CudaExecutionProvider` and invoke the **same trait method
//! the executor calls**, asserting the whole matrix (disabled no-op, hot-set
//! group transition + window advance, unsafe-boundary reject before
//! consume/reset, fail-closed defective/foreign windows, and driver-fault
//! rollback) flows through the production caller and surfaces typed diagnostics.
//!
//! The producer window is supplied by a controllable `RouteTelemetrySource`
//! double (a compile-time assertion in `route_residency.rs` proves the real
//! `QMoEKernel` satisfies the same trait), built from the producer's own
//! `cpu_bitmap` oracle over an explicit eager+replay route sequence — exactly
//! the precedent set by the Slice-7B consume tests.
//!
//! Requires an idle CUDA device with host-NUMA capability. Run:
//! ```text
//! CUDA_VISIBLE_DEVICES=<idle> cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda,cuda-13000,gpu-tests --release \
//!   --test route_residency_boundary_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```

#![allow(
    clippy::too_many_arguments,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_cuda_memory::capability::{CapabilityGateFailure, host_numa_capability};
use onnx_runtime_cuda_memory::release::{DriverFaultPlan, DriverOperation};
use onnx_runtime_cuda_memory::virtual_memory::{PhysicalHandlePool, PhysicalLocation};
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_ep_api::{ExecutionProvider, ExpertWeightGroup, LazyWeightBoundary, Result};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::kernels::expert_route_telemetry::{
    H_COUNT, H_DEVICE, H_EPOCH, H_OVERFLOW, H_POISON, H_REQUEST, HEADER_LEN, TelemetrySnapshot,
    cpu_bitmap,
};
use onnx_runtime_ep_cuda::route_residency::{RouteResidencyBoundary, RouteTelemetrySource};
use onnx_runtime_ep_cuda::weight_paging::CudaWeightResidency;
use onnx_runtime_ir::{DataType, NodeId, ValueId, WeightRef};
use onnx_runtime_loader::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};

// ---------------------------------------------------------------------------
// Feature-gated wrapper for the phase-8 fault EP entry point (mirrors the
// consume test): the method is `#[cfg(any(test, feature = "gpu-tests"))]`, so
// an integration test only sees it with the feature on. Keeps this file in both
// test inventories.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-tests")]
fn drive_boundary_with_faults(
    provider: &CudaExecutionProvider,
    faults: HashMap<ValueId, Arc<DriverFaultPlan>>,
) -> Result<()> {
    provider.consume_route_residency_at_boundary_with_phase8_faults(faults)
}

#[cfg(not(feature = "gpu-tests"))]
fn drive_boundary_with_faults(
    _provider: &CudaExecutionProvider,
    _faults: HashMap<ValueId, Arc<DriverFaultPlan>>,
) -> Result<()> {
    unreachable!("Phase-8 fault injection is only compiled under the gpu-tests feature");
}

// ---------------------------------------------------------------------------
// Controllable producer double. The boundary caller drives this exactly like a
// live armed `QMoEKernel`: snapshot returns the current window; reset advances
// to the next (here: pops it, so the following window starts empty). Counters
// prove the caller took the snapshot / advanced the window the right number of
// times, and that a rejected/disabled boundary did neither.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct WindowSource {
    windows: Mutex<VecDeque<TelemetrySnapshot>>,
    snapshots: AtomicUsize,
    resets: AtomicUsize,
}

impl WindowSource {
    fn with_window(window: TelemetrySnapshot) -> Arc<Self> {
        let src = Arc::new(Self::default());
        src.push(window);
        src
    }

    fn push(&self, window: TelemetrySnapshot) {
        self.windows.lock().unwrap().push_back(window);
    }

    fn snapshot_calls(&self) -> usize {
        self.snapshots.load(Ordering::Relaxed)
    }

    fn reset_calls(&self) -> usize {
        self.resets.load(Ordering::Relaxed)
    }
}

impl RouteTelemetrySource for WindowSource {
    fn route_telemetry_snapshot(&self) -> Result<Option<TelemetrySnapshot>> {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        Ok(self.windows.lock().unwrap().front().cloned())
    }

    fn reset_route_telemetry_boundary(&self) -> Result<()> {
        self.resets.fetch_add(1, Ordering::Relaxed);
        self.windows.lock().unwrap().pop_front();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared harness (copied from route_residency_consume_gpu.rs; integration test
// files are separate crates and cannot share private helpers).
// ---------------------------------------------------------------------------

static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn provider_or_skip(label: &str) -> Option<CudaExecutionProvider> {
    match CudaExecutionProvider::new(0) {
        Ok(p) => Some(p),
        Err(e) => {
            println!("SKIP [{label}]: no CUDA device: {e}");
            None
        }
    }
}

fn make_governor(device_bytes: u64, host_bytes: u64) -> &'static LedgerGovernor {
    let ledger = LeaseLedger::new_for_device(DeviceKey::device(0), device_bytes, host_bytes, 0);
    Box::leak(Box::new(LedgerGovernor::new(ledger)))
}

struct TestPools {
    device_pool: Arc<PhysicalHandlePool>,
    host_pool: Arc<PhysicalHandlePool>,
}

fn make_pools(
    provider: &CudaExecutionProvider,
    pool_bytes: usize,
    governor: &'static LedgerGovernor,
) -> Option<TestPools> {
    let device_ordinal = 0_i32;
    let context = provider.runtime().cuda_context();
    let cap = match host_numa_capability(device_ordinal) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return None;
        }
    };
    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        device_ordinal,
        PhysicalLocation::Device {
            ordinal: device_ordinal,
        },
        pool_bytes,
        governor,
        HolderId::new(110),
        MemoryRole::Weights,
    )
    .expect("device pool");
    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        device_ordinal,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        pool_bytes,
        governor,
        HolderId::new(111),
        MemoryRole::Weights,
    )
    .expect("host pool");
    Some(TestPools {
        device_pool,
        host_pool,
    })
}

fn build_precommitted_allocator(
    provider: &CudaExecutionProvider,
    n_experts: usize,
    gran: usize,
    pool_bytes: usize,
    governor: &'static LedgerGovernor,
    holder: HolderId,
) -> (Arc<CudaVmmAllocator>, u64) {
    let context = provider.runtime().cuda_context();
    let total_bytes = n_experts * gran;
    unsafe { std::env::set_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, pool_bytes.to_string()) };
    let allocator = Arc::new(
        CudaVmmAllocator::new(
            Arc::clone(&context),
            DeviceKey::device(0),
            0_i32,
            total_bytes * 2,
            governor,
            holder,
            MemoryRole::Weights,
        )
        .expect("build allocator"),
    );
    unsafe { std::env::remove_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV) };
    let ptr = allocator
        .allocate(total_bytes, gran)
        .expect("allocate total bytes");
    let base_ptr = ptr.as_ptr() as u64;
    (allocator, base_ptr)
}

fn make_aligned_catalog(n_experts: usize, gran: usize, file_offset: usize) -> WeightRegionCatalog {
    let rows = 512_usize;
    let cols = gran / rows;
    let layout = ExpertTensorLayout {
        version: 1,
        experts: n_experts,
        rows_per_expert: rows,
        storage_elements_per_row: cols,
        order: ExpertStorageOrder::ExpertMajor,
        quantization: Some(ExpertQuantization {
            bits: 4,
            block_size: 16,
            blocks_per_row: cols / 16,
        }),
    };
    let total = n_experts * rows * cols;
    let weight = WeightRef::External {
        path: std::path::PathBuf::from("/nonexistent/weights.bin"),
        offset: file_offset,
        length: total,
        dtype: DataType::Uint8,
        dims: vec![n_experts, rows, cols],
    };
    WeightRegionCatalog::classify(&weight, layout)
}

/// The exact `TelemetrySnapshot` an armed record would hold after a window that
/// accumulated `route_lists` (one per eager call / replay) with no reset,
/// stamped with the given identity/epoch. The bitmap is the producer's own
/// `cpu_bitmap` union; `count` is the bounded in-range route total.
fn window_snapshot(
    route_lists: &[&[i32]],
    num_experts: usize,
    epoch: u32,
    request: u32,
    device: u32,
) -> TelemetrySnapshot {
    let mut bitmap = vec![0u32; num_experts.div_ceil(32)];
    let mut poison = false;
    let mut count: u32 = 0;
    for routes in route_lists {
        let (bits, this_poison) = cpu_bitmap(routes, num_experts);
        for (word, bit) in bitmap.iter_mut().zip(bits) {
            *word |= bit;
        }
        poison |= this_poison;
        count += routes
            .iter()
            .filter(|&&r| r >= 0 && (r as usize) < num_experts)
            .count() as u32;
    }
    let mut header = [0u32; HEADER_LEN];
    header[H_EPOCH] = epoch;
    header[H_REQUEST] = request;
    header[H_DEVICE] = device;
    header[H_POISON] = u32::from(poison);
    header[H_COUNT] = count;
    TelemetrySnapshot {
        header,
        bitmap,
        num_experts,
    }
}

fn gate_on() {
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
}

fn gate_off() {
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
}

fn ambient_gate_is_on() -> bool {
    matches!(
        std::env::var(COARSE_RESIDENCY_ENABLE_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

fn gran_or_skip() -> Option<usize> {
    match host_numa_capability(0) {
        Ok(c) => Some(c.granularity),
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: default-off gate → the production caller is a structural no-op. Even
// with a valid boundary installed, the consumer never runs: no snapshot, no
// reset, no allocator change, byte-identical content. Proves the disabled path
// creates no consumer work.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn boundary_disabled_gate_is_structural_no_op() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== boundary_disabled_gate_is_structural_no_op ===");
    if ambient_gate_is_on() {
        println!("SKIP: {COARSE_RESIDENCY_ENABLE_ENV} is truthy in the ambient env");
        return;
    }
    let provider = match provider_or_skip("disabled") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };

    let n_experts = 8_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let (allocator, base_ptr) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(120),
    );
    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe { runtime.htod(&pattern, base_ptr).expect("htod pattern") };
    let (committed_before, reserved_before) = allocator.committed_and_reserved();

    let value = ValueId(1200);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, make_aligned_catalog(n_experts, gran, 0));
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = Arc::new(CudaWeightResidency::new(
        Arc::clone(runtime),
        total_bytes as u64,
    ));
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let source = WindowSource::with_window(window_snapshot(
        &[&[0, 2], &[2, 3], &[5, 0], &[3, 5]],
        n_experts,
        1,
        7,
        runtime.ordinal(),
    ));
    let boundary = RouteResidencyBoundary::new(
        Arc::clone(&source) as Arc<dyn RouteTelemetrySource>,
        Arc::clone(&residency),
        vec![value],
        LazyWeightBoundary::QMoe,
        catalogs,
        allocators,
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        1,
        0,
        7,
        runtime.ordinal(),
        1,
        Vec::new(),
    );
    provider.install_route_residency_boundary(Arc::new(boundary));

    gate_off();
    provider
        .consume_route_residency_at_boundary()
        .expect("disabled boundary must be Ok");

    let diag = provider.route_residency_diagnostics();
    assert_eq!(
        diag.boundaries(),
        0,
        "disabled path must not run the consumer"
    );
    assert_eq!(
        source.snapshot_calls(),
        0,
        "disabled path must not snapshot"
    );
    assert_eq!(source.reset_calls(), 0, "disabled path must not reset");
    let (committed_after, reserved_after) = allocator.committed_and_reserved();
    assert_eq!(
        committed_before, committed_after,
        "committed bytes unchanged"
    );
    assert_eq!(reserved_before, reserved_after, "reserved bytes unchanged");
    let mut readback = vec![0u8; total_bytes];
    unsafe { runtime.dtoh(&mut readback, base_ptr).expect("dtoh") };
    assert_eq!(pattern, readback, "content byte-identical when disabled");
    println!("disabled production caller is a structural no-op ✓");
}

// ---------------------------------------------------------------------------
// Test 2: the enabled production caller applies a routed hot-set for a
// two-member expert group atomically, advances the window exactly once, and the
// following boundary sees an empty window. Proves: >=3 replay accumulation ->
// boundary -> atomic expert-group transition -> next empty window, all through
// the executor's trait method.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn boundary_applies_group_hot_set_and_advances_window() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== boundary_applies_group_hot_set_and_advances_window ===");
    let provider = match provider_or_skip("group") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };

    let n_experts = 8_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let (allocator_fc1, base_fc1) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(130),
    );
    let (allocator_fc2, base_fc2) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(131),
    );
    let pat_fc1: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    let pat_fc2: Vec<u8> = (0..total_bytes).map(|j| ((j + 91) & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pat_fc1, base_fc1).expect("htod fc1");
        runtime.htod(&pat_fc2, base_fc2).expect("htod fc2");
    }

    let value_fc1 = ValueId(1300);
    let value_fc2 = ValueId(1301);
    let mut catalogs = HashMap::new();
    catalogs.insert(value_fc1, make_aligned_catalog(n_experts, gran, 0));
    catalogs.insert(
        value_fc2,
        make_aligned_catalog(n_experts, gran, total_bytes),
    );
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value_fc1, Arc::clone(&allocator_fc1));
    allocators.insert(value_fc2, Arc::clone(&allocator_fc2));
    let residency = Arc::new(CudaWeightResidency::new(
        Arc::clone(runtime),
        pool_bytes as u64,
    ));
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };
    let groups = vec![ExpertWeightGroup {
        node: NodeId(0),
        boundary: LazyWeightBoundary::QMoe,
        members: vec![value_fc1, value_fc2],
    }];

    // A window that accumulated a union across 1 eager call + 3 replays.
    let request = 9_u32;
    let source = WindowSource::with_window(window_snapshot(
        &[&[0, 2], &[2, 3], &[5, 0], &[3, 5]],
        n_experts,
        1,
        request,
        runtime.ordinal(),
    ));
    let boundary = RouteResidencyBoundary::new(
        Arc::clone(&source) as Arc<dyn RouteTelemetrySource>,
        Arc::clone(&residency),
        vec![value_fc1, value_fc2],
        LazyWeightBoundary::QMoe,
        catalogs,
        allocators,
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        1,
        0,
        request,
        runtime.ordinal(),
        1,
        groups,
    );
    provider.install_route_residency_boundary(Arc::new(boundary));

    gate_on();
    // Boundary 1: consume the accumulated window and transition both members.
    provider
        .consume_route_residency_at_boundary()
        .expect("boundary 1 Ok");
    let diag = provider.route_residency_diagnostics();
    assert_eq!(diag.boundaries(), 1);
    assert_eq!(diag.applied(), 1, "the routed union must be applied");
    assert_eq!(
        source.snapshot_calls(),
        1,
        "exactly one snapshot per boundary"
    );
    assert_eq!(
        source.reset_calls(),
        1,
        "the consumed window must be advanced once"
    );
    let reason = diag.last_reason().unwrap_or_default();
    assert!(
        reason.contains("applied") && reason.contains("epoch 1"),
        "diagnostics must surface the applied hot-set at the window epoch: {reason}"
    );

    // Boundary 2: the window was advanced, so this boundary sees an empty window
    // and does nothing (no second reset, no allocator change).
    provider
        .consume_route_residency_at_boundary()
        .expect("boundary 2 Ok");
    assert_eq!(diag.boundaries(), 2);
    assert_eq!(
        diag.empty(),
        1,
        "the next window must be empty after the reset"
    );
    assert_eq!(source.snapshot_calls(), 2);
    assert_eq!(
        source.reset_calls(),
        1,
        "an empty window must not reset again"
    );
    gate_off();

    // Both group members' bytes are still bit-identical (stable VA preserved).
    let mut got_fc1 = vec![0u8; total_bytes];
    let mut got_fc2 = vec![0u8; total_bytes];
    unsafe {
        runtime.dtoh(&mut got_fc1, base_fc1).expect("dtoh fc1");
        runtime.dtoh(&mut got_fc2, base_fc2).expect("dtoh fc2");
    }
    assert_eq!(pat_fc1, got_fc1, "fc1 content corrupted");
    assert_eq!(pat_fc2, got_fc2, "fc2 content corrupted");
    println!("group hot-set applied atomically, window advanced, next window empty ✓");
}

// ---------------------------------------------------------------------------
// Test 3: an unsafe boundary is rejected *before* the snapshot dtoh and before
// any reset. Two independent unsafe conditions, both through the production
// caller: (a) multi-device execution, (b) an active graph capture. Neither may
// snapshot, reset, or touch the allocator.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn boundary_unsafe_point_rejects_before_consume_and_reset() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== boundary_unsafe_point_rejects_before_consume_and_reset ===");
    let provider = match provider_or_skip("reject") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };

    let n_experts = 8_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let (allocator, base_ptr) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(140),
    );
    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe { runtime.htod(&pattern, base_ptr).expect("htod") };
    let (committed_before, reserved_before) = allocator.committed_and_reserved();

    let value = ValueId(1400);
    let good_window = window_snapshot(&[&[0, 2], &[3, 5]], n_experts, 1, 7, runtime.ordinal());

    // (a) multi-device: device_count = 2 blocks the boundary with no capture.
    {
        let mut catalogs = HashMap::new();
        catalogs.insert(value, make_aligned_catalog(n_experts, gran, 0));
        let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
        allocators.insert(value, Arc::clone(&allocator));
        let residency = Arc::new(CudaWeightResidency::new(
            Arc::clone(runtime),
            total_bytes as u64,
        ));
        let pools = match make_pools(&provider, pool_bytes, governor) {
            Some(p) => p,
            None => return,
        };
        let source = WindowSource::with_window(good_window.clone());
        let boundary = RouteResidencyBoundary::new(
            Arc::clone(&source) as Arc<dyn RouteTelemetrySource>,
            residency,
            vec![value],
            LazyWeightBoundary::QMoe,
            catalogs,
            allocators,
            Arc::clone(&pools.device_pool),
            Arc::clone(&pools.host_pool),
            2, // multi-device
            0,
            7,
            runtime.ordinal(),
            1,
            Vec::new(),
        );
        provider.install_route_residency_boundary(Arc::new(boundary));

        gate_on();
        provider
            .consume_route_residency_at_boundary()
            .expect("reject is Ok, not Err");
        gate_off();

        let diag = provider.route_residency_diagnostics();
        assert_eq!(diag.boundaries(), 1);
        assert_eq!(diag.rejected(), 1, "multi-device must reject");
        assert_eq!(
            source.snapshot_calls(),
            0,
            "reject must precede the snapshot"
        );
        assert_eq!(source.reset_calls(), 0, "reject must precede any reset");
        let reason = diag.last_reason().unwrap_or_default();
        assert!(
            reason.contains("rejected"),
            "reason must name the rejection: {reason}"
        );
    }

    // (b) active capture: a real in-flight graph capture blocks the boundary.
    {
        let mut catalogs = HashMap::new();
        catalogs.insert(value, make_aligned_catalog(n_experts, gran, 0));
        let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
        allocators.insert(value, Arc::clone(&allocator));
        let residency = Arc::new(CudaWeightResidency::new(
            Arc::clone(runtime),
            total_bytes as u64,
        ));
        let pools = match make_pools(&provider, pool_bytes, governor) {
            Some(p) => p,
            None => return,
        };
        let source = WindowSource::with_window(good_window.clone());
        let boundary = RouteResidencyBoundary::new(
            Arc::clone(&source) as Arc<dyn RouteTelemetrySource>,
            residency,
            vec![value],
            LazyWeightBoundary::QMoe,
            catalogs,
            allocators,
            Arc::clone(&pools.device_pool),
            Arc::clone(&pools.host_pool),
            1,
            0,
            7,
            runtime.ordinal(),
            1,
            Vec::new(),
        );
        provider.install_route_residency_boundary(Arc::new(boundary));

        gate_on();
        runtime.begin_graph_capture(&[]).expect("begin capture");
        let result = provider.consume_route_residency_at_boundary();
        runtime.abort_graph_capture().expect("abort capture");
        gate_off();
        result.expect("reject is Ok, not Err");

        let diag = provider.route_residency_diagnostics();
        assert_eq!(diag.boundaries(), 2);
        assert_eq!(diag.rejected(), 2, "active capture must reject");
        assert_eq!(source.snapshot_calls(), 0, "no snapshot during capture");
        assert_eq!(source.reset_calls(), 0, "no reset during capture");
    }

    let (committed_after, reserved_after) = allocator.committed_and_reserved();
    assert_eq!(committed_before, committed_after, "no tiering on reject");
    assert_eq!(reserved_before, reserved_after);
    let mut readback = vec![0u8; total_bytes];
    unsafe { runtime.dtoh(&mut readback, base_ptr).expect("dtoh") };
    assert_eq!(pattern, readback, "content untouched across rejects");
    println!("multi-device and active-capture reject before consume/reset ✓");
}

// ---------------------------------------------------------------------------
// Test 4: defective / foreign-identity windows fail closed to whole-bank
// through the production caller, tiering nothing. The foreign-request case is
// multi-request isolation: a window stamped for another request must not tier
// this request's bank.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn boundary_defective_windows_fail_closed() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== boundary_defective_windows_fail_closed ===");
    let provider = match provider_or_skip("failclosed") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };
    let device = runtime.ordinal();

    let n_experts = 8_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let (allocator, base_ptr) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(160),
    );
    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe { runtime.htod(&pattern, base_ptr).expect("htod") };
    let (committed_before, reserved_before) = allocator.committed_and_reserved();

    let value = ValueId(1600);
    let good_routes: [&[i32]; 2] = [&[0, 2], &[3, 5]];
    let expected_request = 7_u32;

    let mut poison_snap = window_snapshot(&good_routes, n_experts, 1, expected_request, device);
    poison_snap.header[H_POISON] = 1;
    let mut overflow_snap = window_snapshot(&good_routes, n_experts, 1, expected_request, device);
    overflow_snap.header[H_OVERFLOW] = 1;
    let empty_snap = window_snapshot(&[&[]], n_experts, 1, expected_request, device);

    // (label, window, initial_epoch this boundary expects)
    let cases: Vec<(&str, TelemetrySnapshot, u32)> = vec![
        (
            "foreign request (multi-request isolation)",
            window_snapshot(&good_routes, n_experts, 1, expected_request + 1, device),
            1,
        ),
        (
            "foreign device",
            window_snapshot(&good_routes, n_experts, 1, expected_request, device + 7),
            1,
        ),
        ("poison", poison_snap, 1),
        ("overflow", overflow_snap, 1),
        (
            "stale epoch",
            window_snapshot(&good_routes, n_experts, 1, expected_request, device),
            2, // boundary expects epoch >= 2; a record still at 1 is stale
        ),
        ("empty routed set", empty_snap, 1),
    ];

    gate_on();
    let mut expected_whole_bank = 0_u64;
    for (label, window, initial_epoch) in cases {
        let mut catalogs = HashMap::new();
        catalogs.insert(value, make_aligned_catalog(n_experts, gran, 0));
        let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
        allocators.insert(value, Arc::clone(&allocator));
        let residency = Arc::new(CudaWeightResidency::new(
            Arc::clone(runtime),
            total_bytes as u64,
        ));
        let pools = match make_pools(&provider, pool_bytes, governor) {
            Some(p) => p,
            None => return,
        };
        let source = WindowSource::with_window(window);
        let boundary = RouteResidencyBoundary::new(
            Arc::clone(&source) as Arc<dyn RouteTelemetrySource>,
            residency,
            vec![value],
            LazyWeightBoundary::QMoe,
            catalogs,
            allocators,
            Arc::clone(&pools.device_pool),
            Arc::clone(&pools.host_pool),
            1,
            0,
            expected_request,
            device,
            initial_epoch,
            Vec::new(),
        );
        provider.install_route_residency_boundary(Arc::new(boundary));

        provider
            .consume_route_residency_at_boundary()
            .expect("fail-closed boundary is Ok");
        expected_whole_bank += 1;
        let diag = provider.route_residency_diagnostics();
        assert_eq!(
            diag.whole_bank(),
            expected_whole_bank,
            "case '{label}' must fail closed to whole-bank"
        );
        let reason = diag.last_reason().unwrap_or_default();
        assert!(
            reason.contains("whole-bank"),
            "case '{label}' reason must name whole-bank: {reason}"
        );
        println!("case '{label}' → whole-bank ✓");
    }
    gate_off();

    let (committed_after, reserved_after) = allocator.committed_and_reserved();
    assert_eq!(
        committed_before, committed_after,
        "no tiering on any fail-closed case"
    );
    assert_eq!(reserved_before, reserved_after);
    let mut readback = vec![0u8; total_bytes];
    unsafe { runtime.dtoh(&mut readback, base_ptr).expect("dtoh") };
    assert_eq!(
        pattern, readback,
        "content untouched across all fail-closed cases"
    );
    println!("all foreign/defective windows fail closed through the caller ✓");
}

// ---------------------------------------------------------------------------
// Test 5: a deterministic driver fault mid-transition, driven through the
// production caller's fault path, rolls back and preserves content. Proves the
// caller-driven transition inherits the #1854 rollback/quarantine authority.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn boundary_injected_fault_rolls_back_through_caller() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== boundary_injected_fault_rolls_back_through_caller ===");
    let provider = match provider_or_skip("rollback") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };

    // 5 experts, routed union {0,2} -> hot={0,2}, cold={1,3,4}. Failing the 3rd
    // Unmap forces a cross-range rollback (mirrors the Slice-7B consume test 6).
    let n_experts = 5_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let (allocator, base_ptr) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(170),
    );
    let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
    for i in 0..n_experts {
        let pat: Vec<u8> = (0..gran).map(|j| ((i * 31 + j) & 0xFF) as u8).collect();
        unsafe {
            runtime
                .htod(&pat, base_ptr + (i * gran) as u64)
                .expect("htod")
        };
        patterns.push(pat);
    }

    let value = ValueId(1700);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, make_aligned_catalog(n_experts, gran, 0));
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = Arc::new(CudaWeightResidency::new(
        Arc::clone(runtime),
        total_bytes as u64,
    ));
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let request = 7_u32;
    let source = WindowSource::with_window(window_snapshot(
        &[&[0, 2], &[2]],
        n_experts,
        1,
        request,
        runtime.ordinal(),
    ));
    let boundary = RouteResidencyBoundary::new(
        Arc::clone(&source) as Arc<dyn RouteTelemetrySource>,
        Arc::clone(&residency),
        vec![value],
        LazyWeightBoundary::QMoe,
        catalogs,
        allocators,
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        1,
        0,
        request,
        runtime.ordinal(),
        1,
        Vec::new(),
    );
    provider.install_route_residency_boundary(Arc::new(boundary));

    let mut phase8_faults: HashMap<ValueId, Arc<DriverFaultPlan>> = HashMap::new();
    phase8_faults.insert(
        value,
        Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 3)),
    );

    gate_on();
    drive_boundary_with_faults(&provider, phase8_faults).expect("fault boundary is Ok");
    gate_off();

    let diag = provider.route_residency_diagnostics();
    assert_eq!(diag.boundaries(), 1);
    assert_eq!(
        diag.applied(),
        1,
        "the fault path still reports an applied boundary"
    );
    assert_eq!(source.snapshot_calls(), 1);
    assert_eq!(
        source.reset_calls(),
        1,
        "the window advances after the transition"
    );

    // Full rollback preserves every expert's bytes (stable VA, content intact).
    for (i, pattern) in patterns.iter().enumerate() {
        let mut got = vec![0u8; gran];
        unsafe {
            runtime
                .dtoh(&mut got, base_ptr + (i * gran) as u64)
                .expect("dtoh")
        };
        assert_eq!(*pattern, got, "expert {i} content changed despite rollback");
    }
    println!("driver-fault transition rolled back through the caller, content preserved ✓");
}
