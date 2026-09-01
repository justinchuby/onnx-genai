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
use onnx_runtime_ep_api::{
    ExecutionProvider, ExpertWeightGroup, LazyWeightBoundary, Result, expert_weight_groups,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::kernels::expert_route_telemetry::{
    H_COUNT, H_DEVICE, H_EPOCH, H_OVERFLOW, H_POISON, H_REQUEST, HEADER_LEN, TelemetrySnapshot,
    cpu_bitmap,
};
use onnx_runtime_ep_cuda::route_residency::{
    RouteResidencyBoundary, RouteResidencyInstallOutcome, RouteTelemetrySource,
    build_route_residency_boundaries,
};
use onnx_runtime_ep_cuda::weight_paging::{
    CudaWeightResidency, DeviceOffloadPolicy, RouteReservationHealth,
};
use onnx_runtime_ir::{DataType, Graph, NodeId, TensorData, ValueId, WeightRef, static_shape};
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
    let routes_per_row = route_lists
        .iter()
        .find(|routes| !routes.is_empty())
        .map_or(1, |routes| routes.len());
    assert!(
        route_lists
            .iter()
            .all(|routes| routes.is_empty() || routes.len() == routes_per_row),
        "a telemetry window must use one prepared routes-per-row contract"
    );
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
        routes_per_row: u32::try_from(routes_per_row).unwrap(),
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
        RouteReservationHealth::new(),
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
        RouteReservationHealth::new(),
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
            RouteReservationHealth::new(),
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
            RouteReservationHealth::new(),
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
            RouteReservationHealth::new(),
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
        &[&[0, 2], &[0, 2]],
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
        RouteReservationHealth::new(),
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

// ===========================================================================
// #1810 Slice 7D — production *binding* construction/installation.
//
// Slice 7C (above) drove an already-installed `RouteResidencyBoundary` through
// the production request caller. Slice 7D constructs that binding by
// property-based discovery over a *loaded model graph's* expert banks and
// installs it through the real CUDA-EP install seam
// (`CudaExecutionProvider::try_install_route_residency_binding`), so the merged
// live caller has a production binding when the feature is enabled.
//
// These tests never call `build_route_residency_boundaries`' inner helper
// directly for the reachability proof: they build a shape-faithful QMoE graph,
// let discovery find the bank identities, install through the EP, then drive
// the same trait method the executor calls — asserting the binding is actually
// installed (via `route_residency_diagnostics().installs()`) and fires.
// ===========================================================================

/// A `u8` initializer-backed weight value (an expert bank tensor).
fn inline_bank_initializer(graph: &mut Graph, name: &str) -> ValueId {
    let value = graph.create_named_value(name, DataType::Uint8, static_shape([4]));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(DataType::Uint8, vec![4], vec![0u8; 4])),
    );
    value
}

/// A shape-faithful `com.microsoft::QMoE` node whose only initializer-backed
/// inputs are two expert weight banks (fc1/fc2). Property-based discovery
/// (`expert_weight_groups`) must therefore yield exactly one group with those
/// two members, in input order — the identities the production builder binds.
/// The hidden-state and router-probs inputs are graph values, not initializers,
/// so they are never mistaken for banks.
fn two_bank_qmoe_graph() -> (Graph, NodeId, ValueId, ValueId) {
    let mut graph = Graph::new();
    let hidden = graph.create_named_value("hidden", DataType::Float32, static_shape([4]));
    let router = graph.create_named_value("router_probs", DataType::Float32, static_shape([4]));
    let fc1 = inline_bank_initializer(&mut graph, "fc1_experts_weights");
    let fc2 = inline_bank_initializer(&mut graph, "fc2_experts_weights");
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    let mut node = onnx_runtime_ir::Node::new(
        NodeId(0),
        "QMoE",
        vec![Some(hidden), Some(router), Some(fc1), Some(fc2)],
        vec![output],
    );
    node.domain = "com.microsoft".to_string();
    let node_id = graph.insert_node(node);
    (graph, node_id, fc1, fc2)
}

/// A dense-only graph (a single `MatMul`): property-based discovery finds no
/// routed expert group, so a binding attempt must fail-closed.
fn dense_only_graph() -> Graph {
    let mut graph = Graph::new();
    let w = inline_bank_initializer(&mut graph, "dense_weight");
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(onnx_runtime_ir::Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(x), Some(w)],
        vec![y],
    ));
    graph
}

/// Build the two committed bank allocators + aligned catalogs + a producer
/// window source keyed by the *discovered* member/node identities, so the maps
/// the builder consumes are addressed exactly as production would key them.
#[allow(clippy::type_complexity)]
fn wire_two_bank_artifacts(
    provider: &CudaExecutionProvider,
    node: NodeId,
    fc1: ValueId,
    fc2: ValueId,
    n_experts: usize,
    gran: usize,
    pool_bytes: usize,
    governor: &'static LedgerGovernor,
    request: u32,
    holder_base: u64,
) -> (
    HashMap<ValueId, WeightRegionCatalog>,
    HashMap<ValueId, Arc<CudaVmmAllocator>>,
    HashMap<NodeId, Arc<dyn RouteTelemetrySource>>,
    Arc<WindowSource>,
    (u64, u64),
) {
    let runtime = provider.runtime();
    let total_bytes = n_experts * gran;
    let (allocator_fc1, base_fc1) = build_precommitted_allocator(
        provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(holder_base),
    );
    let (allocator_fc2, base_fc2) = build_precommitted_allocator(
        provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(holder_base + 1),
    );
    let pat_fc1: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    let pat_fc2: Vec<u8> = (0..total_bytes).map(|j| ((j + 91) & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pat_fc1, base_fc1).expect("htod fc1");
        runtime.htod(&pat_fc2, base_fc2).expect("htod fc2");
    }

    let mut catalogs = HashMap::new();
    catalogs.insert(fc1, make_aligned_catalog(n_experts, gran, 0));
    catalogs.insert(fc2, make_aligned_catalog(n_experts, gran, total_bytes));
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(fc1, Arc::clone(&allocator_fc1));
    allocators.insert(fc2, Arc::clone(&allocator_fc2));

    let source = WindowSource::with_window(window_snapshot(
        &[&[0, 2], &[2, 3], &[5, 0], &[3, 5]],
        n_experts,
        1,
        request,
        runtime.ordinal(),
    ));
    let mut sources: HashMap<NodeId, Arc<dyn RouteTelemetrySource>> = HashMap::new();
    sources.insert(node, Arc::clone(&source) as Arc<dyn RouteTelemetrySource>);

    (catalogs, allocators, sources, source, (base_fc1, base_fc2))
}

// ---------------------------------------------------------------------------
// Slice 7D Test 1: the production builder assembles a *firing* binding purely
// from graph-property discovery. Build a shape-faithful two-bank QMoE graph,
// let `expert_weight_groups` discover the bank identities, construct the
// binding with `build_route_residency_boundaries`, install it through the EP, and
// drive the executor's trait method: the routed union transitions both
// discovered members atomically, the window advances once, the next boundary is
// empty, and draining removes the binding so no further boundary work occurs.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn builder_assembles_firing_binding_from_graph_banks() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== builder_assembles_firing_binding_from_graph_banks ===");
    let provider = match provider_or_skip("builder") {
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

    // Property-based discovery is the sole source of bank identity/membership.
    let (graph, node, fc1, fc2) = two_bank_qmoe_graph();
    let discovered = expert_weight_groups(&graph);
    assert_eq!(discovered.len(), 1, "one QMoE node -> one bindable group");
    assert_eq!(
        discovered[0].members,
        vec![fc1, fc2],
        "discovery yields the exact fc1/fc2 bank members in input order"
    );

    let request = 11_u32;
    let (catalogs, allocators, sources, source, (base_fc1, base_fc2)) = wire_two_bank_artifacts(
        &provider, node, fc1, fc2, n_experts, gran, pool_bytes, governor, request, 150,
    );
    let pat_fc1: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    let pat_fc2: Vec<u8> = (0..total_bytes).map(|j| ((j + 91) & 0xFF) as u8).collect();
    let residency = Arc::new(CudaWeightResidency::new(
        Arc::clone(runtime),
        pool_bytes as u64,
    ));
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    // The production builder: no op/name allowlist — it consumes the discovered
    // group and the EP's existing catalog/allocator/pool authorities.
    let mut bindings = build_route_residency_boundaries(
        &graph,
        Arc::clone(&residency),
        &sources,
        &catalogs,
        &allocators,
        RouteReservationHealth::new(),
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        1,
        0,
        request,
        runtime.ordinal(),
        1,
    )
    .expect("builder must assemble a binding from a valid two-bank graph");
    assert_eq!(bindings.len(), 1);
    let binding = bindings.remove(0);
    assert_eq!(
        binding.bank_value_count(),
        2,
        "binding covers exactly the two discovered banks"
    );
    provider.install_route_residency_boundary(Arc::new(binding));

    gate_on();
    // Boundary 1: the discovered banks transition atomically to the routed set.
    provider
        .consume_route_residency_at_boundary()
        .expect("boundary 1 Ok");
    let diag = provider.route_residency_diagnostics();
    assert_eq!(diag.applied(), 1, "the routed union is applied once");
    assert_eq!(source.snapshot_calls(), 1, "one snapshot per boundary");
    assert_eq!(source.reset_calls(), 1, "consumed window advanced once");

    // Boundary 2: the window advanced, so the next boundary is empty.
    provider
        .consume_route_residency_at_boundary()
        .expect("boundary 2 Ok");
    assert_eq!(diag.empty(), 1, "next window empty after the reset");
    assert_eq!(source.snapshot_calls(), 2);
    assert_eq!(source.reset_calls(), 1, "empty window does not reset again");
    gate_off();

    // Both discovered banks' bytes survive (stable VA preserved). Read back
    // while the binding — the sole owner of the bank allocators — is still
    // installed, so the arena's stable VA is still mapped.
    let mut got_fc1 = vec![0u8; total_bytes];
    let mut got_fc2 = vec![0u8; total_bytes];
    unsafe {
        runtime.dtoh(&mut got_fc1, base_fc1).expect("dtoh fc1");
        runtime.dtoh(&mut got_fc2, base_fc2).expect("dtoh fc2");
    }
    assert_eq!(pat_fc1, got_fc1, "fc1 content corrupted");
    assert_eq!(pat_fc2, got_fc2, "fc2 content corrupted");

    // Draining removes the binding: the consumer is inert again.
    gate_on();
    provider.drain_route_residency_boundary();
    provider
        .consume_route_residency_at_boundary()
        .expect("drained boundary Ok");
    assert_eq!(
        source.snapshot_calls(),
        2,
        "a drained binding is not snapshotted"
    );
    gate_off();
    println!("builder assembled a firing binding from graph discovery, drained clean ✓");
}

// ---------------------------------------------------------------------------
// Slice 7D Test 2: the real EP install seam fail-closes and installs *nothing*
// when it is not a valid binding authority. On a default provider (no
// coarse-residency authority) `try_install_route_residency_binding` returns
// `OffloadDisabled` with the gate on and `GateDisabled` with the gate off; both
// record a decline and leave the boundary consumer inert (no binding to drive).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn try_install_fail_closed_installs_nothing() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== try_install_fail_closed_installs_nothing ===");
    let provider = match provider_or_skip("decline") {
        Some(p) => p,
        None => return,
    };
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };
    let n_experts = 8_usize;
    let pool_bytes = n_experts * gran * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };
    let (graph, node, _fc1, _fc2) = two_bank_qmoe_graph();
    let mut sources: HashMap<NodeId, Arc<dyn RouteTelemetrySource>> = HashMap::new();
    sources.insert(
        node,
        WindowSource::with_window(window_snapshot(&[&[0]], n_experts, 1, 5, 0))
            as Arc<dyn RouteTelemetrySource>,
    );
    let diag = provider.route_residency_diagnostics();

    // Gate on, but this EP has no coarse-residency authority -> OffloadDisabled.
    gate_on();
    let outcome = provider.try_install_route_residency_binding(
        &graph,
        &sources,
        HashMap::new(),
        HashMap::new(),
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        5,
        1,
    );
    assert_eq!(outcome, RouteResidencyInstallOutcome::OffloadDisabled);
    assert_eq!(diag.declines(), 1, "offload-disabled decline recorded");
    // Nothing installed: a gate-on boundary runs no consumer.
    provider
        .consume_route_residency_at_boundary()
        .expect("no-binding boundary Ok");
    assert_eq!(diag.boundaries(), 0, "no binding was installed");

    // Gate off -> GateDisabled before any discovery/allocation.
    gate_off();
    let outcome = provider.try_install_route_residency_binding(
        &graph,
        &sources,
        HashMap::new(),
        HashMap::new(),
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        5,
        1,
    );
    assert_eq!(outcome, RouteResidencyInstallOutcome::GateDisabled);
    assert_eq!(diag.declines(), 2, "gate-disabled decline recorded");
    assert_eq!(diag.installs(), 0, "no binding ever installed");
    println!("try_install fail-closed to no binding on a non-authority EP ✓");
}

// ---------------------------------------------------------------------------
// Slice 7D Test 3: the real EP install seam on a coarse-residency authority
// (offload-enabled provider). A dense-only graph is a typed `Rejected`
// (`NoExpertGroups`) that installs nothing; a valid two-bank graph is
// `Installed`, is counted in diagnostics, and fires through the production
// caller. Proves the *installed* binding — not a direct helper — drives the
// transition, then drains clean.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn try_install_on_offload_authority_installs_and_fires() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== try_install_on_offload_authority_installs_and_fires ===");
    let gran = match gran_or_skip() {
        Some(g) => g,
        None => return,
    };
    let n_experts = 8_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;

    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some(pool_bytes as u64),
        ..DeviceOffloadPolicy::default()
    };
    let provider = match CudaExecutionProvider::new_with_offload_policy(0, policy) {
        Ok(p) => p,
        Err(e) => {
            println!("SKIP [offload]: cannot build offload-enabled EP: {e}");
            return;
        }
    };
    let runtime = provider.runtime();
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);
    let diag = provider.route_residency_diagnostics();

    // A dense-only graph is a typed reject that installs nothing.
    let dense = dense_only_graph();
    let empty_sources: HashMap<NodeId, Arc<dyn RouteTelemetrySource>> = HashMap::new();
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };
    gate_on();
    let rejected = provider.try_install_route_residency_binding(
        &dense,
        &empty_sources,
        HashMap::new(),
        HashMap::new(),
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        13,
        1,
    );
    assert!(
        matches!(rejected, RouteResidencyInstallOutcome::Rejected(_)),
        "dense-only graph must be a typed reject, got {rejected:?}"
    );
    assert_eq!(diag.installs(), 0, "reject installs nothing");
    provider
        .consume_route_residency_at_boundary()
        .expect("post-reject boundary Ok");
    assert_eq!(diag.boundaries(), 0, "no binding installed after reject");

    // A valid two-bank graph installs a real binding through the EP seam.
    let (graph, node, fc1, fc2) = two_bank_qmoe_graph();
    let request = 13_u32;
    let (catalogs, allocators, sources, source, _) = wire_two_bank_artifacts(
        &provider, node, fc1, fc2, n_experts, gran, pool_bytes, governor, request, 160,
    );
    let installed = provider.try_install_route_residency_binding(
        &graph,
        &sources,
        catalogs,
        allocators,
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        request,
        1,
    );
    assert_eq!(
        installed,
        RouteResidencyInstallOutcome::Installed { banks: 2 },
        "valid graph installs a two-bank binding through the EP seam"
    );
    assert_eq!(diag.installs(), 1, "one binding installed");

    // The *installed* binding fires through the executor's trait method.
    provider
        .consume_route_residency_at_boundary()
        .expect("installed boundary Ok");
    assert_eq!(diag.applied(), 1, "the installed binding applied a hot-set");
    assert_eq!(
        source.snapshot_calls(),
        1,
        "installed binding was snapshotted"
    );
    assert_eq!(
        source.reset_calls(),
        1,
        "installed binding advanced its window"
    );

    // Teardown drains the binding (mirrors `shutdown`).
    provider.drain_route_residency_boundary();
    provider
        .consume_route_residency_at_boundary()
        .expect("drained boundary Ok");
    assert_eq!(
        source.snapshot_calls(),
        1,
        "drained binding not snapshotted"
    );
    gate_off();
    let _ = runtime;
    println!("try_install installed a firing binding on the offload EP, drained clean ✓");
}

// ---------------------------------------------------------------------------
// Slice 7D Test 4: coarse host-overhead of the per-boundary hook, gate ON vs
// OFF, on the steady-state empty-window path (no transition). This is a host-
// only micro-measurement of the boundary caller — NOT a full-model or tok/s
// claim. Serial on an idle GPU; warm up, then take the median of n>=5 samples.
// The disabled path is a single env read; the enabled empty path adds only the
// safe-point check + a snapshot that returns `None`.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn boundary_host_overhead_on_vs_off() {
    use std::time::Instant;

    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== boundary_host_overhead_on_vs_off ===");
    if ambient_gate_is_on() {
        println!("SKIP: {COARSE_RESIDENCY_ENABLE_ENV} is truthy in the ambient env");
        return;
    }
    let provider = match provider_or_skip("overhead") {
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
    let (graph, node, fc1, fc2) = two_bank_qmoe_graph();
    let (catalogs, allocators, sources, source, _) = wire_two_bank_artifacts(
        &provider, node, fc1, fc2, n_experts, gran, pool_bytes, governor, 21, 170,
    );
    // Drain the single armed window so every measured boundary takes the
    // steady-state *empty* path (safe-point check + snapshot -> None).
    source
        .reset_route_telemetry_boundary()
        .expect("drain window");
    let residency = Arc::new(CudaWeightResidency::new(
        Arc::clone(runtime),
        pool_bytes as u64,
    ));
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };
    let mut bindings = build_route_residency_boundaries(
        &graph,
        residency,
        &sources,
        &catalogs,
        &allocators,
        RouteReservationHealth::new(),
        Arc::clone(&pools.device_pool),
        Arc::clone(&pools.host_pool),
        1,
        0,
        21,
        runtime.ordinal(),
        1,
    )
    .expect("binding");
    assert_eq!(bindings.len(), 1);
    let binding = bindings.remove(0);
    provider.install_route_residency_boundary(Arc::new(binding));

    let sample = |gate_enabled: bool| -> f64 {
        if gate_enabled {
            gate_on();
        } else {
            gate_off();
        }
        // Warm up (clock ramp) before timing.
        for _ in 0..64 {
            provider
                .consume_route_residency_at_boundary()
                .expect("warm");
        }
        let mut samples = Vec::new();
        for _ in 0..9 {
            let iters = 1000u32;
            let start = Instant::now();
            for _ in 0..iters {
                provider
                    .consume_route_residency_at_boundary()
                    .expect("timed");
            }
            samples.push(start.elapsed().as_nanos() as f64 / f64::from(iters));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };

    let off_ns = sample(false);
    let on_ns = sample(true);
    gate_off();
    // The empty path must never have consumed/advanced the window.
    assert_eq!(
        source.reset_calls(),
        1,
        "overhead sampling must stay on the empty path (no extra reset)"
    );
    let delta_us = (on_ns - off_ns).max(0.0) / 1000.0;
    println!(
        "boundary host overhead: OFF median {off_ns:.0} ns, ON median {on_ns:.0} ns, delta {delta_us:.3} us/boundary (host-only, empty-window path; no full-model claim)"
    );
    // Coarse non-flaky guard against a gross host regression; the design intent
    // is <= ~2 us/boundary, reported above for the record.
    assert!(
        delta_us <= 50.0,
        "per-boundary host overhead {delta_us:.3} us exceeds the coarse guard"
    );
    println!("boundary host overhead measured ON vs OFF ✓");
}
