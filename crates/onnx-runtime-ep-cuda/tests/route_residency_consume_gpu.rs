//! #1810 Slice 7B — GPU tests for the boundary-time route-telemetry *consumer*
//! (`route_residency::consume_route_window_at_boundary`).
//!
//! These tests require an idle CUDA device with host-NUMA capability
//! (A100-SXM4-80GB, driver 580.x, CUDA 13.0).
//!
//! Run:
//! ```text
//! CUDA_VISIBLE_DEVICES=<idle> cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda,cuda-13000,gpu-tests --release \
//!   --test route_residency_consume_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## What these prove (non-vacuous, real VMM transitions)
//!
//! The consumer's contract begins at "given one completed coarse-boundary
//! window, produce+apply the residency plan". The device-side *accumulation*
//! of that window (a routed-expert union over many eager calls and >=3 graph
//! replays with a fixed epoch, VA stable, fail-closed on overflow/poison/
//! foreign identity) is proven by the merged Slice-6/7A producer tests
//! (`qmoe_gpu.rs::route_telemetry::*`, PR #1922). To keep this file's claims
//! about the *consumer* self-contained and deterministic, each window here is
//! built with the producer's own `cpu_bitmap` oracle over an explicit
//! eager+replay route sequence and cross-checked with the producer's own
//! `consume_and_validate` — the exact bytes the armed record would hold — then
//! fed to the consumer, which drives a **real** granule transition on **real**
//! device/host-NUMA VMM memory. Readback proves stable-VA byte identity; the
//! allocator's committed/reserved accounting and the boundary outcome's
//! `host_bytes_committed` prove the cold experts actually moved through
//! PMM/VMM.
//!
//! Coverage:
//! 1. `disabled_gate_is_structural_no_op` — gate off → `Disabled`, allocator
//!    bytes and content unchanged (ordinary inference byte-identical).
//! 2. `route_window_hot_set_transitions_cold_experts` — one bank, routed union
//!    kept device-resident, complement tiered to host; readback bit-identical.
//! 3. `expert_group_transitions_atomically_from_window` — two cross-tensor
//!    members of one `ExpertWeightGroup` both transition on the same window.
//! 4. `active_capture_and_multi_device_reject_consume` — mid-capture and
//!    multi-device both fail closed with no side effects.
//! 5. `foreign_identity_and_defective_windows_fail_closed` — foreign request/
//!    device, poison, overflow, stale epoch, and an empty routed set each
//!    fail closed to whole-bank with zero side effects.
//! 6. `injected_fault_rolls_back_consumer_transition` — a deterministic driver
//!    fault mid-transition rolls back range-precisely; content preserved.

#![allow(
    clippy::too_many_arguments,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use onnx_runtime_cuda_memory::capability::{CapabilityGateFailure, host_numa_capability};
use onnx_runtime_cuda_memory::release::{DriverFaultPlan, DriverOperation};
use onnx_runtime_cuda_memory::virtual_memory::{PhysicalHandlePool, PhysicalLocation};
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_ep_api::{ExpertWeightGroup, LazyWeightBoundary};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::kernels::expert_route_telemetry::{
    H_COUNT, H_DEVICE, H_EPOCH, H_OVERFLOW, H_POISON, H_REQUEST, HEADER_LEN, RouteDecision,
    TelemetrySnapshot, consume_and_validate, cpu_bitmap,
};
use onnx_runtime_ep_cuda::route_residency::{
    RouteWindowConsumeOutcome, consume_route_window_at_boundary,
};
use onnx_runtime_ep_cuda::weight_paging::CudaWeightResidency;
use onnx_runtime_ir::{DataType, NodeId, ValueId, WeightRef};
use onnx_runtime_loader::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};

// ---------------------------------------------------------------------------
// Fault-injection seam (mirrors coarse_residency_plan_gpu.rs). The consumer's
// `consume_route_window_at_boundary_with_phase8_faults` is gated
// `#[cfg(any(test, feature = "gpu-tests"))]`; an integration test only sees it
// with the feature on, so wrap it once here to keep this file in both test
// inventories.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-tests")]
fn consume_route_window_with_faults(
    runtime: &Arc<onnx_runtime_ep_cuda::CudaRuntime>,
    residency: &CudaWeightResidency,
    snapshot: &TelemetrySnapshot,
    expected_epoch: u32,
    expected_request: u32,
    expected_device: u32,
    bank_values: &[ValueId],
    boundary: LazyWeightBoundary,
    catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    device_pool: &Arc<PhysicalHandlePool>,
    host_pool: &Arc<PhysicalHandlePool>,
    device_count: usize,
    device_ordinal: i32,
    expert_groups: &[ExpertWeightGroup],
    phase8_faults: HashMap<ValueId, Arc<DriverFaultPlan>>,
) -> RouteWindowConsumeOutcome {
    onnx_runtime_ep_cuda::route_residency::consume_route_window_at_boundary_with_phase8_faults(
        runtime,
        residency,
        snapshot,
        expected_epoch,
        expected_request,
        expected_device,
        bank_values,
        boundary,
        catalogs,
        allocators,
        device_pool,
        host_pool,
        device_count,
        device_ordinal,
        expert_groups,
        phase8_faults,
    )
}

#[cfg(not(feature = "gpu-tests"))]
fn consume_route_window_with_faults(
    _runtime: &Arc<onnx_runtime_ep_cuda::CudaRuntime>,
    _residency: &CudaWeightResidency,
    _snapshot: &TelemetrySnapshot,
    _expected_epoch: u32,
    _expected_request: u32,
    _expected_device: u32,
    _bank_values: &[ValueId],
    _boundary: LazyWeightBoundary,
    _catalogs: &HashMap<ValueId, WeightRegionCatalog>,
    _allocators: &HashMap<ValueId, Arc<CudaVmmAllocator>>,
    _device_pool: &Arc<PhysicalHandlePool>,
    _host_pool: &Arc<PhysicalHandlePool>,
    _device_count: usize,
    _device_ordinal: i32,
    _expert_groups: &[ExpertWeightGroup],
    _phase8_faults: HashMap<ValueId, Arc<DriverFaultPlan>>,
) -> RouteWindowConsumeOutcome {
    unreachable!("Phase-8 fault injection is only compiled under the gpu-tests feature");
}

// ---------------------------------------------------------------------------
// Shared harness (same shape as coarse_residency_plan_gpu.rs).
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
        HolderId::new(10),
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
        HolderId::new(11),
        MemoryRole::Weights,
    )
    .expect("host pool");
    Some(TestPools {
        device_pool,
        host_pool,
    })
}

/// Build a `CudaVmmAllocator` with all `n_experts * gran` bytes committed on
/// Device; returns the allocator and the stable base VA (offset 0).
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

/// Granule-aligned catalog where every expert is exactly `gran` bytes.
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

/// Build the exact `TelemetrySnapshot` an armed record would hold after a
/// window that accumulated `route_lists` (one per eager call / replay) with no
/// reset, stamped with the given identity/epoch. The bitmap is the producer's
/// own `cpu_bitmap` union; `count` is the bounded in-range route total.
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

// ---------------------------------------------------------------------------
// Test 1: gate off → structural no-op, allocator bytes + content unchanged.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn disabled_gate_is_structural_no_op() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== disabled_gate_is_structural_no_op ===");
    if ambient_gate_is_on() {
        println!("SKIP: {COARSE_RESIDENCY_ENABLE_ENV} is truthy in the ambient env");
        return;
    }
    let provider = match provider_or_skip("disabled") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match host_numa_capability(0) {
        Ok(c) => c.granularity,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return;
        }
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
        HolderId::new(20),
    );

    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe { runtime.htod(&pattern, base_ptr).expect("htod pattern") };
    let (committed_before, reserved_before) = allocator.committed_and_reserved();

    let value = ValueId(42);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let snapshot = window_snapshot(&[&[0, 2], &[3, 5]], n_experts, 1, 7, runtime.ordinal());

    gate_off();
    let outcome = consume_route_window_at_boundary(
        &residency,
        &snapshot,
        1,
        7,
        runtime.ordinal(),
        &[value],
        LazyWeightBoundary::QMoe,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
    );

    assert!(
        matches!(outcome, RouteWindowConsumeOutcome::Disabled),
        "gate off must return Disabled, got {outcome:?}"
    );
    let (committed_after, reserved_after) = allocator.committed_and_reserved();
    assert_eq!(
        committed_before, committed_after,
        "committed bytes must not change"
    );
    assert_eq!(
        reserved_before, reserved_after,
        "reserved bytes must not change"
    );
    let mut readback = vec![0u8; total_bytes];
    unsafe { runtime.dtoh(&mut readback, base_ptr).expect("dtoh") };
    assert_eq!(
        pattern, readback,
        "content must be byte-identical when disabled"
    );
    println!("disabled path is a structural no-op ✓");
}

// ---------------------------------------------------------------------------
// Test 2: routed union kept resident, complement tiered to host.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn route_window_hot_set_transitions_cold_experts() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== route_window_hot_set_transitions_cold_experts ===");
    let provider = match provider_or_skip("hotset") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match host_numa_capability(0) {
        Ok(c) => c.granularity,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return;
        }
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
        HolderId::new(30),
    );

    // Unique per-expert patterns for byte-identity readback.
    let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
    for i in 0..n_experts {
        let pat: Vec<u8> = (0..gran).map(|j| ((i * 17 + j) & 0xFF) as u8).collect();
        unsafe {
            runtime
                .htod(&pat, base_ptr + (i * gran) as u64)
                .expect("htod")
        };
        patterns.push(pat);
    }

    // A window that accumulated a union across 1 eager call + 3 replays.
    let routes: [&[i32]; 4] = [&[0, 2], &[2, 3], &[5, 0], &[3, 5]];
    let hot = [0usize, 2, 3, 5];
    let cold = [1usize, 4, 6, 7];
    let request = 7_u32;
    let snapshot = window_snapshot(&routes, n_experts, 1, request, runtime.ordinal());
    // Cross-check against the producer's own validator: this is a HotSet whose
    // decoded experts are exactly the union.
    assert!(matches!(
        consume_and_validate(
            &snapshot.header,
            &snapshot.bitmap,
            1,
            request,
            runtime.ordinal(),
            n_experts,
        ),
        RouteDecision::HotSet(_)
    ));
    assert_eq!(snapshot.routed_experts(), hot.to_vec(), "decoded union");

    let value = ValueId(100);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    gate_on();
    let outcome = consume_route_window_at_boundary(
        &residency,
        &snapshot,
        1,
        request,
        runtime.ordinal(),
        &[value],
        LazyWeightBoundary::QMoe,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
    );
    gate_off();

    match outcome {
        RouteWindowConsumeOutcome::Applied {
            routed_experts,
            epoch,
            outcome,
            ..
        } => {
            assert_eq!(routed_experts, hot.to_vec(), "kept the routed union hot");
            assert_eq!(epoch, 1, "fixed window epoch echoed");
            assert!(
                outcome.fallback_reason.is_none(),
                "no structural fallback: {outcome:#?}"
            );
            assert_eq!(outcome.values_touched, 1);
            assert_eq!(outcome.failure_count, 0);
            assert_eq!(
                outcome.host_bytes_committed,
                (cold.len() * gran) as u64,
                "cold experts must move to host"
            );
        }
        other => panic!("expected Applied, got {other:?}"),
    }

    // Every expert's bytes are still bit-identical (stable VA, content preserved).
    for (i, pattern) in patterns.iter().enumerate() {
        let mut got = vec![0u8; gran];
        unsafe {
            runtime
                .dtoh(&mut got, base_ptr + (i * gran) as u64)
                .expect("dtoh")
        };
        assert_eq!(*pattern, got, "expert {i} content corrupted");
    }
    println!("routed union {hot:?} kept resident, cold {cold:?} tiered to host, bytes identical ✓");
}

// ---------------------------------------------------------------------------
// Test 3: two cross-tensor group members transition atomically on one window.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn expert_group_transitions_atomically_from_window() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== expert_group_transitions_atomically_from_window ===");
    let provider = match provider_or_skip("group") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match host_numa_capability(0) {
        Ok(c) => c.granularity,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return;
        }
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
        HolderId::new(40),
    );
    let (allocator_fc2, base_fc2) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(41),
    );
    let pat_fc1: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    let pat_fc2: Vec<u8> = (0..total_bytes).map(|j| ((j + 91) & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pat_fc1, base_fc1).expect("htod fc1");
        runtime.htod(&pat_fc2, base_fc2).expect("htod fc2");
    }

    let value_fc1 = ValueId(800);
    let value_fc2 = ValueId(801);
    let catalog_fc1 = make_aligned_catalog(n_experts, gran, 0);
    let catalog_fc2 = make_aligned_catalog(n_experts, gran, total_bytes);
    let mut catalogs = HashMap::new();
    catalogs.insert(value_fc1, catalog_fc1);
    catalogs.insert(value_fc2, catalog_fc2);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value_fc1, Arc::clone(&allocator_fc1));
    allocators.insert(value_fc2, Arc::clone(&allocator_fc2));
    let residency = CudaWeightResidency::new(Arc::clone(runtime), pool_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let groups = vec![ExpertWeightGroup {
        node: NodeId(0),
        boundary: LazyWeightBoundary::QMoe,
        members: vec![value_fc1, value_fc2],
    }];

    let routes: [&[i32]; 4] = [&[0, 2], &[2, 3], &[5, 0], &[3, 5]];
    let hot = [0usize, 2, 3, 5];
    let cold_per_value = 4_usize;
    let request = 9_u32;
    let snapshot = window_snapshot(&routes, n_experts, 1, request, runtime.ordinal());

    gate_on();
    let outcome = consume_route_window_at_boundary(
        &residency,
        &snapshot,
        1,
        request,
        runtime.ordinal(),
        &[value_fc1, value_fc2],
        LazyWeightBoundary::QMoe,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &groups,
    );
    gate_off();

    match outcome {
        RouteWindowConsumeOutcome::Applied {
            routed_experts,
            outcome,
            ..
        } => {
            assert_eq!(routed_experts, hot.to_vec());
            assert!(
                outcome.fallback_reason.is_none(),
                "no fallback: {outcome:#?}"
            );
            assert!(
                outcome.per_value_fallbacks.is_empty(),
                "both members must transition, no per-value fallback: {:?}",
                outcome.per_value_fallbacks
            );
            assert_eq!(outcome.values_touched, 2, "both group members touched");
            assert_eq!(
                outcome.host_bytes_committed,
                (2 * cold_per_value * gran) as u64,
                "both members tier their cold experts"
            );
        }
        other => panic!("expected Applied, got {other:?}"),
    }

    let mut got_fc1 = vec![0u8; total_bytes];
    let mut got_fc2 = vec![0u8; total_bytes];
    unsafe {
        runtime.dtoh(&mut got_fc1, base_fc1).expect("dtoh fc1");
        runtime.dtoh(&mut got_fc2, base_fc2).expect("dtoh fc2");
    }
    assert_eq!(pat_fc1, got_fc1, "fc1 content corrupted");
    assert_eq!(pat_fc2, got_fc2, "fc2 content corrupted");
    println!("both group members transitioned atomically, content identical ✓");
}

// ---------------------------------------------------------------------------
// Test 4: mid-capture and multi-device both fail closed.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn active_capture_and_multi_device_reject_consume() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== active_capture_and_multi_device_reject_consume ===");
    let provider = match provider_or_skip("reject") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match host_numa_capability(0) {
        Ok(c) => c.granularity,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return;
        }
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
        HolderId::new(50),
    );
    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe { runtime.htod(&pattern, base_ptr).expect("htod") };
    let (committed_before, _) = allocator.committed_and_reserved();

    let value = ValueId(500);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };
    let request = 7_u32;
    let snapshot = window_snapshot(
        &[&[0, 2], &[3, 5]],
        n_experts,
        1,
        request,
        runtime.ordinal(),
    );

    // (a) multi-device (device_count = 2) fails closed with no capture needed.
    gate_on();
    let outcome_md = consume_route_window_at_boundary(
        &residency,
        &snapshot,
        1,
        request,
        runtime.ordinal(),
        &[value],
        LazyWeightBoundary::QMoe,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        2, // device_count > 1
        0,
        &[],
    );
    assert!(
        matches!(
            outcome_md,
            RouteWindowConsumeOutcome::RejectedNotSafeBoundary { .. }
        ),
        "multi-device must fail closed, got {outcome_md:?}"
    );

    // (b) begin a real capture so is_capturing() is true, then consume.
    runtime.begin_graph_capture(&[]).expect("begin capture");
    let outcome_cap = consume_route_window_at_boundary(
        &residency,
        &snapshot,
        1,
        request,
        runtime.ordinal(),
        &[value],
        LazyWeightBoundary::QMoe,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
    );
    // Abort the empty capture and return the graph lifecycle to idle.
    runtime.abort_graph_capture().expect("abort capture");
    let _ = runtime.reset_graph();
    gate_off();

    match outcome_cap {
        RouteWindowConsumeOutcome::RejectedNotSafeBoundary { reason } => {
            assert!(
                reason.contains("capturing") || reason.contains("replay"),
                "capture rejection reason should name capture, got: {reason}"
            );
        }
        other => panic!("active capture must fail closed, got {other:?}"),
    }

    let (committed_after, _) = allocator.committed_and_reserved();
    assert_eq!(committed_before, committed_after, "no tiering on rejection");
    let mut readback = vec![0u8; total_bytes];
    unsafe { runtime.dtoh(&mut readback, base_ptr).expect("dtoh") };
    assert_eq!(pattern, readback, "content untouched on rejection");
    println!("multi-device and active-capture both fail closed with no side effects ✓");
}

// ---------------------------------------------------------------------------
// Test 5: foreign identity + defective windows fail closed (no tiering).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn foreign_identity_and_defective_windows_fail_closed() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== foreign_identity_and_defective_windows_fail_closed ===");
    let provider = match provider_or_skip("failclosed") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match host_numa_capability(0) {
        Ok(c) => c.granularity,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return;
        }
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
        HolderId::new(60),
    );
    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe { runtime.htod(&pattern, base_ptr).expect("htod") };
    let (committed_before, reserved_before) = allocator.committed_and_reserved();

    let value = ValueId(600);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let good_routes: [&[i32]; 2] = [&[0, 2], &[3, 5]];
    let expected_request = 7_u32;

    // Each case is (label, snapshot, expected_epoch, expected_request, expected_device).
    let mut poison_snap = window_snapshot(&good_routes, n_experts, 1, expected_request, device);
    poison_snap.header[H_POISON] = 1;
    let mut overflow_snap = window_snapshot(&good_routes, n_experts, 1, expected_request, device);
    overflow_snap.header[H_OVERFLOW] = 1;
    let empty_snap = window_snapshot(&[&[]], n_experts, 1, expected_request, device);

    let cases: Vec<(&str, TelemetrySnapshot, u32, u32, u32)> = vec![
        (
            "foreign request",
            window_snapshot(&good_routes, n_experts, 1, expected_request + 1, device),
            1,
            expected_request,
            device,
        ),
        (
            "foreign device",
            window_snapshot(&good_routes, n_experts, 1, expected_request, device + 7),
            1,
            expected_request,
            device,
        ),
        ("poison", poison_snap, 1, expected_request, device),
        ("overflow", overflow_snap, 1, expected_request, device),
        (
            "stale epoch",
            window_snapshot(&good_routes, n_experts, 1, expected_request, device),
            2, // boundary expects epoch >= 2
            expected_request,
            device,
        ),
        ("empty routed set", empty_snap, 1, expected_request, device),
    ];

    gate_on();
    for (label, snapshot, exp_epoch, exp_req, exp_dev) in &cases {
        let outcome = consume_route_window_at_boundary(
            &residency,
            snapshot,
            *exp_epoch,
            *exp_req,
            *exp_dev,
            &[value],
            LazyWeightBoundary::QMoe,
            &catalogs,
            &allocators,
            &pools.device_pool,
            &pools.host_pool,
            1,
            0,
            &[],
        );
        assert!(
            matches!(outcome, RouteWindowConsumeOutcome::WholeBank { .. }),
            "case '{label}' must fail closed to WholeBank, got {outcome:?}"
        );
        println!("case '{label}' → WholeBank ✓");
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
    println!("all foreign/defective windows fail closed with zero side effects ✓");
}

// ---------------------------------------------------------------------------
// Test 6: injected driver fault mid-transition rolls back range-precisely.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn injected_fault_rolls_back_consumer_transition() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== injected_fault_rolls_back_consumer_transition ===");
    let provider = match provider_or_skip("rollback") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();
    let gran = match host_numa_capability(0) {
        Ok(c) => c.granularity,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return;
        }
    };

    // 5 experts, routed union {0,2} -> hot={0,2}, cold={1,3,4}. Expert 1 is an
    // isolated 1-granule cold range that fully commits as the 1st Unmap call;
    // experts 3,4 merge into one 2-granule range (2nd + 3rd Unmap calls).
    // Failing the 3rd Unmap makes the merged range's first granule commit and
    // its second hit Fatal, so the prior fully-committed range (expert 1) must
    // be reverted by the rollback loop -- a genuine cross-range rollback driven
    // through the consumer (mirrors coarse_residency_plan_gpu.rs test 5).
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
        HolderId::new(70),
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

    let value = ValueId(700);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));
    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let request = 7_u32;
    let snapshot = window_snapshot(&[&[0, 2], &[2]], n_experts, 1, request, runtime.ordinal());
    assert_eq!(snapshot.routed_experts(), vec![0, 2]);
    let faults = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 3));
    let mut phase8_faults: HashMap<ValueId, Arc<DriverFaultPlan>> = HashMap::new();
    phase8_faults.insert(value, faults);

    gate_on();
    let outcome = consume_route_window_with_faults(
        &Arc::clone(runtime),
        &residency,
        &snapshot,
        1,
        request,
        runtime.ordinal(),
        &[value],
        LazyWeightBoundary::QMoe,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
        phase8_faults,
    );
    gate_off();

    match outcome {
        RouteWindowConsumeOutcome::Applied { outcome, .. } => {
            println!("boundary outcome: {outcome:#?}");
            assert!(
                !outcome.fatal_progress.is_empty(),
                "expected a Fatal from the injected fault, got: {:?}",
                outcome.fatal_progress
            );
            assert_eq!(
                outcome.rollback_count, 1,
                "the prior fully-committed range (expert 1) must be reverted, got {}",
                outcome.rollback_count
            );
            assert_eq!(
                outcome.values_touched, 0,
                "a fully rolled-back value must not report a committed transition, got {}",
                outcome.values_touched
            );
            assert!(
                outcome.committed_values.is_empty(),
                "no value may remain committed after full rollback, got: {:?}",
                outcome.committed_values
            );
            assert!(
                outcome.rollback_failures.is_empty(),
                "rollback must fully succeed, got: {:?}",
                outcome.rollback_failures
            );
        }
        other => panic!("expected Applied (with fault outcome), got {other:?}"),
    }

    // Content preserved end-to-end: forward partial-commit + full rollback keep
    // every expert's bytes bit-identical on the stable VA.
    for (i, pat) in patterns.iter().enumerate() {
        let mut readback = vec![0u8; gran];
        unsafe {
            runtime
                .dtoh(&mut readback, base_ptr + (i * gran) as u64)
                .expect("dtoh")
        };
        assert_eq!(
            pat, &readback,
            "expert {i} content must survive the rolled-back transition"
        );
    }
    println!("injected fault rolled back range-precisely, content identical ✓");
}
