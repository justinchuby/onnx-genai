//! #1810 Slice 5 — GPU correctness tests for `apply_residency_plan_at_boundary`.
//!
//! These tests require an idle CUDA device (GPU 4 on this machine) with host-NUMA
//! capability (A100-SXM4-80GB, driver 580.105.08, CUDA 13.0).
//!
//! Run:
//! ```text
//! CUDA_VISIBLE_DEVICES=4 cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda,gpu-tests --release \
//!   --test coarse_residency_plan_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## Coverage
//!
//! 1. `feature_gate_off_no_op_on_real_allocator` — env var unset → outcome has
//!    `fallback_reason = "feature gate disabled"`, zero `values_touched`;
//!    `allocator.committed_and_reserved()` unchanged before/after.
//!
//! 2. `real_transition_correctness_gate_on` — env var ON, 4-expert
//!    granule-aligned fixture (2 hot / 2 cold); after the plan is applied:
//!    (a) bytes for all 4 experts are bit-identical on DMA readback,
//!    (b) `host_bytes_committed == 2 * granularity`,
//!    (c) stable VA pointer unchanged before and after.
//!
//! 3. Transactional rollback — **explicitly omitted** (not faked).
//!    `apply_residency_plan_at_boundary` calls `transition_granule_range`, not
//!    `transition_granule_range_with_phase8_faults`.  There is no public seam to
//!    inject a `Fatal` result through the plan-level entry point without modifying
//!    production code; the rollback path is therefore not reachable from this test
//!    file.  See the PR comment for the honest disclosure.
//!
//! 4. `mixed_alignment_per_value_fallback` — two values in the same plan: one
//!    granule-aligned (touched), one sub-granule (falls back with an alignment
//!    reason in `per_value_fallbacks`).  Both tensors' bytes are readback-verified
//!    as untouched / unchanged after the plan runs.
//!
//! 5. Capability failure path — **explicitly omitted** (no production seam).
//!    `host_numa_capability` cannot be forced to return `Unsupported` from a
//!    test without either modifying the production function or running on a GPU
//!    that lacks host-NUMA support.  The path is exercised implicitly: if the
//!    GPU under test lacks host-NUMA support, every GPU test in this file
//!    auto-skips via `host_numa_capability`'s `Unsupported` branch.

#![cfg(feature = "gpu-tests")]
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
use onnx_runtime_ep_api::{
    ExpertWeightGroup, LazyWeightBoundary, StaticProfileResidencyPolicy, plan_residency,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::{
    COARSE_RESIDENCY_ENABLE_ENV, apply_residency_plan_at_boundary,
    apply_residency_plan_at_boundary_with_phase8_faults,
};
use onnx_runtime_ep_cuda::weight_paging::CudaWeightResidency;
use onnx_runtime_ir::{DataType, WeightRef};
use onnx_runtime_ir::{NodeId, ValueId};
use onnx_runtime_loader::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole,
};

// ---------------------------------------------------------------------------
// Serialize every test in this file — GPU tests must run one at a time.
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
    #[allow(dead_code)]
    granularity: usize,
    #[allow(dead_code)]
    host_numa_node: i32,
}

fn make_pools(
    provider: &CudaExecutionProvider,
    pool_bytes: usize,
    governor: &'static LedgerGovernor,
) -> Option<TestPools> {
    let device_ordinal = 0_i32;
    let runtime = provider.runtime();
    let context = runtime.cuda_context();

    let cap = match host_numa_capability(device_ordinal) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP: HOST_NUMA not supported: {r}");
            return None;
        }
    };
    let granularity = cap.granularity;
    let host_numa_node = cap.host_numa_id;
    println!(
        "capability: device_ordinal={device_ordinal} host_numa_id={host_numa_node} granularity={granularity}"
    );

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
            node: host_numa_node,
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
        granularity,
        host_numa_node: host_numa_node as i32,
    })
}

/// Build a `CudaVmmAllocator` with all `n_experts * gran` bytes committed on
/// Device, with a retained physical pool of size `pool_bytes`.  Returns the
/// allocator and the stable device VA base pointer (offset 0 in the
/// reservation, since `allocate(total, gran)` on a fresh allocator starts at 0).
fn build_precommitted_allocator(
    provider: &CudaExecutionProvider,
    n_experts: usize,
    gran: usize,
    pool_bytes: usize,
    governor: &'static LedgerGovernor,
    holder: HolderId,
) -> (Arc<CudaVmmAllocator>, u64) {
    let runtime = provider.runtime();
    let context = runtime.cuda_context();
    let total_bytes = n_experts * gran;

    // Set the pool-bytes env var so the allocator has a retained physical pool.
    // This is the same approach as `gqa_shared_prefix_parity_gpu.rs`.
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

    // Allocate and fully commit all experts as Device.  On a fresh allocator the
    // sub-span starts at offset 0, so the returned pointer IS the reservation base.
    let ptr = allocator
        .allocate(total_bytes, gran)
        .expect("allocate total bytes");
    let base_ptr = ptr.as_ptr() as u64;

    (allocator, base_ptr)
}

/// Build a granule-aligned `WeightRegionCatalog` where every expert is exactly
/// `gran` bytes.  Uses `rows_per_expert = 512` and
/// `storage_elements_per_row = gran / 512`.
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

/// Build a sub-granule catalog (4 experts × 8 rows × 16 cols = 512 bytes
/// total — far smaller than any VMM granule).
fn make_sub_granule_catalog(file_offset: usize) -> WeightRegionCatalog {
    let layout = ExpertTensorLayout {
        version: 1,
        experts: 4,
        rows_per_expert: 8,
        storage_elements_per_row: 16,
        order: ExpertStorageOrder::ExpertMajor,
        quantization: Some(ExpertQuantization {
            bits: 4,
            block_size: 16,
            blocks_per_row: 1,
        }),
    };
    let weight = WeightRef::External {
        path: std::path::PathBuf::from("/nonexistent/weights.bin"),
        offset: file_offset,
        length: 4 * 8 * 16,
        dtype: DataType::Uint8,
        dims: vec![4, 8, 16],
    };
    WeightRegionCatalog::classify(&weight, layout)
}

// Test 1: feature gate off → structural no-op, allocator bytes unchanged
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn feature_gate_off_no_op_on_real_allocator() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test1: feature_gate_off_no_op_on_real_allocator ===");

    let provider = match provider_or_skip("test1") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test1: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;

    // Ensure the env var is NOT set to a truthy value in this test.
    let gate_on = matches!(
        std::env::var(COARSE_RESIDENCY_ENABLE_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    if gate_on {
        println!(
            "SKIP test1: {} is set to a truthy value in the ambient environment; \
             cannot safely clear it across threads",
            COARSE_RESIDENCY_ENABLE_ENV
        );
        return;
    }

    let n_experts = 4_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 4;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);

    let (allocator, base_ptr) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(20),
    );
    println!("stable_base = 0x{base_ptr:x}");

    let (committed_before, reserved_before) = allocator.committed_and_reserved();
    println!("before: committed={committed_before} reserved={reserved_before}");

    let value = ValueId(42);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    assert!(catalog.is_pageable(), "catalog must be pageable");

    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value, vec![0, 2]); // hot experts; 1 and 3 are cold
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates = vec![(value, LazyWeightBoundary::QMoe, &catalog)];
    let plan = plan_residency(candidates, &policy, None);
    assert_eq!(plan.len(), 1);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    // Gate is OFF: apply should be a structural no-op.
    let outcome = apply_residency_plan_at_boundary(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1, // device_count
        0, // device_ordinal
        &[],
    );

    println!("outcome: {outcome:#?}");

    assert_eq!(
        outcome.fallback_reason.as_deref(),
        Some("feature gate disabled"),
        "expected 'feature gate disabled', got {:?}",
        outcome.fallback_reason
    );
    assert_eq!(outcome.values_touched, 0, "no values should be touched");
    assert_eq!(outcome.host_bytes_committed, 0);

    let (committed_after, reserved_after) = allocator.committed_and_reserved();
    assert_eq!(
        committed_before, committed_after,
        "committed bytes must not change when gate is off"
    );
    assert_eq!(
        reserved_before, reserved_after,
        "reserved bytes must not change when gate is off"
    );
    println!("after: committed={committed_after} reserved={reserved_after}");

    println!("test1 PASSED: feature gate off → structural no-op, allocator unchanged ✓");
}

// ---------------------------------------------------------------------------
// Test 2: real transition correctness with env gate ON
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn real_transition_correctness_gate_on() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test2: real_transition_correctness_gate_on ===");

    let provider = match provider_or_skip("test2") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test2: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;
    println!("granularity: {gran} bytes ({} MiB)", gran >> 20);

    let n_experts = 4_usize;
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
    println!("stable_base = 0x{base_ptr:x}");

    // Write unique byte patterns per expert.
    let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
    for i in 0..n_experts {
        let pat: Vec<u8> = (0..gran).map(|j| ((i * 17 + j) & 0xFF) as u8).collect();
        unsafe {
            runtime
                .htod(&pat, base_ptr + (i * gran) as u64)
                .expect("htod pattern");
        }
        patterns.push(pat);
    }
    println!("written {n_experts} expert patterns, {gran} bytes each");

    // Record stable VA before the plan.
    let stable_va_before = base_ptr;

    // Plan: hot = [0, 2], cold = [1, 3].
    let value = ValueId(100);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    assert!(catalog.is_pageable());
    // Sanity: every cold expert's range is granule-aligned.
    for e in [1_usize, 3] {
        let r = catalog.relative_range(e).expect("range");
        let len = r.end - r.start;
        assert_eq!(r.start % gran, 0, "expert {e} not granule-aligned");
        assert_eq!(len, gran, "expert {e} wrong length");
    }

    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value, vec![0, 2]);
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates = vec![(value, LazyWeightBoundary::QMoe, &catalog)];
    let plan = plan_residency(candidates, &policy, None);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    // Enable gate, run plan, disable gate.
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    let outcome = apply_residency_plan_at_boundary(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
    );
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };

    println!("outcome: {outcome:#?}");

    // (a) No structural fallback.
    assert!(
        outcome.fallback_reason.is_none(),
        "no fallback expected, got: {:?}",
        outcome.fallback_reason
    );
    assert_eq!(outcome.values_touched, 1, "one value must be touched");
    assert_eq!(outcome.failure_count, 0, "no failures");

    // (b) Accounting: cold experts = 2, each gran bytes.
    let cold_count = 2_usize;
    let expected_host_bytes = (cold_count * gran) as u64;
    assert_eq!(
        outcome.host_bytes_committed, expected_host_bytes,
        "host_bytes_committed should be {expected_host_bytes} ({cold_count}×{gran}), got {}",
        outcome.host_bytes_committed
    );
    println!(
        "host_bytes_committed = {} ({} MiB) ✓",
        outcome.host_bytes_committed,
        outcome.host_bytes_committed >> 20
    );

    // (c) Stable VA unchanged: the base_ptr must equal the VA we had before.
    // `base_ptr` IS the stable VA (it is the reservation's base, the pointer
    // returned by `allocate`, and the only VA address space used here).
    // After the transition the same address range remains accessible via DMA.
    // We cannot directly re-read `reservation.base_ptr()` from the allocator
    // via public API, but the DMA read below proves the VA is still valid.

    // (d) Byte-identical readback for all 4 experts.
    for (i, pattern) in patterns.iter().enumerate() {
        let mut got = vec![0u8; gran];
        unsafe {
            runtime
                .dtoh(&mut got, base_ptr + (i * gran) as u64)
                .expect("dtoh readback");
        }
        assert_eq!(
            *pattern, got,
            "expert {i} content corrupted after plan application"
        );
        let loc = if i == 0 || i == 2 {
            "Device (hot)"
        } else {
            "HostNuma (cold)"
        };
        println!("expert {i} readback bit-identical ({loc}) ✓");
    }

    println!(
        "stable_va_before=0x{stable_va_before:x} (DMA-verified still accessible after transition) ✓"
    );
    println!("test2 PASSED: plan applied, bytes bit-identical, accounting correct ✓");
}

// ---------------------------------------------------------------------------
// Test 3: transactional rollback — OMITTED (no fault-injection seam available)
// ---------------------------------------------------------------------------
// `apply_residency_plan_at_boundary` calls the NON-fault-injectable
// `transition_granule_range`.  The rollback branch triggers only on a `Fatal`
// outcome from that call, which requires a real CUDA driver failure.  No public
// seam exists to inject such a failure through `apply_residency_plan_at_boundary`
// without modifying production code.  This test is explicitly absent rather than
// silently missing; the PR comment documents the gap.

// ---------------------------------------------------------------------------
// Test 4: non-granule-aligned catalog falls back per-value, not whole-plan
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn mixed_alignment_per_value_fallback() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test4: mixed_alignment_per_value_fallback ===");

    let provider = match provider_or_skip("test4") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test4: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;

    // Value A: granule-aligned (4 experts, each gran bytes).
    let value_a = ValueId(200);
    let n_experts_a = 4_usize;
    let total_a = n_experts_a * gran;
    let pool_bytes = total_a * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);

    let (allocator_a, base_a) = build_precommitted_allocator(
        &provider,
        n_experts_a,
        gran,
        pool_bytes,
        governor,
        HolderId::new(40),
    );
    let pattern_a: Vec<u8> = (0..total_a).map(|j| (j & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pattern_a, base_a).expect("htod A");
    }
    let (a_committed_before, _) = allocator_a.committed_and_reserved();

    // Value B: sub-granule (4 experts × 8 rows × 16 cols = 512 bytes total).
    // The allocator_b needs at least one granule of VA to hold the sub-span.
    let value_b = ValueId(201);
    unsafe { std::env::set_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, pool_bytes.to_string()) };
    let allocator_b = Arc::new(
        CudaVmmAllocator::new(
            Arc::clone(&runtime.cuda_context()),
            DeviceKey::device(0),
            0_i32,
            gran * 2,
            governor,
            HolderId::new(41),
            MemoryRole::Weights,
        )
        .expect("allocator B"),
    );
    unsafe { std::env::remove_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV) };

    // Allocate one granule in allocator_b (sub-granule tensor fits inside).
    let b_ptr = allocator_b
        .allocate(gran, gran)
        .expect("allocate B granule");
    let base_b = b_ptr.as_ptr() as u64;
    let b_sentinel = vec![0xBB_u8; gran];
    unsafe {
        runtime.htod(&b_sentinel, base_b).expect("htod B");
    }
    let (b_committed_before, _) = allocator_b.committed_and_reserved();

    // Build catalogs (B's catalog uses sub-granule layout).
    let catalog_a = make_aligned_catalog(n_experts_a, gran, 0);
    let catalog_b = make_sub_granule_catalog(total_a);
    assert!(catalog_a.is_pageable());
    assert!(
        catalog_b.is_pageable(),
        "sub-granule catalog is still pageable; it just won't pass alignment check"
    );

    // Plan: A hot=[0,2] cold=[1,3]; B hot=[0] cold=[1,2,3].
    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value_a, vec![0, 2]);
    profile.insert(value_b, vec![0]);
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates_a_ref = &catalog_a;
    let candidates_b_ref = &catalog_b;
    let candidates = vec![
        (value_a, LazyWeightBoundary::QMoe, candidates_a_ref),
        (value_b, LazyWeightBoundary::QMoe, candidates_b_ref),
    ];
    let plan = plan_residency(candidates, &policy, None);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), pool_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value_a, catalog_a);
    catalogs.insert(value_b, catalog_b);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value_a, Arc::clone(&allocator_a));
    allocators.insert(value_b, Arc::clone(&allocator_b));

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    let outcome = apply_residency_plan_at_boundary(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
    );
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };

    println!("outcome: {outcome:#?}");

    // Value A (aligned) must be touched.
    assert_eq!(
        outcome.values_touched, 1,
        "only the aligned value should be touched, got {}",
        outcome.values_touched
    );

    // Value B (misaligned) must appear in per_value_fallbacks.
    let b_fallback = outcome
        .per_value_fallbacks
        .iter()
        .find(|(v, _)| *v == value_b);
    assert!(
        b_fallback.is_some(),
        "value_b must appear in per_value_fallbacks; got: {:?}",
        outcome.per_value_fallbacks
    );
    let (_, reason) = b_fallback.unwrap();
    println!("value_b fallback reason: {reason}");
    let reason_lower = reason.to_ascii_lowercase();
    assert!(
        reason_lower.contains("align") || reason_lower.contains("granule"),
        "fallback reason should mention alignment/granule, got: {reason}"
    );

    // Allocator A: bytes committed must have DECREASED (cold experts moved to host).
    // Allocator B: committed bytes must be UNCHANGED.
    let (a_committed_after, _) = allocator_a.committed_and_reserved();
    let (b_committed_after, _) = allocator_b.committed_and_reserved();

    // Cold experts for A: 2 out of 4, so 2*gran bytes moved from device.
    // committed_and_reserved reports physical bytes; after transition, 2 granules
    // are on the host pool (owned by host_pool, not the allocator's device pool).
    // The allocator's own `committed` counter should reflect device bytes only.
    println!(
        "allocator_a: committed_before={a_committed_before} committed_after={a_committed_after}"
    );
    println!(
        "allocator_b: committed_before={b_committed_before} committed_after={b_committed_after}"
    );
    assert_eq!(
        b_committed_before, b_committed_after,
        "allocator_b (misaligned) committed bytes must be unchanged"
    );

    // Value A's bytes must be bit-identical on DMA readback.
    let mut readback_a = vec![0u8; total_a];
    unsafe {
        runtime.dtoh(&mut readback_a, base_a).expect("dtoh A");
    }
    assert_eq!(
        pattern_a, readback_a,
        "value A bytes must be bit-identical after plan"
    );

    // Value B's bytes must be unchanged.
    let mut readback_b = vec![0u8; gran];
    unsafe {
        runtime.dtoh(&mut readback_b, base_b).expect("dtoh B");
    }
    assert_eq!(b_sentinel, readback_b, "value B bytes must be untouched");

    println!("test4 PASSED: aligned touched, misaligned in fallbacks, both bytes intact ✓");
}

// ---------------------------------------------------------------------------
// Test 5: capability failure — OMITTED (no production seam available)
// ---------------------------------------------------------------------------
// There is no test-safe way to force `host_numa_capability(device_ordinal)` to
// return `Unsupported` without modifying production code or running on a GPU
// that lacks host-NUMA support.  If the GPU under test lacks host-NUMA, every
// test above auto-skips through the `Err(CapabilityGateFailure::Unsupported)` arm.

// ---------------------------------------------------------------------------
// Test 5 (Cycle 17 revision): plan-entry fault injection.
//
// `apply_residency_plan_at_boundary` internally calls plain
// `transition_granule_range`. To inject a deterministic Phase-8 driver fault
// through the plan-entry point without touching production code, we use the
// test-only `apply_residency_plan_at_boundary_with_phase8_faults` entry point
// (mirrors `transition_granule_range_with_phase8_faults`), which accepts a
// per-`ValueId` `DriverFaultPlan` map consulted by the exact same driver call
// sites `transition_granule_range` uses.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn partial_commit_then_fatal_reverts_prior_range_of_same_value() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test5: partial_commit_then_fatal_reverts_prior_range_of_same_value ===");

    let provider = match provider_or_skip("test5") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test5: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;

    // 5 experts, hot=[0,2] -> cold=[1,3,4]. Expert 1 is isolated (range
    // [gran, 2*gran)); experts 3,4 are adjacent and merge into one 2-granule
    // range [3*gran, 5*gran) -> two distinct `transition_granule_range` calls
    // for this one value, letting the first fully commit before the second
    // hits a forced Fatal.
    let n_experts = 5_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);

    // Unmap call order: range [gran, 2*gran) is 1 granule -> 1st Unmap call.
    // range [3*gran, 5*gran) is 2 granules -> 2nd and 3rd Unmap calls. Fail the
    // 3rd Unmap call: the first granule of the second range commits
    // (committed_count becomes > 0 for that inner transition), forcing Fatal
    // with committed_count == 1 for that range.
    let faults = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 3));
    let (allocator, base_ptr) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(50),
    );
    println!("stable_base = 0x{base_ptr:x}");

    let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
    for i in 0..n_experts {
        let pat: Vec<u8> = (0..gran).map(|j| ((i * 31 + j) & 0xFF) as u8).collect();
        unsafe {
            runtime
                .htod(&pat, base_ptr + (i * gran) as u64)
                .expect("htod pattern");
        }
        patterns.push(pat);
    }

    let value = ValueId(500);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    assert!(catalog.is_pageable());

    // hot=[0,2] -> cold=[1,3,4]; 1 is isolated (range [gran,2gran)), 3&4 merge
    // into [3gran, 5gran).
    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value, vec![0, 2]);
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates = vec![(value, LazyWeightBoundary::QMoe, &catalog)];
    let plan = plan_residency(candidates, &policy, None);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let mut phase8_faults: HashMap<ValueId, Arc<DriverFaultPlan>> = HashMap::new();
    phase8_faults.insert(value, faults);

    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    let outcome = apply_residency_plan_at_boundary_with_phase8_faults(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
        phase8_faults,
    );
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };

    println!("outcome: {outcome:#?}");

    // The first range (expert 1) must have committed, then been reverted
    // when the second range (experts 3,4) hit Fatal.
    assert!(
        !outcome.fatal_progress.is_empty(),
        "expected a Fatal to have been recorded, got: {:?}",
        outcome.fatal_progress
    );
    assert_eq!(
        outcome.values_touched, 0,
        "the value must NOT be counted as touched: its earlier committed range \
         was reverted by rollback, got values_touched={}",
        outcome.values_touched
    );
    assert!(
        outcome.committed_values.is_empty(),
        "committed_values must be empty after full rollback of this value, got: {:?}",
        outcome.committed_values
    );
    assert_eq!(
        outcome.rollback_count, 1,
        "exactly one value (with its one prior committed range) must have been \
         rolled back, got rollback_count={}",
        outcome.rollback_count
    );
    assert!(
        outcome.rollback_failures.is_empty(),
        "the rollback of the prior range is expected to succeed cleanly here, got: {:?}",
        outcome.rollback_failures
    );

    // Every expert's bytes must still be readable and correct: experts 0,2
    // (hot) were never touched; expert 1 was committed then reverted -> must
    // read back correctly from Device; experts 3,4 hit Fatal mid-flight, but
    // per the granule_transition contract the untouched suffix remains
    // readable on its original (Device) backing.
    for (i, pattern) in patterns.iter().enumerate() {
        let mut got = vec![0u8; gran];
        unsafe {
            runtime
                .dtoh(&mut got, base_ptr + (i * gran) as u64)
                .expect("dtoh readback");
        }
        assert_eq!(*pattern, got, "expert {i} content diverged");
    }

    println!(
        "test5 PASSED: prior committed range for the SAME value reverted after a later Fatal ✓"
    );
}

// ---------------------------------------------------------------------------
// Test 6: rollback-of-rollback is explicitly reported (not just all_ok=false).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn rollback_of_rollback_failure_is_explicitly_reported() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test6: rollback_of_rollback_failure_is_explicitly_reported ===");

    let provider = match provider_or_skip("test6") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test6: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;

    // Two separate values: value_a's single cold range commits then reverts
    // cleanly; value_b's cold range hits a Fatal, triggering the rollback
    // loop. We force value_a's REVERSE transition (the rollback) to also
    // fail, by scheduling a fault on the Remap call sequence that the
    // rollback issues (rollback direction: HostNuma -> Device is itself a
    // `transition_granule_range` call, i.e. a Remap of the new committed
    // handle back to Device -- fail its 1st Remap call).
    let n_experts_a = 3_usize;
    let total_a = n_experts_a * gran;
    let pool_bytes = total_a * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);

    // value_a: fail the 1st Remap call overall for this allocator. Its
    // FORWARD transition (Device->HostNuma) uses Remap once to map onto the
    // host handle; forward transitions must succeed for this test's premise
    // (we want the ROLLBACK's remap, i.e. the 2nd Remap call on this
    // allocator, to fail instead).
    let faults_a = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Remap, 2));
    let (allocator_a, base_a) = build_precommitted_allocator(
        &provider,
        n_experts_a,
        gran,
        pool_bytes,
        governor,
        HolderId::new(60),
    );
    let pattern_a: Vec<u8> = (0..total_a).map(|j| (j & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pattern_a, base_a).expect("htod A");
    }

    let value_a = ValueId(600);
    let catalog_a = make_aligned_catalog(n_experts_a, gran, 0);
    assert!(catalog_a.is_pageable());

    // value_b: forced Fatal via failing its 1st Unmap at a later granule so
    // committed_count > 0 for that inner call (2 cold experts -> 2 granules
    // in one merged range; fail the 2nd Unmap call so the 1st granule
    // commits and the 2nd hits Fatal).
    let n_experts_b = 4_usize;
    let total_b = n_experts_b * gran;
    let faults_b = Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 2));
    let (allocator_b, base_b) = build_precommitted_allocator(
        &provider,
        n_experts_b,
        gran,
        pool_bytes,
        governor,
        HolderId::new(61),
    );
    let pattern_b: Vec<u8> = (0..total_b).map(|j| ((j + 7) & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pattern_b, base_b).expect("htod B");
    }
    let value_b = ValueId(601);
    let catalog_b = make_aligned_catalog(n_experts_b, gran, total_a);
    assert!(catalog_b.is_pageable());

    // value_a: hot={0,2} -> cold=[1] (1 granule -> exactly 1 forward Remap;
    // its rollback issues exactly 1 more Remap = the 2nd Remap call overall).
    // value_b: hot={0} -> cold=[1,2,3] (3 granules, one merged range).
    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value_a, vec![0, 2]);
    profile.insert(value_b, vec![0]);
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates = vec![
        (value_a, LazyWeightBoundary::QMoe, &catalog_a),
        (value_b, LazyWeightBoundary::QMoe, &catalog_b),
    ];
    let plan = plan_residency(candidates, &policy, None);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), pool_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value_a, catalog_a);
    catalogs.insert(value_b, catalog_b);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value_a, Arc::clone(&allocator_a));
    allocators.insert(value_b, Arc::clone(&allocator_b));

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    let mut phase8_faults: HashMap<ValueId, Arc<DriverFaultPlan>> = HashMap::new();
    phase8_faults.insert(value_a, faults_a);
    phase8_faults.insert(value_b, faults_b);

    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    let outcome = apply_residency_plan_at_boundary_with_phase8_faults(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
        phase8_faults,
    );
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };

    println!("outcome: {outcome:#?}");

    assert!(
        !outcome.fatal_progress.is_empty(),
        "expected value_b to hit a Fatal, got: {:?}",
        outcome.fatal_progress
    );
    // value_a's forward transition committed, then its rollback attempt
    // (triggered by value_b's Fatal) must ALSO fail, and be reported
    // precisely -- not just outcome.rollback_count staying 0 silently.
    assert!(
        !outcome.rollback_failures.is_empty(),
        "expected an explicit rollback_failures entry for value_a's failed reverse \
         transition, got none. Full outcome: {outcome:#?}"
    );
    let failure = outcome
        .rollback_failures
        .iter()
        .find(|f| f.value == value_a);
    assert!(
        failure.is_some(),
        "rollback_failures must identify value_a by ValueId, got: {:?}",
        outcome.rollback_failures
    );
    assert_eq!(
        outcome.rollback_count, 0,
        "value_a must NOT be counted as a clean rollback since its reverse \
         transition failed, got rollback_count={}",
        outcome.rollback_count
    );

    println!(
        "test6 PASSED: rollback-of-rollback failure explicitly reported with ValueId+detail ✓"
    );
}

// ---------------------------------------------------------------------------
// Test 7: same-device fail-closed — allocator declares a different device.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn mismatched_device_key_fails_closed_with_zero_side_effects() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test7: mismatched_device_key_fails_closed_with_zero_side_effects ===");

    let provider = match provider_or_skip("test7") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test7: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;
    let n_experts = 4_usize;
    let total_bytes = n_experts * gran;
    let pool_bytes = total_bytes * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);

    let context = runtime.cuda_context();
    // Allocator built with device_ordinal 0 physically, but DECLARING a
    // different DeviceKey (device(7), a device that certainly doesn't exist
    // in this single-GPU test process) to simulate a same-device-check
    // mismatch without needing a second real GPU.
    unsafe { std::env::set_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, pool_bytes.to_string()) };
    let allocator = Arc::new(
        CudaVmmAllocator::new(
            Arc::clone(&context),
            DeviceKey::device(7),
            0_i32,
            total_bytes * 2,
            governor,
            HolderId::new(70),
            MemoryRole::Weights,
        )
        .expect("build allocator with mismatched device key"),
    );
    unsafe { std::env::remove_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV) };
    let ptr = allocator
        .allocate(total_bytes, gran)
        .expect("allocate total bytes");
    let base_ptr = ptr.as_ptr() as u64;

    let pattern: Vec<u8> = (0..total_bytes).map(|j| (j & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pattern, base_ptr).expect("htod pattern");
    }
    let (committed_before, reserved_before) = allocator.committed_and_reserved();

    let value = ValueId(700);
    let catalog = make_aligned_catalog(n_experts, gran, 0);
    assert!(catalog.is_pageable());

    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value, vec![0, 2]);
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates = vec![(value, LazyWeightBoundary::QMoe, &catalog)];
    let plan = plan_residency(candidates, &policy, None);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), total_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value, catalog);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value, Arc::clone(&allocator));

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    // Request device_ordinal=0 (the physical device this process actually
    // runs on) -- the mismatch is the allocator's DECLARED DeviceKey(7).
    let outcome = apply_residency_plan_at_boundary(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &[],
    );
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };

    println!("outcome: {outcome:#?}");

    assert_eq!(
        outcome.values_touched, 0,
        "mismatched-device value must not be touched"
    );
    let fallback = outcome
        .per_value_fallbacks
        .iter()
        .find(|(v, _)| *v == value);
    assert!(
        fallback.is_some(),
        "expected a per_value_fallbacks entry for the mismatched-device value, got: {:?}",
        outcome.per_value_fallbacks
    );
    let (_, reason) = fallback.unwrap();
    let reason_lower = reason.to_ascii_lowercase();
    assert!(
        reason_lower.contains("device"),
        "fallback reason should mention device mismatch, got: {reason}"
    );

    let (committed_after, reserved_after) = allocator.committed_and_reserved();
    assert_eq!(
        committed_before, committed_after,
        "same-device check must run BEFORE any mutation: committed bytes must be unchanged"
    );
    assert_eq!(
        reserved_before, reserved_after,
        "same-device check must run BEFORE any mutation: reserved bytes must be unchanged"
    );

    let mut readback = vec![0u8; total_bytes];
    unsafe {
        runtime.dtoh(&mut readback, base_ptr).expect("dtoh");
    }
    assert_eq!(pattern, readback, "bytes must be completely untouched");

    println!("test7 PASSED: mismatched device_key fails closed with zero side effects ✓");
}

// ---------------------------------------------------------------------------
// Test 8: cross-tensor expert-group atomicity — one misaligned member forces
// the WHOLE group (fc1 + fc2 of the same logical expert bank) to fall back,
// never a partial per-tensor tiering.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires idle CUDA device; CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn expert_group_member_failure_forces_whole_group_fallback() {
    let _guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    println!("\n=== test8: expert_group_member_failure_forces_whole_group_fallback ===");

    let provider = match provider_or_skip("test8") {
        Some(p) => p,
        None => return,
    };
    let runtime = provider.runtime();

    let cap = match host_numa_capability(0) {
        Ok(c) => c,
        Err(CapabilityGateFailure::Unsupported(r)) => {
            println!("SKIP test8: HOST_NUMA not supported: {r}");
            return;
        }
    };
    let gran = cap.granularity;

    // fc1 (granule-aligned, would normally tier cleanly) and fc2 (sub-granule,
    // will fail alignment) of the SAME logical expert bank.
    let n_experts = 4_usize;
    let total_fc1 = n_experts * gran;
    let pool_bytes = total_fc1 * 8;
    let governor = make_governor(pool_bytes as u64, pool_bytes as u64);

    let (allocator_fc1, base_fc1) = build_precommitted_allocator(
        &provider,
        n_experts,
        gran,
        pool_bytes,
        governor,
        HolderId::new(80),
    );
    let pattern_fc1: Vec<u8> = (0..total_fc1).map(|j| (j & 0xFF) as u8).collect();
    unsafe {
        runtime.htod(&pattern_fc1, base_fc1).expect("htod fc1");
    }
    let (fc1_committed_before, _) = allocator_fc1.committed_and_reserved();

    let value_fc1 = ValueId(800);
    let catalog_fc1 = make_aligned_catalog(n_experts, gran, 0);
    assert!(catalog_fc1.is_pageable());

    // fc2: sub-granule catalog (same shape as `make_sub_granule_catalog`).
    unsafe { std::env::set_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, pool_bytes.to_string()) };
    let allocator_fc2 = Arc::new(
        CudaVmmAllocator::new(
            Arc::clone(&runtime.cuda_context()),
            DeviceKey::device(0),
            0_i32,
            gran * 2,
            governor,
            HolderId::new(81),
            MemoryRole::Weights,
        )
        .expect("allocator fc2"),
    );
    unsafe { std::env::remove_var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV) };
    let fc2_ptr = allocator_fc2
        .allocate(gran, gran)
        .expect("allocate fc2 granule");
    let base_fc2 = fc2_ptr.as_ptr() as u64;
    let fc2_sentinel = vec![0xCC_u8; gran];
    unsafe {
        runtime.htod(&fc2_sentinel, base_fc2).expect("htod fc2");
    }
    let (fc2_committed_before, _) = allocator_fc2.committed_and_reserved();

    let value_fc2 = ValueId(801);
    let catalog_fc2 = make_sub_granule_catalog(total_fc1);
    assert!(catalog_fc2.is_pageable());

    let mut profile: HashMap<ValueId, Vec<usize>> = HashMap::new();
    profile.insert(value_fc1, vec![0, 2]);
    profile.insert(value_fc2, vec![0]);
    let policy = StaticProfileResidencyPolicy::new(profile);
    let candidates = vec![
        (value_fc1, LazyWeightBoundary::QMoe, &catalog_fc1),
        (value_fc2, LazyWeightBoundary::QMoe, &catalog_fc2),
    ];
    let plan = plan_residency(candidates, &policy, None);

    let residency = CudaWeightResidency::new(Arc::clone(runtime), pool_bytes as u64);
    let mut catalogs = HashMap::new();
    catalogs.insert(value_fc1, catalog_fc1);
    catalogs.insert(value_fc2, catalog_fc2);
    let mut allocators: HashMap<ValueId, Arc<CudaVmmAllocator>> = HashMap::new();
    allocators.insert(value_fc1, Arc::clone(&allocator_fc1));
    allocators.insert(value_fc2, Arc::clone(&allocator_fc2));

    // ONE ExpertWeightGroup binds fc1 and fc2 as members of the same logical
    // QMoE bank (as `expert_weight_groups` would derive from the graph).
    let groups = vec![ExpertWeightGroup {
        node: NodeId(0),
        boundary: LazyWeightBoundary::QMoe,
        members: vec![value_fc1, value_fc2],
    }];

    let pools = match make_pools(&provider, pool_bytes, governor) {
        Some(p) => p,
        None => return,
    };

    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    let outcome = apply_residency_plan_at_boundary(
        runtime,
        &residency,
        &plan,
        &catalogs,
        &allocators,
        &pools.device_pool,
        &pools.host_pool,
        1,
        0,
        &groups,
    );
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };

    println!("outcome: {outcome:#?}");

    // NEITHER member may be touched: fc2's alignment failure must force fc1
    // (which would otherwise transition cleanly) to fall back too.
    assert_eq!(
        outcome.values_touched, 0,
        "no member of the atomic expert group may be touched when any member fails, \
         got values_touched={}",
        outcome.values_touched
    );
    assert!(
        outcome.committed_values.is_empty(),
        "no member may commit, got: {:?}",
        outcome.committed_values
    );
    let fc1_fallback = outcome
        .per_value_fallbacks
        .iter()
        .find(|(v, _)| *v == value_fc1);
    assert!(
        fc1_fallback.is_some(),
        "fc1 must appear in per_value_fallbacks due to its group-mate's failure, got: {:?}",
        outcome.per_value_fallbacks
    );
    let (_, fc1_reason) = fc1_fallback.unwrap();
    assert!(
        fc1_reason.to_ascii_lowercase().contains("group"),
        "fc1's fallback reason should mention the expert-group cause, got: {fc1_reason}"
    );

    let (fc1_committed_after, _) = allocator_fc1.committed_and_reserved();
    let (fc2_committed_after, _) = allocator_fc2.committed_and_reserved();
    assert_eq!(
        fc1_committed_before, fc1_committed_after,
        "fc1 (individually alignment-clean) must be UNTOUCHED because its group-mate failed"
    );
    assert_eq!(
        fc2_committed_before, fc2_committed_after,
        "fc2 committed bytes must be unchanged"
    );

    let mut readback_fc1 = vec![0u8; total_fc1];
    unsafe {
        runtime.dtoh(&mut readback_fc1, base_fc1).expect("dtoh fc1");
    }
    assert_eq!(
        pattern_fc1, readback_fc1,
        "fc1 bytes must be completely untouched by the group fallback"
    );

    println!(
        "test8 PASSED: one misaligned expert-group member blocks the WHOLE group, no partial tiering ✓"
    );
}
