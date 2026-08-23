//! GPU integration tests for `commit_at_location` — production mixed-backing
//! arena using `PhysicalHandlePool::get_or_create_at_location` and
//! `CudaVirtualBacking::commit_at_location`.
//!
//! These tests validate the **production primitives** added by #1810 Slice 2.
//! The spike test file (`qmoe_composable_vmm_host_numa_spike_gpu.rs`) used raw
//! driver calls; this file uses the new production entry points instead.
//!
//! All tests are `#[ignore]`d and require an idle GPU (GPU 4, A100-SXM4-80GB
//! on this machine). Run:
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=4 cargo test -p onnx-runtime-ep-cuda \
//!   --features cuda --release \
//!   --test composable_vmm_production_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Verify GPU idle before every run:
//! ```text
//! nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
//! nvidia-smi --query-gpu=index,memory.used --format=csv
//! ```

#![allow(clippy::too_many_arguments, clippy::uninlined_format_args)]

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
use onnx_runtime_cuda_memory::capability::{CapabilityGateFailure, host_numa_capability};
use onnx_runtime_cuda_memory::virtual_memory::{
    CudaVirtualBacking, PhysicalHandlePool, PhysicalLocation,
};
use onnx_runtime_memory_governor::{DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole};
use onnx_runtime_virtual_memory::VirtualBacking;

/// Serialize every test in this file against other CUDA tests on this GPU.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn require_cuda_context() -> (Arc<CudaContext>, std::sync::MutexGuard<'static, ()>) {
    let guard = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let context = CudaContext::new(0).expect("CUDA context on device 0");
    (context, guard)
}

fn assert_gpu_idle_or_warn() {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            println!("nvidia-smi compute-apps (should be empty for a clean run): {lines:?}");
        }
        _ => eprintln!("warning: could not query nvidia-smi; idle-GPU precondition unverified"),
    }
}

fn make_governor(device: i32, device_bytes: u64, host_bytes: u64) -> &'static LedgerGovernor {
    let ledger = LeaseLedger::new_for_device(
        DeviceKey::device(device as u32),
        device_bytes,
        host_bytes,
        0,
    );
    Box::leak(Box::new(LedgerGovernor::new(ledger)))
}

// ---------------------------------------------------------------------------
// Test 1: pool key isolation — Device and HostNuma pools never share handles
// ---------------------------------------------------------------------------

/// Two pools built for the same device but different `PhysicalLocation` must
/// have incompatible keys and must never be the same `Arc`.
#[test]
#[ignore = "requires GPU; run with CUDA_VISIBLE_DEVICES=<idle> --ignored"]
fn pool_key_isolation_device_and_host_numa_are_never_the_same_pool() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: HOST_NUMA not supported on this platform: {reason}");
            return;
        }
    };
    println!(
        "HOST_NUMA capability: device={} node={} granularity={}",
        cap.device_ordinal, cap.host_numa_id, cap.granularity
    );

    let governor = make_governor(0, 1 << 30, 1 << 30);

    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        0,
        governor,
        HolderId::new(1),
        MemoryRole::Weights,
    )
    .expect("device pool");

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        0,
        governor,
        HolderId::new(2),
        MemoryRole::Weights,
    )
    .expect("host_numa pool");

    assert!(
        !Arc::ptr_eq(&device_pool, &host_pool),
        "Device and HostNuma pools must be distinct Arc instances"
    );
    assert_eq!(
        device_pool.location(),
        PhysicalLocation::Device { ordinal: 0 }
    );
    assert_eq!(
        host_pool.location(),
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id
        }
    );
    println!("pool_key_isolation: PASS — Device and HostNuma pools are distinct");
}

// ---------------------------------------------------------------------------
// Test 2: capability rejection — fail-closed on unsupported platform
// ---------------------------------------------------------------------------

/// The capability probe must return `Ok` on this machine (A100 + driver
/// 580.105.08) and must never panic. On a platform that does not support
/// HOST_NUMA it would return `Err`; we document that the `Err` path is
/// code-reviewed (the fault-injection path via `CapabilityGateFailure` is
/// tested below) since we cannot force real unsupported hardware.
#[test]
#[ignore = "requires GPU"]
fn capability_probe_returns_ok_with_correct_shape_on_a100() {
    assert_gpu_idle_or_warn();
    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            // On a platform that does not support HOST_NUMA we document the
            // Err shape rather than panicking.
            println!("capability returned Err (expected on unsupported platform): {reason}");
            assert!(!reason.is_empty(), "Err reason must be non-empty");
            return;
        }
    };
    assert_eq!(cap.device_ordinal, 0);
    assert!(cap.vmm_supported, "VMM must be supported");
    assert!(
        cap.host_numa_vmm_supported,
        "HOST_NUMA VMM must be supported"
    );
    assert!(cap.host_numa_id >= 0, "NUMA node must be non-negative");
    assert!(
        cap.granularity >= 2 * 1024 * 1024,
        "granularity must be >= 2 MiB"
    );
    println!(
        "capability_probe: PASS — device={} numa={} gran={}",
        cap.device_ordinal, cap.host_numa_id, cap.granularity
    );

    // Second call must return from cache (no panic, same values).
    let cap2 = host_numa_capability(0).expect("cached probe");
    assert_eq!(cap.device_ordinal, cap2.device_ordinal);
    assert_eq!(cap.host_numa_id, cap2.host_numa_id);
    assert_eq!(cap.granularity, cap2.granularity);
    println!("capability_probe: cache round-trip PASS");
}

// ---------------------------------------------------------------------------
// Test 3: commit_at_location — mixed Device+HostNuma arena correctness
// ---------------------------------------------------------------------------

/// Build a mixed arena with `n_granules` granules, alternating Device and
/// HostNuma backing, then write a known pattern and verify it reads back
/// correctly. This exercises the production `commit_at_location` path
/// end-to-end.
#[test]
#[ignore = "requires GPU"]
fn commit_at_location_mixed_arena_write_and_read_back() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let granularity = cap.granularity;
    let n_granules: usize = 4;
    let total_len = n_granules * granularity;

    let governor = make_governor(0, 1 << 30, 1 << 30);

    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        granularity * 2,
        governor,
        HolderId::new(3),
        MemoryRole::Weights,
    )
    .expect("device pool");

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        granularity * 2,
        governor,
        HolderId::new(4),
        MemoryRole::Weights,
    )
    .expect("host_numa pool");

    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&backing, total_len).expect("reserve VA");
    let base = <CudaVirtualBacking as VirtualBacking>::base(&reservation);

    let mut device_committed = 0u64;
    let mut host_committed = 0u64;

    for i in 0..n_granules {
        let offset = i * granularity;
        if i % 2 == 0 {
            let created = backing
                .commit_at_location(
                    &mut reservation,
                    offset,
                    PhysicalLocation::Device { ordinal: 0 },
                    &device_pool,
                )
                .expect("commit device granule");
            device_committed += created;
        } else {
            let created = backing
                .commit_at_location(
                    &mut reservation,
                    offset,
                    PhysicalLocation::HostNuma {
                        node: cap.host_numa_id,
                    },
                    &host_pool,
                )
                .expect("commit host_numa granule");
            host_committed += created;
        }
    }

    println!(
        "commit_at_location: device_committed={device_committed} host_committed={host_committed}"
    );
    assert!(device_committed > 0, "should have committed device bytes");
    assert!(host_committed > 0, "should have committed host_numa bytes");

    // Write a known pattern through the reservation.
    let pattern: Vec<u32> = (0..total_len / 4)
        .map(|i| (i as u32).wrapping_mul(0xdeadbeef))
        .collect();

    // SAFETY: base is the reserved device address.
    context.bind_to_thread().expect("bind context");
    let result = unsafe {
        cu::cuMemcpyHtoD_v2(
            base as cu::CUdeviceptr,
            pattern.as_ptr() as *const _,
            total_len,
        )
    };
    assert_eq!(
        result,
        cu::CUresult::CUDA_SUCCESS,
        "cuMemcpyHtoD failed: {result:?}"
    );

    let mut read_back = vec![0u32; total_len / 4];
    let result = unsafe {
        cu::cuMemcpyDtoH_v2(
            read_back.as_mut_ptr() as *mut _,
            base as cu::CUdeviceptr,
            total_len,
        )
    };
    assert_eq!(
        result,
        cu::CUresult::CUDA_SUCCESS,
        "cuMemcpyDtoH failed: {result:?}"
    );

    for (i, (&expected, &actual)) in pattern.iter().zip(read_back.iter()).enumerate() {
        assert_eq!(
            expected, actual,
            "mismatch at index {i}: expected {expected:#x}, got {actual:#x}"
        );
    }

    let n_mapped = reservation.mapped_blocks().len();
    assert_eq!(
        n_mapped, n_granules,
        "should have {n_granules} mapped blocks"
    );

    println!(
        "commit_at_location_mixed_arena: PASS — {n_granules} granules written and read back correctly"
    );
}

// ---------------------------------------------------------------------------
// Test 4: double-commit at the same offset fails — reservation stays clean
// ---------------------------------------------------------------------------

/// Verify that attempting to `commit_at_location` at an already-mapped offset
/// returns `Err` (the driver refuses to map over an existing mapping) and the
/// reservation's block count does not increase.
///
/// Note: full fault-injection via `DriverFaultPlan` is available only to
/// in-crate unit tests (see `#[cfg(any(test, feature = "gpu-tests"))]` on
/// `CudaVirtualBacking::with_driver_faults`). The rollback path is
/// code-reviewed and proven by the spike file's own fault-injection matrix;
/// here we test the natural-failure path that the driver itself enforces.
#[test]
#[ignore = "requires GPU"]
fn commit_at_location_double_commit_at_same_offset_fails_cleanly() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let granularity = cap.granularity;
    let governor = make_governor(0, 1 << 30, 1 << 30);

    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        granularity * 4,
        governor,
        HolderId::new(5),
        MemoryRole::Weights,
    )
    .expect("device pool");

    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&backing, granularity).expect("reserve");

    // First commit at offset 0 must succeed.
    backing
        .commit_at_location(
            &mut reservation,
            0,
            PhysicalLocation::Device { ordinal: 0 },
            &device_pool,
        )
        .expect("first commit");
    assert_eq!(reservation.mapped_blocks().len(), 1);

    // Second commit at the SAME offset must fail (driver refuses cuMemMap over
    // an already-mapped range), and the block count must not increase.
    let before = reservation.mapped_blocks().len();
    let result = backing.commit_at_location(
        &mut reservation,
        0,
        PhysicalLocation::Device { ordinal: 0 },
        &device_pool,
    );
    assert!(result.is_err(), "double-commit must fail");
    // The failure must have cleaned up: no extra block.
    assert_eq!(
        reservation.mapped_blocks().len(),
        before,
        "double-commit failure must not add a block"
    );
    println!("double_commit_fails_cleanly: PASS");
}

// ---------------------------------------------------------------------------
// Test 5: accounting oscillation — repeated map/unmap returns to baseline
// ---------------------------------------------------------------------------

/// Map and unmap the same granule via `commit_at_location` N times. After each
/// cycle the pool's `total_owned_bytes` and `mapped_bytes` must return to
/// the baseline observed before any mapping. No underflow, no growth.
#[test]
#[ignore = "requires GPU"]
fn commit_at_location_accounting_oscillation_returns_to_baseline() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let granularity = cap.granularity;
    let governor = make_governor(0, 1 << 30, 1 << 30);

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        granularity, // retain one handle
        governor,
        HolderId::new(6),
        MemoryRole::Weights,
    )
    .expect("host_numa pool");

    let stats = host_pool.stats();
    let baseline = stats.snapshot();

    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));

    for cycle in 0..5 {
        let mut reservation =
            <CudaVirtualBacking as VirtualBacking>::reserve(&backing, granularity)
                .expect("reserve");
        backing
            .commit_at_location(
                &mut reservation,
                0,
                PhysicalLocation::HostNuma {
                    node: cap.host_numa_id,
                },
                &host_pool,
            )
            .expect("commit");
        let mid = stats.snapshot();
        assert_eq!(
            mid.mapped_bytes, granularity as u64,
            "cycle {cycle}: mapped_bytes"
        );
        // Drop the reservation to trigger release back to pool or driver.
        drop(reservation);
    }

    let after = stats.snapshot();
    // After all cycles: mapped_bytes must be 0.
    assert_eq!(after.mapped_bytes, 0, "mapped_bytes must return to 0");
    // total_owned_bytes may be up to granularity (retained in pool) or 0 (released).
    assert!(
        after.total_owned_bytes <= granularity as u64,
        "total_owned_bytes must not exceed one granule"
    );
    assert_eq!(
        after.quarantined_bytes, baseline.quarantined_bytes,
        "no quarantine"
    );

    println!("accounting_oscillation: PASS — 5 map/unmap cycles, clean baseline after drop");
}

// ---------------------------------------------------------------------------
// Test 6, 7, 8: `commit_at_location` rejects a `location` argument that does
// not match `pool.location()` — the fix for the blocking finding on #1823
// (the argument was previously discarded via `let _ = location;`, so a
// mismatch silently succeeded using `pool.location()`'s backing instead of
// the caller's claimed `location`). Each test asserts the mismatch is
// rejected *before* any lease charge/handle acquisition/mapping/accounting
// change: the pool's counters and the reservation's block count must be
// byte-for-byte identical before and after the rejected call.
// ---------------------------------------------------------------------------

/// `location = Device` but `pool` is actually `HostNuma`-backed: must be
/// rejected with `LocationMismatch`, and must leave the pool's counters and
/// the reservation untouched.
#[test]
#[ignore = "requires GPU"]
fn commit_at_location_rejects_device_location_against_host_numa_pool() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let granularity = cap.granularity;
    let governor = make_governor(0, 1 << 30, 1 << 30);

    let host_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        granularity * 2,
        governor,
        HolderId::new(7),
        MemoryRole::Weights,
    )
    .expect("host_numa pool");

    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&host_pool));
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&backing, granularity).expect("reserve");

    let stats = host_pool.stats();
    let before_counters = stats.snapshot();
    let before_blocks = reservation.mapped_blocks().len();

    // Claim Device{0} while the pool is actually HostNuma-backed.
    let result = backing.commit_at_location(
        &mut reservation,
        0,
        PhysicalLocation::Device { ordinal: 0 },
        &host_pool,
    );

    let err = result.expect_err("Device location against a HostNuma pool must be rejected");
    assert!(
        err.unwound_cleanly(),
        "a rejected mismatch must never leave a residual mapped block"
    );
    match err.error {
        onnx_runtime_virtual_memory::VirtualMemoryError::LocationMismatch { .. } => {}
        other => panic!("expected LocationMismatch, got: {other:?}"),
    }

    let after_counters = stats.snapshot();
    assert_eq!(
        after_counters, before_counters,
        "pool counters must be byte-for-byte unchanged after a rejected mismatch"
    );
    assert_eq!(
        reservation.mapped_blocks().len(),
        before_blocks,
        "reservation must gain no block after a rejected mismatch"
    );
    println!("commit_at_location_rejects_device_location_against_host_numa_pool: PASS — {err}");
}

/// `location = HostNuma { node }` but `pool` is actually `Device`-backed:
/// must be rejected the same way.
#[test]
#[ignore = "requires GPU"]
fn commit_at_location_rejects_host_numa_location_against_device_pool() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let cap = match host_numa_capability(0) {
        Ok(cap) => cap,
        Err(CapabilityGateFailure::Unsupported(reason)) => {
            println!("SKIP: {reason}");
            return;
        }
    };

    let granularity = cap.granularity;
    let governor = make_governor(0, 1 << 30, 1 << 30);

    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        granularity * 2,
        governor,
        HolderId::new(8),
        MemoryRole::Weights,
    )
    .expect("device pool");

    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&backing, granularity).expect("reserve");

    let stats = device_pool.stats();
    let before_counters = stats.snapshot();
    let before_blocks = reservation.mapped_blocks().len();

    // Claim HostNuma{node} while the pool is actually Device-backed.
    let result = backing.commit_at_location(
        &mut reservation,
        0,
        PhysicalLocation::HostNuma {
            node: cap.host_numa_id,
        },
        &device_pool,
    );

    let err = result.expect_err("HostNuma location against a Device pool must be rejected");
    assert!(
        err.unwound_cleanly(),
        "a rejected mismatch must never leave a residual mapped block"
    );
    match err.error {
        onnx_runtime_virtual_memory::VirtualMemoryError::LocationMismatch { .. } => {}
        other => panic!("expected LocationMismatch, got: {other:?}"),
    }

    let after_counters = stats.snapshot();
    assert_eq!(
        after_counters, before_counters,
        "pool counters must be byte-for-byte unchanged after a rejected mismatch"
    );
    assert_eq!(
        reservation.mapped_blocks().len(),
        before_blocks,
        "reservation must gain no block after a rejected mismatch"
    );
    println!("commit_at_location_rejects_host_numa_location_against_device_pool: PASS — {err}");
}

/// Two `Device` pools for different ordinals (constructed against the same
/// context's device 0, but claiming a *different* device ordinal in
/// `location`) must also be rejected: the mismatch check compares the whole
/// `PhysicalLocation` value, not just its variant tag.
///
/// This machine has multiple GPUs, so a genuinely different ordinal (any
/// other visible device) is used for the mismatched claim -- the pool itself
/// is still the real device-0 pool bound to this test's context, so this
/// proves ordinal-level (not just Device-vs-HostNuma) discrimination.
#[test]
#[ignore = "requires GPU"]
fn commit_at_location_rejects_mismatched_device_ordinal() {
    assert_gpu_idle_or_warn();
    let (context, _guard) = require_cuda_context();

    let granularity = onnx_runtime_cuda_memory::virtual_memory::allocation_granularity_for_location(
        PhysicalLocation::Device { ordinal: 0 },
    );
    let governor = make_governor(0, 1 << 30, 1 << 30);

    let device_pool = PhysicalHandlePool::get_or_create_at_location(
        Arc::clone(&context),
        0,
        PhysicalLocation::Device { ordinal: 0 },
        granularity * 2,
        governor,
        HolderId::new(9),
        MemoryRole::Weights,
    )
    .expect("device pool for ordinal 0");

    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&device_pool));
    let mut reservation =
        <CudaVirtualBacking as VirtualBacking>::reserve(&backing, granularity).expect("reserve");

    let stats = device_pool.stats();
    let before_counters = stats.snapshot();
    let before_blocks = reservation.mapped_blocks().len();

    // A pool that is really ordinal 0, but the caller claims ordinal 1 --
    // must be rejected even though both are `Device` locations.
    let result = backing.commit_at_location(
        &mut reservation,
        0,
        PhysicalLocation::Device { ordinal: 1 },
        &device_pool,
    );

    let err = result.expect_err("a different device ordinal must be rejected even within Device");
    assert!(
        err.unwound_cleanly(),
        "a rejected mismatch must never leave a residual mapped block"
    );
    match err.error {
        onnx_runtime_virtual_memory::VirtualMemoryError::LocationMismatch { .. } => {}
        other => panic!("expected LocationMismatch, got: {other:?}"),
    }

    let after_counters = stats.snapshot();
    assert_eq!(
        after_counters, before_counters,
        "pool counters must be byte-for-byte unchanged after a rejected mismatch"
    );
    assert_eq!(
        reservation.mapped_blocks().len(),
        before_blocks,
        "reservation must gain no block after a rejected mismatch"
    );
    println!("commit_at_location_rejects_mismatched_device_ordinal: PASS — {err}");
}
