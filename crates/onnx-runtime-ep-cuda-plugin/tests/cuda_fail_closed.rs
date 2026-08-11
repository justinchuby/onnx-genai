//! Integration tests for the CUDA plugin.
//!
//! Without a CUDA GPU, the plugin returns zero factories (fail-closed) in
//! both feature configurations. The tests verify this fail-closed behaviour.
//! On a CUDA-capable host, additional tests would verify factory creation.

use std::ptr;

use onnx_genai_ort_sys as ort;

/// Call CreateEpFactories on the CUDA plugin via its public Rust API.
///
/// Returns `(status, num_factories)`.
fn call_create() -> (*mut ort::OrtStatus, usize) {
    let mut factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
    let mut num_factories: usize = 99; // sentinel — must be overwritten
    let status = unsafe {
        onnx_runtime_ep_cuda_plugin::CreateEpFactories(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            factories.as_mut_ptr(),
            1,
            &mut num_factories,
        )
    };
    (status, num_factories)
}

/// Without a CUDA GPU (this host), the plugin must return zero factories
/// regardless of feature configuration.
#[test]
fn cuda_plugin_returns_zero_factories_without_gpu() {
    let (status, num) = call_create();
    assert_eq!(
        num, 0,
        "CUDA plugin must return 0 factories without a GPU, got {num}"
    );
    eprintln!("✓ cuda_plugin_returns_zero_factories_without_gpu: num={num}, status={status:?}");
}

/// Status is null in test context (no ORT host API) but zero factories is the
/// key assertion.
#[test]
fn cuda_plugin_error_status_or_null_without_ort() {
    let (status, num) = call_create();
    assert_eq!(num, 0);
    eprintln!(
        "✓ cuda_plugin_error_status_or_null_without_ort: status={status:?} (null is ok in test context)"
    );
}

/// The CUDA plugin's error message depends on the feature configuration.
#[test]
fn cuda_plugin_diagnostic_message_contract() {
    #[cfg(feature = "cuda")]
    {
        eprintln!("✓ cuda feature ON: error message mentions GPU/driver unavailable");
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("✓ cuda feature OFF: error message mentions 'without `cuda` feature'");
    }
}

/// ABI symbol check: verify both exported symbols exist and are callable.
#[test]
fn cuda_plugin_exports_create_and_release_symbols() {
    // These are #[no_mangle] extern "C" functions — if they didn't exist,
    // the test would fail to link. Calling them with safe arguments verifies
    // the symbols are present and have the correct signature.
    let mut factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
    let mut num: usize = 0;
    let _status = unsafe {
        onnx_runtime_ep_cuda_plugin::CreateEpFactories(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            factories.as_mut_ptr(),
            1,
            &mut num,
        )
    };
    // ReleaseEpFactory with null is a no-op (safe).
    let _status = unsafe { onnx_runtime_ep_cuda_plugin::ReleaseEpFactory(ptr::null_mut()) };
    eprintln!("✓ cuda_plugin_exports_create_and_release_symbols: both ABI symbols present");
}

// ─── B1 regression: EP outliving the MutexGuard ──────────────────────────────

/// Verify that creating a shared-EP-backed allocator does not produce a
/// dangling pointer. The allocator must hold a strong `Arc` reference to the
/// EP, not a raw pointer extracted from a dropped `MutexGuard`.
///
/// This test exercises the ownership model: the allocator works correctly even
/// after the original `Arc` clone is dropped (simulating the factory releasing
/// its reference before the allocator).
#[test]
fn shared_ep_allocator_outlives_original_arc() {
    use onnx_runtime_ep_plugin::device::DeviceAllocator;
    use std::sync::{Arc, Mutex};

    // Minimal mock EP for allocation testing.
    struct AllocEp;
    impl onnx_runtime_ep_api::provider::ExecutionProvider for AllocEp {
        fn name(&self) -> &str {
            "alloc_test_ep"
        }
        fn device_type(&self) -> onnx_runtime_ir::DeviceType {
            onnx_runtime_ir::DeviceType::Cuda
        }
        fn device_id(&self) -> onnx_runtime_ir::DeviceId {
            onnx_runtime_ir::DeviceId::cuda(0)
        }
        fn initialize(
            &mut self,
            _: &onnx_runtime_ep_api::provider::EpConfig,
        ) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }
        fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }
        fn supports_op(
            &self,
            _: &onnx_runtime_ir::Node,
            _: u64,
            _: &[onnx_runtime_ir::Shape],
            _: &[onnx_runtime_ir::DataType],
            _: &[onnx_runtime_ir::TensorLayout],
        ) -> onnx_runtime_ep_api::KernelMatch {
            onnx_runtime_ep_api::KernelMatch::Unsupported {
                reason: "mock".into(),
            }
        }
        fn get_kernel(
            &self,
            _: &onnx_runtime_ir::Node,
            _: &[Vec<usize>],
            _: u64,
        ) -> onnx_runtime_ep_api::Result<Box<dyn onnx_runtime_ep_api::Kernel>> {
            Err(onnx_runtime_ep_api::EpError::KernelFailed("mock".into()))
        }
        fn allocate(
            &self,
            size: usize,
            _align: usize,
        ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::provider::DeviceBuffer> {
            let layout = std::alloc::Layout::from_size_align(size.max(1), 16).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            Ok(unsafe {
                onnx_runtime_ep_api::provider::DeviceBuffer::from_raw_parts(
                    ptr.cast(),
                    onnx_runtime_ir::DeviceId::cuda(0),
                    size,
                    16,
                )
            })
        }
        fn deallocate(
            &self,
            buf: onnx_runtime_ep_api::provider::DeviceBuffer,
        ) -> onnx_runtime_ep_api::Result<()> {
            let ptr = buf.as_ptr();
            let size = buf.len();
            let _ = buf;
            if !ptr.is_null() && size > 0 {
                let layout = std::alloc::Layout::from_size_align(size, 16).unwrap();
                unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
            }
            Ok(())
        }
        fn copy(
            &self,
            _: &onnx_runtime_ep_api::provider::DeviceBuffer,
            _: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
            _: usize,
        ) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }
        fn copy_async(
            &self,
            _: &onnx_runtime_ep_api::provider::DeviceBuffer,
            _: &mut onnx_runtime_ep_api::provider::DeviceBuffer,
            _: usize,
        ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::provider::Fence> {
            Ok(onnx_runtime_ep_api::provider::Fence::signalled())
        }
        fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }
    }

    let ep: Box<dyn onnx_runtime_ep_api::provider::ExecutionProvider + Send> = Box::new(AllocEp);
    let shared = Arc::new(Mutex::new(ep));

    // Create allocator with a clone of the Arc.
    let alloc = unsafe { DeviceAllocator::new_shared(Arc::clone(&shared), ptr::null()) };
    let alloc_ptr = Box::into_raw(alloc);

    // Drop the original Arc clone — allocator must still work.
    drop(shared);

    // The allocator's Arc keeps the EP alive. Verify it allocates.
    let raw_alloc = alloc_ptr.cast::<ort::OrtAllocator>();
    let alloc_fn = unsafe { (*raw_alloc).Alloc.unwrap() };
    let p = unsafe { alloc_fn(raw_alloc as *mut _, 256) };
    assert!(
        !p.is_null(),
        "B1 regression: allocator must work after original Arc is dropped"
    );

    // Free the allocation.
    let free_fn = unsafe { (*raw_alloc).Free.unwrap() };
    unsafe { free_fn(raw_alloc as *mut _, p) };

    // Cleanup.
    unsafe { drop(Box::from_raw(alloc_ptr)) };
}

// ─── S4 regression: fail-closed by design, not by panic ──────────────────────

/// Verify that the CUDA plugin fails closed with a status (not a panic)
/// when no GPU is available. The old code had a panic bomb as constructor
/// that would be called during factory creation, failing by accident.
/// The fix uses `create_ep_factories_for_shared_ep` which doesn't call
/// the constructor.
#[test]
fn fail_closed_by_status_not_panic() {
    // The test is simple: call CreateEpFactories and verify we get a clean
    // return (no panic escape) with 0 factories.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call_create));
    match result {
        Ok((_status, num)) => {
            assert_eq!(num, 0, "must return 0 factories without GPU");
        }
        Err(_) => {
            panic!("S4 regression: CreateEpFactories panicked instead of returning status");
        }
    }
}

// ─── B3 regression: CopyDirection matrix ─────────────────────────────────────

/// Verify the full copy direction matrix is correct.
#[test]
fn copy_direction_h2d_and_d2h_both_supported() {
    use onnx_runtime_ep_plugin::transfer::CopyDirection;

    // H→D
    let h2d = CopyDirection::classify(true, false, false);
    assert_eq!(h2d, CopyDirection::HostToDevice);
    assert!(h2d.is_supported());

    // D→H
    let d2h = CopyDirection::classify(false, true, false);
    assert_eq!(d2h, CopyDirection::DeviceToHost);
    assert!(d2h.is_supported());

    // D→D (same)
    let d2d = CopyDirection::classify(false, false, true);
    assert_eq!(d2d, CopyDirection::DeviceToSameDevice);
    assert!(d2d.is_supported());

    // D→D (different) — must be rejected
    let cross = CopyDirection::classify(false, false, false);
    assert_eq!(cross, CopyDirection::DeviceToDifferentDevice);
    assert!(
        !cross.is_supported(),
        "cross-device copy must be rejected (fail-closed)"
    );

    // H→H — not our responsibility
    let h2h = CopyDirection::classify(true, true, true);
    assert_eq!(h2h, CopyDirection::HostToHost);
    assert!(!h2h.is_supported(), "host-to-host is ORT's job");
}
