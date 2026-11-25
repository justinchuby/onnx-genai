//! Real-device lifetime/transaction regression; compilation is not GPU validation.
use super::*;
use onnx_runtime_ep_api::{DeviceGraphSlot, DevicePtr, DevicePtrMut, ExecutionProvider};

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests"
)]
fn rewarm_preserves_success_pair_and_both_graph_owners() {
    let ep = crate::CudaExecutionProvider::new_default().expect("CUDA device must be available");
    let runtime = ep.runtime();
    runtime.set_defer_eager_sync(true);
    for circular in [false, true] {
        let past = ep.allocate(24, 256).unwrap();
        let update = ep.allocate(16, 256).unwrap();
        let indices = ep.allocate(8, 256).unwrap();
        let out_a = ep.allocate(24, 256).unwrap();
        let out_b = ep.allocate(24, 256).unwrap();
        // SAFETY: each live allocation covers its complete native-endian payload.
        unsafe {
            runtime
                .htod(
                    &[1u32, 2, 3, 4, 5, 6].map(u32::to_ne_bytes).concat(),
                    cuptr(past.as_ptr()),
                )
                .unwrap();
            runtime
                .htod(
                    &[10u32, 20, 30, 40].map(u32::to_ne_bytes).concat(),
                    cuptr(update.as_ptr()),
                )
                .unwrap();
            runtime
                .htod(&2i64.to_ne_bytes(), cuptr(indices.as_ptr()))
                .unwrap();
        }
        let kernel = TensorScatterKernel {
            runtime: runtime.clone(),
            raw_axis: 1,
            circular,
            warm: Mutex::new(WarmMetadata::new(PersistentMetadata::new(runtime.clone()))),
        };
        // Both layouts contain six cache elements but require different strides.
        // A uses explicit start=2 (wraps only in circular mode); B defaults to 0.
        let execute =
            |shape: &[usize], update_shape: &[usize], indexed: bool, output: *mut c_void| {
                let strides = compute_contiguous_strides(shape);
                let update_strides = compute_contiguous_strides(update_shape);
                let mut inputs = vec![
                    TensorView::new(
                        DevicePtr(past.as_ptr()),
                        DataType::Uint32,
                        shape,
                        &strides,
                        ep.device_id(),
                    ),
                    TensorView::new(
                        DevicePtr(update.as_ptr()),
                        DataType::Uint32,
                        update_shape,
                        &update_strides,
                        ep.device_id(),
                    ),
                ];
                if indexed {
                    inputs.push(TensorView::new(
                        DevicePtr(indices.as_ptr()),
                        DataType::Int64,
                        &[1],
                        &[1],
                        ep.device_id(),
                    ));
                }
                kernel.execute(
                    &inputs,
                    &mut [TensorMut::new(
                        DevicePtrMut(output),
                        DataType::Uint32,
                        shape,
                        &strides,
                        ep.device_id(),
                    )],
                )
            };
        let read = |ptr: *const c_void| {
            let mut bytes = [0u8; 24];
            // SAFETY: each output owns 24 bytes; readback waits for its producer.
            unsafe {
                runtime.dtoh(&mut bytes, cuptr(ptr)).unwrap();
            }
            bytes
                .chunks_exact(4)
                .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let counters = || {
            (
                runtime.allocation_counts(),
                runtime.transfer_counts(),
                runtime.forced_synchronization_count(),
            )
        };
        let a = [1, 3, 2];
        let ua = [1, 2, 2];
        let b = [1, 6, 1];
        let ub = [1, 2, 1];
        let ap = out_a.as_ptr().cast_mut();
        let bp = out_b.as_ptr().cast_mut();
        execute(&a, &ua, true, ap).unwrap();
        let expected_a = if circular {
            [30, 40, 3, 4, 10, 20]
        } else {
            [1, 2, 3, 4, 10, 20]
        };
        assert_eq!(read(out_a.as_ptr()), expected_a);
        let owner_a = kernel.device_graph_resources()[0].identity();
        let baseline = counters();
        for _ in 0..8 {
            execute(&a, &ua, true, ap).unwrap();
        }
        assert_eq!(
            counters(),
            baseline,
            "warm A must not allocate/upload/synchronize metadata"
        );

        #[cfg(feature = "gpu-tests")]
        {
            runtime.fail_warm_transaction_at_for_test(1);
            let error = execute(&b, &ub, false, bp).unwrap_err().to_string();
            assert!(
                error.contains("injected staged warm-cache failure"),
                "{error}"
            );
            let warm = kernel.warm.lock().unwrap();
            assert_eq!(warm.signature.as_ref().unwrap().cache_shape, a);
            assert_eq!(
                warm.metadata.device_graph_resource().unwrap().identity(),
                owner_a
            );
        }
        ep.wait_for_deferred_releases().unwrap();
        let baseline = counters();
        execute(&a, &ua, true, ap).unwrap();
        assert_eq!(
            counters(),
            baseline,
            "failed B must preserve the entire warm A pair"
        );
        assert_eq!(read(out_a.as_ptr()), expected_a);

        // An exact signature alone is insufficient: the active capture must own
        // the actual immutable metadata address before recording its consumer.
        runtime
            .begin_graph_capture_with_resources(&[], vec![])
            .unwrap();
        let error = execute(&a, &ua, true, ap).unwrap_err().to_string();
        assert!(error.contains("owner"), "{error}");
        runtime.abort_graph_capture().unwrap();
        runtime.begin_graph_capture(&[&kernel]).unwrap();
        let baseline = counters();
        let error = execute(&b, &ub, false, bp).unwrap_err().to_string();
        assert!(error.contains("warm the exact signature"), "{error}");
        assert_eq!(
            counters(),
            baseline,
            "capture mismatch must refuse before mutation"
        );
        execute(&a, &ua, true, ap).unwrap();
        runtime.end_graph_capture().unwrap();

        execute(&b, &ub, false, bp).unwrap();
        let expected_b = [10, 20, 3, 4, 5, 6];
        assert_eq!(read(out_b.as_ptr()), expected_b);
        assert_ne!(kernel.device_graph_resources()[0].identity(), owner_a);
        let baseline = counters();
        for _ in 0..8 {
            execute(&b, &ub, false, bp).unwrap();
        }
        assert_eq!(
            counters(),
            baseline,
            "warm B must not allocate/upload/synchronize metadata"
        );
        runtime
            .begin_graph_capture_in(DeviceGraphSlot::Verify, &[&kernel])
            .unwrap();
        execute(&b, &ub, false, bp).unwrap();
        runtime
            .end_graph_capture_in(DeviceGraphSlot::Verify)
            .unwrap();
        // A zero-update copy publishes its signature without replacing the
        // unused descriptor; an empty output does no device work at all.
        let owner_b = kernel.device_graph_resources()[0].identity();
        let allocations = runtime.allocation_counts();
        execute(&b, &[1, 0, 1], false, bp).unwrap();
        assert_eq!(runtime.allocation_counts(), allocations);
        assert_eq!(read(out_b.as_ptr()), [1, 2, 3, 4, 5, 6]);
        assert_eq!(kernel.device_graph_resources()[0].identity(), owner_b);
        assert_eq!(
            kernel
                .warm
                .lock()
                .unwrap()
                .signature
                .as_ref()
                .unwrap()
                .update_shape,
            [1, 0, 1]
        );
        let baseline = counters();
        execute(&[1, 0, 1], &[1, 0, 1], false, bp).unwrap();
        assert_eq!(counters(), baseline);
        assert!(kernel.cuda_graph_compatible());
        ep.wait_for_deferred_releases().unwrap();
        let baseline = runtime.allocation_counts();
        drop(kernel);
        ep.wait_for_deferred_releases().unwrap();
        assert_eq!(
            runtime.allocation_counts(),
            baseline,
            "both graphs retain their own descriptor after kernel drop"
        );
        // SAFETY: both outputs retain their complete 24-byte backing.
        unsafe {
            runtime.htod(&[0xa5; 24], cuptr(out_a.as_ptr())).unwrap();
            runtime.htod(&[0xa5; 24], cuptr(out_b.as_ptr())).unwrap();
        }
        runtime.replay_graph().unwrap();
        runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
        assert_eq!(read(out_a.as_ptr()), expected_a);
        assert_eq!(read(out_b.as_ptr()), expected_b);
        assert_eq!(runtime.check_capture_error().unwrap(), 0);
        let before_reset = runtime.allocation_counts();
        assert!(runtime.reset_graph().unwrap());
        ep.wait_for_deferred_releases().unwrap();
        assert!(runtime.allocation_counts().frees > before_reset.frees);
        // SAFETY: B retains its complete output. Poisoning prevents this replay
        // from passing on output left over from the previous replay.
        unsafe {
            runtime.htod(&[0xa5; 24], cuptr(out_b.as_ptr())).unwrap();
        }
        runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
        assert_eq!(read(out_b.as_ptr()), expected_b);
        let before_reset = runtime.allocation_counts();
        assert!(runtime.reset_graph_in(DeviceGraphSlot::Verify).unwrap());
        ep.wait_for_deferred_releases().unwrap();
        assert!(runtime.allocation_counts().frees > before_reset.frees);
        for buffer in [past, update, indices, out_a, out_b] {
            ep.deallocate(buffer).unwrap();
        }
    }
}
