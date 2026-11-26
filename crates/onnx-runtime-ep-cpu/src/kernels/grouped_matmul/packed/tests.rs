use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel, Repacked};
use onnx_runtime_ep_api::{DevicePtrMut, WorkspaceRequirement};
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, dequantize_packed::Format};
fn kernel(format: Format, shape: [usize; 3]) -> Packed {
    let mut node = Node::new(NodeId(0), PACKED_OP, vec![], vec![]);
    node.attributes
        .insert("format".into(), Attribute::String(format.name().into()));
    node.attributes.insert(
        "shape".into(),
        Attribute::Ints(shape.map(|d| d as i64).to_vec()),
    );
    Packed {
        metadata: Metadata::from_node(&node).unwrap(),
    }
}
fn fixture(format: Format) -> [Owned; 3] {
    let (width, bytes) = format.block();
    let k = 2 * width;
    [
        Owned::f32(
            &[2, k],
            &(0..2 * k)
                .map(|i| (i as f32 % 7. - 3.) * 0.125)
                .collect::<Vec<_>>(),
        ),
        Owned::u8(
            &[12 * bytes],
            &(0..12 * bytes)
                .map(|i| ((i * 17 + 93) % 117) as u8)
                .collect::<Vec<_>>(),
        ),
        Owned::i64(&[2, 2], &[2, 0, 1, 2]),
    ]
}
fn requirement(k: &Packed, x: &[TensorView], y: &TensorMut) -> WorkspaceRequirement {
    let m = x
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect::<Vec<_>>();
    k.workspace_requirement_for_execution(
        x,
        std::slice::from_ref(y),
        &m,
        &[OutputMetadata::tensor(y.dtype, y.shape)],
        None,
    )
    .unwrap()
    .require_ready()
    .unwrap()
}
fn bytes(values: usize, routes: usize) -> usize {
    (values * 4).next_multiple_of(align_of::<usize>()) + routes * size_of::<usize>()
}
fn bits(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}
// Previous flattened route/block order, retaining full X and route snapshots.
fn reference(kernel: &Packed, inputs: &[TensorView]) -> Vec<f32> {
    let mut y = Owned::zeros_f32(&[2, 2, 2]);
    let l = packed_group_layout(&kernel.metadata, inputs, &[y.view_mut()]).unwrap();
    let routes =
        super::super::route_indices(Op::GroupedMatMul { transposed: true }, &inputs[2], l.groups)
            .unwrap();
    let activation = crate::kernels::to_dense_f32(&inputs[0]).unwrap();
    let packed = crate::dtype::to_dense::<u8>(&inputs[1]).unwrap();
    let (width, bytes) = kernel.metadata.format().block();
    let blocks = l.k / width;
    let decode = crate::kernels::dequantize_packed::decoder(kernel.metadata.format());
    let mut values = [0f32; 256];
    let mut output = vec![0f32; l.lanes];
    for (route, row) in output.chunks_exact_mut(l.n).enumerate() {
        let expert = routes[route];
        let token = route / l.selections;
        for (column, out) in row.iter_mut().enumerate() {
            let packed_row = (expert * l.n + column) * blocks * bytes;
            let mut sum = 0f32;
            for block in 0..blocks {
                let source = packed_row + block * bytes;
                decode(&packed[source..source + bytes], &mut values[..width]);
                let activation =
                    &activation[token * l.k + block * width..token * l.k + (block + 1) * width];
                for (&a, &weight) in activation.iter().zip(&values[..width]) {
                    sum += a * weight;
                }
            }
            *out = sum;
        }
    }
    output
}
#[test]
fn every_profile_and_strided_x_routes_match_previous_accumulation_bits() {
    for &format in Format::ALL {
        let width = format.block().0;
        let k = kernel(format, [3, 2, 2 * width]);
        for int32 in [false, true] {
            let mut f = fixture(format);
            if int32 {
                f[2] = Owned::i32(&[2, 2], &[2, 0, 1, 2]);
            }
            let expected = reference(&k, &f.iter().map(Owned::view).collect::<Vec<_>>());
            for mode in 0..6 {
                let x = Repacked::new(&f[0], mode);
                let routes = Repacked::new(&f[2], mode);
                let bank = Repacked::new(&f[1], if mode == 4 { 4 } else { 0 });
                let inputs = [x.view(), bank.view(), routes.view()];
                let mut output = Owned::zeros_f32(&[2, 2, 2]);
                assert_eq!(
                    requirement(&k, &inputs, &output.view_mut()).bytes,
                    bytes(
                        8 + Rows::new(PACKED_OP, "X", &inputs[0]).unwrap().scratch_len(),
                        4
                    ) as u64
                );
                assert!(k.supports_strided_input(0) && k.supports_strided_input(2));
                assert!(!k.supports_strided_input(1));
                k.execute_prepared(&inputs, &mut [output.view_mut()])
                    .unwrap();
                assert_eq!(
                    bits(output.to_f32()),
                    bits(expected.clone()),
                    "{} mode={mode} int32={int32}",
                    format.name()
                );
            }
        }
    }
}
#[test]
fn short_or_misaligned_lane_refuses_before_value_reads_or_initialization() {
    let k = kernel(Format::Q4_0, [3, 2, 64]);
    let f = fixture(Format::Q4_0);
    for mode in [0, 2, 4] {
        let x = Repacked::new(&f[0], mode);
        let mut inputs = [x.view(), f[1].view(), f[2].view()];
        inputs[0].data = DevicePtr(4usize as *const std::ffi::c_void);
        inputs[1].data = DevicePtr(8usize as *const std::ffi::c_void);
        inputs[2].data = DevicePtr(8usize as *const std::ffi::c_void);
        let mut y = Owned::f32(&[2, 2, 2], &[123.; 8]);
        let req = requirement(&k, &inputs, &y.view_mut());
        let mut backing = vec![0xaaaaaaaaaaaaaaaau128; (req.bytes as usize).div_ceil(16) + 1];
        let ptr = backing.as_mut_ptr().cast::<u8>();
        for lane in [
            None,
            Some(WorkspaceView::new(
                DevicePtrMut(ptr.cast()),
                req.bytes as usize - 1,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut(unsafe { ptr.add(1).cast() }),
                req.bytes as usize,
            )),
        ] {
            assert!(
                k.execute_with_workspace(&inputs, &mut [y.view_mut()], lane)
                    .is_err()
            );
            assert!(backing.iter().all(|&v| v == 0xaaaaaaaaaaaaaaaau128));
            assert_eq!(y.to_f32(), [123.; 8]);
        }
    }
}

#[test]
fn y_publication_follows_all_three_aliased_sources_and_keeps_padding_untouched() {
    let k = kernel(Format::Q4_0, [3, 2, 64]);
    let f = fixture(Format::Q4_0);
    let expected = reference(&k, &f.iter().map(Owned::view).collect::<Vec<_>>());
    for source in 0..3 {
        for destination in [24usize, 32, 40, 32 + f[source].bytes.len() / 2] {
            let mut memory = vec![
                0xccccccccccccccccccccccccccccccccu128;
                (f[source].bytes.len() + 192).div_ceil(16)
            ];
            let ptr = memory.as_mut_ptr().cast::<u8>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f[source].bytes.as_ptr(),
                    ptr.add(32),
                    f[source].bytes.len(),
                );
            }
            let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
            inputs[source].data = DevicePtr(unsafe { ptr.add(32).cast() });
            let mut out = TensorMut::new(
                DevicePtrMut(unsafe { ptr.add(destination).cast() }),
                DataType::Float32,
                &[2, 2, 2],
                &[4, 2, 1],
                DeviceId::cpu(),
            );
            k.execute_prepared(&inputs, std::slice::from_mut(&mut out))
                .unwrap();
            let actual = (0..8)
                .map(|i| unsafe { ptr.add(destination + i * 4).cast::<f32>().read_unaligned() })
                .collect();
            assert_eq!(
                bits(actual),
                bits(expected.clone()),
                "source={source},destination={destination}"
            );
        }
    }
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut memory = [123f32; 64];
    let mut out = TensorMut::new(
        DevicePtrMut(memory.as_mut_ptr().cast()),
        DataType::Float32,
        &[2, 2, 2],
        &[30, 10, -2],
        DeviceId::cpu(),
    )
    .with_byte_offset(8);
    k.execute_prepared(&inputs, std::slice::from_mut(&mut out))
        .unwrap();
    let offsets = [2, 0, 12, 10, 32, 30, 42, 40];
    for (i, &offset) in offsets.iter().enumerate() {
        assert_eq!(memory[offset].to_bits(), expected[i].to_bits());
    }
    for (i, &value) in memory.iter().enumerate() {
        if !offsets.contains(&i) {
            assert_eq!(value, 123.);
        }
    }
}
#[test]
fn zero_lanes_remain_read_free_without_admission_even_with_nonempty_bad_routes() {
    let k = kernel(Format::Q4_0, [3, 0, 64]);
    let x = Owned::f32(&[2, 64], &[0.; 128]);
    let bank = Owned::u8(&[0], &[]);
    let routes = Owned::i64(&[2, 2], &[-1; 4]);
    let mut inputs = [x.view(), bank.view(), routes.view()];
    for v in &mut inputs {
        v.device = DeviceId::cuda(0);
        v.data = DevicePtr(8usize as *const std::ffi::c_void);
    }
    let mut out = Owned::zeros_f32(&[2, 2, 0]);
    let mut y = out.view_mut();
    y.device = DeviceId::cuda(0);
    assert_eq!(requirement(&k, &inputs, &y).bytes, 0);
    k.execute(&inputs, std::slice::from_mut(&mut y)).unwrap();
}
#[test]
fn zero_k_still_checks_routes_and_host_access_without_reading_empty_banks() {
    let experts = isize::MAX as usize;
    let k = kernel(Format::Q4_0, [experts, 5, 0]);
    let x = TensorView::new(
        DevicePtr(8usize as *const std::ffi::c_void),
        DataType::Float32,
        &[1, 0],
        &[0, 1],
        DeviceId::cpu(),
    );
    let bank = TensorView::new(
        DevicePtr(std::ptr::dangling()),
        DataType::Uint8,
        &[0],
        &[1],
        DeviceId::cpu(),
    );
    let routes = Owned::i64(&[1, 1], &[(experts - 1) as i64]);
    let mut y = Owned::f32(&[1, 1, 5], &[123.; 5]);
    k.execute_prepared(&[x, bank, routes.view()], &mut [y.view_mut()])
        .unwrap();
    assert!(y.to_f32().iter().all(|v| v.to_bits() == 0));
    y = Owned::f32(&[1, 1, 5], &[123.; 5]);
    let bad = Owned::i64(&[1, 1], &[-1]);
    let text = k
        .execute_prepared(&[x, bank, bad.view()], &mut [y.view_mut()])
        .unwrap_err()
        .to_string();
    assert!(text.contains("group_indices[0] is -1"), "{text}");
    assert_eq!(y.to_f32(), [123.; 5]);
    for slot in 0..3 {
        let mut inputs = [x, bank, routes.view()];
        inputs[slot].device = DeviceId::cuda(0);
        assert!(
            k.execute_prepared(&inputs, &mut [y.view_mut()])
                .unwrap_err()
                .to_string()
                .contains("Host")
        );
    }
    for slot in 0..2 {
        let mut inputs = [x, bank, routes.view()];
        inputs[slot].data = DevicePtr(std::ptr::null());
        assert!(
            k.execute_prepared(&inputs, &mut [y.view_mut()])
                .unwrap_err()
                .to_string()
                .contains("invalid input")
        );
    }
}
#[test]
fn unsupported_packed_stride_and_invalid_routes_never_publish_output() {
    let k = kernel(Format::Q4_0, [3, 2, 64]);
    let mut f = fixture(Format::Q4_0);
    let strided = Repacked::new(&f[1], 2);
    let mut y = Owned::f32(&[2, 2, 2], &[123.; 8]);
    let text = k
        .execute_prepared(
            &[f[0].view(), strided.view(), f[2].view()],
            &mut [y.view_mut()],
        )
        .unwrap_err()
        .to_string();
    assert!(text.contains("packed weights must be contiguous"), "{text}");
    assert_eq!(y.to_f32(), [123.; 8]);
    f[2] = Owned::i64(&[2, 2], &[2, 0, 3, 2]);
    let text = k
        .execute_prepared(
            &f.iter().map(Owned::view).collect::<Vec<_>>(),
            &mut [y.view_mut()],
        )
        .unwrap_err()
        .to_string();
    assert!(text.contains("group_indices[2] is 3"), "{text}");
    assert_eq!(y.to_f32(), [123.; 8]);
}

#[test]
fn one_retained_packed_workspace_has_no_warm_issuer_callbacks_and_trims_exactly() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let expected = bytes(72, 4);
    let books = Arc::new(RecordingGovernor::new_host(expected as u64));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let k = kernel(Format::Q4_0, [3, 2, 64]);
    let f = fixture(Format::Q4_0);
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut y = Owned::zeros_f32(&[2, 2, 2]);
    let request = requirement(&k, &inputs, &y.view_mut());
    assert_eq!(request.bytes, bytes(8, 4) as u64);
    let first = provider
        .allocate_workspace(expected, request.alignment, request.role())
        .unwrap();
    let identity = first.managed_owner().unwrap().identity();
    assert!(
        provider
            .allocate_workspace(expected, request.alignment, request.role())
            .is_err()
    );
    provider.deallocate_workspace(first).unwrap();
    let before = (books.claims(), books.releases());
    drop(
        books
            .reserve(Tier::Host, 0, request.role(), HolderId::new(99))
            .unwrap(),
    );
    books.probe_zero_release();
    assert_eq!(
        (books.claims(), books.releases()),
        (before.0 + 1, before.1 + 1)
    );
    let calls = (books.claims(), books.releases());
    let strided_x = Repacked::new(&f[0], 2);
    let strided_inputs = [strided_x.view(), f[1].view(), f[2].view()];
    for iteration in 0..4096 {
        let inputs = if iteration % 2 == 0 {
            inputs.as_slice()
        } else {
            &strided_inputs
        };
        let request = requirement(&k, inputs, &y.view_mut());
        assert_eq!(
            request.bytes,
            bytes(if iteration % 2 == 0 { 8 } else { 72 }, 4) as u64
        );
        let mut allocation = provider
            .allocate_workspace(request.bytes as usize, request.alignment, request.role())
            .unwrap();
        assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
        let workspace = WorkspaceView::new(DevicePtrMut(allocation.as_mut_ptr()), allocation.len());
        k.execute_with_workspace(inputs, &mut [y.view_mut()], Some(workspace))
            .unwrap();
        provider.deallocate_workspace(allocation).unwrap();
    }
    assert_eq!(
        (books.claims(), books.releases()),
        calls,
        "including zero-byte callbacks"
    );
    assert_eq!(books.used(Tier::Host), expected as u64);
    assert!(y.to_f32().iter().all(|v| v.is_finite()));
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}

#[test]
fn metadata_cache_refines_every_format_for_actual_rows_and_rejects_stale_geometry() {
    for &format in Format::ALL {
        let depth = 2 * format.block().0;
        let k = kernel(format, [3, 2, depth]);
        let f = fixture(format);
        let mut metadata = f
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, &v.shape, true))
            .collect::<Vec<_>>();
        let facts = [OutputMetadata::Unresolved {
            dtype: DataType::Float32,
        }];
        let cached = Some(k.workspace_requirement(&metadata, &facts).unwrap());
        assert_eq!(cached, Some(WorkspacePreparation::Deferred));
        let mut y = Owned::zeros_f32(&[2, 2, 2]);
        for mode in [0, 2, 1, 3, 5, 4, 0] {
            let x = Repacked::new(&f[0], mode);
            let inputs = [x.view(), f[1].view(), f[2].view()];
            let row = Rows::new(PACKED_OP, "X", &inputs[0]).unwrap().scratch_len();
            let req = k
                .workspace_requirement_for_execution(
                    &inputs,
                    &[y.view_mut()],
                    &metadata,
                    &facts,
                    cached,
                )
                .unwrap()
                .require_ready()
                .unwrap();
            assert_eq!(
                req.bytes,
                bytes(8 + row, 4) as u64,
                "{} mode={mode}",
                format.name()
            );
            let mut backing = vec![0u128; (req.bytes as usize).div_ceil(16)];
            let lane = WorkspaceView::new(
                DevicePtrMut(backing.as_mut_ptr().cast()),
                req.bytes as usize,
            );
            k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(lane))
                .unwrap();
            assert_eq!(bits(y.to_f32()), bits(reference(&k, &inputs)));
        }
        let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        let mut wrong_y = Owned::zeros_f32(&[2, 2, 3]);
        assert!(
            k.workspace_requirement_for_execution(
                &inputs,
                &[wrong_y.view_mut()],
                &metadata,
                &facts,
                cached
            )
            .is_err()
        );
        let wrong_shape = [1, depth];
        metadata[0].shape = &wrong_shape;
        assert!(
            k.workspace_requirement_for_execution(
                &inputs,
                &[y.view_mut()],
                &metadata,
                &facts,
                cached
            )
            .is_err()
        );
    }
}
