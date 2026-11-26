use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel};
use onnx_runtime_ir::{Attribute, NodeId};
fn kernel(axes: &[i64], epsilon: f32) -> MeanVarianceNormalization {
    let mut node = Node::new(NodeId(0), mvn::OP, vec![], vec![]);
    node.attributes
        .insert("axes".into(), Attribute::Ints(axes.to_vec()));
    node.attributes
        .insert("epsilon".into(), Attribute::Float(epsilon));
    MeanVarianceNormalization(mvn::Attributes::parse(&node).unwrap())
}
fn refined(
    k: &MeanVarianceNormalization,
    x: &TensorView,
    y: &TensorMut,
) -> Result<WorkspaceRequirement> {
    k.workspace_requirement_for_execution(
        &[*x],
        std::slice::from_ref(y),
        &[TensorMetadata::new(x.dtype, x.shape, !x.is_absent())],
        &[OutputMetadata::from_view(y)],
        Some(WorkspacePreparation::Ready(WorkspaceRequirement::NONE)),
    )?
    .require_ready()
}
fn reference<T: NumericElem>(values: &[T], axes: &[i64], epsilon: f32) -> Vec<T>
where
    T::Acc: NormAcc,
{
    let coords = |j: usize| [j / 12, (j / 4) % 3, j % 4];
    (0..24)
        .map(|i| {
            let c = coords(i);
            let group = |j: usize| {
                let other = coords(j);
                (0..3).all(|d| axes.is_empty() || axes.contains(&(d as i64)) || c[d] == other[d])
            };
            let width = (0..24).filter(|&j| group(j)).count();
            let sum = (0..24)
                .filter(|&j| group(j))
                .fold(T::Acc::default(), |sum, j| sum.c_add(values[j].to_acc()));
            let mean = sum.c_div_usize(width);
            let squared = (0..24)
                .filter(|&j| group(j))
                .fold(T::Acc::default(), |sum, j| {
                    let d = values[j].to_acc().c_sub(mean);
                    sum.c_add(d.c_mul(d))
                });
            let denominator = squared
                .c_div_usize(width)
                .sqrt()
                .c_add(T::Acc::from_f32(epsilon));
            T::from_acc(values[i].to_acc().c_sub(mean).c_div(denominator))
        })
        .collect()
}
fn bits<T: NumericElem>(values: &[T]) -> &[u8] {
    // Tests instantiate numeric storage without padding or invalid bit patterns.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
#[test]
fn all_storage_and_interleaved_axes_match_immutable_coordinates_in_every_route() {
    fn check<T: NumericElem + Default>(values: Vec<T>)
    where
        T::Acc: NormAcc,
    {
        let shape = [2, 3, 4];
        let strides = [12, 4, 1];
        let bytes = std::mem::size_of_val(values.as_slice());
        let x = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(values.as_ptr().cast()),
            T::DTYPE,
            &shape,
            &strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        for axes in [
            vec![0],
            vec![1],
            vec![2],
            vec![0, 2],
            vec![0, 1],
            vec![1, 2],
            vec![],
        ] {
            let k = kernel(&axes, 0.125);
            let expected = reference(&values, &axes, 0.125);
            let mut direct = vec![T::default(); 24];
            let mut y = TensorMut::new(
                onnx_runtime_ep_api::DevicePtrMut(direct.as_mut_ptr().cast()),
                T::DTYPE,
                &shape,
                &strides,
                onnx_runtime_ir::DeviceId::cpu(),
            );
            assert_eq!(refined(&k, &x, &y).unwrap().bytes, 0);
            k.execute(&[x], std::slice::from_mut(&mut y)).unwrap();
            assert_eq!(bits(&direct), bits(&expected), "direct {axes:?}");
            let mut inplace = values.clone();
            let mut source = x;
            source.data = onnx_runtime_ep_api::DevicePtr(inplace.as_ptr().cast());
            let mut y = TensorMut::new(
                onnx_runtime_ep_api::DevicePtrMut(inplace.as_mut_ptr().cast()),
                T::DTYPE,
                &shape,
                &strides,
                onnx_runtime_ir::DeviceId::cpu(),
            );
            assert_eq!(refined(&k, &source, &y).unwrap().bytes, 0);
            k.execute(&[source], std::slice::from_mut(&mut y)).unwrap();
            assert_eq!(bits(&inplace), bits(&expected), "inplace {axes:?}");
            for reverse in [false, true] {
                let mut overlapped = vec![T::default(); 25];
                let (source_index, destination) = if reverse { (1, 0) } else { (0, 1) };
                overlapped[source_index..source_index + 24].copy_from_slice(&values);
                let before = overlapped.clone();
                let mut source = x;
                source.data = onnx_runtime_ep_api::DevicePtr(overlapped.as_ptr().cast());
                source.byte_offset = source_index * size_of::<T>();
                let mut y = TensorMut::new(
                    onnx_runtime_ep_api::DevicePtrMut(unsafe {
                        overlapped.as_mut_ptr().add(destination).cast()
                    }),
                    T::DTYPE,
                    &shape,
                    &strides,
                    onnx_runtime_ir::DeviceId::cpu(),
                );
                assert_eq!(refined(&k, &source, &y).unwrap().bytes, bytes as u64);
                assert!(
                    k.execute(&[source], std::slice::from_mut(&mut y))
                        .unwrap_err()
                        .to_string()
                        .contains("workspace")
                );
                assert_eq!(bits(&overlapped), bits(&before));
                let mut scratch = vec![0u128; bytes.div_ceil(16)];
                k.execute_with_workspace(
                    &[source],
                    std::slice::from_mut(&mut y),
                    Some(WorkspaceView::new(
                        onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
                        bytes,
                    )),
                )
                .unwrap();
                assert_eq!(
                    bits(&overlapped[destination..destination + 24]),
                    bits(&expected),
                    "shifted {axes:?} reverse={reverse}"
                );
            }
        }
    }
    let values = (0..24)
        .map(|i| (i as f32 - 7.25) * 0.375 + (i % 3) as f32)
        .collect::<Vec<_>>();
    check(values.clone());
    check(values.iter().map(|&v| f64::from(v)).collect());
    check(values.iter().copied().map(half::f16::from_f32).collect());
    check(values.iter().copied().map(half::bf16::from_f32).collect());
}
#[test]
fn epsilon_stays_outside_sqrt_and_f64_never_narrows_through_f32() {
    let x = Owned::f64(&[2], &[16777217., 16777219.]);
    let mut y = Owned::f64(&[2], &[0.; 2]);
    kernel(&[], 3.)
        .execute(&[x.view()], &mut [y.view_mut()])
        .unwrap();
    assert_eq!(y.to_f64(), [-0.25, 0.25]);
    let x = Owned::f32(&[2], &[0., 2.]);
    let mut y = Owned::zeros_f32(&[2]);
    kernel(&[], 3.)
        .execute(&[x.view()], &mut [y.view_mut()])
        .unwrap();
    assert_eq!(y.to_f32(), [-0.25, 0.25]);
}
#[test]
fn negative_strides_strided_output_and_packed_direct_origins_keep_values() {
    let k = kernel(&[0, 2], 0.125);
    let shape = [2, 3, 4];
    let values = (0..24).map(|i| i as f32 * 0.175 - 2.).collect::<Vec<_>>();
    let owner = Owned::f32(&shape, &values);
    let mut x = owner.view();
    x.byte_offset = 12 * 4;
    x.strides = &[-12, 4, 1];
    let reversed = values[12..]
        .iter()
        .chain(values[..12].iter())
        .copied()
        .collect::<Vec<_>>();
    let expected = reference(&reversed, &[0, 2], 0.125);
    let mut padded = [-99f32; 48];
    let mut y = TensorMut::new(
        onnx_runtime_ep_api::DevicePtrMut(padded.as_mut_ptr().cast()),
        DataType::Float32,
        &shape,
        &[24, 8, 2],
        onnx_runtime_ir::DeviceId::cpu(),
    );
    assert_eq!(refined(&k, &x, &y).unwrap().bytes, 96);
    k.execute_prepared(&[x], std::slice::from_mut(&mut y))
        .unwrap();
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(padded[2 * i].to_bits(), want.to_bits());
        assert_eq!(padded[2 * i + 1], -99.);
    }
    let mut packed_x = [0u32; 25];
    let mut packed_y = [0u32; 25];
    let xptr = unsafe { packed_x.as_mut_ptr().cast::<u8>().add(1) };
    let yptr = unsafe { packed_y.as_mut_ptr().cast::<u8>().add(1) };
    for (i, &v) in values.iter().enumerate() {
        unsafe {
            xptr.add(i * 4).cast::<f32>().write_unaligned(v);
        }
    }
    let mut x = owner.view();
    x.data = onnx_runtime_ep_api::DevicePtr(xptr.cast());
    let mut y = TensorMut::new(
        onnx_runtime_ep_api::DevicePtrMut(yptr.cast()),
        DataType::Float32,
        &shape,
        &[12, 4, 1],
        onnx_runtime_ir::DeviceId::cpu(),
    );
    assert_eq!(refined(&k, &x, &y).unwrap().bytes, 0);
    k.execute(&[x], std::slice::from_mut(&mut y)).unwrap();
    for (i, want) in reference(&values, &[0, 2], 0.125).iter().enumerate() {
        assert_eq!(
            unsafe { yptr.add(i * 4).cast::<f32>().read_unaligned() }.to_bits(),
            want.to_bits()
        );
    }
}
#[test]
fn metadata_is_deferred_and_missing_views_keep_the_complete_storage_bound() {
    for dtype in mvn::DTYPES {
        let k = kernel(&[0, 2], 0.125);
        let meta = [TensorMetadata::new(*dtype, &[2, 3, 4], true)];
        let facts = [OutputMetadata::Unresolved { dtype: *dtype }];
        assert!(matches!(
            k.workspace_requirement(&meta, &facts).unwrap(),
            WorkspacePreparation::Deferred
        ));
        assert_eq!(
            k.workspace_requirement_for_execution(&[], &[], &meta, &facts, None)
                .unwrap()
                .require_ready()
                .unwrap()
                .bytes,
            (24 * dtype.byte_size()) as u64
        );
        for bad in [
            vec![],
            vec![OutputMetadata::Absent],
            vec![OutputMetadata::tensor(*dtype, &[24])],
            vec![OutputMetadata::tensor(DataType::Int32, &[2, 3, 4])],
        ] {
            assert!(
                k.workspace_requirement(&meta, &bad)
                    .unwrap_err()
                    .to_string()
                    .contains(mvn::OP)
            );
        }
        for shape in [&[0, 3, 4][..], &[2, 0, 4], &[2, 3, 0]] {
            let x = Owned::zeros(*dtype, shape);
            let mut y = Owned::zeros(*dtype, shape);
            assert_eq!(
                k.workspace_requirement(
                    &[TensorMetadata::new(*dtype, shape, true)],
                    &[OutputMetadata::tensor(*dtype, shape)]
                )
                .unwrap()
                .require_ready()
                .unwrap()
                .bytes,
                0
            );
            k.execute(&[x.view()], &mut [y.view_mut()]).unwrap();
        }
        for shape in [&[0, usize::MAX, 2][..], &[isize::MAX as usize, 2, 1]] {
            assert!(
                k.workspace_requirement(
                    &[TensorMetadata::new(*dtype, shape, true)],
                    &[OutputMetadata::Unresolved { dtype: *dtype }]
                )
                .is_err()
            );
        }
    }
    for axes in [vec![0, 0], vec![3], vec![-4], vec![i64::MIN]] {
        assert!(
            kernel(&axes, 0.125)
                .workspace_requirement(
                    &[TensorMetadata::new(DataType::Float32, &[2, 3, 4], true)],
                    &[OutputMetadata::tensor(DataType::Float32, &[2, 3, 4])]
                )
                .unwrap_err()
                .to_string()
                .contains("axes")
        );
    }
}
#[test]
fn host_view_and_address_refusal_precedes_any_pointer_access_even_when_empty() {
    let k = kernel(&[], 0.125);
    let x = Owned::f32(&[2, 3], &[1.; 6]);
    let mut y = Owned::f32(&[2, 3], &[-99.; 6]);
    for case in 0..8 {
        let mut source = x.view();
        let mut out = y.view_mut();
        match case {
            0 => {
                source.device = onnx_runtime_ir::DeviceId::cuda(0);
                source.data = onnx_runtime_ep_api::DevicePtr(4usize as *const _);
            }
            1 => {
                out.device = onnx_runtime_ir::DeviceId::cuda(0);
                out.data = onnx_runtime_ep_api::DevicePtrMut(4usize as *mut _);
            }
            2 => source.strides = &[],
            3 => out.strides = &[],
            4 => {
                source.data = onnx_runtime_ep_api::DevicePtr((usize::MAX - 3) as *const _);
                source.byte_offset = 8;
            }
            5 => source.data = onnx_runtime_ep_api::DevicePtr((usize::MAX - 3) as *const _),
            6 => out.data = onnx_runtime_ep_api::DevicePtrMut((usize::MAX - 3) as *mut _),
            _ => {
                out.data = onnx_runtime_ep_api::DevicePtrMut((usize::MAX - 3) as *mut _);
                out.byte_offset = 8;
            }
        }
        let error = refined(&k, &source, &out).unwrap_err();
        assert!(error.to_string().contains(mvn::OP), "case {case}: {error}");
        assert_eq!(y.to_f32(), [-99.; 6]);
    }
    let empty = Owned::zeros_f32(&[0]);
    let mut empty_y = Owned::zeros_f32(&[0]);
    let mut source = empty.view();
    source.device = onnx_runtime_ir::DeviceId::cuda(0);
    assert!(
        k.execute(&[source], &mut [empty_y.view_mut()])
            .unwrap_err()
            .to_string()
            .contains("Host")
    );
    let mut out = y.view_mut();
    out.dtype = DataType::Float64;
    let error = k
        .workspace_requirement_for_execution(
            &[x.view()],
            &[out],
            &[TensorMetadata::new(DataType::Float32, &[2, 3], true)],
            &[OutputMetadata::tensor(DataType::Float32, &[2, 3])],
            None,
        )
        .unwrap_err();
    assert!(error.to_string().contains("refresh metadata"));
}
#[test]
fn complete_fallback_binding_is_checked_before_initialization_or_output_writes() {
    let k = kernel(&[-1], 0.125);
    let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let mut y = Owned::f32(&[2, 3], &[-99.; 6]);
    let mut output = [-99f32; 12];
    let mut view = y.view_mut();
    view.data = onnx_runtime_ep_api::DevicePtrMut(output.as_mut_ptr().cast());
    view.strides = &[6, 2];
    let mut backing = [0x7fc01234u32; 8];
    let ptr = unsafe { backing.as_mut_ptr().add(1).cast::<u8>() };
    for ws in [
        None,
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(ptr.cast()),
            23,
        )),
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(unsafe { ptr.add(1).cast() }),
            24,
        )),
    ] {
        assert!(
            k.execute_with_workspace(&[x.view()], std::slice::from_mut(&mut view), ws)
                .unwrap_err()
                .to_string()
                .contains("workspace")
        );
        assert_eq!(output, [-99.; 12]);
        assert_eq!(backing, [0x7fc01234; 8]);
    }
    k.execute_with_workspace(
        &[x.view()],
        std::slice::from_mut(&mut view),
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(ptr.cast()),
            24,
        )),
    )
    .unwrap();
    assert_eq!((backing[0], backing[7]), (0x7fc01234, 0x7fc01234));
    let expected = output;
    let mut fresh = [std::mem::MaybeUninit::<f32>::uninit(); 6];
    k.execute_with_workspace(
        &[x.view()],
        std::slice::from_mut(&mut view),
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(fresh.as_mut_ptr().cast()),
            24,
        )),
    )
    .unwrap();
    assert_eq!(output, expected);
    for row in output.chunks_exact(6) {
        assert!(row[0] < 0. && row[2].abs() < 1e-6 && row[4] > 0.);
        assert_eq!((row[1], row[3], row[5]), (-99., -99., -99.));
    }
}

#[test]
fn direct_exact_alias_uses_actual_origins_instead_of_allocation_bases() {
    let k = kernel(&[], 0.125);
    let owner = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let mut data = [-99f32, 1., 2., 3., 4., 5., 6., -99.];
    let mut x = owner.view();
    x.data = onnx_runtime_ep_api::DevicePtr(data.as_ptr().cast());
    x.byte_offset = 4;
    let mut y = TensorMut::new(
        onnx_runtime_ep_api::DevicePtrMut(unsafe { data.as_mut_ptr().add(1).cast() }),
        DataType::Float32,
        &[2, 3],
        &[3, 1],
        onnx_runtime_ir::DeviceId::cpu(),
    );
    assert_eq!(refined(&k, &x, &y).unwrap().bytes, 0);
    k.execute(&[x], std::slice::from_mut(&mut y)).unwrap();
    assert_eq!((data[0], data[7]), (-99., -99.));
    assert!(data[1] < 0. && data[6] > 0.);
    let mut shifted = x;
    shifted.byte_offset = 0;
    assert_eq!(refined(&k, &shifted, &y).unwrap().bytes, 24);
}

#[test]
fn governed_snapshot_reuses_one_peak_and_direct_calls_make_no_warm_issuer_callbacks() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let books = Arc::new(RecordingGovernor::new_host(32));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let k = kernel(&[-1], 0.);
    let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 4., 6., 8.]);
    let mut y = Owned::zeros_f32(&[2, 4]);
    let mut direct_y = Owned::zeros_f32(&[2, 4]);
    let mut padded = [0f32; 16];
    let mut output = y.view_mut();
    output.data = onnx_runtime_ep_api::DevicePtrMut(padded.as_mut_ptr().cast());
    output.strides = &[8, 2];
    let request = refined(&k, &x.view(), &output).unwrap();
    assert_eq!(request.bytes, 32);
    let first = provider
        .allocate_workspace(32, request.alignment, request.role())
        .unwrap();
    let identity = first.managed_owner().unwrap().identity();
    assert!(
        provider
            .allocate_workspace(32, request.alignment, request.role())
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
    for _ in 0..4096 {
        assert_eq!(
            refined(&k, &x.view(), &direct_y.view_mut()).unwrap().bytes,
            0
        );
        k.execute(&[x.view()], &mut [direct_y.view_mut()]).unwrap();
        let request = refined(&k, &x.view(), &output).unwrap();
        let mut allocation = provider
            .allocate_workspace(request.bytes as usize, request.alignment, request.role())
            .unwrap();
        assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
        k.execute_with_workspace(
            &[x.view()],
            std::slice::from_mut(&mut output),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            )),
        )
        .unwrap();
        provider.deallocate_workspace(allocation).unwrap();
    }
    assert_eq!(
        (books.claims(), books.releases()),
        calls,
        "including zero-byte callbacks"
    );
    assert_eq!(books.used(Tier::Host), 32);
    for row in padded.chunks_exact(8) {
        let values = [row[0], row[2], row[4], row[6]];
        assert!(values.iter().sum::<f32>().abs() < 1e-6);
        assert!((values.iter().map(|x| x * x).sum::<f32>() / 4. - 1.).abs() < 1e-6);
    }
    pool.trim_idle(32).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}
