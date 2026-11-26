use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel};
use onnx_runtime_ep_api::{DevicePtrMut, WorkspaceRequirement};
use onnx_runtime_ir::{DataType, DeviceId, compute_contiguous_strides};
fn kernel() -> Affine {
    Affine(AffineGrouped {
        depth: 4,
        block: 2,
        zero: 7,
    })
}
fn fixture() -> [Owned; 5] {
    [
        Owned::f32(&[2, 4], &[1., -2., 3., 4., 0.5, 2., -1., 3.]),
        Owned::u8(
            &[3, 3, 2],
            &(0..18).map(|i| (i * 19 + 3) as u8).collect::<Vec<_>>(),
        ),
        Owned::i64(&[2, 2], &[2, 0, 1, 2]),
        Owned::f32(
            &[3, 3, 2],
            &(0..18).map(|i| (i + 1) as f32 * 0.125).collect::<Vec<_>>(),
        ),
        Owned::u8(&[3, 3, 1], &[0x77; 9]),
    ]
}
fn requirement(k: &Affine, inputs: &[TensorView], out: &TensorMut) -> WorkspaceRequirement {
    let meta = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect::<Vec<_>>();
    k.workspace_requirement(&meta, &[OutputMetadata::tensor(out.dtype, out.shape)])
        .unwrap()
        .require_ready()
        .unwrap()
}
fn bytes(floats: usize, routes: usize) -> usize {
    (floats * 4).next_multiple_of(align_of::<usize>()) + routes * size_of::<usize>()
}
fn reference(inputs: &[TensorView]) -> Vec<f32> {
    let k = kernel();
    let mut y = Owned::zeros_f32(&[2, 2, 3]);
    let l = k.0.layout(inputs, &[y.view_mut()]).unwrap();
    let routes =
        super::super::route_indices(Op::GroupedMatMul { transposed: true }, &inputs[2], l.groups)
            .unwrap();
    let bank = AffineBank::prepare(
        &inputs[1],
        &inputs[3],
        inputs.get(4),
        [l.groups, l.n, l.k],
        2,
        7,
    )
    .and_then(|bank| bank.validate())
    .unwrap();
    let x = crate::kernels::to_dense_f32(&inputs[0]).unwrap();
    let mut result = vec![0f32; l.lanes];
    for (route, row) in result.chunks_exact_mut(l.n).enumerate() {
        let token = route / l.selections;
        for (col, value) in row.iter_mut().enumerate() {
            *value = bank.dot(
                routes[route] * l.n + col,
                &x[token * l.k..(token + 1) * l.k],
            );
        }
    }
    result
}
fn bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}
struct Repacked {
    backing: Vec<u128>,
    shape: Vec<usize>,
    strides: Vec<i64>,
    base: usize,
    offset: usize,
    dtype: DataType,
}
impl Repacked {
    fn new(source: &Owned, mode: usize) -> Self {
        let shape = source.shape.clone();
        let width = source.dtype.byte_size();
        let mut strides = compute_contiguous_strides(&shape);
        if matches!(mode, 1..=3 | 5) {
            let mut step = if mode == 2 { 2 } else { 1 };
            for (axis, &dim) in shape.iter().enumerate().rev() {
                strides[axis] = step;
                step = step * dim as i64 + 3;
            }
        }
        let extent = shape
            .iter()
            .zip(&strides)
            .map(|(&d, &s)| (d - 1) * s as usize)
            .sum::<usize>();
        let offset = if mode == 3 {
            extent * width
        } else if mode == 5 {
            (shape[0] - 1) * strides[0] as usize * width
        } else {
            0
        };
        if mode == 3 {
            strides.iter_mut().for_each(|s| *s = -*s);
        }
        if mode == 5 {
            strides[0] = -strides[0];
        }
        let base = usize::from(mode == 4);
        let len = (extent + 1) * width + base;
        let mut backing = vec![0xccccccccccccccccccccccccccccccccu128; len.div_ceil(16)];
        let storage = unsafe {
            std::slice::from_raw_parts_mut(backing.as_mut_ptr().cast::<u8>(), backing.len() * 16)
        };
        for (logical, value) in source.bytes.chunks_exact(width).enumerate() {
            let mut n = logical;
            let mut at = (offset + base) as i64;
            for (axis, &dim) in shape.iter().enumerate().rev() {
                let coordinate = n % dim;
                n /= dim;
                at += coordinate as i64 * strides[axis] * width as i64;
            }
            let at = at as usize;
            storage[at..at + width].copy_from_slice(value);
        }
        Self {
            backing,
            shape,
            strides,
            base,
            offset,
            dtype: source.dtype,
        }
    }
    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(unsafe { self.backing.as_ptr().cast::<u8>().add(self.base).cast() }),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
        .with_byte_offset(self.offset)
    }
}
#[test]
fn whole_workspace_preserves_route_order_for_every_source_layout_and_optional_zero() {
    let k = kernel();
    for int32 in [false, true] {
        for optional in [false, true] {
            let mut f = fixture();
            if int32 {
                f[2] = Owned::i32(&[2, 2], &[2, 0, 1, 2]);
            }
            let slots = if optional { 4 } else { 5 };
            let expected = reference(&f.iter().take(slots).map(Owned::view).collect::<Vec<_>>());
            for mode in 0..6 {
                let storage = f.iter().map(|v| Repacked::new(v, mode)).collect::<Vec<_>>();
                let inputs = storage
                    .iter()
                    .take(slots)
                    .map(Repacked::view)
                    .collect::<Vec<_>>();
                let rows = Rows::new(OP, "X", &inputs[0]).unwrap();
                assert_eq!(
                    rows.scratch_len(),
                    if matches!(mode, 0 | 1 | 5) { 0 } else { 4 }
                );
                let mut y = Owned::zeros_f32(&[2, 2, 3]);
                assert_eq!(
                    requirement(&k, &inputs, &y.view_mut()).bytes,
                    bytes(16, 4) as u64
                );
                for i in 0..slots {
                    assert!(k.supports_strided_input(i));
                }
                k.execute_prepared(&inputs, &mut [y.view_mut()]).unwrap();
                assert_eq!(
                    bits(y.to_f32()),
                    bits(expected.clone()),
                    "mode={mode},int32={int32},optional={optional}"
                );
            }
        }
    }
}
#[test]
fn short_or_misaligned_workspace_refuses_before_source_reads_and_initialization() {
    let f = fixture();
    let k = kernel();
    let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut out = Owned::f32(&[2, 2, 3], &[123.; 12]);
    inputs[2].data = DevicePtr(8usize as *const std::ffi::c_void);
    inputs[3].data = DevicePtr(4usize as *const std::ffi::c_void);
    let req = requirement(&k, &inputs, &out.view_mut());
    let mut memory = vec![0xaaaaaaaaaaaaaaaau128; (req.bytes as usize).div_ceil(16) + 1];
    let pointer = memory.as_mut_ptr().cast::<u8>();
    for lane in [
        None,
        Some(WorkspaceView::new(
            DevicePtrMut(pointer.cast()),
            req.bytes as usize - 1,
        )),
        Some(WorkspaceView::new(
            DevicePtrMut(unsafe { pointer.add(1).cast() }),
            req.bytes as usize,
        )),
    ] {
        assert!(
            k.execute_with_workspace(&inputs, &mut [out.view_mut()], lane)
                .is_err()
        );
        assert_eq!(out.to_f32(), [123.; 12]);
        assert!(memory.iter().all(|&v| v == 0xaaaaaaaaaaaaaaaau128));
    }
}
#[test]
fn staging_protects_every_source_from_shifted_y_and_respects_output_strides() {
    let k = kernel();
    let f = fixture();
    let expected = reference(&f.iter().map(Owned::view).collect::<Vec<_>>());
    for source in 0..5 {
        for shift in [0usize, 8, 16] {
            let mut backing = vec![
                0xccccccccccccccccccccccccccccccccu128;
                (f[source].bytes.len() + 160).div_ceil(16)
            ];
            let pointer = backing.as_mut_ptr().cast::<u8>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f[source].bytes.as_ptr(),
                    pointer.add(16),
                    f[source].bytes.len(),
                );
            }
            let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
            inputs[source].data = DevicePtr(unsafe { pointer.add(16).cast() });
            let mut y = TensorMut::new(
                DevicePtrMut(unsafe { pointer.add(8 + shift).cast() }),
                DataType::Float32,
                &[2, 2, 3],
                &[6, 3, 1],
                DeviceId::cpu(),
            );
            k.execute_prepared(&inputs, std::slice::from_mut(&mut y))
                .unwrap();
            let actual = (0..12)
                .map(|i| unsafe {
                    pointer
                        .add(8 + shift + i * 4)
                        .cast::<f32>()
                        .read_unaligned()
                })
                .collect();
            assert_eq!(
                bits(actual),
                bits(expected.clone()),
                "source={source},shift={shift}"
            );
        }
    }
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut storage = [123f32; 64];
    let mut output = TensorMut::new(
        DevicePtrMut(storage.as_mut_ptr().cast()),
        DataType::Float32,
        &[2, 2, 3],
        &[31, 13, -2],
        DeviceId::cpu(),
    )
    .with_byte_offset(16);
    k.execute_prepared(&inputs, std::slice::from_mut(&mut output))
        .unwrap();
    let offsets = [4, 2, 0, 17, 15, 13, 35, 33, 31, 48, 46, 44];
    for (i, &offset) in offsets.iter().enumerate() {
        assert_eq!(storage[offset].to_bits(), expected[i].to_bits());
    }
    for (i, &value) in storage.iter().enumerate() {
        if !offsets.contains(&i) {
            assert_eq!(value, 123.);
        }
    }
}
#[test]
fn invalid_routes_or_scales_report_context_without_output_publication() {
    let k = kernel();
    for slot in [2, 3] {
        let mut f = fixture();
        if slot == 2 {
            f[2] = Owned::i64(&[2, 2], &[2, 0, -1, 2]);
        } else {
            f[2] = Owned::i64(&[2, 2], &[2, 2, 2, 2]);
            let mut values = f[3].to_f32();
            values[0] = 0.;
            f[3] = Owned::f32(&[3, 3, 2], &values);
        }
        let mut out = Owned::f32(&[2, 2, 3], &[123.; 12]);
        let text = k
            .execute_prepared(
                &f.iter().map(Owned::view).collect::<Vec<_>>(),
                &mut [out.view_mut()],
            )
            .unwrap_err()
            .to_string();
        assert!(
            text.contains(if slot == 2 {
                "group_indices[2] is -1"
            } else {
                "positive and finite"
            }),
            "{text}"
        );
        assert_eq!(out.to_f32(), [123.; 12]);
    }
}
#[test]
fn empty_affine_still_validates_routes_and_scales_without_accessing_x_or_y() {
    let k = kernel();
    let f = fixture();
    let empty = Owned::i64(&[2, 0], &[]);
    let mut out = Owned::zeros_f32(&[2, 0, 3]);
    let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    inputs[2] = empty.view();
    inputs[0].device = DeviceId::cuda(0);
    inputs[0].data = DevicePtr(4usize as *const std::ffi::c_void);
    let mut y = out.view_mut();
    y.device = DeviceId::cuda(0);
    assert_eq!(requirement(&k, &inputs, &y).bytes, 0);
    k.execute(&inputs, std::slice::from_mut(&mut y)).unwrap();
    let scales = Owned::f32(&[3, 3, 2], &[0.; 18]);
    inputs[3] = scales.view();
    assert!(
        k.execute(&inputs, &mut [out.view_mut()])
            .unwrap_err()
            .to_string()
            .contains("positive and finite")
    );
    let weights = Owned::u8(&[3, 0, 2], &[]);
    let scales = Owned::f32(&[3, 0, 2], &[]);
    let zeros = Owned::u8(&[3, 0, 1], &[]);
    let routes = Owned::i64(&[2, 2], &[0, 1, 3, 0]);
    let inputs = [
        f[0].view(),
        weights.view(),
        routes.view(),
        scales.view(),
        zeros.view(),
    ];
    let mut out = Owned::zeros_f32(&[2, 2, 0]);
    assert!(
        k.execute_prepared(&inputs, &mut [out.view_mut()])
            .unwrap_err()
            .to_string()
            .contains("group_indices[2] is 3")
    );
}
#[test]
fn metadata_handles_unresolved_output_and_invalid_formats_without_panics() {
    let k = kernel();
    let f = fixture();
    let mut meta = f
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, &v.shape, true))
        .collect::<Vec<_>>();
    let facts = [OutputMetadata::Unresolved {
        dtype: DataType::Float32,
    }];
    assert_eq!(
        k.workspace_requirement(&meta, &facts)
            .unwrap()
            .require_ready()
            .unwrap()
            .bytes,
        bytes(16, 4) as u64
    );
    for config in [
        AffineGrouped {
            depth: 4,
            block: 0,
            zero: 7,
        },
        AffineGrouped {
            depth: 0,
            block: 2,
            zero: 7,
        },
        AffineGrouped {
            depth: 4,
            block: 2,
            zero: 16,
        },
    ] {
        assert!(Affine(config).workspace_requirement(&meta, &facts).is_err());
    }
    meta[1].shape = &[usize::MAX, 3, 2];
    assert!(k.workspace_requirement(&meta, &facts).is_err());
}

#[test]
fn one_retained_affine_workspace_has_no_warm_issuer_callbacks_and_trims_exactly() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let expected = bytes(16, 4);
    let books = Arc::new(RecordingGovernor::new_host(expected as u64));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let k = kernel();
    let f = fixture();
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut y = Owned::zeros_f32(&[2, 2, 3]);
    let request = requirement(&k, &inputs, &y.view_mut());
    assert_eq!(request.bytes, expected as u64);
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
    for _ in 0..4096 {
        let request = requirement(&k, &inputs, &y.view_mut());
        assert_eq!(request.bytes, expected as u64);
        let mut allocation = provider
            .allocate_workspace(expected, request.alignment, request.role())
            .unwrap();
        assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
        let workspace = WorkspaceView::new(DevicePtrMut(allocation.as_mut_ptr()), allocation.len());
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
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
fn odd_depth_and_block_counts_preserve_scalar_accumulation_and_nonfinite_x_bits() {
    let k = Affine(AffineGrouped {
        depth: 3,
        block: 1,
        zero: 7,
    });
    let values = [
        1.,
        -2.,
        3.,
        f32::NAN,
        0.,
        1.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.,
    ];
    let routes = [0i64, 1, 0];
    let codes = (0..20).map(|i| (i * 19 + 3) as u8).collect::<Vec<_>>();
    let scales = (0..30).map(|i| (i + 1) as f32 * 0.125).collect::<Vec<_>>();
    let x = Owned::f32(&[3, 3], &values);
    let w = Owned::u8(&[2, 5, 2], &codes);
    let r = Owned::i64(&[3, 1], &routes);
    let s = Owned::f32(&[2, 5, 3], &scales);
    let mut out = Owned::zeros_f32(&[3, 1, 5]);
    k.execute_prepared(
        &[x.view(), w.view(), r.view(), s.view()],
        &mut [out.view_mut()],
    )
    .unwrap();
    let mut expected = Vec::new();
    for token in 0..3 {
        for column in 0..5 {
            let row = routes[token] as usize * 5 + column;
            let mut sum = 0f32;
            for j in 0..3 {
                let q = (codes[row * 2 + j / 2] >> ((j % 2) * 4)) & 15;
                let weight = f32::from(i16::from(q) - 7) * scales[row * 3 + j];
                sum += values[token * 3 + j] * weight;
            }
            expected.push(sum);
        }
    }
    assert_eq!(bits(out.to_f32()), bits(expected));
}
