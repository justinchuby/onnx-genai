use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel};
use onnx_runtime_ep_api::{DevicePtrMut, WorkspaceRequirement};
use onnx_runtime_ir::DeviceId;

fn workspace_bytes(floats: usize, experts: usize) -> usize {
    (floats * size_of::<f32>()).next_multiple_of(align_of::<usize>()) + experts * size_of::<usize>()
}

fn kernel(k: usize) -> AffineMoe {
    AffineMoe {
        k,
        alpha: 1.3,
        beta: 0.2,
        limit: 7.,
        first: BankFormat { block: 2, zero: 7 },
        down: BankFormat { block: 2, zero: 7 },
    }
}
fn fixture() -> [Owned; 8] {
    [
        Owned::f32(&[2, 4], &[1., -2., 3., -4., 2., 0., -1., 3.]),
        Owned::f32(&[2, 3], &[1., 1., 1., 0., -20., 2.]),
        Owned::u8(
            &[3, 8, 2],
            &(0..48).map(|n| (n * 17 + 3) as u8).collect::<Vec<_>>(),
        ),
        Owned::f32(&[3, 8, 2], &[0.125; 48]),
        Owned::u8(&[3, 8, 1], &[0x77; 24]),
        Owned::u8(
            &[3, 4, 2],
            &(0..24).map(|n| (n * 19 + 9) as u8).collect::<Vec<_>>(),
        ),
        Owned::f32(&[3, 4, 2], &[0.25; 24]),
        Owned::u8(&[3, 4, 1], &[0x77; 12]),
    ]
}
fn requirement(
    k: &AffineMoe,
    inputs: &[TensorView],
    outputs: &[TensorMut],
) -> WorkspaceRequirement {
    let metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect::<Vec<_>>();
    let facts = outputs
        .iter()
        .map(OutputMetadata::from_view)
        .collect::<Vec<_>>();
    assert!(matches!(
        k.workspace_requirement(&metadata, &facts).unwrap(),
        WorkspacePreparation::Deferred
    ));
    k.workspace_requirement_for_execution(
        inputs,
        outputs,
        &metadata,
        &facts,
        Some(WorkspacePreparation::Ready(WorkspaceRequirement::NONE)),
    )
    .unwrap()
    .require_ready()
    .unwrap()
}
fn reference(kernel: &AffineMoe, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
    let [x, router, w1, s1, z1, w2, s2, z2] = inputs else {
        return Err(error(
            "expected activation, router and two packed expert banks (each with an optional zero-point slot)",
        ));
    };
    let [out] = outputs else {
        return Err(error("expected one FLOAT output"));
    };
    let &[tokens, experts] = router.shape else {
        return Err(error("router must be [tokens, experts]"));
    };
    let &[_, _, packed_intermediate] = w2.shape else {
        return Err(error(
            "second packed bank must be [experts, hidden, packed_intermediate]",
        ));
    };
    let intermediate = packed_intermediate
        .checked_mul(2)
        .ok_or_else(|| error("intermediate width overflows"))?;
    let inter2 = intermediate
        .checked_mul(2)
        .ok_or_else(|| error("interleaved width overflows"))?;
    let Some((&hidden, leading)) = x.shape.split_last() else {
        return Err(error("input has no hidden axis"));
    };
    if !(1..=2).contains(&leading.len())
        || hidden == 0
        || tokens == 0
        || intermediate == 0
        || leading.iter().try_fold(1usize, |a, b| a.checked_mul(*b)) != Some(tokens)
        || kernel.k > experts
        || x.shape != out.shape
        || [x.dtype, router.dtype, out.dtype]
            .iter()
            .any(|&d| d != DataType::Float32)
    {
        return Err(error(format!(
            "expected FLOAT input and output with matching token geometry; input={:?}, router={:?}, output={:?}, k={}",
            x.shape, router.shape, out.shape, kernel.k
        )));
    }
    let first = AffineBank::new(
        w1,
        s1,
        (!z1.is_absent()).then_some(z1),
        [experts, inter2, hidden],
        kernel.first.block,
        kernel.first.zero,
    )
    .map_err(|cause| error(format!("first expert bank (inputs 2..4): {cause}")))?;
    let down = AffineBank::new(
        w2,
        s2,
        (!z2.is_absent()).then_some(z2),
        [experts, hidden, intermediate],
        kernel.down.block,
        kernel.down.zero,
    )
    .map_err(|cause| error(format!("down expert bank (inputs 5..7): {cause}")))?;
    let x = crate::kernels::to_dense_f32(x)?;
    let scores = crate::kernels::to_dense_f32(router)?;
    let mut y = vec![0.0; x.len()];
    let mut indices: Vec<_> = (0..experts).collect();
    let mut selected_logits = vec![0.0; kernel.k];
    let mut probabilities = vec![0.0; kernel.k];
    let mut activated = vec![0.0; intermediate];
    for ((token, logits), output) in x
        .chunks_exact(hidden)
        .zip(scores.chunks_exact(experts))
        .zip(y.chunks_exact_mut(hidden))
    {
        // TopK on the reversed expert axis picks larger original IDs on ties.
        let order = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| b.cmp(a));
        if kernel.k < experts {
            indices.select_nth_unstable_by(kernel.k, order);
        }
        let selected = &mut indices[..kernel.k];
        selected.sort_unstable_by(order);
        for (p, &id) in selected_logits.iter_mut().zip(selected.iter()) {
            *p = logits[id];
        }
        crate::kernels::softmax::softmax_rows(&selected_logits, &mut probabilities, 1, kernel.k);
        for (&expert, &p) in selected.iter().zip(&probabilities) {
            if p.partial_cmp(&THRESHOLD) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            for (col, act) in activated.iter_mut().enumerate() {
                let row = expert * inter2 + col * 2;
                let gate = first.dot(row, token);
                // ONNX Min propagates NaN, unlike f32::min.
                let gate = if gate > kernel.limit {
                    kernel.limit
                } else {
                    gate
                };
                let up = first.dot(row + 1, token).clamp(-kernel.limit, kernel.limit) + kernel.beta;
                let z = kernel.alpha * gate;
                let exp = (-z.abs()).exp();
                let sigmoid = if z >= 0.0 {
                    1.0 / (1.0 + exp)
                } else {
                    exp / (1.0 + exp)
                };
                *act = (gate * sigmoid) * up;
            }
            for (col, sum) in output.iter_mut().enumerate() {
                *sum += down.dot(expert * hidden + col, &activated) * p;
            }
        }
    }
    crate::kernels::write_dense_f32(out, &y)
}
fn bits(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}

#[test]
fn affine_moe_whole_workspace_preserves_routing_bits_and_strided_rows() {
    for k in [1, 2, 3] {
        for logits in [
            [1., 1., 1., 0., -20., 2.],
            [f32::NAN, f32::INFINITY, 0., f32::NEG_INFINITY, 1., 2.],
            [0., -18., -19., 0., -18.5, -17.5],
        ] {
            let mut f = fixture();
            f[1] = Owned::f32(&[2, 3], &logits);
            let mut expected = Owned::zeros_f32(&[2, 4]);
            let dense = f.iter().map(Owned::view).collect::<Vec<_>>();
            reference(&kernel(k), &dense, &mut [expected.view_mut()]).unwrap();
            for mode in 0..5 {
                let mut payloads = Vec::new();
                for source in &f[..2] {
                    let width = source.shape[1];
                    let values = source.to_f32();
                    let (stride, base, size) = match mode {
                        0 => (vec![width as i64, 1], 0, values.len() * 4),
                        1 => (vec![(width * 2) as i64, 1], 0, values.len() * 8),
                        2 => (vec![(width * 2) as i64, 2], 0, values.len() * 8),
                        3 => (vec![width as i64, -1], (width - 1) * 4, values.len() * 4),
                        _ => (vec![width as i64, 1], 1, values.len() * 4 + 1),
                    };
                    let mut backing =
                        vec![0xccccccccccccccccccccccccccccccccu128; size.div_ceil(16)];
                    // Known alignment makes the one-byte origin shift a real
                    // unaligned fixture on every allocator.
                    let bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            backing.as_mut_ptr().cast::<u8>(),
                            backing.len() * 16,
                        )
                    };
                    for row in 0..2 {
                        for col in 0..width {
                            let offset = (base as i64
                                + (row as i64 * stride[0] + col as i64 * stride[1]) * 4)
                                as usize;
                            bytes[offset..offset + 4]
                                .copy_from_slice(&values[row * width + col].to_ne_bytes());
                        }
                    }
                    payloads.push((backing, stride, base));
                }
                let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
                for index in 0..2 {
                    inputs[index] = TensorView::new(
                        DevicePtr(unsafe {
                            payloads[index]
                                .0
                                .as_ptr()
                                .cast::<u8>()
                                .add(usize::from(mode == 4))
                                .cast()
                        }),
                        DataType::Float32,
                        &f[index].shape,
                        &payloads[index].1,
                        DeviceId::cpu(),
                    )
                    .with_byte_offset(if mode == 4 {
                        0
                    } else {
                        payloads[index].2
                    });
                }
                if mode == 1 {
                    inputs[4] = TensorView::absent(DataType::Uint8);
                }
                if mode == 2 {
                    inputs[7] = TensorView::absent(DataType::Uint8);
                }
                let mut y = Owned::zeros_f32(&[2, 4]);
                let req = requirement(&kernel(k), &inputs, &[y.view_mut()]);
                let floats = 8 + 2 * k + 4 + if mode >= 2 { 7 } else { 0 };
                assert_eq!(
                    req.bytes,
                    ((floats * 4).next_multiple_of(align_of::<usize>()) + 3 * size_of::<usize>())
                        as u64
                );
                kernel(k)
                    .execute_prepared(&inputs, &mut [y.view_mut()])
                    .unwrap();
                assert_eq!(
                    bits(y.to_f32()),
                    bits(expected.to_f32()),
                    "mode={mode},k={k}"
                );
            }
        }
    }
}

#[test]
fn affine_moe_refuses_before_data_reads_or_partial_workspace_initialization() {
    let f = fixture();
    let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    // Non-null, aligned Host pointer with valid metadata; it must never be read
    // on workspace refusal or during execution refinement.
    inputs[3].data = DevicePtr(4usize as *const std::ffi::c_void);
    let mut out = Owned::f32(&[2, 4], &[123.; 8]);
    let k = kernel(2);
    let req = requirement(&k, &inputs, &[out.view_mut()]);
    let mut backing = vec![0xaaaaaaaaaaaaaaaau128; (req.bytes as usize).div_ceil(16) + 1];
    let ptr = backing.as_mut_ptr().cast::<u8>();
    for view in [
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
            k.execute_with_workspace(&inputs, &mut [out.view_mut()], view)
                .is_err()
        );
        assert_eq!(out.to_f32(), [123.; 8]);
        assert!(backing.iter().all(|&v| v == 0xaaaaaaaaaaaaaaaau128));
    }
    inputs[7].device = DeviceId::cuda(0);
    assert!(requirement_error(&k, &inputs, &mut out).contains("Host"));
}
fn requirement_error(k: &AffineMoe, inputs: &[TensorView], out: &mut Owned) -> String {
    let metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect::<Vec<_>>();
    k.workspace_requirement_for_execution(
        inputs,
        &[out.view_mut()],
        &metadata,
        &[OutputMetadata::tensor(DataType::Float32, &[2, 4])],
        None,
    )
    .unwrap_err()
    .to_string()
}

#[test]
fn affine_moe_stages_output_until_all_aliased_sources_are_read() {
    let f = fixture();
    let k = kernel(2);
    let mut expected = Owned::zeros_f32(&[2, 4]);
    reference(
        &k,
        &f.iter().map(Owned::view).collect::<Vec<_>>(),
        &mut [expected.view_mut()],
    )
    .unwrap();
    // Alias X, router and both scale banks with Y, including shifted overlaps.
    for source in 0..8 {
        for shift in [0, 4, 8] {
            let mut bytes = vec![0xcc; f[source].bytes.len() + 64];
            bytes[16..16 + f[source].bytes.len()].copy_from_slice(&f[source].bytes);
            let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
            inputs[source].data = DevicePtr(unsafe { bytes.as_ptr().add(16).cast() });
            let mut y = TensorMut::new(
                DevicePtrMut(unsafe { bytes.as_mut_ptr().add(8 + shift).cast() }),
                DataType::Float32,
                &[2, 4],
                &[4, 1],
                DeviceId::cpu(),
            );
            k.execute_prepared(&inputs, std::slice::from_mut(&mut y))
                .unwrap();
            let actual = (0..8)
                .map(|i| {
                    f32::from_ne_bytes(
                        bytes[8 + shift + i * 4..12 + shift + i * 4]
                            .try_into()
                            .unwrap(),
                    )
                })
                .collect();
            assert_eq!(
                bits(actual),
                bits(expected.to_f32()),
                "source={source},shift={shift}"
            );
        }
    }
}

#[test]
fn one_retained_moe_workspace_has_no_warm_issuer_callbacks_and_trims_exactly() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let expected = workspace_bytes(16, 3);
    let books = Arc::new(RecordingGovernor::new_host(expected as u64));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let k = kernel(2);
    let f = fixture();
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut y = Owned::zeros_f32(&[2, 4]);
    let request = requirement(&k, &inputs, &[y.view_mut()]);
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
        let request = requirement(&k, &inputs, &[y.view_mut()]);
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
fn row_borrowing_depends_on_inner_axis_not_whole_tensor_contiguity() {
    let values = (0..16).map(|n| n as f32).collect::<Vec<_>>();
    let source = Owned::f32(&[2, 2, 4], &values);
    let view = TensorView::new(
        source.view().data,
        DataType::Float32,
        &[2, 2, 4],
        &[4, 8, 1],
        DeviceId::cpu(),
    );
    let rows = Rows::new("X", &view).unwrap();
    assert_eq!(rows.scratch_len(), 0);
    for (row, start) in [0, 8, 4, 12].into_iter().enumerate() {
        assert_eq!(rows.get(row, &mut []).unwrap(), &values[start..start + 4]);
    }
    let reversed = TensorView::new(
        source.view().data,
        DataType::Float32,
        &[2, 2, 4],
        &[-4, -8, 1],
        DeviceId::cpu(),
    )
    .with_byte_offset(48);
    let rows = Rows::new("X", &reversed).unwrap();
    assert_eq!(rows.scratch_len(), 0);
    for (row, start) in [12, 4, 8, 0].into_iter().enumerate() {
        assert_eq!(rows.get(row, &mut []).unwrap(), &values[start..start + 4]);
    }
}

#[test]
fn missing_actuals_keep_complete_conservative_budget_and_stale_metadata_fails() {
    let f = fixture();
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let k = kernel(2);
    let mut metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
        .collect::<Vec<_>>();
    let facts = [OutputMetadata::Unresolved {
        dtype: DataType::Float32,
    }];
    let req = k
        .workspace_requirement_for_execution(
            &[],
            &[],
            &metadata,
            &facts,
            Some(WorkspacePreparation::Ready(WorkspaceRequirement::NONE)),
        )
        .unwrap()
        .require_ready()
        .unwrap();
    assert_eq!(req.bytes, workspace_bytes(23, 3) as u64);
    let mut out = Owned::zeros_f32(&[2, 4]);
    metadata[1].dtype = DataType::Float64;
    assert!(
        k.workspace_requirement_for_execution(&inputs, &[out.view_mut()], &metadata, &facts, None)
            .is_err()
    );
    metadata[1].dtype = DataType::Float32;
    metadata[0].shape = &[0, 4];
    assert!(
        k.workspace_requirement(&metadata, &facts)
            .unwrap_err()
            .to_string()
            .contains("geometry")
    );
}

#[test]
fn affine_moe_publishes_only_logical_output_elements_for_reversed_padded_y() {
    let f = fixture();
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let k = kernel(2);
    let mut expected = Owned::zeros_f32(&[2, 4]);
    reference(&k, &inputs, &mut [expected.view_mut()]).unwrap();
    let mut storage = [123f32; 24];
    let shape = [2, 4];
    let strides = [9, -2];
    let mut output = TensorMut::new(
        DevicePtrMut(storage.as_mut_ptr().cast()),
        DataType::Float32,
        &shape,
        &strides,
        DeviceId::cpu(),
    )
    .with_byte_offset(24);
    k.execute_prepared(&inputs, std::slice::from_mut(&mut output))
        .unwrap();
    let offsets = [6usize, 4, 2, 0, 15, 13, 11, 9];
    for (n, &offset) in offsets.iter().enumerate() {
        assert_eq!(storage[offset].to_bits(), expected.to_f32()[n].to_bits());
    }
    for (n, &value) in storage.iter().enumerate() {
        if !offsets.contains(&n) {
            assert_eq!(value, 123.);
        }
    }
}
