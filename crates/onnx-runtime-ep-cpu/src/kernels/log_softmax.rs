//! Numerically stable `LogSoftmax` with ONNX-versioned axis semantics.

use crate::dtype::{
    can_borrow_dense, to_dense_f32_widen, to_dense_f32_widen_into, validate_f32_widen_input,
    write_dense_f32_narrow,
};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::Node;

use super::check_arity;
use super::softmax_geometry::SoftmaxGeometry;

/// `LogSoftmax` kernel carrying its axis and whether it uses the pre-opset-13
/// coerce-to-2D definition.
pub struct LogSoftmaxKernel {
    axis: i64,
    coerce_2d: bool,
}

/// Factory for opset ≥13, where the default axis is -1 and normalization is
/// performed across the one selected axis.
pub struct LogSoftmaxFactory;

/// Factory for opset ≤12, where the default axis is 1 and the trailing axes are
/// flattened into a single row dimension.
pub struct LogSoftmaxLegacyFactory;

impl KernelFactory for LogSoftmaxFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LogSoftmaxKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            coerce_2d: false,
        }))
    }
}

impl KernelFactory for LogSoftmaxLegacyFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LogSoftmaxKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1),
            coerce_2d: true,
        }))
    }
}

/// Write `x - logsumexp(x)` over strided reduction slices. Subtracting the
/// slice maximum before exponentiation makes the logsumexp calculation stable.
// `read` selects either an immutable borrowed source or the current staging
// lane. Every element belongs to exactly one (outer, inner) group; the max and
// sum finish before that group's first write, so in-place reads never observe
// another group's normalized outputs. Static closures keep both routes in one
// arithmetic implementation without constructing aliased shared/mutable slices.
fn log_softmax_slices(
    out: &mut [f32],
    read: impl Fn(&[f32], usize) -> f32,
    outer: usize,
    axis_dim: usize,
    inner: usize,
) {
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_dim * inner + i;
            let mut max = f32::NEG_INFINITY;
            for a in 0..axis_dim {
                max = max.max(read(out, base + a * inner));
            }
            let mut sum = 0.0;
            for a in 0..axis_dim {
                sum += (read(out, base + a * inner) - max).exp();
            }
            let log_sum = sum.ln();
            for a in 0..axis_dim {
                out[base + a * inner] = (read(out, base + a * inner) - max) - log_sum;
            }
        }
    }
}

impl LogSoftmaxKernel {
    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<SoftmaxGeometry> {
        SoftmaxGeometry::new("LogSoftmax", self.axis, self.coerce_2d, inputs, outputs)
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        check_arity("LogSoftmax", inputs, outputs, 1, 1, 1)?;
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "LogSoftmax: provide exactly one Y output, got {} slots",
                outputs.len()
            )));
        }
        let input = TensorMetadata::new(inputs[0].dtype, inputs[0].shape, !inputs[0].is_absent());
        let geometry = self.geometry(&[input], &[OutputMetadata::from_view(&outputs[0])])?;
        validate_f32_widen_input("LogSoftmax", &inputs[0]).map_err(|error| EpError::KernelFailed(format!("LogSoftmax: X view is invalid: {error}; bind a valid floating Host-accessible input view")))?;
        outputs[0].validate().map_err(|error| {
            EpError::KernelFailed(format!(
                "LogSoftmax: Y view is invalid: {error}; bind valid Host-accessible output storage"
            ))
        })?;
        if !outputs[0].device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "LogSoftmax: Y on {:?} is not Host-accessible; explicitly transfer it before CPU execution",
                outputs[0].device
            )));
        }
        // SAFETY: synchronous CPU execution owns exclusive live Host scratch,
        // disjoint from tensors. The shared helper validates and initializes it.
        unsafe {
            geometry.workspace.with_buffers(workspace, |[y]| {
                if geometry.count == 0 {
                    return Ok(());
                }
                if can_borrow_dense::<f32>(&inputs[0]) {
                    // Keep the existing zero-copy source path. Y is exclusive
                    // workspace, disjoint from X even when the real output aliases X.
                    let x = to_dense_f32_widen("LogSoftmax", &inputs[0])?;
                    log_softmax_slices(
                        y,
                        |_, index| x[index],
                        geometry.outer,
                        geometry.width,
                        geometry.inner,
                    );
                } else {
                    // Snapshot every source value before reducing or publishing.
                    // Conversion retains its existing ISA paths and f64->f32 semantics.
                    to_dense_f32_widen_into("LogSoftmax", &inputs[0], y)?;
                    log_softmax_slices(
                        y,
                        |lane, index| lane[index],
                        geometry.outer,
                        geometry.width,
                        geometry.inner,
                    );
                }
                // Finish all reads before publishing, preserving overlapping X/Y.
                write_dense_f32_narrow("LogSoftmax", &mut outputs[0], y)
            })
        }
    }
}

impl Kernel for LogSoftmaxKernel {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            self.geometry(inputs, outputs)?.workspace.requirement(),
        ))
    }
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};

    fn run(axis: i64, coerce_2d: bool, x: &Owned, out: &mut Owned) {
        LogSoftmaxKernel { axis, coerce_2d }
            .execute_prepared(&[x.view()], &mut [out.view_mut()])
            .unwrap();
    }

    fn approx(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (&actual, &want) in got.iter().zip(expected) {
            assert!((actual - want).abs() < 1e-6, "{got:?} vs {expected:?}");
        }
    }

    #[test]
    fn log_softmax_opset13_last_axis_has_expected_values_and_normalizes() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        run(-1, false, &x, &mut out);
        let expected = [-2.407_606, -1.407_606, -0.407_605_95];
        let result = out.to_f32();
        approx(&result[..3], &expected);
        approx(&result[3..], &expected);
        for row in result.as_chunks::<3>().0 {
            assert!((row.iter().map(|v| v.exp()).sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn log_softmax_opset11_axis_one_coerces_trailing_dimensions() {
        let x = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 1., 2., 3., 4.]);
        let mut legacy = Owned::zeros_f32(&[2, 2, 2]);
        let mut modern = Owned::zeros_f32(&[2, 2, 2]);
        run(1, true, &x, &mut legacy);
        run(1, false, &x, &mut modern);
        // Legacy treats each [2,2] block as one 4-element row.
        approx(
            &legacy.to_f32()[..4],
            &[-3.440_189_6, -2.440_189_6, -1.440_189_7, -0.440_189_7],
        );
        // Opset 13 reduces only axis 1, leaving the final dimension independent.
        approx(
            &modern.to_f32()[..4],
            &[-2.126_928, -2.126_928, -0.126_928_05, -0.126_928_05],
        );
    }

    #[test]
    fn log_softmax_is_translation_invariant_for_large_logits() {
        let small = Owned::f32(&[2], &[0., 1.]);
        let shifted = Owned::f32(&[2], &[1_000_000., 1_000_001.]);
        let equal_large = Owned::f32(&[2], &[1e10, 1e10]);
        let mut small_out = Owned::zeros_f32(&[2]);
        let mut shifted_out = Owned::zeros_f32(&[2]);
        let mut equal_large_out = Owned::zeros_f32(&[2]);

        run(-1, false, &small, &mut small_out);
        run(-1, false, &shifted, &mut shifted_out);
        run(-1, false, &equal_large, &mut equal_large_out);

        approx(&shifted_out.to_f32(), &small_out.to_f32());
        approx(&equal_large_out.to_f32(), &[-core::f32::consts::LN_2; 2]);
        for output in [&shifted_out, &equal_large_out] {
            assert!((output.to_f32().iter().map(|v| v.exp()).sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
    #[test]
    fn log_softmax_bf16_matches_widened_f32_reference() {
        let values = [-10.0, 0.0, 1.0, 80.0, -80.0, 2.0];
        let x = Owned::bf16(&[2, 3], &values);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        run(-1, false, &x, &mut out);
        let rounded = x.to_bf16_as_f32();
        let mut reference = vec![0.0; rounded.len()];
        log_softmax_slices(&mut reference, |_, index| rounded[index], 2, 3, 1);
        let expected: Vec<_> = reference
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        assert_eq!(out.to_bf16_as_f32(), expected);
    }
    #[test]
    fn workspace_checks_geometry_capacity_and_empty_shapes() {
        use onnx_runtime_ir::DataType;
        for coerce_2d in [false, true] {
            let k = LogSoftmaxKernel { axis: 1, coerce_2d };
            let input = [TensorMetadata::new(DataType::Float32, &[2, 3, 4], true)];
            let out = OutputMetadata::tensor(DataType::Float32, &[2, 3, 4]);
            let request = k
                .workspace_requirement(&input, std::slice::from_ref(&out))
                .unwrap()
                .require_ready()
                .unwrap();
            assert_eq!(request.bytes, 96);
            assert_eq!(
                request.lifetime,
                onnx_runtime_ep_api::WorkspaceLifetime::StepScoped
            );
            assert_eq!(
                k.workspace_requirement(
                    &input,
                    &[OutputMetadata::Unresolved {
                        dtype: DataType::Float32
                    }]
                )
                .unwrap()
                .require_ready()
                .unwrap()
                .bytes,
                96
            );
            for bad in [
                vec![],
                vec![OutputMetadata::Absent],
                vec![OutputMetadata::tensor(DataType::Float32, &[24])],
                vec![OutputMetadata::tensor(DataType::Int32, &[2, 3, 4])],
                vec![out.clone(), out.clone()],
            ] {
                assert!(
                    k.workspace_requirement(&input, &bad)
                        .unwrap_err()
                        .to_string()
                        .contains("LogSoftmax")
                );
            }
            for shape in [&[0, 3, 4][..], &[2, 0, 4], &[2, 3, 0]] {
                let x = Owned::f32(shape, &[]);
                let mut y = Owned::zeros_f32(shape);
                assert_eq!(
                    k.workspace_requirement(
                        &[TensorMetadata::new(DataType::Float32, shape, true)],
                        &[OutputMetadata::tensor(DataType::Float32, shape)]
                    )
                    .unwrap()
                    .require_ready()
                    .unwrap()
                    .bytes,
                    0
                );
                k.execute(&[x.view()], &mut [y.view_mut()]).unwrap();
            }
            let shape = [0, usize::MAX, 2];
            assert!(
                k.workspace_requirement(
                    &[TensorMetadata::new(DataType::Float32, &shape, true)],
                    &[OutputMetadata::tensor(DataType::Float32, &shape)]
                )
                .unwrap_err()
                .to_string()
                .contains("overflow")
            );
            let shape = [isize::MAX as usize / 4 + 1];
            let k = LogSoftmaxKernel { axis: 0, coerce_2d };
            assert!(
                k.workspace_requirement(
                    &[TensorMetadata::new(DataType::Float32, &shape, true)],
                    &[OutputMetadata::tensor(DataType::Float32, &shape)]
                )
                .is_err()
            );
            for axis in [i64::MIN, -4, 3, i64::MAX] {
                assert!(
                    LogSoftmaxKernel { axis, coerce_2d }
                        .workspace_requirement(&input, std::slice::from_ref(&out))
                        .unwrap_err()
                        .to_string()
                        .contains("axis")
                );
            }
        }
    }

    #[test]
    fn workspace_refusal_precedes_writes_and_fresh_storage_is_initialized() {
        use onnx_runtime_ep_api::DevicePtrMut;
        let k = LogSoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        };
        let x = Owned::f32(&[2, 3], &[1., 2., 3., -2., -1., 4.]);
        let mut y = Owned::f32(&[2, 3], &[-123.; 6]);
        let mut storage = [0x7fc01234u32; 8];
        let pointer = unsafe { storage.as_mut_ptr().add(1).cast::<u8>() };
        for ws in [
            None,
            Some(WorkspaceView::new(DevicePtrMut(pointer.cast()), 23)),
            Some(WorkspaceView::new(
                DevicePtrMut(unsafe { pointer.add(1).cast() }),
                24,
            )),
        ] {
            let error = k
                .execute_with_workspace(&[x.view()], &mut [y.view_mut()], ws)
                .unwrap_err();
            assert!(error.to_string().contains("workspace"), "{error}");
            assert_eq!(y.to_f32(), [-123.; 6]);
            assert_eq!(storage, [0x7fc01234; 8]);
        }
        k.execute_with_workspace(
            &[x.view()],
            &mut [y.view_mut()],
            Some(WorkspaceView::new(DevicePtrMut(pointer.cast()), 24)),
        )
        .unwrap();
        assert_eq!(storage[0], 0x7fc01234);
        assert_eq!(storage[7], 0x7fc01234);
        let expected = y.to_f32();
        let mut fresh = [std::mem::MaybeUninit::<f32>::uninit(); 6];
        k.execute_with_workspace(
            &[x.view()],
            &mut [y.view_mut()],
            Some(WorkspaceView::new(
                DevicePtrMut(fresh.as_mut_ptr().cast()),
                24,
            )),
        )
        .unwrap();
        assert_eq!(y.to_f32(), expected);
    }

    #[test]
    fn both_opset_semantics_and_axis_spellings_match_independent_f64_coordinates() {
        let values = (0..24)
            .map(|i| (i as f32 * 0.2).sin() * 4.)
            .collect::<Vec<_>>();
        let x = Owned::f32(&[2, 3, 4], &values);
        for legacy in [false, true] {
            for axis in 0..3 {
                for negative in [false, true] {
                    let mut y = Owned::zeros_f32(&[2, 3, 4]);
                    run(
                        if negative {
                            axis as i64 - 3
                        } else {
                            axis as i64
                        },
                        legacy,
                        &x,
                        &mut y,
                    );
                    let actual = y.to_f32();
                    for flat in 0..24 {
                        let coords = [flat / 12, (flat / 4) % 3, flat % 4];
                        let same_group = |other: usize| {
                            let c = [other / 12, (other / 4) % 3, other % 4];
                            (0..3).all(|d| d == axis || (legacy && d > axis) || coords[d] == c[d])
                        };
                        let max = (0..24)
                            .filter(|&j| same_group(j))
                            .map(|j| f64::from(values[j]))
                            .fold(f64::NEG_INFINITY, f64::max);
                        let sum = (0..24)
                            .filter(|&j| same_group(j))
                            .map(|j| (f64::from(values[j]) - max).exp())
                            .sum::<f64>();
                        let expected = f64::from(values[flat]) - max - sum.ln();
                        assert!(
                            (f64::from(actual[flat]) - expected).abs() < 2e-6,
                            "legacy={legacy} axis={axis} flat={flat}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_input_output_and_strided_half_storage_preserve_values() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId};
        let values = [1., 2., 3., 4., 5., 6.];
        let x = Owned::f32(&[2, 3], &values);
        let mut y = Owned::zeros_f32(&[2, 3]);
        run(1, false, &x, &mut y);
        let mut shared = values;
        let pointer = shared.as_mut_ptr();
        let input = TensorView::new(
            DevicePtr(pointer.cast()),
            DataType::Float32,
            &[2, 3],
            &[3, 1],
            DeviceId::cpu(),
        );
        let output = TensorMut::new(
            DevicePtrMut(pointer.cast()),
            DataType::Float32,
            &[2, 3],
            &[3, 1],
            DeviceId::cpu(),
        );
        LogSoftmaxKernel {
            axis: 1,
            coerce_2d: false,
        }
        .execute_prepared(&[input], &mut [output])
        .unwrap();
        assert_eq!(shared.as_slice(), y.to_f32());
        let padded = values
            .into_iter()
            .flat_map(|x| [x, 123.])
            .collect::<Vec<_>>();
        let x = Owned::f16(&[12], &padded).with_view(&[2, 3], &[6, 2]);
        let mut out = Owned::f16(&[12], &[-123.; 12]).with_view(&[2, 3], &[6, 2]);
        run(1, false, &x, &mut out);
        let physical = out.to_f16_as_f32();
        for (i, expected) in y.to_f32().iter().enumerate() {
            assert_eq!(physical[2 * i], half::f16::from_f32(*expected).to_f32());
            assert_eq!(physical[2 * i + 1], -123.);
        }
        let x = Owned::f64(&[2, 3], &values.map(f64::from));
        let mut out = Owned::zeros(DataType::Float64, &[2, 3]);
        run(1, false, &x, &mut out);
        assert_eq!(
            out.to_f64(),
            y.to_f32()
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn widened_snapshot_finishes_before_shifted_overlapping_output_is_published() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId};
        for values in [
            [1., -2., 3., -4., 5., -6.],
            [f32::NEG_INFINITY, 0., 1., -0., f32::NAN, f32::INFINITY],
        ] {
            for legacy in [false, true] {
                for axis in [0, 1] {
                    let source = Owned::f32(&[2, 3], &values);
                    let mut expected = Owned::zeros_f32(&[2, 3]);
                    run(axis, legacy, &source, &mut expected);
                    for destination in [0, 6] {
                        let mut storage = [123f32; 16];
                        for (i, &value) in values.iter().enumerate() {
                            storage[1 + 2 * i] = value;
                        }
                        let before = storage.map(f32::to_bits);
                        let pointer = storage.as_mut_ptr();
                        let mut x = TensorView::new(
                            DevicePtr(pointer.cast()),
                            DataType::Float32,
                            &[2, 3],
                            &[6, 2],
                            DeviceId::cpu(),
                        );
                        x.byte_offset = size_of::<f32>();
                        let mut y = TensorMut::new(
                            DevicePtrMut(pointer.cast()),
                            DataType::Float32,
                            &[2, 3],
                            &[3, 1],
                            DeviceId::cpu(),
                        );
                        y.byte_offset = destination * size_of::<f32>();
                        LogSoftmaxKernel {
                            axis,
                            coerce_2d: legacy,
                        }
                        .execute_prepared(&[x], &mut [y])
                        .unwrap();
                        for (actual, want) in storage[destination..destination + 6]
                            .iter()
                            .zip(expected.to_f32())
                        {
                            assert_eq!(actual.to_bits(), want.to_bits());
                        }
                        for i in 0..16 {
                            if !(destination..destination + 6).contains(&i) {
                                assert_eq!(storage[i].to_bits(), before[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn governed_logsoftmax_staging_reuses_one_lane_without_warm_issuer_callbacks() {
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
        let resources = crate::CpuMemoryResources::new(
            books.weight_memory(HolderId::new(13)),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let kernel = LogSoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        };
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 4., 6., 8.]);
        let inputs = [x.view()];
        let meta = [TensorMetadata::new(inputs[0].dtype, inputs[0].shape, true)];
        let mut y = Owned::zeros_f32(&[2, 4]);
        let outputs = [OutputMetadata::tensor(
            onnx_runtime_ir::DataType::Float32,
            &[2, 4],
        )];
        let request = kernel
            .workspace_requirement(&meta, &outputs)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(request.bytes, 32);
        let first = provider
            .allocate_workspace(32, request.alignment, request.role())
            .unwrap();
        let identity = first.managed_owner().unwrap().identity();
        assert!(
            provider
                .allocate_workspace(32, request.alignment, request.role())
                .is_err(),
            "active capacity cannot fund a concurrent execution"
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
            let request = kernel
                .workspace_requirement_for_execution(&inputs, &[], &meta, &outputs, None)
                .unwrap()
                .require_ready()
                .unwrap();
            let mut allocation = provider
                .allocate_workspace(request.bytes as usize, request.alignment, request.role())
                .unwrap();
            assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
            let workspace = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            );
            kernel
                .execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
                .unwrap();
            provider.deallocate_workspace(allocation).unwrap();
        }
        assert_eq!(
            (books.claims(), books.releases()),
            calls,
            "including zero-byte callbacks"
        );
        assert_eq!(books.used(Tier::Host), 32);
        for row in y.to_f32().chunks_exact(4) {
            assert!((row.iter().map(|v| v.exp()).sum::<f32>() - 1.).abs() < 1e-6);
        }
        pool.trim_idle(32).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
    #[test]
    fn invalid_views_are_rejected_before_staging_or_output_writes() {
        use onnx_runtime_ep_api::DevicePtrMut;
        use onnx_runtime_ir::DeviceId;
        let kernel = LogSoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        };
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut y = Owned::f32(&[2, 3], &[-123.; 6]);
        let mut backing = [0x7fc01234u32; 6];
        for case in 0..6 {
            let mut input = x.view();
            let mut output = y.view_mut();
            match case {
                0 => input.strides = &[],
                1 => input.byte_offset = usize::MAX,
                2 => input.device = DeviceId::cuda(0),
                3 => output.strides = &[],
                4 => output.byte_offset = usize::MAX,
                _ => output.device = DeviceId::cuda(0),
            }
            let error = kernel
                .execute_with_workspace(
                    &[input],
                    &mut [output],
                    Some(WorkspaceView::new(
                        DevicePtrMut(backing.as_mut_ptr().cast()),
                        24,
                    )),
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("LogSoftmax"),
                "case {case}: {error}"
            );
            assert_eq!(y.to_f32(), [-123.; 6]);
            assert_eq!(backing, [0x7fc01234; 6]);
        }
    }
}
