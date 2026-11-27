//! Real caller-lane reuse for eager record-sized metadata, not a timing benchmark.
mod common;

use common::{Tensor, input};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, OutputMetadata, TensorMut, TensorView,
    WorkspacePlanning, WorkspaceView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{Attribute, DataType, compute_contiguous_strides, static_shape};

struct Case {
    op: &'static str,
    opset: u64,
    inputs: Vec<Tensor>,
    output: (DataType, Vec<usize>),
    attributes: Vec<(&'static str, Attribute)>,
    metadata_bytes: u64,
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires a CUDA device; enable gpu-tests"
)]
#[test]
fn eager_shape_metadata_reuses_one_owner_without_private_allocations() {
    let ep = common::require_cuda();
    let runtime = ep.runtime();
    let cases = [
        Case {
            op: "LpPool",
            opset: 18,
            inputs: vec![input(
                DataType::Float32,
                &[1, 1, 2, 3],
                &[1f32, 2., 3., 4., 5., 6.],
            )],
            output: (DataType::Float32, vec![1, 1, 1, 2]),
            attributes: vec![
                ("kernel_shape", Attribute::Ints(vec![2, 2])),
                ("p", Attribute::Int(1)),
            ],
            metadata_bytes: 96,
        },
        Case {
            op: "Mod",
            opset: 13,
            inputs: vec![
                input(DataType::Int32, &[2, 3], &[-8i32, -5, -1, 1, 5, 8]),
                input(DataType::Int32, &[3], &[3i32, 4, 2]),
            ],
            output: (DataType::Int32, vec![2, 3]),
            attributes: vec![],
            metadata_bytes: 48,
        },
        Case {
            op: "NonZero",
            opset: 13,
            inputs: vec![input(
                DataType::Float32,
                &[2, 3],
                &[0f32, 2., 0., 3., 0., 4.],
            )],
            output: (DataType::Int64, vec![2, 3]),
            attributes: vec![],
            metadata_bytes: 16,
        },
        Case {
            op: "CenterCropPad",
            opset: 18,
            inputs: vec![
                input(DataType::Int32, &[3, 2], &[1i32, 2, 3, 4, 5, 6]),
                input(DataType::Int64, &[2], &[2i64, 5]),
            ],
            output: (DataType::Int32, vec![2, 5]),
            attributes: vec![],
            metadata_bytes: 64,
        },
        Case {
            op: "Col2Im",
            opset: 18,
            inputs: vec![
                input(
                    DataType::Float32,
                    &[1, 4, 4],
                    &(1..=16).map(|n| n as f32).collect::<Vec<_>>(),
                ),
                input(DataType::Int64, &[1, 2], &[3i64, 3]),
                input(DataType::Int64, &[2, 1], &[2i64, 2]),
            ],
            output: (DataType::Float32, vec![1, 1, 3, 3]),
            attributes: vec![],
            metadata_bytes: 96,
        },
        Case {
            op: "QLinearMatMul",
            opset: 10,
            inputs: vec![
                input(
                    DataType::Uint8,
                    &[2, 2, 3],
                    &[120u8, 125, 131, 118, 129, 140, 121, 126, 132, 119, 130, 141],
                ),
                input(DataType::Float32, &[], &[0.25f32]),
                input(DataType::Uint8, &[], &[123u8]),
                input(
                    DataType::Uint8,
                    &[1, 3, 2],
                    &[126u8, 119, 130, 121, 124, 135],
                ),
                input(DataType::Float32, &[], &[0.5f32]),
                input(DataType::Uint8, &[], &[127u8]),
                input(DataType::Float32, &[], &[0.2f32]),
                input(DataType::Uint8, &[], &[111u8]),
            ],
            output: (DataType::Uint8, vec![2, 2, 2]),
            attributes: vec![],
            metadata_bytes: 32,
        },
        Case {
            op: "QLinearMatMul",
            opset: 10,
            inputs: vec![
                input(DataType::Uint8, &[3], &[2u8, 3, 4]),
                input(DataType::Float32, &[], &[1f32]),
                input(DataType::Uint8, &[], &[0u8]),
                input(DataType::Uint8, &[3], &[4u8, 3, 2]),
                input(DataType::Float32, &[], &[1f32]),
                input(DataType::Uint8, &[], &[0u8]),
                input(DataType::Float32, &[], &[1f32]),
                input(DataType::Uint8, &[], &[0u8]),
            ],
            output: (DataType::Uint8, vec![]),
            attributes: vec![],
            metadata_bytes: 16,
        },
        Case {
            op: "Pad",
            opset: 18,
            inputs: vec![
                input(DataType::Uint8, &[2, 2], &[1u8, 2, 3, 4]),
                input(DataType::Int64, &[4], &[1i64, 0, 0, 1]),
                input(DataType::Uint8, &[], &[9u8]),
            ],
            output: (DataType::Uint8, vec![3, 3]),
            attributes: vec![],
            metadata_bytes: 65,
        },
        Case {
            op: "Pad",
            opset: 18,
            inputs: vec![
                input(DataType::Uint8, &[], &[2u8]),
                input(DataType::Int64, &[0], &[] as &[i64]),
                input(DataType::Uint8, &[], &[9u8]),
            ],
            output: (DataType::Uint8, vec![]),
            attributes: vec![],
            metadata_bytes: 1,
        },
        Case {
            op: "Resize",
            opset: 19,
            inputs: vec![
                input(DataType::Float32, &[2, 2], &[1f32, 2., 3., 4.]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[2], &[1.6f32, 1.6]),
            ],
            output: (DataType::Float32, vec![3, 3]),
            attributes: vec![(
                "coordinate_transformation_mode",
                Attribute::String("half_pixel_symmetric".into()),
            )],
            metadata_bytes: 96,
        },
        Case {
            op: "Resize",
            opset: 19,
            inputs: vec![
                input(DataType::Float32, &[2, 2], &[1f32, 2., 3., 4.]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Int64, &[2], &[3i64, 3]),
            ],
            output: (DataType::Float32, vec![3, 3]),
            attributes: vec![(
                "coordinate_transformation_mode",
                Attribute::String("half_pixel_symmetric".into()),
            )],
            metadata_bytes: 80,
        },
    ];
    let mut workspace = None;
    let mut original_identity = None;
    for case in cases {
        let outputs = [case.output];
        let expected = common::run_cpu(
            case.op,
            "",
            case.opset,
            &case.inputs,
            &outputs,
            &case.attributes,
        );
        let (graph, node) = common::build_graph(
            case.op,
            "",
            case.opset,
            &case.inputs,
            &outputs,
            &case.attributes,
        );
        let shapes = case
            .inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>();
        let claim_shapes = shapes
            .iter()
            .map(|shape| static_shape(shape.iter().copied()))
            .collect::<Vec<_>>();
        let dtypes = case
            .inputs
            .iter()
            .map(|input| input.dtype)
            .collect::<Vec<_>>();
        assert!(matches!(
            ep.supports_op(graph.node(node), case.opset, &claim_shapes, &dtypes, &[]),
            KernelMatch::Supported {
                workspace_planning: WorkspacePlanning::Metadata,
                ..
            }
        ));
        let kernel = ep
            .get_kernel(graph.node(node), &shapes, case.opset)
            .unwrap();
        assert!(!kernel.capture_support().is_supported());
        let mut buffers = Vec::new();
        for input in &case.inputs {
            let mut buffer = ep.allocate(input.bytes.len().max(1), 256).unwrap();
            unsafe {
                runtime
                    .htod(&input.bytes, cuptr(buffer.as_mut_ptr()))
                    .unwrap();
            }
            buffers.push(buffer);
        }
        let strides = shapes
            .iter()
            .map(|shape| {
                compute_contiguous_strides(shape)
                    .expect("fixture shape has representable contiguous strides")
            })
            .collect::<Vec<_>>();
        let views = case
            .inputs
            .iter()
            .zip(&buffers)
            .zip(&strides)
            .map(|((input, buffer), strides)| {
                TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    input.dtype,
                    &input.shape,
                    strides,
                    ep.device_id(),
                )
            })
            .collect::<Vec<_>>();
        let (dtype, shape) = &outputs[0];
        let output_bytes = dtype.storage_bytes(shape.iter().product());
        let mut output_buffer = ep.allocate(output_bytes.max(1), 256).unwrap();
        let output_strides = compute_contiguous_strides(shape)
            .expect("fixture shape has representable contiguous strides");
        let mut output = [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            *dtype,
            shape,
            &output_strides,
            ep.device_id(),
        )];
        let requirement = common::runtime_workspace_requirement(
            kernel.as_ref(),
            &views,
            &[OutputMetadata::from_view(&output[0])],
            &output,
        )
        .unwrap();
        assert_eq!(requirement.bytes, case.metadata_bytes);
        let view = common::prepare_workspace(&ep, requirement, &mut workspace)
            .unwrap()
            .unwrap();
        let owner = workspace.as_ref().unwrap();
        let identity = owner.managed_owner().unwrap().identity();
        assert_eq!(
            *original_identity.get_or_insert(identity),
            identity,
            "different metadata kernels should share one fitting caller owner"
        );
        // No kernel may assume a prior node left its own descriptor at this address.
        unsafe {
            runtime
                .htod(&vec![0xff; view.bytes()], cuptr(view.ptr().0))
                .unwrap();
        }
        let transfers = runtime.transfer_counts();
        for invalid in [
            None,
            Some(WorkspaceView::new(
                view.ptr(),
                requirement.bytes as usize - 1,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut(view.ptr().0.wrapping_add(1)),
                view.bytes(),
            )),
        ] {
            assert!(
                kernel
                    .execute_with_workspace(&views, &mut output, invalid)
                    .is_err()
            );
            assert_eq!(
                runtime.transfer_counts(),
                transfers,
                "refuse incomplete backing before upload"
            );
        }
        kernel
            .execute_with_workspace(&views, &mut output, Some(view))
            .unwrap();
        let after_first = runtime.transfer_counts();
        assert_eq!(
            after_first.host_to_device,
            transfers.host_to_device + 1,
            "exactly one metadata upload per nonempty call"
        );
        let expected_readbacks = match case.op {
            "CenterCropPad" => 1,
            "Col2Im" => 2,
            "QLinearMatMul" => 3,
            "Pad" => u64::from(!case.inputs[1].bytes.is_empty()) + 1,
            "Resize" => 1,
            _ => 0,
        };
        assert_eq!(
            after_first.device_to_host,
            transfers.device_to_host + expected_readbacks,
            "metadata planning must not add control/scale readbacks"
        );
        let allocations = runtime.allocation_counts();
        for _ in 0..4 {
            let fitted = common::prepare_kernel_workspace(
                &ep,
                kernel.as_ref(),
                &views,
                &output,
                &mut workspace,
            )
            .unwrap();
            kernel
                .execute_with_workspace(&views, &mut output, fitted)
                .unwrap();
        }
        assert_eq!(
            runtime.allocation_counts(),
            allocations,
            "warm dispatch must not allocate private metadata"
        );
        assert!(kernel.device_graph_resources().is_empty());
        let mut actual = vec![0; output_bytes];
        unsafe {
            runtime
                .dtoh(&mut actual, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
        assert_eq!(actual, expected[0], "{} dirty-lane parity", case.op);
        // Same address and length, changed contents: preparation must never
        // substitute a stale host value for this dispatch's control inputs.
        if matches!(case.op, "CenterCropPad" | "Col2Im") {
            let (slot, changed) = if case.op == "CenterCropPad" {
                (1, common::raw(&[-1i64, 5]))
            } else {
                (2, common::raw(&[-2i64, 2]))
            };
            unsafe {
                runtime
                    .htod(&changed, cuptr(buffers[slot].as_ptr()))
                    .unwrap();
            }
            let before = runtime.transfer_counts();
            let error = kernel
                .execute_with_workspace(&views, &mut output, Some(view))
                .unwrap_err();
            assert!(error.to_string().contains(case.op), "{error}");
            assert_eq!(
                runtime.transfer_counts().host_to_device,
                before.host_to_device
            );
            unsafe {
                runtime
                    .htod(&case.inputs[slot].bytes, cuptr(buffers[slot].as_ptr()))
                    .unwrap();
            }
            kernel
                .execute_with_workspace(&views, &mut output, Some(view))
                .unwrap();
        } else if case.op == "QLinearMatMul" {
            let mut updated = case.inputs.clone();
            updated[1].bytes = common::raw(&[0.5f32]);
            unsafe {
                runtime
                    .htod(&updated[1].bytes, cuptr(buffers[1].as_ptr()))
                    .unwrap();
            }
            kernel
                .execute_with_workspace(&views, &mut output, Some(view))
                .unwrap();
            let updated_expected = common::run_cpu(
                case.op,
                "",
                case.opset,
                &updated,
                &outputs,
                &case.attributes,
            );
            assert_ne!(
                updated_expected, expected,
                "the changed scale must affect the oracle"
            );
            let mut updated_output = vec![0; output_bytes];
            unsafe {
                runtime
                    .dtoh(&mut updated_output, cuptr(output_buffer.as_ptr()))
                    .unwrap();
            }
            assert_eq!(updated_output, updated_expected[0]);
            unsafe {
                runtime
                    .htod(&case.inputs[1].bytes, cuptr(buffers[1].as_ptr()))
                    .unwrap();
            }
        }
        if case.op == "Pad" && !outputs[0].1.is_empty() {
            let mut updated = case.inputs.clone();
            updated[2].bytes = vec![7u8];
            unsafe {
                runtime
                    .htod(&updated[2].bytes, cuptr(buffers[2].as_ptr()))
                    .unwrap();
            }
            kernel
                .execute_with_workspace(&views, &mut output, Some(view))
                .unwrap();
            let wanted = common::run_cpu(
                case.op,
                "",
                case.opset,
                &updated,
                &outputs,
                &case.attributes,
            );
            let mut actual = vec![0; output_bytes];
            unsafe {
                runtime
                    .dtoh(&mut actual, cuptr(output_buffer.as_ptr()))
                    .unwrap();
            }
            assert_eq!(actual, wanted[0]);
            assert_ne!(wanted, expected);
            unsafe {
                runtime
                    .htod(&case.inputs[2].bytes, cuptr(buffers[2].as_ptr()))
                    .unwrap();
            }
        } else if case.op == "Resize" {
            let (slot, bad) = if case.inputs.len() == 3 {
                (2, common::raw(&[f32::NAN, 1.6]))
            } else {
                (3, common::raw(&[-1i64, 3]))
            };
            unsafe {
                runtime.htod(&bad, cuptr(buffers[slot].as_ptr())).unwrap();
            }
            let before = runtime.transfer_counts();
            let error = kernel
                .execute_with_workspace(&views, &mut output, Some(view))
                .unwrap_err();
            assert!(error.to_string().contains("Resize"), "{error}");
            assert_eq!(
                runtime.transfer_counts().host_to_device,
                before.host_to_device
            );
            unsafe {
                runtime
                    .htod(&case.inputs[slot].bytes, cuptr(buffers[slot].as_ptr()))
                    .unwrap();
            }
        }
        // A raw recording cannot bypass the kernel's eager-upload restriction.
        runtime.begin_graph_capture(&[]).unwrap();
        let error = kernel
            .execute_with_workspace(&views, &mut output, Some(view))
            .unwrap_err();
        assert!(error.to_string().contains("CUDA graph capture"), "{error}");
        runtime.abort_graph_capture().unwrap();
        ep.deallocate(output_buffer).unwrap();
        for buffer in buffers {
            ep.deallocate(buffer).unwrap();
        }
    }
    let owner = workspace.unwrap();
    let settlement = owner.managed_settlement_wait().unwrap();
    ep.deallocate_workspace(owner).unwrap();
    assert_eq!(
        settlement.wait(std::time::Duration::from_secs(30)),
        onnx_runtime_memory_governor::AllocationSettlementStatus::Released
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires a CUDA device; enable gpu-tests"
)]
#[test]
fn scalar_and_empty_shape_metadata_need_no_caller_capacity() {
    use onnx_runtime_ep_api::{TensorMetadata, WorkspacePreparation, WorkspaceRequirement};
    let ep = common::require_cuda();
    let cases = [
        Case {
            op: "Mod",
            opset: 13,
            inputs: vec![
                input(DataType::Int32, &[], &[-8i32]),
                input(DataType::Int32, &[], &[3i32]),
            ],
            output: (DataType::Int32, vec![]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "Mod",
            opset: 13,
            inputs: vec![
                input(DataType::Int32, &[2, 0, 3], &[] as &[i32]),
                input(DataType::Int32, &[3], &[1i32, 2, 3]),
            ],
            output: (DataType::Int32, vec![2, 0, 3]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "NonZero",
            opset: 13,
            inputs: vec![input(DataType::Float32, &[2, 3], &[0f32; 6])],
            output: (DataType::Int64, vec![2, 0]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "NonZero",
            opset: 13,
            inputs: vec![input(DataType::Float32, &[], &[2f32])],
            output: (DataType::Int64, vec![0, 1]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "LpPool",
            opset: 18,
            inputs: vec![input(DataType::Float32, &[0, 1, 2, 3], &[] as &[f32])],
            output: (DataType::Float32, vec![0, 1, 1, 2]),
            attributes: vec![
                ("kernel_shape", Attribute::Ints(vec![2, 2])),
                ("p", Attribute::Int(1)),
            ],
            metadata_bytes: 0,
        },
        Case {
            op: "CenterCropPad",
            opset: 18,
            inputs: vec![
                input(DataType::Int32, &[3, 2], &[1i32, 2, 3, 4, 5, 6]),
                input(DataType::Int64, &[2], &[0i64, 5]),
            ],
            output: (DataType::Int32, vec![0, 5]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "Col2Im",
            opset: 18,
            inputs: vec![
                input(DataType::Float32, &[0, 4, 4], &[] as &[f32]),
                input(DataType::Int64, &[2], &[3i64, 3]),
                input(DataType::Int64, &[2], &[2i64, 2]),
            ],
            output: (DataType::Float32, vec![0, 1, 3, 3]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "QLinearMatMul",
            opset: 10,
            inputs: vec![
                input(DataType::Uint8, &[0, 3], &[] as &[u8]),
                input(DataType::Float32, &[], &[1f32]),
                input(DataType::Uint8, &[], &[0u8]),
                input(DataType::Uint8, &[3, 2], &[1u8, 2, 3, 4, 5, 6]),
                input(DataType::Float32, &[], &[1f32]),
                input(DataType::Uint8, &[], &[0u8]),
                input(DataType::Float32, &[], &[1f32]),
                input(DataType::Uint8, &[], &[0u8]),
            ],
            output: (DataType::Uint8, vec![0, 2]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "Pad",
            opset: 18,
            inputs: vec![
                input(DataType::Uint8, &[2], &[1u8, 2]),
                input(DataType::Int64, &[2], &[-1i64, -1]),
            ],
            output: (DataType::Uint8, vec![0]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "Resize",
            opset: 19,
            inputs: vec![
                input(DataType::Float32, &[2, 2], &[1f32, 2., 3., 4.]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[2], &[0.1f32, 0.1]),
            ],
            output: (DataType::Float32, vec![0, 0]),
            attributes: vec![],
            metadata_bytes: 0,
        },
        Case {
            op: "Resize",
            opset: 10,
            inputs: vec![
                input(DataType::Float32, &[], &[2f32]),
                input(DataType::Float32, &[0], &[] as &[f32]),
            ],
            output: (DataType::Float32, vec![]),
            attributes: vec![],
            metadata_bytes: 0,
        },
    ];
    for case in cases {
        let outputs = [case.output];
        let (graph, node) = common::build_graph(
            case.op,
            "",
            case.opset,
            &case.inputs,
            &outputs,
            &case.attributes,
        );
        let shapes = case
            .inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>();
        let kernel = ep
            .get_kernel(graph.node(node), &shapes, case.opset)
            .unwrap();
        let metadata = case
            .inputs
            .iter()
            .map(|input| TensorMetadata::new(input.dtype, &input.shape, true))
            .collect::<Vec<_>>();
        assert_eq!(
            kernel
                .workspace_requirement(
                    &metadata,
                    &[OutputMetadata::tensor(outputs[0].0, &outputs[0].1)]
                )
                .unwrap(),
            WorkspacePreparation::Ready(WorkspaceRequirement::NONE)
        );
        // The common harness passes None for this zero requirement, so these
        // are real dispatch checks rather than layout-only assertions.
        assert_eq!(
            common::run_cuda(
                &ep,
                case.op,
                "",
                case.opset,
                &case.inputs,
                &outputs,
                &case.attributes
            ),
            common::run_cpu(
                case.op,
                "",
                case.opset,
                &case.inputs,
                &outputs,
                &case.attributes
            ),
            "{} zero-work parity",
            case.op
        );
        let mut invalid = case.inputs.clone();
        match case.op {
            "CenterCropPad" => invalid[1].bytes = common::raw(&[-1i64, 5]),
            "Col2Im" => invalid[2].bytes = common::raw(&[-2i64, 2]),
            "QLinearMatMul" => invalid[1].bytes = common::raw(&[f32::NAN]),
            "Pad" => invalid[1].bytes = common::raw(&[-3i64, 0]),
            "Resize" if case.opset >= 11 => invalid[2].bytes = common::raw(&[f32::NAN, 0.1]),
            _ => continue,
        }
        let error = common::try_run_cuda(
            &ep,
            case.op,
            "",
            case.opset,
            &invalid,
            &outputs,
            &case.attributes,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(case.op),
            "empty output must still validate values: {error}"
        );
    }
}
