#![cfg(test)]

use super::*;
use crate::runtime::GraphDeviceAllocation;
use onnx_runtime_ep_api::{DeviceId, DevicePtrMut};

fn inputs() -> [TensorMetadata<'static>; 4] {
    [
        TensorMetadata::new(DataType::BFloat16, &[2, 5], true),
        TensorMetadata::new(DataType::BFloat16, &[5], true),
        TensorMetadata::new(DataType::Float32, &[5], true),
        TensorMetadata::new(DataType::BFloat16, &[5], true),
    ]
}
fn outputs() -> [OutputMetadata<'static>; 4] {
    [
        OutputMetadata::tensor(DataType::BFloat16, &[2, 5]),
        OutputMetadata::Absent,
        OutputMetadata::tensor(DataType::Float32, &[2]),
        OutputMetadata::tensor(DataType::BFloat16, &[2, 5]),
    ]
}
fn staged(route: SkipNormRoute) -> WorkspaceLayout<8> {
    match route {
        SkipNormRoute::Bf16Staged(layout) => layout,
        other => panic!("expected staging, got {other:?}"),
    }
}

#[test]
fn skip_norm_workspace_partitions_one_peak_with_holes_and_mixed_parameter_dtypes() {
    let route = skip_norm_route(&inputs(), &outputs(), true).unwrap();
    let requirement = route.requirement();
    assert_eq!((requirement.bytes, requirement.alignment), (216, 16));
    assert_eq!(requirement.lifetime, WorkspaceLifetime::SessionPersistent);
    let layout = staged(route);
    let regions = layout
        .bind(Some(WorkspaceView::new(
            DevicePtrMut(0x1000usize as *mut _),
            512,
        )))
        .unwrap();
    assert_eq!(
        regions.map(|view| view.bytes()),
        [40, 20, 0, 20, 40, 0, 8, 40]
    );
    assert_eq!(
        regions.map(|view| view.ptr().0 as usize - 0x1000),
        [0, 48, 80, 80, 112, 160, 160, 176]
    );
    for pair in regions
        .iter()
        .filter(|view| view.bytes() != 0)
        .collect::<Vec<_>>()
        .windows(2)
    {
        assert!(pair[0].ptr().0 as usize + pair[0].bytes() <= pair[1].ptr().0 as usize);
    }
}

#[test]
fn skip_norm_workspace_native_and_direct_are_zero_but_fallback_is_prepared() {
    let mut input = inputs();
    input[1].shape = &[2, 5];
    input[3].present = false;
    let mut output = outputs();
    output[1] = OutputMetadata::tensor(DataType::BFloat16, &[2]);
    assert!(matches!(
        skip_norm_route(&input, &output, true).unwrap(),
        SkipNormRoute::Bf16Native
    ));
    assert_eq!(
        skip_norm_route(&input, &output, true)
            .unwrap()
            .requirement(),
        WorkspaceRequirement::NONE
    );
    assert!(
        skip_norm_route(&input, &output, false)
            .unwrap()
            .requirement()
            .bytes
            > 0
    );
    input[0].dtype = DataType::Float16;
    assert_eq!(
        skip_norm_route(&input, &output, false)
            .unwrap()
            .requirement(),
        WorkspaceRequirement::NONE
    );
    input[0].dtype = DataType::Float32;
    assert_eq!(
        skip_norm_route(&input, &output, true)
            .unwrap()
            .requirement(),
        WorkspaceRequirement::NONE
    );
}

#[test]
fn skip_norm_workspace_derives_unresolved_counts_without_inventing_a_scalar() {
    let mut output = outputs();
    for value in &mut output {
        if let OutputMetadata::Tensor { dtype, .. } = value {
            *value = OutputMetadata::Unresolved { dtype: *dtype };
        }
    }
    let inferred = skip_norm_route(&inputs(), &output, true)
        .unwrap()
        .requirement();
    assert_eq!(
        inferred,
        skip_norm_route(&inputs(), &outputs(), true)
            .unwrap()
            .requirement()
    );
    output[2] = OutputMetadata::tensor(DataType::Float32, &[]);
    let error = skip_norm_route(&inputs(), &output, true)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("output slot 2") && error.contains("2 elements"),
        "{error}"
    );
}

#[test]
fn skip_norm_workspace_rejects_unfunded_or_misaligned_views_before_device_access() {
    let layout = staged(skip_norm_route(&inputs(), &outputs(), true).unwrap());
    for view in [
        None,
        Some(WorkspaceView::new(DevicePtrMut(0x1000usize as *mut _), 215)),
        Some(WorkspaceView::new(DevicePtrMut(0x1001usize as *mut _), 216)),
    ] {
        let error = layout.bind(view).unwrap_err().to_string();
        assert!(error.contains("216 bytes aligned to 16"), "{error}");
    }
}

#[test]
fn skip_norm_workspace_zero_groups_are_empty_and_overflow_is_actionable() {
    let mut input = inputs();
    input[0].shape = &[0, 5];
    let output = [OutputMetadata::tensor(DataType::BFloat16, &[0, 5])];
    let layout = staged(skip_norm_route(&input, &output, true).unwrap());
    assert_eq!(layout.bytes(), 0);
    layout.bind(None).unwrap();
    input[0].shape = &[usize::MAX, 5];
    let output = [OutputMetadata::Unresolved {
        dtype: DataType::BFloat16,
    }];
    let error = skip_norm_route(&input, &output, false)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("element count exceeds usize") && error.contains("reduce the shape"),
        "{error}"
    );
    input[0].shape = &[usize::MAX / 2];
    input[1].shape = &[1];
    input[2].shape = input[0].shape;
    input[3].shape = input[0].shape;
    let error = skip_norm_route(&input, &output, false)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("staging byte count exceeds usize"),
        "{error}"
    );
}

#[test]
fn skip_norm_workspace_rejects_bad_optional_facts_and_required_holes() {
    let input = inputs();
    let mut output = outputs();
    output[0] = OutputMetadata::Absent;
    assert!(
        skip_norm_route(&input, &output, true)
            .unwrap_err()
            .to_string()
            .contains("required output slot 0")
    );
    output = outputs();
    output[2] = OutputMetadata::tensor(DataType::Int64, &[2]);
    assert!(
        skip_norm_route(&input, &output, true)
            .unwrap_err()
            .to_string()
            .contains("output slot 2")
    );
    let mut input = input;
    input[2].present = false;
    assert!(
        skip_norm_route(&input, &outputs(), false)
            .unwrap_err()
            .to_string()
            .contains("required input slot 2")
    );
}

#[test]
fn skip_norm_workspace_stat_widths_are_independent_and_absent_helpers_never_write() {
    let make = |dtype| {
        TensorMut::new(
            DevicePtrMut(std::ptr::null_mut()),
            dtype,
            &[],
            &[],
            DeviceId::cpu(),
        )
    };
    let mut outputs = [
        make(DataType::BFloat16),
        make(DataType::Float32),
        make(DataType::BFloat16),
    ];
    assert_eq!(skip_stat_bf16_mask(&outputs), 2);
    outputs[1].dtype = DataType::BFloat16;
    outputs[2].dtype = DataType::Float32;
    assert_eq!(skip_stat_bf16_mask(&outputs), 1);
    outputs[1].absent = true;
    outputs[2].absent = true;
    outputs[1].dtype = DataType::Int64;
    assert_eq!(skip_stat_bf16_mask(&outputs), 0);
    assert_eq!(
        optional_bf16_stat_ptr(
            "SkipSimplifiedLayerNormalization",
            "Mean",
            &mut outputs,
            1,
            27
        )
        .unwrap(),
        0
    );
    assert_eq!(
        optional_half_stat_ptr(
            "SkipSimplifiedLayerNormalization",
            "Mean",
            &mut outputs,
            1,
            27
        )
        .unwrap(),
        0
    );
    assert_eq!(
        optional_out_ptr(
            "SkipSimplifiedLayerNormalization",
            "Mean",
            &mut outputs,
            1,
            27
        )
        .unwrap(),
        0
    );
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
fn skip_norm_workspace_bf16_broadcast_holes_capture_and_failure_revoke_warm_proof() {
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::compute_contiguous_strides;
    let ep =
        crate::CudaExecutionProvider::new(0).expect("CUDA workspace regression requires a device");
    let runtime = ep.runtime();
    let upload = |values: &[f32], dtype: DataType| {
        let bytes = if dtype == DataType::BFloat16 {
            values
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_ne_bytes())
                .collect::<Vec<_>>()
        } else {
            values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let owner = GraphDeviceAllocation::allocate(
            runtime,
            bytes.len(),
            onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
        )
        .unwrap();
        unsafe {
            runtime.htod(&bytes, owner.ptr()).unwrap();
        }
        owner
    };
    let x = [-0.5, 0.25, 1.0, 2.0, -1.0, 0.5, -0.75, 1.5, 0.0, 0.25];
    let skip = [0.25, -0.5, 0.75, 0.5, -0.25];
    let gamma = [1.0, 0.5, 1.25, -0.5, 2.0];
    let bias = [0.125, 0.25, -0.5, 0.0, 0.5];
    let facts = inputs();
    let strides: [_; 4] =
        std::array::from_fn(|index| compute_contiguous_strides(facts[index].shape));
    let input_owners = [
        upload(&x, DataType::BFloat16),
        upload(&skip, DataType::BFloat16),
        upload(&gamma, DataType::Float32),
        upload(&bias, DataType::BFloat16),
    ];
    let views: [_; 4] = std::array::from_fn(|index| {
        TensorView::new(
            DevicePtr(raw_ptr(input_owners[index].ptr()).cast_const()),
            facts[index].dtype,
            facts[index].shape,
            &strides[index],
            ep.device_id(),
        )
    });
    let output_owners = [
        GraphDeviceAllocation::allocate(
            runtime,
            20,
            onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
        )
        .unwrap(),
        GraphDeviceAllocation::allocate(
            runtime,
            8,
            onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
        )
        .unwrap(),
        GraphDeviceAllocation::allocate(
            runtime,
            20,
            onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
        )
        .unwrap(),
    ];
    let mut node = onnx_runtime_ir::Node::new_with_output_slots(
        onnx_runtime_ir::NodeId(0),
        "SkipSimplifiedLayerNormalization",
        (0..4)
            .map(|index| Some(onnx_runtime_ir::ValueId(index)))
            .collect(),
        vec![
            Some(onnx_runtime_ir::ValueId(4)),
            None,
            Some(onnx_runtime_ir::ValueId(5)),
            Some(onnx_runtime_ir::ValueId(6)),
        ],
    );
    node.domain = "com.microsoft".into();
    let shapes = facts.map(|input| onnx_runtime_ir::static_shape(input.shape.iter().copied()));
    let dtypes = facts.map(|input| input.dtype);
    let claim = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
    assert!(
        matches!(
            claim,
            onnx_runtime_ep_api::KernelMatch::Supported {
                workspace_planning: onnx_runtime_ep_api::WorkspacePlanning::Metadata,
                ..
            }
        ),
        "{claim:?}"
    );
    let concrete_shapes = facts.map(|input| input.shape.to_vec());
    let kernel = ep.get_kernel(&node, &concrete_shapes, 1).unwrap();
    let requirement = kernel
        .workspace_requirement(&facts, &outputs())
        .unwrap()
        .require_ready()
        .unwrap();
    let mut workspace = ep
        .allocate_workspace(
            requirement.bytes as usize,
            requirement.alignment,
            requirement.role(),
        )
        .unwrap();
    let scratch = WorkspaceView::new(
        DevicePtrMut(workspace.as_mut_ptr()),
        requirement.bytes as usize,
    );
    let execute = |workspace| {
        let mut outputs = [
            TensorMut::new(
                DevicePtrMut(raw_ptr(output_owners[0].ptr())),
                DataType::BFloat16,
                &[2, 5],
                &[5, 1],
                ep.device_id(),
            ),
            TensorMut::new(
                DevicePtrMut(std::ptr::null_mut()),
                DataType::Int64,
                &[],
                &[],
                ep.device_id(),
            )
            .mark_absent(),
            TensorMut::new(
                DevicePtrMut(raw_ptr(output_owners[1].ptr())),
                DataType::Float32,
                &[2],
                &[1],
                ep.device_id(),
            ),
            TensorMut::new(
                DevicePtrMut(raw_ptr(output_owners[2].ptr())),
                DataType::BFloat16,
                &[2, 5],
                &[5, 1],
                ep.device_id(),
            ),
        ];
        kernel.execute_with_workspace(&views, &mut outputs, workspace)
    };
    execute(Some(scratch)).unwrap();
    runtime.synchronize().unwrap();
    let read = || {
        output_owners
            .iter()
            .map(|owner| {
                let mut bytes = vec![0; owner.bytes()];
                unsafe {
                    runtime.dtoh(&mut bytes, owner.ptr()).unwrap();
                }
                bytes
            })
            .collect::<Vec<_>>()
    };
    let expected = read();
    for row in 0..2 {
        let residual: [f32; 5] = std::array::from_fn(|j| x[row * 5 + j] + skip[j] + bias[j]);
        let inv = 1.0 / (residual.iter().map(|x| x * x).sum::<f32>() / 5.0 + 1e-5).sqrt();
        let got = f32::from_ne_bytes(expected[1][row * 4..row * 4 + 4].try_into().unwrap());
        assert!((got - inv).abs() < 2e-6);
        for j in 0..5 {
            let offset = (row * 5 + j) * 2;
            for (output, reference) in [(0, residual[j] * inv * gamma[j]), (2, residual[j])] {
                let got = half::bf16::from_bits(u16::from_ne_bytes(
                    expected[output][offset..offset + 2].try_into().unwrap(),
                ))
                .to_f32();
                assert!((got - half::bf16::from_f32(reference).to_f32()).abs() <= 0.015625);
            }
        }
    }
    assert!(kernel.cuda_graph_compatible());
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    assert!(
        execute(Some(scratch))
            .unwrap_err()
            .to_string()
            .contains("real workspace owner")
    );
    runtime.abort_graph_capture().unwrap();
    assert!(!kernel.cuda_graph_compatible());
    execute(Some(scratch)).unwrap();
    runtime.synchronize().unwrap();
    let allocations = runtime.allocation_counts();
    let resource = workspace
        .device_graph_resource()
        .unwrap()
        .expect("workspace must expose a real caller pin");
    runtime
        .begin_graph_capture_with_resources(&[kernel.as_ref()], vec![resource])
        .unwrap();
    execute(Some(scratch)).unwrap();
    runtime.end_graph_capture().unwrap();
    unsafe {
        runtime
            .htod(&vec![0xa5; scratch.bytes()], cuptr(scratch.ptr().0))
            .unwrap();
    }
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    assert_eq!(
        read(),
        expected,
        "captured casts must rewrite dirty caller scratch"
    );
    assert_eq!(runtime.allocation_counts(), allocations);
    assert!(runtime.reset_graph().unwrap());
    assert!(
        execute(None)
            .unwrap_err()
            .to_string()
            .contains("complete caller workspace")
    );
    assert!(!kernel.cuda_graph_compatible());
    ep.deallocate_workspace(workspace).unwrap();
}
