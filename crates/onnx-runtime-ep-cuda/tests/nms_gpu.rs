mod common;

use std::sync::{Mutex, MutexGuard};

use common::{Tensor, absent_input, build_graph, input, prepare_workspace, require_cuda};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMetadata, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::kernels::selection::non_max_suppression;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CUDA_COVERED_OPS, CudaExecutionProvider, nms_execution_stats, reset_nms_execution_stats,
};
use onnx_runtime_ir::{
    Attribute, DataType, Dim, Shape, SymbolId, TensorLayout, compute_contiguous_strides,
    static_shape,
};
use onnx_runtime_loader::Model;

static NMS_GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_nms_gpu() -> MutexGuard<'static, ()> {
    NMS_GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn f32_tensor(shape: &[usize], values: &[f32]) -> Tensor {
    input(DataType::Float32, shape, values)
}

fn nms_inputs(
    boxes: Tensor,
    scores: Tensor,
    max_output: Option<i64>,
    iou_threshold: Option<f32>,
    score_threshold: Option<f32>,
) -> Vec<Tensor> {
    vec![
        boxes,
        scores,
        max_output.map_or_else(
            || absent_input(DataType::Undefined),
            |value| input(DataType::Int64, &[], &[value]),
        ),
        iou_threshold.map_or_else(
            || absent_input(DataType::Undefined),
            |value| input(DataType::Float32, &[], &[value]),
        ),
        score_threshold.map_or_else(
            || absent_input(DataType::Undefined),
            |value| input(DataType::Float32, &[], &[value]),
        ),
    ]
}

fn run_nms(ep: &CudaExecutionProvider, inputs: &[Tensor], attrs: &[(&str, Attribute)]) -> Vec<i64> {
    let (graph, node_id) = build_graph(
        "NonMaxSuppression",
        "",
        10,
        inputs,
        &[(DataType::Int64, vec![0, 3])],
        attrs,
    );
    let model = Model::new(&graph);
    let shapes = inputs
        .iter()
        .map(|input| input.shape.clone())
        .collect::<Vec<_>>();
    let claim_shapes = inputs
        .iter()
        .map(|input| static_shape(input.shape.iter().copied()))
        .collect::<Vec<_>>();
    let dtypes = inputs
        .iter()
        .map(|input| {
            if input.absent {
                DataType::Undefined
            } else {
                input.dtype
            }
        })
        .collect::<Vec<_>>();
    let claim = ep.supports_op(model.graph.node(node_id), 10, &claim_shapes, &dtypes, &[]);
    assert!(
        claim.is_supported(),
        "NMS claim failed: {:?}",
        claim.reason()
    );
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &shapes, 10)
        .unwrap();

    let buffers = inputs
        .iter()
        .map(|input| {
            let buffer = ep.allocate(input.bytes.len().max(1), 256).unwrap();
            if !input.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&input.bytes, cuptr(buffer.as_ptr()))
                        .unwrap();
                }
            }
            buffer
        })
        .collect::<Vec<_>>();
    let strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let views = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .map(|((input, buffer), strides)| {
            if input.absent {
                TensorView::absent(input.dtype)
            } else {
                TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    input.dtype,
                    &input.shape,
                    strides,
                    ep.device_id(),
                )
            }
        })
        .collect::<Vec<_>>();
    let metadata = views
        .iter()
        .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
        .collect::<Vec<_>>();
    let requirement = kernel.workspace_requirement(&metadata).unwrap();
    let mut workspace = None;
    let workspace_view = prepare_workspace(ep, requirement, &mut workspace).unwrap();
    let output_metadata = kernel
        .prepare_kernel_sized_device(&views, &[true], workspace_view)
        .unwrap();
    let output_metadata = output_metadata[0].as_ref().unwrap();
    let mut output_buffer = ep
        .allocate(
            output_metadata
                .dtype
                .storage_bytes(output_metadata.shape.iter().product())
                .max(1),
            256,
        )
        .unwrap();
    let output_strides = compute_contiguous_strides(&output_metadata.shape);
    let mut output = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        output_metadata.dtype,
        &output_metadata.shape,
        &output_strides,
        ep.device_id(),
    );
    kernel
        .materialize_kernel_sized_device(&views, std::slice::from_mut(&mut output), workspace_view)
        .unwrap();
    let mut bytes = vec![0u8; output_metadata.shape.iter().product::<usize>() * 8];
    if !bytes.is_empty() {
        unsafe {
            ep.runtime()
                .dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
    }
    let result = bytes
        .chunks_exact(8)
        .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output_buffer).unwrap();
    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace).unwrap();
    }
    result
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn multi_batch_multi_class_rows_match_cpu_exactly() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let boxes = vec![
        0., 0., 1., 1., 0., 0., 0.9, 0.9, 2., 2., 3., 3., // batch 0
        0., 0., 1., 1., 0., 0., 0.9, 0.9, 2., 2., 3., 3., // batch 1
    ];
    let scores = vec![
        0.9, 0.8, 0.7, 0.1, 0.95, 0.2, // batch 0
        0.6, 0.7, 0.8, 0.99, 0.4, 0.3, // batch 1
    ];
    let inputs = nms_inputs(
        f32_tensor(&[2, 3, 4], &boxes),
        f32_tensor(&[2, 2, 3], &scores),
        Some(2),
        Some(0.5),
        Some(0.15),
    );
    let actual = run_nms(&ep, &inputs, &[]);
    let expected = non_max_suppression(&boxes, &[2, 3, 4], &scores, &[2, 2, 3], 2, 0.5, 0.15, 0)
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn center_mode_thresholds_and_ties_match_cpu() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let boxes = vec![
        0., 0., 4., 2., //
        1.5, 0., 4., 2., //
        10., 10., 2., 2., //
        20., 20., 2., 2.,
    ];
    let scores = vec![0.9, 0.8, 0.7, 0.6];
    let inputs = nms_inputs(
        f32_tensor(&[1, 4, 4], &boxes),
        f32_tensor(&[1, 1, 4], &scores),
        Some(4),
        Some(0.4),
        Some(0.65),
    );
    let attrs = [("center_point_box", Attribute::Int(1))];
    let actual = run_nms(&ep, &inputs, &attrs);
    let expected = non_max_suppression(&boxes, &[1, 4, 4], &scores, &[1, 1, 4], 4, 0.4, 0.65, 1)
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(&actual[3..], &[0, 0, 2]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn equal_score_ties_keep_lower_indices() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let boxes = vec![
        0., 0., 1., 1., //
        2., 2., 3., 3., //
        4., 4., 5., 5.,
    ];
    let scores = vec![0.5, 0.5, 0.5];
    let actual = run_nms(
        &ep,
        &nms_inputs(
            f32_tensor(&[1, 3, 4], &boxes),
            f32_tensor(&[1, 1, 3], &scores),
            Some(3),
            Some(0.5),
            Some(0.0),
        ),
        &[],
    );
    assert_eq!(actual, [0, 0, 0, 0, 0, 1, 0, 0, 2]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn zero_default_and_no_selection_outputs_are_empty() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let boxes = f32_tensor(&[1, 1, 4], &[0., 0., 1., 1.]);
    let scores = f32_tensor(&[1, 1, 1], &[0.9]);
    assert!(
        run_nms(
            &ep,
            &nms_inputs(boxes.clone(), scores.clone(), None, None, None),
            &[]
        )
        .is_empty()
    );
    assert!(
        run_nms(
            &ep,
            &nms_inputs(boxes.clone(), scores.clone(), Some(0), Some(0.5), None),
            &[]
        )
        .is_empty()
    );
    assert!(
        run_nms(
            &ep,
            &nms_inputs(boxes, scores, Some(1), Some(0.5), Some(1.0)),
            &[]
        )
        .is_empty()
    );
    assert!(
        run_nms(
            &ep,
            &nms_inputs(
                f32_tensor(&[1, 0, 4], &[]),
                f32_tensor(&[1, 1, 0], &[]),
                Some(4),
                Some(0.5),
                None,
            ),
            &[]
        )
        .is_empty()
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn policy_moves_only_count_and_runs_prepare_once() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    reset_nms_execution_stats();
    let boxes = (0..32)
        .flat_map(|index| {
            let x = index as f32 * 2.0;
            [x, x, x + 1.0, x + 1.0]
        })
        .collect::<Vec<_>>();
    let scores = (0..2 * 32)
        .map(|index| 1.0 - index as f32 / 100.0)
        .collect::<Vec<_>>();
    let result = run_nms(
        &ep,
        &nms_inputs(
            f32_tensor(&[1, 32, 4], &boxes),
            f32_tensor(&[1, 2, 32], &scores),
            Some(8),
            Some(0.5),
            Some(0.0),
        ),
        &[],
    );
    assert_eq!(result.len(), 16 * 3);
    let stats = nms_execution_stats();
    assert_eq!(stats.prepare_launches, 1);
    assert_eq!(stats.count_launches, 1);
    assert_eq!(stats.materialize_launches, 1);
    assert_eq!(stats.d2h_bytes, 8);
    assert_eq!(stats.full_input_d2h_bytes, 0);
    assert_eq!(stats.workspace_bytes, 272);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn unsupported_dtype_strided_oversize_and_capture_decline() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    assert!(CUDA_COVERED_OPS.contains(&"NonMaxSuppression"));
    let inputs = nms_inputs(
        f32_tensor(&[1, 4, 4], &[0.; 16]),
        f32_tensor(&[1, 1, 4], &[0.; 4]),
        Some(2),
        None,
        None,
    );
    let (graph, id) = build_graph(
        "NonMaxSuppression",
        "",
        10,
        &inputs,
        &[(DataType::Int64, vec![0, 3])],
        &[],
    );
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            10,
            &[static_shape([1, 4, 4]), static_shape([1, 1, 4])],
            &[DataType::Float16, DataType::Float32],
            &[],
        ),
        KernelMatch::Unsupported { .. }
    ));
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            10,
            &[static_shape([1, 4, 4]), static_shape([1, 1, 4])],
            &[DataType::Float32, DataType::Float32],
            &[
                TensorLayout::strided(vec![32, 8, 1]),
                TensorLayout::contiguous()
            ],
        ),
        KernelMatch::Unsupported { .. }
    ));
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            10,
            &[static_shape([1, 257, 4]), static_shape([1, 1, 257]),],
            &[DataType::Float32, DataType::Float32],
            &[],
        ),
        KernelMatch::Unsupported { .. }
    ));
    let kernel = ep
        .get_kernel(model.graph.node(id), &[vec![1, 4, 4], vec![1, 1, 4]], 10)
        .unwrap();
    assert!(!kernel.capture_support().is_supported());
    assert!(
        kernel
            .capture_support()
            .reason()
            .unwrap()
            .contains("8-byte")
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn omitted_score_threshold_matches_negative_infinity_cpu_default() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let boxes = vec![
        0., 0., 1., 1., //
        2., 2., 3., 3.,
    ];
    let scores = vec![f32::MIN, f32::NEG_INFINITY];
    let omitted_inputs = nms_inputs(
        f32_tensor(&[1, 2, 4], &boxes),
        f32_tensor(&[1, 1, 2], &scores),
        Some(2),
        Some(0.5),
        None,
    );
    let explicit_inputs = nms_inputs(
        f32_tensor(&[1, 2, 4], &boxes),
        f32_tensor(&[1, 1, 2], &scores),
        Some(2),
        Some(0.5),
        Some(f32::NEG_INFINITY),
    );
    let omitted = run_nms(&ep, &omitted_inputs, &[]);
    let explicit = run_nms(&ep, &explicit_inputs, &[]);
    let expected = non_max_suppression(
        &boxes,
        &[1, 2, 4],
        &scores,
        &[1, 1, 2],
        2,
        0.5,
        f32::NEG_INFINITY,
        0,
    )
    .unwrap()
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    assert_eq!(omitted, expected);
    assert_eq!(explicit, expected);
    assert_eq!(
        omitted,
        [0, 0, 0],
        "f32::MIN must pass an omitted -infinity threshold, while -infinity itself must not"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn signed_zero_scores_follow_cpu_total_order() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let boxes = vec![
        0., 0., 1., 1., //
        0., 0., 1., 1.,
    ];
    let scores = vec![-0.0, 0.0];
    let actual = run_nms(
        &ep,
        &nms_inputs(
            f32_tensor(&[1, 2, 4], &boxes),
            f32_tensor(&[1, 1, 2], &scores),
            Some(2),
            Some(0.5),
            None,
        ),
        &[],
    );
    let expected = non_max_suppression(
        &boxes,
        &[1, 2, 4],
        &scores,
        &[1, 1, 2],
        2,
        0.5,
        f32::NEG_INFINITY,
        0,
    )
    .unwrap()
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(
        actual,
        [0, 0, 1],
        "+0 at the higher index must sort before -0 at the lower index"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn optional_scalar_shapes_decline_before_kernel_creation() {
    let _lock = lock_nms_gpu();
    let ep = require_cuda();
    let inputs = nms_inputs(
        f32_tensor(&[1, 4, 4], &[0.; 16]),
        f32_tensor(&[1, 1, 4], &[0.; 4]),
        Some(2),
        Some(0.5),
        Some(0.0),
    );
    let (graph, id) = build_graph(
        "NonMaxSuppression",
        "",
        10,
        &inputs,
        &[(DataType::Int64, vec![0, 3])],
        &[],
    );
    let model = Model::new(&graph);
    let dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Int64,
        DataType::Float32,
        DataType::Float32,
    ];
    let valid = vec![
        static_shape([1, 4, 4]),
        static_shape([1, 1, 4]),
        static_shape([]),
        static_shape([]),
        static_shape([]),
    ];
    assert!(
        ep.supports_op(model.graph.node(id), 10, &valid, &dtypes, &[])
            .is_supported(),
        "known rank-0 optional scalars must remain claimable"
    );

    let symbolic: Shape = vec![Dim::Symbolic(SymbolId(77))];
    for (slot, invalid_shape) in [
        (2usize, static_shape([1])),
        (3usize, static_shape([0])),
        (4usize, symbolic),
        (4usize, static_shape([1, 1])),
    ] {
        let mut shapes = valid.clone();
        shapes[slot] = invalid_shape;
        let claim = ep.supports_op(model.graph.node(id), 10, &shapes, &dtypes, &[]);
        assert!(
            matches!(claim, KernelMatch::Unsupported { .. }),
            "optional input {slot} shape {:?} must decline at claim time",
            shapes[slot]
        );
        let reason = claim.reason().unwrap();
        assert!(
            reason.contains("rank-0") || reason.contains("scalar"),
            "claim reason must name the scalar-shape contract: {reason}"
        );
    }
}
