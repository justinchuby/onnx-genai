#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! CUDA conformance tests for movement/construction operators and `Where`.

use onnx_runtime_ep_api::{
    CaptureSupport, DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut,
    TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

#[derive(Clone)]
struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw<T: Copy>(values: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn tensor<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: raw(values),
    }
}

fn require_cuda() -> CudaExecutionProvider {
    CudaExecutionProvider::new_default().expect("CUDA runtime must be available")
}

fn run(
    ep: &CudaExecutionProvider,
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let value = graph.create_named_value(
                format!("input_{i}"),
                input.dtype,
                static_shape(input.shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{i}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.iter().copied().map(Some).collect(),
        output_values.clone(),
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &[], opset)
        .unwrap();

    let input_buffers = inputs
        .iter()
        .map(|input| {
            let buffer = ep.allocate(input.bytes.len(), 256).unwrap();
            if !input.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&input.bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            buffer
        })
        .collect::<Vec<_>>();
    let input_strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
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

    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| {
            ep.allocate(dtype.storage_bytes(shape.iter().product()), 256)
                .unwrap()
        })
        .collect::<Vec<DeviceBuffer>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), buffer), strides)| {
            TensorMut::new(
                DevicePtrMut(buffer.as_mut_ptr()),
                *dtype,
                shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    kernel
        .execute(
            &input_views,
            &mut output_views.into_iter().collect::<Vec<_>>(),
        )
        .unwrap();

    let result = outputs
        .iter()
        .zip(&output_buffers)
        .map(|((dtype, shape), buffer)| {
            let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
            if !bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            bytes
        })
        .collect();
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
    result
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn concat_negative_axis_and_multiple_inputs() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Float32, &[2, 1], &[1_f32, 2.]),
        tensor(DataType::Float32, &[2, 2], &[3_f32, 4., 5., 6.]),
        tensor(DataType::Float32, &[2, 1], &[7_f32, 8.]),
    ];
    let out = run(
        &ep,
        "Concat",
        13,
        &inputs,
        &[(DataType::Float32, vec![2, 4])],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(f32s(&out[0]), vec![1., 3., 4., 7., 2., 5., 6., 8.]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn expand_right_aligned_broadcast() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Int64, &[3], &[7_i64, 8, 9]),
        tensor(DataType::Int64, &[2], &[2_i64, 1]),
    ];
    let out = run(
        &ep,
        "Expand",
        13,
        &inputs,
        &[(DataType::Int64, vec![2, 3])],
        &[],
    );
    assert_eq!(out[0], raw(&[7_i64, 8, 9, 7, 8, 9]));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn reshape_preserves_dtype_agnostic_bytes() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Int64, &[2, 3], &[1_i64, 2, 3, 4, 5, 6]),
        tensor(DataType::Int64, &[2], &[3_i64, 2]),
    ];
    let out = run(
        &ep,
        "Reshape",
        13,
        &inputs,
        &[(DataType::Int64, vec![3, 2])],
        &[],
    );
    assert_eq!(out[0], raw(&[1_i64, 2, 3, 4, 5, 6]));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn slice_multi_axis_negative_axis_and_step() {
    let ep = require_cuda();
    let data = (0..24).map(|v| v as f32).collect::<Vec<_>>();
    let inputs = [
        tensor(DataType::Float32, &[2, 3, 4], &data),
        tensor(DataType::Int64, &[2], &[2_i64, 3]),
        tensor(DataType::Int64, &[2], &[1_i64, 0]),
        tensor(DataType::Int64, &[2], &[1_i64, -1]),
        tensor(DataType::Int64, &[2], &[-1_i64, -2]),
    ];
    let out = run(
        &ep,
        "Slice",
        13,
        &inputs,
        &[(DataType::Float32, vec![2, 1, 2])],
        &[],
    );
    assert_eq!(f32s(&out[0]), vec![11., 9., 23., 21.]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_negative_axis_via_split_input() {
    let ep = require_cuda();
    let inputs = [
        tensor(
            DataType::Float32,
            &[2, 4],
            &[1_f32, 2., 3., 4., 5., 6., 7., 8.],
        ),
        tensor(DataType::Int64, &[2], &[1_i64, 3]),
    ];
    let out = run(
        &ep,
        "Split",
        13,
        &inputs,
        &[
            (DataType::Float32, vec![2, 1]),
            (DataType::Float32, vec![2, 3]),
        ],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(f32s(&out[0]), vec![1., 5.]);
    assert_eq!(f32s(&out[1]), vec![2., 3., 4., 6., 7., 8.]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn squeeze_axes_input_preserves_bytes() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Int64, &[1, 3, 1], &[7_i64, 8, 9]),
        tensor(DataType::Int64, &[2], &[0_i64, 2]),
    ];
    let out = run(
        &ep,
        "Squeeze",
        13,
        &inputs,
        &[(DataType::Int64, vec![3])],
        &[],
    );
    assert_eq!(out[0], raw(&[7_i64, 8, 9]));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn tile_multi_axis_repeats() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Float32, &[2, 1], &[1_f32, 2.]),
        tensor(DataType::Int64, &[2], &[2_i64, 3]),
    ];
    let out = run(
        &ep,
        "Tile",
        13,
        &inputs,
        &[(DataType::Float32, vec![4, 3])],
        &[],
    );
    assert_eq!(
        f32s(&out[0]),
        vec![1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2.]
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn transpose_explicit_three_axis_permutation() {
    let ep = require_cuda();
    let inputs = [tensor(
        DataType::Float32,
        &[2, 1, 3],
        &[1_f32, 2., 3., 4., 5., 6.],
    )];
    let out = run(
        &ep,
        "Transpose",
        13,
        &inputs,
        &[(DataType::Float32, vec![3, 2, 1])],
        &[("perm", Attribute::Ints(vec![2, 0, 1]))],
    );
    assert_eq!(f32s(&out[0]), vec![1., 4., 2., 5., 3., 6.]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn unsqueeze_multiple_axes_input_preserves_bytes() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Int64, &[2], &[5_i64, 9]),
        tensor(DataType::Int64, &[2], &[0_i64, 2]),
    ];
    let out = run(
        &ep,
        "Unsqueeze",
        13,
        &inputs,
        &[(DataType::Int64, vec![1, 2, 1])],
        &[],
    );
    assert_eq!(out[0], raw(&[5_i64, 9]));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn where_broadcasts_all_three_inputs() {
    let ep = require_cuda();
    let inputs = [
        tensor(DataType::Bool, &[2, 1], &[1_u8, 0]),
        tensor(DataType::Int64, &[1, 3], &[1_i64, 2, 3]),
        tensor(DataType::Int64, &[], &[9_i64]),
    ];
    let out = run(
        &ep,
        "Where",
        13,
        &inputs,
        &[(DataType::Int64, vec![2, 3])],
        &[],
    );
    assert_eq!(out[0], raw(&[1_i64, 2, 3, 9, 9, 9]));
}

/// Build a `Split` kernel for a single data input of `data_shape` producing
/// `output_shapes`, with an optional runtime split-sizes input. Returns the
/// kernel so a test can inspect its device-graph capture eligibility. The
/// resolved input shapes are forwarded to `get_kernel` so the kernel can plan
/// the static, capturable form at build time exactly as the executor does.
fn build_split_kernel(
    ep: &CudaExecutionProvider,
    data_shape: &[usize],
    output_shapes: &[Vec<usize>],
    attrs: &[(&str, Attribute)],
    runtime_split_shape: Option<&[usize]>,
) -> Box<dyn Kernel> {
    let opset = 13;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let data = graph.create_named_value(
        "data",
        DataType::Float32,
        static_shape(data_shape.iter().copied()),
    );
    graph.add_input(data);
    let mut node_inputs = vec![Some(data)];
    let mut input_shapes = vec![data_shape.to_vec()];
    if let Some(split_shape) = runtime_split_shape {
        let split = graph.create_named_value(
            "split_sizes",
            DataType::Int64,
            static_shape(split_shape.iter().copied()),
        );
        graph.add_input(split);
        node_inputs.push(Some(split));
        input_shapes.push(split_shape.to_vec());
    }
    let outputs = output_shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            graph.create_named_value(
                format!("output_{i}"),
                DataType::Float32,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(NodeId(0), "Split", node_inputs, outputs.clone());
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in outputs {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    ep.get_kernel(model.graph.node(node_id), &input_shapes, opset)
        .unwrap()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_static_even_num_outputs_is_capture_supported() {
    // The GLM-4 fused-MLP activation split: single data input, num_outputs=2,
    // axis=-1, statically resolved even halves. This must be capturable.
    let ep = require_cuda();
    let kernel = build_split_kernel(
        &ep,
        &[1, 4, 8],
        &[vec![1, 4, 4], vec![1, 4, 4]],
        &[
            ("axis", Attribute::Int(-1)),
            ("num_outputs", Attribute::Int(2)),
        ],
        None,
    );
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_static_explicit_split_attribute_is_capture_supported() {
    // Explicit, uneven but statically known split sizes are also capturable.
    let ep = require_cuda();
    let kernel = build_split_kernel(
        &ep,
        &[2, 5],
        &[vec![2, 2], vec![2, 3]],
        &[
            ("axis", Attribute::Int(1)),
            ("split", Attribute::Ints(vec![2, 3])),
        ],
        None,
    );
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_dynamic_runtime_sizes_is_not_capture_supported() {
    // A wired runtime split-sizes input keeps the host-read-plus-synchronize
    // path and must never be admitted to capture.
    let ep = require_cuda();
    let kernel = build_split_kernel(
        &ep,
        &[2, 4],
        &[vec![2, 1], vec![2, 3]],
        &[("axis", Attribute::Int(-1))],
        Some(&[2]),
    );
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Unsupported { .. }
    ));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_static_even_num_outputs_matches_eager_bytes() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    let input_shape = [1, 2, 4];
    let output_shapes = [vec![1, 2, 2], vec![1, 2, 2]];
    let initial = raw(&[1_f32, 2., 3., 4., 5., 6., 7., 8.]);
    let mutated = raw(&[9_f32, 10., 11., 12., 13., 14., 15., 16.]);
    let kernel = build_split_kernel(
        &ep,
        &input_shape,
        &output_shapes,
        &[
            ("axis", Attribute::Int(-1)),
            ("num_outputs", Attribute::Int(2)),
        ],
        None,
    );
    assert_eq!(
        kernel.capture_support(),
        CaptureSupport::Supported,
        "concrete input shapes must select the static Split plan"
    );

    let input_buffer = ep.allocate(initial.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&initial, cuptr(input_buffer.as_ptr()))
            .unwrap();
    };
    let input_strides = compute_contiguous_strides(&input_shape);
    let input = TensorView::new(
        DevicePtr(input_buffer.as_ptr()),
        DataType::Float32,
        &input_shape,
        &input_strides,
        device,
    );
    let mut output_buffers = output_shapes
        .iter()
        .map(|shape| {
            ep.allocate(DataType::Float32.storage_bytes(shape.iter().product()), 256)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let output_strides = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();

    macro_rules! execute {
        () => {{
            let mut outputs = output_buffers
                .iter_mut()
                .zip(&output_shapes)
                .zip(&output_strides)
                .map(|((buffer, shape), strides)| {
                    TensorMut::new(
                        DevicePtrMut(buffer.as_mut_ptr()),
                        DataType::Float32,
                        shape,
                        strides,
                        device,
                    )
                })
                .collect::<Vec<_>>();
            kernel.execute(&[input], &mut outputs).unwrap();
        }};
    }

    execute!();
    unsafe {
        runtime
            .htod(&mutated, cuptr(input_buffer.as_ptr()))
            .unwrap();
    };
    execute!();
    let eager = output_buffers
        .iter()
        .map(|buffer| {
            let mut bytes = vec![0; 4 * 4];
            unsafe {
                runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr())).unwrap();
            };
            bytes
        })
        .collect::<Vec<_>>();
    assert_eq!(eager[0], raw(&[9_f32, 10., 13., 14.]));
    assert_eq!(eager[1], raw(&[11_f32, 12., 15., 16.]));

    unsafe {
        runtime
            .htod(&initial, cuptr(input_buffer.as_ptr()))
            .unwrap();
    };
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute!();
    runtime.end_graph_capture().unwrap();
    assert!(runtime.has_graph_executable().unwrap());

    unsafe {
        runtime
            .htod(&mutated, cuptr(input_buffer.as_ptr()))
            .unwrap();
    };
    runtime.replay_graph().unwrap();
    let replayed = output_buffers
        .iter()
        .map(|buffer| {
            let mut bytes = vec![0; 4 * 4];
            unsafe {
                runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr())).unwrap();
            };
            bytes
        })
        .collect::<Vec<_>>();
    assert_eq!(replayed, eager);
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(input_buffer).unwrap();
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_constant_input_warms_and_captures() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // DeepSeek-V2-Lite decode: [B,S,16,192] -> [B,S,16,128] + [B,S,16,64].
    let data_shape = [1, 1, 16, 192];
    let split_shape = [2];
    let output_shapes = [vec![1, 1, 16, 128], vec![1, 1, 16, 64]];
    let data = (0..16 * 192).map(|value| value as f32).collect::<Vec<_>>();
    let data_bytes = raw(&data);
    let split_bytes = raw(&[128_i64, 64]);
    let mut kernel = build_split_kernel(
        &ep,
        &data_shape,
        &output_shapes,
        &[("axis", Attribute::Int(-1))],
        Some(&split_shape),
    );
    kernel.set_constant_inputs(&[false, true]);
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Unsupported { .. }
    ));

    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let split_buffer = ep.allocate(split_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&split_bytes, cuptr(split_buffer.as_ptr()))
            .unwrap();
    }
    let data_strides = compute_contiguous_strides(&data_shape);
    let split_strides = compute_contiguous_strides(&split_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &data_shape,
            &data_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(split_buffer.as_ptr()),
            DataType::Int64,
            &split_shape,
            &split_strides,
            device,
        ),
    ];
    let mut output_buffers = output_shapes
        .iter()
        .map(|shape| {
            ep.allocate(DataType::Float32.storage_bytes(shape.iter().product()), 256)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let output_strides = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();

    let mut execute = || {
        let mut outputs = output_buffers
            .iter_mut()
            .zip(&output_shapes)
            .zip(&output_strides)
            .map(|((buffer, shape), strides)| {
                TensorMut::new(
                    DevicePtrMut(buffer.as_mut_ptr()),
                    DataType::Float32,
                    shape,
                    strides,
                    device,
                )
            })
            .collect::<Vec<_>>();
        kernel.execute(&inputs, &mut outputs).unwrap();
    };
    execute();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();

    let mut first = vec![0; 16 * 128 * std::mem::size_of::<f32>()];
    let mut second = vec![0; 16 * 64 * std::mem::size_of::<f32>()];
    unsafe {
        runtime
            .dtoh(&mut first, cuptr(output_buffers[0].as_ptr()))
            .unwrap();
        runtime
            .dtoh(&mut second, cuptr(output_buffers[1].as_ptr()))
            .unwrap();
    }
    let expected_first = data
        .chunks_exact(192)
        .flat_map(|head| &head[..128])
        .copied()
        .collect::<Vec<_>>();
    let expected_second = data
        .chunks_exact(192)
        .flat_map(|head| &head[128..])
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(first, raw(&expected_first));
    assert_eq!(second, raw(&expected_second));
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(split_buffer).unwrap();
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn split_runtime_sizes_with_resolved_outputs_warms_and_captures() {
    // The GatedDeltaNet decode Split (C3 of the GDN capture fix): a *runtime*
    // split-sizes input that is NOT flagged constant (its sizes come from a
    // `Constant` node the default `OptimizationLevel::None` never folds into an
    // initializer). The old path host-read the sizes and synchronized every
    // step, de-capturing the surrounding decode block. Because the executor
    // pre-allocates each output at its statically-inferred shape, the sizes are
    // fully determined by the output shapes — so after warmup the kernel derives
    // a static plan and becomes capture-safe with byte-identical results.
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // Qwen3.5 GatedDeltaNet split: [1,1,H,320] -> [1,1,H,128] + [1,1,H,128] +
    // [1,1,H,64] (q/k/v-style fan-out), axis=-1.
    let heads = 4usize;
    let data_shape = [1, 1, heads, 320];
    let split_shape = [3];
    let output_shapes = [
        vec![1, 1, heads, 128],
        vec![1, 1, heads, 128],
        vec![1, 1, heads, 64],
    ];
    let data = (0..heads * 320)
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let data_bytes = raw(&data);
    let split_bytes = raw(&[128_i64, 128, 64]);
    // NOTE: no `set_constant_inputs` — constant_split_input stays false, which is
    // exactly the GDN case the C3 fix targets.
    let kernel = build_split_kernel(
        &ep,
        &data_shape,
        &output_shapes,
        &[("axis", Attribute::Int(-1))],
        Some(&split_shape),
    );
    // Before warmup the kernel has no static plan, so it must decline capture
    // (the dynamic host-read/synchronize path).
    assert!(
        matches!(kernel.capture_support(), CaptureSupport::Unsupported { .. }),
        "a cold runtime-split Split must decline capture until warmed"
    );

    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let split_buffer = ep.allocate(split_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&split_bytes, cuptr(split_buffer.as_ptr()))
            .unwrap();
    }
    let data_strides = compute_contiguous_strides(&data_shape);
    let split_strides = compute_contiguous_strides(&split_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &data_shape,
            &data_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(split_buffer.as_ptr()),
            DataType::Int64,
            &split_shape,
            &split_strides,
            device,
        ),
    ];
    let mut output_buffers = output_shapes
        .iter()
        .map(|shape| {
            ep.allocate(DataType::Float32.storage_bytes(shape.iter().product()), 256)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let output_strides = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();

    macro_rules! execute {
        () => {{
            let mut outputs = output_buffers
                .iter_mut()
                .zip(&output_shapes)
                .zip(&output_strides)
                .map(|((buffer, shape), strides)| {
                    TensorMut::new(
                        DevicePtrMut(buffer.as_mut_ptr()),
                        DataType::Float32,
                        shape,
                        strides,
                        device,
                    )
                })
                .collect::<Vec<_>>();
            kernel.execute(&inputs, &mut outputs).unwrap();
        }};
    }

    // Warm eagerly: the output-derived plan is cached, flipping capture_support
    // to Supported WITHOUT ever host-reading the split-size input.
    execute!();
    assert_eq!(
        kernel.capture_support(),
        CaptureSupport::Supported,
        "after warmup the output-derived static plan must admit capture"
    );

    let eager = output_buffers
        .iter()
        .zip(&output_shapes)
        .map(|(buffer, shape)| {
            let mut bytes = vec![0; DataType::Float32.storage_bytes(shape.iter().product())];
            unsafe {
                runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr())).unwrap();
            };
            bytes
        })
        .collect::<Vec<_>>();

    // Capture + replay must succeed (no host-read/sync in the captured region)
    // and reproduce the eager bytes exactly.
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute!();
    runtime.end_graph_capture().unwrap();
    assert!(runtime.has_graph_executable().unwrap());
    runtime.replay_graph().unwrap();
    let replayed = output_buffers
        .iter()
        .zip(&output_shapes)
        .map(|(buffer, shape)| {
            let mut bytes = vec![0; DataType::Float32.storage_bytes(shape.iter().product())];
            unsafe {
                runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr())).unwrap();
            };
            bytes
        })
        .collect::<Vec<_>>();
    assert_eq!(replayed, eager, "replayed Split bytes must equal eager");

    // Byte-exact vs the reference chunking.
    let expected: Vec<Vec<f32>> = {
        let bounds = [(0usize, 128usize), (128, 256), (256, 320)];
        bounds
            .iter()
            .map(|&(lo, hi)| {
                data.chunks_exact(320)
                    .flat_map(|head| &head[lo..hi])
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    for (idx, want) in expected.iter().enumerate() {
        assert_eq!(eager[idx], raw(want), "output {idx} byte mismatch");
    }
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(split_buffer).unwrap();
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
}

fn build_movement_kernel(
    ep: &CudaExecutionProvider,
    op: &str,
    input_shapes: &[Vec<usize>],
    output_shapes: &[Vec<usize>],
    attrs: &[(&str, Attribute)],
) -> Box<dyn Kernel> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let inputs = input_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            graph.create_named_value(
                format!("input_{index}"),
                if (index == 1 && matches!(op, "Expand" | "Reshape" | "Tile"))
                    || (op == "Slice" && index > 0)
                {
                    DataType::Int64
                } else {
                    DataType::Float32
                },
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let outputs = output_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        inputs.iter().copied().map(Some).collect(),
        outputs,
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    let model = Model::new(&graph);
    ep.get_kernel(model.graph.node(node_id), input_shapes, 13)
        .unwrap()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn concat_fixed_shape_captures_and_matches_eager() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // DeepSeek-V2-Lite decode: concatenate per-head q_nope and q_rope.
    let input_shapes = [vec![1, 1, 16, 128], vec![1, 1, 16, 64]];
    let output_shape = [1, 1, 16, 192];
    let kernel = build_movement_kernel(
        &ep,
        "Concat",
        &input_shapes,
        &[output_shape.to_vec()],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);

    let first = (0..16 * 128).map(|value| value as f32).collect::<Vec<_>>();
    let second = (0..16 * 64)
        .map(|value| 10_000.0 + value as f32)
        .collect::<Vec<_>>();
    let input_bytes = [raw(&first), raw(&second)];
    let input_buffers = input_bytes
        .iter()
        .map(|bytes| {
            let buffer = ep.allocate(bytes.len(), 256).unwrap();
            unsafe { runtime.htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
            buffer
        })
        .collect::<Vec<_>>();
    let input_strides = input_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let inputs = input_buffers
        .iter()
        .zip(&input_shapes)
        .zip(&input_strides)
        .map(|((buffer, shape), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                DataType::Float32,
                shape,
                strides,
                device,
            )
        })
        .collect::<Vec<_>>();
    let mut output_buffer = ep
        .allocate(16 * 192 * std::mem::size_of::<f32>(), 256)
        .unwrap();
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut execute = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            device,
        )];
        kernel.execute(&inputs, &mut output).unwrap();
    };
    execute();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();

    let mut actual = vec![0; 16 * 192 * std::mem::size_of::<f32>()];
    unsafe {
        runtime
            .dtoh(&mut actual, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };
    let mut expected = Vec::with_capacity(16 * 192);
    for head in 0..16 {
        expected.extend_from_slice(&first[head * 128..(head + 1) * 128]);
        expected.extend_from_slice(&second[head * 64..(head + 1) * 64]);
    }
    assert_eq!(actual, raw(&expected));
    assert!(runtime.reset_graph().unwrap());
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output_buffer).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn reshape_exact_signature_captures_async_copy() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // DeepSeek-V2-Lite decode projection reshape.
    let input_shape = [1, 1, 3072];
    let shape_shape = [4];
    let output_shape = [1, 1, 16, 192];
    let kernel = build_movement_kernel(
        &ep,
        "Reshape",
        &[input_shape.to_vec(), shape_shape.to_vec()],
        &[output_shape.to_vec()],
        &[],
    );
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Unsupported { .. }
    ));

    let data = (0..3072).map(|value| value as f32).collect::<Vec<_>>();
    let data_bytes = raw(&data);
    let shape_bytes = raw(&[0_i64, 0, 16, 192]);
    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let shape_buffer = ep.allocate(shape_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&shape_bytes, cuptr(shape_buffer.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape);
    let shape_strides = compute_contiguous_strides(&shape_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &input_shape,
            &input_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(shape_buffer.as_ptr()),
            DataType::Int64,
            &shape_shape,
            &shape_strides,
            device,
        ),
    ];
    let mut output_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut execute = || {
        let mut outputs = [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            device,
        )];
        kernel.execute(&inputs, &mut outputs).unwrap();
    };
    execute();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute();
    runtime.end_graph_capture().unwrap();

    let replay_data = data
        .iter()
        .map(|value| value + 10_000.0)
        .collect::<Vec<_>>();
    let replay_bytes = raw(&replay_data);
    unsafe {
        runtime
            .htod(&replay_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap()
    };
    runtime.replay_graph().unwrap();
    let mut copied = vec![0; replay_bytes.len()];
    unsafe {
        runtime
            .dtoh(&mut copied, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(copied, replay_bytes);
    assert!(runtime.reset_graph().unwrap());

    let changed_output_shape = [1, 1, 8, 384];
    let changed_output_strides = compute_contiguous_strides(&changed_output_shape);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    let mut changed_output = [TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        &changed_output_shape,
        &changed_output_strides,
        device,
    )];
    let error = kernel.execute(&inputs, &mut changed_output).unwrap_err();
    assert!(error.to_string().contains("warm the exact signature first"));
    runtime.abort_graph_capture().unwrap();

    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(shape_buffer).unwrap();
    ep.deallocate(output_buffer).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn expand_warmed_metadata_captures_and_matches_eager() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // DeepSeek-V2-Lite decode broadcasts one rotary-key head across 16 heads.
    let input_shape = [1, 1, 1, 64];
    let shape_shape = [4];
    let output_shape = [1, 1, 16, 64];
    let kernel = build_movement_kernel(
        &ep,
        "Expand",
        &[input_shape.to_vec(), shape_shape.to_vec()],
        &[output_shape.to_vec()],
        &[],
    );

    let data = (0..64).map(|value| value as f32).collect::<Vec<_>>();
    let data_bytes = raw(&data);
    let shape_bytes = raw(&[1_i64, 1, 16, 1]);
    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let shape_buffer = ep.allocate(shape_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&shape_bytes, cuptr(shape_buffer.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape);
    let shape_strides = compute_contiguous_strides(&shape_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &input_shape,
            &input_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(shape_buffer.as_ptr()),
            DataType::Int64,
            &shape_shape,
            &shape_strides,
            device,
        ),
    ];
    let mut output_buffer = ep
        .allocate(16 * 64 * std::mem::size_of::<f32>(), 256)
        .unwrap();
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut execute = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            device,
        )];
        kernel.execute(&inputs, &mut output).unwrap();
    };
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Unsupported { .. }
    ));
    execute();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();

    let mut actual = vec![0; 16 * 64 * std::mem::size_of::<f32>()];
    unsafe {
        runtime
            .dtoh(&mut actual, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(actual, raw(&data.repeat(16)));
    assert!(runtime.reset_graph().unwrap());
    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(shape_buffer).unwrap();
    ep.deallocate(output_buffer).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn transpose_warmed_metadata_captures_and_matches_eager() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // LinearAttention decode transposes carry a fixed perm on a fixed decode
    // shape, so the permutation metadata is stable and must survive capture.
    let input_shape = [2usize, 1, 3];
    let output_shape = [3usize, 2, 1];
    let kernel = build_movement_kernel(
        &ep,
        "Transpose",
        &[input_shape.to_vec()],
        &[output_shape.to_vec()],
        &[("perm", Attribute::Ints(vec![2, 0, 1]))],
    );

    let data = vec![1_f32, 2., 3., 4., 5., 6.];
    let data_bytes = raw(&data);
    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape);
    let inputs = [TensorView::new(
        DevicePtr(data_buffer.as_ptr()),
        DataType::Float32,
        &input_shape,
        &input_strides,
        device,
    )];
    let mut output_buffer = ep.allocate(6 * std::mem::size_of::<f32>(), 256).unwrap();
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut execute = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            device,
        )];
        kernel.execute(&inputs, &mut output).unwrap();
    };
    // Unwarmed: the kernel must decline capture until a fixed shape is seen.
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Unsupported { .. }
    ));
    // First eager run warms the fixed shape/perm signature.
    execute();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    // Re-running eagerly reuses the cached metadata (no realloc/sync) and must
    // still produce identical bytes.
    execute();
    // Capture + replay must equal the eager result byte-for-byte.
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();

    let mut actual = vec![0; 6 * std::mem::size_of::<f32>()];
    unsafe {
        runtime
            .dtoh(&mut actual, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(actual, raw(&[1_f32, 4., 2., 5., 3., 6.]));
    assert!(runtime.reset_graph().unwrap());
    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(output_buffer).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn transpose_captured_metadata_outlives_rewarm_and_kernel_drop() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    let perm = [2_i64, 0, 1];
    let a_input_shape = [2usize, 1, 3];
    let a_output_shape = [3usize, 2, 1];
    let kernel = build_movement_kernel(
        &ep,
        "Transpose",
        &[a_input_shape.to_vec()],
        &[a_output_shape.to_vec()],
        &[("perm", Attribute::Ints(perm.to_vec()))],
    );

    let a_data = [1_f32, 2., 3., 4., 5., 6.];
    let a_data_buffer = ep.allocate(std::mem::size_of_val(&a_data), 256).unwrap();
    let mut a_output_buffer = ep.allocate(std::mem::size_of_val(&a_data), 256).unwrap();
    unsafe {
        runtime
            .htod(&raw(&a_data), cuptr(a_data_buffer.as_ptr()))
            .unwrap();
    }
    let a_input_strides = compute_contiguous_strides(&a_input_shape);
    let a_output_strides = compute_contiguous_strides(&a_output_shape);
    let a_inputs = [TensorView::new(
        DevicePtr(a_data_buffer.as_ptr()),
        DataType::Float32,
        &a_input_shape,
        &a_input_strides,
        device,
    )];
    let mut run_a = || {
        kernel.execute(
            &a_inputs,
            &mut [TensorMut::new(
                DevicePtrMut(a_output_buffer.as_mut_ptr()),
                DataType::Float32,
                &a_output_shape,
                &a_output_strides,
                device,
            )],
        )
    };

    run_a().unwrap();
    let a_resource = kernel.device_graph_resources();
    assert_eq!(a_resource.len(), 1);
    let a_resource_id = a_resource[0].identity();
    drop(a_resource);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    run_a().unwrap();
    runtime.end_graph_capture().unwrap();

    let b_input_shape = [2usize, 2, 3];
    let b_output_shape = [3usize, 2, 2];
    let b_data = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let b_data_buffer = ep.allocate(b_data.len() * 4, 256).unwrap();
    let mut b_output_buffer = ep.allocate(b_data.len() * 4, 256).unwrap();
    unsafe {
        runtime
            .htod(&raw(&b_data), cuptr(b_data_buffer.as_ptr()))
            .unwrap();
    }
    let b_input_strides = compute_contiguous_strides(&b_input_shape);
    let b_output_strides = compute_contiguous_strides(&b_output_shape);
    let b_inputs = [TensorView::new(
        DevicePtr(b_data_buffer.as_ptr()),
        DataType::Float32,
        &b_input_shape,
        &b_input_strides,
        device,
    )];
    kernel
        .execute(
            &b_inputs,
            &mut [TensorMut::new(
                DevicePtrMut(b_output_buffer.as_mut_ptr()),
                DataType::Float32,
                &b_output_shape,
                &b_output_strides,
                device,
            )],
        )
        .unwrap();
    let b_resource = kernel.device_graph_resources();
    assert_eq!(b_resource.len(), 1);
    assert_ne!(
        b_resource[0].identity(),
        a_resource_id,
        "rewarming must install a new eager metadata owner"
    );
    drop(b_resource);

    let counts_before_drop = runtime.allocation_counts();
    let pooled_before_drop = runtime.raw_pool_retained_bytes();
    drop(kernel);
    assert!(
        runtime.allocation_counts().frees > counts_before_drop.frees
            || runtime.raw_pool_retained_bytes() > pooled_before_drop,
        "dropping the kernel must release only its newer eager metadata owner"
    );

    runtime.replay_graph().unwrap();
    let mut actual = vec![0; std::mem::size_of_val(&a_data)];
    unsafe {
        runtime
            .dtoh(&mut actual, cuptr(a_output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(actual, raw(&[1_f32, 4., 2., 5., 3., 6.]));

    let counts_before_reset = runtime.allocation_counts();
    let pooled_before_reset = runtime.raw_pool_retained_bytes();
    assert!(runtime.reset_graph().unwrap());
    assert!(
        runtime.allocation_counts().frees > counts_before_reset.frees
            || runtime.raw_pool_retained_bytes() > pooled_before_reset,
        "reset must release the older metadata owner retained by the graph"
    );

    for buffer in [
        a_data_buffer,
        a_output_buffer,
        b_data_buffer,
        b_output_buffer,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn slice_captured_metadata_outlives_rewarm_and_kernel_drop() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    let a_data_shape = [3usize, 4];
    let a_output_shape = [2usize, 2];
    let bounds_shape = [2usize];
    let kernel = build_movement_kernel(
        &ep,
        "Slice",
        &[
            a_data_shape.to_vec(),
            bounds_shape.to_vec(),
            bounds_shape.to_vec(),
            bounds_shape.to_vec(),
            bounds_shape.to_vec(),
        ],
        &[a_output_shape.to_vec()],
        &[],
    );

    let a_data = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let a_metadata = [
        raw(&[0_i64, 3]),
        raw(&[3_i64, -5]),
        raw(&[0_i64, 1]),
        raw(&[2_i64, -2]),
    ];
    let a_data_buffer = ep.allocate(a_data.len() * 4, 256).unwrap();
    let a_metadata_buffers = a_metadata
        .iter()
        .map(|bytes| {
            let buffer = ep.allocate(bytes.len(), 256).unwrap();
            unsafe { runtime.htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
            buffer
        })
        .collect::<Vec<_>>();
    let mut a_output_buffer = ep
        .allocate(a_output_shape.iter().product::<usize>() * 4, 256)
        .unwrap();
    unsafe {
        runtime
            .htod(&raw(&a_data), cuptr(a_data_buffer.as_ptr()))
            .unwrap();
    }
    let a_data_strides = compute_contiguous_strides(&a_data_shape);
    let bounds_strides = compute_contiguous_strides(&bounds_shape);
    let a_output_strides = compute_contiguous_strides(&a_output_shape);
    let a_inputs = std::iter::once(TensorView::new(
        DevicePtr(a_data_buffer.as_ptr()),
        DataType::Float32,
        &a_data_shape,
        &a_data_strides,
        device,
    ))
    .chain(a_metadata_buffers.iter().map(|buffer| {
        TensorView::new(
            DevicePtr(buffer.as_ptr()),
            DataType::Int64,
            &bounds_shape,
            &bounds_strides,
            device,
        )
    }))
    .collect::<Vec<_>>();
    let mut run_a = || {
        kernel.execute(
            &a_inputs,
            &mut [TensorMut::new(
                DevicePtrMut(a_output_buffer.as_mut_ptr()),
                DataType::Float32,
                &a_output_shape,
                &a_output_strides,
                device,
            )],
        )
    };

    run_a().unwrap();
    let mut a_resource_ids = kernel
        .device_graph_resources()
        .into_iter()
        .map(|resource| resource.identity())
        .collect::<Vec<_>>();
    a_resource_ids.sort_unstable();
    assert_eq!(a_resource_ids.len(), 2);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    run_a().unwrap();
    runtime.end_graph_capture().unwrap();

    let b_data_shape = [2usize, 6];
    let b_output_shape = [1usize, 3];
    let b_data = (100..112).map(|value| value as f32).collect::<Vec<_>>();
    let b_metadata = [
        raw(&[1_i64, 0]),
        raw(&[2_i64, 6]),
        raw(&[0_i64, 1]),
        raw(&[1_i64, 2]),
    ];
    let b_data_buffer = ep.allocate(b_data.len() * 4, 256).unwrap();
    let b_metadata_buffers = b_metadata
        .iter()
        .map(|bytes| {
            let buffer = ep.allocate(bytes.len(), 256).unwrap();
            unsafe { runtime.htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
            buffer
        })
        .collect::<Vec<_>>();
    let mut b_output_buffer = ep
        .allocate(b_output_shape.iter().product::<usize>() * 4, 256)
        .unwrap();
    unsafe {
        runtime
            .htod(&raw(&b_data), cuptr(b_data_buffer.as_ptr()))
            .unwrap();
    }
    let b_data_strides = compute_contiguous_strides(&b_data_shape);
    let b_output_strides = compute_contiguous_strides(&b_output_shape);
    let b_inputs = std::iter::once(TensorView::new(
        DevicePtr(b_data_buffer.as_ptr()),
        DataType::Float32,
        &b_data_shape,
        &b_data_strides,
        device,
    ))
    .chain(b_metadata_buffers.iter().map(|buffer| {
        TensorView::new(
            DevicePtr(buffer.as_ptr()),
            DataType::Int64,
            &bounds_shape,
            &bounds_strides,
            device,
        )
    }))
    .collect::<Vec<_>>();
    kernel
        .execute(
            &b_inputs,
            &mut [TensorMut::new(
                DevicePtrMut(b_output_buffer.as_mut_ptr()),
                DataType::Float32,
                &b_output_shape,
                &b_output_strides,
                device,
            )],
        )
        .unwrap();
    let mut b_resource_ids = kernel
        .device_graph_resources()
        .into_iter()
        .map(|resource| resource.identity())
        .collect::<Vec<_>>();
    b_resource_ids.sort_unstable();
    assert_eq!(b_resource_ids.len(), 2);
    assert!(
        a_resource_ids
            .iter()
            .all(|identity| !b_resource_ids.contains(identity)),
        "both Slice metadata allocations must be independently replaced"
    );

    drop(kernel);
    runtime.replay_graph().unwrap();
    let mut actual = vec![0; a_output_shape.iter().product::<usize>() * 4];
    unsafe {
        runtime
            .dtoh(&mut actual, cuptr(a_output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(actual, raw(&[3_f32, 1., 11., 9.]));

    let counts_before_reset = runtime.allocation_counts();
    let pooled_before_reset = runtime.raw_pool_retained_bytes();
    assert!(runtime.reset_graph().unwrap());
    assert!(
        runtime.allocation_counts().frees >= counts_before_reset.frees + 2
            || runtime.raw_pool_retained_bytes() > pooled_before_reset,
        "reset must release both Slice metadata owners retained by the graph"
    );

    ep.deallocate(a_data_buffer).unwrap();
    ep.deallocate(a_output_buffer).unwrap();
    for buffer in a_metadata_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(b_data_buffer).unwrap();
    ep.deallocate(b_output_buffer).unwrap();
    for buffer in b_metadata_buffers {
        ep.deallocate(buffer).unwrap();
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn transpose_rejects_signature_change_during_capture() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // Signature A is the warmed decode shape; the persistent-metadata guard must
    // refuse any DIFFERENT shape once a CUDA graph capture is in flight so a
    // stale metadata buffer can never be replayed against the wrong geometry.
    let perm = [2_i64, 0, 1];
    let a_input_shape = [2usize, 1, 3];
    let a_output_shape = [3usize, 2, 1];
    let kernel = build_movement_kernel(
        &ep,
        "Transpose",
        &[a_input_shape.to_vec()],
        &[a_output_shape.to_vec()],
        &[("perm", Attribute::Ints(perm.to_vec()))],
    );

    let a_data = vec![1_f32, 2., 3., 4., 5., 6.];
    let a_data_bytes = raw(&a_data);
    let a_data_buffer = ep.allocate(a_data_bytes.len(), 256).unwrap();
    let mut a_output_buffer = ep.allocate(6 * std::mem::size_of::<f32>(), 256).unwrap();
    unsafe {
        runtime
            .htod(&a_data_bytes, cuptr(a_data_buffer.as_ptr()))
            .unwrap();
    }
    let a_input_strides = compute_contiguous_strides(&a_input_shape);
    let a_output_strides = compute_contiguous_strides(&a_output_shape);
    let a_inputs = [TensorView::new(
        DevicePtr(a_data_buffer.as_ptr()),
        DataType::Float32,
        &a_input_shape,
        &a_input_strides,
        device,
    )];
    let mut run_a = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(a_output_buffer.as_mut_ptr()),
            DataType::Float32,
            &a_output_shape,
            &a_output_strides,
            device,
        )];
        kernel.execute(&a_inputs, &mut output)
    };

    // A larger, genuinely different signature B (input/output shape both change).
    let b_input_shape = [2usize, 2, 3];
    let b_output_shape = [3usize, 2, 2];
    let b_data = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let b_data_bytes = raw(&b_data);
    let b_data_buffer = ep.allocate(b_data_bytes.len(), 256).unwrap();
    let mut b_output_buffer = ep.allocate(12 * std::mem::size_of::<f32>(), 256).unwrap();
    unsafe {
        runtime
            .htod(&b_data_bytes, cuptr(b_data_buffer.as_ptr()))
            .unwrap();
    }
    let b_input_strides = compute_contiguous_strides(&b_input_shape);
    let b_output_strides = compute_contiguous_strides(&b_output_shape);
    let b_inputs = [TensorView::new(
        DevicePtr(b_data_buffer.as_ptr()),
        DataType::Float32,
        &b_input_shape,
        &b_input_strides,
        device,
    )];
    let mut run_b = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(b_output_buffer.as_mut_ptr()),
            DataType::Float32,
            &b_output_shape,
            &b_output_strides,
            device,
        )];
        kernel.execute(&b_inputs, &mut output)
    };

    // Warm signature A eagerly, then open a capture keyed on that warmed kernel.
    run_a().unwrap();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    // Recording the warmed signature A into the graph is fine.
    run_a().unwrap();
    // Feeding a DIFFERENT signature mid-capture must be rejected, not silently
    // replayed against the stale metadata buffer.
    let err = run_b().expect_err("signature change during capture must be rejected");
    assert!(
        err.to_string()
            .contains("changed during CUDA graph capture"),
        "unexpected error message: {err}"
    );
    // Close the capture cleanly (it recorded only the warmed A op) and reset.
    runtime.end_graph_capture().unwrap();
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(a_data_buffer).unwrap();
    ep.deallocate(a_output_buffer).unwrap();
    ep.deallocate(b_data_buffer).unwrap();
    ep.deallocate(b_output_buffer).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn tile_warmed_metadata_captures_and_matches_eager() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // Fixed decode-shape Tile: the repeats/geometry are stable, so the persistent
    // metadata must survive capture with no per-call host read or sync.
    let input_shape = [2usize, 1];
    let repeats_shape = [2usize];
    let output_shape = [4usize, 3];
    let kernel = build_movement_kernel(
        &ep,
        "Tile",
        &[input_shape.to_vec(), repeats_shape.to_vec()],
        &[output_shape.to_vec()],
        &[],
    );

    let data = vec![1_f32, 2.];
    let data_bytes = raw(&data);
    let repeats_bytes = raw(&[2_i64, 3]);
    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let repeats_buffer = ep.allocate(repeats_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&repeats_bytes, cuptr(repeats_buffer.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape);
    let repeats_strides = compute_contiguous_strides(&repeats_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &input_shape,
            &input_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(repeats_buffer.as_ptr()),
            DataType::Int64,
            &repeats_shape,
            &repeats_strides,
            device,
        ),
    ];
    let mut output_buffer = ep.allocate(12 * std::mem::size_of::<f32>(), 256).unwrap();
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut execute = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            device,
        )];
        kernel.execute(&inputs, &mut output).unwrap();
    };
    // Unwarmed: the kernel must decline capture until a fixed shape is seen.
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Unsupported { .. }
    ));
    // First eager run reads/validates repeats once and warms the signature.
    execute();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    // Re-running eagerly reuses cached metadata with no host read/sync and must
    // still produce identical bytes.
    execute();
    // Capture + replay must equal the eager result byte-for-byte.
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();

    let mut actual = vec![0; 12 * std::mem::size_of::<f32>()];
    unsafe {
        runtime
            .dtoh(&mut actual, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(
        actual,
        raw(&[1_f32, 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2.])
    );
    assert!(runtime.reset_graph().unwrap());
    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(repeats_buffer).unwrap();
    ep.deallocate(output_buffer).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn tile_rejects_signature_change_during_capture() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    // Signature A is the warmed decode shape; feeding a different tiled geometry
    // mid-capture must be rejected, not replayed against the stale metadata.
    let input_shape = [2usize, 1];
    let repeats_shape = [2usize];
    let a_output_shape = [4usize, 3];
    let kernel = build_movement_kernel(
        &ep,
        "Tile",
        &[input_shape.to_vec(), repeats_shape.to_vec()],
        &[a_output_shape.to_vec()],
        &[],
    );

    let data = vec![1_f32, 2.];
    let data_bytes = raw(&data);
    let data_buffer = ep.allocate(data_bytes.len(), 256).unwrap();
    let a_repeats_bytes = raw(&[2_i64, 3]);
    let a_repeats_buffer = ep.allocate(a_repeats_bytes.len(), 256).unwrap();
    let b_repeats_bytes = raw(&[2_i64, 2]);
    let b_repeats_buffer = ep.allocate(b_repeats_bytes.len(), 256).unwrap();
    unsafe {
        runtime
            .htod(&data_bytes, cuptr(data_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&a_repeats_bytes, cuptr(a_repeats_buffer.as_ptr()))
            .unwrap();
        runtime
            .htod(&b_repeats_bytes, cuptr(b_repeats_buffer.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape);
    let repeats_strides = compute_contiguous_strides(&repeats_shape);
    let a_inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &input_shape,
            &input_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(a_repeats_buffer.as_ptr()),
            DataType::Int64,
            &repeats_shape,
            &repeats_strides,
            device,
        ),
    ];
    let b_inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            &input_shape,
            &input_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(b_repeats_buffer.as_ptr()),
            DataType::Int64,
            &repeats_shape,
            &repeats_strides,
            device,
        ),
    ];
    let mut a_output_buffer = ep.allocate(12 * std::mem::size_of::<f32>(), 256).unwrap();
    let a_output_strides = compute_contiguous_strides(&a_output_shape);
    let b_output_shape = [4usize, 2];
    let mut b_output_buffer = ep.allocate(8 * std::mem::size_of::<f32>(), 256).unwrap();
    let b_output_strides = compute_contiguous_strides(&b_output_shape);
    let mut run_a = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(a_output_buffer.as_mut_ptr()),
            DataType::Float32,
            &a_output_shape,
            &a_output_strides,
            device,
        )];
        kernel.execute(&a_inputs, &mut output)
    };
    let mut run_b = || {
        let mut output = [TensorMut::new(
            DevicePtrMut(b_output_buffer.as_mut_ptr()),
            DataType::Float32,
            &b_output_shape,
            &b_output_strides,
            device,
        )];
        kernel.execute(&b_inputs, &mut output)
    };

    // Warm signature A eagerly, then open a capture keyed on that warmed kernel.
    run_a().unwrap();
    assert_eq!(kernel.capture_support(), CaptureSupport::Supported);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    // Recording the warmed signature A into the graph is fine.
    run_a().unwrap();
    // A DIFFERENT tiled output geometry mid-capture must be rejected.
    let err = run_b().expect_err("signature change during capture must be rejected");
    assert!(
        err.to_string()
            .contains("changed during CUDA graph capture"),
        "unexpected error message: {err}"
    );
    runtime.end_graph_capture().unwrap();
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(a_repeats_buffer).unwrap();
    ep.deallocate(b_repeats_buffer).unwrap();
    ep.deallocate(a_output_buffer).unwrap();
    ep.deallocate(b_output_buffer).unwrap();
}
