//! CUDA conformance tests for movement/construction operators and `Where`.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
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

fn gpu() -> CudaExecutionProvider {
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
                &format!("input_{i}"),
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
                &format!("output_{i}"),
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

#[test]
fn concat_negative_axis_and_multiple_inputs() {
    let ep = gpu();
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

#[test]
fn expand_right_aligned_broadcast() {
    let ep = gpu();
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

#[test]
fn reshape_preserves_dtype_agnostic_bytes() {
    let ep = gpu();
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

#[test]
fn slice_multi_axis_negative_axis_and_step() {
    let ep = gpu();
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

#[test]
fn split_negative_axis_via_split_input() {
    let ep = gpu();
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

#[test]
fn squeeze_axes_input_preserves_bytes() {
    let ep = gpu();
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

#[test]
fn tile_multi_axis_repeats() {
    let ep = gpu();
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

#[test]
fn transpose_explicit_three_axis_permutation() {
    let ep = gpu();
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

#[test]
fn unsqueeze_multiple_axes_input_preserves_bytes() {
    let ep = gpu();
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

#[test]
fn where_broadcasts_all_three_inputs() {
    let ep = gpu();
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
