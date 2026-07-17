//! CUDA conformance tests for router/mask indexing and scan operators.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

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

fn run(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
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
        input_values.into_iter().map(Some).collect(),
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
    let mut output_views = outputs
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
    kernel.execute(&input_views, &mut output_views).unwrap();

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

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|v| i64::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}

#[test]
fn topk_non_final_axes_match_cpu_layout_and_are_deterministic() {
    let axis_zero = || {
        run(
            "TopK",
            10,
            &[
                tensor(
                    DataType::Float32,
                    &[3, 4],
                    &[5_f32, 1., 7., 2., 5., 4., 6., 9., 3., 4., 8., 9.],
                ),
                tensor(DataType::Int64, &[], &[2_i64]),
            ],
            &[
                (DataType::Float32, vec![2, 4]),
                (DataType::Int64, vec![2, 4]),
            ],
            &[("axis", Attribute::Int(0))],
        )
    };
    let out = axis_zero();
    assert_eq!(f32s(&out[0]), vec![5., 4., 8., 9., 5., 4., 7., 9.]);
    assert_eq!(i64s(&out[1]), vec![0, 1, 2, 1, 1, 2, 0, 2]);
    assert_eq!(out, axis_zero());

    let middle_axis = || {
        run(
            "TopK",
            10,
            &[
                tensor(
                    DataType::Float32,
                    &[2, 3, 2],
                    &[1_f32, 9., 5., 9., 5., 2., 7., 4., 6., 8., 7., 8.],
                ),
                tensor(DataType::Int64, &[], &[2_i64]),
            ],
            &[
                (DataType::Float32, vec![2, 2, 2]),
                (DataType::Int64, vec![2, 2, 2]),
            ],
            &[("axis", Attribute::Int(1))],
        )
    };
    let out = middle_axis();
    assert_eq!(f32s(&out[0]), vec![5., 9., 5., 9., 7., 8., 7., 8.]);
    assert_eq!(i64s(&out[1]), vec![1, 0, 2, 1, 0, 1, 2, 2]);
    assert_eq!(out, middle_axis());
}

#[test]
fn topk_largest_ties_k_input() {
    let out = run(
        "TopK",
        10,
        &[
            tensor(DataType::Float32, &[4], &[4_f32, 5., 5., 2.]),
            tensor(DataType::Int64, &[], &[2_i64]),
        ],
        &[(DataType::Float32, vec![2]), (DataType::Int64, vec![2])],
        &[],
    );
    assert_eq!(f32s(&out[0]), vec![5., 5.]);
    assert_eq!(i64s(&out[1]), vec![1, 2]);
}

#[test]
fn topk_smallest_and_sorted_false_match_cpu_order() {
    let out = run(
        "TopK",
        10,
        &[
            tensor(DataType::Float32, &[5], &[3_f32, 1., 1., 4., 2.]),
            tensor(DataType::Int64, &[], &[3_i64]),
        ],
        &[(DataType::Float32, vec![3]), (DataType::Int64, vec![3])],
        &[
            ("largest", Attribute::Int(0)),
            ("sorted", Attribute::Int(0)),
        ],
    );
    assert_eq!(f32s(&out[0]), vec![1., 1., 2.]);
    assert_eq!(i64s(&out[1]), vec![1, 2, 4]);
}

#[test]
fn gather_elements_negative_axis_and_indices() {
    let out = run(
        "GatherElements",
        11,
        &[
            tensor(DataType::Float32, &[2, 3], &[1_f32, 2., 3., 4., 5., 6.]),
            tensor(DataType::Int64, &[2, 2], &[2_i64, 0, -1, 1]),
        ],
        &[(DataType::Float32, vec![2, 2])],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(f32s(&out[0]), vec![3., 1., 6., 5.]);
}

fn scatter(reduction: &str) -> Vec<f32> {
    let attrs = if reduction == "none" {
        vec![("axis", Attribute::Int(-1))]
    } else {
        vec![
            ("axis", Attribute::Int(-1)),
            ("reduction", Attribute::String(reduction.into())),
        ]
    };
    let out = run(
        "ScatterElements",
        if reduction == "none" { 11 } else { 16 },
        &[
            tensor(DataType::Float32, &[3], &[10_f32, 20., 30.]),
            tensor(DataType::Int64, &[3], &[1_i64, 1, -3]),
            tensor(DataType::Float32, &[3], &[2_f32, 3., 4.]),
        ],
        &[(DataType::Float32, vec![3])],
        &attrs,
    );
    f32s(&out[0])
}

#[test]
fn scatter_elements_all_reductions_are_ordered_and_deterministic() {
    assert_eq!(scatter("none"), vec![4., 3., 30.]);
    assert_eq!(scatter("add"), vec![14., 25., 30.]);
    assert_eq!(scatter("mul"), vec![40., 120., 30.]);
    assert_eq!(scatter("max"), vec![10., 20., 30.]);
    assert_eq!(scatter("min"), vec![4., 2., 30.]);
}

#[test]
fn cumsum_exclusive_reverse_matrix_with_negative_axis() {
    let expected = [
        ((0, 0), vec![1., 3., 6., 4., 9., 15.]),
        ((1, 0), vec![0., 1., 3., 0., 4., 9.]),
        ((0, 1), vec![6., 5., 3., 15., 11., 6.]),
        ((1, 1), vec![5., 3., 0., 11., 6., 0.]),
    ];
    for ((exclusive, reverse), expected) in expected {
        let out = run(
            "CumSum",
            11,
            &[
                tensor(DataType::Float32, &[2, 3], &[1_f32, 2., 3., 4., 5., 6.]),
                tensor(DataType::Int64, &[], &[-1_i64]),
            ],
            &[(DataType::Float32, vec![2, 3])],
            &[
                ("exclusive", Attribute::Int(exclusive)),
                ("reverse", Attribute::Int(reverse)),
            ],
        );
        assert_eq!(f32s(&out[0]), expected);
    }
}
