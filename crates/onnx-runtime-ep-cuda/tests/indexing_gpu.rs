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
//! CUDA conformance tests for router/mask indexing and scan operators.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut,
    TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{CudaExecutionProvider, SCATTER_CAPTURE_ERROR_INDEX};
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

fn graph(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> (Graph, NodeId) {
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
    (graph, node_id)
}

fn run(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let (graph, node_id) = graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|input| input.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
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

fn run_warmed_capture(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let (graph, node_id) = graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let shapes = inputs
        .iter()
        .map(|input| input.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &shapes, opset)
        .unwrap();
    let mut input_buffers = inputs
        .iter()
        .map(|input| {
            let mut buffer = ep.allocate(input.bytes.len(), 256).unwrap();
            if !input.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&input.bytes, cuptr(buffer.as_mut_ptr()))
                        .unwrap();
                }
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
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let execute = |buffers: &mut [DeviceBuffer]| {
        let mut output_views = outputs
            .iter()
            .zip(buffers.iter_mut())
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
    };
    let download = |buffers: &[DeviceBuffer]| {
        outputs
            .iter()
            .zip(buffers)
            .map(|((dtype, shape), buffer)| {
                let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
                if !bytes.is_empty() {
                    unsafe {
                        ep.runtime()
                            .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
                            .unwrap();
                    }
                }
                bytes
            })
            .collect::<Vec<_>>()
    };

    execute(&mut output_buffers);
    let eager = download(&output_buffers);
    assert!(
        kernel.cuda_graph_compatible(),
        "{op} must advertise capture support after warmup"
    );
    let allocation_counts = ep.runtime().allocation_counts();
    ep.runtime()
        .begin_graph_capture(&[kernel.as_ref()])
        .unwrap();
    execute(&mut output_buffers);
    ep.runtime().end_graph_capture().unwrap();
    assert_eq!(ep.runtime().allocation_counts(), allocation_counts);
    ep.runtime().replay_graph().unwrap();
    let replay = download(&output_buffers);
    assert!(ep.runtime().reset_graph().unwrap());

    drop(input_views);
    drop(kernel);
    for buffer in input_buffers.drain(..) {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
    (eager, replay)
}

fn run_cpu(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CpuExecutionProvider::new();
    let (graph, node_id) = graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|input| input.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();
    let input_strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_strides)
        .map(|(input, strides)| {
            TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_bytes = outputs
        .iter()
        .map(|(dtype, shape)| vec![0_u8; dtype.storage_bytes(shape.iter().product())])
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(&output_strides)
        .zip(output_bytes.iter_mut())
        .map(|(((dtype, shape), strides), bytes)| {
            TensorMut::new(
                DevicePtrMut(bytes.as_mut_ptr().cast()),
                *dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();
    drop(output_views);
    output_bytes
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

fn f16s(bytes: &[u8]) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|v| f16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())))
        .collect()
}

fn bf16s(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(2)
        .map(|v| bf16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())))
        .collect()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn topk_deepseek_router_k6_of_64_is_deterministic() {
    let values = (0..64)
        .map(|expert| (expert % 11) as f32)
        .collect::<Vec<_>>();
    let out = run(
        "TopK",
        10,
        &[
            tensor(DataType::Float32, &[1, 64], &values),
            tensor(DataType::Int64, &[], &[6_i64]),
        ],
        &[
            (DataType::Float32, vec![1, 6]),
            (DataType::Int64, vec![1, 6]),
        ],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(f32s(&out[0]), vec![10., 10., 10., 10., 10., 9.]);
    assert_eq!(i64s(&out[1]), vec![10, 21, 32, 43, 54, 9]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn topk_fp16_and_bf16_router_values_match_cpu_order() {
    let input = [4.0_f32, 7.0, 7.0, -2.0, 6.0];
    let expected_indices = vec![1, 2, 4];

    let fp16_input = input.map(f16::from_f32);
    let fp16_inputs = [
        tensor(DataType::Float16, &[1, 5], &fp16_input),
        tensor(DataType::Int64, &[], &[3_i64]),
    ];
    let fp16_outputs = [
        (DataType::Float16, vec![1, 3]),
        (DataType::Int64, vec![1, 3]),
    ];
    let attrs = [("axis", Attribute::Int(-1))];
    let fp16_out = run("TopK", 10, &fp16_inputs, &fp16_outputs, &attrs);
    assert_eq!(
        fp16_out,
        run_cpu("TopK", 10, &fp16_inputs, &fp16_outputs, &attrs)
    );
    assert_eq!(
        f16s(&fp16_out[0]),
        vec![f16::from_f32(7.0), f16::from_f32(7.0), f16::from_f32(6.0)]
    );
    assert_eq!(i64s(&fp16_out[1]), expected_indices);

    let bf16_input = input.map(bf16::from_f32);
    let bf16_inputs = [
        tensor(DataType::BFloat16, &[1, 5], &bf16_input),
        tensor(DataType::Int64, &[], &[3_i64]),
    ];
    let bf16_outputs = [
        (DataType::BFloat16, vec![1, 3]),
        (DataType::Int64, vec![1, 3]),
    ];
    let bf16_out = run("TopK", 10, &bf16_inputs, &bf16_outputs, &attrs);
    assert_eq!(
        bf16_out,
        run_cpu("TopK", 10, &bf16_inputs, &bf16_outputs, &attrs)
    );
    assert_eq!(
        bf16s(&bf16_out[0]),
        vec![
            bf16::from_f32(7.0),
            bf16::from_f32(7.0),
            bf16::from_f32(6.0)
        ]
    );
    assert_eq!(i64s(&bf16_out[1]), expected_indices);
}

/// Router-scale fp16 parity: the Qwen3.6-35B-A3B (`dense_fallback` MoE) gate is a
/// 256-expert / top-8 `TopK` in fp16. Prove the CUDA kernel is byte-identical to
/// the CPU EP oracle at that exact shape (large last-dim, k=8, many ties) and that
/// the fp16 path handles a non-final axis identically to the proven f32 kernel.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn topk_fp16_router_scale_and_non_final_axis_match_cpu() {
    // 256-expert / top-8 router on the final axis (two token rows). The `% 37`
    // pattern repeats every 37 experts, so each row has many ties, forcing the
    // ascending-index tie-break the upcast-for-compare path must reproduce.
    let (rows, width) = (2_usize, 256_usize);
    let data = (0..rows * width)
        .map(|n| f16::from_f32((n % 37) as f32 - 18.0))
        .collect::<Vec<_>>();
    let inputs = [
        tensor(DataType::Float16, &[rows, width], &data),
        tensor(DataType::Int64, &[], &[8_i64]),
    ];
    let outputs = [
        (DataType::Float16, vec![rows, 8]),
        (DataType::Int64, vec![rows, 8]),
    ];
    let attrs = [("axis", Attribute::Int(-1))];
    let router = run("TopK", 10, &inputs, &outputs, &attrs);
    assert_eq!(
        router,
        run_cpu("TopK", 10, &inputs, &outputs, &attrs),
        "256-expert fp16 router GPU/CPU byte parity"
    );
    // Each token's 8 selected experts must be distinct and in range.
    let indices = i64s(&router[1]);
    assert_eq!(indices.len(), rows * 8);
    for row in indices.chunks(8) {
        for (a, &idx) in row.iter().enumerate() {
            assert!((0..256).contains(&idx), "index {idx} out of range");
            for &other in &row[a + 1..] {
                assert_ne!(idx, other, "router selected a duplicate expert");
            }
        }
    }

    // Non-final axis (axis 0): oracle the fp16 kernel against the proven f32 GPU
    // path on identical, exactly-representable integer inputs. (The CPU EP writes
    // non-final-axis TopK in push order while the CUDA kernel and ONNX use the
    // k-major layout asserted by `topk_non_final_axes_*`; that divergence is
    // pre-existing and independent of dtype, so f32 GPU is the layout oracle here.)
    let axis0 = [("axis", Attribute::Int(0))];
    let f32_data = (0..20).map(|n| (n % 37) as f32 - 18.0).collect::<Vec<_>>();
    let f16_data = f32_data
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect::<Vec<_>>();
    let f32_out = run(
        "TopK",
        10,
        &[
            tensor(DataType::Float32, &[5, 4], &f32_data),
            tensor(DataType::Int64, &[], &[2_i64]),
        ],
        &[
            (DataType::Float32, vec![2, 4]),
            (DataType::Int64, vec![2, 4]),
        ],
        &axis0,
    );
    let f16_out = run(
        "TopK",
        10,
        &[
            tensor(DataType::Float16, &[5, 4], &f16_data),
            tensor(DataType::Int64, &[], &[2_i64]),
        ],
        &[
            (DataType::Float16, vec![2, 4]),
            (DataType::Int64, vec![2, 4]),
        ],
        &axis0,
    );
    assert_eq!(
        i64s(&f16_out[1]),
        i64s(&f32_out[1]),
        "fp16 non-final-axis indices match the f32 kernel"
    );
    assert_eq!(
        f16s(&f16_out[0])
            .iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        f32s(&f32_out[0]),
        "fp16 non-final-axis values match the f32 kernel"
    );
}

/// Regression guard for the Qwen3.6-35B-A3B unblock: the model's 40 MoE router
/// gates are fp16 `TopK`. Before fp16 support the CUDA claim gate rejected them
/// with `input 0 ('X') dtype Float16 unsupported; expected Float32`, forcing a
/// whole-session CPU fallback. The EP must now CLAIM fp16 TopK (and still claim
/// f32, so no dtype is regressed).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cuda_ep_claims_fp16_topk_router_nodes() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let claims = |dtype: DataType| {
        let (graph, node_id) = graph(
            "TopK",
            10,
            &[
                tensor(dtype, &[1, 256], &vec![0_u16; 256]),
                tensor(DataType::Int64, &[], &[8_i64]),
            ],
            &[(dtype, vec![1, 8]), (DataType::Int64, vec![1, 8])],
            &[("axis", Attribute::Int(-1))],
        );
        let model = Model::new(&graph);
        let shapes = [
            static_shape([1_usize, 256]),
            static_shape(std::iter::empty::<usize>()),
        ];
        matches!(
            ep.supports_op(
                model.graph.node(node_id),
                10,
                &shapes,
                &[dtype, DataType::Int64],
                &[],
            ),
            KernelMatch::Supported { .. }
        )
    };
    assert!(claims(DataType::Float16), "CUDA EP must claim fp16 TopK");
    assert!(claims(DataType::BFloat16), "CUDA EP must claim bf16 TopK");
    assert!(
        claims(DataType::Float32),
        "CUDA EP must still claim f32 TopK"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn warmed_moe_routing_ops_capture_without_allocations() {
    let router = (0..64)
        .map(|expert| (expert % 11) as f32)
        .collect::<Vec<_>>();
    let (eager, replay) = run_warmed_capture(
        "TopK",
        10,
        &[
            tensor(DataType::Float32, &[1, 64], &router),
            tensor(DataType::Int64, &[], &[6_i64]),
        ],
        &[
            (DataType::Float32, vec![1, 6]),
            (DataType::Int64, vec![1, 6]),
        ],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(replay, eager);

    let (eager, replay) = run_warmed_capture(
        "GatherElements",
        13,
        &[
            tensor(DataType::Float32, &[1, 64], &router),
            tensor(DataType::Int64, &[1, 6], &[63_i64, 0, 17, 31, 42, -1]),
        ],
        &[(DataType::Float32, vec![1, 6])],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(replay, eager);

    let (eager, replay) = run_warmed_capture(
        "Softmax",
        13,
        &[tensor(DataType::Float32, &[1, 64], &router)],
        &[(DataType::Float32, vec![1, 64])],
        &[("axis", Attribute::Int(-1))],
    );
    assert_eq!(replay, eager);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn scatter_elements_all_reductions_are_ordered_and_deterministic() {
    assert_eq!(scatter("none"), vec![4., 3., 30.]);
    assert_eq!(scatter("add"), vec![14., 25., 30.]);
    assert_eq!(scatter("mul"), vec![40., 120., 30.]);
    assert_eq!(scatter("max"), vec![10., 20., 30.]);
    assert_eq!(scatter("min"), vec![4., 2., 30.]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn scatter_elements_fp16_data_matches_cpu_oracle() {
    let data = [
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
    ];
    let updates = [f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
    for reduction in ["none", "add", "mul", "max", "min"] {
        let attrs = if reduction == "none" {
            vec![("axis", Attribute::Int(-1))]
        } else {
            vec![
                ("axis", Attribute::Int(-1)),
                ("reduction", Attribute::String(reduction.into())),
            ]
        };
        let opset = if reduction == "none" { 11 } else { 16 };
        let gpu = run(
            "ScatterElements",
            opset,
            &[
                tensor(DataType::Float16, &[3], &data),
                tensor(DataType::Int64, &[3], &[1_i64, 1, -3]),
                tensor(DataType::Float16, &[3], &updates),
            ],
            &[(DataType::Float16, vec![3])],
            &attrs,
        );
        let cpu = run_cpu(
            "ScatterElements",
            opset,
            &[
                tensor(DataType::Float16, &[3], &data),
                tensor(DataType::Int64, &[3], &[1_i64, 1, -3]),
                tensor(DataType::Float16, &[3], &updates),
            ],
            &[(DataType::Float16, vec![3])],
            &attrs,
        );
        assert_eq!(gpu, cpu, "fp16 reduction={reduction}");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn scatter_elements_int32_indices_match_cpu_oracle() {
    for reduction in ["none", "add", "mul", "max", "min"] {
        let attrs = if reduction == "none" {
            vec![("axis", Attribute::Int(-1))]
        } else {
            vec![
                ("axis", Attribute::Int(-1)),
                ("reduction", Attribute::String(reduction.into())),
            ]
        };
        let opset = if reduction == "none" { 11 } else { 16 };
        let gpu = run(
            "ScatterElements",
            opset,
            &[
                tensor(DataType::Float32, &[3], &[10_f32, 20.0, 30.0]),
                tensor(DataType::Int32, &[3], &[1_i32, 1, -3]),
                tensor(DataType::Float32, &[3], &[2_f32, 3.0, 4.0]),
            ],
            &[(DataType::Float32, vec![3])],
            &attrs,
        );
        let cpu = run_cpu(
            "ScatterElements",
            opset,
            &[
                tensor(DataType::Float32, &[3], &[10_f32, 20.0, 30.0]),
                tensor(DataType::Int64, &[3], &[1_i64, 1, -3]),
                tensor(DataType::Float32, &[3], &[2_f32, 3.0, 4.0]),
            ],
            &[(DataType::Float32, vec![3])],
            &attrs,
        );
        assert_eq!(gpu, cpu, "int32 indices reduction={reduction}");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn scatter_elements_bf16_data_and_int32_indices_match_cpu_oracle() {
    let data = [
        bf16::from_f32(10.0),
        bf16::from_f32(20.0),
        bf16::from_f32(30.0),
    ];
    let updates = [
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
    ];
    let attrs = [
        ("axis", Attribute::Int(-1)),
        ("reduction", Attribute::String(b"add".to_vec())),
    ];
    let gpu = run(
        "ScatterElements",
        16,
        &[
            tensor(DataType::BFloat16, &[3], &data),
            tensor(DataType::Int32, &[3], &[1_i32, 1, -3]),
            tensor(DataType::BFloat16, &[3], &updates),
        ],
        &[(DataType::BFloat16, vec![3])],
        &attrs,
    );
    let cpu = run_cpu(
        "ScatterElements",
        16,
        &[
            tensor(DataType::BFloat16, &[3], &data),
            tensor(DataType::Int64, &[3], &[1_i64, 1, -3]),
            tensor(DataType::BFloat16, &[3], &updates),
        ],
        &[(DataType::BFloat16, vec![3])],
        &attrs,
    );
    assert_eq!(gpu, cpu);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn scatter_elements_warmed_fp16_int32_path_is_capture_safe() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let data = [
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
    ];
    let indices = [1_i32, 1, -3];
    let updates = [f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
    let inputs = [
        tensor(DataType::Float16, &[3], &data),
        tensor(DataType::Int32, &[3], &indices),
        tensor(DataType::Float16, &[3], &updates),
    ];
    let outputs = [(DataType::Float16, vec![3])];
    let attrs = [
        ("axis", Attribute::Int(-1)),
        ("reduction", Attribute::String(b"add".to_vec())),
    ];
    let (graph, node_id) = graph("ScatterElements", 16, &inputs, &outputs, &attrs);
    let model = Model::new(&graph);
    let shapes = inputs
        .iter()
        .map(|input| input.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &shapes, 16)
        .unwrap();
    let mut input_buffers = inputs
        .iter()
        .map(|input| {
            let buffer = ep.allocate(input.bytes.len(), 256).unwrap();
            unsafe {
                ep.runtime()
                    .htod(&input.bytes, cuptr(buffer.as_ptr()))
                    .unwrap()
            };
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
    let mut output = ep.allocate(3 * std::mem::size_of::<f16>(), 256).unwrap();
    let output_strides = compute_contiguous_strides(&[3]);
    let execute = |output: &mut DeviceBuffer| {
        let output_view = TensorMut::new(
            DevicePtrMut(output.as_mut_ptr()),
            DataType::Float16,
            &[3],
            &output_strides,
            ep.device_id(),
        );
        kernel.execute(&input_views, &mut [output_view]).unwrap();
    };

    execute(&mut output);
    let mut eager = vec![0_u8; 3 * std::mem::size_of::<f16>()];
    unsafe {
        ep.runtime()
            .dtoh(&mut eager, cuptr(output.as_ptr()))
            .unwrap()
    };
    assert!(kernel.cuda_graph_compatible());
    let allocation_counts = ep.runtime().allocation_counts();

    ep.runtime()
        .begin_graph_capture(&[kernel.as_ref()])
        .unwrap();
    execute(&mut output);
    ep.runtime().end_graph_capture().unwrap();
    ep.runtime().replay_graph().unwrap();
    let mut replay = vec![0_u8; eager.len()];
    unsafe {
        ep.runtime()
            .dtoh(&mut replay, cuptr(output.as_ptr()))
            .unwrap()
    };
    assert_eq!(replay, eager);
    assert_eq!(ep.runtime().allocation_counts(), allocation_counts);

    let invalid = [-4_i32, 1, -3];
    unsafe {
        ep.runtime()
            .htod(
                raw(&invalid).as_slice(),
                cuptr(input_buffers[1].as_mut_ptr()),
            )
            .unwrap()
    };
    ep.runtime().replay_graph().unwrap();
    assert_ne!(
        ep.runtime().check_capture_error().unwrap() & SCATTER_CAPTURE_ERROR_INDEX,
        0
    );
    assert!(ep.runtime().reset_graph().unwrap());

    drop(input_views);
    drop(kernel);
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
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
