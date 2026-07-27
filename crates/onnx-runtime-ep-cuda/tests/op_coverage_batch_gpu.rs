//! GPU parity tests for the issue #67 CUDA op-coverage batch: the
//! trigonometric/hyperbolic unary-math family (`Tan`, `Sinh`, `Cosh`, `Asin`,
//! `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`), the metadata/movement ops
//! `Identity`, `Flatten`, `Size`, and the triangular mask `Trilu`.
//!
//! Every check runs the real CUDA kernel and compares its result against the CPU
//! execution provider running the same ONNX node (the reference path), asserting
//! concrete values. The suite skips cleanly when no CUDA runtime is present so a
//! host without a GPU still passes.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
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
    // SAFETY: primitive test values are plain old data with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn input<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: raw(values),
    }
}

fn encode_floats(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        other => panic!("unsupported float dtype {other:?}"),
    }
}

fn decode_floats(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported float dtype {other:?}"),
    }
}

fn build_graph(
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
        .map(|(index, tensor)| {
            let value = graph.create_named_value(
                format!("input_{index}"),
                tensor.dtype,
                static_shape(tensor.shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(index, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{index}"),
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

fn run_cuda(
    ep: &CudaExecutionProvider,
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let (graph, node_id) = build_graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();

    let input_buffers = inputs
        .iter()
        .map(|tensor| {
            let buffer = ep.allocate(tensor.bytes.len().max(1), 256).unwrap();
            if !tensor.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&tensor.bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            buffer
        })
        .collect::<Vec<_>>();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((tensor, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| {
            ep.allocate(dtype.storage_bytes(shape.iter().product()).max(1), 256)
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

fn run_cpu(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CpuExecutionProvider::new();
    let (graph, node_id) = build_graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_strides)
        .map(|(tensor, strides)| {
            TensorView::new(
                DevicePtr(tensor.bytes.as_ptr().cast()),
                tensor.dtype,
                &tensor.shape,
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

fn cuda_ep() -> Option<CudaExecutionProvider> {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => Some(ep),
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            None
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked");
            None
        }
    }
}

fn assert_close(op: &str, dtype: DataType, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len(), "{op} {dtype:?}: length mismatch");
    for (index, (&got, &want)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "{op} {dtype:?} index {index}: got {got}, expected {want}, tolerance {tolerance}"
        );
    }
}

/// Trigonometric / hyperbolic unary math parity across f32/f16/bf16, each op fed
/// domain-safe values, compared element-wise against the CPU EP.
#[test]
fn trig_hyperbolic_unary_matches_cpu() {
    let Some(ep) = cuda_ep() else { return };
    // (op, domain-safe inputs).
    let cases: &[(&str, &[f32])] = &[
        ("Tan", &[-1.2, -0.4, 0.0, 0.5, 1.1]),
        ("Sinh", &[-2.0, -0.7, 0.0, 0.9, 2.5]),
        ("Cosh", &[-2.0, -0.7, 0.0, 0.9, 2.5]),
        ("Asin", &[-0.95, -0.4, 0.0, 0.5, 0.9]),
        ("Acos", &[-0.95, -0.4, 0.0, 0.5, 0.9]),
        ("Atan", &[-5.0, -0.7, 0.0, 1.3, 4.0]),
        ("Asinh", &[-5.0, -0.7, 0.0, 1.3, 4.0]),
        ("Acosh", &[1.0, 1.4, 2.0, 3.5, 6.0]),
        ("Atanh", &[-0.9, -0.4, 0.0, 0.5, 0.85]),
    ];
    for &(op, values) in cases {
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let shape = vec![values.len()];
            let inputs = [Tensor {
                dtype,
                shape: shape.clone(),
                bytes: encode_floats(values, dtype),
            }];
            let outputs = [(dtype, shape.clone())];
            let cuda = run_cuda(&ep, op, 17, &inputs, &outputs, &[]);
            let cpu = run_cpu(op, 17, &inputs, &outputs, &[]);
            let cuda_f = decode_floats(&cuda[0], dtype);
            let cpu_f = decode_floats(&cpu[0], dtype);
            // f32 device intrinsics vs host libm agree to a few ulp; the half
            // formats are dominated by their own rounding step.
            let tolerance = match dtype {
                DataType::Float32 => 1e-4,
                DataType::Float16 => 3e-3,
                DataType::BFloat16 => 3e-2,
                _ => unreachable!(),
            };
            assert_close(op, dtype, &cuda_f, &cpu_f, tolerance);
        }
        eprintln!("{op}: CUDA matches CPU EP across f32/f16/bf16");
    }
}

/// `Identity` and `Flatten` are dtype-agnostic byte copies; assert exact byte
/// equality with the CPU EP for float and integer payloads.
#[test]
fn identity_and_flatten_match_cpu() {
    let Some(ep) = cuda_ep() else { return };
    let f32_values: Vec<f32> = (0..12).map(|v| v as f32 * 0.5 - 3.0).collect();
    let i64_values: Vec<i64> = (0..12).map(|v| (v as i64) * 7 - 5).collect();

    // Identity: same shape in and out.
    for (dtype, bytes) in [
        (DataType::Float32, raw(&f32_values)),
        (DataType::Int64, raw(&i64_values)),
    ] {
        let inputs = [Tensor {
            dtype,
            shape: vec![3, 4],
            bytes,
        }];
        let outputs = [(dtype, vec![3, 4])];
        let cuda = run_cuda(&ep, "Identity", 16, &inputs, &outputs, &[]);
        let cpu = run_cpu("Identity", 16, &inputs, &outputs, &[]);
        assert_eq!(cuda, cpu, "Identity {dtype:?} bytes mismatch");
    }

    // Flatten: axis=2 collapses [2,2,3] -> [4,3] but preserves row-major order.
    let inputs = [input(DataType::Float32, &[2, 2, 3], &f32_values)];
    let outputs = [(DataType::Float32, vec![4, 3])];
    let attrs = [("axis", Attribute::Int(2))];
    let cuda = run_cuda(&ep, "Flatten", 13, &inputs, &outputs, &attrs);
    let cpu = run_cpu("Flatten", 13, &inputs, &outputs, &attrs);
    assert_eq!(cuda, cpu, "Flatten bytes mismatch");
    eprintln!("Identity/Flatten: CUDA matches CPU EP");
}

/// `Size` yields the input's element count as an Int64 scalar.
#[test]
fn size_matches_cpu() {
    let Some(ep) = cuda_ep() else { return };
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let inputs = [input(DataType::Float32, &[2, 3, 4], &values)];
    let outputs = [(DataType::Int64, vec![])];
    let cuda = run_cuda(&ep, "Size", 13, &inputs, &outputs, &[]);
    let cpu = run_cpu("Size", 13, &inputs, &outputs, &[]);
    assert_eq!(cuda, cpu, "Size bytes mismatch");
    let scalar = i64::from_ne_bytes(cuda[0].as_slice().try_into().unwrap());
    assert_eq!(scalar, 24, "Size scalar value");
    eprintln!("Size: CUDA matches CPU EP (= {scalar})");
}

/// `Trilu` upper/lower masks with several diagonal offsets and dtypes, compared
/// byte-for-byte against the CPU EP over batched trailing 2-D matrices. `k` is a
/// runtime Int64 scalar so the CUDA path exercises its device-scalar read.
#[test]
fn trilu_matches_cpu() {
    let Some(ep) = cuda_ep() else { return };
    // A batch of two 3x4 matrices with distinct values.
    let f32_values: Vec<f32> = (0..24).map(|v| v as f32 + 1.0).collect();
    let i64_values: Vec<i64> = (0..24).map(|v| (v as i64) + 1).collect();
    let shape = [2usize, 3, 4];

    for (upper, k) in [
        (1_i64, None),
        (0, None),
        (1, Some(1_i64)),
        (1, Some(-1)),
        (0, Some(2)),
        (0, Some(-2)),
    ] {
        for (dtype, bytes) in [
            (DataType::Float32, raw(&f32_values)),
            (DataType::Int64, raw(&i64_values)),
        ] {
            let mut inputs = vec![Tensor {
                dtype,
                shape: shape.to_vec(),
                bytes,
            }];
            if let Some(k) = k {
                inputs.push(input(DataType::Int64, &[], &[k]));
            }
            let outputs = [(dtype, shape.to_vec())];
            let attrs = [("upper", Attribute::Int(upper))];
            let cuda = run_cuda(&ep, "Trilu", 14, &inputs, &outputs, &attrs);
            let cpu = run_cpu("Trilu", 14, &inputs, &outputs, &attrs);
            assert_eq!(
                cuda, cpu,
                "Trilu upper={upper} k={k:?} {dtype:?} bytes mismatch"
            );
        }
    }
    eprintln!("Trilu: CUDA matches CPU EP across upper/lower, k offsets, f32/i64");
}
