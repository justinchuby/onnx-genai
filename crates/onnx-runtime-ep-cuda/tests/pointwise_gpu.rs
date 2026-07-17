//! GPU-vs-CPU checks for floating pointwise dtype and broadcasting coverage.

use half::{bf16, f16};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape};
use onnx_runtime_loader::Model;

fn encode(values: &[f32], dtype: DataType) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| match dtype {
            DataType::Float16 => f16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
            DataType::BFloat16 => bf16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}

fn decode(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            match dtype {
                DataType::Float16 => f16::from_bits(bits).to_f32(),
                DataType::BFloat16 => bf16::from_bits(bits).to_f32(),
                _ => unreachable!(),
            }
        })
        .collect()
}

fn quantize(values: &[f32], dtype: DataType) -> Vec<f32> {
    decode(&encode(values, dtype), dtype)
}

fn run_binary(
    ep: &CudaExecutionProvider,
    op: &str,
    dtype: DataType,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let rt = ep.runtime();
    let a_bytes = encode(a, dtype);
    let b_bytes = encode(b, dtype);
    let a_buf = ep.allocate(a_bytes.len(), 256).unwrap();
    let b_buf = ep.allocate(b_bytes.len(), 256).unwrap();
    let mut y_buf = ep
        .allocate(out_shape.iter().product::<usize>() * 2, 256)
        .unwrap();
    unsafe {
        rt.htod(&a_bytes, cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(&b_bytes, cuptr(b_buf.as_ptr())).unwrap();
    }

    let a_strides = compute_contiguous_strides(a_shape);
    let b_strides = compute_contiguous_strides(b_shape);
    let y_strides = compute_contiguous_strides(out_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(a_buf.as_ptr()),
            dtype,
            a_shape,
            &a_strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(b_buf.as_ptr()),
            dtype,
            b_shape,
            &b_strides,
            ep.device_id(),
        ),
    ];
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        dtype,
        out_shape,
        &y_strides,
        ep.device_id(),
    );
    let kernel = ep
        .get_kernel(&Node::new(NodeId(0), op, vec![], vec![]), &[], 17)
        .unwrap();
    kernel.execute(&inputs, &mut [output]).unwrap();

    let mut out = vec![0u8; out_shape.iter().product::<usize>() * 2];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    decode(&out, dtype)
}

fn run_unary(ep: &CudaExecutionProvider, op: &str, dtype: DataType, x: &[f32]) -> Vec<f32> {
    let rt = ep.runtime();
    let bytes = encode(x, dtype);
    let x_buf = ep.allocate(bytes.len(), 256).unwrap();
    let mut y_buf = ep.allocate(bytes.len(), 256).unwrap();
    unsafe { rt.htod(&bytes, cuptr(x_buf.as_ptr())).unwrap() };
    let shape = [x.len()];
    let strides = compute_contiguous_strides(&shape);
    let input = TensorView::new(
        DevicePtr(x_buf.as_ptr()),
        dtype,
        &shape,
        &strides,
        ep.device_id(),
    );
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        dtype,
        &shape,
        &strides,
        ep.device_id(),
    );
    let kernel = ep
        .get_kernel(&Node::new(NodeId(0), op, vec![], vec![]), &[], 17)
        .unwrap();
    kernel.execute(&[input], &mut [output]).unwrap();
    let mut out = vec![0u8; bytes.len()];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(x_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    decode(&out, dtype)
}

fn run_predicate(
    ep: &CudaExecutionProvider,
    op: &str,
    dtype: DataType,
    a_bytes: &[u8],
    a_shape: &[usize],
    b_bytes: &[u8],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<u8> {
    let rt = ep.runtime();
    let a_buf = ep.allocate(a_bytes.len(), 256).unwrap();
    let b_buf = ep.allocate(b_bytes.len(), 256).unwrap();
    let mut y_buf = ep.allocate(out_shape.iter().product(), 256).unwrap();
    unsafe {
        rt.htod(a_bytes, cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(b_bytes, cuptr(b_buf.as_ptr())).unwrap();
    }
    let a_strides = compute_contiguous_strides(a_shape);
    let b_strides = compute_contiguous_strides(b_shape);
    let y_strides = compute_contiguous_strides(out_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(a_buf.as_ptr()),
            dtype,
            a_shape,
            &a_strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(b_buf.as_ptr()),
            dtype,
            b_shape,
            &b_strides,
            ep.device_id(),
        ),
    ];
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Bool,
        out_shape,
        &y_strides,
        ep.device_id(),
    );
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a_value = graph.create_named_value("a", dtype, static_shape(a_shape.iter().copied()));
    let b_value = graph.create_named_value("b", dtype, static_shape(b_shape.iter().copied()));
    let y_value =
        graph.create_named_value("y", DataType::Bool, static_shape(out_shape.iter().copied()));
    graph.add_input(a_value);
    graph.add_input(b_value);
    let node_id = graph.insert_node(Node::new(
        NodeId(0),
        op,
        vec![Some(a_value), Some(b_value)],
        vec![y_value],
    ));
    graph.add_output(y_value);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], 17).unwrap();
    kernel.execute(&inputs, &mut [output]).unwrap();
    let mut out = vec![0; out_shape.iter().product()];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    out
}

fn cpu_broadcast(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let strides = |shape: &[usize]| {
        let contiguous = compute_contiguous_strides(shape);
        let leading = out_shape.len() - shape.len();
        (0..out_shape.len())
            .map(|axis| {
                if axis < leading || shape[axis - leading] == 1 {
                    0
                } else {
                    contiguous[axis - leading] as usize
                }
            })
            .collect::<Vec<_>>()
    };
    let a_strides = strides(a_shape);
    let b_strides = strides(b_shape);
    let n = out_shape.iter().product();
    (0..n)
        .map(|index| {
            let mut linear = index;
            let mut ai = 0;
            let mut bi = 0;
            for axis in (0..out_shape.len()).rev() {
                let coord = linear % out_shape[axis];
                linear /= out_shape[axis];
                ai += coord * a_strides[axis];
                bi += coord * b_strides[axis];
            }
            op(a[ai], b[bi])
        })
        .collect()
}

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "index {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
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

#[test]
fn f16_bf16_arithmetic_matches_cpu_compute_domain() {
    let Some(ep) = cuda_ep() else { return };
    let a = [-3.0, -1.5, 0.5, 2.0, 5.0, 8.0];
    let b = [0.5, 2.0, -4.0, 0.25, 2.0, -2.0];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let aq = quantize(&a, dtype);
        let bq = quantize(&b, dtype);
        let tolerance = if dtype == DataType::Float16 {
            0.002
        } else {
            0.04
        };
        for (op, f) in [
            ("Add", (|x, y| x + y) as fn(f32, f32) -> f32),
            ("Sub", (|x, y| x - y) as fn(f32, f32) -> f32),
            ("Mul", (|x, y| x * y) as fn(f32, f32) -> f32),
            ("Div", (|x, y| x / y) as fn(f32, f32) -> f32),
        ] {
            let expected = quantize(
                &aq.iter()
                    .zip(&bq)
                    .map(|(&x, &y)| f(x, y))
                    .collect::<Vec<_>>(),
                dtype,
            );
            let got = run_binary(&ep, op, dtype, &a, &[6], &b, &[6], &[6]);
            assert_close(&got, &expected, tolerance);
        }
    }
}

#[test]
fn f16_bf16_numpy_broadcast_matches_cpu_reference() {
    let Some(ep) = cuda_ep() else { return };
    let a = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect::<Vec<_>>();
    let b = (0..15).map(|i| i as f32 * 0.1 + 0.5).collect::<Vec<_>>();
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let aq = quantize(&a, dtype);
        let bq = quantize(&b, dtype);
        let expected = quantize(
            &cpu_broadcast(&aq, &[4, 1, 3], &bq, &[1, 5, 3], &[4, 5, 3], |x, y| x + y),
            dtype,
        );
        let got = run_binary(
            &ep,
            "Add",
            dtype,
            &a,
            &[4, 1, 3],
            &b,
            &[1, 5, 3],
            &[4, 5, 3],
        );
        assert_close(
            &got,
            &expected,
            if dtype == DataType::Float16 {
                0.002
            } else {
                0.04
            },
        );
    }
}

#[test]
fn half_unary_and_activation_families_match_cpu_reference() {
    let Some(ep) = cuda_ep() else { return };
    let x = [-3.0, -1.0, -0.0, 0.5, 2.0];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let xq = quantize(&x, dtype);
        let tolerance = if dtype == DataType::Float16 {
            0.003
        } else {
            0.04
        };
        for (op, f) in [
            (
                "Relu",
                (|v: f32| if v.is_nan() { v } else { v.max(0.0) }) as fn(f32) -> f32,
            ),
            ("Exp", (|v: f32| v.exp()) as fn(f32) -> f32),
            (
                "LeakyRelu",
                (|v: f32| if v >= 0.0 { v } else { 0.01 * v }) as fn(f32) -> f32,
            ),
        ] {
            let expected = quantize(&xq.iter().copied().map(f).collect::<Vec<_>>(), dtype);
            assert_close(&run_unary(&ep, op, dtype, &x), &expected, tolerance);
        }
    }
}

#[test]
fn logical_family_numpy_broadcast_matches_cpu_reference() {
    let Some(ep) = cuda_ep() else { return };
    let a = [0_u8, 1];
    let b = [0_u8, 1, 1];
    let expected = [
        ("And", vec![0, 0, 0, 0, 1, 1]),
        ("Or", vec![0, 1, 1, 1, 1, 1]),
        ("Xor", vec![0, 1, 1, 1, 0, 0]),
    ];
    for (op, expected) in expected {
        assert_eq!(
            run_predicate(&ep, op, DataType::Bool, &a, &[2, 1], &b, &[1, 3], &[2, 3]),
            expected,
            "{op}"
        );
    }
}

#[test]
fn comparison_family_numpy_broadcast_matches_cpu_reference() {
    let Some(ep) = cuda_ep() else { return };
    let a = [1.0_f32, 3.0];
    let b = [2.0_f32, 3.0, 4.0];
    let a_bytes =
        unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), std::mem::size_of_val(&a)) };
    let b_bytes =
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u8>(), std::mem::size_of_val(&b)) };
    let expected = [
        ("Equal", vec![0, 0, 0, 0, 1, 0]),
        ("Greater", vec![0, 0, 0, 1, 0, 0]),
        ("Less", vec![1, 1, 1, 0, 0, 1]),
        ("GreaterOrEqual", vec![0, 0, 0, 1, 1, 0]),
        ("LessOrEqual", vec![1, 1, 1, 0, 1, 1]),
    ];
    for (op, expected) in expected {
        assert_eq!(
            run_predicate(
                &ep,
                op,
                DataType::Float32,
                a_bytes,
                &[2, 1],
                b_bytes,
                &[1, 3],
                &[2, 3],
            ),
            expected,
            "{op}"
        );
    }
}
