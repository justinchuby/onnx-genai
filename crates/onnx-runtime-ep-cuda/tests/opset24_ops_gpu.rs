//! CUDA correctness checks for standard ops needed by opset-24 models.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, TensorData, compute_contiguous_strides};

struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: the test's numeric input types are plain data.
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

fn run(
    ep: &CudaExecutionProvider,
    node: &Node,
    inputs: &[Tensor],
    output_dtype: DataType,
    output_shape: &[usize],
) -> Vec<u8> {
    let input_buffers = inputs
        .iter()
        .map(|input| {
            let buffer = ep.allocate(input.bytes.len().max(1), 256).unwrap();
            if !input.bytes.is_empty() {
                // SAFETY: the device allocation covers the source bytes.
                unsafe {
                    ep.runtime()
                        .htod(&input.bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            buffer
        })
        .collect::<Vec<DeviceBuffer>>();
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

    let output_bytes = output_dtype.storage_bytes(output_shape.iter().product());
    let mut output_buffer = ep.allocate(output_bytes.max(1), 256).unwrap();
    let output_strides = compute_contiguous_strides(output_shape);
    let output = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        output_dtype,
        output_shape,
        &output_strides,
        ep.device_id(),
    );
    let kernel = ep.get_kernel(node, &[], 24).unwrap();
    kernel.execute(&input_views, &mut [output]).unwrap();

    let mut bytes = vec![0_u8; output_bytes];
    if !bytes.is_empty() {
        // SAFETY: the host destination matches the device output size.
        unsafe {
            ep.runtime()
                .dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))
                .unwrap()
        };
    }
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output_buffer).unwrap();
    bytes
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn constantofshape_opset24_fills_float_and_int_outputs() {
    let Some(ep) = cuda_ep() else { return };
    let shape = tensor(DataType::Int64, &[2], &[2_i64, 3]);

    let mut float_node = Node::new(NodeId(0), "ConstantOfShape", vec![], vec![]);
    float_node.attributes.insert(
        "value".into(),
        Attribute::Tensor(TensorData::from_raw(
            DataType::Float32,
            vec![1],
            1.25_f32.to_ne_bytes().to_vec(),
        )),
    );
    assert_eq!(
        decode_f32(&run(&ep, &float_node, &[shape], DataType::Float32, &[2, 3])),
        vec![1.25; 6]
    );

    let mut int_node = Node::new(NodeId(0), "ConstantOfShape", vec![], vec![]);
    int_node.attributes.insert(
        "value".into(),
        Attribute::Tensor(TensorData::from_raw(
            DataType::Int32,
            vec![1],
            (-7_i32).to_ne_bytes().to_vec(),
        )),
    );
    let int_shape = tensor(DataType::Int64, &[1], &[4_i64]);
    let got = run(&ep, &int_node, &[int_shape], DataType::Int32, &[4]);
    assert_eq!(
        got.chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>(),
        vec![-7; 4]
    );
}

fn gelu_exact(x: f32) -> f32 {
    let z = (x / std::f32::consts::SQRT_2).abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let polynomial =
        (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    let erf = (1.0 - polynomial * (-z * z).exp()).copysign(x);
    0.5 * x * (1.0 + erf)
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x
        * (1.0
            + (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2
                * (x + 0.044715 * x * x * x))
                .tanh())
}

#[test]
fn gelu_opset24_exact_and_tanh_include_fp16_and_bf16() {
    let Some(ep) = cuda_ep() else { return };
    let x = [-3.0_f32, -1.0, -0.25, 0.0, 0.5, 2.0];

    let exact_node = Node::new(NodeId(0), "Gelu", vec![], vec![]);
    let got = decode_f32(&run(
        &ep,
        &exact_node,
        &[tensor(DataType::Float32, &[x.len()], &x)],
        DataType::Float32,
        &[x.len()],
    ));
    for (&got, want) in got.iter().zip(x.map(gelu_exact)) {
        assert!((got - want).abs() <= 2e-6, "got {got}, want {want}");
    }

    let mut tanh_node = Node::new(NodeId(0), "Gelu", vec![], vec![]);
    tanh_node
        .attributes
        .insert("approximate".into(), Attribute::String(b"tanh".to_vec()));
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let bytes = x
            .iter()
            .flat_map(|&value| match dtype {
                DataType::Float16 => f16::from_f32(value).to_bits().to_ne_bytes(),
                DataType::BFloat16 => bf16::from_f32(value).to_bits().to_ne_bytes(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let got = run(
            &ep,
            &tanh_node,
            &[Tensor {
                dtype,
                shape: vec![x.len()],
                bytes,
            }],
            dtype,
            &[x.len()],
        )
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_ne_bytes(bytes.try_into().unwrap());
            match dtype {
                DataType::Float16 => f16::from_bits(bits).to_f32(),
                DataType::BFloat16 => bf16::from_bits(bits).to_f32(),
                _ => unreachable!(),
            }
        })
        .collect::<Vec<_>>();
        for (&got, &input) in got.iter().zip(&x) {
            let input = match dtype {
                DataType::Float16 => f16::from_f32(input).to_f32(),
                DataType::BFloat16 => bf16::from_f32(input).to_f32(),
                _ => unreachable!(),
            };
            let want = match dtype {
                DataType::Float16 => f16::from_f32(gelu_tanh(input)).to_f32(),
                DataType::BFloat16 => bf16::from_f32(gelu_tanh(input)).to_f32(),
                _ => unreachable!(),
            };
            let tolerance = if dtype == DataType::Float16 {
                0.002
            } else {
                0.02
            };
            assert!((got - want).abs() <= tolerance, "got {got}, want {want}");
        }
    }
}

#[test]
fn onehot_opset24_matches_scalar_oracle_with_negative_indices() {
    let Some(ep) = cuda_ep() else { return };
    let mut node = Node::new(NodeId(0), "OneHot", vec![], vec![]);
    node.attributes.insert("axis".into(), Attribute::Int(1));
    let got = decode_f32(&run(
        &ep,
        &node,
        &[
            tensor(DataType::Int32, &[2, 2], &[0_i32, -1, 3, -4]),
            tensor(DataType::Int64, &[], &[3_i64]),
            tensor(DataType::Float32, &[2], &[-1_f32, 5.0]),
        ],
        DataType::Float32,
        &[2, 3, 2],
    ));
    assert_eq!(
        got,
        vec![
            5.0, -1.0, -1.0, -1.0, -1.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
        ]
    );
}
