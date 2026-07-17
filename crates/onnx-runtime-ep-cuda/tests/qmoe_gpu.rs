use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

#[derive(Clone)]
struct HostTensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl HostTensor {
    fn f32(shape: &[usize], values: &[f32]) -> Self {
        Self {
            dtype: DataType::Float32,
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
        }
    }

    fn u8(shape: &[usize], values: Vec<u8>) -> Self {
        Self {
            dtype: DataType::Uint8,
            shape: shape.to_vec(),
            bytes: values,
        }
    }

    fn activation(dtype: DataType, shape: &[usize], values: &[f32]) -> Self {
        let bytes = match dtype {
            DataType::Float32 => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            DataType::Float16 => values
                .iter()
                .flat_map(|value| f16::from_f32(*value).to_bits().to_ne_bytes())
                .collect(),
            DataType::BFloat16 => values
                .iter()
                .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
                .collect(),
            other => panic!("unsupported activation dtype {other:?}"),
        };
        Self {
            dtype,
            shape: shape.to_vec(),
            bytes,
        }
    }
}

#[derive(Clone)]
struct Quantized {
    packed: HostTensor,
    scales: HostTensor,
    zero_points: Option<HostTensor>,
}

#[allow(clippy::too_many_arguments)]
fn quantize(
    experts: usize,
    out_features: usize,
    in_features: usize,
    bits: usize,
    block_size: usize,
    affine: bool,
    seed: usize,
) -> Quantized {
    let pack_size = 8 / bits;
    let blocks = in_features / block_size;
    let packed_in = in_features / pack_size;
    let zero_point_bytes = blocks.div_ceil(pack_size);
    let mask = if bits == 8 {
        u8::MAX
    } else {
        (1u8 << bits) - 1
    };
    let default_zero = 1u8 << (bits - 1);
    let mut packed = vec![0u8; experts * out_features * packed_in];
    let mut scales = vec![0.0f32; experts * out_features * blocks];
    let mut zero_points = affine.then(|| vec![0u8; experts * out_features * zero_point_bytes]);
    for expert in 0..experts {
        for output in 0..out_features {
            let expert_row = expert * out_features + output;
            for block in 0..blocks {
                let scale =
                    0.025 + 0.0125 * ((seed + expert * 3 + output * 5 + block * 7) % 5) as f32;
                scales[expert_row * blocks + block] = scale;
                let zero = if affine {
                    default_zero.saturating_sub(
                        ((seed + expert + output + block) % 3).min(default_zero as usize) as u8,
                    )
                } else {
                    default_zero
                };
                if let Some(points) = &mut zero_points {
                    points[expert_row * zero_point_bytes + block / pack_size] |=
                        zero << ((block % pack_size) * bits);
                }
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let span = if bits == 8 { 31 } else { 7 };
                    let centered = ((seed + expert * 11 + output * 13 + depth * 17) % span) as i16
                        - (span / 2) as i16;
                    let quantized = (centered + i16::from(zero)).clamp(0, i16::from(mask)) as u8;
                    packed[expert_row * packed_in + depth / pack_size] |=
                        quantized << ((depth % pack_size) * bits);
                }
            }
        }
    }
    Quantized {
        packed: HostTensor::u8(&[experts, out_features, packed_in], packed),
        scales: HostTensor::f32(&[experts, out_features, blocks], &scales),
        zero_points: zero_points
            .map(|points| HostTensor::u8(&[experts, out_features, zero_point_bytes], points)),
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    experts: usize,
    rows: usize,
    hidden: usize,
    inter: usize,
    bits: usize,
    top_k: usize,
    activation: &'static str,
    swiglu_fusion: usize,
    affine: bool,
    fc3: bool,
    biases: bool,
    normalize: bool,
    router_weights: bool,
}

fn case_inputs(case: Case, dtype: DataType) -> Vec<Option<HostTensor>> {
    let fc1_size = if case.activation == "swiglu" && case.swiglu_fusion != 0 {
        case.inter * 2
    } else {
        case.inter
    };
    let x: Vec<f32> = (0..case.rows * case.hidden)
        .map(|index| ((index * 19 + 3) % 29) as f32 / 13.0 - 1.0)
        .collect();
    let router: Vec<f32> = (0..case.rows * case.experts)
        .map(|index| ((index * 7 + 5) % 17) as f32 / 4.0 - 2.0)
        .collect();
    let aggregation: Vec<f32> = (0..case.rows * case.experts)
        .map(|index| 0.1 + ((index * 5 + 2) % 11) as f32 / 10.0)
        .collect();
    let fc1 = quantize(
        case.experts,
        fc1_size,
        case.hidden,
        case.bits,
        16,
        case.affine,
        1,
    );
    let fc2 = quantize(
        case.experts,
        case.hidden,
        case.inter,
        case.bits,
        16,
        case.affine,
        2,
    );
    match (case.activation, case.swiglu_fusion) {
        ("swiglu", 0) => assert!(case.fc3, "unfused SwiGLU requires FC3"),
        ("swiglu", _) => assert!(!case.fc3, "fused SwiGLU must not provide FC3"),
        ("silu", 0) => {}
        _ => assert!(!case.fc3, "FC3 is only valid for SwiGLU or gated SiLU"),
    }
    let fc3 = case.fc3.then(|| {
        quantize(
            case.experts,
            case.inter,
            case.hidden,
            case.bits,
            16,
            case.affine,
            3,
        )
    });
    let bias = |width: usize, seed: usize| {
        let values: Vec<f32> = (0..case.experts * width)
            .map(|index| ((index * 3 + seed) % 7) as f32 * 0.01 - 0.03)
            .collect();
        HostTensor::f32(&[case.experts, width], &values)
    };
    vec![
        Some(HostTensor::activation(dtype, &[case.rows, case.hidden], &x)),
        Some(HostTensor::f32(&[case.rows, case.experts], &router)),
        Some(fc1.packed),
        Some(fc1.scales),
        case.biases.then(|| bias(fc1_size, 1)),
        Some(fc2.packed),
        Some(fc2.scales),
        case.biases.then(|| bias(case.hidden, 2)),
        fc3.as_ref().map(|weights| weights.packed.clone()),
        fc3.as_ref().map(|weights| weights.scales.clone()),
        (case.biases && case.fc3).then(|| bias(case.inter, 3)),
        fc1.zero_points,
        fc2.zero_points,
        fc3.and_then(|weights| weights.zero_points),
        case.router_weights
            .then(|| HostTensor::f32(&[case.rows, case.experts], &aggregation)),
    ]
}

fn model_node(
    inputs: &[Option<HostTensor>],
    output_dtype: DataType,
    output_shape: &[usize],
    case: Case,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let values = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            input.as_ref().map(|input| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect();
    let output = graph.create_named_value(
        "output",
        output_dtype,
        static_shape(output_shape.iter().copied()),
    );
    let mut node = Node::new(NodeId(0), "QMoE", values, vec![output]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("expert_weight_bits", Attribute::Int(case.bits as i64)),
        ("block_size", Attribute::Int(16)),
        ("k", Attribute::Int(case.top_k as i64)),
        (
            "activation_type",
            Attribute::String(case.activation.as_bytes().to_vec()),
        ),
        (
            "normalize_routing_weights",
            Attribute::Int(i64::from(case.normalize)),
        ),
        ("swiglu_fusion", Attribute::Int(case.swiglu_fusion as i64)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    node.attributes
        .insert("activation_alpha".into(), Attribute::Float(1.125));
    node.attributes
        .insert("activation_beta".into(), Attribute::Float(-0.0625));
    node.attributes
        .insert("swiglu_limit".into(), Attribute::Float(4.0));
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
}

fn absent_dtype(index: usize, activation_dtype: DataType) -> DataType {
    match index {
        0 => activation_dtype,
        2 | 5 | 8 | 11 | 12 | 13 => DataType::Uint8,
        _ => DataType::Float32,
    }
}

fn run_cpu(case: Case, inputs: &[Option<HostTensor>]) -> Vec<f32> {
    let output_shape = [case.rows, case.hidden];
    let (graph, node) = model_node(inputs, DataType::Float32, &output_shape, case);
    let model = Model::new(&graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node), &[], 1)
        .unwrap();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .zip(&strides)
        .enumerate()
        .map(|(index, (input, strides))| match (input, strides) {
            (Some(input), Some(strides)) => TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            ),
            _ => TensorView::absent(absent_dtype(index, DataType::Float32)),
        })
        .collect();
    let mut output = vec![0u8; case.rows * case.hidden * 4];
    let output_strides = compute_contiguous_strides(&output_shape);
    kernel
        .execute(
            &views,
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr().cast()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                DeviceId::cpu(),
            )],
        )
        .unwrap();
    output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn gpu() -> Option<CudaExecutionProvider> {
    match CudaExecutionProvider::new_default() {
        Ok(ep) => Some(ep),
        Err(error) => {
            eprintln!("skip: no CUDA GPU available ({error})");
            None
        }
    }
}

fn run_gpu(
    ep: &CudaExecutionProvider,
    case: Case,
    inputs: &[Option<HostTensor>],
    dtype: DataType,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    run_gpu_with_prefill_min_tokens(ep, case, inputs, dtype, None)
}

fn run_gpu_with_prefill_min_tokens(
    ep: &CudaExecutionProvider,
    case: Case,
    inputs: &[Option<HostTensor>],
    dtype: DataType,
    prefill_min_tokens: Option<usize>,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let output_shape = [case.rows, case.hidden];
    let (mut graph, node) = model_node(inputs, dtype, &output_shape, case);
    if let Some(prefill_min_tokens) = prefill_min_tokens {
        graph.node_mut(node).attributes.insert(
            "prefill_min_tokens".into(),
            Attribute::Int(prefill_min_tokens as i64),
        );
    }
    let model = Model::new(&graph);
    let concrete_shapes: Vec<_> = inputs
        .iter()
        .filter_map(|input| input.as_ref().map(|input| input.shape.clone()))
        .collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes, 1)?;
    let runtime = ep.runtime();
    let mut buffers = Vec::<Option<DeviceBuffer>>::new();
    for input in inputs {
        if let Some(input) = input {
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            // SAFETY: allocation size equals the source tensor byte length.
            unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            buffers.push(Some(buffer));
        } else {
            buffers.push(None);
        }
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .enumerate()
        .map(
            |(index, ((input, buffer), strides))| match (input, buffer, strides) {
                (Some(input), Some(buffer), Some(strides)) => TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    input.dtype,
                    &input.shape,
                    strides,
                    ep.device_id(),
                ),
                _ => TensorView::absent(absent_dtype(index, dtype)),
            },
        )
        .collect();
    let output_bytes = case.rows * case.hidden * dtype.byte_size();
    let mut output_buffer = ep.allocate(output_bytes, 256)?;
    let output_strides = compute_contiguous_strides(&output_shape);
    kernel.execute(
        &views,
        &mut [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            dtype,
            &output_shape,
            &output_strides,
            ep.device_id(),
        )],
    )?;
    let mut bytes = vec![0u8; output_bytes];
    // SAFETY: output allocation contains exactly the requested output tensor.
    unsafe { runtime.dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))? };
    drop(views);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    Ok(match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported output dtype {other:?}"),
    })
}

fn rounded_cpu_inputs(inputs: &[Option<HostTensor>], dtype: DataType) -> Vec<Option<HostTensor>> {
    let mut rounded = inputs.to_vec();
    let activation = rounded[0].as_ref().unwrap();
    let values: Vec<f32> = match dtype {
        DataType::Float32 => activation
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => activation
            .bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => activation
            .bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    };
    rounded[0] = Some(HostTensor::f32(&activation.shape, &values));
    rounded
}

fn assert_conforms(actual: &[f32], expected: &[f32], case: Case, dtype: DataType) {
    assert_eq!(actual.len(), expected.len());
    let (absolute, relative) = match dtype {
        DataType::Float32 => (2e-5, 1e-4),
        DataType::Float16 => (2e-5, 6e-4),
        DataType::BFloat16 => (2e-5, 4.1e-3),
        other => panic!("unsupported dtype {other:?}"),
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = absolute + relative * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}, \
             absolute={absolute}, relative={relative}, dtype={dtype:?}, case={case:?}"
        );
    }
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> (f32, u32) {
    actual.iter().zip(expected).fold(
        (0.0f32, 0u32),
        |(max_abs, max_ulp), (&actual, &expected)| {
            let actual_key = if actual.is_sign_negative() {
                !actual.to_bits()
            } else {
                actual.to_bits() | 0x8000_0000
            };
            let expected_key = if expected.is_sign_negative() {
                !expected.to_bits()
            } else {
                expected.to_bits() | 0x8000_0000
            };
            (
                max_abs.max((actual - expected).abs()),
                max_ulp.max(actual_key.abs_diff(expected_key)),
            )
        },
    )
}

fn compare(case: Case, dtype: DataType) -> (f32, u32) {
    let Some(ep) = gpu() else { return (0.0, 0) };
    let gpu_inputs = case_inputs(case, dtype);
    let cpu_inputs = rounded_cpu_inputs(&gpu_inputs, dtype);
    let expected = run_cpu(case, &cpu_inputs);
    let actual = run_gpu(&ep, case, &gpu_inputs, dtype).unwrap();
    assert_conforms(&actual, &expected, case, dtype);
    error_metrics(&actual, &expected)
}

fn compare_gemv_gemm_and_cpu(case: Case) {
    assert_eq!(case.rows, 6);
    assert_eq!(case.experts, 4);
    assert_eq!(case.top_k, 2);
    let Some(ep) = gpu() else { return };
    let mut inputs = case_inputs(case, DataType::Float32);
    inputs[1] = Some(HostTensor::f32(
        &[case.rows, case.experts],
        &[
            9.0, 8.0, 0.0, -1.0, 8.0, 9.0, -1.0, 0.0, 9.0, 7.0, 1.0, 0.0, 0.0, -1.0, 9.0, 8.0,
            -1.0, 0.0, 8.0, 9.0, 1.0, 0.0, 9.0, 7.0,
        ],
    ));
    let expected = run_cpu(case, &inputs);
    let gemm =
        run_gpu_with_prefill_min_tokens(&ep, case, &inputs, DataType::Float32, Some(2)).unwrap();
    let gemv =
        run_gpu_with_prefill_min_tokens(&ep, case, &inputs, DataType::Float32, Some(1024)).unwrap();
    assert_conforms(&gemm, &gemv, case, DataType::Float32);
    assert_conforms(&gemm, &expected, case, DataType::Float32);
    assert_conforms(&gemv, &expected, case, DataType::Float32);
}

fn activation_case(activation: &'static str, swiglu_fusion: usize, fc3: bool) -> Case {
    Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation,
        swiglu_fusion,
        affine: true,
        fc3,
        biases: false,
        normalize: true,
        router_weights: false,
    }
}

macro_rules! activation_path_test {
    ($name:ident, $activation:literal, $fusion:expr, $separate_gate:expr) => {
        #[test]
        fn $name() {
            compare_gemv_gemm_and_cpu(activation_case($activation, $fusion, $separate_gate));
        }
    };
}

activation_path_test!(qmoe_relu_gemv_gemm_matches_cpu, "relu", 0, false);
activation_path_test!(qmoe_gelu_gemv_gemm_matches_cpu, "gelu", 0, false);
activation_path_test!(qmoe_silu_gemv_gemm_matches_cpu, "silu", 0, false);
activation_path_test!(qmoe_silu_gated_gemv_gemm_matches_cpu, "silu", 0, true);
activation_path_test!(qmoe_swiglu_unfused_gemv_gemm_matches_cpu, "swiglu", 0, true);
activation_path_test!(
    qmoe_swiglu_interleaved_gemv_gemm_matches_cpu,
    "swiglu",
    1,
    false
);
activation_path_test!(qmoe_swiglu_split_gemv_gemm_matches_cpu, "swiglu", 2, false);
activation_path_test!(qmoe_identity_gemv_gemm_matches_cpu, "identity", 0, false);

#[test]
fn qmoe_biases_gemv_gemm_match_cpu() {
    compare_gemv_gemm_and_cpu(Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "identity",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: true,
        router_weights: false,
    });
}

#[test]
fn qmoe_separate_router_weights_gemv_gemm_match_cpu() {
    compare_gemv_gemm_and_cpu(Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "identity",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: false,
        normalize: true,
        router_weights: true,
    });
}

#[test]
fn qmoe_glm_silu_fc3_biases_separate_router_matches_cpu_all_dtypes() {
    let case = Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: true,
        biases: true,
        normalize: true,
        router_weights: true,
    };
    compare_gemv_gemm_and_cpu(case);
    compare(case, DataType::Float16);
    compare(case, DataType::BFloat16);
}

macro_rules! sub_byte_path_test {
    ($name:ident, $bits:expr, $affine:expr) => {
        #[test]
        fn $name() {
            compare_gemv_gemm_and_cpu(Case {
                experts: 4,
                rows: 6,
                hidden: 16,
                inter: 16,
                bits: $bits,
                top_k: 2,
                activation: "identity",
                swiglu_fusion: 0,
                affine: $affine,
                fc3: false,
                biases: true,
                normalize: true,
                router_weights: false,
            });
        }
    };
}

sub_byte_path_test!(qmoe_int1_symmetric_gemv_gemm_matches_cpu, 1, false);
sub_byte_path_test!(qmoe_int1_affine_gemv_gemm_matches_cpu, 1, true);
sub_byte_path_test!(qmoe_int2_symmetric_gemv_gemm_matches_cpu, 2, false);
sub_byte_path_test!(qmoe_int2_affine_gemv_gemm_matches_cpu, 2, true);

#[test]
fn qmoe_int4_top2_symmetric_matches_cpu() {
    let (max_abs, max_ulp) = compare(
        Case {
            experts: 4,
            rows: 3,
            hidden: 16,
            inter: 16,
            bits: 4,
            top_k: 2,
            activation: "identity",
            swiglu_fusion: 0,
            affine: false,
            fc3: false,
            biases: false,
            normalize: true,
            router_weights: false,
        },
        DataType::Float32,
    );
    eprintln!("QMoE int4 top-2 CPU/CUDA max_abs_diff={max_abs:e} max_ulp_diff={max_ulp}");
}

#[test]
fn qmoe_int8_top1_affine_bias_matches_cpu() {
    compare(
        Case {
            experts: 4,
            rows: 2,
            hidden: 16,
            inter: 16,
            bits: 8,
            top_k: 1,
            activation: "relu",
            swiglu_fusion: 0,
            affine: true,
            fc3: false,
            biases: true,
            normalize: false,
            router_weights: true,
        },
        DataType::Float32,
    );
}

#[test]
fn qmoe_single_expert_top1_matches_cpu() {
    compare(
        Case {
            experts: 1,
            rows: 2,
            hidden: 16,
            inter: 16,
            bits: 4,
            top_k: 1,
            activation: "gelu",
            swiglu_fusion: 0,
            affine: true,
            fc3: false,
            biases: true,
            normalize: true,
            router_weights: false,
        },
        DataType::Float32,
    );
}

#[test]
fn qmoe_fp16_and_bf16_storage_match_rounded_cpu_reference() {
    let case = Case {
        experts: 4,
        rows: 2,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: false,
        router_weights: false,
    };
    compare(case, DataType::Float16);
    compare(case, DataType::BFloat16);
}

#[test]
fn qmoe_prefill_gemm_matches_gemv_and_cpu_oracle() {
    let case = Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: true,
        router_weights: false,
    };
    compare_gemv_gemm_and_cpu(case);
}

#[test]
fn qmoe_prefill_handles_empty_experts_and_all_routes_to_one_expert() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        experts: 4,
        rows: 5,
        hidden: 16,
        inter: 16,
        bits: 8,
        top_k: 1,
        activation: "identity",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: false,
        router_weights: true,
    };
    let mut inputs = case_inputs(case, DataType::Float32);
    inputs[1] = Some(HostTensor::f32(
        &[case.rows, case.experts],
        &[
            0.0, 1.0, 9.0, -1.0, 1.0, 0.0, 8.0, -1.0, -1.0, 0.0, 7.0, 1.0, 0.0, -1.0, 9.0, 1.0,
            1.0, 0.0, 8.0, -1.0,
        ],
    ));

    let expected = run_cpu(case, &inputs);
    let actual = run_gpu(&ep, case, &inputs, DataType::Float32).unwrap();
    assert_conforms(&actual, &expected, case, DataType::Float32);
}
