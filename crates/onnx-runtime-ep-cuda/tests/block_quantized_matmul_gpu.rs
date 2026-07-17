use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Dim, Graph, Node, NodeId, SymbolId, compute_contiguous_strides,
    static_shape,
};
use onnx_runtime_loader::Model;

const DOMAIN: &str = "pkg.nxrt";

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

    fn u8(shape: &[usize], values: &[u8]) -> Self {
        Self {
            dtype: DataType::Uint8,
            shape: shape.to_vec(),
            bytes: values.to_vec(),
        }
    }
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

fn model_node(
    format: &str,
    a_shape: &[usize],
    packed_shape: &[usize],
    output_shape: &[usize],
    k: usize,
    n: usize,
    with_bias: bool,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(DOMAIN.into(), 1);
    let a = graph.create_named_value(
        "A",
        DataType::Float32,
        static_shape(a_shape.iter().copied()),
    );
    let packed = graph.create_named_value(
        "packed_B",
        DataType::Uint8,
        static_shape(packed_shape.iter().copied()),
    );
    graph.add_input(a);
    graph.add_input(packed);
    let mut inputs = vec![Some(a), Some(packed)];
    if with_bias {
        let bias = graph.create_named_value("bias", DataType::Float32, static_shape([n]));
        graph.add_input(bias);
        inputs.push(Some(bias));
    }
    let output = graph.create_named_value(
        "Y",
        DataType::Float32,
        static_shape(output_shape.iter().copied()),
    );
    let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", inputs, vec![output]);
    node.domain = DOMAIN.into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes.insert(
        "format".into(),
        Attribute::String(format.as_bytes().to_vec()),
    );
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
}

fn run_cpu(graph: &Graph, node: NodeId, inputs: &[HostTensor], output_shape: &[usize]) -> Vec<f32> {
    let model = Model::new(graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node), &[], 1)
        .unwrap();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .zip(&strides)
        .map(|(input, strides)| {
            TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();
    let output_strides = compute_contiguous_strides(output_shape);
    let mut output = vec![0u8; output_shape.iter().product::<usize>() * 4];
    let output_view = TensorMut::new(
        DevicePtrMut(output.as_mut_ptr().cast()),
        DataType::Float32,
        output_shape,
        &output_strides,
        DeviceId::cpu(),
    );
    kernel.execute(&views, &mut [output_view]).unwrap();
    output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn run_gpu(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_shape: &[usize],
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let model = Model::new(graph);
    let concrete_shapes: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape.clone()).collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes, 1)?;
    let runtime = ep.runtime();
    let mut buffers = Vec::<DeviceBuffer>::new();
    for input in inputs {
        let buffer = ep.allocate(input.bytes.len(), 256)?;
        // SAFETY: each allocation exactly covers its source tensor.
        unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
        buffers.push(buffer);
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect();
    let output_len = output_shape.iter().product::<usize>();
    let mut output_buffer = ep.allocate(output_len * 4, 256)?;
    let output_strides = compute_contiguous_strides(output_shape);
    let output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        output_shape,
        &output_strides,
        ep.device_id(),
    );
    kernel.execute(&views, &mut [output_view])?;
    let mut output = vec![0u8; output_len * 4];
    // SAFETY: the destination exactly covers the f32 output allocation.
    unsafe { runtime.dtoh(&mut output, cuptr(output_buffer.as_ptr()))? };
    drop(views);
    for buffer in buffers {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    Ok(output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn format_info(format: &str) -> (usize, usize) {
    match format {
        "mxfp4" => (32, 17),
        "iq4_nl" => (32, 18),
        "iq4_xs" => (256, 136),
        "iq2_xxs" => (256, 66),
        "iq3_xxs" => (256, 98),
        "iq2_xs" => (256, 74),
        "iq2_s" => (256, 82),
        "iq3_s" => (256, 110),
        "iq1_s" => (256, 50),
        "iq1_m" => (256, 56),
        other => panic!("unknown test format {other}"),
    }
}

fn random_case(format: &str, k: usize, n: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut state = 0x942e_81f5_c3a7_6d0bu64 ^ format.len() as u64;
    let activations = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 3.0)
        .collect();
    let bias = (0..n)
        .map(|_| random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let (qk, block_bytes) = format_info(format);
    let blocks = k.div_ceil(qk);
    let mut packed = vec![0u8; n * blocks * block_bytes];
    for block in packed.chunks_exact_mut(block_bytes) {
        if format == "mxfp4" {
            block[0] = 120 + (random_u32(&mut state) % 16) as u8;
            for byte in &mut block[1..] {
                *byte = random_u32(&mut state) as u8;
            }
        } else if format == "iq1_m" {
            for byte in &mut *block {
                *byte = random_u32(&mut state) as u8;
            }
            let scale =
                half::f16::from_f32(0.002 + random_u32(&mut state) as f32 / u32::MAX as f32 * 0.05);
            for index in 0..4 {
                let offset = 48 + index * 2;
                let packed = u16::from_le_bytes([block[offset], block[offset + 1]]);
                let scale_nibble = (scale.to_bits() >> (4 * index)) & 0x0f;
                block[offset..offset + 2]
                    .copy_from_slice(&((packed & 0x0fff) | (scale_nibble << 12)).to_le_bytes());
            }
        } else {
            let scale =
                half::f16::from_f32(0.002 + random_u32(&mut state) as f32 / u32::MAX as f32 * 0.05);
            block[..2].copy_from_slice(&scale.to_le_bytes());
            for byte in &mut block[2..] {
                *byte = random_u32(&mut state) as u8;
            }
        }
    }
    (activations, packed, bias)
}

fn random_gemm_case(format: &str, m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let (_, packed, bias) = random_case(format, k, n);
    let mut state = 0x6e2a_953d_b47c_018fu64 ^ m as u64 ^ ((k as u64) << 16);
    let activations = (0..m * k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 3.0)
        .collect();
    (activations, packed, bias)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5e-3_f32.max(expected.abs() * 5e-5);
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn block_quantized_gemv_random_supported_formats_match_cpu() {
    let Some(ep) = gpu() else { return };
    let (k, n) = (1003usize, 37usize);
    for format in [
        "mxfp4", "iq4_nl", "iq4_xs", "iq2_xxs", "iq3_xxs", "iq2_xs", "iq2_s", "iq3_s", "iq1_s",
        "iq1_m",
    ] {
        let (qk, block_bytes) = format_info(format);
        let packed_shape = [n, k.div_ceil(qk), block_bytes];
        let (activations, packed, bias) = random_case(format, k, n);
        let inputs = [
            HostTensor::f32(&[1, k], &activations),
            HostTensor::u8(&packed_shape, &packed),
            HostTensor::f32(&[n], &bias),
        ];
        let (graph, node) = model_node(format, &[1, k], &packed_shape, &[1, n], k, n, true);
        let expected = run_cpu(&graph, node, &inputs, &[1, n]);
        let actual = run_gpu(&ep, &graph, node, &inputs, &[1, n]).unwrap();
        assert_close(&actual, &expected);
    }
}

#[test]
fn block_quantized_gemm_prefill_matches_cpu_for_partial_row_tiles() {
    let Some(ep) = gpu() else { return };
    for (format, a_shape, k, n, with_bias) in [
        ("mxfp4", vec![3, 99], 99usize, 13usize, true),
        ("iq4_xs", vec![1, 7, 515], 515usize, 11usize, false),
    ] {
        let m = a_shape[..a_shape.len() - 1].iter().product();
        let (qk, block_bytes) = format_info(format);
        let packed_shape = [n, k.div_ceil(qk), block_bytes];
        let output_shape = [&a_shape[..a_shape.len() - 1], &[n]].concat();
        let (activations, packed, bias) = random_gemm_case(format, m, k, n);
        let mut inputs = vec![
            HostTensor::f32(&a_shape, &activations),
            HostTensor::u8(&packed_shape, &packed),
        ];
        if with_bias {
            inputs.push(HostTensor::f32(&[n], &bias));
        }
        let (graph, node) = model_node(
            format,
            &a_shape,
            &packed_shape,
            &output_shape,
            k,
            n,
            with_bias,
        );
        let expected = run_cpu(&graph, node, &inputs, &output_shape);
        let actual = run_gpu(&ep, &graph, node, &inputs, &output_shape).unwrap();
        assert_close(&actual, &expected);
    }
}

#[test]
fn block_quantized_gemv_dequant_is_bit_exact_against_cpu() {
    let Some(ep) = gpu() else { return };
    let n = 2usize;
    for format in [
        "mxfp4", "iq4_nl", "iq4_xs", "iq2_xxs", "iq3_xxs", "iq2_xs", "iq2_s", "iq3_s", "iq1_s",
        "iq1_m",
    ] {
        let (qk, block_bytes) = format_info(format);
        let k = qk;
        let packed_shape = [n, 1, block_bytes];
        let (_, packed, _) = random_case(format, k, n);
        let (graph, node) = model_node(format, &[1, k], &packed_shape, &[1, n], k, n, false);
        for depth in 0..k {
            let mut activation = vec![0.0f32; k];
            activation[depth] = 1.0;
            let inputs = [
                HostTensor::f32(&[1, k], &activation),
                HostTensor::u8(&packed_shape, &packed),
            ];
            let expected = run_cpu(&graph, node, &inputs, &[1, n]);
            let actual = run_gpu(&ep, &graph, node, &inputs, &[1, n]).unwrap();
            for (column, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{format} depth {depth}, column {column}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn block_quantized_known_blocks_match_cpu_semantics_on_gpu() {
    let Some(ep) = gpu() else { return };
    for (format, packed, depth, expected) in [
        (
            "mxfp4",
            {
                let mut block = vec![0u8; 17];
                block[0] = 128;
                block[1] = 0xd7;
                block
            },
            0usize,
            12.0f32,
        ),
        (
            "mxfp4",
            {
                let mut block = vec![0u8; 17];
                block[0] = 128;
                block[1] = 0xd7;
                block
            },
            16usize,
            -6.0f32,
        ),
        (
            "iq4_nl",
            {
                let mut block = half::f16::from_f32(0.5).to_le_bytes().to_vec();
                block.extend([0xf0]);
                block.resize(18, 0);
                block
            },
            0usize,
            -63.5f32,
        ),
        (
            "iq4_nl",
            {
                let mut block = half::f16::from_f32(0.5).to_le_bytes().to_vec();
                block.extend([0xf0]);
                block.resize(18, 0);
                block
            },
            16usize,
            56.5f32,
        ),
        (
            "iq4_xs",
            {
                let mut block = vec![0u8; 136];
                block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
                block[2] = 2;
                block[4] = 0x22;
                block
            },
            0usize,
            -127.0f32,
        ),
        (
            "iq2_xxs",
            {
                let mut block = vec![0u8; 66];
                block[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
                block
            },
            0usize,
            2.0f32,
        ),
    ] {
        let block_bytes = packed.len();
        let packed_shape = [1, 1, block_bytes];
        let (qk, _) = format_info(format);
        let mut activation = vec![0.0f32; qk];
        activation[depth] = 1.0;
        let inputs = [
            HostTensor::f32(&[1, qk], &activation),
            HostTensor::u8(&packed_shape, &packed),
        ];
        let (graph, node) = model_node(format, &[1, qk], &packed_shape, &[1, 1], qk, 1, false);
        let cpu = run_cpu(&graph, node, &inputs, &[1, 1]);
        let cuda = run_gpu(&ep, &graph, node, &inputs, &[1, 1]).unwrap();
        assert_eq!(cpu, [expected]);
        assert_eq!(cuda, [expected]);
    }
}

#[test]
fn block_quantized_iq1_known_blocks_match_cpu_semantics_on_gpu() {
    let Some(ep) = gpu() else { return };
    let cases = [
        (
            "iq1_s",
            {
                let mut block = vec![0u8; 50];
                block[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
                block[4] = 0xff;
                block[34..36].copy_from_slice(&0xa1c0u16.to_le_bytes());
                block
            },
            [-11.25f32, -11.25, 8.75, -11.25],
        ),
        (
            "iq1_m",
            {
                let mut block = vec![0u8; 56];
                block[1] = 0xff;
                block[2] = 0xff;
                block[32] = 0xf0;
                block[33] = 0x8f;
                block[48..56].copy_from_slice(&[0x1a, 0, 0, 0, 0, 0, 0, 0x40]);
                block
            },
            [-8.75f32, 8.75, 12.25, -15.75],
        ),
    ];
    for (format, packed, expected) in cases {
        let packed_shape = [1, 1, packed.len()];
        let (qk, _) = format_info(format);
        let (graph, node) = model_node(format, &[1, qk], &packed_shape, &[1, 1], qk, 1, false);
        for (depth, expected) in [0usize, 8, 16, 24].into_iter().zip(expected) {
            let mut activation = vec![0.0f32; qk];
            activation[depth] = 1.0;
            let inputs = [
                HostTensor::f32(&[1, qk], &activation),
                HostTensor::u8(&packed_shape, &packed),
            ];
            let cpu = run_cpu(&graph, node, &inputs, &[1, 1]);
            let cuda = run_gpu(&ep, &graph, node, &inputs, &[1, 1]).unwrap();
            assert_eq!(cpu, [expected], "{format} depth {depth} CPU");
            assert_eq!(cuda, [expected], "{format} depth {depth} CUDA");
        }
    }
}

#[test]
fn supported_formats_and_prefill_route_to_cuda() {
    let Some(ep) = gpu() else { return };
    for format in [
        "mxfp4", "iq4_nl", "iq4_xs", "iq2_xxs", "iq3_xxs", "iq2_xs", "iq2_s", "iq3_s", "iq1_s",
        "iq1_m",
    ] {
        let (qk, block_bytes) = format_info(format);
        let (graph, node) = model_node(
            format,
            &[1, qk],
            &[1, 1, block_bytes],
            &[1, 1],
            qk,
            1,
            false,
        );
        let model = Model::new(&graph);
        assert!(matches!(
            ep.supports_op(
                model.graph.node(node),
                &[static_shape([1, qk]), static_shape([1, 1, block_bytes])],
                &[]
            ),
            KernelMatch::Supported { .. }
        ));
    }

    let (graph, node) = model_node("q4_0", &[1, 32], &[1, 1, 18], &[1, 1], 32, 1, false);
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(node),
            &[static_shape([1, 32]), static_shape([1, 1, 18])],
            &[]
        ),
        KernelMatch::Unsupported
    ));

    for format in ["mxfp4", "iq1_s", "iq1_m"] {
        let (qk, block_bytes) = format_info(format);
        let (graph, node) = model_node(
            format,
            &[2, qk],
            &[1, 1, block_bytes],
            &[2, 1],
            qk,
            1,
            false,
        );
        let model = Model::new(&graph);
        assert!(matches!(
            ep.supports_op(
                model.graph.node(node),
                &[static_shape([2, qk]), static_shape([1, 1, block_bytes])],
                &[]
            ),
            KernelMatch::Supported { .. }
        ));
    }

    let (graph, node) = model_node("mxfp4", &[2, 32], &[1, 1, 17], &[2, 1], 32, 1, false);
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(node),
            &[
                vec![Dim::Symbolic(SymbolId(0)), Dim::Static(32)],
                static_shape([1, 1, 17]),
            ],
            &[]
        ),
        KernelMatch::Supported { .. }
    ));
}
