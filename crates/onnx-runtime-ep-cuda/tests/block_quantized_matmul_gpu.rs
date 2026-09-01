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
use std::io::{Read, Seek, SeekFrom};

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

    fn raw(dtype: DataType, shape: &[usize], values: &[u8]) -> Self {
        Self {
            dtype,
            shape: shape.to_vec(),
            bytes: values.to_vec(),
        }
    }
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
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
    let mut inputs = vec![Some(a), Some(packed), None, None];
    if with_bias {
        let bias = graph.create_named_value("bias", DataType::Float32, static_shape([n]));
        graph.add_input(bias);
        inputs[3] = Some(bias);
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
    let compact_views: Vec<_> = inputs
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
    let views = [
        compact_views[0],
        compact_views[1],
        TensorView::absent(DataType::Float8E8M0),
        compact_views
            .get(2)
            .copied()
            .unwrap_or_else(|| TensorView::absent(DataType::Float32)),
    ];
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
    let concrete_shapes = vec![
        inputs[0].shape.clone(),
        inputs[1].shape.clone(),
        vec![],
        inputs
            .get(2)
            .map_or_else(Vec::new, |input| input.shape.clone()),
    ];
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
    let compact_views: Vec<_> = inputs
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
    let views = [
        compact_views[0],
        compact_views[1],
        TensorView::absent(DataType::Float8E8M0),
        compact_views
            .get(2)
            .copied()
            .unwrap_or_else(|| TensorView::absent(DataType::Float32)),
    ];
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
    for buffer in buffers {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    Ok(output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn planar_model_node(
    format: &str,
    a_shape: &[usize],
    packed: &HostTensor,
    scale: &HostTensor,
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
    let weight = graph.create_named_value(
        "packed_B",
        packed.dtype,
        static_shape(packed.shape.iter().copied()),
    );
    let aux_scale = graph.create_named_value(
        "aux_scale_B",
        scale.dtype,
        static_shape(scale.shape.iter().copied()),
    );
    for value in [a, weight, aux_scale] {
        graph.add_input(value);
    }
    let mut inputs = vec![Some(a), Some(weight), Some(aux_scale), None];
    if with_bias {
        let bias = graph.create_named_value("bias", DataType::Float32, static_shape([n]));
        graph.add_input(bias);
        inputs[3] = Some(bias);
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
    let (block_out, block_in) = if format == "fp4_planar" {
        (1, 32)
    } else {
        (n.div_ceil(scale.shape[0]), k.div_ceil(scale.shape[1]))
    };
    node.attributes
        .insert("block_size_out".into(), Attribute::Int(block_out as i64));
    node.attributes
        .insert("block_size_in".into(), Attribute::Int(block_in as i64));
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
}

fn run_planar_cpu(
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_shape: &[usize],
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let model = Model::new(graph);
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node), &[], 1)
        .unwrap();
    kernel.set_constant_inputs(&[false, true, true, false]);
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let views = [
        TensorView::new(
            DevicePtr(inputs[0].bytes.as_ptr().cast()),
            inputs[0].dtype,
            &inputs[0].shape,
            &strides[0],
            DeviceId::cpu(),
        ),
        TensorView::new(
            DevicePtr(inputs[1].bytes.as_ptr().cast()),
            inputs[1].dtype,
            &inputs[1].shape,
            &strides[1],
            DeviceId::cpu(),
        ),
        TensorView::new(
            DevicePtr(inputs[2].bytes.as_ptr().cast()),
            inputs[2].dtype,
            &inputs[2].shape,
            &strides[2],
            DeviceId::cpu(),
        ),
        inputs.get(3).map_or_else(
            || TensorView::absent(DataType::Float32),
            |bias| {
                TensorView::new(
                    DevicePtr(bias.bytes.as_ptr().cast()),
                    bias.dtype,
                    &bias.shape,
                    &strides[3],
                    DeviceId::cpu(),
                )
            },
        ),
    ];
    let output_strides = compute_contiguous_strides(output_shape);
    let mut output = vec![0u8; output_shape.iter().product::<usize>() * 4];
    kernel.execute(
        &views,
        &mut [TensorMut::new(
            DevicePtrMut(output.as_mut_ptr().cast()),
            DataType::Float32,
            output_shape,
            &output_strides,
            DeviceId::cpu(),
        )],
    )?;
    Ok(output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn run_planar_gpu(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_shape: &[usize],
    graph_replays: usize,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let model = Model::new(graph);
    let concrete_shapes = vec![
        inputs[0].shape.clone(),
        inputs[1].shape.clone(),
        inputs[2].shape.clone(),
        inputs
            .get(3)
            .map_or_else(Vec::new, |input| input.shape.clone()),
    ];
    let mut kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes, 1)?;
    kernel.set_constant_inputs(&[false, true, true, false]);
    let runtime = ep.runtime();
    let mut buffers = Vec::new();
    for input in inputs {
        let buffer = ep.allocate(input.bytes.len(), 256)?;
        unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
        buffers.push(buffer);
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let views = [
        TensorView::new(
            DevicePtr(buffers[0].as_ptr()),
            inputs[0].dtype,
            &inputs[0].shape,
            &strides[0],
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(buffers[1].as_ptr()),
            inputs[1].dtype,
            &inputs[1].shape,
            &strides[1],
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(buffers[2].as_ptr()),
            inputs[2].dtype,
            &inputs[2].shape,
            &strides[2],
            ep.device_id(),
        ),
        inputs.get(3).map_or_else(
            || TensorView::absent(DataType::Float32),
            |bias| {
                TensorView::new(
                    DevicePtr(buffers[3].as_ptr()),
                    bias.dtype,
                    &bias.shape,
                    &strides[3],
                    ep.device_id(),
                )
            },
        ),
    ];
    let output_len = output_shape.iter().product::<usize>();
    let mut output_buffer = ep.allocate(output_len * 4, 256)?;
    let output_strides = compute_contiguous_strides(output_shape);
    let mut execute = || {
        kernel.execute(
            &views,
            &mut [TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float32,
                output_shape,
                &output_strides,
                ep.device_id(),
            )],
        )
    };
    if let Err(error) = execute() {
        drop(execute);
        for buffer in buffers {
            ep.deallocate(buffer)?;
        }
        ep.deallocate(output_buffer)?;
        return Err(error);
    }
    if graph_replays != 0 {
        runtime.begin_graph_capture(&[kernel.as_ref()])?;
        execute()?;
        runtime.end_graph_capture()?;
        for _ in 0..graph_replays {
            runtime.replay_graph()?;
        }
    }
    let mut output = vec![0u8; output_len * 4];
    unsafe { runtime.dtoh(&mut output, cuptr(output_buffer.as_ptr()))? };
    if graph_replays != 0 {
        assert!(runtime.has_graph_executable()?);
        assert_eq!(runtime.graph_segment_count()?, 1);
        assert!(runtime.reset_graph()?);
    }
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
        "q2_k" => (256, 84),
        "q3_k" => (256, 110),
        "q5_k" => (256, 176),
        "q6_k" => (256, 210),
        "q8_0" => (32, 34),
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
        } else if format == "q2_k" {
            for byte in &mut block[..80] {
                *byte = random_u32(&mut state) as u8;
            }
            block[80..82].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            block[82..84].copy_from_slice(&half::f16::from_f32(0.005).to_le_bytes());
        } else if format == "q3_k" {
            for byte in &mut block[..108] {
                *byte = random_u32(&mut state) as u8;
            }
            block[108..110].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        } else if format == "q5_k" {
            block[..2].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
            for byte in &mut block[4..] {
                *byte = random_u32(&mut state) as u8;
            }
        } else if format == "q6_k" {
            for byte in &mut block[..208] {
                *byte = random_u32(&mut state) as u8;
            }
            block[208..210].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
        } else if format == "q8_0" {
            block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            for byte in &mut block[2..] {
                *byte = random_u32(&mut state) as u8;
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_gemv_random_supported_formats_match_cpu() {
    let ep = require_cuda();
    let (k, n) = (1003usize, 37usize);
    for format in [
        "mxfp4", "iq4_nl", "iq4_xs", "iq2_xxs", "iq3_xxs", "iq2_xs", "iq2_s", "iq3_s", "iq1_s",
        "iq1_m", "q2_k", "q3_k", "q5_k", "q6_k", "q8_0",
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_gemm_prefill_matches_cpu_for_partial_and_grid_stride_tiles() {
    let ep = require_cuda();
    for (format, a_shape, k, n, with_bias) in [
        ("mxfp4", vec![3, 99], 99usize, 13usize, true),
        ("iq4_xs", vec![1, 7, 515], 515usize, 11usize, false),
        ("q2_k", vec![3, 512], 512usize, 7usize, false),
        ("q8_0", vec![5, 96], 96usize, 9usize, true),
        ("mxfp4", vec![32_769, 32], 32usize, 2usize, false),
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_gemv_dequant_is_bit_exact_against_cpu() {
    let ep = require_cuda();
    let n = 2usize;
    for format in [
        "mxfp4", "iq4_nl", "iq4_xs", "iq2_xxs", "iq3_xxs", "iq2_xs", "iq2_s", "iq3_s", "iq1_s",
        "iq1_m", "q2_k", "q3_k", "q5_k", "q6_k", "q8_0",
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_known_blocks_match_cpu_semantics_on_gpu() {
    let ep = require_cuda();
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_iq1_known_blocks_match_cpu_semantics_on_gpu() {
    let ep = require_cuda();
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn planar_fp8_and_fp4_production_matmul_match_cpu_with_tails_and_graph_replay() {
    let ep = require_cuda();
    let cases = [
        (
            "block_fp8",
            vec![2usize, 5],
            HostTensor::raw(
                DataType::Float8E4M3FN,
                &[3, 5],
                &[
                    0x38, 0x40, 0x3c, 0x00, 0xb8, 0x40, 0x38, 0xbc, 0x30, 0x00, 0x34, 0xb4, 0x38,
                    0x40, 0xc0,
                ],
            ),
            HostTensor::raw(DataType::Float8E8M0, &[2, 2], &[127, 128, 126, 127]),
            vec![3usize],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, -0.5, 1.5, 0.25, -3.0],
            vec![0.25, -0.5, 1.0],
        ),
        (
            "fp4_planar",
            vec![1usize, 32],
            HostTensor::raw(
                DataType::Int8,
                &[3, 16],
                &[
                    0x21, 0x43, 0x65, 0x07, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x70, 0x98,
                    0xba, 0xdc, 0xfe, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x00, 0x99, 0xaa,
                    0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x88, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc,
                    0xfe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                ],
            ),
            HostTensor::raw(DataType::Float8E8M0, &[3, 1], &[127, 126, 128]),
            vec![3usize],
            (0..32).map(|index| index as f32 / 16.0 - 1.0).collect(),
            vec![0.125, -0.25, 0.5],
        ),
    ];
    for (format, a_shape, packed, scale, output_shape, activations, bias) in cases {
        let k = *a_shape.last().unwrap();
        let n = packed.shape[0];
        let output_shape = [&a_shape[..a_shape.len() - 1], output_shape.as_slice()].concat();
        let inputs = [
            HostTensor::f32(&a_shape, &activations),
            packed.clone(),
            scale.clone(),
            HostTensor::f32(&[n], &bias),
        ];
        let (graph, node) =
            planar_model_node(format, &a_shape, &packed, &scale, &output_shape, k, n, true);
        let expected = run_planar_cpu(&graph, node, &inputs, &output_shape).unwrap();
        let actual = run_planar_gpu(&ep, &graph, node, &inputs, &output_shape, 3).unwrap();
        assert_close(&actual, &expected);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn planar_production_matmul_rejects_reserved_and_overflow_values() {
    let ep = require_cuda();
    for (label, weight, scale, expected) in [
        ("reserved scale", vec![0x38u8; 32], 0xff, "reserved E8M0"),
        (
            "reserved weight",
            {
                let mut values = vec![0x38u8; 32];
                values[7] = 0x7f;
                values
            },
            127,
            "reserved E4M3",
        ),
        ("overflow", vec![0x7eu8; 32], 0xfe, "overflow"),
    ] {
        let packed = HostTensor::raw(DataType::Float8E4M3FN, &[1, 32], &weight);
        let aux_scale = HostTensor::raw(DataType::Float8E8M0, &[1, 1], &[scale]);
        let inputs = [
            HostTensor::f32(&[1, 32], &[1.0; 32]),
            packed.clone(),
            aux_scale.clone(),
        ];
        let (graph, node) = planar_model_node(
            "block_fp8",
            &[1, 32],
            &packed,
            &aux_scale,
            &[1, 1],
            32,
            1,
            false,
        );
        let error = run_planar_gpu(&ep, &graph, node, &inputs, &[1, 1], 0)
            .expect_err(label)
            .to_string();
        assert!(
            error.contains(expected),
            "{label}: expected '{expected}' in '{error}'"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn supported_formats_and_prefill_route_to_cuda() {
    let ep = require_cuda();
    for format in [
        "mxfp4", "iq4_nl", "iq4_xs", "iq2_xxs", "iq3_xxs", "iq2_xs", "iq2_s", "iq3_s", "iq1_s",
        "iq1_m", "q2_k", "q3_k", "q5_k", "q6_k", "q8_0",
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
                1,
                &[
                    static_shape([1, qk]),
                    static_shape([1, 1, block_bytes]),
                    vec![],
                    vec![],
                ],
                &[
                    DataType::Float32,
                    DataType::Uint8,
                    DataType::Undefined,
                    DataType::Undefined,
                ],
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
            1,
            &[
                static_shape([1, 32]),
                static_shape([1, 1, 18]),
                vec![],
                vec![],
            ],
            &[
                DataType::Float32,
                DataType::Uint8,
                DataType::Undefined,
                DataType::Undefined,
            ],
            &[]
        ),
        KernelMatch::Unsupported { .. }
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
                1,
                &[
                    static_shape([2, qk]),
                    static_shape([1, 1, block_bytes]),
                    vec![],
                    vec![],
                ],
                &[
                    DataType::Float32,
                    DataType::Uint8,
                    DataType::Undefined,
                    DataType::Undefined,
                ],
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
            1,
            &[
                vec![Dim::Symbolic(SymbolId(0)), Dim::Static(32)],
                static_shape([1, 1, 17]),
                vec![],
                vec![],
            ],
            &[
                DataType::Float32,
                DataType::Uint8,
                DataType::Undefined,
                DataType::Undefined,
            ],
            &[]
        ),
        KernelMatch::Supported { .. }
    ));
}

#[test]
#[ignore = "opt-in real-checkpoint test; set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT"]
fn glm52_ud_iq1s_real_blocks_match_cpu_on_cuda() {
    let root = std::env::var("ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT")
        .expect("set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT to the official checkpoint directory");
    let ep = require_cuda();
    let cases = [
        (
            "iq1_s",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            3_374_539_520u64,
        ),
        (
            "iq2_xxs",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            17_617_400_576,
        ),
        (
            "iq3_xxs",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            2_131_092_224,
        ),
        (
            "iq4_xs",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            15_892_755_200,
        ),
        (
            "q2_k",
            "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            16_942_690_656,
        ),
        (
            "q3_k",
            "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            15_548_248_416,
        ),
        (
            "q5_k",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            1_265_416_960,
        ),
        (
            "q6_k",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            1_203_485_440,
        ),
        (
            "q8_0",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            17_604_031_232,
        ),
    ];
    for (format, shard, offset) in cases {
        let (qk, block_bytes) = format_info(format);
        let mut packed = vec![0u8; block_bytes];
        let mut file = std::fs::File::open(std::path::Path::new(&root).join(shard)).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.read_exact(&mut packed).unwrap();
        let activation: Vec<f32> = (0..qk)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 16.0)
            .collect();
        let packed_shape = [1, 1, block_bytes];
        let inputs = [
            HostTensor::f32(&[1, qk], &activation),
            HostTensor::u8(&packed_shape, &packed),
        ];
        let (graph, node) = model_node(format, &[1, qk], &packed_shape, &[1, 1], qk, 1, false);
        let expected = run_cpu(&graph, node, &inputs, &[1, 1]);
        let actual = run_gpu(&ep, &graph, node, &inputs, &[1, 1]).unwrap();
        assert_close(&actual, &expected);
    }
}
