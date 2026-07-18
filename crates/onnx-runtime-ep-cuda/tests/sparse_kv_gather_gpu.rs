//! GPU parity tests for `pkg.nxrt::SparseKvGather` v1.
//!
//! Each case builds a tiny single-node model with the `onnx-runtime-ir` graph
//! API, runs it through the CUDA EP, and compares the device output against the
//! CPU reference kernel (for f32) or hand-computed expected records (for the
//! f16/bf16 byte-copy paths the CPU op does not cover). Tests skip cleanly when
//! no CUDA device is present.

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
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        }
    }

    fn f16(shape: &[usize], values: &[f32]) -> Self {
        Self {
            dtype: DataType::Float16,
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
                .collect(),
        }
    }

    fn bf16(shape: &[usize], values: &[f32]) -> Self {
        Self {
            dtype: DataType::BFloat16,
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
                .collect(),
        }
    }

    fn i32(shape: &[usize], values: &[i32]) -> Self {
        Self {
            dtype: DataType::Int32,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        }
    }

    fn i64(shape: &[usize], values: &[i64]) -> Self {
        Self {
            dtype: DataType::Int64,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
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

/// Build a single `SparseKvGather` node model. `input_specs` names and dtypes
/// map 1:1 onto the runtime inputs; the output dtype matches the cache.
fn model_node(
    inputs: &[HostTensor],
    output_shape: &[usize],
    out_dtype: DataType,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(DOMAIN.into(), 1);
    let names = ["cache", "indices", "valid_lengths"];
    let mut node_inputs = Vec::new();
    for (input, name) in inputs.iter().zip(names) {
        let value =
            graph.create_named_value(name, input.dtype, static_shape(input.shape.iter().copied()));
        graph.add_input(value);
        node_inputs.push(Some(value));
    }
    let output = graph.create_named_value(
        "selected",
        out_dtype,
        static_shape(output_shape.iter().copied()),
    );
    let mut node = Node::new(NodeId(0), "SparseKvGather", node_inputs, vec![output]);
    node.domain = DOMAIN.into();
    node.attributes
        .insert("index_layout_version".into(), Attribute::Int(1));
    node.attributes.insert(
        "out_of_range".into(),
        Attribute::String("error".as_bytes().to_vec()),
    );
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
}

fn views<'a>(
    inputs: &'a [HostTensor],
    ptrs: &'a [*const u8],
    strides: &'a [Vec<i64>],
    device: DeviceId,
) -> Vec<TensorView<'a>> {
    inputs
        .iter()
        .zip(ptrs)
        .zip(strides)
        .map(|((input, ptr), strides)| {
            TensorView::new(
                DevicePtr(*ptr as *const _),
                input.dtype,
                &input.shape,
                strides,
                device,
            )
        })
        .collect()
}

fn run_cpu(
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_shape: &[usize],
    out_dtype: DataType,
) -> Vec<u8> {
    let model = Model::new(graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node), &[], 1)
        .unwrap();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let ptrs: Vec<*const u8> = inputs.iter().map(|input| input.bytes.as_ptr()).collect();
    let input_views = views(inputs, &ptrs, &strides, DeviceId::cpu());
    let output_strides = compute_contiguous_strides(output_shape);
    let output_len = output_shape.iter().product::<usize>() * out_dtype.byte_size();
    let mut output = vec![0u8; output_len];
    let output_view = TensorMut::new(
        DevicePtrMut(output.as_mut_ptr().cast()),
        out_dtype,
        output_shape,
        &output_strides,
        DeviceId::cpu(),
    );
    kernel.execute(&input_views, &mut [output_view]).unwrap();
    output
}

fn run_gpu(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_shape: &[usize],
    out_dtype: DataType,
) -> onnx_runtime_ep_api::Result<Vec<u8>> {
    let model = Model::new(graph);
    let concrete_shapes: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape.clone()).collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes, 1)?;
    let runtime = ep.runtime();
    let mut buffers = Vec::<DeviceBuffer>::new();
    for input in inputs {
        let buffer = ep.allocate(input.bytes.len().max(1), 256)?;
        if !input.bytes.is_empty() {
            // SAFETY: each allocation exactly covers its source tensor.
            unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
        }
        buffers.push(buffer);
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let ptrs: Vec<*const u8> = buffers.iter().map(|b| b.as_ptr() as *const u8).collect();
    let input_views = views(inputs, &ptrs, &strides, ep.device_id());
    let output_len = output_shape.iter().product::<usize>();
    let output_bytes = output_len * out_dtype.byte_size();
    let mut output_buffer = ep.allocate(output_bytes.max(1), 256)?;
    let output_strides = compute_contiguous_strides(output_shape);
    let output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        out_dtype,
        output_shape,
        &output_strides,
        ep.device_id(),
    );
    let result = kernel.execute(&input_views, &mut [output_view]);
    let mut output = vec![0u8; output_bytes];
    if result.is_ok() && !output.is_empty() {
        // SAFETY: the destination exactly covers the output allocation.
        unsafe { runtime.dtoh(&mut output, cuptr(output_buffer.as_ptr()))? };
    }
    drop(input_views);
    for buffer in buffers {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    result.map(|()| output)
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn basic_gather_matches_cpu_and_expected() {
    let Some(ep) = gpu() else { return };
    // cache [B=1,G=1,C=4,D=2], indices [1,1,1,4] selecting 2,0,2,3 (dup + order).
    let cache = HostTensor::f32(
        &[1, 1, 4, 2],
        &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
    );
    let indices = HostTensor::i64(&[1, 1, 1, 4], &[2, 0, 2, 3]);
    let out_shape = [1, 1, 1, 4, 2];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let gpu_out =
        as_f32(&run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap());
    let cpu_out = as_f32(&run_cpu(
        &graph,
        node,
        &inputs,
        &out_shape,
        DataType::Float32,
    ));
    assert_eq!(gpu_out, [20.0, 21.0, 0.0, 1.0, 20.0, 21.0, 30.0, 31.0]);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn int32_indices_and_multibatch_group_match_cpu() {
    let Some(ep) = gpu() else { return };
    // B=2, G=2, C=3, D=2; Q=2, K=3. Distinct values per (b,g) record.
    let mut cache_vals = Vec::new();
    for bg in 0..4 {
        for c in 0..3 {
            cache_vals.push((bg * 100 + c * 10) as f32);
            cache_vals.push((bg * 100 + c * 10 + 1) as f32);
        }
    }
    let cache = HostTensor::f32(&[2, 2, 3, 2], &cache_vals);
    let indices = HostTensor::i32(
        &[2, 2, 2, 3],
        &[
            0, 1, 2, 2, 1, 0, // b0g0
            1, 1, 0, 2, 0, 2, // b0g1
            0, 2, 1, 1, 2, 0, // b1g0
            2, 0, 1, 0, 1, 2, // b1g1
        ],
    );
    let out_shape = [2, 2, 2, 3, 2];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let gpu_out =
        as_f32(&run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap());
    let cpu_out = as_f32(&run_cpu(
        &graph,
        node,
        &inputs,
        &out_shape,
        DataType::Float32,
    ));
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn valid_lengths_limits_gather_and_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let cache = HostTensor::f32(
        &[1, 1, 4, 2],
        &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
    );
    let indices = HostTensor::i64(&[1, 1, 1, 3], &[0, 2, 1]);
    let valid = HostTensor::i64(&[1], &[3]);
    let out_shape = [1, 1, 1, 3, 2];
    let inputs = [cache, indices, valid];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let gpu_out =
        as_f32(&run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap());
    let cpu_out = as_f32(&run_cpu(
        &graph,
        node,
        &inputs,
        &out_shape,
        DataType::Float32,
    ));
    assert_eq!(gpu_out, [0.0, 1.0, 20.0, 21.0, 10.0, 11.0]);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn index_at_or_beyond_valid_length_errors() {
    let Some(ep) = gpu() else { return };
    // valid_lengths=3 makes index 3 out of range even though cache has 4 rows.
    let cache = HostTensor::f32(&[1, 1, 4, 2], &[0.0; 8]);
    let indices = HostTensor::i64(&[1, 1, 1, 3], &[0, 1, 3]);
    let valid = HostTensor::i64(&[1], &[3]);
    let out_shape = [1, 1, 1, 3, 2];
    let inputs = [cache, indices, valid];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let err = run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("out of range"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("valid length 3"),
        "unexpected error: {message}"
    );
}

#[test]
fn negative_index_errors() {
    let Some(ep) = gpu() else { return };
    let cache = HostTensor::f32(&[1, 1, 4, 2], &[0.0; 8]);
    let indices = HostTensor::i64(&[1, 1, 1, 3], &[0, -1, 2]);
    let out_shape = [1, 1, 1, 3, 2];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let err = run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap_err();
    assert!(
        err.to_string().contains("negative index"),
        "unexpected error: {err}"
    );
}

#[test]
fn empty_selection_yields_empty_output() {
    let Some(ep) = gpu() else { return };
    // K=0 -> zero selected records, a valid empty contiguous output.
    let cache = HostTensor::f32(&[1, 1, 1, 2], &[1.0, 2.0]);
    let indices = HostTensor::i64(&[1, 1, 3, 0], &[]);
    let out_shape = [1, 1, 3, 0, 2];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let gpu_out = run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap();
    assert!(gpu_out.is_empty());
}

#[test]
fn f16_gather_matches_bit_exact_copy() {
    let Some(ep) = gpu() else { return };
    let values = [0.0f32, 1.5, -2.25, 3.0, 4.5, -5.5, 6.0, 7.75];
    let cache = HostTensor::f16(&[1, 1, 4, 2], &values);
    let indices = HostTensor::i64(&[1, 1, 1, 4], &[3, 0, 3, 1]);
    let out_shape = [1, 1, 1, 4, 2];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float16);
    let out = run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float16).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
        .collect();
    // Records 3,0,3,1 of [ (0,1.5) (-2.25,3) (4.5,-5.5) (6,7.75) ].
    assert_eq!(got, [6.0, 7.75, 0.0, 1.5, 6.0, 7.75, -2.25, 3.0]);
}

#[test]
fn bf16_gather_matches_bit_exact_copy() {
    let Some(ep) = gpu() else { return };
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cache = HostTensor::bf16(&[1, 1, 3, 2], &values);
    let indices = HostTensor::i32(&[1, 1, 1, 3], &[2, 2, 0]);
    let out_shape = [1, 1, 1, 3, 2];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::BFloat16);
    let out = run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::BFloat16).unwrap();
    let got: Vec<f32> = out
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
        .collect();
    assert_eq!(got, [5.0, 6.0, 5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn deepseek_kv_layout_shape_matches_cpu() {
    let Some(ep) = gpu() else { return };
    // Realistic compressed-sparse-attention decode geometry: a small head count
    // and head dim, a 130-wide candidate row (128 window + 2 compressed), and a
    // KV cache long enough to exercise the ring/compressed offset span.
    let batch = 2usize;
    let groups = 4usize;
    let cache_len = 160usize;
    let dim = 64usize;
    let queries = 1usize;
    let selections = 130usize;

    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };

    let cache_elems = batch * groups * cache_len * dim;
    let cache_vals: Vec<f32> = (0..cache_elems)
        .map(|_| (next() as f32 / u32::MAX as f32 - 0.5) * 4.0)
        .collect();
    let cache = HostTensor::f32(&[batch, groups, cache_len, dim], &cache_vals);

    let index_elems = batch * groups * queries * selections;
    let index_vals: Vec<i64> = (0..index_elems)
        .map(|_| (next() % cache_len as u32) as i64)
        .collect();
    let indices = HostTensor::i64(&[batch, groups, queries, selections], &index_vals);

    let out_shape = [batch, groups, queries, selections, dim];
    let inputs = [cache, indices];
    let (graph, node) = model_node(&inputs, &out_shape, DataType::Float32);
    let gpu_out =
        as_f32(&run_gpu(&ep, &graph, node, &inputs, &out_shape, DataType::Float32).unwrap());
    let cpu_out = as_f32(&run_cpu(
        &graph,
        node,
        &inputs,
        &out_shape,
        DataType::Float32,
    ));
    assert_eq!(gpu_out.len(), out_shape.iter().product::<usize>());
    assert_eq!(gpu_out, cpu_out);
}
