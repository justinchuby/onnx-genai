//! GPU parity tests for `pkg.nxrt::IndexShare` v1.
//!
//! Each case builds a tiny single-node model with the `onnx-runtime-ir` graph
//! API, runs it through both the CPU reference kernel (the authoritative
//! numerical oracle) and the device-resident CUDA kernel, and compares every
//! output. The shapes and value generators mirror the CPU kernel's own unit
//! tests, plus a `prefill → decode → decode` sequence that threads each step's
//! `present_*` outputs back in as the next step's `past_*` inputs.
//!
//! ## Parity contract
//!
//! * `present_key` / `present_value` are a pure `past ⧺ current` copy, so they
//!   are asserted **bit-identical**.
//! * The attention output `Y` runs a softmax whose `expf` (and the fused
//!   multiply-adds in the score / value reductions) are evaluated by the device
//!   math library, which is not bit-identical to host `libm`. `Y` is therefore
//!   asserted within a tight absolute tolerance (a few ULP); identical bit
//!   patterns — including ±inf — always pass, and the observed max |Δ| is
//!   printed for the record.
//!
//! Tests skip cleanly when no CUDA device is present.

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
/// Absolute tolerance for the softmax attention output `Y`. Device `expf` and
/// fused multiply-adds differ from host `libm` by ~1 ULP at these magnitudes.
const Y_TOLERANCE: f32 = 1e-4;

#[derive(Clone)]
struct HostTensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl HostTensor {
    fn f32(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self {
            dtype: DataType::Float32,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        }
    }

    fn f16(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self {
            dtype: DataType::Float16,
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|&value| f16::from_f32(value).to_bits().to_ne_bytes())
                .collect(),
        }
    }

    fn bf16(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self {
            dtype: DataType::BFloat16,
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|&value| bf16::from_f32(value).to_bits().to_ne_bytes())
                .collect(),
        }
    }

    fn i64(shape: &[usize], values: &[i64]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self {
            dtype: DataType::Int64,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        }
    }

    fn i32(shape: &[usize], values: &[i32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self {
            dtype: DataType::Int32,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        }
    }
}

/// Output tensor spec (dtype + shape) for one node output slot.
#[derive(Clone)]
struct OutputSpec {
    dtype: DataType,
    shape: Vec<usize>,
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

/// The IndexShare geometry, mirroring the CPU kernel's own test `Case`.
#[derive(Clone, Copy)]
struct Case {
    q_heads: usize,
    kv_heads: usize,
    q_seq: usize,
    head_size: usize,
    total_seq: usize,
    scale: f32,
}

/// Build a single `IndexShare` node model from a full positional input list
/// (`[query, key, value, past_key, past_value, selected_indices,
/// attention_bias]`; `None` = an omitted optional). `outputs` is 1 or 3.
fn build_node(
    inputs: &[Option<HostTensor>],
    case: Case,
    outputs: usize,
) -> (Graph, NodeId, Vec<OutputSpec>) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(DOMAIN.into(), 1);
    let node_inputs: Vec<Option<_>> = inputs
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.as_ref().map(|tensor| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    tensor.dtype,
                    static_shape(tensor.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect();

    let batch = inputs[0].as_ref().expect("query is required").shape[0];
    let y_shape = vec![batch, case.q_heads, case.q_seq, case.head_size];
    let mut node_outputs = Vec::new();
    let mut output_specs = Vec::new();
    let dtype = inputs[0].as_ref().expect("query is required").dtype;
    node_outputs.push(graph.create_named_value(
        "output",
        dtype,
        static_shape(y_shape.iter().copied()),
    ));
    output_specs.push(OutputSpec {
        dtype,
        shape: y_shape,
    });
    if outputs == 3 {
        for name in ["present_key", "present_value"] {
            let shape = vec![batch, case.kv_heads, case.total_seq, case.head_size];
            node_outputs.push(graph.create_named_value(
                name,
                dtype,
                static_shape(shape.iter().copied()),
            ));
            output_specs.push(OutputSpec { dtype, shape });
        }
    }

    let mut node = Node::new(NodeId(0), "IndexShare", node_inputs, node_outputs.clone());
    node.domain = DOMAIN.into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(case.q_heads as i64));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(case.kv_heads as i64));
    node.attributes
        .insert("scale".into(), Attribute::Float(case.scale));
    let id = graph.insert_node(node);
    for out in node_outputs {
        graph.add_output(out);
    }
    (graph, id, output_specs)
}

/// Like [`build_node`], but the 3-output present is sized to the fixed capacity
/// `present_seq` (the past cache length) rather than `total_seq`, exercising the
/// in-place capacity present path where the present aliases past.
fn build_node_capacity(
    inputs: &[Option<HostTensor>],
    case: Case,
    present_seq: usize,
) -> (Graph, NodeId, Vec<OutputSpec>) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(DOMAIN.into(), 1);
    let node_inputs: Vec<Option<_>> = inputs
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.as_ref().map(|tensor| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    tensor.dtype,
                    static_shape(tensor.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect();

    let batch = inputs[0].as_ref().expect("query is required").shape[0];
    let dtype = inputs[0].as_ref().expect("query is required").dtype;
    let y_shape = vec![batch, case.q_heads, case.q_seq, case.head_size];
    let mut node_outputs = Vec::new();
    let mut output_specs = Vec::new();
    node_outputs.push(graph.create_named_value(
        "output",
        dtype,
        static_shape(y_shape.iter().copied()),
    ));
    output_specs.push(OutputSpec {
        dtype,
        shape: y_shape,
    });
    for name in ["present_key", "present_value"] {
        let shape = vec![batch, case.kv_heads, present_seq, case.head_size];
        node_outputs.push(graph.create_named_value(
            name,
            dtype,
            static_shape(shape.iter().copied()),
        ));
        output_specs.push(OutputSpec { dtype, shape });
    }

    let mut node = Node::new(NodeId(0), "IndexShare", node_inputs, node_outputs.clone());
    node.domain = DOMAIN.into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(case.q_heads as i64));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(case.kv_heads as i64));
    node.attributes
        .insert("scale".into(), Attribute::Float(case.scale));
    let id = graph.insert_node(node);
    for out in node_outputs {
        graph.add_output(out);
    }
    (graph, id, output_specs)
}
fn concrete_shapes(inputs: &[Option<HostTensor>]) -> Vec<Vec<usize>> {
    inputs
        .iter()
        .map(|slot| slot.as_ref().map_or_else(Vec::new, |t| t.shape.clone()))
        .collect()
}

fn run_cpu(
    graph: &Graph,
    node: NodeId,
    inputs: &[Option<HostTensor>],
    output_specs: &[OutputSpec],
) -> onnx_runtime_ep_api::Result<Vec<Vec<u8>>> {
    let model = Model::new(graph);
    let kernel = CpuExecutionProvider::new().get_kernel(
        model.graph.node(node),
        &concrete_shapes(inputs),
        1,
    )?;

    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let input_views: Vec<TensorView> = inputs
        .iter()
        .zip(&strides)
        .map(|(slot, strides)| match (slot, strides) {
            (Some(tensor), Some(strides)) => TensorView::new(
                DevicePtr(tensor.bytes.as_ptr() as *const _),
                tensor.dtype,
                &tensor.shape,
                strides,
                DeviceId::cpu(),
            ),
            _ => TensorView::absent(DataType::Undefined),
        })
        .collect();

    let out_strides: Vec<_> = output_specs
        .iter()
        .map(|spec| compute_contiguous_strides(&spec.shape))
        .collect();
    let mut out_bufs: Vec<Vec<u8>> = output_specs
        .iter()
        .map(|spec| vec![0u8; spec.shape.iter().product::<usize>() * spec.dtype.byte_size()])
        .collect();
    let mut out_views: Vec<TensorMut> = out_bufs
        .iter_mut()
        .zip(output_specs.iter().zip(&out_strides))
        .map(|(buf, (spec, strides))| {
            TensorMut::new(
                DevicePtrMut(buf.as_mut_ptr().cast()),
                spec.dtype,
                &spec.shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();
    kernel.execute(&input_views, &mut out_views)?;
    drop(out_views);
    Ok(out_bufs)
}

fn run_gpu(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[Option<HostTensor>],
    output_specs: &[OutputSpec],
) -> onnx_runtime_ep_api::Result<Vec<Vec<u8>>> {
    let model = Model::new(graph);
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes(inputs), 1)?;
    let runtime = ep.runtime();

    // Allocate + upload each present input; omitted optionals get no buffer.
    let mut buffers: Vec<Option<DeviceBuffer>> = Vec::new();
    for slot in inputs {
        match slot {
            Some(tensor) => {
                let buffer = ep.allocate(tensor.bytes.len().max(1), 256)?;
                if !tensor.bytes.is_empty() {
                    // SAFETY: allocation exactly covers the source tensor bytes.
                    unsafe { runtime.htod(&tensor.bytes, cuptr(buffer.as_ptr()))? };
                }
                buffers.push(Some(buffer));
            }
            None => buffers.push(None),
        }
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let input_views: Vec<TensorView> = inputs
        .iter()
        .zip(buffers.iter().zip(&strides))
        .map(|(slot, (buffer, strides))| match (slot, buffer, strides) {
            (Some(tensor), Some(buffer), Some(strides)) => TensorView::new(
                DevicePtr(buffer.as_ptr() as *const _),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            ),
            _ => TensorView::absent(DataType::Undefined),
        })
        .collect();

    let out_strides: Vec<_> = output_specs
        .iter()
        .map(|spec| compute_contiguous_strides(&spec.shape))
        .collect();
    let out_lens: Vec<usize> = output_specs
        .iter()
        .map(|spec| spec.shape.iter().product::<usize>() * spec.dtype.byte_size())
        .collect();
    let mut out_buffers: Vec<DeviceBuffer> = out_lens
        .iter()
        .map(|len| ep.allocate((*len).max(1), 256))
        .collect::<onnx_runtime_ep_api::Result<_>>()?;
    let mut out_views: Vec<TensorMut> = out_buffers
        .iter_mut()
        .zip(output_specs.iter().zip(&out_strides))
        .map(|(buffer, (spec, strides))| {
            TensorMut::new(
                DevicePtrMut(buffer.as_mut_ptr()),
                spec.dtype,
                &spec.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect();
    let result = kernel.execute(&input_views, &mut out_views);
    drop(out_views);
    drop(input_views);

    let mut outputs = Vec::new();
    if result.is_ok() {
        for (buffer, len) in out_buffers.iter().zip(&out_lens) {
            let mut host = vec![0u8; *len];
            if *len > 0 {
                // SAFETY: destination exactly covers the output allocation.
                unsafe { runtime.dtoh(&mut host, cuptr(buffer.as_ptr()))? };
            }
            outputs.push(host);
        }
    }
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    for buffer in out_buffers.drain(..) {
        ep.deallocate(buffer)?;
    }
    result.map(|()| outputs)
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

fn decode_floating(dtype: DataType, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        DataType::Float32 => as_f32(bytes),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        _ => panic!("unsupported floating test dtype {dtype:?}"),
    }
}

/// Convert storage values to the exact f32 values visible to the CUDA kernel,
/// so the frozen f32 CPU oracle is a valid reference for low-precision inputs.
fn exact_rounded_f32(tensor: &HostTensor) -> HostTensor {
    HostTensor::f32(&tensor.shape, &decode_floating(tensor.dtype, &tensor.bytes))
}

fn assert_close_values(gpu: &[f32], cpu: &[f32], tolerance: f32, what: &str) -> f32 {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length mismatch");
    let mut max_delta = 0.0f32;
    for (index, (&gpu, &cpu)) in gpu.iter().zip(cpu).enumerate() {
        if gpu.to_bits() == cpu.to_bits() {
            continue;
        }
        let delta = (gpu - cpu).abs();
        max_delta = max_delta.max(delta);
        assert!(
            delta <= tolerance,
            "{what}: mismatch at {index}: gpu={gpu} cpu={cpu} (|Δ|={delta} > {tolerance})"
        );
    }
    max_delta
}

/// Assert `Y` matches within [`Y_TOLERANCE`]; return the max |Δ| observed.
fn assert_close(gpu: &[u8], cpu: &[u8], what: &str) -> f32 {
    assert_close_values(&as_f32(gpu), &as_f32(cpu), Y_TOLERANCE, what)
}

/// Assert a pure-copy output is bit-identical.
fn assert_bit_exact(gpu: &[u8], cpu: &[u8], what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length mismatch");
    assert_eq!(gpu, cpu, "{what}: bytes must be bit-identical");
}

/// Run one node on both backends. Asserts `Y` within tolerance and the present
/// cache bit-exact; returns `(cpu_outputs, max |Δ| on Y)`.
fn assert_parity(
    ep: &CudaExecutionProvider,
    inputs: &[Option<HostTensor>],
    case: Case,
    outputs: usize,
) -> (Vec<Vec<u8>>, f32) {
    let (graph, node, specs) = build_node(inputs, case, outputs);
    let cpu = run_cpu(&graph, node, inputs, &specs).expect("CPU IndexShare kernel");
    let gpu = run_gpu(ep, &graph, node, inputs, &specs).expect("CUDA IndexShare kernel");
    let max_delta = assert_close(&gpu[0], &cpu[0], "output");
    if outputs == 3 {
        assert_bit_exact(&gpu[1], &cpu[1], "present_key");
        assert_bit_exact(&gpu[2], &cpu[2], "present_value");
    }
    (cpu, max_delta)
}

fn sequence(count: usize, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| offset + index as f32 * 0.0625)
        .collect()
}

#[test]
fn selected_subset_and_trailing_padding_match_cpu() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.5,
    };
    let q = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.25));
    let k = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, 0.125));
    let v = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, -1.0));
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, 3, -1, -1]);
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(indices)];
    let (_, delta) = assert_parity(&ep, &inputs, case, 3);
    eprintln!("selected_subset: Y max|Δ|={delta}");
}

/// Runs low-precision storage through CUDA, comparing its decoded result to the
/// f32 CPU oracle fed the *exact* values obtained by rounding every input to
/// the requested storage type first.
fn low_precision_matches_exact_rounded_cpu(dtype: DataType, tolerance: f32) {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.5,
    };
    let make = |shape, values: Vec<f32>| match dtype {
        DataType::Float16 => HostTensor::f16(shape, &values),
        DataType::BFloat16 => HostTensor::bf16(shape, &values),
        _ => unreachable!("low precision test only covers f16/bf16"),
    };
    let q = make(&[1, 2, 1, 3], sequence(6, -0.25));
    let k = make(&[1, 2, 5, 3], sequence(30, 0.125));
    let v = make(&[1, 2, 5, 3], sequence(30, -1.0));
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, 3, -1, -1]);
    let gpu_inputs = [
        Some(q.clone()),
        Some(k.clone()),
        Some(v.clone()),
        None,
        None,
        Some(indices.clone()),
    ];
    let cpu_inputs = [
        Some(exact_rounded_f32(&q)),
        Some(exact_rounded_f32(&k)),
        Some(exact_rounded_f32(&v)),
        None,
        None,
        Some(indices),
    ];
    let (cpu_graph, cpu_node, cpu_specs) = build_node(&cpu_inputs, case, 3);
    let cpu = run_cpu(&cpu_graph, cpu_node, &cpu_inputs, &cpu_specs).expect("CPU f32 oracle");
    let (gpu_graph, gpu_node, gpu_specs) = build_node(&gpu_inputs, case, 3);
    let gpu = run_gpu(&ep, &gpu_graph, gpu_node, &gpu_inputs, &gpu_specs)
        .expect("CUDA low-precision IndexShare");

    let delta = assert_close_values(
        &decode_floating(dtype, &gpu[0]),
        &as_f32(&cpu[0]),
        tolerance,
        "low-precision output",
    );
    assert_bit_exact(&gpu[1], &k.bytes, "present_key low-precision copy");
    assert_bit_exact(&gpu[2], &v.bytes, "present_value low-precision copy");
    eprintln!("{dtype:?} exact-rounded CPU parity: Y max|Δ|={delta}");
}

#[test]
fn f16_storage_matches_exact_rounded_cpu_oracle() {
    low_precision_matches_exact_rounded_cpu(DataType::Float16, 1e-3);
}

#[test]
fn bf16_storage_matches_exact_rounded_cpu_oracle() {
    low_precision_matches_exact_rounded_cpu(DataType::BFloat16, 1e-2);
}

#[test]
fn gqa_shared_indices_match_cpu() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 4,
        kv_heads: 2,
        q_seq: 2,
        head_size: 2,
        total_seq: 5,
        scale: 0.25,
    };
    let q = HostTensor::f32(&[1, 4, 2, 2], &sequence(16, -0.5));
    let past_k = HostTensor::f32(&[1, 2, 1, 2], &sequence(4, 0.25));
    let past_v = HostTensor::f32(&[1, 2, 1, 2], &sequence(4, -0.75));
    let k = HostTensor::f32(&[1, 2, 4, 2], &sequence(16, 0.5));
    let v = HostTensor::f32(&[1, 2, 4, 2], &sequence(16, 1.0));
    let indices = HostTensor::i64(&[1, 1, 2, 3], &[0, 2, 4, 1, 3, 4]);
    let inputs = [
        Some(q),
        Some(k),
        Some(v),
        Some(past_k),
        Some(past_v),
        Some(indices),
    ];
    let (_, delta) = assert_parity(&ep, &inputs, case, 3);
    eprintln!("gqa_shared_indices: Y max|Δ|={delta}");
}

#[test]
fn causal_and_padding_bias_composition_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 1,
        kv_heads: 1,
        q_seq: 2,
        head_size: 2,
        total_seq: 5,
        scale: 0.5,
    };
    let past_seq = 2usize;
    let q = HostTensor::f32(&[1, 1, 2, 2], &sequence(4, 0.25));
    let past_k = HostTensor::f32(&[1, 1, 2, 2], &sequence(4, -0.5));
    let past_v = HostTensor::f32(&[1, 1, 2, 2], &sequence(4, 0.75));
    let k = HostTensor::f32(&[1, 1, 3, 2], &sequence(6, 0.125));
    let v = HostTensor::f32(&[1, 1, 3, 2], &sequence(6, -1.25));
    let indices = HostTensor::i64(&[1, 1, 2, 4], &[0, 1, 2, 4, 0, 1, 3, 4]);
    let mut bias_values = vec![0.0f32; 10];
    for qi in 0..case.q_seq {
        for key in 0..5 {
            if key > past_seq + qi || key == 1 {
                bias_values[qi * 5 + key] = f32::NEG_INFINITY;
            }
        }
    }
    let bias = HostTensor::f32(&[1, 1, 2, 5], &bias_values);
    let inputs = [
        Some(q),
        Some(k),
        Some(v),
        Some(past_k),
        Some(past_v),
        Some(indices),
        Some(bias),
    ];
    let (_, delta) = assert_parity(&ep, &inputs, case, 3);
    eprintln!("causal_padding_bias: Y max|Δ|={delta}");
}

#[test]
fn int32_indices_and_broadcast_bias_match_cpu() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 1,
        q_seq: 1,
        head_size: 4,
        total_seq: 4,
        scale: 0.35,
    };
    let q = HostTensor::f32(&[2, 2, 1, 4], &sequence(16, -0.4));
    let k = HostTensor::f32(&[2, 1, 4, 4], &sequence(32, 0.1));
    let v = HostTensor::f32(&[2, 1, 4, 4], &sequence(32, -0.6));
    // int32 indices, one row broadcast across heads (index_heads=1).
    let indices = HostTensor::i32(&[2, 1, 1, 3], &[0, 2, 3, 1, 2, -1]);
    // Bias broadcast over heads: shape [B, 1, S_q, total].
    let bias = HostTensor::f32(&[2, 1, 1, 4], &sequence(8, -0.2));
    let inputs = [
        Some(q),
        Some(k),
        Some(v),
        None,
        None,
        Some(indices),
        Some(bias),
    ];
    let (_, delta) = assert_parity(&ep, &inputs, case, 1);
    eprintln!("int32_broadcast_bias: Y max|Δ|={delta}");
}

/// Rank-0 (scalar) `attention_bias`: the CPU oracle accepts it (broadcasts the
/// single value everywhere), so the CUDA kernel must too. Guards against the
/// claim-then-fail divergence where the gate (delegating to the CPU oracle)
/// claims a scalar-bias node but the kernel rejects rank 0 at execution.
#[test]
fn scalar_bias_broadcasts_and_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.4,
    };
    let q = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.25));
    let k = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, 0.125));
    let v = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, -1.0));
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, 3, -1, -1]);
    // Rank-0 scalar bias: broadcasts across every [B, N, S_q, total] position.
    let bias = HostTensor::f32(&[], &[0.5]);
    let inputs = [
        Some(q),
        Some(k),
        Some(v),
        None,
        None,
        Some(indices),
        Some(bias),
    ];
    let (_, delta) = assert_parity(&ep, &inputs, case, 1);
    eprintln!("scalar_bias: Y max|Δ|={delta}");
}

/// prefill → decode → decode, threading present_key/present_value into the next
/// step's past_key/past_value. Every step is checked against the CPU oracle.
#[test]
fn prefill_then_two_decodes_thread_present_to_past() {
    let Some(ep) = gpu() else { return };
    let head_size = 4;
    let q_heads = 4;
    let kv_heads = 2;

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 2.0
    };
    let mut make = |count: usize| -> Vec<f32> { (0..count).map(|_| next()).collect() };

    // Step 1: prefill. current_seq=6, no past cache.
    let prefill_case = Case {
        q_heads,
        kv_heads,
        q_seq: 6,
        head_size,
        total_seq: 6,
        scale: 0.4,
    };
    let q = HostTensor::f32(&[1, q_heads, 6, head_size], &make(q_heads * 6 * head_size));
    let k = HostTensor::f32(
        &[1, kv_heads, 6, head_size],
        &make(kv_heads * 6 * head_size),
    );
    let v = HostTensor::f32(
        &[1, kv_heads, 6, head_size],
        &make(kv_heads * 6 * head_size),
    );
    let mut prefill_indices = Vec::new();
    for qi in 0..6usize {
        let mut row: Vec<i64> = (0..=qi as i64).take(4).collect();
        while row.len() < 4 {
            row.push(-1);
        }
        prefill_indices.extend_from_slice(&row);
    }
    let indices = HostTensor::i64(&[1, 1, 6, 4], &prefill_indices);
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(indices)];
    let (after_prefill, delta) = assert_parity(&ep, &inputs, prefill_case, 3);
    eprintln!("prefill: Y max|Δ|={delta}");
    let mut past_k = HostTensor::f32(&[1, kv_heads, 6, head_size], &as_f32(&after_prefill[1]));
    let mut past_v = HostTensor::f32(&[1, kv_heads, 6, head_size], &as_f32(&after_prefill[2]));

    // Steps 2 and 3: single-token decode, cache grows each step.
    let mut past_seq = 6usize;
    for step in 0..2usize {
        let total = past_seq + 1;
        let decode_case = Case {
            q_heads,
            kv_heads,
            q_seq: 1,
            head_size,
            total_seq: total,
            scale: 0.4,
        };
        let q = HostTensor::f32(&[1, q_heads, 1, head_size], &make(q_heads * head_size));
        let k = HostTensor::f32(&[1, kv_heads, 1, head_size], &make(kv_heads * head_size));
        let v = HostTensor::f32(&[1, kv_heads, 1, head_size], &make(kv_heads * head_size));
        // Select up to 4 keys spread over the cache, strictly increasing.
        let stride = (total / 4).max(1);
        let mut row: Vec<i64> = Vec::new();
        let mut pos = 0i64;
        while row.len() < 4 && (pos as usize) < total {
            row.push(pos);
            pos += stride as i64;
        }
        while row.len() < 4 {
            row.push(-1);
        }
        let indices = HostTensor::i64(&[1, 1, 1, 4], &row);
        let inputs = [
            Some(q),
            Some(k),
            Some(v),
            Some(past_k.clone()),
            Some(past_v.clone()),
            Some(indices),
        ];
        let (present, delta) = assert_parity(&ep, &inputs, decode_case, 3);
        eprintln!("decode step {step}: Y max|Δ|={delta} (cache now {total})");
        past_k = HostTensor::f32(&[1, kv_heads, total, head_size], &as_f32(&present[1]));
        past_v = HostTensor::f32(&[1, kv_heads, total, head_size], &as_f32(&present[2]));
        past_seq = total;
    }
    assert_eq!(past_seq, 8);
}

#[test]
fn single_output_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.5,
    };
    let q = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.25));
    let k = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, 0.125));
    let v = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, -1.0));
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, 3, -1, -1]);
    // outputs=1: only Y is produced (present cache not requested).
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(indices)];
    let (_, delta) = assert_parity(&ep, &inputs, case, 1);
    eprintln!("single_output: Y max|Δ|={delta}");
}

#[test]
fn captured_f16_replay_is_byte_identical_to_eager_and_preserves_latch() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.5,
    };
    let q = HostTensor::f16(&[1, 2, 1, 3], &sequence(6, -0.25));
    let k = HostTensor::f16(&[1, 2, 5, 3], &sequence(30, 0.125));
    let v = HostTensor::f16(&[1, 2, 5, 3], &sequence(30, -1.0));
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, 3, -1, -1]);
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(indices)];
    let (graph, node, specs) = build_node(&inputs, case, 3);
    let eager = run_gpu(&ep, &graph, node, &inputs, &specs).expect("eager f16 IndexShare");
    let replay = run_gpu_capture_replay(&ep, &graph, node, &inputs, &specs)
        .expect("captured f16 IndexShare");
    for (label, (eager, replay)) in ["Y", "present_key", "present_value"]
        .iter()
        .zip(eager.iter().zip(&replay))
    {
        assert_bit_exact(replay, eager, label);
    }
    assert_eq!(
        ep.runtime()
            .check_capture_error()
            .expect("read capture-error latch"),
        0,
        "valid f16 capture must not poison the capture-error latch"
    );
    eprintln!("captured f16 replay: byte-identical to eager with a clear capture-error latch");
}

/// Capture one IndexShare `execute` into a CUDA graph and replay it, returning
/// the bytes produced solely by the replay. Mirrors the CSA B6 / GQA-decode
/// capture harness: a single kernel instance warms the pooled scratch (stable
/// addresses across warmup → capture → replay), the output buffers are zeroed
/// after warmup and before capture, and the captured `execute` performs no host
/// staging, no per-call alloc/free, and no stream sync while recording.
fn run_gpu_capture_replay(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[Option<HostTensor>],
    output_specs: &[OutputSpec],
) -> onnx_runtime_ep_api::Result<Vec<Vec<u8>>> {
    use cudarc::driver::sys::{
        CUgraph, CUgraphExec, CUstreamCaptureMode, cuGraphDestroy, cuGraphExecDestroy,
        cuGraphInstantiateWithFlags, cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture,
    };

    let model = Model::new(graph);
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes(inputs), 1)?;
    let runtime = ep.runtime();

    // Upload each present input; omitted optionals get no buffer.
    let mut buffers: Vec<Option<DeviceBuffer>> = Vec::new();
    for slot in inputs {
        match slot {
            Some(tensor) => {
                let buffer = ep.allocate(tensor.bytes.len().max(1), 256)?;
                if !tensor.bytes.is_empty() {
                    // SAFETY: allocation exactly covers the source tensor bytes.
                    unsafe { runtime.htod(&tensor.bytes, cuptr(buffer.as_ptr()))? };
                }
                buffers.push(Some(buffer));
            }
            None => buffers.push(None),
        }
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let input_views: Vec<TensorView> = inputs
        .iter()
        .zip(buffers.iter().zip(&strides))
        .map(|(slot, (buffer, strides))| match (slot, buffer, strides) {
            (Some(tensor), Some(buffer), Some(strides)) => TensorView::new(
                DevicePtr(buffer.as_ptr() as *const _),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            ),
            _ => TensorView::absent(DataType::Undefined),
        })
        .collect();

    let out_strides: Vec<_> = output_specs
        .iter()
        .map(|spec| compute_contiguous_strides(&spec.shape))
        .collect();
    let out_lens: Vec<usize> = output_specs
        .iter()
        .map(|spec| spec.shape.iter().product::<usize>() * spec.dtype.byte_size())
        .collect();
    let mut out_buffers: Vec<DeviceBuffer> = out_lens
        .iter()
        .map(|len| ep.allocate((*len).max(1), 256))
        .collect::<onnx_runtime_ep_api::Result<_>>()?;

    let make_out_views = |out_buffers: &mut [DeviceBuffer]| -> Vec<TensorMut> {
        out_buffers
            .iter_mut()
            .zip(output_specs.iter().zip(&out_strides))
            .map(|(buffer, (spec, strides))| {
                TensorMut::new(
                    DevicePtrMut(buffer.as_mut_ptr()),
                    spec.dtype,
                    &spec.shape,
                    strides,
                    ep.device_id(),
                )
            })
            .collect()
    };

    // A fresh generation starts un-poisoned.
    runtime.reset_capture_error()?;

    // Warmup: an eager execute compiles/caches every NVRTC kernel and sizes the
    // pooled scratch before capture. Only after this does the kernel advertise
    // capture eligibility.
    {
        let mut out_views = make_out_views(&mut out_buffers);
        kernel.execute(&input_views, &mut out_views)?;
    }
    runtime.synchronize()?;
    assert!(
        kernel.cuda_graph_compatible(),
        "IndexShare must advertise CUDA-graph capture eligibility after warmup"
    );

    // Zero the outputs so the returned bytes come only from the graph replay.
    for (buffer, len) in out_buffers.iter().zip(&out_lens) {
        if *len > 0 {
            let zeros = vec![0u8; *len];
            // SAFETY: destination exactly covers the output allocation.
            unsafe { runtime.htod(&zeros, cuptr(buffer.as_ptr()))? };
        }
    }
    runtime.synchronize()?;

    let stream = runtime.stream_ptr();
    let mut graph_handle: CUgraph = std::ptr::null_mut();
    let mut graph_exec: CUgraphExec = std::ptr::null_mut();

    let captured = (|| -> onnx_runtime_ep_api::Result<()> {
        // SAFETY: `stream` is the EP's live compute stream.
        unsafe {
            cuStreamBeginCapture_v2(
                stream,
                CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!(
                    "cuStreamBeginCapture_v2: {error:?}"
                ))
            })?;
        }
        let mut out_views = make_out_views(&mut out_buffers);
        let record = kernel.execute(&input_views, &mut out_views);
        drop(out_views);
        // Always end capture to leave the stream clean, even on error.
        // SAFETY: `stream` is capturing; `graph_handle` is a valid out-pointer.
        let end = unsafe { cuStreamEndCapture(stream, &mut graph_handle) }
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!("cuStreamEndCapture: {error:?}"))
            });
        record?;
        end?;
        // SAFETY: `graph_handle` is a freshly captured non-null graph.
        unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph_handle, 0) }
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!(
                    "cuGraphInstantiateWithFlags: {error:?}"
                ))
            })?;
        // SAFETY: `graph_exec` is instantiated; `stream` is the EP stream.
        unsafe { cuGraphLaunch(graph_exec, stream) }
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!("cuGraphLaunch: {error:?}"))
            })?;
        runtime.synchronize()
    })();

    let mut outputs = Vec::new();
    if captured.is_ok() {
        for (buffer, len) in out_buffers.iter().zip(&out_lens) {
            let mut host = vec![0u8; *len];
            if *len > 0 {
                // SAFETY: destination exactly covers the output allocation.
                unsafe { runtime.dtoh(&mut host, cuptr(buffer.as_ptr()))? };
            }
            outputs.push(host);
        }
    }

    if !graph_exec.is_null() {
        // SAFETY: `graph_exec` was instantiated above and is destroyed once.
        let _ = unsafe { cuGraphExecDestroy(graph_exec) }.result();
    }
    if !graph_handle.is_null() {
        // SAFETY: `graph_handle` was captured above and is destroyed once.
        let _ = unsafe { cuGraphDestroy(graph_handle) }.result();
    }
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    for buffer in out_buffers.drain(..) {
        ep.deallocate(buffer)?;
    }
    captured.map(|()| outputs)
}

/// Capture a prefill → decode → decode IndexShare sequence into CUDA graphs and
/// assert the replayed decode is **byte-identical** to the eager device decode
/// AND bit-parity (present cache) / tight-tolerance (`Y`) against the independent
/// CPU oracle at every step. Threads each step's present_key/present_value into
/// the next step's past_key/past_value, exactly like the eager sequence test.
#[test]
fn captured_prefill_then_two_decodes_match_eager_and_cpu() {
    let Some(ep) = gpu() else { return };
    let head_size = 4;
    let q_heads = 4;
    let kv_heads = 2;

    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 2.0
    };
    let mut make = |count: usize| -> Vec<f32> { (0..count).map(|_| next()).collect() };

    // Step 1: prefill.
    let prefill_case = Case {
        q_heads,
        kv_heads,
        q_seq: 6,
        head_size,
        total_seq: 6,
        scale: 0.4,
    };
    let q = HostTensor::f32(&[1, q_heads, 6, head_size], &make(q_heads * 6 * head_size));
    let k = HostTensor::f32(
        &[1, kv_heads, 6, head_size],
        &make(kv_heads * 6 * head_size),
    );
    let v = HostTensor::f32(
        &[1, kv_heads, 6, head_size],
        &make(kv_heads * 6 * head_size),
    );
    let mut prefill_indices = Vec::new();
    for qi in 0..6usize {
        let mut row: Vec<i64> = (0..=qi as i64).take(4).collect();
        while row.len() < 4 {
            row.push(-1);
        }
        prefill_indices.extend_from_slice(&row);
    }
    let indices = HostTensor::i64(&[1, 1, 6, 4], &prefill_indices);
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(indices)];

    let (graph, node, specs) = build_node(&inputs, prefill_case, 3);
    let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU prefill oracle");
    let eager = run_gpu(&ep, &graph, node, &inputs, &specs).expect("eager prefill");
    let replay =
        run_gpu_capture_replay(&ep, &graph, node, &inputs, &specs).expect("captured prefill");
    for (index, label) in ["Y", "present_key", "present_value"].iter().enumerate() {
        assert_eq!(
            replay[index], eager[index],
            "captured prefill must be byte-identical to eager for {label}"
        );
    }
    assert_close(&eager[0], &cpu[0], "prefill Y (eager vs cpu)");
    assert_bit_exact(&eager[1], &cpu[1], "prefill present_key (eager vs cpu)");
    assert_bit_exact(&eager[2], &cpu[2], "prefill present_value (eager vs cpu)");
    eprintln!("captured prefill: byte-identical to eager, bit-parity with CPU");

    let mut past_k = HostTensor::f32(&[1, kv_heads, 6, head_size], &as_f32(&cpu[1]));
    let mut past_v = HostTensor::f32(&[1, kv_heads, 6, head_size], &as_f32(&cpu[2]));

    // Steps 2 and 3: single-token decode, cache grows each step.
    let mut past_seq = 6usize;
    for step in 0..2usize {
        let total = past_seq + 1;
        let decode_case = Case {
            q_heads,
            kv_heads,
            q_seq: 1,
            head_size,
            total_seq: total,
            scale: 0.4,
        };
        let q = HostTensor::f32(&[1, q_heads, 1, head_size], &make(q_heads * head_size));
        let k = HostTensor::f32(&[1, kv_heads, 1, head_size], &make(kv_heads * head_size));
        let v = HostTensor::f32(&[1, kv_heads, 1, head_size], &make(kv_heads * head_size));
        let stride = (total / 4).max(1);
        let mut row: Vec<i64> = Vec::new();
        let mut pos = 0i64;
        while row.len() < 4 && (pos as usize) < total {
            row.push(pos);
            pos += stride as i64;
        }
        while row.len() < 4 {
            row.push(-1);
        }
        let indices = HostTensor::i64(&[1, 1, 1, 4], &row);
        let inputs = [
            Some(q),
            Some(k),
            Some(v),
            Some(past_k.clone()),
            Some(past_v.clone()),
            Some(indices),
        ];
        let (graph, node, specs) = build_node(&inputs, decode_case, 3);
        let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU decode oracle");
        let eager = run_gpu(&ep, &graph, node, &inputs, &specs).expect("eager decode");
        let replay =
            run_gpu_capture_replay(&ep, &graph, node, &inputs, &specs).expect("captured decode");
        for (index, label) in ["Y", "present_key", "present_value"].iter().enumerate() {
            assert_eq!(
                replay[index], eager[index],
                "captured decode step {step} must be byte-identical to eager for {label}"
            );
        }
        assert_close(&eager[0], &cpu[0], "decode Y (eager vs cpu)");
        assert_bit_exact(&eager[1], &cpu[1], "decode present_key (eager vs cpu)");
        assert_bit_exact(&eager[2], &cpu[2], "decode present_value (eager vs cpu)");
        // A well-formed captured replay must never poison the capture-error latch.
        assert_eq!(
            ep.runtime()
                .check_capture_error()
                .expect("read capture-error latch"),
            0,
            "valid captured decode must not latch a capture error"
        );
        eprintln!("captured decode step {step}: byte-identical to eager, cache now {total}");
        past_k = HostTensor::f32(&[1, kv_heads, total, head_size], &as_f32(&cpu[1]));
        past_v = HostTensor::f32(&[1, kv_heads, total, head_size], &as_f32(&cpu[2]));
        past_seq = total;
    }
    assert_eq!(past_seq, 8);
}

/// Eager execution rejects a malformed `selected_indices` row (an index that
/// follows trailing `-1` padding) exactly like the CPU oracle, and the CUDA EP
/// surfaces the same hard error rather than silently producing garbage.
#[test]
fn eager_rejects_invalid_indices() {
    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.5,
    };
    let q = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.25));
    let k = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, 0.125));
    let v = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, -1.0));
    // Second row: index 3 follows a -1 padding entry — invalid.
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, -1, 3, -1]);
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(indices)];
    let (graph, node, specs) = build_node(&inputs, case, 1);
    let result = run_gpu(&ep, &graph, node, &inputs, &specs);
    assert!(
        result.is_err(),
        "eager IndexShare must reject an index that follows trailing -1 padding"
    );
}

/// The device-side validation path latches the shared capture-error word when a
/// captured replay is fed an out-of-range index, so a poisoned decode is caught
/// by the host at the per-step sync (outside the captured region) instead of
/// issuing an out-of-bounds device load. Warmup uses a **valid** row (so capture
/// is eligible and the scratch is sized); the captured replay then observes an
/// out-of-range index written into the same stable index buffer.
#[test]
fn captured_replay_latches_capture_error_on_invalid_index() {
    use cudarc::driver::sys::{
        CUgraph, CUgraphExec, CUstreamCaptureMode, cuGraphDestroy, cuGraphExecDestroy,
        cuGraphInstantiateWithFlags, cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture,
    };

    let Some(ep) = gpu() else { return };
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: 5,
        scale: 0.5,
    };
    let q = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.25));
    let k = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, 0.125));
    let v = HostTensor::f32(&[1, 2, 5, 3], &sequence(30, -1.0));
    let valid = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 4, -1, 1, 3, -1, -1]);
    // Index 9 is out of range for a cache of length 5 (valid range [0, 5)).
    let invalid = HostTensor::i64(&[1, 2, 1, 4], &[0, 2, 9, -1, 1, 3, -1, -1]);
    let inputs = [Some(q), Some(k), Some(v), None, None, Some(valid.clone())];
    let (graph, node, specs) = build_node(&inputs, case, 1);

    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(node), &concrete_shapes(&inputs), 1)
        .expect("kernel");
    let runtime = ep.runtime();

    // Upload inputs to stable device buffers (the index buffer is reused with a
    // different payload for the captured replay).
    let mut buffers: Vec<Option<DeviceBuffer>> = Vec::new();
    for slot in &inputs {
        match slot {
            Some(tensor) => {
                let buffer = ep.allocate(tensor.bytes.len().max(1), 256).expect("alloc");
                // SAFETY: allocation exactly covers the source tensor bytes.
                unsafe {
                    runtime
                        .htod(&tensor.bytes, cuptr(buffer.as_ptr()))
                        .expect("htod")
                };
                buffers.push(Some(buffer));
            }
            None => buffers.push(None),
        }
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let input_views: Vec<TensorView> = inputs
        .iter()
        .zip(buffers.iter().zip(&strides))
        .map(|(slot, (buffer, strides))| match (slot, buffer, strides) {
            (Some(tensor), Some(buffer), Some(strides)) => TensorView::new(
                DevicePtr(buffer.as_ptr() as *const _),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            ),
            _ => TensorView::absent(DataType::Undefined),
        })
        .collect();

    let out_len = specs[0].shape.iter().product::<usize>() * specs[0].dtype.byte_size();
    let out_stride = compute_contiguous_strides(&specs[0].shape);
    let mut out_buffer = ep.allocate(out_len.max(1), 256).expect("alloc out");
    let make_out = |buffer: &mut DeviceBuffer| -> Vec<TensorMut> {
        vec![TensorMut::new(
            DevicePtrMut(buffer.as_mut_ptr()),
            specs[0].dtype,
            &specs[0].shape,
            &out_stride,
            ep.device_id(),
        )]
    };

    runtime.reset_capture_error().expect("reset latch");

    // Warmup with the valid indices: sizes scratch, primes NVRTC, makes capture
    // eligible, and (eager) does not poison the latch.
    {
        let mut out = make_out(&mut out_buffer);
        kernel
            .execute(&input_views, &mut out)
            .expect("warmup execute");
    }
    runtime.synchronize().expect("sync");
    assert!(kernel.cuda_graph_compatible(), "eligible after warmup");
    assert_eq!(
        runtime.check_capture_error().expect("latch"),
        0,
        "valid warmup must not latch"
    );

    // Overwrite the index buffer in place with the out-of-range payload.
    let index_buffer = buffers[5].as_ref().expect("index buffer");
    // SAFETY: the invalid payload has the same byte length as the valid one.
    unsafe {
        runtime
            .htod(&invalid.bytes, cuptr(index_buffer.as_ptr()))
            .expect("htod invalid");
    }
    runtime.synchronize().expect("sync");

    let stream = runtime.stream_ptr();
    let mut graph_handle: CUgraph = std::ptr::null_mut();
    let mut graph_exec: CUgraphExec = std::ptr::null_mut();
    let captured = (|| -> onnx_runtime_ep_api::Result<()> {
        // SAFETY: `stream` is the EP's live compute stream.
        unsafe {
            cuStreamBeginCapture_v2(
                stream,
                CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!("begin capture: {error:?}"))
            })?;
        }
        let mut out = make_out(&mut out_buffer);
        let record = kernel.execute(&input_views, &mut out);
        drop(out);
        // SAFETY: `stream` is capturing; `graph_handle` is a valid out-pointer.
        let end = unsafe { cuStreamEndCapture(stream, &mut graph_handle) }
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!("end capture: {error:?}"))
            });
        record?;
        end?;
        // SAFETY: `graph_handle` is a freshly captured non-null graph.
        unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph_handle, 0) }
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!("instantiate: {error:?}"))
            })?;
        // SAFETY: `graph_exec` is instantiated; `stream` is the EP stream.
        unsafe { cuGraphLaunch(graph_exec, stream) }
            .result()
            .map_err(|error| {
                onnx_runtime_ep_api::EpError::KernelFailed(format!("launch: {error:?}"))
            })?;
        runtime.synchronize()
    })();

    if !graph_exec.is_null() {
        // SAFETY: instantiated above, destroyed once.
        let _ = unsafe { cuGraphExecDestroy(graph_exec) }.result();
    }
    if !graph_handle.is_null() {
        // SAFETY: captured above, destroyed once.
        let _ = unsafe { cuGraphDestroy(graph_handle) }.result();
    }
    captured.expect("captured replay must stay memory-safe (clamped gather)");

    let latched = runtime.check_capture_error().expect("read latch");
    assert_ne!(
        latched & onnx_runtime_ep_cuda::INDEX_SHARE_CAPTURE_ERROR_INDEX,
        0,
        "an out-of-range index in a captured replay must latch the capture-error word"
    );
    runtime.reset_capture_error().expect("reset latch");

    ep.deallocate(out_buffer).expect("free out");
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer).expect("free input");
    }
    eprintln!("captured invalid index latched capture-error word 0x{latched:x}");
}

/// The in-place fixed-capacity present path: the 3-output present aliases the
/// fixed-capacity `past` bindings (present sequence == past sequence, no growing
/// `past ⧺ current`), and the valid length — hence the current-token write
/// position and the index range — is carried by the causal/padding
/// `attention_bias` frontier. The CUDA kernel derives that frontier on-device
/// (capture-safe) and must reproduce the CPU oracle: `Y` within tolerance and
/// the aliased present bit-exact.
#[test]
fn capacity_present_aliases_past_matches_cpu() {
    let Some(ep) = gpu() else { return };
    // Fixed capacity 5, one current token, logical valid length 4 (bias finite
    // for k in [0,4), -inf at k=4). write_pos = valid_len - current_seq = 3, so
    // the current token overwrites capacity row 3; row 4 is never gathered.
    let capacity = 5usize;
    let case = Case {
        q_heads: 2,
        kv_heads: 2,
        q_seq: 1,
        head_size: 3,
        total_seq: capacity, // gather stride is the capacity for this path
        scale: 0.5,
    };
    let q = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.25));
    // Current token K/V (written at capacity row 3).
    let cur_k = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, 0.5));
    let cur_v = HostTensor::f32(&[1, 2, 1, 3], &sequence(6, -0.5));
    // Fixed-capacity past caches [B, kv, capacity, H]; rows 3 and 4 hold
    // arbitrary values (row 3 is overwritten by the current token, row 4 masked).
    let past_k = HostTensor::f32(&[1, 2, capacity, 3], &sequence(2 * capacity * 3, 0.125));
    let past_v = HostTensor::f32(&[1, 2, capacity, 3], &sequence(2 * capacity * 3, -1.0));
    // Indices reference only valid positions (< 4), per index head.
    let indices = HostTensor::i64(&[1, 2, 1, 4], &[0, 1, 3, -1, 0, 2, 3, -1]);
    // Additive bias [B, N, S_q, capacity]: finite in [0,4), -inf at 4.
    let mut bias_vals = Vec::new();
    for _ in 0..(case.q_heads * case.q_seq) {
        for k in 0..capacity {
            bias_vals.push(if k < 4 { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    let bias = HostTensor::f32(&[1, case.q_heads, 1, capacity], &bias_vals);
    let inputs = [
        Some(q),
        Some(cur_k),
        Some(cur_v),
        Some(past_k),
        Some(past_v),
        Some(indices),
        Some(bias),
    ];

    let (graph, node, specs) = build_node_capacity(&inputs, case, capacity);
    let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU capacity IndexShare");
    let gpu = run_gpu(&ep, &graph, node, &inputs, &specs).expect("CUDA capacity IndexShare");
    let delta = assert_close(&gpu[0], &cpu[0], "capacity output");
    assert_bit_exact(&gpu[1], &cpu[1], "capacity present_key");
    assert_bit_exact(&gpu[2], &cpu[2], "capacity present_value");
    // The present must have the fixed-capacity shape (aliases past), not grow.
    assert_eq!(
        specs[1].shape[2], capacity,
        "present must be capacity-sized"
    );
    eprintln!("capacity_present: Y max|Δ|={delta}");
}
