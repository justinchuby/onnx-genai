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
    node_outputs.push(graph.create_named_value(
        "output",
        DataType::Float32,
        static_shape(y_shape.iter().copied()),
    ));
    output_specs.push(OutputSpec {
        dtype: DataType::Float32,
        shape: y_shape,
    });
    if outputs == 3 {
        for name in ["present_key", "present_value"] {
            let shape = vec![batch, case.kv_heads, case.total_seq, case.head_size];
            node_outputs.push(graph.create_named_value(
                name,
                DataType::Float32,
                static_shape(shape.iter().copied()),
            ));
            output_specs.push(OutputSpec {
                dtype: DataType::Float32,
                shape,
            });
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

/// Concrete per-slot shapes (empty for omitted optionals) for `get_kernel`.
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

/// Assert `Y` matches within [`Y_TOLERANCE`]; return the max |Δ| observed.
fn assert_close(gpu: &[u8], cpu: &[u8], what: &str) -> f32 {
    let g = as_f32(gpu);
    let c = as_f32(cpu);
    assert_eq!(g.len(), c.len(), "{what}: length mismatch");
    let mut max_delta = 0.0f32;
    for (index, (gv, cv)) in g.iter().zip(&c).enumerate() {
        if gv.to_bits() == cv.to_bits() {
            continue;
        }
        let delta = (gv - cv).abs();
        max_delta = max_delta.max(delta);
        assert!(
            delta <= Y_TOLERANCE,
            "{what}: mismatch at {index}: gpu={gv} cpu={cv} (|Δ|={delta} > {Y_TOLERANCE})"
        );
    }
    max_delta
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
    let k = HostTensor::f32(&[1, kv_heads, 6, head_size], &make(kv_heads * 6 * head_size));
    let v = HostTensor::f32(&[1, kv_heads, 6, head_size], &make(kv_heads * 6 * head_size));
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
