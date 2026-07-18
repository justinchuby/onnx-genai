//! GPU parity tests for `pkg.nxrt::CompressedSparseAttention` v1 (host-staged).
//!
//! Each case builds a single-node model with the `onnx-runtime-ir` graph API and
//! runs the SAME inputs through both the CPU CSA kernel (the authoritative
//! oracle) and the CUDA CSA kernel, asserting bit-parity on every output. The
//! ratio-128 case threads a `prefill → decode → decode` sequence, feeding each
//! step's `present_*` outputs back as the next step's `past_*` inputs, so the
//! stateful compressed-cache / carry lifecycle is validated across steps.
//!
//! Input construction mirrors the CPU kernel's own
//! `ratio128_stateful_carry_matches_full_recompute_across_decode_boundary` test
//! (same value generators, D=512, RD=64, ratio=128, cache_format
//! `fp8_e4m3_block64`) so the oracle comparison is apples-to-apples. The last
//! decode step crosses a 128-position block boundary, exercising the stateful
//! compression / FP8 quantized-cache write and carry reset during decode.
//!
//! Tests skip cleanly when no CUDA device is present.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

const DOMAIN: &str = "pkg.nxrt";
const DIM: usize = 512;
const ROPE_DIM: usize = 64;
const RATIO: usize = 128;
// Hybrid FP8/BF16 record: (D-RD)/64 FP8 blocks of (64 codes + 1 E8M0 scale) byte
// each, followed by RD little-endian BF16 RoPE values (2 bytes each).
const STORED_WIDTH: usize = ((DIM - ROPE_DIM) / 64) * (64 + 1) + ROPE_DIM * 2;

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

    fn u8(shape: &[usize], values: &[u8]) -> Self {
        Self {
            dtype: DataType::Uint8,
            shape: shape.to_vec(),
            bytes: values.to_vec(),
        }
    }

    fn zeros(dtype: DataType, shape: &[usize]) -> Self {
        let bytes = vec![0u8; shape.iter().product::<usize>() * dtype.byte_size()];
        Self {
            dtype,
            shape: shape.to_vec(),
            bytes,
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

// --- ratio-128 value generators (copied verbatim from the CPU kernel test so
// the two kernels see byte-identical inputs). ------------------------------

fn compressor_value(position: usize, d: usize) -> f32 {
    0.4 + (position % RATIO) as f32 * 0.00625
        + (position / RATIO) as f32 * 0.03125
        + (d % 23) as f32 * 0.009
        + ((position * 11 + d * 3) % 7) as f32 * 0.001
}

fn compressor_score(position: usize, d: usize) -> f32 {
    ((position * 3 + d * 5) % 19) as f32 * 0.0625 - 0.5625 + ((position + d) % 3) as f32 * 0.015625
}

fn ape_value(slot: usize, d: usize) -> f32 {
    0.03125 + ((slot * 5 + d * 7) % 17) as f32 * 0.0078125 - 0.0625
}

fn query_value(position: usize, d: usize) -> f32 {
    0.01 + ((position * 17 + d * 13) % 37) as f32 * 0.00025
}

fn kv_value(position: usize, d: usize) -> f32 {
    0.2 + ((position * 7 + d * 11) % 41) as f32 * 0.0125 + (position % 5) as f32 * 0.003
}

fn rows(start: usize, count: usize, value: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(count * DIM);
    for position in start..start + count {
        for d in 0..DIM {
            values.push(value(position, d));
        }
    }
    values
}

/// Compressed-attention state threaded across steps (the `present_*` outputs).
struct CsaState {
    cache: HostTensor,
    carry: HostTensor,
}

/// The 11 frozen-v1 ratio-128 inputs, in positional order.
fn ratio128_inputs(
    sequence: usize,
    first_position: usize,
    current_kv_start: usize,
    current_kv_len: usize,
    total: usize,
    ape: &HostTensor,
    norm: &HostTensor,
    sink: &HostTensor,
    past: &CsaState,
) -> Vec<HostTensor> {
    let query = HostTensor::f32(
        &[1, sequence, 1, DIM],
        &rows(first_position, sequence, query_value),
    );
    let current_kv = HostTensor::f32(
        &[1, current_kv_len, DIM],
        &rows(current_kv_start, current_kv_len, kv_value),
    );
    let compressor_kv = HostTensor::f32(
        &[1, sequence, DIM],
        &rows(first_position, sequence, compressor_value),
    );
    let compressor_gate = HostTensor::f32(
        &[1, sequence, DIM],
        &rows(first_position, sequence, |position, d| {
            compressor_score(position, d) - ape_value(position % RATIO, d)
        }),
    );
    let seqlens = HostTensor::i32(&[1], &[(total - 1) as i32]);
    let total_len = HostTensor::i64(&[], &[total as i64]);
    vec![
        query,
        current_kv,
        compressor_kv,
        compressor_gate,
        ape.clone(),
        norm.clone(),
        past.cache.clone(),
        past.carry.clone(),
        seqlens,
        total_len,
        sink.clone(),
    ]
}

fn ratio128_node(inputs: &[HostTensor], next_records: usize) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(DOMAIN.into(), 1);
    let names = [
        "query",
        "current_kv",
        "compressor_kv",
        "compressor_gate",
        "compressor_ape",
        "compressor_norm",
        "past_compressed_kv",
        "past_compression_carry",
        "seqlens_k",
        "total_sequence_length",
        "head_sink",
    ];
    let node_inputs: Vec<_> = inputs
        .iter()
        .zip(names)
        .map(|(input, name)| {
            let value = graph.create_named_value(
                name,
                input.dtype,
                static_shape(input.shape.iter().copied()),
            );
            graph.add_input(value);
            Some(value)
        })
        .collect();

    let batch = inputs[0].shape[0];
    let sequence = inputs[0].shape[1];
    let outputs = vec![
        graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape([batch, sequence, 1, DIM]),
        ),
        graph.create_named_value(
            "present_compressed_kv",
            DataType::Uint8,
            static_shape([batch, next_records, STORED_WIDTH]),
        ),
        graph.create_named_value(
            "present_compression_carry",
            DataType::Float32,
            static_shape([batch, RATIO, 2, DIM]),
        ),
    ];
    let mut node = Node::new(
        NodeId(0),
        "CompressedSparseAttention",
        node_inputs,
        outputs.clone(),
    );
    node.domain = DOMAIN.into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(1));
    node.attributes
        .insert("head_dim".into(), Attribute::Int(DIM as i64));
    node.attributes
        .insert("qk_rope_head_dim".into(), Attribute::Int(ROPE_DIM as i64));
    node.attributes
        .insert("compression_ratio".into(), Attribute::Int(RATIO as i64));
    node.attributes.insert("causal".into(), Attribute::Int(1));
    node.attributes
        .insert("cache_layout_version".into(), Attribute::Int(1));
    node.attributes
        .insert("index_layout_version".into(), Attribute::Int(1));
    node.attributes.insert(
        "cache_format".into(),
        Attribute::String("fp8_e4m3_block64".as_bytes().to_vec()),
    );
    let node = graph.insert_node(node);
    for output in outputs {
        graph.add_output(output);
    }
    (graph, node)
}

struct OutputSpec {
    dtype: DataType,
    shape: Vec<usize>,
}

/// Run the node on the CPU EP over host-resident tensors, returning each
/// output's raw bytes.
fn run_cpu(
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_specs: &[OutputSpec],
) -> onnx_runtime_ep_api::Result<Vec<Vec<u8>>> {
    let model = Model::new(graph);
    let concrete: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape.clone()).collect();
    let kernel = CpuExecutionProvider::new().get_kernel(model.graph.node(node), &concrete, 1)?;

    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<TensorView> = inputs
        .iter()
        .zip(&strides)
        .map(|(input, strides)| {
            TensorView::new(
                DevicePtr(input.bytes.as_ptr() as *const _),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            )
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

/// Run the node on the CUDA EP: upload inputs, execute host-staged, download
/// each output's raw bytes.
fn run_gpu(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_specs: &[OutputSpec],
) -> onnx_runtime_ep_api::Result<Vec<Vec<u8>>> {
    let model = Model::new(graph);
    let concrete: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape.clone()).collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete, 1)?;
    let runtime = ep.runtime();

    let mut buffers = Vec::<DeviceBuffer>::new();
    for input in inputs {
        let buffer = ep.allocate(input.bytes.len().max(1), 256)?;
        if !input.bytes.is_empty() {
            // SAFETY: allocation exactly covers the source tensor bytes.
            unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
        }
        buffers.push(buffer);
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<TensorView> = inputs
        .iter()
        .zip(buffers.iter().zip(&strides))
        .map(|(input, (buffer, strides))| {
            TensorView::new(
                DevicePtr(buffer.as_ptr() as *const _),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
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
    for buffer in buffers {
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

fn assert_f32_close(gpu: &[u8], cpu: &[u8], tol: f32, what: &str) {
    let gpu = as_f32(gpu);
    let cpu = as_f32(cpu);
    assert_eq!(gpu.len(), cpu.len(), "{what}: length mismatch");
    for (index, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        // Host-staging delegates to the same CPU compute, so values are expected
        // bit-identical; identical bit patterns (incl. ±inf) always pass. The
        // tolerance is a guard against any incidental FP reassociation.
        if g.to_bits() == c.to_bits() {
            continue;
        }
        assert!(
            (g - c).abs() <= tol,
            "{what}: mismatch at {index}: gpu={g} cpu={c} (tol {tol})"
        );
    }
}

/// One `prefill → decode → decode` step: builds the node, runs both kernels,
/// asserts parity on all three outputs, and returns the (CPU) present state to
/// thread into the next step.
fn run_step(ep: &CudaExecutionProvider, inputs: &[HostTensor], next_records: usize) -> CsaState {
    let sequence = inputs[0].shape[1];
    let output_specs = vec![
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, sequence, 1, DIM],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, next_records, STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, RATIO, 2, DIM],
        },
    ];
    let (graph, node) = ratio128_node(inputs, next_records);
    let cpu = run_cpu(&graph, node, inputs, &output_specs).expect("CPU CSA kernel");
    let gpu = run_gpu(ep, &graph, node, inputs, &output_specs).expect("CUDA CSA kernel");

    assert_f32_close(&gpu[0], &cpu[0], 1e-4, "Y");
    assert_eq!(
        gpu[1], cpu[1],
        "present_compressed_kv bytes must match exactly"
    );
    assert_f32_close(&gpu[2], &cpu[2], 1e-4, "present_compression_carry");

    CsaState {
        cache: HostTensor::u8(&output_specs[1].shape, &cpu[1]),
        carry: HostTensor::f32(&output_specs[2].shape, &as_f32(&cpu[2])),
    }
}

#[test]
fn ratio128_prefill_then_two_decodes_matches_cpu() {
    let Some(ep) = gpu() else { return };

    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(
        &[DIM],
        &(0..DIM)
            .map(|d| 0.75 + (d % 17) as f32 * 0.03125)
            .collect::<Vec<_>>(),
    );
    let sink = HostTensor::f32(&[1], &[-0.375]);

    // Prefill from scratch: positions 0..125 (S=126, total=126). start=0 so the
    // carry is reset internally; the empty past cache has 0 records. next_records
    // = 126/128 = 0 (all attention is in the dense window + sink — the dense
    // fallback / sink path).
    let initial = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let prefill_inputs = ratio128_inputs(126, 0, 0, 126, 126, &ape, &norm, &sink, &initial);
    let after_prefill = run_step(&ep, &prefill_inputs, 0);

    // Decode position 126: total=127, still 0 compressed records.
    let decode1_inputs = ratio128_inputs(1, 126, 0, 127, 127, &ape, &norm, &sink, &after_prefill);
    let after_decode1 = run_step(&ep, &decode1_inputs, 0);

    // Decode position 127: total=128 crosses the 128-block boundary → emits the
    // first FP8-quantized compressed record and resets the carry. This exercises
    // the stateful compression + quantized-cache write during decode.
    let decode2_inputs = ratio128_inputs(1, 127, 0, 128, 128, &ape, &norm, &sink, &after_decode1);
    let after_decode2 = run_step(&ep, &decode2_inputs, 1);

    assert_eq!(after_decode2.cache.shape, vec![1, 1, STORED_WIDTH]);
}

/// `supports_op` must reject, at claim time, ratio/dtype/attribute combinations
/// the kernel does not correctly handle — rather than claiming the node and
/// failing inside `execute` (doc §4.8).
#[test]
fn supports_op_rejects_unsupported_configs() {
    let Some(ep) = gpu() else { return };

    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(&[DIM], &vec![0.8f32; DIM]);
    let sink = HostTensor::f32(&[1], &[-0.375]);
    let state = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let inputs = ratio128_inputs(126, 0, 0, 126, 126, &ape, &norm, &sink, &state);
    let input_dtypes: Vec<DataType> = inputs.iter().map(|input| input.dtype).collect();
    let shapes: Vec<_> = inputs
        .iter()
        .map(|input| static_shape(input.shape.iter().copied()))
        .collect();

    // Baseline: a valid ratio-128 config is claimed.
    let (graph, node) = ratio128_node(&inputs, 0);
    let model = Model::new(&graph);
    assert!(
        matches!(
            ep.supports_op(model.graph.node(node), 1, &shapes, &input_dtypes, &[]),
            KernelMatch::Supported { .. }
        ),
        "valid ratio-128 CSA must be claimed"
    );

    // Unsupported compression_ratio (only 4 and 128 are frozen) → rejected.
    let (mut graph_bad, node_bad) = ratio128_node(&inputs, 0);
    let node_ref = graph_bad.node_mut(node_bad);
    node_ref
        .attributes
        .insert("compression_ratio".into(), Attribute::Int(8));
    let model_bad = Model::new(&graph_bad);
    assert!(
        matches!(
            ep.supports_op(
                model_bad.graph.node(node_bad),
                1,
                &shapes,
                &input_dtypes,
                &[]
            ),
            KernelMatch::Unsupported { .. }
        ),
        "unsupported compression_ratio must be rejected at claim time"
    );

    // Unsupported cache_format string → rejected.
    let (mut graph_fmt, node_fmt) = ratio128_node(&inputs, 0);
    graph_fmt.node_mut(node_fmt).attributes.insert(
        "cache_format".into(),
        Attribute::String("int4_block16".as_bytes().to_vec()),
    );
    let model_fmt = Model::new(&graph_fmt);
    assert!(
        matches!(
            ep.supports_op(
                model_fmt.graph.node(node_fmt),
                1,
                &shapes,
                &input_dtypes,
                &[]
            ),
            KernelMatch::Unsupported { .. }
        ),
        "unsupported cache_format must be rejected at claim time"
    );

    // Unsupported query dtype (query must be f32) → rejected via dtype gating.
    let mut bad_dtypes = input_dtypes.clone();
    bad_dtypes[0] = DataType::Float16;
    assert!(
        matches!(
            ep.supports_op(model.graph.node(node), 1, &shapes, &bad_dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "non-f32 query dtype must be rejected at claim time"
    );
}
