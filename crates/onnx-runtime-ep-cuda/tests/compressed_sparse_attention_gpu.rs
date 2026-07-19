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
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{CsaAttentionMode, CsaCheckpointJournal, CudaExecutionProvider};
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
const RATIO4: usize = 4;
const RATIO4_SEQUENCE: usize = 4;
const RATIO4_INDEX_HEADS: usize = 2;
const RATIO4_INDEX_DIM: usize = 128;
const RATIO4_MAIN_WIDTH: usize = DIM * 2;
const RATIO4_INDEX_COMPRESSOR_WIDTH: usize = RATIO4_INDEX_DIM * 2;
const RATIO4_INDEX_STORED_WIDTH: usize = (RATIO4_INDEX_DIM / 32) * (16 + 1);

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

/// Serializes GPU test bodies within this binary. The capture/replay test uses
/// `CU_STREAM_CAPTURE_MODE_GLOBAL`, under which any concurrent CUDA alloc/launch
/// from another test thread in the same process/context errors out. Holding this
/// lock for the whole test body (via [`GpuGuard`]) keeps capture from overlapping
/// other CUDA work. Separate test binaries run in separate processes/contexts, so
/// no cross-binary serialization is needed.
static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A live CUDA EP plus the held [`GPU_SERIAL`] guard. Derefs to the EP so every
/// existing `gpu()` call site is unchanged.
struct GpuGuard {
    ep: CudaExecutionProvider,
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for GpuGuard {
    type Target = CudaExecutionProvider;
    fn deref(&self) -> &CudaExecutionProvider {
        &self.ep
    }
}

fn gpu() -> Option<GpuGuard> {
    // Ignore poisoning: a panicking test still leaves the device usable, and we
    // must not cascade one failure into spurious lock failures elsewhere.
    let serial = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    match CudaExecutionProvider::new_default() {
        Ok(ep) => Some(GpuGuard {
            ep,
            _serial: serial,
        }),
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

fn ratio128_node_with_bias(
    inputs: &[HostTensor],
    next_records: usize,
    bias: &HostTensor,
) -> (Graph, NodeId) {
    let (mut graph, node) = ratio128_node(inputs, next_records);
    let attention_bias = graph.create_named_value(
        "attention_bias",
        bias.dtype,
        static_shape(bias.shape.iter().copied()),
    );
    graph.add_input(attention_bias);
    graph.node_mut(node).inputs.resize(19, None);
    graph.node_mut(node).inputs.push(Some(attention_bias));
    (graph, node)
}

fn ratio128_bias_claim_metadata(
    inputs: &[HostTensor],
    bias: &HostTensor,
) -> (Vec<onnx_runtime_ir::Shape>, Vec<DataType>) {
    let (mut shapes, mut dtypes) = claim_metadata(inputs);
    shapes.resize(19, Vec::new());
    dtypes.resize(19, DataType::Undefined);
    shapes.push(static_shape(bias.shape.iter().copied()));
    dtypes.push(bias.dtype);
    (shapes, dtypes)
}

fn ratio4_values(len: usize, offset: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| 0.05 + ((index * 17 + offset) % 97) as f32 * scale)
        .collect()
}

fn ratio4_inputs() -> Vec<HostTensor> {
    vec![
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, 1, DIM],
            &ratio4_values(RATIO4_SEQUENCE * DIM, 3, 0.0005),
        ),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, DIM],
            &ratio4_values(RATIO4_SEQUENCE * DIM, 5, 0.003),
        ),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, RATIO4_MAIN_WIDTH],
            &ratio4_values(RATIO4_SEQUENCE * RATIO4_MAIN_WIDTH, 7, 0.002),
        ),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, RATIO4_MAIN_WIDTH],
            &ratio4_values(RATIO4_SEQUENCE * RATIO4_MAIN_WIDTH, 11, 0.001),
        ),
        HostTensor::f32(
            &[RATIO4, RATIO4_MAIN_WIDTH],
            &ratio4_values(RATIO4 * RATIO4_MAIN_WIDTH, 13, 0.0002),
        ),
        HostTensor::f32(
            &[DIM],
            &(0..DIM)
                .map(|index| 0.75 + (index % 19) as f32 * 0.01)
                .collect::<Vec<_>>(),
        ),
        HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        HostTensor::zeros(DataType::Float32, &[1, 8, 2, RATIO4_MAIN_WIDTH]),
        HostTensor::i32(&[1], &[(RATIO4_SEQUENCE - 1) as i32]),
        HostTensor::i64(&[], &[RATIO4_SEQUENCE as i64]),
        HostTensor::f32(&[1], &[-0.41]),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, RATIO4_INDEX_HEADS, RATIO4_INDEX_DIM],
            &ratio4_values(
                RATIO4_SEQUENCE * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
                17,
                0.0015,
            ),
        ),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, RATIO4_INDEX_HEADS],
            &ratio4_values(RATIO4_SEQUENCE * RATIO4_INDEX_HEADS, 19, 0.01),
        ),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &ratio4_values(RATIO4_SEQUENCE * RATIO4_INDEX_COMPRESSOR_WIDTH, 23, 0.002),
        ),
        HostTensor::f32(
            &[1, RATIO4_SEQUENCE, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &ratio4_values(RATIO4_SEQUENCE * RATIO4_INDEX_COMPRESSOR_WIDTH, 29, 0.001),
        ),
        HostTensor::f32(
            &[RATIO4, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &ratio4_values(RATIO4 * RATIO4_INDEX_COMPRESSOR_WIDTH, 31, 0.0002),
        ),
        HostTensor::f32(
            &[RATIO4_INDEX_DIM],
            &(0..RATIO4_INDEX_DIM)
                .map(|index| 0.8 + (index % 13) as f32 * 0.0125)
                .collect::<Vec<_>>(),
        ),
        HostTensor::zeros(DataType::Uint8, &[1, 0, RATIO4_INDEX_STORED_WIDTH]),
        HostTensor::zeros(DataType::Float32, &[1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH]),
    ]
}

fn ratio4_node(inputs: &[HostTensor]) -> (Graph, NodeId) {
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
        "index_query",
        "index_weight",
        "index_compressor_kv",
        "index_compressor_gate",
        "index_compressor_ape",
        "index_compressor_norm",
        "past_index_key",
        "past_index_carry",
    ];
    let node_inputs = inputs
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
    let sequence = inputs[0].shape[1];
    let total = i64::from_ne_bytes(inputs[9].bytes.as_slice().try_into().unwrap()) as usize;
    let start = total - sequence;
    let records = inputs[17].shape[1] + total / RATIO4 - start / RATIO4;
    let topk = (total / RATIO4).min(512);
    let outputs = vec![
        graph.create_named_value("Y", DataType::Float32, static_shape([1, sequence, 1, DIM])),
        graph.create_named_value(
            "present_compressed_kv",
            DataType::Uint8,
            static_shape([1, records, STORED_WIDTH]),
        ),
        graph.create_named_value(
            "present_compression_carry",
            DataType::Float32,
            static_shape([1, 8, 2, RATIO4_MAIN_WIDTH]),
        ),
        graph.create_named_value(
            "present_index_key",
            DataType::Uint8,
            static_shape([1, records, RATIO4_INDEX_STORED_WIDTH]),
        ),
        graph.create_named_value(
            "present_index_carry",
            DataType::Float32,
            static_shape([1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH]),
        ),
        graph.create_named_value(
            "selected_indices",
            DataType::Int32,
            static_shape([1, RATIO4_INDEX_HEADS, sequence, topk]),
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
        .insert("compression_ratio".into(), Attribute::Int(RATIO4 as i64));
    node.attributes.insert(
        "index_num_heads".into(),
        Attribute::Int(RATIO4_INDEX_HEADS as i64),
    );
    node.attributes.insert(
        "index_head_dim".into(),
        Attribute::Int(RATIO4_INDEX_DIM as i64),
    );
    node.attributes
        .insert("index_topk".into(), Attribute::Int(1));
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

fn ratio4_node_with_bias(inputs: &[HostTensor], bias: &HostTensor) -> (Graph, NodeId) {
    let (mut graph, node) = ratio4_node(inputs);
    let value = graph.create_named_value(
        "attention_bias",
        bias.dtype,
        static_shape(bias.shape.iter().copied()),
    );
    graph.add_input(value);
    graph.node_mut(node).inputs.resize(19, None);
    graph.node_mut(node).inputs.push(Some(value));
    (graph, node)
}

/// Same node as [`ratio4_node`] but omitting the optional `selected_indices`
/// (output 5), yielding a valid **5-output** ratio-4 node. `selected_indices` is
/// optional for ratio-4, so both the CPU `validate_ratio_specific_v1_schema` and
/// the CUDA `validate_ratio4_claim` accept 5..=6 outputs and still claim it. With
/// no device-selected record stream to dereference, its fused-attention `Y` must
/// come from the host oracle — never the ratio-128 device kernel.
fn ratio4_node_five_outputs(inputs: &[HostTensor]) -> (Graph, NodeId) {
    let (mut graph, node) = ratio4_node(inputs);
    let dropped = graph
        .node_mut(node)
        .outputs
        .pop()
        .expect("ratio4_node builds 6 outputs");
    if let Some(index) = graph.outputs.iter().position(|&value| value == dropped) {
        graph.remove_output(index);
    }
    (graph, node)
}

fn claim_metadata(inputs: &[HostTensor]) -> (Vec<onnx_runtime_ir::Shape>, Vec<DataType>) {
    (
        inputs
            .iter()
            .map(|input| static_shape(input.shape.iter().copied()))
            .collect(),
        inputs.iter().map(|input| input.dtype).collect(),
    )
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

#[test]
fn ratio128_device_compression_crosses_two_blocks_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(
        &[DIM],
        &(0..DIM)
            .map(|d| 0.75 + (d % 17) as f32 * 0.03125)
            .collect::<Vec<_>>(),
    );
    let sink = HostTensor::f32(&[1], &[-0.375]);
    let initial = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };

    // The prefill emits block [0, 128), leaves [128, 255) in carry, then the
    // final decode emits [128, 256).  This catches both FP8 record placement and
    // the full carry reset between adjacent ratio-128 blocks.
    let prefill = ratio128_inputs(255, 0, 0, 255, 255, &ape, &norm, &sink, &initial);
    let after_prefill = run_step(&ep, &prefill, 1);
    let decode = ratio128_inputs(1, 255, 0, 256, 256, &ape, &norm, &sink, &after_prefill);
    let after_decode = run_step(&ep, &decode, 2);
    assert_eq!(after_decode.cache.shape, vec![1, 2, STORED_WIDTH]);
}

// ---------------------------------------------------------------------------
// B1: ratio-128 f32-cache DEVICE sink-softmax attention parity.
//
// With cache_format="f32", `present_compressed_kv` is the f32 dequantized
// candidate-record buffer, so the CUDA kernel runs stage-6 (candidate read) +
// stage-7 (sparse sink-softmax attention) on device (compression/state stay
// host). Y must match the CPU oracle bit-for-bit (ULP=0). The all-Host fp8
// tests above stay green (the device path only engages for f32 cache).
// ---------------------------------------------------------------------------

/// f32-record ratio-128 node: cache_format="f32", `present_compressed_kv` is
/// f32 `[batch, next_records, DIM]`.
fn ratio128_node_f32(inputs: &[HostTensor], next_records: usize) -> (Graph, NodeId) {
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
            DataType::Float32,
            static_shape([batch, next_records, DIM]),
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
        Attribute::String("f32".as_bytes().to_vec()),
    );
    let node = graph.insert_node(node);
    for output in outputs {
        graph.add_output(output);
    }
    (graph, node)
}

/// Max-ULP (sign-magnitude ordered) between two f32 buffers, matching the GQA
/// reference-parity metric.
fn max_ulp(gpu: &[u8], cpu: &[u8]) -> u32 {
    let gpu = as_f32(gpu);
    let cpu = as_f32(cpu);
    assert_eq!(gpu.len(), cpu.len(), "length mismatch");
    gpu.iter()
        .zip(&cpu)
        .map(|(&g, &c)| {
            let key = |v: f32| {
                if v.is_sign_negative() {
                    !v.to_bits()
                } else {
                    v.to_bits() | 0x8000_0000
                }
            };
            key(g).abs_diff(key(c))
        })
        .max()
        .unwrap_or(0)
}

/// One f32-cache `prefill/decode` step with the device attention stage engaged.
/// Asserts bit-exact (ULP=0) `Y` parity plus state parity, returning the present
/// state (with an f32 cache) and the device/CPU `Y` for further checks.
fn run_step_f32(
    ep: &CudaExecutionProvider,
    inputs: &[HostTensor],
    next_records: usize,
) -> (Vec<f32>, Vec<f32>, CsaState) {
    let sequence = inputs[0].shape[1];
    let output_specs = vec![
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, sequence, 1, DIM],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, next_records, DIM],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, RATIO, 2, DIM],
        },
    ];
    let (graph, node) = ratio128_node_f32(inputs, next_records);
    let cpu = run_cpu(&graph, node, inputs, &output_specs).expect("CPU CSA kernel");
    let gpu = run_gpu(ep, &graph, node, inputs, &output_specs).expect("CUDA CSA kernel");

    let ulp = max_ulp(&gpu[0], &cpu[0]);
    eprintln!("ratio128 f32 device attention: Y max_ulp={ulp}");
    assert_eq!(
        ulp, 0,
        "device sink-softmax attention Y must match the CPU oracle bit-for-bit"
    );
    // State stays host-staged, so it is byte-identical to the oracle.
    assert_eq!(gpu[1], cpu[1], "present_compressed_kv must match exactly");
    assert_f32_close(&gpu[2], &cpu[2], 1e-4, "present_compression_carry");

    let state = CsaState {
        cache: HostTensor::f32(&output_specs[1].shape, &as_f32(&cpu[1])),
        carry: HostTensor::f32(&output_specs[2].shape, &as_f32(&cpu[2])),
    };
    (as_f32(&gpu[0]), as_f32(&cpu[0]), state)
}

#[test]
fn ratio128_f32_device_attention_matches_cpu() {
    let Some(ep) = gpu() else { return };

    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(
        &[DIM],
        &(0..DIM)
            .map(|d| 0.75 + (d % 17) as f32 * 0.03125)
            .collect::<Vec<_>>(),
    );
    let sink = HostTensor::f32(&[1], &[-0.375]);

    // Prefill positions 0..125 (S=126, total=126, next_records=0). Each query s
    // sees only dense candidates with absolute<=s; the rest are `-1` invalid
    // (skipped) — the fused invalid-candidate path. Attention is dense window +
    // sink only.
    let initial = CsaState {
        cache: HostTensor::f32(&[1, 0, DIM], &[]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let prefill = ratio128_inputs(126, 0, 0, 126, 126, &ape, &norm, &sink, &initial);
    let (_, _, after_prefill) = run_step_f32(&ep, &prefill, 0);

    // Decode position 126 (total=127, still 0 compressed records): full 127-wide
    // dense window + sink.
    let decode1 = ratio128_inputs(1, 126, 0, 127, 127, &ape, &norm, &sink, &after_prefill);
    let (_, _, after_decode1) = run_step_f32(&ep, &decode1, 0);

    // Decode position 127 (total=128) crosses the 128 block boundary → emits the
    // first f32 compressed record. valid_compressed=1, so the device attention
    // now includes a completed compressed candidate alongside the full 128-wide
    // dense window and the sink.
    let decode2 = ratio128_inputs(1, 127, 0, 128, 128, &ape, &norm, &sink, &after_decode1);
    let (_, _, after_decode2) = run_step_f32(&ep, &decode2, 1);
    assert_eq!(after_decode2.cache.shape, vec![1, 1, DIM]);
}

#[test]
fn ratio128_f32_device_attention_sink_material_matches_cpu() {
    let Some(ep) = gpu() else { return };

    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(
        &[DIM],
        &(0..DIM)
            .map(|d| 0.75 + (d % 17) as f32 * 0.03125)
            .collect::<Vec<_>>(),
    );

    // A single decode at position 4 (total=5, dense window 0..4, no compressed
    // records). Run once with a negligible sink and once with a large positive
    // sink: the large sink adds `exp(sink - max)` mass to the denominator,
    // measurably shrinking `Y`. Both device runs must match the CPU oracle
    // bit-for-bit, proving the sink-after-max term is reproduced exactly.
    let carry = HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]);
    let past = CsaState {
        cache: HostTensor::f32(&[1, 0, DIM], &[]),
        carry: carry.clone(),
    };

    let small_sink = HostTensor::f32(&[1], &[-30.0]);
    let inputs_small = ratio128_inputs(1, 4, 0, 5, 5, &ape, &norm, &small_sink, &past);
    let (y_small_gpu, y_small_cpu, _) = run_step_f32(&ep, &inputs_small, 0);

    let large_sink = HostTensor::f32(&[1], &[6.0]);
    let inputs_large = ratio128_inputs(1, 4, 0, 5, 5, &ape, &norm, &large_sink, &past);
    let (y_large_gpu, y_large_cpu, _) = run_step_f32(&ep, &inputs_large, 0);

    // The large sink must materially change the output (denominator dominated by
    // the sink mass), not just perturb it in the last bits.
    let max_delta = y_small_cpu
        .iter()
        .zip(&y_large_cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-2,
        "sink term should materially change Y; max delta was {max_delta:e}"
    );
    // Device already asserted ULP=0 vs CPU inside run_step_f32; double-check the
    // two device runs differ too (sink genuinely flows through the device path).
    let device_delta = y_small_gpu
        .iter()
        .zip(&y_large_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        device_delta > 1e-2,
        "device sink path inert: {device_delta:e}"
    );
}

#[test]
fn ratio4_prefill_claim_and_execute_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let inputs = ratio4_inputs();
    let (graph, node) = ratio4_node(&inputs);
    let (shapes, dtypes) = claim_metadata(&inputs);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Supported { .. }
        ),
        "valid ratio-4 CSA must be claimed"
    );

    let output_specs = vec![
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, RATIO4_SEQUENCE, 1, DIM],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, 1, STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_MAIN_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, 1, RATIO4_INDEX_STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Int32,
            shape: vec![1, RATIO4_INDEX_HEADS, RATIO4_SEQUENCE, 1],
        },
    ];
    let cpu = run_cpu(&graph, node, &inputs, &output_specs).expect("CPU ratio-4 CSA kernel");
    let gpu = run_gpu(&ep, &graph, node, &inputs, &output_specs).expect("CUDA ratio-4 CSA kernel");
    assert_eq!(
        max_ulp(&gpu[0], &cpu[0]),
        0,
        "ratio-4 fused Y (no bias) must be bit-exact vs the CPU oracle"
    );
    assert_eq!(gpu[1], cpu[1], "ratio-4 compressed cache");
    assert_f32_close(&gpu[2], &cpu[2], 1e-4, "ratio-4 compression carry");
    assert_eq!(gpu[3], cpu[3], "ratio-4 index cache");
    assert_f32_close(&gpu[4], &cpu[4], 1e-4, "ratio-4 index carry");
    assert_eq!(gpu[5], cpu[5], "ratio-4 selected indices");
}

/// Build a full ratio-4 prefill input set for an arbitrary sequence length,
/// letting the caller override the `index_query` / `index_weight` streams so a
/// test can engineer clustered (near-tie) candidate scores. `start == 0`, so
/// `total == seq` and `next_records == seq / 4`.
fn ratio4_inputs_prefill(
    seq: usize,
    index_query: Vec<f32>,
    index_weight: Vec<f32>,
) -> Vec<HostTensor> {
    vec![
        HostTensor::f32(&[1, seq, 1, DIM], &ratio4_values(seq * DIM, 3, 0.0005)),
        HostTensor::f32(&[1, seq, DIM], &ratio4_values(seq * DIM, 5, 0.003)),
        HostTensor::f32(
            &[1, seq, RATIO4_MAIN_WIDTH],
            &ratio4_values(seq * RATIO4_MAIN_WIDTH, 7, 0.002),
        ),
        HostTensor::f32(
            &[1, seq, RATIO4_MAIN_WIDTH],
            &ratio4_values(seq * RATIO4_MAIN_WIDTH, 11, 0.001),
        ),
        HostTensor::f32(
            &[RATIO4, RATIO4_MAIN_WIDTH],
            &ratio4_values(RATIO4 * RATIO4_MAIN_WIDTH, 13, 0.0002),
        ),
        HostTensor::f32(
            &[DIM],
            &(0..DIM)
                .map(|index| 0.75 + (index % 19) as f32 * 0.01)
                .collect::<Vec<_>>(),
        ),
        HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        HostTensor::zeros(DataType::Float32, &[1, 8, 2, RATIO4_MAIN_WIDTH]),
        HostTensor::i32(&[1], &[(seq - 1) as i32]),
        HostTensor::i64(&[], &[seq as i64]),
        HostTensor::f32(&[1], &[-0.41]),
        HostTensor::f32(
            &[1, seq, RATIO4_INDEX_HEADS, RATIO4_INDEX_DIM],
            &index_query,
        ),
        HostTensor::f32(&[1, seq, RATIO4_INDEX_HEADS], &index_weight),
        HostTensor::f32(
            &[1, seq, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &ratio4_values(seq * RATIO4_INDEX_COMPRESSOR_WIDTH, 23, 0.002),
        ),
        HostTensor::f32(
            &[1, seq, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &ratio4_values(seq * RATIO4_INDEX_COMPRESSOR_WIDTH, 29, 0.001),
        ),
        HostTensor::f32(
            &[RATIO4, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &ratio4_values(RATIO4 * RATIO4_INDEX_COMPRESSOR_WIDTH, 31, 0.0002),
        ),
        HostTensor::f32(
            &[RATIO4_INDEX_DIM],
            &(0..RATIO4_INDEX_DIM)
                .map(|index| 0.8 + (index % 13) as f32 * 0.0125)
                .collect::<Vec<_>>(),
        ),
        HostTensor::zeros(DataType::Uint8, &[1, 0, RATIO4_INDEX_STORED_WIDTH]),
        HostTensor::zeros(DataType::Float32, &[1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH]),
    ]
}

/// Same node as [`ratio4_node`] but with a caller-chosen `index_topk`, sizing
/// `selected_indices` to `index_topk.min(next_records)` so top-k selection over
/// multiple candidate records is exercised.
fn ratio4_node_topk(inputs: &[HostTensor], index_topk: usize) -> (Graph, NodeId) {
    let (mut graph, node) = ratio4_node(inputs);
    let sequence = inputs[0].shape[1];
    let total = i64::from_ne_bytes(inputs[9].bytes.as_slice().try_into().unwrap()) as usize;
    let start = total - sequence;
    let next_records = total / RATIO4;
    let records = inputs[17].shape[1] + next_records - start / RATIO4;
    let topk = index_topk.min(next_records);
    graph
        .node_mut(node)
        .attributes
        .insert("index_topk".into(), Attribute::Int(index_topk as i64));
    // Re-point `selected_indices` (output 5) at a value with the correct top-k
    // width. The other outputs already match `records` from `ratio4_node`.
    let selected = graph.create_named_value(
        "selected_indices_topk",
        DataType::Int32,
        static_shape([1, RATIO4_INDEX_HEADS, sequence, topk]),
    );
    graph.node_mut(node).outputs[5] = selected;
    let _ = records;
    (graph, node)
}

fn ratio4_topk_output_specs(sequence: usize, records: usize, topk: usize) -> Vec<OutputSpec> {
    vec![
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, sequence, 1, DIM],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, records, STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_MAIN_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, records, RATIO4_INDEX_STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Int32,
            shape: vec![1, RATIO4_INDEX_HEADS, sequence, topk],
        },
    ]
}

/// B4 — device stages 3–5 must reproduce the CPU oracle's `selected_indices`
/// **bit-for-bit** across many causal candidate records and a wide top-k, so the
/// full `sorted=True, largest=True` ordering and the `-1` causal padding are all
/// exercised (not just the single-record prefill case). Compared against the
/// INDEPENDENT CPU oracle, never the device against itself.
#[test]
fn ratio4_device_topk_selection_multi_record_matches_cpu() {
    let Some(ep) = gpu() else { return };
    const SEQ: usize = 16;
    let index_query = ratio4_values(SEQ * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM, 17, 0.0015);
    let index_weight = ratio4_values(SEQ * RATIO4_INDEX_HEADS, 19, 0.01);
    let inputs = ratio4_inputs_prefill(SEQ, index_query, index_weight);
    // next_records = 16/4 = 4; a top-k of 4 fully orders every causal record.
    let index_topk = 4;
    let records = SEQ / RATIO4;
    let (graph, node) = ratio4_node_topk(&inputs, index_topk);
    let specs = ratio4_topk_output_specs(SEQ, records, index_topk.min(records));

    let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU ratio-4 top-k oracle");
    let gpu = run_gpu(&ep, &graph, node, &inputs, &specs).expect("CUDA ratio-4 top-k");

    // The selection must actually be non-trivial: the deepest queries fill every
    // top-k slot with a real (non `-1`) record, and shallow queries pad with -1.
    let selected: Vec<i32> = gpu[5]
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        selected.iter().any(|&v| v >= 0) && selected.contains(&-1),
        "multi-record top-k must exercise both real selections and -1 padding: {selected:?}"
    );
    assert_eq!(
        gpu[5], cpu[5],
        "device deterministic top-k selection must match the CPU oracle bit-for-bit"
    );
    assert_eq!(gpu[3], cpu[3], "ratio-4 index cache");
}

/// B4 adversarial tie fixture — engineer clustered candidate scores (tiny,
/// near-equal per-record contributions) so the deterministic tie order is
/// genuinely exercised. If the device reduction order or the total-order
/// comparator diverged from the oracle by a single ULP, the winning record would
/// flip; asserting bit-identical `selected_indices` vs the independent oracle
/// catches it.
#[test]
fn ratio4_device_topk_tie_break_matches_cpu() {
    let Some(ep) = gpu() else { return };
    const SEQ: usize = 12;
    // Near-constant, tiny index queries make every `dot` tiny and the per-record
    // scores cluster in the last mantissa bits — the worst case for tie order.
    let index_query: Vec<f32> = (0..SEQ * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM)
        .map(|index| 0.5 + ((index % 7) as f32) * 1.0e-4)
        .collect();
    let index_weight: Vec<f32> = (0..SEQ * RATIO4_INDEX_HEADS)
        .map(|index| 0.25 + ((index % 3) as f32) * 1.0e-3)
        .collect();
    let inputs = ratio4_inputs_prefill(SEQ, index_query, index_weight);
    let index_topk = 3;
    let records = SEQ / RATIO4;
    let (graph, node) = ratio4_node_topk(&inputs, index_topk);
    let specs = ratio4_topk_output_specs(SEQ, records, index_topk.min(records));

    let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU ratio-4 tie oracle");
    let gpu = run_gpu(&ep, &graph, node, &inputs, &specs).expect("CUDA ratio-4 tie selection");
    assert_eq!(
        gpu[5], cpu[5],
        "device tie-break order must be bit-identical to the CPU oracle"
    );
}

fn ratio4_sequence_slice(tensor: &HostTensor, first: usize, count: usize) -> HostTensor {
    let mut shape = tensor.shape.clone();
    let row_bytes = tensor.bytes.len() / shape[1];
    shape[1] = count;
    HostTensor {
        dtype: tensor.dtype,
        shape,
        bytes: tensor.bytes[first * row_bytes..(first + count) * row_bytes].to_vec(),
    }
}

#[test]
fn ratio4_device_index_stream_matches_cpu_oracle_across_decode_boundary() {
    let Some(ep) = gpu() else { return };
    let full = ratio4_inputs();
    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, 3);
    }
    prefill[8] = HostTensor::i32(&[1], &[2]);
    prefill[9] = HostTensor::i64(&[], &[3]);
    let prefill_specs = vec![
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 3, 1, DIM],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, 0, STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_MAIN_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, 0, RATIO4_INDEX_STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Int32,
            shape: vec![1, RATIO4_INDEX_HEADS, 3, 0],
        },
    ];
    let (prefill_graph, prefill_node) = ratio4_node(&prefill);
    let prefill_cpu = run_cpu(&prefill_graph, prefill_node, &prefill, &prefill_specs).unwrap();
    let prefill_gpu = run_gpu(&ep, &prefill_graph, prefill_node, &prefill, &prefill_specs).unwrap();
    assert_eq!(prefill_gpu[3], prefill_cpu[3], "prefix index key");
    assert_eq!(prefill_gpu[4], prefill_cpu[4], "prefix index carry");
    assert_eq!(
        prefill_gpu[5], prefill_cpu[5],
        "prefix selected_indices (empty top-k)"
    );

    let mut decode = full.clone();
    for index in [0usize, 2, 3, 11, 12, 13, 14] {
        decode[index] = ratio4_sequence_slice(&full[index], 3, 1);
    }
    decode[6] = HostTensor::u8(&[1, 0, STORED_WIDTH], &prefill_cpu[1]);
    decode[7] = HostTensor::f32(&[1, 8, 2, RATIO4_MAIN_WIDTH], &as_f32(&prefill_cpu[2]));
    decode[8] = HostTensor::i32(&[1], &[3]);
    decode[9] = HostTensor::i64(&[], &[4]);
    decode[17] = HostTensor::u8(&[1, 0, RATIO4_INDEX_STORED_WIDTH], &prefill_cpu[3]);
    decode[18] = HostTensor::f32(
        &[1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
        &as_f32(&prefill_cpu[4]),
    );
    let decode_specs = vec![
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 1, 1, DIM],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, 1, STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_MAIN_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Uint8,
            shape: vec![1, 1, RATIO4_INDEX_STORED_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Float32,
            shape: vec![1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
        },
        OutputSpec {
            dtype: DataType::Int32,
            shape: vec![1, RATIO4_INDEX_HEADS, 1, 1],
        },
    ];
    let (decode_graph, decode_node) = ratio4_node(&decode);
    let decode_cpu = run_cpu(&decode_graph, decode_node, &decode, &decode_specs).unwrap();
    let decode_gpu = run_gpu(&ep, &decode_graph, decode_node, &decode, &decode_specs).unwrap();
    assert_eq!(
        decode_gpu[3], decode_cpu[3],
        "device FP4 index key at boundary"
    );
    assert_eq!(
        decode_gpu[4], decode_cpu[4],
        "device index carry including overlap shift and c=0"
    );
    assert_eq!(
        decode_gpu[5], decode_cpu[5],
        "device deterministic top-k selection at the decode boundary"
    );
}

/// B5 exercises the complete ratio-4 device path through a prefill and two
/// decodes. The bias is rank-4 so its candidate axis proves the device fused
/// candidate ordering is `[dense window, selected slot 0]`, not record order.
#[test]
fn ratio4_device_fused_attention_prefill_then_two_decodes_with_bias_matches_cpu() {
    let Some(ep) = gpu() else { return };
    const PREFILL: usize = 12;
    let full = ratio4_inputs_prefill(
        PREFILL + 2,
        ratio4_values(
            (PREFILL + 2) * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
            17,
            0.0015,
        ),
        ratio4_values((PREFILL + 2) * RATIO4_INDEX_HEADS, 19, 0.01),
    );

    let run = |mut inputs: Vec<HostTensor>, records: usize| -> Vec<Vec<u8>> {
        let sequence = inputs[0].shape[1];
        let dense_candidates = if inputs[1].shape[1] == sequence {
            inputs[1].shape[1].min(128)
        } else {
            128
        };
        let bias = HostTensor::f32(
            &[1, 1, sequence, dense_candidates + 1],
            &ratio4_values(sequence * (dense_candidates + 1), 43, 0.0003),
        );
        let (graph, node) = ratio4_node_with_bias(&inputs, &bias);
        inputs.push(bias);
        let specs = ratio4_topk_output_specs(sequence, records, 1);
        let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU ratio-4 CSA oracle");
        let gpu = run_gpu(&ep, &graph, node, &inputs, &specs).expect("CUDA ratio-4 CSA");
        assert_eq!(max_ulp(&gpu[0], &cpu[0]), 0, "ratio-4 fused Y");
        for index in 1..6 {
            assert_eq!(gpu[index], cpu[index], "ratio-4 output {index}");
        }
        cpu
    };

    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, PREFILL);
    }
    prefill[8] = HostTensor::i32(&[1], &[(PREFILL - 1) as i32]);
    prefill[9] = HostTensor::i64(&[], &[PREFILL as i64]);
    let first = run(prefill, PREFILL / RATIO4);

    let decode = |position: usize, previous: &[Vec<u8>]| {
        let mut inputs = full.clone();
        for index in [0usize, 2, 3, 11, 12, 13, 14] {
            inputs[index] = ratio4_sequence_slice(&full[index], position, 1);
        }
        inputs[1] = ratio4_sequence_slice(&full[1], 0, position + 1);
        inputs[6] = HostTensor::u8(&[1, position / RATIO4, STORED_WIDTH], &previous[1]);
        inputs[7] = HostTensor::f32(&[1, 8, 2, RATIO4_MAIN_WIDTH], &as_f32(&previous[2]));
        inputs[8] = HostTensor::i32(&[1], &[position as i32]);
        inputs[9] = HostTensor::i64(&[], &[(position + 1) as i64]);
        inputs[17] = HostTensor::u8(
            &[1, position / RATIO4, RATIO4_INDEX_STORED_WIDTH],
            &previous[3],
        );
        inputs[18] = HostTensor::f32(
            &[1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &as_f32(&previous[4]),
        );
        inputs
    };
    let second = run(decode(PREFILL, &first), (PREFILL + 1) / RATIO4);
    let _third = run(decode(PREFILL + 1, &second), (PREFILL + 2) / RATIO4);
}

/// B5-1 regression cover. A ratio-4 node may OMIT the optional
/// `selected_indices` (output 5) and still be claimed (both validators accept
/// 5..=6 outputs). With no device-selected record stream, its fused `Y` must
/// fall back to the host-staged oracle. The B5 regression instead keyed the
/// device dispatch on `outputs.len() == 6`: a 5-output ratio-4 node fell into
/// the `else` arm and ran `run_device_attention` (the ratio-128 kernel) over the
/// ratio-4 583-byte packed FP8 cache reinterpreted as `f32×512` — out of bounds,
/// clobbering the correct host-staged `Y`. Under the buggy dispatch this test's
/// `Y` assertion fails; the ratio-keyed fix restores host-oracle fallback. Drives
/// prefill→decode→decode and asserts every present output is bit-exact vs the
/// independent CPU oracle.
#[test]
fn ratio4_five_output_fused_attention_falls_back_to_host_oracle_bit_exact() {
    let Some(ep) = gpu() else { return };
    const PREFILL: usize = 12;
    let full = ratio4_inputs_prefill(
        PREFILL + 2,
        ratio4_values(
            (PREFILL + 2) * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
            17,
            0.0015,
        ),
        ratio4_values((PREFILL + 2) * RATIO4_INDEX_HEADS, 19, 0.01),
    );

    let run = |inputs: Vec<HostTensor>, records: usize| -> Vec<Vec<u8>> {
        let sequence = inputs[0].shape[1];
        let (graph, node) = ratio4_node_five_outputs(&inputs);
        // Drop the `selected_indices` spec so exactly 5 outputs are requested,
        // matching the 5-output node.
        let mut specs = ratio4_topk_output_specs(sequence, records, 1);
        specs.pop();
        assert_eq!(specs.len(), 5, "5-output ratio-4 node has 5 output specs");
        let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU ratio-4 CSA oracle");
        let gpu = run_gpu(&ep, &graph, node, &inputs, &specs).expect("CUDA ratio-4 CSA");
        assert_eq!(
            max_ulp(&gpu[0], &cpu[0]),
            0,
            "5-output ratio-4 Y must be the bit-exact host oracle result, never \
             the ratio-128 device kernel"
        );
        for index in 1..5 {
            assert_eq!(gpu[index], cpu[index], "5-output ratio-4 output {index}");
        }
        cpu
    };

    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, PREFILL);
    }
    prefill[8] = HostTensor::i32(&[1], &[(PREFILL - 1) as i32]);
    prefill[9] = HostTensor::i64(&[], &[PREFILL as i64]);
    let first = run(prefill, PREFILL / RATIO4);

    let decode = |position: usize, previous: &[Vec<u8>]| {
        let mut inputs = full.clone();
        for index in [0usize, 2, 3, 11, 12, 13, 14] {
            inputs[index] = ratio4_sequence_slice(&full[index], position, 1);
        }
        inputs[1] = ratio4_sequence_slice(&full[1], 0, position + 1);
        inputs[6] = HostTensor::u8(&[1, position / RATIO4, STORED_WIDTH], &previous[1]);
        inputs[7] = HostTensor::f32(&[1, 8, 2, RATIO4_MAIN_WIDTH], &as_f32(&previous[2]));
        inputs[8] = HostTensor::i32(&[1], &[position as i32]);
        inputs[9] = HostTensor::i64(&[], &[(position + 1) as i64]);
        inputs[17] = HostTensor::u8(
            &[1, position / RATIO4, RATIO4_INDEX_STORED_WIDTH],
            &previous[3],
        );
        inputs[18] = HostTensor::f32(
            &[1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
            &as_f32(&previous[4]),
        );
        inputs
    };
    let second = run(decode(PREFILL, &first), (PREFILL + 1) / RATIO4);
    let _third = run(decode(PREFILL + 1, &second), (PREFILL + 2) / RATIO4);
}
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

#[test]
fn supports_op_rejects_non_query_fixed_input_dtype() {
    let Some(ep) = gpu() else { return };
    let inputs = ratio4_inputs();
    let (graph, node) = ratio4_node(&inputs);
    let (shapes, mut dtypes) = claim_metadata(&inputs);
    dtypes[1] = DataType::Float16;
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "Float16 current_kv must be rejected at claim time"
    );
}

#[test]
fn supports_op_rejects_ratio4_non_128_index_head_dim() {
    let Some(ep) = gpu() else { return };
    let inputs = ratio4_inputs();
    let (mut graph, node) = ratio4_node(&inputs);
    graph
        .node_mut(node)
        .attributes
        .insert("index_head_dim".into(), Attribute::Int(64));
    let (shapes, dtypes) = claim_metadata(&inputs);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-4 index_head_dim != 128 must be rejected at claim time"
    );
}

#[test]
fn supports_op_rejects_ratio4_missing_index_inputs() {
    let Some(ep) = gpu() else { return };
    let inputs = ratio4_inputs();
    let (mut graph, node) = ratio4_node(&inputs);
    graph.node_mut(node).inputs.truncate(11);
    let (mut shapes, mut dtypes) = claim_metadata(&inputs);
    shapes.truncate(11);
    dtypes.truncate(11);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-4 without inputs 11..18 must be rejected at claim time"
    );
}

#[test]
fn supports_op_rejects_ratio4_wrong_output_count() {
    let Some(ep) = gpu() else { return };
    let inputs = ratio4_inputs();
    let (mut graph, node) = ratio4_node(&inputs);
    graph.node_mut(node).outputs.truncate(4);
    let (shapes, dtypes) = claim_metadata(&inputs);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-4 with fewer than five outputs must be rejected at claim time"
    );
}

#[test]
fn supports_op_rejects_ratio128_fp4_cache_format() {
    let Some(ep) = gpu() else { return };
    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(&[DIM], &vec![0.8f32; DIM]);
    let sink = HostTensor::f32(&[1], &[-0.375]);
    let state = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let inputs = ratio128_inputs(1, 0, 0, 1, 1, &ape, &norm, &sink, &state);
    let (mut graph, node) = ratio128_node(&inputs, 0);
    graph.node_mut(node).attributes.insert(
        "cache_format".into(),
        Attribute::String("fp4_e2m1_block32".as_bytes().to_vec()),
    );
    let (shapes, dtypes) = claim_metadata(&inputs);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-128 FP4 cache format must be rejected at claim time"
    );
}

#[test]
fn supports_op_rejects_ratio128_ratio4_only_input() {
    let Some(ep) = gpu() else { return };
    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(&[DIM], &vec![0.8f32; DIM]);
    let sink = HostTensor::f32(&[1], &[-0.375]);
    let state = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let inputs = ratio128_inputs(1, 0, 0, 1, 1, &ape, &norm, &sink, &state);
    let (mut graph, node) = ratio128_node(&inputs, 0);
    let index_query = graph.create_named_value(
        "index_query",
        DataType::Float32,
        static_shape([1, 1, 1, RATIO4_INDEX_DIM]),
    );
    graph.add_input(index_query);
    graph.node_mut(node).inputs.push(Some(index_query));
    let (mut shapes, mut dtypes) = claim_metadata(&inputs);
    shapes.push(static_shape([1, 1, 1, RATIO4_INDEX_DIM]));
    dtypes.push(DataType::Float32);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-4-only inputs must be rejected for ratio-128 at claim time"
    );
}

#[test]
fn supports_op_validates_ratio128_attention_bias_at_input_19() {
    let Some(ep) = gpu() else { return };
    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(&[DIM], &vec![0.8f32; DIM]);
    let sink = HostTensor::f32(&[1], &[-0.375]);
    let state = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let inputs = ratio128_inputs(1, 0, 0, 1, 1, &ape, &norm, &sink, &state);

    let bad_dtype = HostTensor::zeros(DataType::Float16, &[1, 1, 1, 1]);
    let (graph, node) = ratio128_node_with_bias(&inputs, 0, &bad_dtype);
    let (shapes, dtypes) = ratio128_bias_claim_metadata(&inputs, &bad_dtype);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-128 Float16 attention_bias at input 19 must be rejected"
    );

    let bad_rank = HostTensor::zeros(DataType::Float32, &[1, 1, 1, 1, 1]);
    let (graph, node) = ratio128_node_with_bias(&inputs, 0, &bad_rank);
    let (shapes, dtypes) = ratio128_bias_claim_metadata(&inputs, &bad_rank);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-128 rank-5 attention_bias at input 19 must be rejected"
    );

    let bad_broadcast = HostTensor::zeros(DataType::Float32, &[2, 1, 1, 1]);
    let (graph, node) = ratio128_node_with_bias(&inputs, 0, &bad_broadcast);
    let (shapes, dtypes) = ratio128_bias_claim_metadata(&inputs, &bad_broadcast);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "ratio-128 statically incompatible attention_bias must be rejected"
    );

    let valid_bias = HostTensor::zeros(DataType::Float32, &[1, 1, 1]);
    let (graph, node) = ratio128_node_with_bias(&inputs, 0, &valid_bias);
    let (shapes, dtypes) = ratio128_bias_claim_metadata(&inputs, &valid_bias);
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Supported { .. }
        ),
        "ratio-128 broadcastable f32 attention_bias at input 19 must remain claimed"
    );
}

#[test]
fn supports_op_claims_omitted_ratio128_attention_bias() {
    let Some(ep) = gpu() else { return };
    let ape = HostTensor::f32(&[RATIO, DIM], &rows(0, RATIO, ape_value));
    let norm = HostTensor::f32(&[DIM], &vec![0.8f32; DIM]);
    let sink = HostTensor::f32(&[1], &[-0.375]);
    let state = CsaState {
        cache: HostTensor::zeros(DataType::Uint8, &[1, 0, STORED_WIDTH]),
        carry: HostTensor::zeros(DataType::Float32, &[1, RATIO, 2, DIM]),
    };
    let inputs = ratio128_inputs(1, 0, 0, 1, 1, &ape, &norm, &sink, &state);
    let (mut graph, node) = ratio128_node(&inputs, 0);
    graph.node_mut(node).inputs.resize(20, None);
    let (mut shapes, mut dtypes) = claim_metadata(&inputs);
    shapes.resize(20, Vec::new());
    dtypes.resize(20, DataType::Undefined);

    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Supported { .. }
        ),
        "an omitted positional attention_bias must be treated as absent"
    );
}

/// Build the ratio-4 fp8 6-output decode step at `position` (a single query
/// token) by threading the previous step's `present_*` state into `past_*`,
/// mirroring the decode wiring used by the fused-attention decode tests.
fn ratio4_decode_step(
    full: &[HostTensor],
    position: usize,
    previous: &[Vec<u8>],
) -> Vec<HostTensor> {
    let mut inputs = full.to_vec();
    for index in [0usize, 2, 3, 11, 12, 13, 14] {
        inputs[index] = ratio4_sequence_slice(&full[index], position, 1);
    }
    inputs[1] = ratio4_sequence_slice(&full[1], 0, position + 1);
    inputs[6] = HostTensor::u8(&[1, position / RATIO4, STORED_WIDTH], &previous[1]);
    inputs[7] = HostTensor::f32(&[1, 8, 2, RATIO4_MAIN_WIDTH], &as_f32(&previous[2]));
    inputs[8] = HostTensor::i32(&[1], &[position as i32]);
    inputs[9] = HostTensor::i64(&[], &[(position + 1) as i64]);
    inputs[17] = HostTensor::u8(
        &[1, position / RATIO4, RATIO4_INDEX_STORED_WIDTH],
        &previous[3],
    );
    inputs[18] = HostTensor::f32(
        &[1, 8, 2, RATIO4_INDEX_COMPRESSOR_WIDTH],
        &as_f32(&previous[4]),
    );
    inputs
}

/// Execute the node on the CUDA EP by capturing a single `execute` into a CUDA
/// graph and replaying it via `cuGraphLaunch`. Uses one kernel instance so the
/// pooled workspace addresses are stable across warmup/capture/replay. The
/// output buffers are zeroed after warmup and before capture, so the returned
/// bytes are produced solely by the graph replay (never the warmup pass).
fn run_gpu_capture_replay(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[HostTensor],
    output_specs: &[OutputSpec],
) -> onnx_runtime_ep_api::Result<Vec<Vec<u8>>> {
    use cudarc::driver::sys::{
        CUgraph, CUgraphExec, CUstreamCaptureMode, cuGraphDestroy, cuGraphExecDestroy,
        cuGraphInstantiateWithFlags, cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture,
    };

    let model = Model::new(graph);
    let concrete: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape.clone()).collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete, 1)?;
    assert!(
        kernel.cuda_graph_compatible(),
        "ratio-4 fp8 6-output decode must advertise CUDA-graph capture eligibility"
    );
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

    // Warmup: an eager execute compiles/caches every NVRTC kernel and primes the
    // fixed-address device state before capture (module loads during capture are
    // avoided by the cache hit).
    {
        let mut out_views = make_out_views(&mut out_buffers);
        kernel.execute(&input_views, &mut out_views)?;
    }
    runtime.synchronize()?;

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

    // Capture the device-only pipeline into a CUDA graph, then instantiate and
    // launch it. The captured `execute` performs no host staging, no per-call
    // alloc/free, and skips the trailing sync while capturing.
    let captured = (|| -> onnx_runtime_ep_api::Result<()> {
        // SAFETY: `stream` is the EP's live compute stream.
        unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
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
        // Always end capture to leave the stream in a clean state, even on error.
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
    for buffer in buffers {
        ep.deallocate(buffer)?;
    }
    for buffer in out_buffers.drain(..) {
        ep.deallocate(buffer)?;
    }
    captured.map(|()| outputs)
}

/// B6 — capture a ratio-4 fp8 6-output decode step into a CUDA graph, replay it,
/// and assert **byte parity** between the eager decode and the replayed decode
/// across every output (`Y` + all present state + `selected_indices`), AND
/// bit-exact parity of the eager decode vs the INDEPENDENT CPU oracle. This is
/// non-tautological: it byte-compares two independent executions (an eager launch
/// and a graph replay from zeroed buffers) and independently checks both against
/// the CPU oracle. It also asserts the config advertises capture eligibility.
#[test]
fn ratio4_capture_replay_decode_is_byte_identical_to_eager_and_cpu() {
    let Some(ep) = gpu() else { return };
    const PREFILL: usize = 12;
    let full = ratio4_inputs_prefill(
        PREFILL + 1,
        ratio4_values(
            (PREFILL + 1) * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
            17,
            0.0015,
        ),
        ratio4_values((PREFILL + 1) * RATIO4_INDEX_HEADS, 19, 0.01),
    );

    // Establish a real decode context: a prefill produces the past state that the
    // single-token decode step then consumes.
    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, PREFILL);
    }
    prefill[8] = HostTensor::i32(&[1], &[(PREFILL - 1) as i32]);
    prefill[9] = HostTensor::i64(&[], &[PREFILL as i64]);
    let (prefill_graph, prefill_node) = ratio4_node(&prefill);
    let prefill_specs = ratio4_topk_output_specs(PREFILL, PREFILL / RATIO4, 1);
    let first = run_gpu(&ep, &prefill_graph, prefill_node, &prefill, &prefill_specs)
        .expect("CUDA ratio-4 prefill");

    // Single-token decode step at position PREFILL.
    let decode = ratio4_decode_step(&full, PREFILL, &first);
    let records = (PREFILL + 1) / RATIO4;
    let (graph, node) = ratio4_node(&decode);
    let specs = ratio4_topk_output_specs(1, records, 1);

    let cpu = run_cpu(&graph, node, &decode, &specs).expect("CPU ratio-4 decode oracle");
    let eager = run_gpu(&ep, &graph, node, &decode, &specs).expect("CUDA ratio-4 eager decode");
    let replay = run_gpu_capture_replay(&ep, &graph, node, &decode, &specs)
        .expect("CUDA ratio-4 captured+replayed decode");

    // Replay must be byte-identical to the eager device decode across all outputs.
    let labels = [
        "Y",
        "present_compressed_kv",
        "present_compression_carry",
        "present_index_key",
        "present_index_carry",
        "selected_indices",
    ];
    for (index, label) in labels.iter().enumerate() {
        assert_eq!(
            replay[index], eager[index],
            "captured+replayed decode must be byte-identical to eager for {label}"
        );
    }

    // Eager device decode must match the independent CPU oracle bit-for-bit.
    assert_eq!(
        max_ulp(&eager[0], &cpu[0]),
        0,
        "eager ratio-4 decode Y must be bit-exact vs the CPU oracle"
    );
    for (index, label) in labels.iter().enumerate().skip(1) {
        assert_eq!(
            eager[index], cpu[index],
            "eager ratio-4 decode {label} must be byte-exact vs the CPU oracle"
        );
    }
}

/// Upload `bytes` into a fresh device buffer sized exactly to it, returning the
/// buffer (kept alive by the caller). Used to stand up the persistent,
/// stable-address carry buffers a speculative in-place cache would rewind.
fn upload_device(ep: &CudaExecutionProvider, bytes: &[u8]) -> DeviceBuffer {
    let buffer = ep.allocate(bytes.len().max(1), 256).expect("device alloc");
    if !bytes.is_empty() {
        // SAFETY: allocation exactly covers `bytes`.
        unsafe {
            ep.runtime()
                .htod(bytes, cuptr(buffer.as_ptr()))
                .expect("htod");
        }
    }
    buffer
}

/// Download `len` bytes from a device buffer.
fn download_device(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut host = vec![0u8; len];
    if len > 0 {
        // SAFETY: destination exactly covers the buffer region read.
        unsafe {
            ep.runtime()
                .dtoh(&mut host, cuptr(buffer.as_ptr()))
                .expect("dtoh");
        }
    }
    host
}

/// B7 — the core speculative-rollback correctness proof (D6, §4.6).
///
/// Establishes an accepted prefix of 13 tokens (prefill 12 + one decode), takes a
/// stream-ordered checkpoint of the bounded carry state, drafts one more token
/// (speculatively overwriting the committed carry buffers in place), then rejects
/// it via `restore_prefix`. It asserts:
///   1. the draft genuinely dirtied the carry (guards against a no-op test),
///   2. `restore_prefix` rolls the carry buffers back **byte-exact** to the
///      checkpoint and resets the device sequence cursor to the accepted prefix,
///   3. a continuation decode from the restored state is **bit-identical** to a
///      fresh run that only processed the accepted tokens, and to the INDEPENDENT
///      CPU oracle (`max_ulp == 0` on `Y`, byte-exact on every present output and
///      `selected_indices`).
/// The rollback counter on the shared metrics surface is asserted to advance.
#[test]
fn ratio4_speculative_rollback_restores_accepted_prefix_bit_exact() {
    let Some(ep) = gpu() else { return };
    const PREFILL: usize = 12;
    // Two positions past the prefill: one to establish the accepted prefix
    // (decode at PREFILL), one to draft-then-reject (decode at PREFILL+1).
    let full = ratio4_inputs_prefill(
        PREFILL + 2,
        ratio4_values(
            (PREFILL + 2) * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
            17,
            0.0015,
        ),
        ratio4_values((PREFILL + 2) * RATIO4_INDEX_HEADS, 19, 0.01),
    );

    // Prefill the first PREFILL tokens.
    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, PREFILL);
    }
    prefill[8] = HostTensor::i32(&[1], &[(PREFILL - 1) as i32]);
    prefill[9] = HostTensor::i64(&[], &[PREFILL as i64]);
    let (pg, pn) = ratio4_node(&prefill);
    let pspecs = ratio4_topk_output_specs(PREFILL, PREFILL / RATIO4, 1);
    let first = run_gpu(&ep, &pg, pn, &prefill, &pspecs).expect("prefill");

    // Decode at PREFILL → the committed accepted prefix of 13 tokens.
    let committed_inputs = ratio4_decode_step(&full, PREFILL, &first);
    let crecords = (PREFILL + 1) / RATIO4;
    let (cg, cn) = ratio4_node(&committed_inputs);
    let cspecs = ratio4_topk_output_specs(1, crecords, 1);
    let committed = run_gpu(&ep, &cg, cn, &committed_inputs, &cspecs).expect("committed decode");

    // Stand up the persistent, stable-address carry buffers an in-place cache
    // would rewind, seeded with the committed (accepted-prefix) carry state.
    let main_carry = upload_device(&ep, &committed[2]);
    let index_carry = upload_device(&ep, &committed[4]);
    // Device sequence cursor scalar (i64), speculatively advanced to 14.
    let seq_scalar = upload_device(&ep, &14i64.to_ne_bytes());

    let metrics = ep.csa_metrics().clone();
    let journal = CsaCheckpointJournal::new(
        ep.runtime().clone(),
        RATIO4 as u64,
        committed[2].len(),
        committed[4].len(),
        metrics.clone(),
    )
    .expect("journal");

    const GENERATION: u64 = 0xB7;
    let accepted_cursor: u64 = 13;
    // SAFETY: both carry buffers are live and sized to the committed carry bytes.
    let checkpoint = unsafe {
        journal.checkpoint(
            cuptr(main_carry.as_ptr()),
            cuptr(index_carry.as_ptr()),
            committed[2].len(),
            committed[4].len(),
            accepted_cursor,
            GENERATION,
        )
    }
    .expect("checkpoint");
    assert_eq!(checkpoint.cursors().seq_cursor, 13);
    assert_eq!(checkpoint.cursors().compressed_len, 3);
    assert_eq!(checkpoint.cursors().compression_carry_len, 1);

    // Draft one token at PREFILL+1, speculatively overwriting the committed carry
    // buffers in place (the reject we are about to roll back).
    let draft_inputs = ratio4_decode_step(&full, PREFILL + 1, &committed);
    let drecords = (PREFILL + 2) / RATIO4;
    let (dg, dn) = ratio4_node(&draft_inputs);
    let dspecs = ratio4_topk_output_specs(1, drecords, 1);
    let draft = run_gpu(&ep, &dg, dn, &draft_inputs, &dspecs).expect("draft decode");
    // SAFETY: buffers/sizes match the committed carry extents.
    unsafe {
        ep.runtime()
            .htod(&draft[2], cuptr(main_carry.as_ptr()))
            .expect("dirty main carry");
        ep.runtime()
            .htod(&draft[4], cuptr(index_carry.as_ptr()))
            .expect("dirty index carry");
    }
    let dirty_main = download_device(&ep, &main_carry, committed[2].len());
    assert_ne!(
        dirty_main, committed[2],
        "the draft must genuinely overwrite the committed carry (non-tautological guard)"
    );

    // Reject: roll back to the accepted prefix.
    let rollbacks_before = metrics.rollback_count();
    // SAFETY: buffers are the same live carry/scalar allocations checkpointed above.
    let restored_cursors = unsafe {
        journal.restore_prefix(
            &checkpoint,
            accepted_cursor,
            GENERATION,
            cuptr(main_carry.as_ptr()),
            cuptr(index_carry.as_ptr()),
            Some(cuptr(seq_scalar.as_ptr())),
        )
    }
    .expect("restore");
    assert_eq!(restored_cursors.seq_cursor, 13);
    assert_eq!(
        metrics.rollback_count(),
        rollbacks_before + 1,
        "restore must record a rollback in the §8 metrics"
    );

    // The device carry buffers and cursor scalar are back at the accepted prefix.
    let restored_main = download_device(&ep, &main_carry, committed[2].len());
    let restored_index = download_device(&ep, &index_carry, committed[4].len());
    assert_eq!(
        restored_main, committed[2],
        "restore_prefix must roll the main carry back byte-exact"
    );
    assert_eq!(
        restored_index, committed[4],
        "restore_prefix must roll the index carry back byte-exact"
    );
    let restored_scalar = download_device(&ep, &seq_scalar, 8);
    assert_eq!(
        i64::from_ne_bytes(restored_scalar.try_into().unwrap()),
        13,
        "restore_prefix must reset the device sequence cursor to the accepted prefix"
    );

    // Continue from the RESTORED state: a decode at the accepted boundary
    // (position 13) fed the rolled-back carry.
    let mut restored_state = committed.clone();
    restored_state[2] = restored_main;
    restored_state[4] = restored_index;
    let spec_inputs = ratio4_decode_step(&full, PREFILL + 1, &restored_state);
    let (sg, sn) = ratio4_node(&spec_inputs);
    let spec_specs = ratio4_topk_output_specs(1, drecords, 1);
    let spec = run_gpu(&ep, &sg, sn, &spec_inputs, &spec_specs).expect("post-rollback continue");

    // Fresh run that only processed the accepted tokens (no speculation) + the
    // independent CPU oracle.
    let fresh_inputs = ratio4_decode_step(&full, PREFILL + 1, &committed);
    let (fg, fn_) = ratio4_node(&fresh_inputs);
    let fresh = run_gpu(&ep, &fg, fn_, &fresh_inputs, &spec_specs).expect("fresh continue");
    let cpu = run_cpu(&fg, fn_, &fresh_inputs, &spec_specs).expect("cpu oracle continue");

    let labels = [
        "Y",
        "present_compressed_kv",
        "present_compression_carry",
        "present_index_key",
        "present_index_carry",
        "selected_indices",
    ];
    assert_eq!(
        max_ulp(&spec[0], &fresh[0]),
        0,
        "post-rollback Y must be bit-identical to the fresh accepted-only run"
    );
    assert_eq!(
        max_ulp(&spec[0], &cpu[0]),
        0,
        "post-rollback Y must be bit-exact vs the independent CPU oracle"
    );
    for (index, label) in labels.iter().enumerate().skip(1) {
        assert_eq!(
            spec[index], fresh[index],
            "post-rollback {label} must equal the fresh accepted-only run"
        );
        assert_eq!(
            spec[index], cpu[index],
            "post-rollback {label} must be byte-exact vs the CPU oracle"
        );
    }

    ep.deallocate(main_carry).unwrap();
    ep.deallocate(index_carry).unwrap();
    ep.deallocate(seq_scalar).unwrap();
}

/// B7 — greedy-token bit-identity across a draft/verify/correct sequence
/// (§10-Q14). The greedy next token is `argmax(Y)`; after a checkpoint, a
/// speculative draft, and a `restore_prefix`, the continuation's `Y` — and thus
/// its argmax greedy token — must be bit-identical to a non-speculative decode
/// and to the independent CPU oracle.
#[test]
fn ratio4_greedy_token_bit_identity_across_draft_verify_correct() {
    let Some(ep) = gpu() else { return };
    const PREFILL: usize = 12;
    let full = ratio4_inputs_prefill(
        PREFILL + 2,
        ratio4_values(
            (PREFILL + 2) * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
            23,
            0.0011,
        ),
        ratio4_values((PREFILL + 2) * RATIO4_INDEX_HEADS, 27, 0.008),
    );

    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, PREFILL);
    }
    prefill[8] = HostTensor::i32(&[1], &[(PREFILL - 1) as i32]);
    prefill[9] = HostTensor::i64(&[], &[PREFILL as i64]);
    let (pg, pn) = ratio4_node(&prefill);
    let pspecs = ratio4_topk_output_specs(PREFILL, PREFILL / RATIO4, 1);
    let first = run_gpu(&ep, &pg, pn, &prefill, &pspecs).expect("prefill");

    let committed_inputs = ratio4_decode_step(&full, PREFILL, &first);
    let (cg, cn) = ratio4_node(&committed_inputs);
    let cspecs = ratio4_topk_output_specs(1, (PREFILL + 1) / RATIO4, 1);
    let committed = run_gpu(&ep, &cg, cn, &committed_inputs, &cspecs).expect("committed decode");

    let main_carry = upload_device(&ep, &committed[2]);
    let index_carry = upload_device(&ep, &committed[4]);
    let journal = CsaCheckpointJournal::new(
        ep.runtime().clone(),
        RATIO4 as u64,
        committed[2].len(),
        committed[4].len(),
        ep.csa_metrics().clone(),
    )
    .expect("journal");
    const GEN: u64 = 42;
    // SAFETY: live carry buffers sized to the committed carry extents.
    let checkpoint = unsafe {
        journal.checkpoint(
            cuptr(main_carry.as_ptr()),
            cuptr(index_carry.as_ptr()),
            committed[2].len(),
            committed[4].len(),
            13,
            GEN,
        )
    }
    .expect("checkpoint");

    // Draft + dirty.
    let draft_inputs = ratio4_decode_step(&full, PREFILL + 1, &committed);
    let (dg, dn) = ratio4_node(&draft_inputs);
    let dspecs = ratio4_topk_output_specs(1, (PREFILL + 2) / RATIO4, 1);
    let draft = run_gpu(&ep, &dg, dn, &draft_inputs, &dspecs).expect("draft");
    // SAFETY: sizes match.
    unsafe {
        ep.runtime()
            .htod(&draft[2], cuptr(main_carry.as_ptr()))
            .unwrap();
        ep.runtime()
            .htod(&draft[4], cuptr(index_carry.as_ptr()))
            .unwrap();
    }

    // Correct: restore and continue.
    // SAFETY: same live buffers as the checkpoint.
    unsafe {
        journal
            .restore_prefix(
                &checkpoint,
                13,
                GEN,
                cuptr(main_carry.as_ptr()),
                cuptr(index_carry.as_ptr()),
                None,
            )
            .expect("restore");
    }
    let mut restored_state = committed.clone();
    restored_state[2] = download_device(&ep, &main_carry, committed[2].len());
    restored_state[4] = download_device(&ep, &index_carry, committed[4].len());
    let spec_inputs = ratio4_decode_step(&full, PREFILL + 1, &restored_state);
    let (sg, sn) = ratio4_node(&spec_inputs);
    let spec = run_gpu(&ep, &sg, sn, &spec_inputs, &dspecs).expect("continue");

    let fresh_inputs = ratio4_decode_step(&full, PREFILL + 1, &committed);
    let (fg, fn_) = ratio4_node(&fresh_inputs);
    let cpu = run_cpu(&fg, fn_, &fresh_inputs, &dspecs).expect("cpu oracle");

    assert_eq!(
        max_ulp(&spec[0], &cpu[0]),
        0,
        "draft/verify/correct continuation Y must be bit-exact vs the CPU oracle"
    );
    // Greedy token = argmax(Y): identical bits ⇒ identical argmax.
    let greedy = |bytes: &[u8]| -> usize {
        as_f32(bytes)
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            })
            .0
    };
    assert_eq!(
        greedy(&spec[0]),
        greedy(&cpu[0]),
        "greedy next token must match across draft/verify/correct"
    );

    ep.deallocate(main_carry).unwrap();
    ep.deallocate(index_carry).unwrap();
}

/// B7 — the §8 observability metrics must be populated on the shared telemetry
/// surface after a default (device-path) decode: attention mode Device, the five
/// cursor lengths, device output bytes, and staging bytes avoided.
#[test]
fn ratio4_device_default_populates_observability_metrics() {
    let Some(ep) = gpu() else { return };
    const PREFILL: usize = 12;
    let full = ratio4_inputs_prefill(
        PREFILL + 1,
        ratio4_values(
            (PREFILL + 1) * RATIO4_INDEX_HEADS * RATIO4_INDEX_DIM,
            17,
            0.0015,
        ),
        ratio4_values((PREFILL + 1) * RATIO4_INDEX_HEADS, 19, 0.01),
    );
    let mut prefill = full.clone();
    for index in [0usize, 1, 2, 3, 11, 12, 13, 14] {
        prefill[index] = ratio4_sequence_slice(&full[index], 0, PREFILL);
    }
    prefill[8] = HostTensor::i32(&[1], &[(PREFILL - 1) as i32]);
    prefill[9] = HostTensor::i64(&[], &[PREFILL as i64]);
    let (pg, pn) = ratio4_node(&prefill);
    let pspecs = ratio4_topk_output_specs(PREFILL, PREFILL / RATIO4, 1);
    let first = run_gpu(&ep, &pg, pn, &prefill, &pspecs).expect("prefill");

    let decode = ratio4_decode_step(&full, PREFILL, &first);
    let (graph, node) = ratio4_node(&decode);
    let specs = ratio4_topk_output_specs(1, (PREFILL + 1) / RATIO4, 1);
    run_gpu(&ep, &graph, node, &decode, &specs).expect("device decode");

    let layer_id = graph.node(node).id.0 as u64;
    let layer = ep
        .csa_metrics()
        .layer(layer_id)
        .expect("device decode must record §8 metrics for the layer");
    assert_eq!(
        layer.mode,
        CsaAttentionMode::Device,
        "the default ratio-4 fp8 decode must report Device attention mode"
    );
    assert_eq!(layer.cursors.seq_cursor, 13, "seq cursor length");
    assert_eq!(layer.cursors.compressed_len, 3, "compressed records length");
    assert_eq!(layer.cursors.index_len, 3, "index records length");
    assert!(
        layer.device_bytes > 0,
        "device output bytes must be recorded"
    );
    assert!(
        layer.bytes_avoided > 0,
        "device path must record host-staging bytes avoided"
    );
    assert_eq!(layer.host_bytes, 0, "device path stages nothing host-side");
    assert!(ep.csa_metrics().bytes_avoided_total() > 0);
}

/// B7 — composite checkpoint atomicity across two CSA layers (the non-model
/// portion of the MTP-6 concern). Two independent journals (target + a second
/// CSA layer) are checkpointed and rolled back together; both must restore their
/// bounded carry state byte-exact, demonstrating the engine can orchestrate a
/// composite rollback over per-layer backend journals (D6).
#[test]
fn csa_composite_checkpoint_rolls_back_all_layers() {
    let Some(ep) = gpu() else { return };
    let committed_a = vec![0.25f32; 64];
    let committed_b = vec![-1.5f32; 96];
    let bytes_a: Vec<u8> = committed_a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let bytes_b: Vec<u8> = committed_b.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let carry_a = upload_device(&ep, &bytes_a);
    let carry_b = upload_device(&ep, &bytes_b);
    let metrics = ep.csa_metrics().clone();
    let journal_a =
        CsaCheckpointJournal::new(ep.runtime().clone(), 4, bytes_a.len(), 1, metrics.clone())
            .unwrap();
    let journal_b =
        CsaCheckpointJournal::new(ep.runtime().clone(), 4, bytes_b.len(), 1, metrics.clone())
            .unwrap();

    const GEN: u64 = 9;
    // SAFETY: live buffers sized to the committed carry extents.
    let (ck_a, ck_b) = unsafe {
        let a = journal_a
            .checkpoint(cuptr(carry_a.as_ptr()), 0, bytes_a.len(), 0, 8, GEN)
            .unwrap();
        let b = journal_b
            .checkpoint(cuptr(carry_b.as_ptr()), 0, bytes_b.len(), 0, 8, GEN)
            .unwrap();
        (a, b)
    };

    // Speculatively overwrite both layers' carry in place.
    let dirty_a: Vec<u8> = vec![0.9f32; 64]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let dirty_b: Vec<u8> = vec![7.0f32; 96]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    // SAFETY: sizes match.
    unsafe {
        ep.runtime()
            .htod(&dirty_a, cuptr(carry_a.as_ptr()))
            .unwrap();
        ep.runtime()
            .htod(&dirty_b, cuptr(carry_b.as_ptr()))
            .unwrap();
    }

    // Composite rollback: reject the draft on every layer.
    // SAFETY: same live buffers as each checkpoint.
    unsafe {
        journal_a
            .restore_prefix(&ck_a, 8, GEN, cuptr(carry_a.as_ptr()), 0, None)
            .unwrap();
        journal_b
            .restore_prefix(&ck_b, 8, GEN, cuptr(carry_b.as_ptr()), 0, None)
            .unwrap();
    }

    assert_eq!(download_device(&ep, &carry_a, bytes_a.len()), bytes_a);
    assert_eq!(download_device(&ep, &carry_b, bytes_b.len()), bytes_b);

    // A generation mismatch must be rejected (identity validation, D6).
    // SAFETY: live buffer.
    let mismatch =
        unsafe { journal_a.restore_prefix(&ck_a, 8, GEN + 1, cuptr(carry_a.as_ptr()), 0, None) };
    assert!(
        mismatch.is_err(),
        "restore must reject a checkpoint from a different generation"
    );

    ep.deallocate(carry_a).unwrap();
    ep.deallocate(carry_b).unwrap();
}

/// B7 — full MTP composite decode smoke. Requires external Mobius/MTP model
/// artifacts (a real target + MTP draft head sharing the CSA cache), which are
/// not vendored here, so it is gated behind `#[ignore]`. The non-model composite
/// atomicity is covered by `csa_composite_checkpoint_rolls_back_all_layers`; the
/// MTP-6 composite-with-target-state path lands with the MTP integration.
#[test]
#[ignore = "requires external Mobius/MTP model artifacts (see decision note roy-csa-b7.md)"]
fn mtp_composite_decode_smoke() {
    // Intentionally empty: the model-dependent MTP smoke is scoped out of B7 core
    // (external artifacts unavailable). See the CSA cursor+carry rollback tests
    // for the in-scope correctness proof.
}
