#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args,
    clippy::type_complexity,
    clippy::drop_non_drop
)]
//! GPU parity tests for `pkg.nxrt::DsaIndexSelect` v1.
//!
//! Each case builds a tiny single-node model with the `onnx-runtime-ir` graph
//! API, runs it through both the CPU reference kernel (the authoritative
//! numerical oracle in `onnx-runtime-ep-cpu`) and the device-resident CUDA
//! kernel, and asserts the produced `selected_indices` are **bit-identical**.
//!
//! Unlike the softmax attention output of `IndexShare`, this op emits exact
//! integer indices, so parity is byte-exact (not tolerance-based): the CPU
//! oracle and the CUDA kernel both widen f16/bf16 storage to f32 losslessly,
//! compute the head-weighted ReLU'd scaled scores in the same operand order, and
//! break score ties by the same `f32::total_cmp` order. The tests cover tiny and
//! real GLM-5.2 indexer dimensions (`H=2`, `D=8`, `top_k=4`), first-token /
//! prefill / decode geometries, block-boundary top-k widths, explicit tie and
//! `-inf` mask handling, `-1` sentinel padding, query-dependent route changes,
//! sequential request isolation, CUDA-graph capture + ≥3 replays with no
//! fallback, eager equivalence, and typed rejection of the f32-only
//! `attention_bias` contract.
//!
//! CPU-only CI reports these tests as ignored unless `gpu-tests` is enabled.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMetadata,
    TensorMut, TensorView, WorkspaceView,
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
}

/// Output tensor spec (dtype + shape) for one node output slot.
#[derive(Clone)]
struct OutputSpec {
    dtype: DataType,
    shape: Vec<usize>,
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

/// The DsaIndexSelect geometry.
#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    q_seq: usize,
    heads: usize,
    head_dim: usize,
    key_seq: usize,
    top_k: usize,
    scale: f32,
    weights_scale: Option<f32>,
}

/// Build a single `DsaIndexSelect` node model from the positional input list
/// `[query, key, weights, attention_bias]`. All four inputs are required.
fn build_node(inputs: &[Option<HostTensor>], case: Case) -> (Graph, NodeId, Vec<OutputSpec>) {
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

    let out_shape = vec![case.batch, 1, case.q_seq, case.top_k];
    let output = graph.create_named_value(
        "selected_indices",
        DataType::Int64,
        static_shape(out_shape.iter().copied()),
    );
    let output_specs = vec![OutputSpec {
        dtype: DataType::Int64,
        shape: out_shape,
    }];

    let mut node = Node::new(NodeId(0), "DsaIndexSelect", node_inputs, vec![output]);
    node.domain = DOMAIN.into();
    node.attributes
        .insert("top_k".into(), Attribute::Int(case.top_k as i64));
    node.attributes
        .insert("scale".into(), Attribute::Float(case.scale));
    if let Some(ws) = case.weights_scale {
        node.attributes
            .insert("weights_scale".into(), Attribute::Float(ws));
    }
    let id = graph.insert_node(node);
    graph.add_output(output);
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

fn workspace_view_for(
    kernel: &dyn onnx_runtime_ep_api::Kernel,
    input_views: &[TensorView],
    buffer: &mut Option<DeviceBuffer>,
    ep: &CudaExecutionProvider,
) -> onnx_runtime_ep_api::Result<Option<WorkspaceView>> {
    let metadata = input_views
        .iter()
        .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
        .collect::<Vec<_>>();
    let requirement = kernel.workspace_requirement(&metadata)?;
    if requirement.bytes == 0 {
        return Ok(None);
    }
    let bytes = usize::try_from(requirement.bytes).map_err(|_| {
        onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "test workspace requirement {} does not fit usize",
            requirement.bytes
        ))
    })?;
    if buffer
        .as_ref()
        .is_none_or(|buffer| buffer.len() < bytes || buffer.alignment() < requirement.alignment)
    {
        if let Some(old) = buffer.take() {
            ep.deallocate(old)?;
        }
        *buffer = Some(ep.allocate(bytes.max(1), requirement.alignment)?);
    }
    Ok(buffer
        .as_mut()
        .map(|buffer| WorkspaceView::new(DevicePtrMut(buffer.as_mut_ptr()), bytes)))
}

fn upload_inputs(
    ep: &CudaExecutionProvider,
    inputs: &[Option<HostTensor>],
) -> onnx_runtime_ep_api::Result<Vec<Option<DeviceBuffer>>> {
    let runtime = ep.runtime();
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
    Ok(buffers)
}

fn input_views<'a>(
    inputs: &'a [Option<HostTensor>],
    buffers: &'a [Option<DeviceBuffer>],
    strides: &'a [Option<Vec<i64>>],
    ep: &CudaExecutionProvider,
) -> Vec<TensorView<'a>> {
    inputs
        .iter()
        .zip(buffers.iter().zip(strides))
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
        .collect()
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

    let buffers = upload_inputs(ep, inputs)?;
    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let views = input_views(inputs, &buffers, &strides, ep);

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
    let mut workspace_buffer = None;
    let workspace = workspace_view_for(kernel.as_ref(), &views, &mut workspace_buffer, ep)?;
    let result = kernel.execute_with_workspace(&views, &mut out_views, workspace);
    drop(out_views);
    drop(views);

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
    if let Some(buffer) = workspace_buffer {
        ep.deallocate(buffer)?;
    }
    result.map(|()| outputs)
}

/// Capture one DsaIndexSelect `execute` into a CUDA graph and replay it
/// `replays` times, returning the bytes produced by each replay. The output is
/// zeroed before every replay so the returned bytes come solely from that graph
/// launch. Warmup compiles the NVRTC kernel and sizes the persistent workspace;
/// the captured `execute` performs no host staging, per-call alloc/free, or
/// stream sync while recording.
fn run_gpu_capture_replay(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[Option<HostTensor>],
    output_specs: &[OutputSpec],
    replays: usize,
) -> onnx_runtime_ep_api::Result<Vec<Vec<Vec<u8>>>> {
    use cudarc::driver::sys::{
        CUgraph, CUgraphExec, CUstreamCaptureMode, cuGraphDestroy, cuGraphExecDestroy,
        cuGraphInstantiateWithFlags, cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture,
    };

    let model = Model::new(graph);
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes(inputs), 1)?;
    let runtime = ep.runtime();

    let buffers = upload_inputs(ep, inputs)?;
    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let views = input_views(inputs, &buffers, &strides, ep);

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
    let mut workspace_buffer = None;
    let workspace = workspace_view_for(kernel.as_ref(), &views, &mut workspace_buffer, ep)?;

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
    let zero_outputs = |out_buffers: &[DeviceBuffer]| -> onnx_runtime_ep_api::Result<()> {
        for (buffer, len) in out_buffers.iter().zip(&out_lens) {
            if *len > 0 {
                let zeros = vec![0u8; *len];
                // SAFETY: destination exactly covers the output allocation.
                unsafe { runtime.htod(&zeros, cuptr(buffer.as_ptr()))? };
            }
        }
        runtime.synchronize()
    };

    // A fresh generation starts un-poisoned.
    runtime.reset_capture_error()?;

    // Warmup: an eager execute compiles/caches the NVRTC kernel and sizes the
    // persistent scratch before capture. Only after this does the kernel
    // advertise capture eligibility.
    {
        let mut out_views = make_out_views(&mut out_buffers);
        kernel.execute_with_workspace(&views, &mut out_views, workspace)?;
    }
    runtime.synchronize()?;
    assert!(
        kernel.cuda_graph_compatible(),
        "DsaIndexSelect must advertise CUDA-graph capture eligibility after warmup"
    );

    zero_outputs(&out_buffers)?;

    let stream = runtime.stream_ptr();
    let mut graph_handle: CUgraph = std::ptr::null_mut();
    let mut graph_exec: CUgraphExec = std::ptr::null_mut();

    let captured = (|| -> onnx_runtime_ep_api::Result<Vec<Vec<Vec<u8>>>> {
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
        let record = kernel.execute_with_workspace(&views, &mut out_views, workspace);
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

        let mut replay_outputs = Vec::with_capacity(replays);
        for _ in 0..replays {
            zero_outputs(&out_buffers)?;
            // SAFETY: `graph_exec` is instantiated; `stream` is the EP stream.
            unsafe { cuGraphLaunch(graph_exec, stream) }
                .result()
                .map_err(|error| {
                    onnx_runtime_ep_api::EpError::KernelFailed(format!("cuGraphLaunch: {error:?}"))
                })?;
            runtime.synchronize()?;
            let mut this = Vec::new();
            for (buffer, len) in out_buffers.iter().zip(&out_lens) {
                let mut host = vec![0u8; *len];
                if *len > 0 {
                    // SAFETY: destination exactly covers the output allocation.
                    unsafe { runtime.dtoh(&mut host, cuptr(buffer.as_ptr()))? };
                }
                this.push(host);
            }
            replay_outputs.push(this);
        }
        Ok(replay_outputs)
    })();

    if !graph_exec.is_null() {
        // SAFETY: `graph_exec` was instantiated above and is destroyed once.
        let _ = unsafe { cuGraphExecDestroy(graph_exec) }.result();
    }
    if !graph_handle.is_null() {
        // SAFETY: `graph_handle` was captured above and is destroyed once.
        let _ = unsafe { cuGraphDestroy(graph_handle) }.result();
    }
    drop(views);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    for buffer in out_buffers.drain(..) {
        ep.deallocate(buffer)?;
    }
    if let Some(buffer) = workspace_buffer {
        ep.deallocate(buffer)?;
    }
    captured
}

fn as_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_indices_bit_exact(gpu: &[u8], cpu: &[u8], what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length mismatch");
    assert_eq!(
        as_i64(gpu),
        as_i64(cpu),
        "{what}: selected_indices must be bit-identical (gpu vs cpu oracle)"
    );
}

/// Run CPU + GPU and assert bit-exact indices; returns the (CPU) indices.
fn assert_parity(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[Option<HostTensor>],
    specs: &[OutputSpec],
) -> Vec<i64> {
    let cpu = run_cpu(graph, node, inputs, specs).expect("CPU DsaIndexSelect kernel");
    let gpu = run_gpu(ep, graph, node, inputs, specs).expect("CUDA DsaIndexSelect kernel");
    assert_indices_bit_exact(&gpu[0], &cpu[0], "selected_indices");
    as_i64(&cpu[0])
}

/// Deterministic value generator (LCG) in `[-1, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    }
}

fn make(count: usize, rng: &mut impl FnMut() -> f32) -> Vec<f32> {
    (0..count).map(|_| rng()).collect()
}

/// Causal bias `[batch, 1, q_seq, key_seq]`: position `t` is allowed for query
/// `s` iff `t <= s + (key_seq - q_seq)` (queries are the trailing `q_seq` of a
/// `key_seq`-length cache). Masked entries are the `-inf` causal fill.
fn causal_bias(batch: usize, q_seq: usize, key_seq: usize) -> HostTensor {
    let offset = key_seq - q_seq;
    let mut values = vec![0.0f32; batch * q_seq * key_seq];
    for b in 0..batch {
        for s in 0..q_seq {
            for t in 0..key_seq {
                let idx = (b * q_seq + s) * key_seq + t;
                values[idx] = if t <= s + offset {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
    }
    HostTensor::f32(&[batch, 1, q_seq, key_seq], &values)
}

fn glm_case(q_seq: usize, key_seq: usize) -> Case {
    Case {
        batch: 1,
        q_seq,
        heads: 2,
        head_dim: 8,
        key_seq,
        top_k: 4,
        scale: (8.0f32).powf(-0.5),
        weights_scale: Some((2.0f32).powf(-0.5)),
    }
}

fn glm_inputs(case: Case, seed: u64) -> [Option<HostTensor>; 4] {
    let mut rng = lcg(seed);
    let q = HostTensor::f32(
        &[case.batch, case.q_seq, case.heads, case.head_dim],
        &make(
            case.batch * case.q_seq * case.heads * case.head_dim,
            &mut rng,
        ),
    );
    let k = HostTensor::f32(
        &[case.batch, case.key_seq, case.head_dim],
        &make(case.batch * case.key_seq * case.head_dim, &mut rng),
    );
    let w = HostTensor::f32(
        &[case.batch, case.q_seq, case.heads],
        &make(case.batch * case.q_seq * case.heads, &mut rng),
    );
    let bias = causal_bias(case.batch, case.q_seq, case.key_seq);
    [Some(q), Some(k), Some(w), Some(bias)]
}

// ---------------------------------------------------------------------------

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn tiny_f32_selection_matches_cpu() {
    let ep = require_cuda();
    let case = Case {
        batch: 1,
        q_seq: 2,
        heads: 2,
        head_dim: 3,
        key_seq: 5,
        top_k: 3,
        scale: 0.5,
        weights_scale: None,
    };
    let mut rng = lcg(0xA1B2_C3D4);
    let q = HostTensor::f32(&[1, 2, 2, 3], &make(2 * 2 * 3, &mut rng));
    let k = HostTensor::f32(&[1, 5, 3], &make(5 * 3, &mut rng));
    let w = HostTensor::f32(&[1, 2, 2], &make(2 * 2, &mut rng));
    let bias = HostTensor::f32(&[1, 1, 2, 5], &[0.0; 10]);
    let inputs = [Some(q), Some(k), Some(w), Some(bias)];
    let (graph, node, specs) = build_node(&inputs, case);
    let indices = assert_parity(&ep, &graph, node, &inputs, &specs);
    // top_k=3 of 5 allowed positions, sorted ascending, no padding.
    for row in indices.chunks(3) {
        assert!(
            row.iter().all(|&i| (0..5).contains(&i)),
            "in-range: {row:?}"
        );
        assert!(row.windows(2).all(|w| w[0] < w[1]), "ascending: {row:?}");
    }
    eprintln!("tiny f32 DsaIndexSelect: bit-identical to CPU oracle {indices:?}");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn real_glm_dims_first_token_prefill_decode_match_cpu() {
    let ep = require_cuda();
    // First token (S=T=1), prefill (S=T=6), and a decode step (S=1, T=10).
    for (label, q_seq, key_seq, seed) in [
        ("first-token", 1usize, 1usize, 0x11u64),
        ("prefill", 6, 6, 0x22),
        ("decode", 1, 10, 0x33),
    ] {
        let case = glm_case(q_seq, key_seq);
        let inputs = glm_inputs(case, seed);
        let (graph, node, specs) = build_node(&inputs, case);
        let indices = assert_parity(&ep, &graph, node, &inputs, &specs);
        eprintln!("GLM {label} (S={q_seq},T={key_seq}) DsaIndexSelect bit-parity: {indices:?}");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn long_decode_16_steps_match_cpu() {
    let ep = require_cuda();
    // A ≥16-length decode context exercises rows wider than a warp of keys.
    let case = glm_case(1, 20);
    let inputs = glm_inputs(case, 0x44);
    let (graph, node, specs) = build_node(&inputs, case);
    assert_parity(&ep, &graph, node, &inputs, &specs);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f16_and_bf16_storage_match_cpu() {
    let ep = require_cuda();
    let case = glm_case(4, 12);
    let mut rng = lcg(0x55AA);
    let qv = make(case.q_seq * case.heads * case.head_dim, &mut rng);
    let kv = make(case.key_seq * case.head_dim, &mut rng);
    let wv = make(case.q_seq * case.heads, &mut rng);
    let bias = causal_bias(1, case.q_seq, case.key_seq);
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let (q, k, w) = match dtype {
            DataType::Float16 => (
                HostTensor::f16(&[1, case.q_seq, case.heads, case.head_dim], &qv),
                HostTensor::f16(&[1, case.key_seq, case.head_dim], &kv),
                HostTensor::f16(&[1, case.q_seq, case.heads], &wv),
            ),
            _ => (
                HostTensor::bf16(&[1, case.q_seq, case.heads, case.head_dim], &qv),
                HostTensor::bf16(&[1, case.key_seq, case.head_dim], &kv),
                HostTensor::bf16(&[1, case.q_seq, case.heads], &wv),
            ),
        };
        let inputs = [Some(q), Some(k), Some(w), Some(bias.clone())];
        let (graph, node, specs) = build_node(&inputs, case);
        assert_parity(&ep, &graph, node, &inputs, &specs);
        eprintln!("{dtype:?} storage DsaIndexSelect: bit-identical indices to CPU oracle");
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn ties_select_lowest_indices_and_match_cpu() {
    let ep = require_cuda();
    // All-ones query/key/weights ⇒ every key scores identically ⇒ the tie break
    // (lower index wins) must select positions [0,1,2,3] ascending.
    let case = Case {
        batch: 1,
        q_seq: 1,
        heads: 2,
        head_dim: 4,
        key_seq: 8,
        top_k: 4,
        scale: 0.25,
        weights_scale: Some(0.5),
    };
    let q = HostTensor::f32(&[1, 1, 2, 4], &[1.0; 8]);
    let k = HostTensor::f32(&[1, 8, 4], &[1.0; 32]);
    let w = HostTensor::f32(&[1, 1, 2], &[1.0; 2]);
    let bias = HostTensor::f32(&[1, 1, 1, 8], &[0.0; 8]);
    let inputs = [Some(q), Some(k), Some(w), Some(bias)];
    let (graph, node, specs) = build_node(&inputs, case);
    let indices = assert_parity(&ep, &graph, node, &inputs, &specs);
    assert_eq!(
        indices,
        vec![0, 1, 2, 3],
        "tie break must pick lowest indices"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn masks_and_sentinel_padding_match_cpu() {
    let ep = require_cuda();
    // Row allows only positions {1, 3} (rest -inf masked); top_k=4 ⇒ two indices
    // then two -1 pads. The masked positions must never be selected.
    let case = Case {
        batch: 1,
        q_seq: 1,
        heads: 1,
        head_dim: 2,
        key_seq: 5,
        top_k: 4,
        scale: 1.0,
        weights_scale: None,
    };
    let q = HostTensor::f32(&[1, 1, 1, 2], &[1.0, 1.0]);
    let k = HostTensor::f32(
        &[1, 5, 2],
        &[9.0, 9.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 7.0, 7.0],
    );
    let w = HostTensor::f32(&[1, 1, 1], &[1.0]);
    let n = f32::NEG_INFINITY;
    let bias = HostTensor::f32(&[1, 1, 1, 5], &[n, 0.0, n, 0.0, n]);
    let inputs = [Some(q), Some(k), Some(w), Some(bias)];
    let (graph, node, specs) = build_node(&inputs, case);
    let indices = assert_parity(&ep, &graph, node, &inputs, &specs);
    assert_eq!(
        indices,
        vec![1, 3, -1, -1],
        "only unmasked positions, -1 padded"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fully_masked_row_is_all_negative_one() {
    let ep = require_cuda();
    let case = Case {
        batch: 1,
        q_seq: 1,
        heads: 1,
        head_dim: 2,
        key_seq: 4,
        top_k: 3,
        scale: 1.0,
        weights_scale: None,
    };
    let q = HostTensor::f32(&[1, 1, 1, 2], &[0.5, -0.5]);
    let k = HostTensor::f32(&[1, 4, 2], &make(8, &mut lcg(7)));
    let w = HostTensor::f32(&[1, 1, 1], &[1.0]);
    let bias = HostTensor::f32(&[1, 1, 1, 4], &[f32::NEG_INFINITY; 4]);
    let inputs = [Some(q), Some(k), Some(w), Some(bias)];
    let (graph, node, specs) = build_node(&inputs, case);
    let indices = assert_parity(&ep, &graph, node, &inputs, &specs);
    assert_eq!(indices, vec![-1, -1, -1], "fully masked ⇒ all -1");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn query_dependent_routes_change_and_match_cpu() {
    let ep = require_cuda();
    let case = glm_case(1, 12);
    let a = glm_inputs(case, 0x1000);
    let b = glm_inputs(case, 0x2000);
    let (ga, na, sa) = build_node(&a, case);
    let (gb, nb, sb) = build_node(&b, case);
    let ia = assert_parity(&ep, &ga, na, &a, &sa);
    let ib = assert_parity(&ep, &gb, nb, &b, &sb);
    assert_ne!(
        ia, ib,
        "different queries must (generally) select different indices: {ia:?} vs {ib:?}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn sequential_requests_are_isolated() {
    let ep = require_cuda();
    // Two different-shaped requests back to back must each match their own CPU
    // oracle — the reused persistent workspace must not leak state.
    let case1 = glm_case(3, 7);
    let case2 = glm_case(5, 5);
    let in1 = glm_inputs(case1, 0xDEAD);
    let in2 = glm_inputs(case2, 0xBEEF);
    let (g1, n1, s1) = build_node(&in1, case1);
    let (g2, n2, s2) = build_node(&in2, case2);
    assert_parity(&ep, &g1, n1, &in1, &s1);
    assert_parity(&ep, &g2, n2, &in2, &s2);
    // Re-run the first: still correct after the second touched the pool.
    assert_parity(&ep, &g1, n1, &in1, &s1);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_replay_is_byte_identical_and_no_fallback() {
    let ep = require_cuda();
    let case = glm_case(4, 12);
    // f16 storage to mirror the real decode dtype.
    let mut rng = lcg(0xCAFE);
    let q = HostTensor::f16(
        &[1, case.q_seq, case.heads, case.head_dim],
        &make(case.q_seq * case.heads * case.head_dim, &mut rng),
    );
    let k = HostTensor::f16(
        &[1, case.key_seq, case.head_dim],
        &make(case.key_seq * case.head_dim, &mut rng),
    );
    let w = HostTensor::f16(
        &[1, case.q_seq, case.heads],
        &make(case.q_seq * case.heads, &mut rng),
    );
    let bias = causal_bias(1, case.q_seq, case.key_seq);
    let inputs = [Some(q), Some(k), Some(w), Some(bias)];
    let (graph, node, specs) = build_node(&inputs, case);

    let eager = run_gpu(&ep, &graph, node, &inputs, &specs).expect("eager DsaIndexSelect");
    let replays = run_gpu_capture_replay(&ep, &graph, node, &inputs, &specs, 3)
        .expect("captured DsaIndexSelect");
    assert_eq!(replays.len(), 3, "expected 3 replays");
    for (r, replay) in replays.iter().enumerate() {
        assert_indices_bit_exact(&replay[0], &eager[0], &format!("replay {r} vs eager"));
    }
    assert_eq!(
        ep.runtime()
            .check_capture_error()
            .expect("read capture-error latch"),
        0,
        "valid DsaIndexSelect capture must not poison the capture-error latch (no fallback)"
    );
    // Bit-parity against the independent CPU oracle too.
    let cpu = run_cpu(&graph, node, &inputs, &specs).expect("CPU oracle");
    assert_indices_bit_exact(&eager[0], &cpu[0], "eager vs cpu oracle");
    eprintln!("captured DsaIndexSelect: 3 replays byte-identical to eager, latch clear");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn eager_equals_capture_warmup_output() {
    // The warmup eager pass inside the capture harness must itself produce the
    // same indices as a standalone eager run (no capture-only divergence).
    let ep = require_cuda();
    let case = glm_case(2, 9);
    let inputs = glm_inputs(case, 0x9999);
    let (graph, node, specs) = build_node(&inputs, case);
    let eager = run_gpu(&ep, &graph, node, &inputs, &specs).expect("eager");
    let replays = run_gpu_capture_replay(&ep, &graph, node, &inputs, &specs, 1).expect("capture");
    assert_indices_bit_exact(&replays[0][0], &eager[0], "capture warmup vs eager");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn bias_must_be_f32_is_rejected() {
    let ep = require_cuda();
    let case = glm_case(2, 6);
    let mut rng = lcg(0x1357);
    let q = HostTensor::f16(
        &[1, case.q_seq, case.heads, case.head_dim],
        &make(case.q_seq * case.heads * case.head_dim, &mut rng),
    );
    let k = HostTensor::f16(
        &[1, case.key_seq, case.head_dim],
        &make(case.key_seq * case.head_dim, &mut rng),
    );
    let w = HostTensor::f16(
        &[1, case.q_seq, case.heads],
        &make(case.q_seq * case.heads, &mut rng),
    );
    // f16 bias violates the strict f32-only mask contract.
    let bias_vals = vec![0.0f32; case.q_seq * case.key_seq];
    let bias = HostTensor::f16(&[1, 1, case.q_seq, case.key_seq], &bias_vals);
    let inputs = [Some(q), Some(k), Some(w), Some(bias)];
    let (graph, node, specs) = build_node(&inputs, case);
    let result = run_gpu(&ep, &graph, node, &inputs, &specs);
    let err = result.expect_err("f16 attention_bias must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("attention_bias") && msg.contains("Float32"),
        "reject reason must cite the f32-only bias contract, got: {msg}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn unknown_attribute_is_rejected() {
    let ep = require_cuda();
    let case = glm_case(1, 4);
    let inputs = glm_inputs(case, 0x2468);
    let (mut graph, node, _specs) = build_node(&inputs, case);
    graph
        .node_mut(node)
        .attributes
        .insert("bogus".into(), Attribute::Int(1));
    let model = Model::new(&graph);
    let result = ep.get_kernel(model.graph.node(node), &concrete_shapes(&inputs), 1);
    let err = match result {
        Ok(_) => panic!("unknown attribute must be rejected"),
        Err(err) => err,
    };
    assert!(
        format!("{err}").contains("frozen v1 ABI"),
        "reject reason must cite the frozen v1 ABI, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Capability-gate (`supports_op`) tests.
//
// These assert the CUDA provider's `supports_op` (the partitioner's capability
// query, the CUDA analogue of ORT `GetCapability`) actually consults the typed
// validator: a valid node is CLAIMED and every malformed node is DECLINED **at
// capability time**, so a bad node is never claimed-then-hard-failed at execute
// with no fallback. The registered kernel factory is *not* enough on its own —
// the capability query must be wired to the validator, and these tests are the
// regression guard that it is (and stays) so.
// ---------------------------------------------------------------------------

/// Model input shapes and dtypes as the partitioner would hand them to
/// `supports_op` (positional, absent slots reported as empty/Undefined).
fn shapes_and_dtypes(
    inputs: &[Option<HostTensor>],
) -> (Vec<onnx_runtime_ir::Shape>, Vec<DataType>) {
    let shapes = inputs
        .iter()
        .map(|slot| match slot {
            Some(t) => static_shape(t.shape.iter().copied()),
            None => static_shape([0usize; 0]),
        })
        .collect();
    let dtypes = inputs
        .iter()
        .map(|slot| slot.as_ref().map_or(DataType::Undefined, |t| t.dtype))
        .collect();
    (shapes, dtypes)
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn supports_op_claims_valid_and_low_precision_storage() {
    let ep = require_cuda();
    let case = glm_case(4, 4);
    let inputs = glm_inputs(case, 0x1357);
    let (graph, node, _specs) = build_node(&inputs, case);
    let model = Model::new(&graph);
    let (shapes, dtypes) = shapes_and_dtypes(&inputs);

    // Valid all-f32 node is claimed at capability time.
    assert!(
        matches!(
            ep.supports_op(model.graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Supported { .. }
        ),
        "a valid DsaIndexSelect node must be claimed by supports_op"
    );

    // f16 / bf16 storage (bias stays f32) is claimed: the CUDA validator projects
    // the query/key/weights trio to f32, so it supports the low-precision storage
    // the f32-only CPU oracle would decline. The strict f32-only bias is retained.
    for storage in [DataType::Float16, DataType::BFloat16] {
        let low = vec![storage, storage, storage, DataType::Float32];
        assert!(
            matches!(
                ep.supports_op(model.graph.node(node), 1, &shapes, &low, &[]),
                KernelMatch::Supported { .. }
            ),
            "{storage:?} query/key/weights storage (f32 bias) must be claimed by supports_op"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn supports_op_declines_malformed_nodes_at_capability_time() {
    let ep = require_cuda();
    let case = glm_case(4, 4);
    let inputs = glm_inputs(case, 0x2461);
    let (graph, node, _specs) = build_node(&inputs, case);
    let model = Model::new(&graph);
    let (shapes, dtypes) = shapes_and_dtypes(&inputs);

    let is_declined = |shapes: &[onnx_runtime_ir::Shape], dtypes: &[DataType]| {
        matches!(
            ep.supports_op(model.graph.node(node), 1, shapes, dtypes, &[]),
            KernelMatch::Unsupported { .. }
        )
    };

    // Non-f32 attention_bias — the exact contract that must be declined, not
    // claimed-then-hard-failed (an f16 finfo.min would misread a masked slot).
    let mut bias_f16 = dtypes.clone();
    bias_f16[3] = DataType::Float16;
    assert!(
        is_declined(&shapes, &bias_f16),
        "non-f32 attention_bias must be declined at capability time"
    );

    // Non-floating query dtype.
    let mut query_int = dtypes.clone();
    query_int[0] = DataType::Int32;
    assert!(
        is_declined(&shapes, &query_int),
        "non-floating query must be declined at capability time"
    );

    // Mixed query/key storage dtype (trio must be homogeneous).
    let mut mixed = dtypes.clone();
    mixed[1] = DataType::Float16;
    assert!(
        is_declined(&shapes, &mixed),
        "mismatched key storage dtype must be declined at capability time"
    );

    // Shape conflict: bias head axis must be 1 (head-broadcast).
    let mut bias_head2 = shapes.clone();
    bias_head2[3] = static_shape([case.batch, 2, case.q_seq, case.key_seq]);
    assert!(
        is_declined(&bias_head2, &dtypes),
        "attention_bias head axis != 1 must be declined at capability time"
    );

    // Unknown attribute outside the frozen v1 ABI.
    let mut bogus = graph.clone();
    bogus
        .node_mut(node)
        .attributes
        .insert("bogus".into(), Attribute::Int(1));
    let bogus_model = Model::new(&bogus);
    assert!(
        matches!(
            ep.supports_op(bogus_model.graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "an unknown attribute must be declined at capability time"
    );
}

// ---------------------------------------------------------------------------
// Measurement probe (not a correctness test).
//
// Reports the DsaIndexSelect CUDA kernel's per-launch **device time** (CUDA
// events bracketing a batch of captured-graph launches) and **host enqueue**
// cost (wall-clock around the same async launches) on an idle A100, after a
// clock ramp, with n>=3 repeats reported as median + spread. It prints an
// nvidia-smi clock/power witness before and after and a first-vs-last drift so
// a reader can confirm the device held still (cuda-perf-measurement Trap 5).
//
// This is a perf probe: it is `#[ignore]`d and only runs when invoked
// explicitly. It makes **no** full-size tok/s claim; it reports per-launch
// microseconds and the fixed scratch VRAM for tiny and real GLM indexer
// dimensions only. onnx-genai-kv remains the sole page authority — this op
// allocates no pages, only a `B*S*T` f32 score buffer plus a `B*S*T` u8 state
// buffer in the executor-owned SessionPersistent workspace.
// ---------------------------------------------------------------------------

struct BenchResult {
    kernel_us_median: f32,
    kernel_us_min: f32,
    kernel_us_max: f32,
    enqueue_us_median: f32,
    workspace_bytes: u64,
}

fn median_min_max(samples: &mut [f32]) -> (f32, f32, f32) {
    samples.sort_by(|a, b| a.total_cmp(b));
    let median = samples[samples.len() / 2];
    (median, samples[0], samples[samples.len() - 1])
}

fn gpu_clock_witness(phase: &str) {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,clocks.sm,power.draw,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output();
    match out {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            // Only the first line is our pinned device under CUDA_VISIBLE_DEVICES.
            let first = text.lines().next().unwrap_or("").trim();
            println!("[clock witness: {phase}] gpu0 -> {first}");
        }
        _ => println!("[clock witness: {phase}] nvidia-smi unavailable"),
    }
}

/// Capture the single-node DsaIndexSelect graph, ramp the device for
/// `ramp_secs`, then time `repeats` batches of `batch` captured-graph launches.
fn bench_config(
    ep: &CudaExecutionProvider,
    label: &str,
    case: Case,
    seed: u64,
    ramp_secs: f32,
    batch: usize,
    repeats: usize,
) -> BenchResult {
    use cudarc::driver::result::event;
    use cudarc::driver::sys::{
        CUevent_flags, CUgraph, CUgraphExec, CUstreamCaptureMode, cuGraphDestroy,
        cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch, cuStreamBeginCapture_v2,
        cuStreamEndCapture,
    };
    use std::time::Instant;

    let inputs = glm_inputs(case, seed);
    let (graph, node, specs) = build_node(&inputs, case);
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(node), &concrete_shapes(&inputs), 1)
        .expect("CUDA DsaIndexSelect kernel");
    let runtime = ep.runtime();

    let buffers = upload_inputs(ep, &inputs).expect("upload inputs");
    let strides: Vec<_> = inputs
        .iter()
        .map(|slot| slot.as_ref().map(|t| compute_contiguous_strides(&t.shape)))
        .collect();
    let views = input_views(&inputs, &buffers, &strides, ep);

    let workspace_bytes = {
        let metadata: Vec<_> = views
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
            .collect();
        kernel
            .workspace_requirement(&metadata)
            .expect("workspace requirement")
            .bytes
    };

    let out_strides: Vec<_> = specs
        .iter()
        .map(|spec| compute_contiguous_strides(&spec.shape))
        .collect();
    let out_lens: Vec<usize> = specs
        .iter()
        .map(|spec| spec.shape.iter().product::<usize>() * spec.dtype.byte_size())
        .collect();
    let mut out_buffers: Vec<DeviceBuffer> = out_lens
        .iter()
        .map(|len| ep.allocate((*len).max(1), 256))
        .collect::<onnx_runtime_ep_api::Result<_>>()
        .expect("output buffers");
    let mut workspace_buffer = None;
    let workspace = workspace_view_for(kernel.as_ref(), &views, &mut workspace_buffer, ep)
        .expect("workspace view");

    let make_out_views = |out_buffers: &mut [DeviceBuffer]| -> Vec<TensorMut> {
        out_buffers
            .iter_mut()
            .zip(specs.iter().zip(&out_strides))
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

    runtime.reset_capture_error().expect("reset capture error");
    // Warmup compiles the NVRTC kernel and sizes the persistent workspace.
    {
        let mut out_views = make_out_views(&mut out_buffers);
        kernel
            .execute_with_workspace(&views, &mut out_views, workspace)
            .expect("warmup execute");
    }
    runtime.synchronize().expect("warmup sync");
    assert!(
        kernel.cuda_graph_compatible(),
        "kernel must advertise capture eligibility after warmup"
    );

    let stream = runtime.stream_ptr();
    let mut graph_handle: CUgraph = std::ptr::null_mut();
    let mut graph_exec: CUgraphExec = std::ptr::null_mut();
    // SAFETY: `stream` is the EP's live compute stream; capture is balanced by
    // the matching `cuStreamEndCapture` below.
    unsafe {
        cuStreamBeginCapture_v2(
            stream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .result()
        .expect("begin capture");
    }
    {
        let mut out_views = make_out_views(&mut out_buffers);
        let record = kernel.execute_with_workspace(&views, &mut out_views, workspace);
        // SAFETY: `stream` is capturing; `graph_handle` is a valid out-pointer.
        let end = unsafe { cuStreamEndCapture(stream, &mut graph_handle) }.result();
        record.expect("captured execute");
        end.expect("end capture");
    }
    // SAFETY: `graph_handle` is a freshly captured non-null graph.
    unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph_handle, 0) }
        .result()
        .expect("instantiate graph");

    let launch = |exec: CUgraphExec| {
        // SAFETY: `exec` is instantiated; `stream` is the EP stream.
        unsafe { cuGraphLaunch(exec, stream) }
            .result()
            .expect("graph launch");
    };

    // Clock ramp: continuous graph replay until the wall-clock floor elapses.
    let ramp_start = Instant::now();
    while ramp_start.elapsed().as_secs_f32() < ramp_secs {
        for _ in 0..64 {
            launch(graph_exec);
        }
        runtime.synchronize().expect("ramp sync");
    }

    // Device time: CUDA events bracket a batch of captured-graph launches.
    let start_ev = event::create(CUevent_flags::CU_EVENT_DEFAULT).expect("start event");
    let end_ev = event::create(CUevent_flags::CU_EVENT_DEFAULT).expect("end event");
    let mut kernel_us = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        // SAFETY: events/stream are live for the duration of the timed batch.
        unsafe { event::record(start_ev, stream) }.expect("record start");
        for _ in 0..batch {
            launch(graph_exec);
        }
        // SAFETY: same live stream/events.
        unsafe { event::record(end_ev, stream) }.expect("record end");
        unsafe { event::synchronize(end_ev) }.expect("sync end event");
        let ms = unsafe { event::elapsed(start_ev, end_ev) }.expect("elapsed");
        kernel_us.push(ms * 1000.0 / batch as f32);
    }

    // Host enqueue: wall-clock around a *small* drained batch of async launches
    // so the driver's launch queue never back-pressures — this isolates the
    // pure host submit cost of `cuGraphLaunch` from device throughput. A large
    // batch would instead measure device time, because the host blocks once the
    // queue fills (cuda-perf-measurement Trap 4).
    const ENQUEUE_BATCH: usize = 8;
    let mut enqueue_us = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        runtime.synchronize().expect("drain before enqueue timing");
        let t = Instant::now();
        for _ in 0..ENQUEUE_BATCH {
            launch(graph_exec);
        }
        let per = t.elapsed().as_secs_f64() * 1e6 / ENQUEUE_BATCH as f64;
        runtime.synchronize().expect("drain enqueue batch");
        enqueue_us.push(per as f32);
    }

    // SAFETY: each handle was created above and is destroyed exactly once.
    unsafe {
        let _ = event::destroy(start_ev);
        let _ = event::destroy(end_ev);
        let _ = cuGraphExecDestroy(graph_exec).result();
        let _ = cuGraphDestroy(graph_handle).result();
    }
    drop(views);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer).expect("free input");
    }
    for buffer in out_buffers.drain(..) {
        ep.deallocate(buffer).expect("free output");
    }
    if let Some(buffer) = workspace_buffer {
        ep.deallocate(buffer).expect("free workspace");
    }

    let (kmed, kmin, kmax) = median_min_max(&mut kernel_us);
    let (emed, _, _) = median_min_max(&mut enqueue_us);
    println!(
        "{label:28} | B={} S={:>3} H={} D={} T={:>3} top_k={} | kernel {kmed:7.2} us \
         (min {kmin:.2}, max {kmax:.2}) | host-enqueue {emed:6.3} us | scratch {} B | \
         batch={batch} n={repeats}",
        case.batch,
        case.q_seq,
        case.heads,
        case.head_dim,
        case.key_seq,
        case.top_k,
        workspace_bytes,
    );

    BenchResult {
        kernel_us_median: kmed,
        kernel_us_min: kmin,
        kernel_us_max: kmax,
        enqueue_us_median: emed,
        workspace_bytes,
    }
}

#[ignore = "perf probe; run explicitly: cargo test -p onnx-runtime-ep-cuda --features gpu-tests --test dsa_index_select_gpu -- --ignored --nocapture measure_dsa_index_select"]
#[test]
fn measure_dsa_index_select_kernel_and_host_enqueue() {
    let ep = require_cuda();
    gpu_clock_witness("start");

    // Tiny decode row and real GLM-5.2 indexer dimensions (H=2, D=8, top_k=4)
    // for first-token/prefill and a wide decode-against-cache geometry.
    let tiny = Case {
        batch: 1,
        q_seq: 1,
        heads: 2,
        head_dim: 8,
        key_seq: 16,
        top_k: 4,
        scale: (8.0f32).powf(-0.5),
        weights_scale: Some((2.0f32).powf(-0.5)),
    };
    let glm_prefill = glm_case(64, 64);
    let glm_decode = glm_case(1, 512);

    // 8s clock ramp on the tiny graph before any measurement (Trap 5), result
    // discarded; the device then stays warm for the measured configs.
    let _ = bench_config(&ep, "tiny-decode (ramp)", tiny, 0x51ce_d5a0, 8.0, 512, 7);
    let first = bench_config(&ep, "tiny-decode", tiny, 0x51ce_d5a0, 1.0, 512, 7);
    let _ = bench_config(
        &ep,
        "glm-prefill S=64 T=64",
        glm_prefill,
        0x6c3f_2d11,
        1.0,
        256,
        7,
    );
    let _ = bench_config(
        &ep,
        "glm-decode S=1 T=512",
        glm_decode,
        0x6c3f_2d12,
        1.0,
        512,
        7,
    );
    // Re-measure the first config last to witness drift (Trap 5 discipline).
    let again = bench_config(
        &ep,
        "tiny-decode (drift re-measure)",
        tiny,
        0x51ce_d5a0,
        0.0,
        512,
        7,
    );

    let drift =
        (again.kernel_us_median - first.kernel_us_median).abs() / first.kernel_us_median * 100.0;
    println!(
        "drift(tiny kernel median first->last): {drift:.1}%  (first {:.2} us, last {:.2} us; \
         scratch {} B)",
        first.kernel_us_median, again.kernel_us_median, first.workspace_bytes,
    );
    let _ = (
        first.kernel_us_min,
        first.kernel_us_max,
        again.enqueue_us_median,
    );
    gpu_clock_witness("end");
}
