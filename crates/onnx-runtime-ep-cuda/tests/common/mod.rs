//! Shared GPU parity harness for the CUDA EP integration tests.
//!
//! Every helper here builds a single-node ONNX graph, runs it on the real CUDA
//! execution provider, and (where a reference is needed) runs the identical node
//! on the CPU execution provider — the conformance oracle. This is the same
//! `run_cuda` vs `run_cpu` pattern used by the hand-written op suites, factored
//! into one place so the data-driven conformance profile
//! (`cuda_conformance_gpu.rs`) and future suites can reuse it instead of
//! copy-pasting the boilerplate per op.
//!
//! The harness skips cleanly when no CUDA runtime is present (see [`require_cuda`]),
//! so a host without a GPU still passes.

// Integration tests are separate crates; a shared `tests/common` module is
// compiled into every test binary that includes it, so not every helper is used
// by every binary. Allow the unused-across-binaries case rather than sprinkle
// per-item attributes.
#![allow(dead_code)]

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

/// A concrete input tensor: dtype, shape, and contiguous little-endian bytes.
#[derive(Clone)]
pub struct Tensor {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Reinterpret a slice of plain-old-data scalars as raw little-endian bytes.
pub fn raw<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: primitive test values are plain old data with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

/// Build a [`Tensor`] from typed scalar values.
pub fn input<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: raw(values),
    }
}

/// Encode f32 logical values into the storage bytes of a float dtype
/// (`Float32`/`Float16`/`BFloat16`).
pub fn encode_floats(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        other => panic!("unsupported float dtype {other:?}"),
    }
}

/// Build a float [`Tensor`] from f32 logical values, encoded for `dtype`.
pub fn float_input(dtype: DataType, shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: encode_floats(values, dtype),
    }
}

/// Decode the storage bytes of a float dtype back to f32 logical values.
pub fn decode_floats(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        DataType::Float64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        other => panic!("unsupported float dtype {other:?}"),
    }
}

/// Assemble a single-node graph named after `op` in `domain` with the given
/// inputs, outputs and attributes. Returns the graph and the node id so callers
/// can look the node back up on the model.
///
/// `domain` is `""` for the default ONNX domain or e.g. `"com.microsoft"` for
/// contrib ops; the domain is recorded on both the node and `opset_imports` so
/// the EP registries key the lookup correctly.
pub fn build_graph(
    op: &str,
    domain: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    // Always import a default-domain opset so shape inference / claim gates have
    // a base version, and additionally import the op's own (contrib) domain.
    graph
        .opset_imports
        .insert(String::new(), if domain.is_empty() { opset } else { 21 });
    if !domain.is_empty() {
        graph.opset_imports.insert(domain.to_string(), opset);
    }
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let value = graph.create_named_value(
                format!("input_{index}"),
                tensor.dtype,
                static_shape(tensor.shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(index, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{index}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().map(Some).collect(),
        output_values.clone(),
    );
    node.domain = domain.to_string();
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    (graph, node_id)
}

/// Execute a single node on the CUDA EP and return each output's raw bytes.
///
/// Asserts the EP's claim gate (`supports_op`) accepts the node — a partitioner
/// would otherwise fall back to the CPU EP and the parity check would be void.
pub fn run_cuda(
    ep: &CudaExecutionProvider,
    op: &str,
    domain: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let (graph, node_id) = build_graph(op, domain, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let claim_shapes = inputs
        .iter()
        .map(|tensor| static_shape(tensor.shape.iter().copied()))
        .collect::<Vec<_>>();
    let claim_dtypes = inputs.iter().map(|tensor| tensor.dtype).collect::<Vec<_>>();
    let claim = ep.supports_op(
        model.graph.node(node_id),
        opset,
        &claim_shapes,
        &claim_dtypes,
        &[],
    );
    assert!(
        claim.is_supported(),
        "{op} (opset {opset}) must be claimed by the CUDA EP, got: {:?}",
        claim.reason()
    );
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();

    let input_buffers = inputs
        .iter()
        .map(|tensor| {
            let buffer = ep.allocate(tensor.bytes.len().max(1), 256).unwrap();
            if !tensor.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&tensor.bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            buffer
        })
        .collect::<Vec<_>>();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((tensor, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| {
            ep.allocate(dtype.storage_bytes(shape.iter().product()).max(1), 256)
                .unwrap()
        })
        .collect::<Vec<DeviceBuffer>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), buffer), strides)| {
            TensorMut::new(
                DevicePtrMut(buffer.as_mut_ptr()),
                *dtype,
                shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();

    let result = outputs
        .iter()
        .zip(&output_buffers)
        .map(|((dtype, shape), buffer)| {
            let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
            if !bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            bytes
        })
        .collect();
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
    result
}

/// Execute the identical node on the CPU EP (the conformance oracle) and return
/// each output's raw bytes.
pub fn run_cpu(
    op: &str,
    domain: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CpuExecutionProvider::new();
    let (graph, node_id) = build_graph(op, domain, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_strides)
        .map(|(tensor, strides)| {
            TensorView::new(
                DevicePtr(tensor.bytes.as_ptr().cast()),
                tensor.dtype,
                &tensor.shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_bytes = outputs
        .iter()
        .map(|(dtype, shape)| vec![0_u8; dtype.storage_bytes(shape.iter().product())])
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(&output_strides)
        .zip(output_bytes.iter_mut())
        .map(|(((dtype, shape), strides), bytes)| {
            TensorMut::new(
                DevicePtrMut(bytes.as_mut_ptr().cast()),
                *dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();
    drop(output_views);
    output_bytes
}

/// Construct the default CUDA EP or panic. CPU-only runs rely on the `gpu-tests`
/// ignore gate; feature-enabled runs must fail loudly if CUDA is unavailable.
pub fn require_cuda() -> CudaExecutionProvider {
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

/// Assert two f32 value vectors agree element-wise within `tolerance`.
pub fn assert_close(label: &str, dtype: DataType, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label} {dtype:?}: length mismatch"
    );
    for (index, (&got, &want)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "{label} {dtype:?} index {index}: got {got}, expected {want}, tolerance {tolerance}"
        );
    }
}
