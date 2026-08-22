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

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::CudaContext;
use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMetadata,
    TensorMut, TensorView, WorkspaceAllocation, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;
use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, MemoryError, Tier};

/// Runtime-exact workspace bytes for one dispatch.
pub fn runtime_workspace_requirement(
    kernel: &dyn Kernel,
    inputs: &[TensorView],
) -> onnx_runtime_ep_api::Result<WorkspaceRequirement> {
    let metadata = inputs
        .iter()
        .map(|input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()))
        .collect::<Vec<_>>();
    kernel.workspace_requirement_for_execution(inputs, &metadata)
}

/// Ensure `workspace` satisfies `requirement`, reusing or replacing it through
/// the provider's governed workspace path.
pub fn prepare_workspace(
    ep: &CudaExecutionProvider,
    requirement: WorkspaceRequirement,
    workspace: &mut Option<WorkspaceAllocation>,
) -> onnx_runtime_ep_api::Result<Option<WorkspaceView>> {
    if requirement.bytes == 0 {
        return Ok(None);
    }
    let required = usize::try_from(requirement.bytes).map_err(|_| {
        onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "test workspace requirement {} does not fit usize",
            requirement.bytes
        ))
    })?;
    let needs_replacement = workspace
        .as_ref()
        .is_none_or(|buffer| buffer.len() < required || buffer.alignment() < requirement.alignment);
    if needs_replacement {
        let old = workspace.take();
        *workspace =
            Some(ep.replace_workspace(old, required, requirement.alignment, requirement.role)?);
    }
    let workspace = workspace
        .as_mut()
        .expect("a non-zero workspace requirement must prepare a buffer");
    Ok(Some(WorkspaceView::new(
        DevicePtrMut(workspace.as_mut_ptr()),
        workspace.len(),
    )))
}

/// Execute one kernel through the same prepared-workspace path the session
/// executor uses.
pub fn execute_kernel(
    ep: &CudaExecutionProvider,
    kernel: &dyn Kernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    workspace: &mut Option<WorkspaceAllocation>,
) -> onnx_runtime_ep_api::Result<()> {
    let requirement = runtime_workspace_requirement(kernel, inputs)?;
    let workspace_view = prepare_workspace(ep, requirement, workspace)?;
    kernel.execute_with_workspace(inputs, outputs, workspace_view)
}

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
    let mut workspace = None;
    execute_kernel(
        ep,
        kernel.as_ref(),
        &input_views,
        &mut output_views,
        &mut workspace,
    )
    .unwrap();

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
    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace).unwrap();
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

/// Construct a raw CUDA context for the current visible device or panic.
pub fn require_context(what: &str) -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "{what} requires the CUDA driver on the visible device, but creating a context failed: {error}"
        ),
    }
}

/// An injected eager `cuMemAlloc` allocator for tests that need to prove the
/// provider routed a workspace through the selected `DeviceAllocator`.
#[derive(Debug)]
pub struct ExternalEagerAllocator {
    context: Arc<CudaContext>,
    device: DeviceKey,
    cumemalloc_calls: AtomicU64,
    frees: AtomicU64,
}

impl ExternalEagerAllocator {
    pub fn new(context: Arc<CudaContext>) -> Self {
        let ordinal = context.ordinal() as u32;
        Self {
            context,
            device: DeviceKey::device(ordinal),
            cumemalloc_calls: AtomicU64::new(0),
            frees: AtomicU64::new(0),
        }
    }

    pub fn cumemalloc_calls(&self) -> u64 {
        self.cumemalloc_calls.load(Ordering::Relaxed)
    }

    pub fn frees(&self) -> u64 {
        self.frees.load(Ordering::Relaxed)
    }
}

impl DeviceAllocator for ExternalEagerAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        if align == 0 || !align.is_power_of_two() || align > 256 {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: "cuMemAlloc guarantees 256-byte alignment and this allocator does not \
                         over-allocate to exceed it",
            });
        }
        self.context
            .bind_to_thread()
            .map_err(|error| MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: format!("could not bind the CUDA context: {error}"),
            })?;
        // SAFETY: a fresh device allocation on the bound context, owned here and
        // freed exactly once in `deallocate`.
        let dptr =
            unsafe { cudarc::driver::result::malloc_sync(bytes.max(1)) }.map_err(|error| {
                MemoryError::AllocationFailed {
                    tier: Tier::Device.name(),
                    requested: bytes as u64,
                    reason: format!("cuMemAlloc refused: {error}"),
                }
            })?;
        NonNull::new(dptr as *mut u8)
            .ok_or(MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: String::from("cuMemAlloc returned a null device pointer"),
            })
            .inspect(|_| {
                self.cumemalloc_calls.fetch_add(1, Ordering::Relaxed);
            })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _bytes: usize, _align: usize) {
        let _ = self.context.bind_to_thread();
        // SAFETY: the pointer came from `allocate` on this allocator and the
        // caller's contract guarantees a single free.
        let _ = unsafe {
            cudarc::driver::result::free_sync(ptr.as_ptr() as cudarc::driver::sys::CUdeviceptr)
        };
        self.frees.fetch_add(1, Ordering::Relaxed);
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

/// Wait until the provider's deferred release queue is idle so allocator-side
/// free counters reflect every enqueued release.
pub fn drain_releases(provider: &CudaExecutionProvider, what: &str) {
    assert!(
        provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "the deferred release queue must drain before {what} is asserted: {:?}",
        provider.deferred_release_stats()
    );
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
