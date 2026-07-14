//! The sequential CPU executor (Track D, `docs/ORT2.md` §20, §11.3).
//!
//! Turns a loaded [`Graph`] plus its live [`WeightStore`] into a runnable plan:
//! allocate a device buffer per value, resolve a kernel per node through the
//! execution provider, then walk the topological order binding
//! [`TensorView`]/[`TensorMut`] windows over those buffers and invoking each
//! kernel. It is generic over any [`Graph`] and any [`ExecutionProvider`]; the
//! Phase-1 build wires in the CPU EP only, but nothing here is op- or
//! model-specific.
//!
//! ## Holden's precondition (ep-api safety review #1) — enforced here
//!
//! A [`TensorView`] carries no backing length, so it cannot self-check storage
//! bounds. This executor owns every buffer, so it is the layer that *can*: for
//! **every** input and output view of **every** node it calls
//! [`strided::view_in_bounds`] (or, for sub-byte dtypes, the `storage_bytes`
//! equivalent in [`view_bounds`]) and refuses to dispatch on failure. That
//! check is the sole thing that makes ep-cpu's unchecked pointer derefs sound.

use std::collections::HashMap;
use std::sync::Arc;

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::strided::view_in_bounds;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    as_static_shape, compute_contiguous_strides, DataType, Dim, Graph, Node, NodeId, Shape,
    TensorLayout, ValueId,
};
use onnx_runtime_loader::WeightStore;

use crate::error::{Result, SessionError};
use crate::tensor::{host_bytes, write_host, Tensor};

/// A per-node compiled entry: everything the run loop needs without touching
/// the graph again except to fetch the (attribute-bearing) [`Node`].
#[derive(Debug)]
pub(crate) struct NodePlan {
    pub node_id: NodeId,
    /// Present (non-skipped) input value ids, in positional order.
    pub inputs: Vec<ValueId>,
    /// Output value ids, in positional order.
    pub outputs: Vec<ValueId>,
    /// Concrete static shapes of the present inputs.
    pub input_shapes: Vec<Vec<usize>>,
    /// Element types of the present inputs.
    pub input_dtypes: Vec<DataType>,
    /// Concrete static shapes of the outputs.
    pub output_shapes: Vec<Vec<usize>>,
    /// Element types of the outputs.
    pub output_dtypes: Vec<DataType>,
}

/// Cache key for a compiled kernel (§11.1). Keyed by the concrete node and its
/// input shapes: attributes are fixed per node, so this is correct, and the
/// shape component makes it *shape-keyed* — a re-run with the same shapes hits,
/// a different shape misses. A future EP-level key `(op, attrs, shapes)` could
/// additionally dedup structurally identical nodes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct KernelKey {
    node: u32,
    shapes: Vec<Vec<usize>>,
}

/// Observable kernel-cache statistics (§11.1) — enough to prove reuse in tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Distinct compiled entries currently held.
    pub entries: usize,
    /// Lookups served from an existing entry.
    pub hits: u64,
    /// Lookups that compiled a new kernel.
    pub misses: u64,
}

/// Shape-keyed kernel cache (§11.1). Owns the compiled kernels for the session.
#[derive(Default)]
pub(crate) struct KernelCache {
    entries: HashMap<KernelKey, Box<dyn onnx_runtime_ep_api::Kernel>>,
    hits: u64,
    misses: u64,
}

impl KernelCache {
    fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Return the cached kernel for `(node, input_shapes)`, compiling and
    /// inserting it on a miss.
    fn get_or_create(
        &mut self,
        plan: &NodePlan,
        node: &Node,
        ep: &CpuExecutionProvider,
    ) -> Result<&dyn onnx_runtime_ep_api::Kernel> {
        let key = KernelKey {
            node: plan.node_id.0,
            shapes: plan.input_shapes.clone(),
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            let kernel = ep.get_kernel(node, &plan.input_shapes)?;
            self.entries.insert(key.clone(), kernel);
            self.misses += 1;
        }
        Ok(self.entries.get(&key).expect("just inserted").as_ref())
    }
}

/// The compiled, runnable graph: buffers + plan + kernel cache. Owned by the
/// public [`InferenceSession`](crate::InferenceSession).
pub(crate) struct Executor {
    graph: Graph,
    /// Kept alive so external-weight memory maps outlive buffer population.
    _weights: Arc<WeightStore>,
    ep: Arc<CpuExecutionProvider>,
    /// One device buffer per value that needs backing storage.
    buffers: HashMap<ValueId, DeviceBuffer>,
    /// Concrete static shape of every backed value.
    value_shapes: HashMap<ValueId, Vec<usize>>,
    /// Element type of every backed value.
    value_dtypes: HashMap<ValueId, DataType>,
    /// Topologically ordered execution plan.
    plan: Vec<NodePlan>,
    /// name → value id for the graph inputs the caller must supply.
    input_index: HashMap<String, ValueId>,
    /// Value ids the caller must supply at `run` (graph inputs minus initializers).
    required_inputs: Vec<ValueId>,
    cache: KernelCache,
}

/// Resolve a value's concrete static shape, preferring its IR shape and falling
/// back to an initializer's declared dims.
fn resolve_shape(graph: &Graph, vid: ValueId, initializer_dims: Option<&[usize]>) -> Result<Vec<usize>> {
    let value = graph.value(vid);
    if let Some(dims) = as_static_shape(&value.shape) {
        return Ok(dims);
    }
    if let Some(dims) = initializer_dims {
        return Ok(dims.to_vec());
    }
    Err(SessionError::DynamicShape {
        value: value
            .name
            .clone()
            .unwrap_or_else(|| format!("value#{}", vid.0)),
    })
}

/// The `[shape, strides, byte_offset]` storage-bounds gate (Holden's
/// precondition). Uses [`view_in_bounds`] for fixed-width dtypes and a
/// `storage_bytes` check for sub-byte packed dtypes (which have no integral
/// per-element byte size).
fn view_bounds(
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    buffer_len: usize,
) -> Result<()> {
    let esize = dtype.byte_size();
    if esize == 0 {
        // Sub-byte (int4/uint4) or variable-width: size via `storage_bytes`.
        let numel: usize = shape.iter().product();
        let need = byte_offset + dtype.storage_bytes(numel);
        if need > buffer_len {
            return Err(SessionError::from(onnx_runtime_ep_api::EpError::InvalidTensorView {
                reason: format!(
                    "sub-byte view needs {need} bytes but backing allocation is {buffer_len}"
                ),
            }));
        }
        return Ok(());
    }
    view_in_bounds(shape, strides, byte_offset, esize, buffer_len)?;
    Ok(())
}

impl Executor {
    /// Compile a graph + weights into a runnable executor on the CPU EP.
    pub(crate) fn build(
        graph: Graph,
        weights: Arc<WeightStore>,
        ep: Arc<CpuExecutionProvider>,
    ) -> Result<Self> {
        // Topological order up front: also validates the graph is a DAG.
        let order = graph.topological_order()?;

        let mut value_shapes: HashMap<ValueId, Vec<usize>> = HashMap::new();
        let mut value_dtypes: HashMap<ValueId, DataType> = HashMap::new();
        let mut buffers: HashMap<ValueId, DeviceBuffer> = HashMap::new();

        // 1) Initializers: resolve dims, allocate, copy weight bytes in.
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            let bytes = weights.bytes(weight).ok_or_else(|| SessionError::Internal(
                format!("weight bytes unavailable for value#{}", vid.0),
            ))?;
            let mut buf = ep.allocate(bytes.len().max(1), TensorLayout::contiguous().alignment)?;
            write_host(&mut buf, bytes)?;
            value_dtypes.insert(vid, dtype);
            value_shapes.insert(vid, dims);
            buffers.insert(vid, buf);
        }

        // 2) Graph inputs (that are not initializers): allocate empty buffers.
        for &vid in &graph.inputs {
            if buffers.contains_key(&vid) {
                continue;
            }
            let dtype = graph.value(vid).dtype;
            let dims = resolve_shape(&graph, vid, None)?;
            Self::alloc_value(&ep, vid, dtype, &dims, &mut buffers, &mut value_shapes, &mut value_dtypes)?;
        }

        // 3) Node outputs (intermediates + graph outputs): allocate.
        for &nid in &order {
            let node = graph.node(nid);
            for &out in &node.outputs {
                if buffers.contains_key(&out) {
                    continue;
                }
                let dtype = graph.value(out).dtype;
                let dims = resolve_shape(&graph, out, None)?;
                Self::alloc_value(&ep, out, dtype, &dims, &mut buffers, &mut value_shapes, &mut value_dtypes)?;
            }
        }

        // 4) Build the per-node plan and verify EP support for each node.
        let mut plan = Vec::with_capacity(order.len());
        for &nid in &order {
            let node = graph.node(nid);
            let inputs: Vec<ValueId> = node.input_values().collect();
            let outputs: Vec<ValueId> = node.outputs.clone();

            let mut input_shapes = Vec::with_capacity(inputs.len());
            let mut input_dtypes = Vec::with_capacity(inputs.len());
            for &vid in &inputs {
                input_shapes.push(
                    value_shapes
                        .get(&vid)
                        .ok_or_else(|| SessionError::Internal(format!(
                            "input value#{} of node {:?} has no backing buffer",
                            vid.0, node.op_type
                        )))?
                        .clone(),
                );
                input_dtypes.push(value_dtypes[&vid]);
            }
            let output_shapes: Vec<Vec<usize>> =
                outputs.iter().map(|v| value_shapes[v].clone()).collect();
            let output_dtypes: Vec<DataType> =
                outputs.iter().map(|v| value_dtypes[v]).collect();

            // Verify the EP claims this op at these shapes/layouts.
            let shape_dims: Vec<Shape> = input_shapes
                .iter()
                .map(|s| s.iter().map(|&d| Dim::Static(d)).collect())
                .collect();
            let layouts = vec![TensorLayout::contiguous(); inputs.len()];
            if !matches!(ep.supports_op(node, &shape_dims, &layouts), KernelMatch::Supported { .. })
            {
                return Err(SessionError::UnsupportedOp {
                    op_type: node.op_type.clone(),
                });
            }

            plan.push(NodePlan {
                node_id: nid,
                inputs,
                outputs,
                input_shapes,
                input_dtypes,
                output_shapes,
                output_dtypes,
            });
        }

        // 5) name → value id and the set of caller-required inputs.
        let mut input_index = HashMap::new();
        let mut required_inputs = Vec::new();
        for &vid in &graph.inputs {
            if graph.initializers.contains_key(&vid) {
                continue; // pre-filled; not a caller input
            }
            required_inputs.push(vid);
            if let Some(name) = &graph.value(vid).name {
                input_index.insert(name.clone(), vid);
            }
        }

        let mut exec = Self {
            graph,
            _weights: weights,
            ep,
            buffers,
            value_shapes,
            value_dtypes,
            plan,
            input_index,
            required_inputs,
            cache: KernelCache::default(),
        };

        // 6) Resolve (compile) every kernel into the cache — the "compiled
        // plan". Subsequent runs with the same shapes hit this cache.
        exec.compile_all()?;
        Ok(exec)
    }

    fn alloc_value(
        ep: &CpuExecutionProvider,
        vid: ValueId,
        dtype: DataType,
        dims: &[usize],
        buffers: &mut HashMap<ValueId, DeviceBuffer>,
        value_shapes: &mut HashMap<ValueId, Vec<usize>>,
        value_dtypes: &mut HashMap<ValueId, DataType>,
    ) -> Result<()> {
        let numel: usize = dims.iter().product();
        let size = dtype.storage_bytes(numel);
        let buf = ep.allocate(size.max(1), TensorLayout::contiguous().alignment)?;
        buffers.insert(vid, buf);
        value_shapes.insert(vid, dims.to_vec());
        value_dtypes.insert(vid, dtype);
        Ok(())
    }

    /// Populate the kernel cache for the compiled plan (build-time + warmup).
    fn compile_all(&mut self) -> Result<()> {
        let graph = &self.graph;
        let ep = &*self.ep;
        for np in &self.plan {
            let node = graph.node(np.node_id);
            self.cache.get_or_create(np, node, ep)?;
        }
        Ok(())
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Warmup: re-touch the shape-keyed cache for the compiled plan so the first
    /// real `run` sees only cache hits (§11.3). Minimal for Phase-1 static
    /// shapes — the plan's shapes already key the cache.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        self.compile_all()
    }

    /// Sequential topological executor.
    pub(crate) fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        // --- Bind inputs -----------------------------------------------------
        let mut bound: Vec<ValueId> = Vec::new();
        for (name, tensor) in inputs {
            let vid = *self
                .input_index
                .get(*name)
                .ok_or_else(|| SessionError::InputNotFound { name: (*name).to_string() })?;
            let want_dtype = self.value_dtypes[&vid];
            let want_shape = &self.value_shapes[&vid];
            if tensor.dtype != want_dtype {
                return Err(SessionError::DtypeMismatch {
                    name: (*name).to_string(),
                    expected: format!("{want_dtype:?}"),
                    got: format!("{:?}", tensor.dtype),
                });
            }
            if &tensor.shape != want_shape {
                return Err(SessionError::ShapeMismatch {
                    name: (*name).to_string(),
                    expected: want_shape.clone(),
                    got: tensor.shape.clone(),
                });
            }
            let buf = self
                .buffers
                .get_mut(&vid)
                .expect("input value has a buffer");
            write_host(buf, tensor.as_bytes())?;
            bound.push(vid);
        }
        for &vid in &self.required_inputs {
            if !bound.contains(&vid) {
                let name = self
                    .graph
                    .value(vid)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("value#{}", vid.0));
                return Err(SessionError::InputNotFound { name });
            }
        }

        // --- Execute nodes ---------------------------------------------------
        // Split borrows by field so the kernel (borrowed from `cache`) and the
        // buffers can be touched in the same iteration.
        let graph = &self.graph;
        let ep = self.ep.clone();
        let cache = &mut self.cache;
        let buffers = &mut self.buffers;

        for np in &self.plan {
            // Precompute contiguous strides for every input/output view; these
            // holders must outlive the views that borrow them.
            let in_strides: Vec<Vec<i64>> =
                np.input_shapes.iter().map(|s| compute_contiguous_strides(s)).collect();
            let out_strides: Vec<Vec<i64>> =
                np.output_shapes.iter().map(|s| compute_contiguous_strides(s)).collect();

            // Input base pointers (raw, no lingering borrow) + bounds gate.
            let mut in_ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(np.inputs.len());
            for (i, &vid) in np.inputs.iter().enumerate() {
                let buf = buffers.get(&vid).ok_or_else(|| SessionError::Internal(
                    format!("missing buffer for input value#{}", vid.0),
                ))?;
                view_bounds(
                    &np.input_shapes[i],
                    &in_strides[i],
                    0,
                    np.input_dtypes[i],
                    buf.len(),
                )?;
                in_ptrs.push(buf.as_ptr());
            }

            // Take output buffers out of the map so they can be borrowed `&mut`
            // without conflicting with the input reads still in the map (SSA
            // guarantees outputs are disjoint from inputs).
            let mut out_bufs: Vec<(ValueId, DeviceBuffer)> = Vec::with_capacity(np.outputs.len());
            for &vid in &np.outputs {
                let buf = buffers.remove(&vid).ok_or_else(|| SessionError::Internal(
                    format!("missing buffer for output value#{}", vid.0),
                ))?;
                out_bufs.push((vid, buf));
            }

            // Build input views over the raw pointers.
            let mut views: Vec<TensorView> = Vec::with_capacity(np.inputs.len());
            for i in 0..np.inputs.len() {
                views.push(TensorView::new(
                    DevicePtr(in_ptrs[i]),
                    np.input_dtypes[i],
                    &np.input_shapes[i],
                    &in_strides[i],
                    onnx_runtime_ir::DeviceId::cpu(),
                ));
            }

            // Build output views + bounds gate.
            let mut outs: Vec<TensorMut> = Vec::with_capacity(out_bufs.len());
            for (i, (_, buf)) in out_bufs.iter_mut().enumerate() {
                view_bounds(
                    &np.output_shapes[i],
                    &out_strides[i],
                    0,
                    np.output_dtypes[i],
                    buf.len(),
                )?;
                let ptr = buf.as_mut_ptr();
                outs.push(TensorMut::new(
                    DevicePtrMut(ptr),
                    np.output_dtypes[i],
                    &np.output_shapes[i],
                    &out_strides[i],
                    onnx_runtime_ir::DeviceId::cpu(),
                ));
            }

            // Resolve the kernel (cache hit at run time) and dispatch.
            let node = graph.node(np.node_id);
            let kernel = cache.get_or_create(np, node, &ep)?;
            kernel.execute(&views, &mut outs)?;

            // Drop the views (they borrow the holders/pointers) before moving
            // the output buffers back into the map.
            drop(views);
            drop(outs);
            for (vid, buf) in out_bufs {
                buffers.insert(vid, buf);
            }
        }

        // --- Collect graph outputs into owned tensors -----------------------
        let mut results = Vec::with_capacity(self.graph.outputs.len());
        for &vid in &self.graph.outputs {
            let dtype = self.value_dtypes[&vid];
            let shape = self.value_shapes[&vid].clone();
            let buf = self
                .buffers
                .get(&vid)
                .ok_or_else(|| SessionError::Internal(format!("output value#{} not produced", vid.0)))?;
            let n = dtype.storage_bytes(shape.iter().product());
            let bytes = &host_bytes(buf)[..n];
            results.push(Tensor::from_raw_in(
                self.ep.clone(),
                dtype,
                shape,
                bytes,
            )?);
        }
        Ok(results)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Free every buffer via the owning EP (DeviceBuffer has no Drop).
        for (_, buf) in self.buffers.drain() {
            let _ = self.ep.deallocate(buf);
        }
    }
}

/// Instantiate and initialize the Phase-1 CPU execution provider (§20.7,
/// CPU-only auto-detection). A GPU/accelerator EP would be prepended here in a
/// later phase; for Phase 1 the CPU EP is the sole, always-available backend.
pub(crate) fn auto_detect_cpu_ep() -> Result<Arc<CpuExecutionProvider>> {
    let mut ep = CpuExecutionProvider::new();
    ep.initialize(&Default::default())?;
    Ok(Arc::new(ep))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holden's precondition: the dispatch-boundary gate must reject a view that
    /// addresses bytes past its backing allocation, rather than letting a kernel
    /// dereference out of bounds (UB).
    #[test]
    fn view_bounds_rejects_out_of_bounds_view() {
        // A [2, 3] f32 view needs 24 bytes; give it a 16-byte backing length.
        let shape = [2usize, 3];
        let strides = compute_contiguous_strides(&shape);
        let err = view_bounds(&shape, &strides, 0, DataType::Float32, 16);
        assert!(err.is_err(), "gate must reject an oversized view");

        // Exactly-fitting length is accepted.
        assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 24).is_ok());
    }

    /// A negative byte offset region (via a byte_offset that pushes the origin
    /// past the buffer) is also rejected.
    #[test]
    fn view_bounds_rejects_offset_overrun() {
        let shape = [4usize];
        let strides = compute_contiguous_strides(&shape);
        // 4 f32 = 16 bytes; origin at byte 8 leaves only 8 bytes → overrun.
        assert!(view_bounds(&shape, &strides, 8, DataType::Float32, 16).is_err());
        assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 16).is_ok());
    }
}
