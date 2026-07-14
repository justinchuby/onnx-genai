//! The sequential CPU executor (Track D, `docs/ORT2.md` §20, §11.3).
//!
//! Turns a loaded [`Graph`] plus its live [`WeightStore`] into a runnable plan:
//! resolve every value's concrete shape from the actual bound inputs at
//! `run`, size a device buffer per value from those *resolved* shapes, resolve
//! a kernel per node through the execution provider (keyed by the resolved
//! input shapes), then walk the topological order binding
//! [`TensorView`]/[`TensorMut`] windows over those buffers and invoking each
//! kernel. It is generic over any [`Graph`] and any [`ExecutionProvider`]; the
//! Phase-1 build wires in the CPU EP only, but nothing here is op- or
//! model-specific.
//!
//! ## Symbolic → concrete shape resolution (§3.2, §11)
//!
//! Real models carry *symbolic* input dims (e.g. `batch`, `max_seq_len`): the
//! loader produces a [`Shape`] whose dims are a mix of [`Dim::Static`] and
//! [`Dim::Symbolic`]. This executor is model-agnostic about them — a symbol is
//! whatever [`SymbolId`] the graph interned. At `run` it reads the actual shape
//! of each bound input, **binds** the graph's symbols to concrete sizes from
//! those inputs (conflicts across inputs are an error), and **substitutes**
//! those bindings into every value's loader shape to obtain a fully-concrete
//! shape. Buffers are sized from the resolved shapes and become run-scoped when
//! shapes are dynamic (reused when the resolved shape is unchanged, re-allocated
//! when it changes). A fully-static graph is simply the special case where
//! there are no symbols: resolution is a no-op and every buffer/kernel is
//! materialized once at build.
//!
//! The session does **not** infer op output shapes — that is the loader's job
//! (`onnx-runtime-loader::shape_inference`). If a value's loader shape still
//! contains an unbound symbol after substitution, the session cannot size its
//! buffer and reports [`SessionError::UnresolvedShape`] naming the value and its
//! producing op, rather than guessing.
//!
//! ## Holden's precondition (ep-api safety review #1) — enforced here
//!
//! A [`TensorView`] carries no backing length, so it cannot self-check storage
//! bounds. This executor owns every buffer, so it is the layer that *can*: for
//! **every** input and output view of **every** node it calls
//! [`strided::view_in_bounds`] (or, for sub-byte dtypes, the `storage_bytes`
//! equivalent in [`view_bounds`]) against the **run-scoped resolved** buffer and
//! refuses to dispatch on failure. That check is the sole thing that makes
//! ep-cpu's unchecked pointer derefs sound.

use std::collections::HashMap;
use std::sync::Arc;

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::strided::view_in_bounds;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    as_static_shape, compute_contiguous_strides, DataType, Dim, Graph, Node, NodeId, Shape,
    SymbolId, TensorLayout, ValueId,
};
use onnx_runtime_loader::WeightStore;

use crate::error::{Result, SessionError};
use crate::tensor::{host_bytes, write_host, Tensor};

/// A per-node compiled entry: the structural facts the run loop needs without
/// re-deriving them from the graph. Shapes are **not** baked here — they are
/// resolved per run from the bound inputs (see module docs).
#[derive(Debug)]
pub(crate) struct NodePlan {
    pub node_id: NodeId,
    /// Present (non-skipped) input value ids, in positional order.
    pub inputs: Vec<ValueId>,
    /// Output value ids, in positional order.
    pub outputs: Vec<ValueId>,
    /// Element types of the present inputs.
    pub input_dtypes: Vec<DataType>,
    /// Element types of the outputs.
    pub output_dtypes: Vec<DataType>,
}

/// Cache key for a compiled kernel (§11.1). Keyed by the concrete node and its
/// **resolved** (concrete) input shapes: attributes are fixed per node, so this
/// is correct, and the shape component makes it *shape-keyed* — a re-run with
/// the same resolved shapes hits, a different shape (e.g. a new batch/seq)
/// misses and re-compiles. This preserves Chew's guarantee: a kernel is never
/// reused for a shape it was not compiled for.
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

    /// Return the cached kernel for `(node, resolved_input_shapes)`, verifying
    /// EP support and compiling+inserting it on a miss. The EP support check
    /// lives on the miss path so a re-planned shape is re-validated exactly
    /// once per distinct shape.
    fn get_or_create(
        &mut self,
        node_id: NodeId,
        node: &Node,
        input_shapes: &[Vec<usize>],
        opset: u64,
        ep: &CpuExecutionProvider,
    ) -> Result<&dyn onnx_runtime_ep_api::Kernel> {
        let key = KernelKey {
            node: node_id.0,
            shapes: input_shapes.to_vec(),
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            // Verify the EP claims this op at these concrete shapes/layouts
            // before compiling — same gate the static path used at build.
            let shape_dims: Vec<Shape> = input_shapes
                .iter()
                .map(|s| s.iter().map(|&d| Dim::Static(d)).collect())
                .collect();
            let layouts = vec![TensorLayout::contiguous(); input_shapes.len()];
            if !matches!(
                ep.supports_op(node, &shape_dims, &layouts),
                KernelMatch::Supported { .. }
            ) {
                return Err(SessionError::UnsupportedOp {
                    op_type: node.op_type.clone(),
                });
            }
            let kernel = ep.get_kernel(node, input_shapes, opset)?;
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
    /// One device buffer per backed value. Static values are allocated once at
    /// build; dynamic (symbol-shaped) values are allocated per run and cached
    /// here so a run whose resolved shape is unchanged reuses the allocation.
    buffers: HashMap<ValueId, DeviceBuffer>,
    /// The concrete shape each live buffer in [`Self::buffers`] is currently
    /// sized for — the reuse key for run-scoped buffers.
    buffer_shapes: HashMap<ValueId, Vec<usize>>,
    /// Loader-produced (possibly symbolic) shape of every value.
    value_shapes: HashMap<ValueId, Shape>,
    /// Element type of every value.
    value_dtypes: HashMap<ValueId, DataType>,
    /// Topologically ordered execution plan (structure only; shapes per run).
    plan: Vec<NodePlan>,
    /// name → value id for the graph inputs the caller must supply.
    input_index: HashMap<String, ValueId>,
    /// Value ids the caller must supply at `run` (graph inputs minus initializers).
    required_inputs: Vec<ValueId>,
    /// Whether any value in the graph carries a symbolic dim. A fully-static
    /// graph is materialized eagerly at build; a symbolic graph defers buffer
    /// allocation and kernel compilation to the first `run` that fixes shapes.
    has_symbols: bool,
    cache: KernelCache,
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
            return Err(SessionError::from(
                onnx_runtime_ep_api::EpError::InvalidTensorView {
                    reason: format!(
                        "sub-byte view needs {need} bytes but backing allocation is {buffer_len}"
                    ),
                },
            ));
        }
        return Ok(());
    }
    view_in_bounds(shape, strides, byte_offset, esize, buffer_len)?;
    Ok(())
}

/// Element count of a shape with overflow checking. A malicious or corrupt
/// shape whose dims multiply past `usize::MAX` would silently wrap under a plain
/// `iter().product()`, under-sizing the backing buffer. Returns
/// [`SessionError::ShapeOverflow`] instead so the caller allocates nothing.
fn checked_numel(dims: &[usize], value: impl FnOnce() -> String) -> Result<usize> {
    let mut acc = 1usize;
    for &d in dims {
        acc = match acc.checked_mul(d) {
            Some(n) => n,
            None => {
                return Err(SessionError::ShapeOverflow {
                    value: value(),
                    dims: dims.to_vec(),
                })
            }
        };
    }
    Ok(acc)
}

/// Byte size of `numel` elements of `dtype` with overflow checking. Even when
/// the element *count* fits in `usize` (guarded by [`checked_numel`]), the
/// element-count → bytes multiply can still wrap for a fixed-width dtype and
/// under-size the backing buffer. Returns [`SessionError::ShapeOverflow`] so the
/// caller allocates nothing rather than a wrapped, undersized buffer.
fn checked_storage_bytes(
    dtype: DataType,
    numel: usize,
    value: impl FnOnce() -> String,
    dims: &[usize],
) -> Result<usize> {
    dtype
        .checked_storage_bytes(numel)
        .ok_or_else(|| SessionError::ShapeOverflow {
            value: value(),
            dims: dims.to_vec(),
        })
}

/// The effective operator-set version governing `node` — the graph's imported
/// opset for the node's domain. The default ONNX domain is spelled both `""`
/// and `"ai.onnx"`; both map to the same import. When a graph omits the import
/// (should not happen for a loaded model), fall back to `u64::MAX` so ops with a
/// single registration still resolve and opset-specialized ops pick their
/// newest kernel.
fn effective_opset(graph: &Graph, node: &Node) -> u64 {
    let domain = node.domain.as_str();
    graph
        .opset_imports
        .get(domain)
        .or_else(|| {
            if domain.is_empty() {
                graph.opset_imports.get("ai.onnx")
            } else if domain == "ai.onnx" {
                graph.opset_imports.get("")
            } else {
                None
            }
        })
        .copied()
        .unwrap_or(u64::MAX)
}

/// Substitute concrete symbol bindings into a (possibly symbolic) shape.
/// Returns `None` if any dim is a symbol with no binding.
fn substitute(shape: &Shape, bindings: &HashMap<SymbolId, usize>) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Symbolic(s) => bindings.get(s).copied(),
        })
        .collect()
}

/// Decode a host buffer's integer elements as `i64` for `dtype`, or `None` if
/// the dtype is not an integer the shape math understands. Used to read the
/// *values* of shape-defining inputs (e.g. `Slice` starts/ends) at run time.
fn buffer_as_i64(buffer: &DeviceBuffer, dtype: DataType) -> Option<Vec<i64>> {
    let bytes = crate::tensor::host_bytes(buffer);
    match dtype {
        DataType::Int64 => Some(
            bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DataType::Int32 => Some(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
        ),
        _ => None,
    }
}

/// Compute the concrete output shapes of a *data-dependent* shape op from its
/// already-resolved input shapes and the runtime *values* of its integer
/// inputs. This is the executor's fallback for the rare value whose shape the
/// loader's static (symbolic) inference could not pin down — e.g. a `Slice`
/// whose `ends` is produced by a runtime `Shape → Min → Cast` chain, so its
/// extent is only known once those upstream nodes have executed.
///
/// Model-agnostic: it dispatches on the op type alone. Returns `None` for ops
/// this executor cannot resolve dynamically, which surfaces as
/// [`SessionError::UnresolvedShape`] exactly as before.
fn dynamic_output_shapes(
    node: &Node,
    input_shapes: &[Vec<usize>],
    input_values: &[Option<Vec<i64>>],
) -> Option<Vec<Vec<usize>>> {
    match node.op_type.as_str() {
        // Opset-10+ `Slice`: data, starts, ends, [axes], [steps] as inputs. The
        // per-axis element count mirrors the `Slice` kernel's clamp semantics
        // exactly (ONNX reference), so the buffer we size here matches what the
        // kernel writes.
        "Slice" => {
            let data_shape = input_shapes.first()?;
            let starts = input_values.get(1)?.as_ref()?;
            let ends = input_values.get(2)?.as_ref()?;
            let (axes, steps) = onnx_runtime_ep_cpu::slice_axes_steps(
                starts.len(),
                input_values.get(3).and_then(|v| v.as_deref()),
                input_values.get(4).and_then(|v| v.as_deref()),
            );
            // Reuse the exact kernel geometry helper so the buffer we size here
            // always matches what the Slice kernel writes. Any error (length
            // mismatch, out-of-range axis, zero step) means "cannot resolve".
            let plan =
                onnx_runtime_ep_cpu::slice_plan(data_shape, starts, ends, &axes, &steps).ok()?;
            let count: Vec<usize> = plan.iter().map(|p| p.count).collect();
            Some(vec![count])
        }
        _ => None,
    }
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

        let mut value_shapes: HashMap<ValueId, Shape> = HashMap::new();
        let mut value_dtypes: HashMap<ValueId, DataType> = HashMap::new();
        let mut buffers: HashMap<ValueId, DeviceBuffer> = HashMap::new();
        let mut buffer_shapes: HashMap<ValueId, Vec<usize>> = HashMap::new();

        // 1) Initializers: always concrete. Record dims, allocate, copy weights.
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!("weight bytes unavailable for value#{}", vid.0))
            })?;
            let mut buf = ep.allocate(bytes.len().max(1), TensorLayout::contiguous().alignment)?;
            write_host(&mut buf, bytes)?;
            value_dtypes.insert(vid, dtype);
            value_shapes.insert(vid, dims.iter().map(|&d| Dim::Static(d)).collect());
            buffer_shapes.insert(vid, dims);
            buffers.insert(vid, buf);
        }

        // 2) Record the loader shape + dtype of every remaining value (graph
        //    inputs and node outputs). No allocation yet — shapes may be
        //    symbolic and are only sized once resolved.
        for &vid in &graph.inputs {
            value_shapes
                .entry(vid)
                .or_insert_with(|| graph.value(vid).shape.clone());
            value_dtypes.entry(vid).or_insert(graph.value(vid).dtype);
        }
        for &nid in &order {
            for &out in &graph.node(nid).outputs {
                value_shapes
                    .entry(out)
                    .or_insert_with(|| graph.value(out).shape.clone());
                value_dtypes.entry(out).or_insert(graph.value(out).dtype);
            }
        }

        let has_symbols = value_shapes.values().any(|s| as_static_shape(s).is_none());

        // 3) Build the structural per-node plan.
        let mut plan = Vec::with_capacity(order.len());
        for &nid in &order {
            let node = graph.node(nid);
            let inputs: Vec<ValueId> = node.input_values().collect();
            let outputs: Vec<ValueId> = node.outputs.clone();
            let input_dtypes: Vec<DataType> = inputs.iter().map(|v| value_dtypes[v]).collect();
            let output_dtypes: Vec<DataType> = outputs.iter().map(|v| value_dtypes[v]).collect();
            plan.push(NodePlan {
                node_id: nid,
                inputs,
                outputs,
                input_dtypes,
                output_dtypes,
            });
        }

        // 4) name → value id and the set of caller-required inputs.
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
            buffer_shapes,
            value_shapes,
            value_dtypes,
            plan,
            input_index,
            required_inputs,
            has_symbols,
            cache: KernelCache::default(),
        };

        // 5) Fully-static graphs are materialized eagerly (buffers + the whole
        //    "compiled plan" of kernels), so the first `run` sees only cache
        //    hits. Symbolic graphs cannot be sized until a `run` fixes their
        //    shapes, so their buffers/kernels are created on first use.
        if !exec.has_symbols {
            let empty = HashMap::new();
            let resolved = exec.resolve_all(&empty)?;
            exec.size_buffers(&resolved)?;
            exec.compile_all(&resolved)?;
        }
        Ok(exec)
    }

    /// Allocate `vid`'s buffer for `dims`, or reuse the existing allocation when
    /// it is already sized for `dims` (the run-scoped reuse path).
    fn ensure_buffer(&mut self, vid: ValueId, dtype: DataType, dims: &[usize]) -> Result<()> {
        if self.buffer_shapes.get(&vid).map(|s| s.as_slice()) == Some(dims) {
            return Ok(()); // identical shape → reuse allocation
        }
        if let Some(old) = self.buffers.remove(&vid) {
            self.ep.deallocate(old)?;
        }
        let numel = checked_numel(dims, || format!("value#{}", vid.0))?;
        let size = checked_storage_bytes(dtype, numel, || format!("value#{}", vid.0), dims)?;
        let buf = self
            .ep
            .allocate(size.max(1), TensorLayout::contiguous().alignment)?;
        self.buffers.insert(vid, buf);
        self.buffer_shapes.insert(vid, dims.to_vec());
        Ok(())
    }

    /// Resolve every value's concrete shape by substituting `bindings` into its
    /// loader shape. A value whose shape stays symbolic (unbound) cannot be
    /// sized: report it as an uninferred shape, naming its producing op.
    fn resolve_all(
        &self,
        bindings: &HashMap<SymbolId, usize>,
    ) -> Result<HashMap<ValueId, Vec<usize>>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            match substitute(shape, bindings) {
                Some(dims) => {
                    resolved.insert(vid, dims);
                }
                None => {
                    let value = self.graph.value(vid);
                    let name = value
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("value#{}", vid.0));
                    let op = value
                        .producer
                        .map(|nid| self.graph.node(nid).op_type.clone())
                        .unwrap_or_else(|| "<graph input>".to_string());
                    return Err(SessionError::UnresolvedShape { value: name, op });
                }
            }
        }
        Ok(resolved)
    }

    /// Like [`Self::resolve_all`] but never errors: values whose shape stays
    /// symbolic (a data-dependent extent the loader could not pin down) are
    /// simply omitted, to be resolved just-in-time during execution once their
    /// producing node's inputs are concrete.
    fn resolve_soft(&self, bindings: &HashMap<SymbolId, usize>) -> HashMap<ValueId, Vec<usize>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            if let Some(dims) = substitute(shape, bindings) {
                resolved.insert(vid, dims);
            }
        }
        resolved
    }

    /// Size (allocate or reuse) a backing buffer for every value from its
    /// resolved concrete shape. Initializers already hold their weights and are
    /// left untouched. Values whose shape is not (yet) in `resolved` — the
    /// data-dependent ones filled in during execution — are skipped here and
    /// sized just-in-time in the run loop.
    fn size_buffers(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        let vids: Vec<ValueId> = self.value_shapes.keys().copied().collect();
        for vid in vids {
            if self.graph.initializers.contains_key(&vid) {
                continue;
            }
            let dtype = self.value_dtypes[&vid];
            let Some(dims) = resolved.get(&vid).cloned() else {
                continue;
            };
            self.ensure_buffer(vid, dtype, &dims)?;
        }
        Ok(())
    }

    /// Resolved input shapes of a plan node, in positional order.
    fn node_input_shapes(
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        plan.inputs.iter().map(|v| resolved[v].clone()).collect()
    }

    /// Resolved output shapes of a plan node, in positional order.
    fn node_output_shapes(
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        plan.outputs.iter().map(|v| resolved[v].clone()).collect()
    }

    /// Populate the kernel cache for the compiled plan against `resolved` shapes.
    fn compile_all(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        for i in 0..self.plan.len() {
            let node_id = self.plan[i].node_id;
            let input_shapes = Self::node_input_shapes(&self.plan[i], resolved);
            let node = self.graph.node(node_id);
            let opset = effective_opset(&self.graph, node);
            self.cache
                .get_or_create(node_id, node, &input_shapes, opset, &self.ep)?;
        }
        Ok(())
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Warmup: re-touch the shape-keyed cache for the compiled plan so the first
    /// real `run` sees only cache hits (§11.3). Only meaningful for fully-static
    /// graphs, whose plan shapes are known at build; symbolic graphs cannot be
    /// pre-compiled without a concrete shape and warm up on their first `run`.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        if self.has_symbols {
            return Ok(());
        }
        let empty = HashMap::new();
        let resolved = self.resolve_all(&empty)?;
        self.compile_all(&resolved)
    }

    /// Bind the graph's symbols to concrete sizes from the actual bound-input
    /// shapes, validating rank and static dims and detecting symbol conflicts.
    fn bind_symbols(
        &self,
        inputs: &[(&str, &Tensor)],
    ) -> Result<HashMap<SymbolId, usize>> {
        let mut bindings: HashMap<SymbolId, usize> = HashMap::new();
        for (name, tensor) in inputs {
            let vid = *self
                .input_index
                .get(*name)
                .ok_or_else(|| SessionError::InputNotFound {
                    name: (*name).to_string(),
                })?;
            let want_dtype = self.value_dtypes[&vid];
            if tensor.dtype != want_dtype {
                return Err(SessionError::DtypeMismatch {
                    name: (*name).to_string(),
                    expected: format!("{want_dtype:?}"),
                    got: format!("{:?}", tensor.dtype),
                });
            }
            let decl = &self.value_shapes[&vid];
            if decl.len() != tensor.shape.len() {
                return Err(SessionError::RankMismatch {
                    name: (*name).to_string(),
                    expected: decl.len(),
                    got: tensor.shape.len(),
                });
            }
            for (dim, &actual) in decl.iter().zip(&tensor.shape) {
                match dim {
                    Dim::Static(n) => {
                        if *n != actual {
                            return Err(SessionError::ShapeMismatch {
                                name: (*name).to_string(),
                                expected: as_static_shape(decl).unwrap_or_default(),
                                got: tensor.shape.clone(),
                            });
                        }
                    }
                    Dim::Symbolic(s) => {
                        if let Some(&prev) = bindings.get(s) {
                            if prev != actual {
                                let sym = self
                                    .symbol_name(*s)
                                    .unwrap_or_else(|| format!("symbol#{}", s.0));
                                return Err(SessionError::SymbolConflict {
                                    symbol: sym,
                                    first: prev,
                                    second: actual,
                                });
                            }
                        } else {
                            bindings.insert(*s, actual);
                        }
                    }
                }
            }
        }
        Ok(bindings)
    }

    /// Human-readable name of a symbol, if the graph recorded one.
    fn symbol_name(&self, s: SymbolId) -> Option<String> {
        self.graph
            .symbol_constraints
            .get(&s)
            .and_then(|c| c.name.clone())
    }

    /// Sequential topological executor.
    pub(crate) fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        // --- Resolve shapes from the actual bound inputs --------------------
        let bindings = self.bind_symbols(inputs)?;

        // Every required input must be supplied.
        let provided: Vec<ValueId> = inputs
            .iter()
            .filter_map(|(name, _)| self.input_index.get(*name).copied())
            .collect();
        for &vid in &self.required_inputs {
            if !provided.contains(&vid) {
                let name = self
                    .graph
                    .value(vid)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("value#{}", vid.0));
                return Err(SessionError::InputNotFound { name });
            }
        }

        // Substitute the bindings into every value → concrete shapes, then size
        // the run-scoped buffers from them (reused when unchanged). Values with a
        // data-dependent shape stay unresolved here and are filled in during the
        // execution loop, once their producing node's inputs are concrete.
        let mut resolved = self.resolve_soft(&bindings);
        self.size_buffers(&resolved)?;

        // --- Bind input bytes into their (now correctly sized) buffers ------
        for (name, tensor) in inputs {
            let vid = self.input_index[*name];
            let buf = self
                .buffers
                .get_mut(&vid)
                .expect("input value has a buffer");
            write_host(buf, tensor.as_bytes())?;
        }

        // --- Execute nodes ---------------------------------------------------
        // Split borrows by field so the kernel (borrowed from `cache`) and the
        // buffers can be touched in the same iteration.
        let graph = &self.graph;
        let ep = self.ep.clone();
        let cache = &mut self.cache;
        let buffers = &mut self.buffers;

        for np in &self.plan {
            let input_shapes = Self::node_input_shapes(np, &resolved);

            // Data-dependent shapes: if any output's shape is still unresolved,
            // compute it now from the concrete input shapes + the runtime values
            // of this node's integer inputs (which upstream nodes have already
            // produced), then size those buffers just-in-time.
            if np.outputs.iter().any(|v| !resolved.contains_key(v)) {
                let input_values: Vec<Option<Vec<i64>>> = np
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        buffers
                            .get(v)
                            .and_then(|b| buffer_as_i64(b, np.input_dtypes[i]))
                    })
                    .collect();
                let node = graph.node(np.node_id);
                let out_shapes = dynamic_output_shapes(node, &input_shapes, &input_values)
                    .ok_or_else(|| {
                        let vid = np
                            .outputs
                            .iter()
                            .find(|v| !resolved.contains_key(v))
                            .copied()
                            .unwrap_or(np.outputs[0]);
                        let value = graph.value(vid);
                        SessionError::UnresolvedShape {
                            value: value
                                .name
                                .clone()
                                .unwrap_or_else(|| format!("value#{}", vid.0)),
                            op: node.op_type.clone(),
                        }
                    })?;
                // A future multi-output data-dependent op would misindex
                // `out_shapes[oi]`; verify the sizer returned exactly one shape
                // per output rather than panicking on a length mismatch.
                if out_shapes.len() != np.outputs.len() {
                    return Err(SessionError::OutputShapeCountMismatch {
                        op: node.op_type.clone(),
                        expected: np.outputs.len(),
                        got: out_shapes.len(),
                    });
                }
                for (oi, &ovid) in np.outputs.iter().enumerate() {
                    let dims = out_shapes[oi].clone();
                    let numel = checked_numel(&dims, || format!("value#{}", ovid.0))?;
                    let need = checked_storage_bytes(
                        np.output_dtypes[oi],
                        numel,
                        || format!("value#{}", ovid.0),
                        &dims,
                    )?
                    .max(1);
                    let fits = buffers.get(&ovid).map(|b| b.len() == need).unwrap_or(false);
                    if !fits {
                        if let Some(old) = buffers.remove(&ovid) {
                            ep.deallocate(old)?;
                        }
                        let buf = ep.allocate(need, TensorLayout::contiguous().alignment)?;
                        buffers.insert(ovid, buf);
                    }
                    resolved.insert(ovid, dims);
                }
            }

            let output_shapes = Self::node_output_shapes(np, &resolved);

            // Precompute contiguous strides for every input/output view; these
            // holders must outlive the views that borrow them.
            let in_strides: Vec<Vec<i64>> = input_shapes
                .iter()
                .map(|s| compute_contiguous_strides(s))
                .collect();
            let out_strides: Vec<Vec<i64>> = output_shapes
                .iter()
                .map(|s| compute_contiguous_strides(s))
                .collect();

            // Input base pointers (raw, no lingering borrow) + bounds gate
            // against the run-scoped resolved buffers.
            let mut in_ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(np.inputs.len());
            for (i, &vid) in np.inputs.iter().enumerate() {
                let buf = buffers.get(&vid).ok_or_else(|| {
                    SessionError::Internal(format!("missing buffer for input value#{}", vid.0))
                })?;
                view_bounds(
                    &input_shapes[i],
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
                let buf = buffers.remove(&vid).ok_or_else(|| {
                    SessionError::Internal(format!("missing buffer for output value#{}", vid.0))
                })?;
                out_bufs.push((vid, buf));
            }

            // Build input views over the raw pointers.
            let mut views: Vec<TensorView> = Vec::with_capacity(np.inputs.len());
            for i in 0..np.inputs.len() {
                views.push(TensorView::new(
                    DevicePtr(in_ptrs[i]),
                    np.input_dtypes[i],
                    &input_shapes[i],
                    &in_strides[i],
                    onnx_runtime_ir::DeviceId::cpu(),
                ));
            }

            // Build output views + bounds gate.
            let mut outs: Vec<TensorMut> = Vec::with_capacity(out_bufs.len());
            for (i, (_, buf)) in out_bufs.iter_mut().enumerate() {
                view_bounds(
                    &output_shapes[i],
                    &out_strides[i],
                    0,
                    np.output_dtypes[i],
                    buf.len(),
                )?;
                let ptr = buf.as_mut_ptr();
                outs.push(TensorMut::new(
                    DevicePtrMut(ptr),
                    np.output_dtypes[i],
                    &output_shapes[i],
                    &out_strides[i],
                    onnx_runtime_ir::DeviceId::cpu(),
                ));
            }

            // Resolve the kernel (shape-keyed by the resolved input shapes) and
            // dispatch.
            let node = graph.node(np.node_id);
            let opset = effective_opset(graph, node);
            let kernel = cache.get_or_create(np.node_id, node, &input_shapes, opset, &ep)?;
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
            let shape = resolved[&vid].clone();
            let buf = self.buffers.get(&vid).ok_or_else(|| {
                SessionError::Internal(format!("output value#{} not produced", vid.0))
            })?;
            let n = dtype.storage_bytes(shape.iter().product());
            let bytes = &host_bytes(buf)[..n];
            results.push(Tensor::from_raw_in(self.ep.clone(), dtype, shape, bytes)?);
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

    /// Symbol substitution: static dims pass through, bound symbols resolve, an
    /// unbound symbol yields `None` (the uninferred-shape signal).
    #[test]
    fn substitute_resolves_bound_symbols_only() {
        let mut bindings = HashMap::new();
        bindings.insert(SymbolId(0), 7usize);
        let shape = vec![Dim::Symbolic(SymbolId(0)), Dim::Static(4)];
        assert_eq!(substitute(&shape, &bindings), Some(vec![7, 4]));

        let unbound = vec![Dim::Symbolic(SymbolId(1)), Dim::Static(4)];
        assert_eq!(substitute(&unbound, &bindings), None);
    }

    /// H-D1: element-count multiplication must be overflow-checked so a huge or
    /// malicious shape reports `ShapeOverflow` instead of wrapping `usize` and
    /// under-sizing the buffer.
    #[test]
    fn checked_numel_detects_overflow() {
        // Well-formed shapes multiply normally.
        assert_eq!(checked_numel(&[2, 3, 4], || "v".into()).unwrap(), 24);
        assert_eq!(checked_numel(&[], || "v".into()).unwrap(), 1);

        // A product past usize::MAX overflows.
        let huge = [usize::MAX, 2];
        let err = checked_numel(&huge, || "value#9".into());
        assert!(matches!(
            err,
            Err(SessionError::ShapeOverflow { .. })
        ));
    }

    /// H-D1 (byte layer): even when the element *count* fits in `usize`, the
    /// count → bytes multiply can wrap for a fixed-width dtype. The allocation
    /// path must report `ShapeOverflow` rather than under-allocating.
    #[test]
    fn checked_storage_bytes_detects_byte_overflow() {
        // `usize::MAX / 4` elements fit in usize (pass checked_numel) but
        // `* 8` bytes for Float64 wraps — this is the exploited under-alloc.
        let numel = usize::MAX / 4;
        let err = checked_storage_bytes(DataType::Float64, numel, || "value#9".into(), &[numel]);
        assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));

        // A well-formed size passes through unchanged.
        assert_eq!(
            checked_storage_bytes(DataType::Float32, 4, || "v".into(), &[4]).unwrap(),
            16
        );
    }

    /// The data-dependent shape sizer must return exactly one shape per output
    /// so the run loop's `out_shapes[oi]` indexing can never misindex. Slice is
    /// single-output, so it returns a 1-element Vec; the run loop additionally
    /// guards the count (see `OutputShapeCountMismatch`).
    #[test]
    fn dynamic_output_shapes_slice_is_single_output() {
        let node = Node::new(NodeId(0), "Slice", vec![], vec![]);
        let input_shapes = vec![vec![4usize, 2]];
        let input_values = vec![
            None,           // data (unused by sizer)
            Some(vec![1]),  // starts
            Some(vec![3]),  // ends
            Some(vec![0]),  // axes
            Some(vec![1]),  // steps
        ];
        let out = dynamic_output_shapes(&node, &input_shapes, &input_values).unwrap();
        assert_eq!(out.len(), 1, "Slice must resolve exactly one output shape");
        assert_eq!(out[0], vec![2, 2]);

        // An op the sizer cannot resolve returns None (surfaces as UnresolvedShape).
        let other = Node::new(NodeId(1), "Conv", vec![], vec![]);
        assert!(dynamic_output_shapes(&other, &input_shapes, &input_values).is_none());
    }

    /// The effective opset is read from the graph's import for the op's domain,
    /// with the default and `ai.onnx` spellings treated as one.
    #[test]
    fn effective_opset_reads_graph_import() {
        let mut graph = Graph::default();
        graph.opset_imports.insert(String::new(), 12);
        let node = Node::new(NodeId(0), "Softmax", vec![], vec![]);
        assert_eq!(effective_opset(&graph, &node), 12);

        // Missing import falls back to u64::MAX (assume latest).
        let empty = Graph::default();
        assert_eq!(effective_opset(&empty, &node), u64::MAX);
    }
}
