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
//! (the loader runs `onnx-runtime-shape-inference` at load time). If a value's
//! loader shape still contains an unbound symbol after substitution, the
//! session resolves genuinely data-dependent extents just-in-time during
//! execution (see [`dynamic_output_shapes`]); anything it still cannot size is
//! reported as [`SessionError::UnresolvedShape`] naming the value and its
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
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

use crate::error::{Result, SessionError};
use crate::tensor::{host_bytes, write_host, Tensor};

/// A per-node compiled entry: the structural facts the run loop needs without
/// re-deriving them from the graph. Shapes are **not** baked here — they are
/// resolved per run from the bound inputs (see module docs).
#[derive(Debug)]
pub(crate) struct NodePlan {
    pub node_id: NodeId,
    /// Positional input value ids in ONNX signature order. An omitted optional
    /// input (ONNX empty-string input name → `None` slot) is preserved as
    /// `None` so a later present input is never misread as the omitted one
    /// (e.g. `Slice(data, starts, ends, "", steps)`). Trailing `None`s are
    /// trimmed — a truly absent trailing optional simply lowers the arity.
    pub inputs: Vec<Option<ValueId>>,
    /// Output value ids, in positional order.
    pub outputs: Vec<ValueId>,
    /// Element types of the inputs, positional (matches `inputs`). A `None`
    /// input slot carries a placeholder dtype that kernels never read.
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
                return Err(SessionError::unsupported_op(node, node_id, opset, ep.name()));
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
    /// name → value id for every named value in this graph (inputs, outputs,
    /// initializers and interior SSA values). Used to resolve outer-scope
    /// captures referenced by name from a nested control-flow subgraph body.
    name_index: HashMap<String, ValueId>,
    /// Compiled child executors for this graph's control-flow subgraph bodies,
    /// keyed by `(control-flow node, subgraph attr key)`. Built lazily on first
    /// execution (once concrete input shapes are known) and **reused across
    /// Loop/Scan iterations** — the whole point of the efficiency directive: a
    /// body's topo-sort, buffer sizing and kernel compilation happen once, then
    /// every iteration is just a re-bind + dispatch. Rebuilt only if a later
    /// invocation's external input shapes differ from the ones it was compiled
    /// for (a shape-varying loop body — rare).
    subgraph_execs: HashMap<(NodeId, String), CompiledSubgraph>,
}

/// A cached child executor for one control-flow subgraph body, plus the
/// external-input shape signature it was compiled for (so a shape change forces
/// a rebuild rather than a silent shape mismatch).
struct CompiledSubgraph {
    exec: Executor,
    /// Ordered names of the body's external inputs — formal parameters first
    /// (positional, from the body's declared `inputs`), then captured
    /// outer-scope values (bound by name).
    formal_inputs: Vec<String>,
    capture_inputs: Vec<String>,
    /// Concrete shapes the child was last compiled for, in
    /// `formal_inputs ++ capture_inputs` order.
    built_shapes: Vec<Vec<usize>>,
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
/// and `"ai.onnx"`; both map to the same import. Loader and programmatic session
/// entry points validate this invariant before executor construction.
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
        .unwrap_or_else(|| {
            unreachable!(
                "internal invariant violated: node #{} ({}::{}) has no opset import",
                node.id.0,
                if node.domain.is_empty() {
                    "ai.onnx"
                } else {
                    &node.domain
                },
                node.op_type
            )
        })
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
            // EPContext nodes are pre-compiled: they bypass placement and were
            // already restored through their owning EP by the session's
            // consume path (§55.3). They must never be resolved as ordinary
            // kernels — the CPU EP has no `EPContext` kernel — so skip them
            // here.
            if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
                continue;
            }
            // Preserve positional input arity: keep interior `None` (omitted
            // optional) slots so a later present input is not misread as the
            // omitted one, but trim trailing `None`s (a trailing omitted
            // optional just lowers the arity, matching ONNX semantics).
            let mut slots: Vec<Option<ValueId>> = node.inputs.clone();
            while matches!(slots.last(), Some(None)) {
                slots.pop();
            }
            let inputs = slots;
            let outputs: Vec<ValueId> = node.outputs.clone();
            let input_dtypes: Vec<DataType> = inputs
                .iter()
                .map(|v| v.map(|vid| value_dtypes[&vid]).unwrap_or(DataType::Float32))
                .collect();
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

        // Full name → value id map (every named value in the graph), used to
        // resolve a nested subgraph's outer-scope captures by name.
        let mut name_index = HashMap::new();
        for (vid, value) in graph.values.iter() {
            if let Some(name) = &value.name {
                name_index.insert(name.clone(), vid);
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
            name_index,
            subgraph_execs: HashMap::new(),
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

    /// Resolved input shapes of a plan node, in positional order. An omitted
    /// optional input (`None` slot) has no shape; it takes an empty shape,
    /// which the run loop only ever pairs with an absent placeholder view.
    fn node_input_shapes(
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        plan.inputs
            .iter()
            .map(|v| v.map(|vid| resolved[&vid].clone()).unwrap_or_default())
            .collect()
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
            let node = self.graph.node(node_id);
            // Control-flow ops (If/Loop/Scan) are not leaf kernels — they execute
            // nested subgraphs through the executor's own path, so they have no
            // entry in the EP kernel registry and must not be compiled here.
            if is_control_flow_op(&node.op_type, &node.domain) {
                continue;
            }
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

    /// The compiled graph, retained for the §55.4 EPContext dump path: the
    /// exporter needs the (post-optimize) graph to serialise a `*_ctx.onnx`
    /// context-cache model with compiled partitions spliced out.
    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Live weight bytes backing the graph, needed alongside [`Self::graph`] so
    /// the EPContext dump can encode initializers into the context model.
    pub(crate) fn weights(&self) -> &Arc<WeightStore> {
        &self._weights
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
        self.run_scoped(inputs, &HashMap::new())
    }

    /// Execute the graph with `inputs` bound by name, plus an `outer_scope` of
    /// enclosing named values a nested control-flow subgraph body may capture.
    /// The top-level session `run` passes an empty scope; a control-flow body's
    /// child executor is invoked with its enclosing graph's live values so a
    /// deeply-nested body can still reach an outer capture (ONNX lexical scope).
    fn run_scoped(
        &mut self,
        inputs: &[(&str, &Tensor)],
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>> {
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
        // Iterate by index so a control-flow node can take `&mut self` (it must
        // build/reuse child executors) while an ordinary kernel node uses the
        // disjoint-field borrow split inside `exec_kernel_node`.
        for pi in 0..self.plan.len() {
            let node_id = self.plan[pi].node_id;
            let node = self.graph.node(node_id);
            if is_control_flow_op(&node.op_type, &node.domain) {
                self.exec_control_flow(pi, &mut resolved, outer_scope)?;
            } else {
                self.exec_kernel_node(pi, &mut resolved)?;
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

    /// Execute one ordinary (leaf-kernel) plan node: resolve any data-dependent
    /// output shapes, size buffers, build the input/output views (with Holden's
    /// bounds gate), resolve the shape-keyed kernel, and dispatch it.
    fn exec_kernel_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        // Disjoint-field borrows: the kernel is borrowed from `cache` while the
        // buffers are touched in the same iteration.
        let np = &self.plan[pi];
        let graph = &self.graph;
        let ep = self.ep.clone();
        let cache = &mut self.cache;
        let buffers = &mut self.buffers;

        let input_shapes = Self::node_input_shapes(np, resolved);

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
                    v.and_then(|vid| {
                        buffers
                            .get(&vid)
                            .and_then(|b| buffer_as_i64(b, np.input_dtypes[i]))
                    })
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

        let output_shapes = Self::node_output_shapes(np, resolved);

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
        // against the run-scoped resolved buffers. An omitted optional
        // (`None`) slot has no buffer: it gets a null pointer and later
        // becomes a `TensorView::absent` placeholder.
        let mut in_ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(np.inputs.len());
        for (i, slot) in np.inputs.iter().enumerate() {
            let Some(vid) = slot else {
                in_ptrs.push(std::ptr::null());
                continue;
            };
            let buf = buffers.get(vid).ok_or_else(|| {
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

        // Build input views over the raw pointers. An omitted optional
        // (`None`) slot becomes an absent placeholder that preserves
        // positional arity for the kernel.
        let mut views: Vec<TensorView> = Vec::with_capacity(np.inputs.len());
        for i in 0..np.inputs.len() {
            if np.inputs[i].is_none() {
                views.push(TensorView::absent(np.input_dtypes[i]));
                continue;
            }
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
        Ok(())
    }
}

/// Whether `(op_type, domain)` is one of the standard subgraph-bearing
/// control-flow ops the executor handles recursively (default `ai.onnx`
/// domain). Kept in lock-step with the loader's `validate_no_control_flow`
/// allow-list.
fn is_control_flow_op(op_type: &str, domain: &str) -> bool {
    (domain.is_empty() || domain == "ai.onnx") && matches!(op_type, "If" | "Loop" | "Scan")
}

/// Read a single scalar `i64`/`i32` element from a length-1 tensor (Loop's `M`).
fn tensor_scalar_i64(t: &Tensor) -> Option<i64> {
    match t.dtype {
        DataType::Int64 => t
            .as_bytes()
            .get(..8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())),
        DataType::Int32 => t
            .as_bytes()
            .get(..4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64),
        _ => None,
    }
}

/// Read a single scalar bool from a length-1 `BOOL` tensor (a `BOOL` is one
/// byte; any nonzero is true, per ONNX).
fn tensor_scalar_bool(t: &Tensor) -> Option<bool> {
    if t.dtype != DataType::Bool {
        return None;
    }
    t.as_bytes().first().map(|&b| b != 0)
}

/// Build a length-1 `i64` scalar tensor (Loop's `iter_num` body input).
fn scalar_i64_tensor(v: i64) -> Result<Tensor> {
    Tensor::from_raw(DataType::Int64, vec![], &v.to_le_bytes())
}

/// Build a scalar `BOOL` tensor (Loop's `cond` body input).
fn scalar_bool_tensor(v: bool) -> Result<Tensor> {
    Tensor::from_raw(DataType::Bool, vec![], &[u8::from(v)])
}

// === Control-flow (subgraph-executing) ops: If / Loop / Scan ===
//
// These are handled at the executor level rather than as leaf kernels because
// they must recursively execute a nested ONNX [`Graph`] with the enclosing
// scope bound — something a `Kernel` (which sees only tensor views, never the
// session/graph context) cannot do. Each body is compiled to a child
// [`Executor`] once and **reused across iterations** (see [`CompiledSubgraph`]).
impl Executor {
    /// Materialize value `vid`'s current bytes into an owned host [`Tensor`],
    /// using its resolved concrete shape and recorded dtype.
    fn value_tensor(
        &self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<Tensor> {
        let dtype = self.value_dtypes[&vid];
        let shape = resolved.get(&vid).cloned().ok_or_else(|| {
            let name = self
                .graph
                .try_value(vid)
                .and_then(|v| v.name.clone())
                .unwrap_or_else(|| format!("value#{}", vid.0));
            SessionError::UnresolvedShape {
                value: name,
                op: "<control-flow input>".to_string(),
            }
        })?;
        let buf = self.buffers.get(&vid).ok_or_else(|| {
            SessionError::Internal(format!("missing buffer for control-flow input value#{}", vid.0))
        })?;
        let n = dtype.storage_bytes(shape.iter().product());
        let bytes = &host_bytes(buf)[..n];
        Tensor::from_raw_in(self.ep.clone(), dtype, shape, bytes)
    }

    /// Store a control-flow op's produced output `tensor` into this graph's
    /// output value `vid`: (re)size the backing buffer, copy the bytes, and
    /// record the runtime dtype/shape so the caller (and the final output
    /// collection) reads them back correctly. Control-flow output shapes are
    /// data-dependent (the loader never inferred inside the body), so they are
    /// resolved here, exactly as the JIT data-dependent path does for kernels.
    fn store_output_tensor(
        &mut self,
        vid: ValueId,
        tensor: &Tensor,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let dims = tensor.shape.clone();
        let numel = checked_numel(&dims, || format!("value#{}", vid.0))?;
        let need = checked_storage_bytes(tensor.dtype, numel, || format!("value#{}", vid.0), &dims)?
            .max(1);
        let fits = self.buffers.get(&vid).map(|b| b.len() == need).unwrap_or(false);
        if !fits {
            if let Some(old) = self.buffers.remove(&vid) {
                self.ep.deallocate(old)?;
            }
            let buf = self
                .ep
                .allocate(need, TensorLayout::contiguous().alignment)?;
            self.buffers.insert(vid, buf);
        }
        let buf = self.buffers.get_mut(&vid).expect("just ensured");
        write_host(buf, tensor.as_bytes())?;
        self.value_dtypes.insert(vid, tensor.dtype);
        self.buffer_shapes.insert(vid, dims.clone());
        resolved.insert(vid, dims);
        Ok(())
    }

    /// Snapshot every currently-materialized named value of this graph into a
    /// name → [`Tensor`] scope, layered on top of the inherited enclosing
    /// `outer_scope`. A nested control-flow body captures free names against
    /// this scope; local names shadow outer ones (ONNX lexical scoping). Built
    /// once per control-flow node execution (and reused across all iterations of
    /// a Loop/Scan), so captures are materialized once, not per iteration.
    fn materialize_scope(
        &self,
        resolved: &HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut scope = outer_scope.clone();
        for (name, &vid) in &self.name_index {
            if resolved.contains_key(&vid) && self.buffers.contains_key(&vid) {
                scope.insert(name.clone(), self.value_tensor(vid, resolved)?);
            }
        }
        Ok(scope)
    }

    /// Compile a control-flow body subgraph to a child [`Executor`], turning
    /// captured outer-scope names into extra graph inputs (so they are supplied
    /// and written every run), seeding the concrete external-input shapes, and
    /// running shape inference so the body's interior buffers can be sized.
    fn build_subgraph_exec(
        &self,
        key: &(NodeId, String),
        formal_names: &[String],
        capture_names: &[String],
        externals: &[Tensor],
    ) -> Result<CompiledSubgraph> {
        let body = self.graph.subgraphs.get(key).ok_or_else(|| {
            SessionError::Internal(format!(
                "control-flow node #{} has no registered subgraph '{}'",
                key.0 .0, key.1
            ))
        })?;
        let mut g = body.clone();

        // name → value id within the body.
        let mut body_names: HashMap<String, ValueId> = HashMap::new();
        for (vid, value) in g.values.iter() {
            if let Some(n) = &value.name {
                body_names.insert(n.clone(), vid);
            }
        }

        // Captures become graph inputs so they are required + written each run.
        for cname in capture_names {
            let vid = *body_names.get(cname).ok_or_else(|| {
                SessionError::Internal(format!(
                    "control-flow body '{}' lost capture value '{cname}'",
                    key.1
                ))
            })?;
            if !g.inputs.contains(&vid) {
                g.add_input(vid);
            }
        }

        // Seed each external input's concrete static shape + runtime dtype so
        // shape inference can flow through the body.
        let all_names = formal_names.iter().chain(capture_names.iter());
        for (name, tensor) in all_names.zip(externals.iter()) {
            let vid = *body_names.get(name).ok_or_else(|| {
                SessionError::Internal(format!(
                    "control-flow body '{}' missing formal/captured input '{name}'",
                    key.1
                ))
            })?;
            let v = g.value_mut(vid);
            v.dtype = tensor.dtype;
            v.shape = tensor.shape.iter().map(|&d| Dim::Static(d)).collect();
        }

        // Run shape inference over the seeded body (best-effort: interior shapes
        // that stay data-dependent are resolved just-in-time at run, exactly as
        // for the top-level graph).
        let registry = InferenceRegistry::default_registry();
        let opset_imports = self.graph.opset_imports.clone();
        registry.infer_graph(&mut g, &opset_imports, MergePolicy::Permissive)?;

        let exec = Executor::build(g, self._weights.clone(), self.ep.clone())?;
        Ok(CompiledSubgraph {
            exec,
            formal_inputs: formal_names.to_vec(),
            capture_inputs: capture_names.to_vec(),
            built_shapes: externals.iter().map(|t| t.shape.clone()).collect(),
        })
    }

    /// Run one control-flow body subgraph with `formal_inputs` bound positionally
    /// and outer-scope captures resolved from `scope`. Builds (or reuses) the
    /// cached child executor and returns the body's outputs in declared order.
    fn run_subgraph(
        &mut self,
        node_id: NodeId,
        attr_key: &str,
        formal_inputs: Vec<Tensor>,
        scope: &HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>> {
        let key = (node_id, attr_key.to_string());
        let body = self.graph.subgraphs.get(&key).ok_or_else(|| {
            SessionError::Internal(format!(
                "control-flow node #{} references missing subgraph '{attr_key}'",
                node_id.0
            ))
        })?;

        // Formal input names, in the body's declared input order.
        let formal_names: Vec<String> = body
            .inputs
            .iter()
            .map(|&vid| {
                body.value(vid)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("value#{}", vid.0))
            })
            .collect();
        if formal_names.len() != formal_inputs.len() {
            return Err(SessionError::Internal(format!(
                "control-flow body '{attr_key}' expects {} formal input(s) but {} were supplied",
                formal_names.len(),
                formal_inputs.len()
            )));
        }

        // Capture names: producer-less named values that are neither a formal
        // input nor a body initializer — free variables bound from outer scope.
        let formal_set: std::collections::HashSet<ValueId> = body.inputs.iter().copied().collect();
        let mut capture_names: Vec<String> = Vec::new();
        for (vid, value) in body.values.iter() {
            if value.producer.is_none()
                && !formal_set.contains(&vid)
                && !body.initializers.contains_key(&vid)
            {
                if let Some(name) = &value.name {
                    capture_names.push(name.clone());
                }
            }
        }
        // Deterministic order so the cached signature is stable.
        capture_names.sort();

        // Resolve captures from the enclosing scope.
        let mut captures: Vec<Tensor> = Vec::with_capacity(capture_names.len());
        for cname in &capture_names {
            let t = scope.get(cname).ok_or_else(|| SessionError::Internal(format!(
                "control-flow body '{attr_key}' captures free variable '{cname}', but it is not \
                 available in the enclosing scope. RULES #1: a subgraph may only reference outer \
                 values that are graph inputs, initializers, or produced by an upstream node in an \
                 enclosing graph; '{cname}' matches none of these"
            )))?;
            captures.push(t.clone());
        }

        // Externals in `formal ++ capture` order.
        let mut externals: Vec<Tensor> = formal_inputs;
        externals.extend(captures);
        let cur_shapes: Vec<Vec<usize>> = externals.iter().map(|t| t.shape.clone()).collect();

        // Rebuild the child only when its signature or input shapes change.
        let rebuild = match self.subgraph_execs.get(&key) {
            Some(cs) => {
                cs.formal_inputs != formal_names
                    || cs.capture_inputs != capture_names
                    || cs.built_shapes != cur_shapes
            }
            None => true,
        };
        if rebuild {
            let child = self.build_subgraph_exec(&key, &formal_names, &capture_names, &externals)?;
            self.subgraph_execs.insert(key.clone(), child);
        }

        let all_names: Vec<String> = formal_names.into_iter().chain(capture_names).collect();
        let inputs: Vec<(&str, &Tensor)> = all_names
            .iter()
            .map(|s| s.as_str())
            .zip(externals.iter())
            .collect();

        let cs = self.subgraph_execs.get_mut(&key).expect("child present");
        cs.exec.run_scoped(&inputs, scope)
    }

    /// Dispatch a control-flow plan node to its op-specific handler.
    fn exec_control_flow(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let node = self.graph.node(self.plan[pi].node_id).clone();
        match node.op_type.as_str() {
            "If" => self.exec_if(&node, resolved, outer_scope),
            "Loop" => self.exec_loop(&node, resolved, outer_scope),
            "Scan" => self.exec_scan(&node, resolved, outer_scope),
            other => Err(SessionError::Internal(format!(
                "exec_control_flow reached non-control-flow op {other:?}"
            ))),
        }
    }

    /// ONNX `If`: read the scalar `cond`, execute exactly one branch subgraph
    /// (0 formal inputs), and route the branch's outputs to `If`'s outputs.
    fn exec_if(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let cond_vid = node.inputs.first().and_then(|s| *s).ok_or_else(|| {
            SessionError::Internal("If node is missing its required 'cond' input".to_string())
        })?;
        let cond_t = self.value_tensor(cond_vid, resolved)?;
        let cond = tensor_scalar_bool(&cond_t).ok_or_else(|| SessionError::Internal(format!(
            "If: 'cond' must be a BOOL scalar, got dtype {:?} shape {:?}",
            cond_t.dtype, cond_t.shape
        )))?;

        let attr_key = if cond { "then_branch" } else { "else_branch" };
        let scope = self.materialize_scope(resolved, outer_scope)?;
        let outs = self.run_subgraph(node.id, attr_key, Vec::new(), &scope)?;

        if outs.len() != node.outputs.len() {
            return Err(SessionError::OutputShapeCountMismatch {
                op: format!("If/{attr_key}"),
                expected: node.outputs.len(),
                got: outs.len(),
            });
        }
        for (vid, t) in node.outputs.iter().zip(outs.iter()) {
            self.store_output_tensor(*vid, t, resolved)?;
        }
        Ok(())
    }

    /// ONNX `Loop`: inputs `[M?, cond?, v_initial...]`, body signature
    /// `(iter_num, cond_in, carried...) -> (cond_out, carried..., scan_out...)`.
    /// Iterates while `cond` is true and `iter < M`, threading loop-carried
    /// values across iterations and stacking each scan output along a new
    /// leading iteration axis.
    fn exec_loop(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Inputs: [M, cond, v_initial...]. M and cond may be omitted (None slot)
        // or an empty-name optional; absence means "unbounded" / "true".
        let m: Option<i64> = match node.inputs.first().and_then(|s| *s) {
            Some(vid) => {
                let t = self.value_tensor(vid, resolved)?;
                let m = tensor_scalar_i64(&t).ok_or_else(|| SessionError::Internal(format!(
                    "Loop: trip-count 'M' must be an INT64/INT32 scalar, got dtype {:?}",
                    t.dtype
                )))?;
                Some(m)
            }
            None => None,
        };
        let mut cond: Option<bool> = match node.inputs.get(1).and_then(|s| *s) {
            Some(vid) => {
                let t = self.value_tensor(vid, resolved)?;
                Some(tensor_scalar_bool(&t).ok_or_else(|| SessionError::Internal(format!(
                    "Loop: 'cond' must be a BOOL scalar, got dtype {:?}",
                    t.dtype
                )))?)
            }
            None => None,
        };

        // Initial loop-carried dependencies (inputs after M and cond).
        let mut carried: Vec<Tensor> = Vec::new();
        for slot in node.inputs.iter().skip(2) {
            let vid = slot.ok_or_else(|| SessionError::Internal(
                "Loop: an interior loop-carried input is omitted (empty), which ONNX does not \
                 allow — every v_initial must be provided".to_string(),
            ))?;
            carried.push(self.value_tensor(vid, resolved)?);
        }
        let num_carried = carried.len();
        // Loop outputs = carried finals ++ scan outputs. Scan-output count is
        // whatever remains after the carried finals.
        let num_outputs = node.outputs.len();
        if num_outputs < num_carried {
            return Err(SessionError::Internal(format!(
                "Loop: node declares {num_outputs} output(s) but has {num_carried} loop-carried \
                 dependency(ies); outputs must be carried-finals followed by scan-outputs"
            )));
        }
        let num_scan = num_outputs - num_carried;
        // Accumulators for scan outputs — one Vec of per-iteration slices each.
        let mut scan_acc: Vec<Vec<Tensor>> = vec![Vec::new(); num_scan];

        // Materialize the enclosing scope once; captures are constant across
        // iterations, so this is not repeated per iteration.
        let scope = self.materialize_scope(resolved, outer_scope)?;

        let mut iter: i64 = 0;
        loop {
            if let Some(m) = m {
                if iter >= m {
                    break;
                }
            }
            if cond == Some(false) {
                break;
            }

            // Body formal inputs: iter_num, cond_in, carried...
            let mut formal: Vec<Tensor> = Vec::with_capacity(2 + num_carried);
            formal.push(scalar_i64_tensor(iter)?);
            formal.push(scalar_bool_tensor(cond.unwrap_or(true))?);
            // Move the carried tensors in (avoids a deep copy each iteration);
            // they are replaced from the body outputs below.
            formal.extend(std::mem::take(&mut carried));

            let outs = self.run_subgraph(node.id, "body", formal, &scope)?;
            // Body outputs: cond_out, carried..., scan_out...
            let expected = 1 + num_carried + num_scan;
            if outs.len() != expected {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: "Loop/body".to_string(),
                    expected,
                    got: outs.len(),
                });
            }
            let mut it = outs.into_iter();
            let cond_out = it.next().expect("cond_out present");
            cond = Some(tensor_scalar_bool(&cond_out).ok_or_else(|| SessionError::Internal(
                format!(
                    "Loop: body's first output 'cond_out' must be a BOOL scalar, got dtype {:?}",
                    cond_out.dtype
                ),
            ))?);
            for _ in 0..num_carried {
                carried.push(it.next().expect("carried output present"));
            }
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"));
            }

            iter += 1;
        }

        // Emit outputs: carried finals, then stacked scan outputs.
        for (i, t) in carried.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved)?;
        }
        for (s, acc) in scan_acc.into_iter().enumerate() {
            let stacked = stack_new_leading_axis(&acc)?;
            self.store_output_tensor(node.outputs[num_carried + s], &stacked, resolved)?;
        }
        Ok(())
    }

    /// ONNX `Scan`: inputs `[initial_state..., scan_input...]`, body signature
    /// `(state..., scan_slice...) -> (state..., scan_out_slice...)`. Iterates
    /// over the scan axis, threading state and stacking scan outputs along their
    /// axis. Phase-1 scope: scan axis 0 for every scan input/output, forward
    /// direction (the common exporter output); anything else is rejected
    /// clearly rather than silently mis-scanned.
    fn exec_scan(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let num_scan_inputs = node
            .attr("num_scan_inputs")
            .and_then(|a| a.as_int())
            .ok_or_else(|| SessionError::Internal(
                "Scan: required attribute 'num_scan_inputs' is missing or not an INT".to_string(),
            ))? as usize;

        // Reject the axis/direction knobs this Phase-1 implementation does not
        // yet honor, rather than silently ignoring them (RULES #1/#5).
        for attr in ["scan_input_axes", "scan_output_axes"] {
            if let Some(a) = node.attr(attr) {
                if let Some(axes) = a.as_ints() {
                    if axes.iter().any(|&ax| ax != 0) {
                        return Err(SessionError::Internal(format!(
                            "Scan: attribute '{attr}' = {axes:?} requests a non-zero scan axis, \
                             which this runtime does not yet support. Expected axis 0 for every \
                             scan input/output; re-export with axis 0 or wait for full Scan-axis \
                             support"
                        )));
                    }
                }
            }
        }
        for attr in ["scan_input_directions", "scan_output_directions"] {
            if let Some(a) = node.attr(attr) {
                if let Some(dirs) = a.as_ints() {
                    if dirs.iter().any(|&d| d != 0) {
                        return Err(SessionError::Internal(format!(
                            "Scan: attribute '{attr}' = {dirs:?} requests reverse iteration, which \
                             this runtime does not yet support (forward only). Re-export forward or \
                             wait for reverse-Scan support"
                        )));
                    }
                }
            }
        }

        let total_inputs = node.inputs.len();
        if total_inputs < num_scan_inputs {
            return Err(SessionError::Internal(format!(
                "Scan: node has {total_inputs} input(s) but num_scan_inputs={num_scan_inputs}"
            )));
        }
        let num_state = total_inputs - num_scan_inputs;

        // Initial state (threaded across iterations) + scan inputs (sliced).
        let mut state: Vec<Tensor> = Vec::with_capacity(num_state);
        for slot in node.inputs.iter().take(num_state) {
            let vid = slot.ok_or_else(|| SessionError::Internal(
                "Scan: an initial-state input is omitted (empty), which ONNX does not allow"
                    .to_string(),
            ))?;
            state.push(self.value_tensor(vid, resolved)?);
        }
        let mut scan_inputs: Vec<Tensor> = Vec::with_capacity(num_scan_inputs);
        for slot in node.inputs.iter().skip(num_state) {
            let vid = slot.ok_or_else(|| SessionError::Internal(
                "Scan: a scan input is omitted (empty), which ONNX does not allow".to_string(),
            ))?;
            scan_inputs.push(self.value_tensor(vid, resolved)?);
        }

        // Sequence length = extent of scan axis 0; all scan inputs must agree.
        let seq_len = scan_inputs
            .first()
            .and_then(|t| t.shape.first().copied())
            .ok_or_else(|| SessionError::Internal(
                "Scan: requires at least one scan input with rank >= 1".to_string(),
            ))?;
        for (i, t) in scan_inputs.iter().enumerate() {
            let this = t.shape.first().copied().unwrap_or(0);
            if this != seq_len {
                return Err(SessionError::Internal(format!(
                    "Scan: scan input #{i} has scan-axis length {this} but the first scan input has \
                     {seq_len}; all scan inputs must share the same scan-axis length"
                )));
            }
        }

        let num_outputs = node.outputs.len();
        if num_outputs < num_state {
            return Err(SessionError::Internal(format!(
                "Scan: declares {num_outputs} output(s) but has {num_state} state variable(s); \
                 outputs must be final-state followed by scan-outputs"
            )));
        }
        let num_scan_out = num_outputs - num_state;
        let mut scan_acc: Vec<Vec<Tensor>> = vec![Vec::new(); num_scan_out];

        let scope = self.materialize_scope(resolved, outer_scope)?;

        for step in 0..seq_len {
            // Body formal inputs: state..., scan_slice...
            let mut formal: Vec<Tensor> = Vec::with_capacity(num_state + num_scan_inputs);
            formal.extend(std::mem::take(&mut state));
            for t in &scan_inputs {
                formal.push(slice_leading_axis(t, step)?);
            }

            let outs = self.run_subgraph(node.id, "body", formal, &scope)?;
            let expected = num_state + num_scan_out;
            if outs.len() != expected {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: "Scan/body".to_string(),
                    expected,
                    got: outs.len(),
                });
            }
            let mut it = outs.into_iter();
            for _ in 0..num_state {
                state.push(it.next().expect("state output present"));
            }
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"));
            }
        }

        for (i, t) in state.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved)?;
        }
        for (s, acc) in scan_acc.into_iter().enumerate() {
            let stacked = stack_new_leading_axis(&acc)?;
            self.store_output_tensor(node.outputs[num_state + s], &stacked, resolved)?;
        }
        Ok(())
    }
}

/// Extract the `index`-th slice along the leading axis of a contiguous tensor,
/// dropping that axis (shape `[d0, d1, ...] → [d1, ...]`). Used to feed one
/// Scan step's slice to the body. A single contiguous `memcpy` — the slice is a
/// contiguous sub-range because the tensor is row-major.
fn slice_leading_axis(t: &Tensor, index: usize) -> Result<Tensor> {
    if t.shape.is_empty() {
        return Err(SessionError::Internal(
            "Scan: cannot slice a scalar scan input along axis 0".to_string(),
        ));
    }
    let outer = t.shape[0];
    if index >= outer {
        return Err(SessionError::Internal(format!(
            "Scan: slice index {index} out of range for scan-axis length {outer}"
        )));
    }
    let inner_shape = t.shape[1..].to_vec();
    let inner_numel: usize = inner_shape.iter().product();
    let esize = t.dtype.byte_size();
    // Sub-byte dtypes have no clean per-element byte stride for slicing; reject.
    if esize == 0 {
        return Err(SessionError::Internal(format!(
            "Scan: sub-byte dtype {:?} scan inputs are not supported",
            t.dtype
        )));
    }
    let slice_bytes = inner_numel * esize;
    let start = index * slice_bytes;
    let bytes = &t.as_bytes()[start..start + slice_bytes];
    Tensor::from_raw(t.dtype, inner_shape, bytes)
}

/// Stack a list of identically-shaped tensors along a **new** leading axis
/// (shape `[s...] × n → [n, s...]`). Used to accumulate Loop/Scan scan outputs.
/// Pre-sizes the destination once and copies each slice contiguously — no
/// per-append reallocation.
fn stack_new_leading_axis(slices: &[Tensor]) -> Result<Tensor> {
    let n = slices.len();
    // A zero-trip loop/scan still produces a well-typed empty stack. Without any
    // slice we cannot know the element shape/dtype; ONNX leaves this shape
    // partially unknown. We emit a rank-1 empty tensor of the (unknown) element
    // type defaulting to Float32 — callers with zero trips and consumed scan
    // outputs are pathological; document rather than guess silently.
    if n == 0 {
        return Tensor::from_raw(DataType::Float32, vec![0], &[]);
    }
    let elem_shape = slices[0].shape.clone();
    let dtype = slices[0].dtype;
    let esize = dtype.byte_size();
    if esize == 0 {
        return Err(SessionError::Internal(format!(
            "Loop/Scan: sub-byte dtype {dtype:?} scan outputs are not supported"
        )));
    }
    let elem_numel: usize = elem_shape.iter().product();
    let elem_bytes = elem_numel * esize;
    let mut out = vec![0u8; n * elem_bytes];
    for (i, s) in slices.iter().enumerate() {
        if s.shape != elem_shape || s.dtype != dtype {
            return Err(SessionError::Internal(format!(
                "Loop/Scan: scan output slice {i} has shape {:?} dtype {:?} but the first slice is \
                 shape {elem_shape:?} dtype {dtype:?}; every iteration's scan output must match",
                s.shape, s.dtype
            )));
        }
        out[i * elem_bytes..(i + 1) * elem_bytes].copy_from_slice(s.as_bytes());
    }
    let mut shape = Vec::with_capacity(1 + elem_shape.len());
    shape.push(n);
    shape.extend(elem_shape);
    Tensor::from_raw(dtype, shape, &out)
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

        graph.opset_imports.insert(String::new(), 0);
        assert_eq!(effective_opset(&graph, &node), 0);

    }

    #[test]
    #[should_panic(expected = "internal invariant violated")]
    fn effective_opset_requires_validated_import() {
        effective_opset(
            &Graph::default(),
            &Node::new(NodeId(0), "Softmax", vec![], vec![]),
        );
    }
}
