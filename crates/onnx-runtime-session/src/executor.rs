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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cpu::strided::view_in_bounds;
use onnx_runtime_ir::{
    DataType, Dim, Graph, Node, NodeId, Shape, SymbolId, TensorLayout, ValueId, as_static_shape,
    compute_contiguous_strides,
};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

use crate::error::{Result, SessionError};
use crate::sequence::{concat_axis, split_axis, stack_new_axis, SeqTensor, SequenceValue};
use crate::tensor::Tensor;

fn profile_ops_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_PROFILE_OPS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

fn print_op_profile(total: Duration, timings: HashMap<String, (Duration, usize)>) {
    let mut timings = timings.into_iter().collect::<Vec<_>>();
    timings.sort_unstable_by(|left, right| right.1.0.cmp(&left.1.0));
    let total_ms = total.as_secs_f64() * 1_000.0;
    eprintln!("[onnx-genai-profile] node execution: {total_ms:.3} ms");
    eprintln!("[onnx-genai-profile] op_type,total_ms,percent,calls");
    for (op_type, (elapsed, calls)) in timings {
        let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
        let percent = if total_ms == 0.0 {
            0.0
        } else {
            elapsed_ms / total_ms * 100.0
        };
        eprintln!("[onnx-genai-profile] {op_type},{elapsed_ms:.3},{percent:.2},{calls}");
    }
}

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

/// Observable control-flow executor statistics. These counters make subgraph
/// reuse deterministic to test without relying on timing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlFlowStats {
    /// Child executors built, including shape-signature rebuilds.
    pub subgraph_builds: u64,
    /// Child subgraph invocations served by those executors.
    pub subgraph_runs: u64,
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
        constant_inputs: &[bool],
        opset: u64,
        ep: &dyn ExecutionProvider,
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
                return Err(SessionError::unsupported_op(
                    node,
                    node_id,
                    opset,
                    ep.name(),
                ));
            }
            let mut kernel = ep.get_kernel(node, input_shapes, opset)?;
            kernel.set_constant_inputs(constant_inputs);
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
    /// Kept alive so external-weight memory maps outlive buffer population —
    /// **and**, since the weight-streaming change, so borrowed initializer
    /// buffers that alias this store's mmap bytes stay valid for the executor's
    /// whole lifetime. `weights` MUST outlive every live use of `buffers`: a
    /// borrowed `DeviceBuffer` in `buffers` points into `weights`' mmap/inline
    /// storage. Teardown is safe because `Executor::drop` **drains and
    /// deallocates `buffers` first** (a borrowed deallocate is a no-op free), so
    /// no buffer still aliases `weights` when the `Arc<WeightStore>` field is
    /// dropped afterwards — no use-after-free regardless of field drop order.
    weights: Arc<WeightStore>,
    ep: Arc<dyn ExecutionProvider>,
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
    control_flow_stats: ControlFlowStats,
    /// Run-scoped zero-copy **view** metadata (§5.4). A value id present here is
    /// a strided view aliasing another value's buffer (a layout/movement-op
    /// output such as `Slice`) rather than an owner in [`Self::buffers`]. Built
    /// during the run loop and cleared at the start of every run.
    views: HashMap<ValueId, ValueView>,
    /// Run-scoped set of buffer-owning value ids that have ≥1 live view alias.
    /// A pinned buffer must not be reused or deallocated for the remainder of
    /// the run (conservative liveness: a source buffer outlives every view that
    /// aliases it, guaranteeing no use-after-free). Cleared each run.
    pinned: HashSet<ValueId>,
    /// Value ids whose runtime value is a **sequence of tensors** rather than a
    /// single tensor (produced by `SequenceEmpty`/`SequenceConstruct`/
    /// `SequenceInsert`/`SequenceErase`/`SplitToSequence`). Computed once at
    /// build; these values own no [`DeviceBuffer`] and are skipped by buffer
    /// sizing — their storage lives in [`Self::sequences`] at run time.
    sequence_values: HashSet<ValueId>,
    /// Run-scoped storage for sequence values: `value id → SequenceValue`. A
    /// [`SequenceValue`] holds its elements as `Arc`-shared immutable tensors,
    /// so a sequence op that inserts/erases/etc. shares element `Arc`s with the
    /// source rather than deep-copying bytes (see [`crate::sequence`] for the
    /// no-copy + no-race invariants). Cleared each run.
    sequences: HashMap<ValueId, SequenceValue>,
    /// Run-scoped **zero-copy** backing for a *tensor* value whose bytes are a
    /// shared sequence element (the output of `SequenceAt`): the tensor aliases
    /// the element's `Arc` instead of owning a `DeviceBuffer`, so no bytes are
    /// copied out of the sequence. A downstream kernel reads it through a
    /// [`TensorView`] over the `Arc`'s bytes; it is materialized to owned bytes
    /// only at the graph-output/control-flow boundary. Cleared each run.
    seq_elem_values: HashMap<ValueId, Arc<SeqTensor>>,
}

/// Run-scoped metadata for a zero-copy view value: it owns no buffer but
/// borrows `source`'s buffer with the given (real, possibly non-contiguous or
/// negative-strided) geometry. `strides`/`byte_offset` are expressed relative
/// to `source`'s allocation base, so a view-of-a-view is flattened to a single
/// hop whose `source` is always a real buffer owner (never itself a view).
#[derive(Clone, Debug)]
struct ValueView {
    source: ValueId,
    shape: Vec<usize>,
    strides: Vec<i64>,
    byte_offset: usize,
}

/// Per-input geometry the run loop resolves once per node: the raw base pointer
/// of the backing (root) buffer plus the real view (shape, element strides —
/// possibly non-contiguous or negative — and byte offset) to read it through.
/// A plain owned value yields contiguous strides at offset 0; a view value
/// yields its recorded strides/offset over its source buffer. `present` is false
/// for an omitted optional input (an absent placeholder).
struct InInfo {
    present: bool,
    dtype: DataType,
    shape: Vec<usize>,
    strides: Vec<i64>,
    byte_offset: usize,
    base_ptr: *const std::ffi::c_void,
    device: onnx_runtime_ir::DeviceId,
    /// Length in bytes of the backing (root) allocation, for the bounds gate.
    root_len: usize,
}

/// A cached child executor for one control-flow subgraph body, plus the
/// external-input shape signature it was compiled for (so a shape change forces
/// a rebuild rather than a silent shape mismatch).
struct CompiledSubgraph {
    exec: Executor,
    /// Ordered names of every external input: formal parameters first, then
    /// captured outer-scope values.
    input_names: Vec<String>,
    /// Concrete shapes the child was last compiled for, in `input_names` order.
    built_shapes: Vec<Vec<usize>>,
}

/// Invocation-invariant binding metadata for one selected subgraph. Loop/Scan
/// prepare this once outside the iteration loop, including one-time capture
/// materialization, then only rebind the changing formal tensors each step.
struct PreparedSubgraph {
    key: (NodeId, String),
    formal_names: Vec<String>,
    capture_names: Vec<String>,
    /// Direct captures plus transitive captures needed only by nested bodies.
    captures: HashMap<String, Tensor>,
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

/// Gather a strided view over `src` into a fresh contiguous row-major byte
/// buffer. `strides` are in **elements** (may be negative); `byte_offset` is the
/// byte position of the element origin within `src`. `esize` is the element
/// size in bytes (fixed-width types only — callers exclude sub-byte dtypes).
/// This is the materialization copy that turns a zero-copy view back into a
/// contiguous tensor for a strided-unaware consumer or the output boundary.
fn gather_view(
    src: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut out = vec![0u8; n * esize];
    if n == 0 {
        return out;
    }
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let mut w = 0usize;
    loop {
        let mut off = byte_offset as i64;
        for d in 0..rank {
            off += strides[d] * idx[d] as i64 * esize as i64;
        }
        let s = off as usize;
        out[w..w + esize].copy_from_slice(&src[s..s + esize]);
        w += esize;
        // Advance the row-major index; stop when it wraps to all-zero.
        let mut carried = true;
        for axis in (0..rank).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                carried = false;
                break;
            }
            idx[axis] = 0;
        }
        if carried {
            break;
        }
    }
    out
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
                });
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

/// Decode raw little-endian integer bytes as `i64` for `dtype`, or `None` if the
/// dtype is not an integer the shape math understands. Shared by the owned-buffer
/// and materialized-view integer-input readers.
fn bytes_as_i64(bytes: &[u8], dtype: DataType) -> Option<Vec<i64>> {
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
        "GroupQueryAttention" if node.domain == "com.microsoft" => {
            let query = input_shapes.first()?;
            let key = input_shapes.get(1)?;
            let past_key = input_shapes.get(3)?;
            if query.len() != 3 || key.len() != 3 || past_key.len() != 4 {
                return None;
            }
            let kv_heads = usize::try_from(node.attr("kv_num_heads")?.as_int()?).ok()?;
            if kv_heads == 0 || !key[2].is_multiple_of(kv_heads) {
                return None;
            }
            let head_dim = key[2] / kv_heads;
            let total_sequence_values = input_values.get(6)?.as_ref()?;
            if total_sequence_values.len() != 1 {
                return None;
            }
            let total_sequence = usize::try_from(total_sequence_values[0]).ok()?;
            let present_sequence = past_key[2].max(total_sequence);
            let present = vec![query[0], kv_heads, present_sequence, head_dim];
            let mut shapes = vec![query.clone()];
            if node.outputs.len() >= 2 {
                shapes.push(present.clone());
            }
            if node.outputs.len() >= 3 {
                shapes.push(present);
            }
            Some(shapes)
        }
        _ => None,
    }
}

/// Lower an exact `x * Sigmoid(x)` pair to the CPU EP's fused SiLU kernel.
///
/// The Sigmoid result must have exactly one consumer and must not be a graph
/// output, so removing its materialized value cannot change observable behavior.
fn fuse_silu_patterns(graph: &mut Graph) -> usize {
    let sigmoid_ids: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.op_type == "Sigmoid"
                && (node.domain.is_empty() || node.domain == "ai.onnx")
                && node.inputs.len() == 1
                && node.outputs.len() == 1)
                .then_some(id)
        })
        .collect();
    let mut fused = 0;

    for sigmoid_id in sigmoid_ids {
        let Some(sigmoid) = graph.try_node(sigmoid_id) else {
            continue;
        };
        let Some(x) = sigmoid.inputs[0] else {
            continue;
        };
        let sigmoid_output = sigmoid.outputs[0];
        if graph.outputs.contains(&sigmoid_output) {
            continue;
        }
        let consumers = graph.value(sigmoid_output).consumers.clone();
        if consumers.len() != 1 {
            continue;
        }
        let mul_id = consumers[0];
        let mul = graph.node(mul_id);
        if mul.op_type != "Mul"
            || !(mul.domain.is_empty() || mul.domain == "ai.onnx")
            || mul.inputs.len() != 2
            || mul.outputs.len() != 1
            || !((mul.inputs[0] == Some(x) && mul.inputs[1] == Some(sigmoid_output))
                || (mul.inputs[1] == Some(x) && mul.inputs[0] == Some(sigmoid_output)))
        {
            continue;
        }

        let mut silu = mul.clone();
        silu.op_type = "Silu".to_string();
        silu.domain = "com.microsoft".to_string();
        silu.inputs = vec![Some(x)];
        silu.attributes.clear();
        graph.replace_node(mul_id, silu);
        graph.remove_node(sigmoid_id);
        fused += 1;
    }

    if fused != 0 {
        graph
            .opset_imports
            .entry("com.microsoft".to_string())
            .or_insert(1);
    }
    fused
}

impl Executor {
    /// Compile a graph + weights into a runnable executor on the CPU EP.
    pub(crate) fn build(
        mut graph: Graph,
        weights: Arc<WeightStore>,
        ep: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        fuse_silu_patterns(&mut graph);
        // Topological order up front: also validates the graph is a DAG.
        let order = graph.topological_order()?;

        let mut value_shapes: HashMap<ValueId, Shape> = HashMap::new();
        let mut value_dtypes: HashMap<ValueId, DataType> = HashMap::new();
        let mut buffers: HashMap<ValueId, DeviceBuffer> = HashMap::new();
        let mut buffer_shapes: HashMap<ValueId, Vec<usize>> = HashMap::new();

        // 1) Initializers: always concrete. Record dims and back each with a
        //    device buffer. Where the mmap'd weight bytes are already suitably
        //    aligned we **borrow** them zero-copy (no RAM allocation, no copy)
        //    so a model whose weights exceed RAM still runs — the OS pages the
        //    mmap in/out on demand. Unaligned or empty slices fall back to the
        //    original allocate + copy path (correctness first).
        let init_align = TensorLayout::contiguous().alignment;
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!("weight bytes unavailable for value#{}", vid.0))
            })?;
            // Only borrow when the value has NO producer. The borrowed
            // `DeviceBuffer` aliases read-only mmap/inline storage, so it must
            // never be written. A legitimate initializer always has
            // `producer == None`; a malformed graph can reuse an initializer's
            // `ValueId` as a node output (see loader `validate_no_initializer_producer`),
            // giving it a producer — a kernel would then write through
            // `as_mut_ptr()` into read-only mmap (SIGSEGV / aliasing UB). In
            // that case fall back to the owned writable copy below.
            let producer_less = graph.value(vid).producer.is_none();
            let buf = if ep.device_id().is_host_accessible()
                && producer_less
                && !bytes.is_empty()
                && (bytes.as_ptr() as usize).is_multiple_of(init_align)
            {
                // Zero-copy: alias the aligned mmap bytes. `weights` (an owned
                // Arc field on this executor) outlives `buffers`, so the pointer
                // stays valid for every run; the borrowed buffer is read-only
                // (initializers are read-only SSA inputs, never a mutable output)
                // and its Drop/deallocate never frees the mmap.
                // SAFETY: `bytes` borrows `weights`' live mmap/inline storage,
                // is `bytes.len()` long and `init_align`-aligned (checked above),
                // is treated as read-only, and `weights` outlives every use.
                unsafe {
                    DeviceBuffer::from_borrowed_parts(
                        bytes.as_ptr() as *mut std::ffi::c_void,
                        ep.device_id(),
                        bytes.len(),
                        init_align,
                    )
                }
            } else {
                let mut owned = ep.allocate(bytes.len().max(1), init_align)?;
                ep.copy_from_host(bytes, &mut owned)?;
                owned
            };
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

        // Sequence-typed values own no tensor buffer: a Sequence op stores its
        // list in `sequences` at run time. Mark every value produced by a
        // sequence-producing op so buffer sizing skips it (and so a Sequence
        // graph output is diagnosed cleanly rather than read as tensor bytes).
        let mut sequence_values: HashSet<ValueId> = HashSet::new();
        for &nid in &order {
            let node = graph.node(nid);
            if produces_sequence_output(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    sequence_values.insert(out);
                }
            }
        }

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
            weights,
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
            control_flow_stats: ControlFlowStats::default(),
            views: HashMap::new(),
            pinned: HashSet::new(),
            sequence_values,
            sequences: HashMap::new(),
            seq_elem_values: HashMap::new(),
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
            // Sequence-typed values have no meaningful tensor shape and are
            // never buffer-sized; skip them so a static graph does not trip the
            // unresolved-shape check on a sequence value.
            if self.sequence_values.contains(&vid) {
                continue;
            }
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
            // Sequence-typed values own no tensor buffer (their list lives in
            // `sequences` at run time), so never size one for them.
            if self.sequence_values.contains(&vid) {
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
            // Sequence ops are executor-handled (they operate on sequence-of-
            // tensor values, not tensor views) — they have no EP kernel and must
            // not be compiled here, exactly like control-flow ops.
            if is_sequence_op(&node.op_type, &node.domain) {
                continue;
            }
            let input_shapes = Self::node_input_shapes(&self.plan[i], resolved);
            let constant_inputs: Vec<bool> = self.plan[i]
                .inputs
                .iter()
                .map(|input| input.is_some_and(|vid| self.graph.initializers.contains_key(&vid)))
                .collect();
            let node = self.graph.node(node_id);
            let opset = effective_opset(&self.graph, node);
            self.cache.get_or_create(
                node_id,
                node,
                &input_shapes,
                &constant_inputs,
                opset,
                self.ep.as_ref(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    pub(crate) fn control_flow_stats(&self) -> ControlFlowStats {
        self.control_flow_stats
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
        &self.weights
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
    fn bind_symbols(&self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<SymbolId, usize>> {
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
        // Zero-copy view metadata is run-scoped: a value that aliased another's
        // buffer last run must not leak into this one (buffers may be resized).
        self.views.clear();
        self.pinned.clear();
        // Sequence values and their zero-copy element-backed tensors are equally
        // run-scoped (element Arcs from a prior run must not leak in).
        self.sequences.clear();
        self.seq_elem_values.clear();

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
            self.ep.copy_from_host(tensor.as_bytes(), buf)?;
        }

        // --- Execute nodes ---------------------------------------------------
        // Iterate by index so a control-flow node can take `&mut self` (it must
        // build/reuse child executors) while an ordinary kernel node uses the
        // disjoint-field borrow split inside `exec_kernel_node`.
        if profile_ops_enabled() {
            let run_start = Instant::now();
            let mut timings: HashMap<String, (Duration, usize)> = HashMap::new();
            for pi in 0..self.plan.len() {
                let node_id = self.plan[pi].node_id;
                let node = self.graph.node(node_id);
                let op_type = node.op_type.clone();
                let start = Instant::now();
                let result = if is_control_flow_op(&node.op_type, &node.domain) {
                    self.exec_control_flow(pi, &mut resolved, outer_scope)
                } else if is_sequence_op(&node.op_type, &node.domain) {
                    self.exec_sequence_node(pi, &mut resolved)
                } else {
                    self.exec_kernel_node(pi, &mut resolved)
                };
                let elapsed = start.elapsed();
                let entry = timings.entry(op_type).or_insert((Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
                result?;
            }
            print_op_profile(run_start.elapsed(), timings);
        } else {
            for pi in 0..self.plan.len() {
                let node_id = self.plan[pi].node_id;
                let node = self.graph.node(node_id);
                if is_control_flow_op(&node.op_type, &node.domain) {
                    self.exec_control_flow(pi, &mut resolved, outer_scope)?;
                } else if is_sequence_op(&node.op_type, &node.domain) {
                    self.exec_sequence_node(pi, &mut resolved)?;
                } else {
                    self.exec_kernel_node(pi, &mut resolved)?;
                }
            }
        }

        // --- Collect graph outputs into owned tensors -----------------------
        // A view output (a layout op whose result aliases an input buffer) is
        // materialized to contiguous owned bytes here — external consumers and
        // the Python/DLPack boundary expect contiguous tensors.
        let mut results = Vec::with_capacity(self.graph.outputs.len());
        for &vid in &self.graph.outputs {
            // A Sequence value cannot be returned through the tensor-typed
            // `run` boundary. Diagnose it clearly instead of misreading it as
            // tensor bytes; consumers extract tensors via SequenceAt /
            // ConcatFromSequence before the graph output.
            if self.sequence_values.contains(&vid) {
                let name = self
                    .graph
                    .try_value(vid)
                    .and_then(|v| v.name.clone())
                    .unwrap_or_else(|| format!("value#{}", vid.0));
                return Err(SessionError::SequenceOp {
                    op: "<graph output>".to_string(),
                    reason: format!(
                        "graph output {name} is a Sequence value, which cannot be \
                         returned through the tensor `run` API. To fix: end the graph \
                         with ConcatFromSequence or SequenceAt to produce tensor \
                         output(s)"
                    ),
                });
            }

            let dtype = self.value_dtypes[&vid];
            let shape = resolved[&vid].clone();
            let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
            results.push(Tensor::from_raw(dtype, shape, &bytes)?);
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
        // Small owned copies of the plan facts so the buffer/view/cache fields
        // can be mutated below without fighting a borrow of `self.plan`.
        let node_id = self.plan[pi].node_id;
        let inputs = self.plan[pi].inputs.clone();
        let outputs = self.plan[pi].outputs.clone();
        let input_dtypes = self.plan[pi].input_dtypes.clone();
        let output_dtypes = self.plan[pi].output_dtypes.clone();

        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|v| v.map(|vid| resolved[&vid].clone()).unwrap_or_default())
            .collect();

        // Data-dependent shapes: if any output's shape is still unresolved,
        // compute it now from the concrete input shapes + the runtime *values*
        // of this node's integer inputs. Buffers are NOT sized here — a view
        // output needs none, and the compute path sizes them just below.
        if outputs.iter().any(|v| !resolved.contains_key(v)) {
            let input_values: Vec<Option<Vec<i64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.and_then(|vid| self.input_i64(vid, &input_shapes[i], input_dtypes[i]))
                })
                .collect();
            let node = self.graph.node(node_id);
            let out_shapes =
                dynamic_output_shapes(node, &input_shapes, &input_values).ok_or_else(|| {
                    let vid = outputs
                        .iter()
                        .find(|v| !resolved.contains_key(v))
                        .copied()
                        .unwrap_or(outputs[0]);
                    let value = self.graph.value(vid);
                    SessionError::UnresolvedShape {
                        value: value
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("value#{}", vid.0)),
                        op: node.op_type.clone(),
                    }
                })?;
            if out_shapes.len() != outputs.len() {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: self.graph.node(node_id).op_type.clone(),
                    expected: outputs.len(),
                    got: out_shapes.len(),
                });
            }
            for (oi, &ovid) in outputs.iter().enumerate() {
                resolved.insert(ovid, out_shapes[oi].clone());
            }
        }

        let output_shapes: Vec<Vec<usize>> = outputs.iter().map(|v| resolved[v].clone()).collect();

        // Resolve each input's real geometry (root buffer + strides/offset) and
        // bounds-check it. View inputs read through their recorded strides.
        let mut in_infos: Vec<InInfo> = Vec::with_capacity(inputs.len());
        for (i, slot) in inputs.iter().enumerate() {
            let Some(vid) = *slot else {
                in_infos.push(InInfo {
                    present: false,
                    dtype: input_dtypes[i],
                    shape: Vec::new(),
                    strides: Vec::new(),
                    byte_offset: 0,
                    base_ptr: std::ptr::null(),
                    device: self.ep.device_id(),
                    root_len: 0,
                });
                continue;
            };
            // A tensor input backed by a shared sequence element (SequenceAt
            // output) owns no DeviceBuffer: read it zero-copy through a
            // contiguous view over the element's immutable `Arc` bytes. The Arc
            // is held live in `self.seq_elem_values` for the whole run, so the
            // pointer stays valid across this kernel dispatch.
            if let Some(elem) = self.seq_elem_values.get(&vid) {
                let shape = input_shapes[i].clone();
                let strides = compute_contiguous_strides(&shape);
                let root_len = elem.data.len();
                let base_ptr = elem.as_ptr() as *const std::ffi::c_void;
                view_bounds(&shape, &strides, 0, input_dtypes[i], root_len)?;
                in_infos.push(InInfo {
                    present: true,
                    dtype: input_dtypes[i],
                    shape,
                    strides,
                    byte_offset: 0,
                    base_ptr,
                    device: onnx_runtime_ir::DeviceId::cpu(),
                    root_len,
                });
                continue;
            }
            let root = self.root_of(vid);
            let buf = self.buffers.get(&root).ok_or_else(|| {
                SessionError::Internal(format!("missing buffer for input value#{}", vid.0))
            })?;
            let root_len = buf.len();
            let base_ptr = buf.as_ptr();
            let (shape, strides, byte_offset) = match self.views.get(&vid) {
                Some(view) => (view.shape.clone(), view.strides.clone(), view.byte_offset),
                None => {
                    let shape = input_shapes[i].clone();
                    let strides = compute_contiguous_strides(&shape);
                    (shape, strides, 0)
                }
            };
            view_bounds(&shape, &strides, byte_offset, input_dtypes[i], root_len)?;
            in_infos.push(InInfo {
                present: true,
                dtype: input_dtypes[i],
                shape,
                strides,
                byte_offset,
                base_ptr,
                device: buf.device(),
                root_len,
            });
        }

        let ep = self.ep.clone();

        // Bind the mutated fields as disjoint locals so `self` is never borrowed
        // whole while the kernel (from `cache`) and the buffers/views are held.
        let graph = &self.graph;
        let cache = &mut self.cache;
        let buffers = &mut self.buffers;
        let buffer_shapes = &mut self.buffer_shapes;
        let views_meta = &mut self.views;
        let pinned = &mut self.pinned;

        // Build the (possibly strided) input views once; they feed both the
        // view-output probe and, on the compute path, the kernel itself.
        let mut views: Vec<TensorView> = Vec::with_capacity(in_infos.len());
        for info in &in_infos {
            if !info.present {
                views.push(TensorView::absent(info.dtype));
                continue;
            }
            views.push(
                TensorView::new(
                    DevicePtr(info.base_ptr),
                    info.dtype,
                    &info.shape,
                    &info.strides,
                    info.device,
                )
                .with_byte_offset(info.byte_offset),
            );
        }

        let node = graph.node(node_id);
        let opset = effective_opset(graph, node);
        let constant_inputs: Vec<bool> = inputs
            .iter()
            .map(|input| input.is_some_and(|vid| graph.initializers.contains_key(&vid)))
            .collect();
        let kernel = cache.get_or_create(
            node_id,
            node,
            &input_shapes,
            &constant_inputs,
            opset,
            ep.as_ref(),
        )?;

        // --- Zero-copy view fast path ---------------------------------------
        // Ask the kernel whether its outputs are strided views over its inputs
        // (a layout/movement op such as Slice). If so, record view metadata
        // aliasing the source buffer and skip compute + allocation entirely.
        if let Some(specs) = kernel.view_outputs(&views, outputs.len()) {
            drop(views);
            if specs.len() != outputs.len() {
                return Err(SessionError::Internal(format!(
                    "op '{}' returned {} view outputs for {} outputs",
                    node.op_type,
                    specs.len(),
                    outputs.len()
                )));
            }
            for (oi, spec) in specs.into_iter().enumerate() {
                let ovid = outputs[oi];
                let Some(in_vid) = inputs.get(spec.input_index).copied().flatten() else {
                    return Err(SessionError::Internal(format!(
                        "op '{}' view output {} references invalid input index {}",
                        node.op_type, oi, spec.input_index
                    )));
                };
                let root = match views_meta.get(&in_vid) {
                    Some(v) => v.source,
                    None => in_vid,
                };
                let root_len = buffers.get(&root).map(|b| b.len()).ok_or_else(|| {
                    SessionError::Internal(format!("view source value#{} has no buffer", root.0))
                })?;
                // Bounds-gate the composed view against the source allocation.
                view_bounds(
                    &spec.shape,
                    &spec.strides,
                    spec.byte_offset,
                    output_dtypes[oi],
                    root_len,
                )?;
                // The output becomes a view: drop any buffer it used to own so a
                // later run re-sizes cleanly, then record the alias and pin the
                // source (conservative liveness — a source with any live view is
                // never reused/freed for the rest of the run; no use-after-free).
                // A freshly-produced output can never already be pinned (its
                // viewers run strictly after it under SSA topo order).
                debug_assert!(
                    !pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being reproduced",
                    ovid.0
                );
                if let Some(old) = buffers.remove(&ovid) {
                    ep.deallocate(old)?;
                }
                buffer_shapes.remove(&ovid);
                views_meta.insert(
                    ovid,
                    ValueView {
                        source: root,
                        shape: spec.shape.clone(),
                        strides: spec.strides,
                        byte_offset: spec.byte_offset,
                    },
                );
                pinned.insert(root);
                resolved.insert(ovid, spec.shape);
            }
            return Ok(());
        }

        // --- Compute path ----------------------------------------------------
        // Size (allocate or reuse) each output's contiguous buffer, JIT-sizing
        // data-dependent ones. A value that was a view on a prior run has no
        // buffer here and is freshly allocated.
        for (oi, &ovid) in outputs.iter().enumerate() {
            let dims = &output_shapes[oi];
            let numel = checked_numel(dims, || format!("value#{}", ovid.0))?;
            let need = checked_storage_bytes(
                output_dtypes[oi],
                numel,
                || format!("value#{}", ovid.0),
                dims,
            )?
            .max(1);
            let fits = buffers.get(&ovid).map(|b| b.len() == need).unwrap_or(false);
            if !fits {
                // Never free a buffer that has a live view alias (would dangle
                // the viewer). Unreachable under SSA topo order, but enforced.
                debug_assert!(
                    !pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being resized",
                    ovid.0
                );
                if let Some(old) = buffers.remove(&ovid) {
                    ep.deallocate(old)?;
                }
                let buf = ep.allocate(need, TensorLayout::contiguous().alignment)?;
                buffers.insert(ovid, buf);
            }
        }

        // Auto-materialization gate: a strided (view) input feeding a kernel
        // that does not accept strided input on that slot is gathered into a
        // private contiguous temp so contiguous-assuming kernels stay correct.
        // Temps must outlive the views that borrow them.
        let mut mat: Vec<Option<(Vec<u8>, Vec<i64>)>> = Vec::with_capacity(in_infos.len());
        for (i, info) in in_infos.iter().enumerate() {
            if !info.present {
                mat.push(None);
                continue;
            }
            let contiguous = onnx_runtime_ir::is_contiguous(&info.shape, &info.strides);
            if contiguous || kernel.supports_strided_input(i) {
                mat.push(None);
                continue;
            }
            if !info.device.is_host_accessible() {
                return Err(SessionError::Internal(format!(
                    "op '{}' requires host-only strided materialization for CUDA input {i}",
                    node.op_type
                )));
            }
            let esize = info.dtype.byte_size();
            if esize == 0 {
                return Err(SessionError::from(
                    onnx_runtime_ep_api::EpError::InvalidTensorView {
                        reason: format!(
                            "cannot materialize sub-byte strided input {i} of op '{}'",
                            node.op_type
                        ),
                    },
                ));
            }
            let src =
                unsafe { std::slice::from_raw_parts(info.base_ptr as *const u8, info.root_len) };
            let gathered = gather_view(src, &info.shape, &info.strides, info.byte_offset, esize);
            let strides = compute_contiguous_strides(&info.shape);
            mat.push(Some((gathered, strides)));
        }

        // Rebuild input views, swapping any materialized slot to its contiguous
        // temp (offset 0, contiguous strides over the fresh buffer).
        drop(views);
        let mut views: Vec<TensorView> = Vec::with_capacity(in_infos.len());
        for (i, info) in in_infos.iter().enumerate() {
            if !info.present {
                views.push(TensorView::absent(info.dtype));
                continue;
            }
            match &mat[i] {
                Some((buf, strides)) => views.push(TensorView::new(
                    DevicePtr(buf.as_ptr() as *const std::ffi::c_void),
                    info.dtype,
                    &info.shape,
                    strides,
                    onnx_runtime_ir::DeviceId::cpu(),
                )),
                None => views.push(
                    TensorView::new(
                        DevicePtr(info.base_ptr),
                        info.dtype,
                        &info.shape,
                        &info.strides,
                        info.device,
                    )
                    .with_byte_offset(info.byte_offset),
                ),
            }
        }

        // Take output buffers out so they can be borrowed `&mut` disjointly from
        // the input reads (SSA guarantees outputs are disjoint from inputs).
        let out_strides: Vec<Vec<i64>> = output_shapes
            .iter()
            .map(|s| compute_contiguous_strides(s))
            .collect();
        let mut out_bufs: Vec<(ValueId, DeviceBuffer)> = Vec::with_capacity(outputs.len());
        for &vid in &outputs {
            let buf = buffers.remove(&vid).ok_or_else(|| {
                SessionError::Internal(format!("missing buffer for output value#{}", vid.0))
            })?;
            out_bufs.push((vid, buf));
        }
        let mut outs: Vec<TensorMut> = Vec::with_capacity(out_bufs.len());
        for (i, (_, buf)) in out_bufs.iter_mut().enumerate() {
            view_bounds(
                &output_shapes[i],
                &out_strides[i],
                0,
                output_dtypes[i],
                buf.len(),
            )?;
            let ptr = buf.as_mut_ptr();
            outs.push(TensorMut::new(
                DevicePtrMut(ptr),
                output_dtypes[i],
                &output_shapes[i],
                &out_strides[i],
                buf.device(),
            ));
        }

        kernel.execute(&views, &mut outs).map_err(|error| {
            let input_types = views.iter().map(|view| view.dtype).collect::<Vec<_>>();
            let output_types = outs.iter().map(|output| output.dtype).collect::<Vec<_>>();
            let input_shapes = views
                .iter()
                .map(|view| view.shape.to_vec())
                .collect::<Vec<_>>();
            let output_shapes = outs
                .iter()
                .map(|output| output.shape.to_vec())
                .collect::<Vec<_>>();
            let input_names = inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| self.graph.value(value).name.as_deref().unwrap_or("<unnamed>"))
                        .unwrap_or("<absent>")
                })
                .collect::<Vec<_>>();
            let output_names = outputs
                .iter()
                .map(|&value| self.graph.value(value).name.as_deref().unwrap_or("<unnamed>"))
                .collect::<Vec<_>>();
            SessionError::Internal(format!(
                "node {} ({:?}, op '{}::{}', inputs {input_names:?} {input_types:?} {input_shapes:?}, outputs {output_names:?} {output_types:?} {output_shapes:?}) failed: {error}",
                node.id.0, node.name, node.domain, node.op_type,
            ))
        })?;

        drop(views);
        drop(outs);
        for (vid, buf) in out_bufs {
            buffers.insert(vid, buf);
        }
        Ok(())
    }

    /// Read the integer *values* of input `vid` as `i64`, materializing a view
    /// first if needed. Used to resolve data-dependent output shapes (e.g. a
    /// `Slice` whose `ends` is produced at runtime). Returns `None` if the value
    /// has no readable buffer/view or its dtype is not an integer.
    fn input_i64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<i64>> {
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_i64(&bytes, dtype)
    }
}

// === Sequence-of-tensors ops: SequenceEmpty / SequenceConstruct /
// SequenceInsert / SequenceErase / SequenceAt / SequenceLength /
// SplitToSequence / ConcatFromSequence ===
//
// These are handled at the executor level (like control-flow ops) rather than as
// leaf kernels, because they operate on a *sequence-of-tensors* runtime value
// that a `Kernel` — which sees only individual tensor views — cannot represent.
//
// ## No-copy design
//
// A sequence stores its elements as `Arc`-shared **immutable** [`SeqTensor`]s
// (see [`crate::sequence`]). Insert/Erase/Construct build a NEW list that SHARES
// the surviving element `Arc`s — only handles (a refcount bump), never element
// bytes, are cloned. `SequenceAt` yields the shared element `Arc` and backs its
// output tensor value with that same allocation (`seq_elem_values`), so a
// downstream kernel reads it through a zero-copy [`TensorView`] and no bytes are
// copied out of the sequence until the graph-output boundary. The only copies
// are unavoidable boundary crossings: a *tensor → sequence* entry (a produced
// `DeviceBuffer`, reused across runs, cannot be aliased so its bytes are moved
// into the element `Arc` exactly once) and the single-alloc `Split`/`Concat`
// data movement.
//
// ## No-race design
//
// Elements are immutable after construction and only ever shared read-only
// through `Arc`; there is no interior mutability, so concurrent readers cannot
// race (the only cross-thread state is `Arc`'s atomic refcount).
impl Executor {
    /// Execute one Sequence-op plan node.
    fn exec_sequence_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let node_id = self.plan[pi].node_id;
        let inputs = self.plan[pi].inputs.clone();
        let outputs = self.plan[pi].outputs.clone();
        let op = self.graph.node(node_id).op_type.clone();

        match op.as_str() {
            "SequenceEmpty" => {
                let dtype_attr = self
                    .graph
                    .node(node_id)
                    .attr("dtype")
                    .and_then(|a| a.as_int());
                let dtype = match dtype_attr {
                    None => DataType::Float32, // ONNX default element type.
                    Some(raw) => {
                        DataType::from_onnx(raw as i32).ok_or_else(|| SessionError::SequenceOp {
                            op: op.clone(),
                            reason: format!(
                                "attribute 'dtype' = {raw} is not a known ONNX \
                                 TensorProto.DataType. To fix: use a valid element \
                                 dtype id (e.g. 1=float32, 7=int64)"
                            ),
                        })?
                    }
                };
                self.sequences
                    .insert(outputs[0], SequenceValue::empty(dtype));
                Ok(())
            }
            "SequenceConstruct" => {
                let mut items = Vec::with_capacity(inputs.len());
                for slot in &inputs {
                    let vid = slot.ok_or_else(|| self.seq_missing_input(&op))?;
                    items.push(self.read_seq_element(vid, resolved)?);
                }
                let seq = SequenceValue::construct(items).map_err(seq_err)?;
                self.sequences.insert(outputs[0], seq);
                Ok(())
            }
            "SequenceInsert" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let tvid = inputs
                    .get(1)
                    .copied()
                    .flatten()
                    .ok_or_else(|| self.seq_missing_input(&op))?;
                let tensor = self.read_seq_element(tvid, resolved)?;
                let position = match inputs.get(2).copied().flatten() {
                    Some(pvid) => Some(self.read_scalar_i64(pvid, resolved, &op)?),
                    None => None,
                };
                let out = seq.insert(tensor, position).map_err(seq_err)?;
                self.sequences.insert(outputs[0], out);
                Ok(())
            }
            "SequenceErase" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let position = match inputs.get(1).copied().flatten() {
                    Some(pvid) => Some(self.read_scalar_i64(pvid, resolved, &op)?),
                    None => None,
                };
                let out = seq.erase(position).map_err(seq_err)?;
                self.sequences.insert(outputs[0], out);
                Ok(())
            }
            "SequenceAt" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let pvid =
                    inputs
                        .get(1)
                        .copied()
                        .flatten()
                        .ok_or_else(|| SessionError::SequenceOp {
                            op: op.clone(),
                            reason: "requires a 'position' input. To fix: supply the \
                                 index tensor of the element to read"
                                .to_string(),
                        })?;
                let pos = self.read_scalar_i64(pvid, resolved, &op)?;
                let elem = seq.at(pos).map_err(seq_err)?;
                self.store_seq_element_output(outputs[0], elem, resolved)
            }
            "SequenceLength" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let len = seq.len() as i64;
                self.store_raw_tensor_output(
                    outputs[0],
                    DataType::Int64,
                    Vec::new(),
                    &len.to_le_bytes(),
                    resolved,
                )
            }
            "SplitToSequence" => self.exec_split_to_sequence(&op, &inputs, &outputs, resolved),
            "ConcatFromSequence" => {
                self.exec_concat_from_sequence(node_id, &op, &inputs, &outputs, resolved)
            }
            other => Err(SessionError::SequenceOp {
                op: other.to_string(),
                reason: "unrecognized Sequence op (executor routing bug)".to_string(),
            }),
        }
    }

    /// `SplitToSequence`: split a tensor into a sequence along `axis`.
    fn exec_split_to_sequence(
        &mut self,
        op: &str,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let node = self.graph.node(self.plan_node_of(outputs[0]));
        let axis_attr = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0);
        let keepdims = node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0;

        let ivid = inputs
            .first()
            .copied()
            .flatten()
            .ok_or_else(|| self.seq_missing_input(op))?;
        let dtype = self.value_dtypes[&ivid];
        let esize = dtype.byte_size();
        if esize == 0 {
            return Err(SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "sub-byte dtype {dtype:?} is not supported for SplitToSequence. \
                     To fix: Cast to a byte-addressable dtype before splitting"
                ),
            });
        }
        let shape = resolved
            .get(&ivid)
            .cloned()
            .ok_or_else(|| self.seq_unresolved(op, ivid))?;
        let rank = shape.len();
        if rank == 0 {
            return Err(SessionError::SequenceOp {
                op: op.to_string(),
                reason: "cannot split a scalar (rank-0) tensor. To fix: split a \
                         tensor with at least one dimension"
                    .to_string(),
            });
        }
        let axis = normalize_axis(axis_attr, rank).ok_or_else(|| SessionError::SequenceOp {
            op: op.to_string(),
            reason: format!(
                "attribute 'axis' = {axis_attr} is out of range for a rank-{rank} \
                 input (valid range is [{}, {}])",
                -(rank as i64),
                rank as i64 - 1
            ),
        })?;
        let axis_dim = shape[axis];
        let bytes = self.contiguous_bytes(ivid, &shape, dtype)?;

        // Determine per-chunk sizes and whether to squeeze the split axis.
        let mut squeeze = false;
        let sizes: Vec<usize> = match inputs.get(1).copied().flatten() {
            None => {
                // No 'split': one element per index along axis; keepdims=0 drops
                // the (size-1) axis from each element.
                squeeze = !keepdims;
                vec![1; axis_dim]
            }
            Some(svid) => {
                let sshape = resolved.get(&svid).cloned().unwrap_or_default();
                let svals = self.read_i64_vec(svid, &sshape, op)?;
                let is_scalar = sshape.is_empty();
                if is_scalar {
                    let chunk = *svals.first().ok_or_else(|| SessionError::SequenceOp {
                        op: op.to_string(),
                        reason: "'split' scalar is empty".to_string(),
                    })?;
                    if chunk <= 0 {
                        return Err(SessionError::SequenceOp {
                            op: op.to_string(),
                            reason: format!("'split' chunk size {chunk} must be positive"),
                        });
                    }
                    let chunk = chunk as usize;
                    let mut v = Vec::new();
                    let mut rem = axis_dim;
                    while rem > 0 {
                        let k = rem.min(chunk);
                        v.push(k);
                        rem -= k;
                    }
                    v
                } else {
                    let v: Vec<usize> = svals.iter().map(|&x| x.max(0) as usize).collect();
                    let sum: usize = v.iter().sum();
                    if sum != axis_dim {
                        return Err(SessionError::SequenceOp {
                            op: op.to_string(),
                            reason: format!(
                                "'split' sizes {v:?} sum to {sum} but axis {axis} has \
                                 extent {axis_dim}. To fix: make the split sizes sum to \
                                 the axis length"
                            ),
                        });
                    }
                    v
                }
            }
        };

        let parts = split_axis(&bytes, &shape, axis, &sizes, esize);
        let items: Vec<std::sync::Arc<SeqTensor>> = parts
            .into_iter()
            .map(|(mut sh, data)| {
                if squeeze {
                    sh.remove(axis);
                }
                SeqTensor::shared(dtype, sh, data)
            })
            .collect();
        self.sequences.insert(
            outputs[0],
            SequenceValue {
                elem_dtype: dtype,
                items,
            },
        );
        Ok(())
    }

    /// `ConcatFromSequence`: concatenate (or stack, when `new_axis=1`) a
    /// sequence's tensors into one freshly-allocated output.
    fn exec_concat_from_sequence(
        &mut self,
        node_id: NodeId,
        op: &str,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let node = self.graph.node(node_id);
        let axis_attr =
            node.attr("axis")
                .and_then(|a| a.as_int())
                .ok_or_else(|| SessionError::SequenceOp {
                    op: op.to_string(),
                    reason: "requires the mandatory 'axis' attribute. To fix: set 'axis'"
                        .to_string(),
                })?;
        let new_axis = node.attr("new_axis").and_then(|a| a.as_int()).unwrap_or(0) != 0;

        let seq = self.get_sequence(inputs.first().copied().flatten(), op)?;
        if seq.is_empty() {
            return Err(SessionError::SequenceOp {
                op: op.to_string(),
                reason: "cannot concatenate an empty sequence (output shape is \
                         undefined). To fix: guard with SequenceLength"
                    .to_string(),
            });
        }
        let dtype = seq.elem_dtype;
        let esize = dtype.byte_size();
        if esize == 0 {
            return Err(SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!("sub-byte dtype {dtype:?} is not supported for ConcatFromSequence"),
            });
        }
        let elem_shapes: Vec<Vec<usize>> = seq.items.iter().map(|t| t.shape.clone()).collect();
        let elem_datas: Vec<&[u8]> = seq.items.iter().map(|t| t.data.as_slice()).collect();
        let rank = elem_shapes[0].len();

        let (oshape, out) = if new_axis {
            // Stack: every element must share one shape; new axis in [-rank-1, rank].
            for (i, s) in elem_shapes.iter().enumerate() {
                if s != &elem_shapes[0] {
                    return Err(SessionError::SequenceOp {
                        op: op.to_string(),
                        reason: format!(
                            "ConcatFromSequence(new_axis=1) requires identical element \
                             shapes, but element {i} has shape {s:?} vs {:?}",
                            elem_shapes[0]
                        ),
                    });
                }
            }
            let axis =
                normalize_axis(axis_attr, rank + 1).ok_or_else(|| SessionError::SequenceOp {
                    op: op.to_string(),
                    reason: format!(
                        "'axis' = {axis_attr} is out of range for new_axis=1 stacking \
                         of rank-{rank} elements (valid range is [{}, {}])",
                        -(rank as i64) - 1,
                        rank as i64
                    ),
                })?;
            stack_new_axis(&elem_datas, &elem_shapes[0], axis, esize)
        } else {
            let axis = normalize_axis(axis_attr, rank).ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "'axis' = {axis_attr} is out of range for rank-{rank} elements \
                     (valid range is [{}, {}])",
                    -(rank as i64),
                    rank as i64 - 1
                ),
            })?;
            // Elements must agree on every dimension except the concat axis.
            for (i, s) in elem_shapes.iter().enumerate() {
                let mismatch = s.len() != rank
                    || s.iter()
                        .enumerate()
                        .any(|(d, &v)| d != axis && v != elem_shapes[0][d]);
                if mismatch {
                    return Err(SessionError::SequenceOp {
                        op: op.to_string(),
                        reason: format!(
                            "ConcatFromSequence requires elements to match on all axes \
                             except {axis}, but element {i} has shape {s:?} vs {:?}",
                            elem_shapes[0]
                        ),
                    });
                }
            }
            concat_axis(&elem_datas, &elem_shapes, axis, esize)
        };
        drop(seq);
        self.store_raw_tensor_output(outputs[0], dtype, oshape, &out, resolved)
    }

    /// Build (or share) an `Arc<SeqTensor>` for a tensor value entering a
    /// sequence. If the value is already a shared sequence element (a
    /// `SequenceAt` result), its `Arc` is **shared** with no copy; otherwise its
    /// contiguous bytes are moved into a fresh element once (the tensor→sequence
    /// entry boundary).
    fn read_seq_element(
        &self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<std::sync::Arc<SeqTensor>> {
        if let Some(elem) = self.seq_elem_values.get(&vid) {
            return Ok(std::sync::Arc::clone(elem)); // zero-copy share
        }
        let dtype = self.value_dtypes[&vid];
        let shape = resolved
            .get(&vid)
            .cloned()
            .ok_or_else(|| self.seq_unresolved("Sequence", vid))?;
        let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
        Ok(SeqTensor::shared(dtype, shape, bytes))
    }

    /// Fetch (clone) the sequence value bound to `vid` (cheap — `Arc` handle
    /// clones), or an actionable error if the input is missing / not a sequence.
    fn get_sequence(&self, vid: Option<ValueId>, op: &str) -> Result<SequenceValue> {
        let vid = vid.ok_or_else(|| self.seq_missing_input(op))?;
        self.sequences
            .get(&vid)
            .cloned()
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "input value#{} is not a live sequence. To fix: ensure it is produced \
                 by a Sequence-producing op (SequenceEmpty/Construct/Insert/Erase/\
                 SplitToSequence)",
                    vid.0
                ),
            })
    }

    /// Read a scalar `i64`/`i32` position input.
    fn read_scalar_i64(
        &self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
    ) -> Result<i64> {
        let shape = resolved.get(&vid).cloned().unwrap_or_default();
        let dtype = self.value_dtypes[&vid];
        let vals = self
            .input_i64(vid, &shape, dtype)
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "position input has dtype {dtype:?}, expected an integer (int32/int64). \
                 To fix: provide an int64 scalar index"
                ),
            })?;
        vals.first()
            .copied()
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: "position input is empty; expected a single scalar index".to_string(),
            })
    }

    /// Read an `i64` vector from an integer tensor input (SplitToSequence's
    /// `split`).
    fn read_i64_vec(&self, vid: ValueId, shape: &[usize], op: &str) -> Result<Vec<i64>> {
        let dtype = self.value_dtypes[&vid];
        self.input_i64(vid, shape, dtype)
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "'split' input has dtype {dtype:?}, expected int32/int64. To fix: \
                 provide integer split sizes"
                ),
            })
    }

    /// Back a tensor *output* value with a shared sequence element (SequenceAt):
    /// the value owns no buffer — a downstream consumer reads the element's
    /// `Arc` bytes zero-copy. Any stale buffer/view for the value is released.
    fn store_seq_element_output(
        &mut self,
        vid: ValueId,
        elem: std::sync::Arc<SeqTensor>,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        if let Some(old) = self.buffers.remove(&vid) {
            self.ep.deallocate(old)?;
        }
        self.buffer_shapes.remove(&vid);
        self.views.remove(&vid);
        resolved.insert(vid, elem.shape.clone());
        self.value_dtypes.insert(vid, elem.dtype);
        self.seq_elem_values.insert(vid, elem);
        Ok(())
    }

    /// Store freshly-computed contiguous bytes into a tensor output value
    /// (SequenceLength / ConcatFromSequence): (re)allocate its buffer, copy the
    /// bytes once, and record its dtype/shape.
    fn store_raw_tensor_output(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: Vec<usize>,
        bytes: &[u8],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        // A value that was seq-element-backed on a prior run must not shadow the
        // fresh buffer we write here.
        self.seq_elem_values.remove(&vid);
        self.views.remove(&vid);
        let need = bytes.len().max(1);
        let fits = self
            .buffers
            .get(&vid)
            .map(|b| b.len() == need)
            .unwrap_or(false);
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
        self.ep.copy_from_host(bytes, buf)?;
        self.value_dtypes.insert(vid, dtype);
        self.buffer_shapes.insert(vid, dims.clone());
        resolved.insert(vid, dims);
        Ok(())
    }

    /// The producing node id of value `vid` (Sequence ops always have a
    /// producer).
    fn plan_node_of(&self, vid: ValueId) -> NodeId {
        self.graph
            .value(vid)
            .producer
            .expect("sequence op output has a producer")
    }

    fn seq_missing_input(&self, op: &str) -> SessionError {
        SessionError::SequenceOp {
            op: op.to_string(),
            reason: "a required input is missing (omitted None slot). To fix: connect \
                     all required inputs of this Sequence op"
                .to_string(),
        }
    }

    fn seq_unresolved(&self, op: &str, vid: ValueId) -> SessionError {
        let name = self
            .graph
            .try_value(vid)
            .and_then(|v| v.name.clone())
            .unwrap_or_else(|| format!("value#{}", vid.0));
        SessionError::SequenceOp {
            op: op.to_string(),
            reason: format!(
                "input {name} has no resolved shape yet. To fix: ensure its producer \
                 runs before this Sequence op"
            ),
        }
    }
}

/// Map a [`crate::sequence::SeqOpError`] into an actionable `SessionError`.
fn seq_err(e: crate::sequence::SeqOpError) -> SessionError {
    SessionError::SequenceOp {
        op: e.op.to_string(),
        reason: e.reason,
    }
}

/// Normalize a possibly-negative ONNX `axis` against `rank`, returning the
/// non-negative axis or `None` when out of `[-rank, rank-1]`.
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        None
    } else {
        Some(a as usize)
    }
}

/// Whether `(op_type, domain)` is one of the standard subgraph-bearing
/// control-flow ops the executor handles recursively (default `ai.onnx`
/// domain). Kept in lock-step with the loader's `validate_no_control_flow`
/// allow-list.
fn is_control_flow_op(op_type: &str, domain: &str) -> bool {
    (domain.is_empty() || domain == "ai.onnx") && matches!(op_type, "If" | "Loop" | "Scan")
}

/// Whether `(op_type, domain)` is an ONNX **Sequence** op the executor handles
/// directly (default `ai.onnx` domain). Like control-flow ops these are handled
/// at the executor level rather than as leaf [`Kernel`](onnx_runtime_ep_api::Kernel)s
/// because a `Kernel` sees only tensor views, never a *sequence-of-tensors*
/// runtime value. Kept as a small self-contained routing predicate (mirroring
/// [`is_control_flow_op`]) so it never collides with the EP kernel registry.
fn is_sequence_op(op_type: &str, domain: &str) -> bool {
    (domain.is_empty() || domain == "ai.onnx")
        && matches!(
            op_type,
            "SequenceEmpty"
                | "SequenceConstruct"
                | "SequenceInsert"
                | "SequenceErase"
                | "SequenceAt"
                | "SequenceLength"
                | "SplitToSequence"
                | "ConcatFromSequence"
        )
}

/// Whether a Sequence op yields a *sequence* value (vs. a tensor). Used at build
/// to mark sequence-typed values so they are excluded from tensor buffer sizing.
fn produces_sequence_output(op_type: &str, domain: &str) -> bool {
    (domain.is_empty() || domain == "ai.onnx")
        && matches!(
            op_type,
            "SequenceEmpty"
                | "SequenceConstruct"
                | "SequenceInsert"
                | "SequenceErase"
                | "SplitToSequence"
        )
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

fn missing_capture_error(attr_key: &str, name: &str) -> SessionError {
    SessionError::Internal(format!(
        "control-flow body '{attr_key}' captures free variable '{name}', but it is not \
         available in the enclosing scope. RULES #1: a subgraph may only reference outer \
         values that are graph inputs, initializers, or produced by an upstream node in an \
         enclosing graph; '{name}' matches none of these"
    ))
}

/// Names a graph or any nested body needs from its enclosing lexical scope.
/// A nested requirement stops propagating when this graph defines that name,
/// because the nested body will bind the local value at execution time.
fn required_outer_names(graph: &Graph) -> HashSet<String> {
    let formal_set: HashSet<ValueId> = graph.inputs.iter().copied().collect();
    let local_names: HashSet<&str> = graph
        .values
        .iter()
        .filter_map(|(_, value)| value.name.as_deref())
        .collect();
    let mut required = HashSet::new();
    for (vid, value) in graph.values.iter() {
        if value.producer.is_none()
            && !formal_set.contains(&vid)
            && !graph.initializers.contains_key(&vid)
            && let Some(name) = &value.name
        {
            required.insert(name.clone());
        }
    }
    for nested in graph.subgraphs.values() {
        for name in required_outer_names(nested) {
            if !local_names.contains(name.as_str()) {
                required.insert(name);
            }
        }
    }
    required
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
        // A view value owns no buffer; materialize its strided bytes contiguous.
        let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
        Tensor::from_raw(dtype, shape, &bytes)
    }

    /// The buffer-owning (root) value backing `vid`: `vid` itself if it owns a
    /// buffer, or the `source` recorded in its view metadata (always a root,
    /// since views are flattened at creation).
    fn root_of(&self, vid: ValueId) -> ValueId {
        match self.views.get(&vid) {
            Some(v) => v.source,
            None => vid,
        }
    }

    /// Contiguous row-major bytes of `vid` for `shape`/`dtype`, materializing a
    /// view (strided gather over its source buffer) or truncating an owned
    /// buffer to its logical size. This is the single materialization seam used
    /// by the graph-output boundary and control-flow scope capture.
    fn contiguous_bytes(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Result<Vec<u8>> {
        let numel: usize = shape.iter().product();
        let n = dtype.storage_bytes(numel);
        // A tensor value backed by a shared sequence element (SequenceAt output)
        // owns no buffer; its bytes are the element's contiguous bytes. This is
        // the one materialization point where they are copied out (the boundary
        // back into owned tensors); the compute path reads them zero-copy.
        if let Some(elem) = self.seq_elem_values.get(&vid) {
            return Ok(elem.data[..n.min(elem.data.len())].to_vec());
        }
        if let Some(view) = self.views.get(&vid) {
            let buf = self.buffers.get(&view.source).ok_or_else(|| {
                SessionError::Internal(format!(
                    "view value#{} aliases missing source buffer value#{}",
                    vid.0, view.source.0
                ))
            })?;
            let esize = dtype.byte_size();
            if esize == 0 {
                // Sub-byte views are never created (Slice falls back to copy),
                // so reaching here is an internal invariant violation.
                return Err(SessionError::Internal(format!(
                    "cannot materialize sub-byte view value#{}",
                    vid.0
                )));
            }
            let mut host = vec![0u8; buf.len()];
            self.ep.copy_to_host(buf, &mut host)?;
            Ok(gather_view(
                &host,
                &view.shape,
                &view.strides,
                view.byte_offset,
                esize,
            ))
        } else {
            let buf = self
                .buffers
                .get(&vid)
                .ok_or_else(|| SessionError::Internal(format!("value#{} not produced", vid.0)))?;
            let mut host = vec![0u8; n];
            self.ep.copy_to_host(buf, &mut host)?;
            Ok(host)
        }
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
        self.store_output_bytes(
            vid,
            tensor.dtype,
            tensor.shape.clone(),
            tensor.as_bytes(),
            resolved,
        )
    }

    fn store_output_bytes(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: Vec<usize>,
        bytes: &[u8],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let numel = checked_numel(&dims, || format!("value#{}", vid.0))?;
        let need =
            checked_storage_bytes(dtype, numel, || format!("value#{}", vid.0), &dims)?.max(1);
        let fits = self
            .buffers
            .get(&vid)
            .map(|b| b.len() == need)
            .unwrap_or(false);
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
        self.ep.copy_from_host(bytes, buf)?;
        self.value_dtypes.insert(vid, dtype);
        self.buffer_shapes.insert(vid, dims.clone());
        resolved.insert(vid, dims);
        Ok(())
    }

    /// Prepare one selected control-flow subgraph and materialize only the free
    /// variables that body actually captures. This avoids copying every named
    /// value in the enclosing graph and, for Loop/Scan, keeps captures stable
    /// across all iterations.
    fn prepare_subgraph(
        &self,
        node_id: NodeId,
        attr_key: &str,
        resolved: &HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<PreparedSubgraph> {
        let key = (node_id, attr_key.to_string());
        let body = self.graph.subgraphs.get(&key).ok_or_else(|| {
            SessionError::Internal(format!(
                "control-flow node #{} references missing subgraph '{attr_key}'",
                node_id.0
            ))
        })?;

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
        let formal_set: HashSet<ValueId> = body.inputs.iter().copied().collect();
        let mut capture_names = Vec::new();
        for (vid, value) in body.values.iter() {
            if value.producer.is_none()
                && !formal_set.contains(&vid)
                && !body.initializers.contains_key(&vid)
                && let Some(name) = &value.name
            {
                capture_names.push(name.clone());
            }
        }
        capture_names.sort();

        let mut scope_names = required_outer_names(body);
        scope_names.extend(capture_names.iter().cloned());
        let mut captures = HashMap::with_capacity(scope_names.len());
        for name in scope_names {
            let tensor = if let Some(&vid) = self.name_index.get(&name) {
                let materialized = self.buffers.contains_key(&vid)
                    || self.views.contains_key(&vid)
                    || self.seq_elem_values.contains_key(&vid);
                if resolved.contains_key(&vid) && materialized {
                    self.value_tensor(vid, resolved)?
                } else {
                    outer_scope
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| missing_capture_error(attr_key, &name))?
                }
            } else {
                outer_scope
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| missing_capture_error(attr_key, &name))?
            };
            captures.insert(name, tensor);
        }

        Ok(PreparedSubgraph {
            key,
            formal_names,
            capture_names,
            captures,
        })
    }

    /// Compile a control-flow body subgraph to a child [`Executor`], turning
    /// captured outer-scope names into extra graph inputs (so they are supplied
    /// and written every run), seeding the concrete external-input shapes, and
    /// running shape inference so the body's interior buffers can be sized.
    fn build_subgraph_exec(
        &self,
        prepared: &PreparedSubgraph,
        externals: &[&Tensor],
    ) -> Result<CompiledSubgraph> {
        let key = &prepared.key;
        let body = self.graph.subgraphs.get(key).ok_or_else(|| {
            SessionError::Internal(format!(
                "control-flow node #{} has no registered subgraph '{}'",
                key.0.0, key.1
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
        for cname in &prepared.capture_names {
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
        let all_names = prepared
            .formal_names
            .iter()
            .chain(prepared.capture_names.iter());
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

        let exec = Executor::build(g, self.weights.clone(), self.ep.clone())?;
        Ok(CompiledSubgraph {
            exec,
            input_names: prepared
                .formal_names
                .iter()
                .chain(prepared.capture_names.iter())
                .cloned()
                .collect(),
            built_shapes: externals.iter().map(|t| t.shape.clone()).collect(),
        })
    }

    /// Run a prepared control-flow body with changing formal inputs. Captures and
    /// signature metadata are reused; only a concrete shape change rebuilds the
    /// child executor.
    fn run_subgraph(
        &mut self,
        prepared: &PreparedSubgraph,
        formal_inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        if prepared.formal_names.len() != formal_inputs.len() {
            return Err(SessionError::Internal(format!(
                "control-flow body '{}' expects {} formal input(s) but {} were supplied",
                prepared.key.1,
                prepared.formal_names.len(),
                formal_inputs.len()
            )));
        }

        let mut externals: Vec<&Tensor> =
            Vec::with_capacity(formal_inputs.len() + prepared.capture_names.len());
        externals.extend_from_slice(formal_inputs);
        for name in &prepared.capture_names {
            externals.push(
                prepared
                    .captures
                    .get(name)
                    .expect("prepared capture must be present"),
            );
        }
        let rebuild = match self.subgraph_execs.get(&prepared.key) {
            Some(cs) => {
                cs.built_shapes.len() != externals.len()
                    || cs
                        .built_shapes
                        .iter()
                        .zip(externals.iter())
                        .any(|(built, tensor)| built != &tensor.shape)
            }
            None => true,
        };
        if rebuild {
            let child = self.build_subgraph_exec(prepared, &externals)?;
            self.subgraph_execs.insert(prepared.key.clone(), child);
            self.control_flow_stats.subgraph_builds += 1;
        }

        self.control_flow_stats.subgraph_runs += 1;
        let cs = self
            .subgraph_execs
            .get_mut(&prepared.key)
            .expect("child present");
        let inputs: Vec<(&str, &Tensor)> = cs
            .input_names
            .iter()
            .map(String::as_str)
            .zip(externals)
            .collect();
        cs.exec.run_scoped(&inputs, &prepared.captures)
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
        let cond = tensor_scalar_bool(&cond_t).ok_or_else(|| {
            SessionError::Internal(format!(
                "If: 'cond' must be a BOOL scalar, got dtype {:?} shape {:?}",
                cond_t.dtype, cond_t.shape
            ))
        })?;

        let attr_key = if cond { "then_branch" } else { "else_branch" };
        let prepared = self.prepare_subgraph(node.id, attr_key, resolved, outer_scope)?;
        let outs = self.run_subgraph(&prepared, &[])?;

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
                let m = tensor_scalar_i64(&t).ok_or_else(|| {
                    SessionError::Internal(format!(
                        "Loop: trip-count 'M' must be an INT64/INT32 scalar, got dtype {:?}",
                        t.dtype
                    ))
                })?;
                Some(m)
            }
            None => None,
        };
        let mut cond: Option<bool> = match node.inputs.get(1).and_then(|s| *s) {
            Some(vid) => {
                let t = self.value_tensor(vid, resolved)?;
                Some(tensor_scalar_bool(&t).ok_or_else(|| {
                    SessionError::Internal(format!(
                        "Loop: 'cond' must be a BOOL scalar, got dtype {:?}",
                        t.dtype
                    ))
                })?)
            }
            None => None,
        };

        // Initial loop-carried dependencies (inputs after M and cond).
        let mut carried: Vec<Tensor> = Vec::new();
        for slot in node.inputs.iter().skip(2) {
            let vid = slot.ok_or_else(|| {
                SessionError::Internal(
                    "Loop: an interior loop-carried input is omitted (empty), which ONNX does not \
                 allow — every v_initial must be provided"
                        .to_string(),
                )
            })?;
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
        let expected_iterations = m.and_then(|n| usize::try_from(n).ok());
        let mut scan_acc: Vec<TensorStackAccumulator> = (0..num_scan)
            .map(|_| TensorStackAccumulator::new(expected_iterations))
            .collect();
        let prepared = self.prepare_subgraph(node.id, "body", resolved, outer_scope)?;
        let mut iter_tensor = scalar_i64_tensor(0)?;
        let mut cond_tensor = scalar_bool_tensor(cond.unwrap_or(true))?;

        let mut iter: i64 = 0;
        loop {
            if let Some(m) = m
                && iter >= m
            {
                break;
            }
            if cond == Some(false) {
                break;
            }

            iter_tensor.overwrite_bytes(&iter.to_le_bytes())?;
            cond_tensor.overwrite_bytes(&[u8::from(cond.unwrap_or(true))])?;
            let mut formal: Vec<&Tensor> = Vec::with_capacity(2 + num_carried);
            formal.push(&iter_tensor);
            formal.push(&cond_tensor);
            formal.extend(carried.iter());

            let outs = self.run_subgraph(&prepared, &formal)?;
            drop(formal);
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
            cond = Some(tensor_scalar_bool(&cond_out).ok_or_else(|| {
                SessionError::Internal(format!(
                    "Loop: body's first output 'cond_out' must be a BOOL scalar, got dtype {:?}",
                    cond_out.dtype
                ))
            })?);
            carried.clear();
            carried.extend((&mut it).take(num_carried));
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"))?;
            }

            iter += 1;
        }

        // Emit outputs: carried finals, then stacked scan outputs.
        for (i, t) in carried.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved)?;
        }
        for (s, acc) in scan_acc.into_iter().enumerate() {
            let (dtype, shape, bytes) = acc.finish();
            self.store_output_bytes(
                node.outputs[num_carried + s],
                dtype,
                shape,
                &bytes,
                resolved,
            )?;
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
            .ok_or_else(|| {
                SessionError::Internal(
                    "Scan: required attribute 'num_scan_inputs' is missing or not an INT"
                        .to_string(),
                )
            })? as usize;

        // Reject the axis/direction knobs this Phase-1 implementation does not
        // yet honor, rather than silently ignoring them (RULES #1/#5).
        for attr in ["scan_input_axes", "scan_output_axes"] {
            if let Some(a) = node.attr(attr)
                && let Some(axes) = a.as_ints()
                && axes.iter().any(|&ax| ax != 0)
            {
                return Err(SessionError::Internal(format!(
                    "Scan: attribute '{attr}' = {axes:?} requests a non-zero scan axis, \
                     which this runtime does not yet support. Expected axis 0 for every \
                     scan input/output; re-export with axis 0 or wait for full Scan-axis \
                     support"
                )));
            }
        }
        for attr in ["scan_input_directions", "scan_output_directions"] {
            if let Some(a) = node.attr(attr)
                && let Some(dirs) = a.as_ints()
                && dirs.iter().any(|&d| d != 0)
            {
                return Err(SessionError::Internal(format!(
                    "Scan: attribute '{attr}' = {dirs:?} requests reverse iteration, which \
                     this runtime does not yet support (forward only). Re-export forward or \
                     wait for reverse-Scan support"
                )));
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
            let vid = slot.ok_or_else(|| {
                SessionError::Internal(
                    "Scan: an initial-state input is omitted (empty), which ONNX does not allow"
                        .to_string(),
                )
            })?;
            state.push(self.value_tensor(vid, resolved)?);
        }
        let mut scan_inputs: Vec<Tensor> = Vec::with_capacity(num_scan_inputs);
        for slot in node.inputs.iter().skip(num_state) {
            let vid = slot.ok_or_else(|| {
                SessionError::Internal(
                    "Scan: a scan input is omitted (empty), which ONNX does not allow".to_string(),
                )
            })?;
            scan_inputs.push(self.value_tensor(vid, resolved)?);
        }

        // Sequence length = extent of scan axis 0; all scan inputs must agree.
        let seq_len = scan_inputs
            .first()
            .and_then(|t| t.shape.first().copied())
            .ok_or_else(|| {
                SessionError::Internal(
                    "Scan: requires at least one scan input with rank >= 1".to_string(),
                )
            })?;
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
        let mut scan_acc: Vec<TensorStackAccumulator> = (0..num_scan_out)
            .map(|_| TensorStackAccumulator::new(Some(seq_len)))
            .collect();
        let prepared = self.prepare_subgraph(node.id, "body", resolved, outer_scope)?;
        let mut scan_slices = Vec::with_capacity(num_scan_inputs);
        if seq_len != 0 {
            for t in &scan_inputs {
                let (shape, bytes) = leading_slice(t, 0)?;
                scan_slices.push(Tensor::from_raw_in(self.ep.clone(), t.dtype, shape, bytes)?);
            }
        }
        for step in 0..seq_len {
            if step != 0 {
                for (source, slice) in scan_inputs.iter().zip(scan_slices.iter_mut()) {
                    let (_, bytes) = leading_slice(source, step)?;
                    slice.overwrite_bytes(bytes)?;
                }
            }
            let mut formal: Vec<&Tensor> = Vec::with_capacity(num_state + num_scan_inputs);
            formal.extend(state.iter());
            formal.extend(scan_slices.iter());

            let outs = self.run_subgraph(&prepared, &formal)?;
            drop(formal);
            let expected = num_state + num_scan_out;
            if outs.len() != expected {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: "Scan/body".to_string(),
                    expected,
                    got: outs.len(),
                });
            }
            let mut it = outs.into_iter();
            state.clear();
            state.extend((&mut it).take(num_state));
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"))?;
            }
        }

        for (i, t) in state.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved)?;
        }
        for (s, acc) in scan_acc.into_iter().enumerate() {
            let (dtype, shape, bytes) = acc.finish();
            self.store_output_bytes(node.outputs[num_state + s], dtype, shape, &bytes, resolved)?;
        }
        Ok(())
    }
}

/// Borrow the `index`-th contiguous slice along a tensor's leading axis while
/// dropping that axis from the returned shape.
fn leading_slice(t: &Tensor, index: usize) -> Result<(Vec<usize>, &[u8])> {
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
    Ok((inner_shape, bytes))
}

/// Single-allocation accumulator for Loop/Scan scan outputs. Each iteration's
/// temporary output is copied directly into its final stacked byte position and
/// then dropped, avoiding a retained tensor allocation per step and a second
/// full stacking pass.
struct TensorStackAccumulator {
    expected_len: Option<usize>,
    dtype: Option<DataType>,
    elem_shape: Vec<usize>,
    len: usize,
    bytes: Vec<u8>,
}

impl TensorStackAccumulator {
    fn new(expected_len: Option<usize>) -> Self {
        Self {
            expected_len,
            dtype: None,
            elem_shape: Vec::new(),
            len: 0,
            bytes: Vec::new(),
        }
    }

    fn push(&mut self, tensor: Tensor) -> Result<()> {
        if let Some(dtype) = self.dtype {
            if tensor.shape != self.elem_shape || tensor.dtype != dtype {
                return Err(SessionError::Internal(format!(
                    "Loop/Scan: scan output slice {} has shape {:?} dtype {:?} but the first slice \
                     is shape {:?} dtype {:?}; every iteration's scan output must match",
                    self.len, tensor.shape, tensor.dtype, self.elem_shape, dtype
                )));
            }
        } else {
            if tensor.dtype.byte_size() == 0 {
                return Err(SessionError::Internal(format!(
                    "Loop/Scan: sub-byte dtype {:?} scan outputs are not supported",
                    tensor.dtype
                )));
            }
            self.dtype = Some(tensor.dtype);
            self.elem_shape = tensor.shape.clone();
            if let Some(expected) = self.expected_len {
                self.bytes
                    .reserve(expected.saturating_mul(tensor.as_bytes().len()));
            }
        }
        self.bytes.extend_from_slice(tensor.as_bytes());
        self.len += 1;
        Ok(())
    }

    fn finish(self) -> (DataType, Vec<usize>, Vec<u8>) {
        if self.len == 0 {
            return (DataType::Float32, vec![0], Vec::new());
        }
        let dtype = self.dtype.expect("non-empty accumulator has dtype");
        let mut shape = Vec::with_capacity(1 + self.elem_shape.len());
        shape.push(self.len);
        shape.extend(self.elem_shape);
        (dtype, shape, self.bytes)
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
pub(crate) fn auto_detect_cpu_ep() -> Result<Arc<dyn ExecutionProvider>> {
    let mut ep = CpuExecutionProvider::new();
    ep.initialize(&Default::default())?;
    Ok(Arc::new(ep))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuses_only_single_consumer_silu_pattern() {
        let mut graph = Graph::new();
        let shape = vec![Dim::Static(2)];
        let x = graph.create_named_value("x", DataType::Float32, shape.clone());
        let sigmoid_out = graph.create_named_value("sigmoid", DataType::Float32, shape.clone());
        let silu_out = graph.create_named_value("silu", DataType::Float32, shape);
        graph.add_input(x);
        graph.add_output(silu_out);
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(x)],
            vec![sigmoid_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(sigmoid_out), Some(x)],
            vec![silu_out],
        ));

        assert_eq!(fuse_silu_patterns(&mut graph), 1);
        assert_eq!(graph.num_nodes(), 1);
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "Silu");
        assert_eq!(fused.domain, "com.microsoft");
        assert_eq!(fused.inputs, vec![Some(x)]);
        assert_eq!(fused.outputs, vec![silu_out]);
        assert_eq!(graph.opset_imports["com.microsoft"], 1);
    }

    #[test]
    fn does_not_fuse_silu_when_sigmoid_has_second_consumer() {
        let mut graph = Graph::new();
        let shape = vec![Dim::Static(2)];
        let x = graph.create_named_value("x", DataType::Float32, shape.clone());
        let sigmoid_out = graph.create_named_value("sigmoid", DataType::Float32, shape.clone());
        let mul_out = graph.create_named_value("mul", DataType::Float32, shape.clone());
        let identity_out = graph.create_named_value("identity", DataType::Float32, shape);
        graph.add_input(x);
        graph.add_output(mul_out);
        graph.add_output(identity_out);
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(x)],
            vec![sigmoid_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(x), Some(sigmoid_out)],
            vec![mul_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(sigmoid_out)],
            vec![identity_out],
        ));

        assert_eq!(fuse_silu_patterns(&mut graph), 0);
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Sigmoid")
                .count(),
            1
        );
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Mul")
                .count(),
            1
        );
        assert!(graph.validate().is_ok());
    }

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
        assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));
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
            None,          // data (unused by sizer)
            Some(vec![1]), // starts
            Some(vec![3]), // ends
            Some(vec![0]), // axes
            Some(vec![1]), // steps
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

    // --- weight-streaming: zero-copy borrowed initializer buffers -----------

    use onnx_runtime_ir::{WeightRef, static_shape};
    use std::path::PathBuf;

    /// A writable scratch dir under the workspace `target/` (never `/tmp`).
    fn weightstream_tmp_dir() -> PathBuf {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/weightstream_test"
        ));
        std::fs::create_dir_all(&dir).expect("create weight-streaming test dir");
        dir
    }

    fn f32_le(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// (b) An aligned external-data initializer is backed **zero-copy** by a
    /// borrowed buffer whose data pointer EQUALS the WeightStore's mmap slice —
    /// no allocation, no copy. A model larger than RAM relies on this.
    #[test]
    fn aligned_external_initializer_is_borrowed_zero_copy() {
        let align = TensorLayout::contiguous().alignment;
        let path = weightstream_tmp_dir().join("aligned_init.bin");
        let w_data = [1.0f32, 2.0, 3.0, 4.0];
        std::fs::write(&path, f32_le(&w_data)).unwrap();

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
        g.set_initializer(
            w,
            WeightRef::External {
                path: path.clone(),
                offset: 0, // mmap base is page-aligned -> 0 is `align`-aligned
                length: 16,
                dtype: DataType::Float32,
                dims: vec![4],
            },
        );
        let y = g.create_value(DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(w)], vec![y]));
        g.add_output(y);

        let ep = auto_detect_cpu_ep().unwrap();
        let exec = Executor::build(g, Arc::new(store), ep).unwrap();

        let weight = &exec.graph.initializers[&w];
        let src = exec.weights().bytes(weight).unwrap();
        assert!(
            (src.as_ptr() as usize).is_multiple_of(align),
            "mmap window must be aligned for this test to exercise the zero-copy path"
        );
        let buf = &exec.buffers[&w];
        assert!(
            buf.is_borrowed(),
            "aligned initializer must be borrowed, not copied"
        );
        assert_eq!(
            buf.as_ptr() as *const u8,
            src.as_ptr(),
            "zero-copy: the buffer must alias the mmap bytes (no copy)"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// (c) An unaligned external-data initializer falls back to an owned copy
    /// (buffer ptr != slice ptr) and is still numerically correct end-to-end.
    #[test]
    fn unaligned_external_initializer_falls_back_to_owned_copy() {
        let align = TensorLayout::contiguous().alignment;
        let path = weightstream_tmp_dir().join("unaligned_init.bin");
        // Prefix the weight window with 8 bytes so it starts at offset 8, which
        // is not a multiple of `align` (64) -> forces the copy fallback.
        let offset = 8usize;
        let w_data = [5.0f32, 6.0, 7.0, 8.0];
        let mut file = vec![0u8; offset];
        file.extend_from_slice(&f32_le(&w_data));
        std::fs::write(&path, &file).unwrap();

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
        g.set_initializer(
            w,
            WeightRef::External {
                path: path.clone(),
                offset,
                length: 16,
                dtype: DataType::Float32,
                dims: vec![4],
            },
        );
        let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
        g.add_input(x);
        let y = g.create_value(DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(w)], vec![y]));
        g.add_output(y);

        let ep = auto_detect_cpu_ep().unwrap();
        let mut exec = Executor::build(g, Arc::new(store), ep).unwrap();

        let weight = &exec.graph.initializers[&w];
        let src = exec.weights().bytes(weight).unwrap();
        assert!(
            !(src.as_ptr() as usize).is_multiple_of(align),
            "window must be unaligned for this test to exercise the fallback"
        );
        let buf = &exec.buffers[&w];
        assert!(
            !buf.is_borrowed(),
            "unaligned initializer must fall back to an owned copy"
        );
        assert_ne!(
            buf.as_ptr() as *const u8,
            src.as_ptr(),
            "fallback: the buffer must be a fresh copy, not an alias"
        );

        // The copy is numerically correct: Y = X + W.
        let x_tensor = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = exec.run(&[("X", &x_tensor)]).unwrap();
        assert_eq!(out.len(), 1);
        let got = out[0].to_vec_f32();
        let want = [15.0f32, 26.0, 37.0, 48.0];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }

        let _ = std::fs::remove_file(&path);
    }

    /// (d) Soundness guard: even when an initializer's mmap bytes are aligned
    /// (so the zero-copy path would otherwise fire), the executor must NOT
    /// borrow them if the value also has a producer — i.e. a malformed graph
    /// reused the initializer's `ValueId` as a node output. Borrowing yields a
    /// read-only buffer; a kernel writing that output would write through the
    /// mmap (SIGSEGV / aliasing UB). The build must fall back to an owned,
    /// writable copy instead.
    #[test]
    fn producer_backed_initializer_is_not_borrowed() {
        let align = TensorLayout::contiguous().alignment;
        let path = weightstream_tmp_dir().join("producer_backed_init.bin");
        let w_data = [1.0f32, 2.0, 3.0, 4.0];
        std::fs::write(&path, f32_le(&w_data)).unwrap();

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
        g.add_input(x);
        let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
        g.set_initializer(
            w,
            WeightRef::External {
                path: path.clone(),
                offset: 0, // aligned: without the producer guard this would borrow
                length: 16,
                dtype: DataType::Float32,
                dims: vec![4],
            },
        );
        // Reuse the initializer's ValueId as a node output -> gives `w` a
        // producer, exactly the malformed shape the loader also rejects.
        g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![w]));
        let y = g.create_value(DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(1), "Add", vec![Some(x), Some(w)], vec![y]));
        g.add_output(y);

        assert!(
            g.value(w).producer.is_some(),
            "test setup: initializer value must have a producer",
        );

        let ep = auto_detect_cpu_ep().unwrap();
        let exec = Executor::build(g, Arc::new(store), ep).unwrap();

        let weight = &exec.graph.initializers[&w];
        let src = exec.weights().bytes(weight).unwrap();
        assert!(
            (src.as_ptr() as usize).is_multiple_of(align),
            "mmap window must be aligned so only the producer guard prevents borrowing",
        );
        let buf = &exec.buffers[&w];
        assert!(
            !buf.is_borrowed(),
            "producer-backed initializer must fall back to an owned writable copy",
        );
        assert_ne!(
            buf.as_ptr() as *const u8,
            src.as_ptr(),
            "producer-backed initializer must not alias read-only mmap bytes",
        );

        let _ = std::fs::remove_file(&path);
    }
}
