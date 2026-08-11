//! The inference context handed to each op rule, plus the supporting type
//! model: [`TypeInfo`], [`TypedShape`], [`MergePolicy`], and the
//! [`SymbolInterner`] that lowers derived [`DimExpr`]s back to IR [`Dim`]s.

use std::collections::HashMap;

use onnx_runtime_ir::{DataType, Dim, Node, SymbolId, ValueId, normalize_domain};

use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::shape_data::ShapeData;

/// An inferred shape: an ordered list of symbolic dimension expressions. The
/// rank is always known (unknown-rank tensors are represented by the *absence*
/// of a [`TypeInfo`], never by a `TypedShape`).
pub type TypedShape = Vec<DimExpr>;

/// The inferred type of a value: element dtype plus a symbolic shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeInfo {
    pub dtype: DataType,
    pub shape: TypedShape,
}

impl TypeInfo {
    /// A new type info from a dtype and shape.
    pub fn new(dtype: DataType, shape: TypedShape) -> Self {
        Self { dtype, shape }
    }

    /// The rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// A tensor leaf inside a [`ValueType`] container element type.
///
/// Unlike the top-level [`TypeInfo`] used by the tensor-only path, the `shape`
/// is *optional*: a container producer such as `SequenceEmpty` knows only the
/// element `dtype`, never its rank. Representing that honestly (rather than
/// fabricating a bogus shape) is why this type exists separately from
/// [`TypeInfo`]. A `TensorType` whose `shape` is known converts to and from a
/// [`TypeInfo`] losslessly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DataType,
    /// `None` when the rank/shape is unknown (a dtype-only tensor).
    pub shape: Option<TypedShape>,
}

impl TensorType {
    /// A tensor leaf with a known shape.
    pub fn new(dtype: DataType, shape: TypedShape) -> Self {
        Self {
            dtype,
            shape: Some(shape),
        }
    }

    /// A tensor leaf whose dtype is known but whose shape is not.
    pub fn dtype_only(dtype: DataType) -> Self {
        Self { dtype, shape: None }
    }

    /// The full [`TypeInfo`], available only when the shape is known.
    pub fn to_type_info(&self) -> Option<TypeInfo> {
        self.shape
            .as_ref()
            .map(|shape| TypeInfo::new(self.dtype, shape.clone()))
    }
}

impl From<TypeInfo> for TensorType {
    fn from(type_info: TypeInfo) -> Self {
        Self {
            dtype: type_info.dtype,
            shape: Some(type_info.shape),
        }
    }
}

/// The full type of a value: a tensor, or a container whose element type is
/// itself a [`ValueType`].
///
/// ONNX values are tensors in the overwhelming majority of graphs; the tensor
/// path never materialises a `ValueType` at all (a value with no recorded
/// `ValueType` is, by construction, a plain tensor). The container variants
/// exist only so `Sequence`/`Optional`/`Map` operators can propagate their
/// element types. This layer is *additive*: it wraps, and never replaces,
/// [`TypeInfo`], so the tensor-only path stays byte-identical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueType {
    /// A tensor value.
    Tensor(TensorType),
    /// A homogeneous sequence of `element`-typed values.
    Sequence(Box<ValueType>),
    /// An optional value that is either present as `element` or absent.
    Optional(Box<ValueType>),
    /// A map from `key` (an integer or string dtype) to `value`-typed values.
    Map(DataType, Box<ValueType>),
}

impl ValueType {
    /// A tensor value with a known shape.
    pub fn tensor(dtype: DataType, shape: TypedShape) -> Self {
        Self::Tensor(TensorType::new(dtype, shape))
    }

    /// A sequence whose elements have type `element`.
    pub fn sequence(element: ValueType) -> Self {
        Self::Sequence(Box::new(element))
    }

    /// The tensor leaf, when this value is a tensor.
    pub fn as_tensor(&self) -> Option<&TensorType> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    /// The element type, when this value is a sequence.
    pub fn as_sequence_element(&self) -> Option<&ValueType> {
        match self {
            Self::Sequence(element) => Some(element),
            _ => None,
        }
    }
}

/// The resolved inference state of a single input or output slot: an optional
/// type and an optional [`ShapeData`] side-value.
#[derive(Clone, Debug, Default)]
pub struct NodeIo {
    pub type_info: Option<TypeInfo>,
    pub shape_data: Option<ShapeData>,
    /// The container type of this slot, when it is a `Sequence`/`Optional`/`Map`
    /// value. `None` for plain tensors — the overwhelming common case — which
    /// keeps the tensor-only path byte-identical.
    pub value_type: Option<ValueType>,
}

impl NodeIo {
    /// An i/o slot carrying only a type.
    pub fn typed(type_info: TypeInfo) -> Self {
        Self {
            type_info: Some(type_info),
            shape_data: None,
            value_type: None,
        }
    }

    /// An i/o slot carrying a container [`ValueType`].
    pub fn container(value_type: ValueType) -> Self {
        Self {
            type_info: None,
            shape_data: None,
            value_type: Some(value_type),
        }
    }
}

/// How to reconcile an inferred shape with a value's declared shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergePolicy {
    /// Prefer the more specific dimension and keep going; never error on a
    /// disagreement. This is the robust default used for whole-graph inference.
    #[default]
    Permissive,
    /// Raise [`ShapeInferError::ShapeConflict`] / [`ShapeInferError::RankConflict`]
    /// on a *concrete* disagreement between inferred and declared shapes.
    /// Symbolic differences are treated as naming and never conflict.
    Strict,
}

/// Allocates and interns fresh symbolic dimensions, and lowers [`DimExpr`]s to
/// IR [`Dim`]s.
///
/// A derived dimension that is neither a pure constant nor a bare symbol (e.g.
/// `floor((d-k)/s)+1` where `d` is symbolic) cannot be stored in the IR's
/// [`Dim`] enum. Such an expression is assigned a *fresh* symbol; because
/// [`DimExpr`] is canonical, two structurally-identical derived dimensions
/// intern to the **same** symbol and stay unified across the graph.
#[derive(Debug)]
pub struct SymbolInterner {
    next: u32,
    /// The floor `next` started at: every symbol id `>= initial_floor` was
    /// minted by inference (an anonymous/derived/data-dependent symbol), while
    /// ids below it are graph-declared roots (`batch`, `seq`, KV length, …).
    /// Persisted so a consumer (the CUDA-graph capture classifier's fail-safe
    /// mode) can tell a *provably-rooted* symbol from an inference-minted one.
    initial_floor: u32,
    cache: HashMap<DimExpr, SymbolId>,
    /// Symbols minted during inference, to be registered on the graph.
    fresh: Vec<SymbolId>,
    /// Every `(loser, winner)` symbol pair that [`broadcast_dim`] unified when
    /// broadcasting two *distinct* symbolic dimensions onto one representative.
    /// This is additive bookkeeping: recording a pair never changes the returned
    /// representative or any inferred dim, so inference output stays byte-
    /// identical. Persisted onto [`Graph::symbol_unifications`] so downstream
    /// consumers (e.g. capture-eligibility) can close over the equivalence
    /// classes without re-implementing a partial copy of inference's unification.
    ///
    /// [`broadcast_dim`]: InferenceContext::broadcast_dim
    /// [`Graph::symbol_unifications`]: onnx_runtime_ir::Graph::symbol_unifications
    unifications: Vec<(SymbolId, SymbolId)>,
    /// Every `(derived, source)` provenance edge recorded when [`lower`] interns
    /// a non-bare *derived* [`DimExpr`] (e.g. `seq_kv * 8` from `Reshape`/
    /// `Flatten`) to a fresh [`SymbolId`]: the fresh `derived` symbol depends on
    /// each `source` symbol the expression was built from. This is the general
    /// lineage record that closes the derived-symbol capture hole (a fresh
    /// symbol built from a growing one must itself be treated as growing). Like
    /// [`unifications`](Self::unifications) it is purely additive — `lower`
    /// returns the same `Dim` regardless — so inference stays byte-identical.
    /// Persisted onto [`Graph::symbol_derivations`].
    ///
    /// [`lower`]: Self::lower
    /// [`Graph::symbol_derivations`]: onnx_runtime_ir::Graph::symbol_derivations
    derivations: Vec<(SymbolId, SymbolId)>,
    /// Symbols minted for an *unknowable* extent from which no source symbol
    /// could be recovered — an arithmetic-overflow degrade or a nonsensical
    /// negative extent (see [`lower`](Self::lower)). Such a symbol has no
    /// provenance, so a conservative consumer must treat it as disqualifying
    /// (eager) rather than assume it is constant. Persisted onto
    /// [`Graph::symbol_opaque`](onnx_runtime_ir::Graph::symbol_opaque).
    opaque: Vec<SymbolId>,
}

impl SymbolInterner {
    /// A new interner that allocates symbol ids starting at `next` (which must
    /// be greater than every symbol id already present in the graph).
    pub fn new(next: u32) -> Self {
        Self {
            next,
            initial_floor: next,
            cache: HashMap::new(),
            fresh: Vec::new(),
            unifications: Vec::new(),
            derivations: Vec::new(),
            opaque: Vec::new(),
        }
    }

    /// The floor id at/above which every symbol was minted by this inference
    /// pass (see [`initial_floor`](Self::initial_floor)).
    pub fn initial_floor(&self) -> u32 {
        self.initial_floor
    }

    /// Record that inference unified two distinct symbolic dimensions onto a
    /// single representative (the `(loser, winner)` substitution in
    /// [`broadcast_dim`](InferenceContext::broadcast_dim)). Order is irrelevant
    /// to consumers (they build an undirected equivalence relation); this is a
    /// pure append and does not influence the inferred shape.
    fn record_unification(&mut self, a: SymbolId, b: SymbolId) {
        self.unifications.push((a, b));
    }

    /// Record that the fresh symbol `derived` was interned from an expression
    /// built out of `source` (a directed provenance edge `derived -> source`).
    /// Pure append; never influences the inferred shape.
    fn record_derivation(&mut self, derived: SymbolId, source: SymbolId) {
        self.derivations.push((derived, source));
    }

    /// Record that `sym` was minted for a genuinely unknowable extent (overflow
    /// or negative degrade) with no recoverable source symbols.
    fn record_opaque(&mut self, sym: SymbolId) {
        self.opaque.push(sym);
    }

    /// The symbol pairs unified during inference (to persist on the graph).
    pub fn unifications(&self) -> &[(SymbolId, SymbolId)] {
        &self.unifications
    }

    /// The `(derived, source)` provenance edges recorded during inference (to
    /// persist on the graph).
    pub fn derivations(&self) -> &[(SymbolId, SymbolId)] {
        &self.derivations
    }

    /// The opaque (unknowable-extent) symbols minted during inference.
    pub fn opaque(&self) -> &[SymbolId] {
        &self.opaque
    }

    /// Mint a brand-new opaque symbol (not tied to any expression).
    pub fn fresh_symbol(&mut self) -> SymbolId {
        let id = SymbolId(self.next);
        // Saturating rather than wrapping: exhausting the u32 symbol space is
        // adversarial/pathological, but must never wrap `next` back into the
        // range of already-minted ids (which would alias symbols).
        self.next = self.next.saturating_add(1);
        self.fresh.push(id);
        id
    }

    /// Mint a fresh opaque dimension expression.
    pub fn fresh_dim(&mut self) -> DimExpr {
        DimExpr::symbol(self.fresh_symbol())
    }

    /// Lower a [`DimExpr`] to an IR [`Dim`], interning derived expressions to a
    /// stable fresh symbol.
    ///
    /// Whenever a *derived* (non-bare, non-const) expression is interned to a
    /// fresh symbol, its symbol lineage is recorded (see
    /// [`derivations`](Self::derivations)): the fresh symbol depends on every
    /// constituent symbol of the expression. This is what lets a downstream
    /// consumer close a growing/pinned set transitively across `Reshape`/
    /// `Flatten`-style derived dims. Recording is purely additive: the returned
    /// `Dim` is identical with or without it, so inference stays byte-identical.
    pub fn lower(&mut self, expr: &DimExpr) -> Dim {
        // An overflowed (unknown) expression has no representable value and must
        // not alias other overflows via the cache: mint a distinct fresh symbol.
        // The overflow sentinel has dropped its terms, so no source symbols are
        // recoverable — mark the minted symbol OPAQUE so a conservative consumer
        // treats it as disqualifying (never assumes it is constant/pinned).
        if expr.is_overflow() {
            let id = self.fresh_symbol();
            self.record_opaque(id);
            return Dim::Symbolic(id);
        }
        if let Some(n) = expr.as_const() {
            if n >= 0 {
                return Dim::Static(n as usize);
            }
            // A negative extent is nonsensical; degrade to a fresh symbol and
            // mark it opaque (a pure constant carries no symbol lineage, but the
            // degrade is unknowable, so err toward disqualifying).
            let id = self.fresh_symbol();
            self.record_opaque(id);
            return Dim::Symbolic(id);
        }
        if let Some(s) = expr.as_symbol() {
            return Dim::Symbolic(s);
        }
        if let Some(&id) = self.cache.get(expr) {
            // Provenance was already recorded when `expr` was first interned in
            // this pass (the interner is fresh per inference run, so a cache hit
            // implies a prior insert this run); re-record for robustness. Consumers
            // dedup, and `infer_graph_scoped` dedups before persisting.
            self.record_expr_derivation(id, expr);
            return Dim::Symbolic(id);
        }
        let id = self.fresh_symbol();
        self.cache.insert(expr.clone(), id);
        self.record_expr_derivation(id, expr);
        Dim::Symbolic(id)
    }

    /// Record a provenance edge from the freshly-minted `derived` symbol to each
    /// distinct symbol appearing in `expr`.
    fn record_expr_derivation(&mut self, derived: SymbolId, expr: &DimExpr) {
        let mut seen = std::collections::HashSet::new();
        for source in expr.symbol_ids() {
            if source != derived && seen.insert(source) {
                self.record_derivation(derived, source);
            }
        }
    }

    /// The symbols minted during inference (to register on the graph).
    pub fn fresh_symbols(&self) -> &[SymbolId] {
        &self.fresh
    }
}

/// The context passed to every op inference rule.
///
/// It exposes each input's inferred type and shape-data, lets a rule mint fresh
/// symbolic dimensions and broadcast shapes, and collects the outputs the rule
/// produces. Rules never touch the [`Graph`](onnx_runtime_ir::Graph) directly —
/// they operate purely on this context, which makes them trivially unit
/// testable in isolation.
pub struct InferenceContext<'a> {
    /// The node being inferred.
    pub node: &'a Node,
    opset_imports: &'a HashMap<String, u64>,
    policy: MergePolicy,
    inputs: Vec<NodeIo>,
    outputs: Vec<NodeIo>,
    interner: &'a mut SymbolInterner,
}

impl<'a> InferenceContext<'a> {
    /// Build a context for `node` from its resolved `inputs` (aligned with
    /// `node.inputs`, skipped slots carrying an empty [`NodeIo`]).
    pub fn new(
        node: &'a Node,
        inputs: Vec<NodeIo>,
        opset_imports: &'a HashMap<String, u64>,
        policy: MergePolicy,
        interner: &'a mut SymbolInterner,
    ) -> Self {
        let outputs = vec![NodeIo::default(); node.outputs.len()];
        Self {
            node,
            opset_imports,
            policy,
            inputs,
            outputs,
            interner,
        }
    }

    // === input access ===

    /// The op type of the node.
    pub fn op(&self) -> &str {
        &self.node.op_type
    }

    /// The number of input slots (including skipped optional ones).
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// The number of output slots.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Whether input slot `i` is present (a value is connected).
    pub fn has_input(&self, i: usize) -> bool {
        self.node
            .inputs
            .get(i)
            .map(Option::is_some)
            .unwrap_or(false)
    }

    /// The inferred type of input `i`, if resolved.
    pub fn input_type(&self, i: usize) -> Option<&TypeInfo> {
        self.inputs.get(i)?.type_info.as_ref()
    }

    /// The container [`ValueType`] of input `i`, if it is a container value.
    pub fn input_value_type(&self, i: usize) -> Option<&ValueType> {
        self.inputs.get(i)?.value_type.as_ref()
    }

    /// The inferred shape of input `i`, if resolved.
    pub fn input_shape(&self, i: usize) -> Option<&[DimExpr]> {
        self.input_type(i).map(|t| t.shape.as_slice())
    }

    /// The inferred dtype of input `i`, if resolved.
    pub fn input_dtype(&self, i: usize) -> Option<DataType> {
        self.input_type(i).map(|t| t.dtype)
    }

    /// The inferred rank of input `i`, if resolved.
    pub fn input_rank(&self, i: usize) -> Option<usize> {
        self.input_type(i).map(TypeInfo::rank)
    }

    /// The propagated shape-data of input `i`, if any.
    pub fn input_shape_data(&self, i: usize) -> Option<&ShapeData> {
        self.inputs.get(i)?.shape_data.as_ref()
    }

    // === output production ===

    /// Set the type of output `i`.
    pub fn set_output_type(&mut self, i: usize, type_info: TypeInfo) {
        if let Some(slot) = self.outputs.get_mut(i) {
            slot.type_info = Some(type_info);
        }
    }

    /// Set the container [`ValueType`] of output `i`.
    pub fn set_output_value_type(&mut self, i: usize, value_type: ValueType) {
        if let Some(slot) = self.outputs.get_mut(i) {
            slot.value_type = Some(value_type);
        }
    }

    /// Set the dtype and shape of output `i`.
    pub fn set_output(&mut self, i: usize, dtype: DataType, shape: TypedShape) {
        self.set_output_type(i, TypeInfo::new(dtype, shape));
    }

    /// Set the propagated shape-data of output `i`.
    pub fn set_output_shape_data(&mut self, i: usize, data: ShapeData) {
        if let Some(slot) = self.outputs.get_mut(i) {
            slot.shape_data = Some(data);
        }
    }

    /// Consume the context, returning the outputs the rule produced.
    pub fn into_outputs(self) -> Vec<NodeIo> {
        self.outputs
    }

    // === helpers available to rules ===

    /// The active merge policy.
    pub fn policy(&self) -> MergePolicy {
        self.policy
    }

    /// The effective opset version for `domain`.
    ///
    /// When asking about the active node's own domain, a node-local
    /// [`Node::version`](onnx_runtime_ir::Node::version) wins over the graph import. Other domains are resolved
    /// from the graph-level imports because a node-local version describes only
    /// that node's operator schema, not every domain a shape rule may consult.
    pub fn opset(&self, domain: &str) -> u64 {
        let domain = normalize_domain(domain);
        if domain == self.node.domain
            && let Some(version) = self.node.local_opset()
        {
            return version;
        }
        self.opset_imports.get(domain).copied().unwrap_or(1)
    }

    /// Mint a fresh opaque dimension.
    pub fn fresh_dim(&mut self) -> DimExpr {
        self.interner.fresh_dim()
    }

    /// Mutable access to the symbol interner, for shared container-type
    /// unification helpers that mint fresh dims.
    pub(crate) fn interner_mut(&mut self) -> &mut SymbolInterner {
        self.interner
    }

    /// Broadcast two shapes under NumPy rules. Where two distinct symbolic dims
    /// must be unified, keeps a deterministic representative symbol (see
    /// [`broadcast_dim`](Self::broadcast_dim)) rather than minting a fresh one.
    /// Errors only under [`MergePolicy::Strict`] on a concrete incompatibility.
    pub fn broadcast(
        &mut self,
        a: &[DimExpr],
        b: &[DimExpr],
    ) -> Result<TypedShape, ShapeInferError> {
        let rank = a.len().max(b.len());
        let mut out = Vec::with_capacity(rank);
        for axis in 0..rank {
            // Align from the right; missing leading dims are implicitly `1`.
            let da = dim_from_right(a, rank, axis);
            let db = dim_from_right(b, rank, axis);
            out.push(self.broadcast_dim(&da, &db)?);
        }
        Ok(out)
    }

    /// Broadcast a single pair of dimensions.
    pub fn broadcast_dim(&mut self, a: &DimExpr, b: &DimExpr) -> Result<DimExpr, ShapeInferError> {
        let ac = a.as_const();
        let bc = b.as_const();
        if ac == Some(1) {
            return Ok(b.clone());
        }
        if bc == Some(1) {
            return Ok(a.clone());
        }
        if a == b {
            return Ok(a.clone());
        }
        match (ac, bc) {
            (Some(x), Some(y)) => {
                if x == y {
                    Ok(a.clone())
                } else if self.policy == MergePolicy::Strict {
                    Err(ShapeInferError::Invalid {
                        op: self.node.op_type.clone(),
                        detail: format!("incompatible broadcast dims {x} and {y}"),
                    })
                } else {
                    // Permissive: two provably-unequal, non-1 concrete extents
                    // are genuinely incompatible. Rather than fabricate a
                    // `max(x, y)` that matches neither operand, degrade to a
                    // fresh symbol (an honest "unknown") so we never assert a
                    // bogus concrete dimension.
                    Ok(self.fresh_dim())
                }
            }
            // A concrete non-`1` extent dominates a symbolic one (the symbol
            // must broadcast up to it, or the model is invalid).
            (Some(_), None) => Ok(a.clone()),
            (None, Some(_)) => Ok(b.clone()),
            // Two distinct symbolic dims. In a valid model they must be equal at
            // this position (or one is 1), so keeping a single *representative*
            // symbol — rather than minting a fresh one no downstream consumer
            // could ever bind — is both conformance-safe and what reference
            // symbolic inference (onnxruntime) does. When both are bare symbols
            // we keep the one with the smaller id, which deterministically
            // prefers a named graph symbol (low-range, e.g. `batch`/`seq`) over
            // an anonymous fresh one (allocated at/above `0x8000_0000`); this is
            // what lets a data-dependent extent re-unify with the graph's real
            // dims (e.g. a `Shape`-driven `Expand` target). A derived expression
            // (not a bare symbol) has no id to compare, so it stays a fresh
            // opaque symbol — the honest "unknown".
            (None, None) => match (a.as_symbol(), b.as_symbol()) {
                (Some(sa), Some(sb)) => {
                    // Record the equivalence before returning. This is the SINGLE
                    // chokepoint every broadcasting handler funnels through
                    // (elementwise `broadcast`, `MatMul` batch dims, `Einsum`
                    // ellipsis, `Concat` non-concat axes, `Expand`), so recording
                    // here captures every symbol substitution inference performs —
                    // complete by construction, with no per-op enumeration. It is
                    // additive: the returned representative is unchanged.
                    self.interner.record_unification(sa, sb);
                    Ok(if sa.0 <= sb.0 { a.clone() } else { b.clone() })
                }
                _ => Ok(self.fresh_dim()),
            },
        }
    }
}

/// The dimension of `shape` at `axis` counting from the right of a rank-`rank`
/// aligned view; leading positions absent from `shape` are `1`.
fn dim_from_right(shape: &[DimExpr], rank: usize, axis: usize) -> DimExpr {
    let offset = rank - shape.len();
    if axis < offset {
        DimExpr::constant(1)
    } else {
        shape[axis - offset].clone()
    }
}

/// Reconcile an inferred shape with a value's declared IR shape under `policy`.
///
/// Returns the merged shape (each dim the more specific of the two). Under
/// [`MergePolicy::Strict`], a concrete-vs-concrete disagreement — or a rank
/// mismatch — is an error; symbolic disagreements are treated as naming and are
/// never conflicts, so that inference using freshly-minted symbols never
/// spuriously clashes with the loader's differently-named symbols.
pub fn merge_shapes(
    value: ValueId,
    inferred: &[DimExpr],
    declared: &[Dim],
    policy: MergePolicy,
) -> Result<Vec<DimExpr>, ShapeInferError> {
    if inferred.len() != declared.len() {
        if policy == MergePolicy::Strict {
            return Err(ShapeInferError::RankConflict {
                value,
                inferred: inferred.len(),
                declared: declared.len(),
            });
        }
        // Permissive: prefer the inferred (known) rank.
        return Ok(inferred.to_vec());
    }
    let mut out = Vec::with_capacity(inferred.len());
    for (axis, (inf, dec)) in inferred.iter().zip(declared.iter()).enumerate() {
        let dec_expr: DimExpr = (*dec).into();
        let merged = match (inf.as_const(), dec_expr.as_const()) {
            (Some(a), Some(b)) if a != b => {
                if policy == MergePolicy::Strict {
                    return Err(ShapeInferError::ShapeConflict {
                        value,
                        axis,
                        inferred: a,
                        declared: b,
                    });
                }
                // Permissive: keep the inferred value.
                inf.clone()
            }
            // Prefer whichever side is concrete (more specific).
            (Some(_), _) => inf.clone(),
            (None, Some(_)) => dec_expr,
            // Both symbolic: keep the inferred symbol.
            (None, None) => inf.clone(),
        };
        out.push(merged);
    }
    Ok(out)
}

/// Per-dimension agreement of two container element shapes. Differing ranks
/// yield `None` (unknown element rank); within a matching rank, structurally
/// equal dims (including symbolic ones) are preserved and disagreements degrade
/// to a fresh symbol.
///
/// Shared by every rule and control-flow reconciliation that unifies a sequence
/// element shape (`SequenceConstruct`/`SequenceInsert`, `If` branch outputs).
pub(crate) fn merge_element_shape(
    interner: &mut SymbolInterner,
    a: &[DimExpr],
    b: &[DimExpr],
) -> Option<TypedShape> {
    if a.len() != b.len() {
        return None;
    }
    let merged = a
        .iter()
        .zip(b.iter())
        .map(|(da, db)| {
            if da == db {
                da.clone()
            } else {
                interner.fresh_dim()
            }
        })
        .collect();
    Some(merged)
}

/// Unify two container element tensor types: dtypes must match (ONNX
/// homogeneity), shapes agree per dimension via [`merge_element_shape`]. A
/// missing shape on *either* side yields an unknown element shape — agreement
/// cannot be confirmed. Shared by `SequenceConstruct`/`SequenceInsert` and the
/// `If`/`Loop` container reconciliation.
pub(crate) fn unify_tensor_type(
    interner: &mut SymbolInterner,
    op: &str,
    acc: TensorType,
    other: TensorType,
) -> Result<TensorType, ShapeInferError> {
    if acc.dtype != other.dtype {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!(
                "sequence elements must share a dtype, found {:?} and {:?}",
                acc.dtype, other.dtype
            ),
        });
    }
    let shape = match (acc.shape, other.shape) {
        (Some(acc_shape), Some(other_shape)) => {
            merge_element_shape(interner, &acc_shape, &other_shape)
        }
        _ => None,
    };
    Ok(TensorType {
        dtype: acc.dtype,
        shape,
    })
}

/// Recursively unify two container [`ValueType`]s. Tensor leaves unify via
/// [`unify_tensor_type`]; `Sequence`/`Optional` recurse into their element type;
/// `Map` requires an equal key dtype and unifies its value type. Mismatched
/// variants (e.g. a `Sequence` against a `Tensor`, or differing `Map` keys) are
/// an error — the honest analogue of the tensor `If` branch dtype-mismatch.
pub(crate) fn unify_value_type(
    interner: &mut SymbolInterner,
    op: &str,
    a: &ValueType,
    b: &ValueType,
) -> Result<ValueType, ShapeInferError> {
    match (a, b) {
        (ValueType::Tensor(a), ValueType::Tensor(b)) => Ok(ValueType::Tensor(unify_tensor_type(
            interner,
            op,
            a.clone(),
            b.clone(),
        )?)),
        (ValueType::Sequence(a), ValueType::Sequence(b)) => {
            Ok(ValueType::sequence(unify_value_type(interner, op, a, b)?))
        }
        (ValueType::Optional(a), ValueType::Optional(b)) => Ok(ValueType::Optional(Box::new(
            unify_value_type(interner, op, a, b)?,
        ))),
        (ValueType::Map(ak, av), ValueType::Map(bk, bv)) if ak == bk => Ok(ValueType::Map(
            *ak,
            Box::new(unify_value_type(interner, op, av, bv)?),
        )),
        _ => Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("container types disagree: {a:?} vs {b:?}"),
        }),
    }
}

#[cfg(test)]
mod opset_resolution_tests {
    use super::*;
    use onnx_runtime_ir::NodeId;

    fn context_for(version: Option<i64>, imports: &HashMap<String, u64>) -> u64 {
        let mut node = Node::new(NodeId(0), "Swish", vec![], vec![]);
        node.version = version;
        let mut interner = SymbolInterner::new(0);
        let context = InferenceContext::new(
            &node,
            Vec::new(),
            imports,
            MergePolicy::default(),
            &mut interner,
        );
        context.opset("")
    }

    /// A usable node-local version wins, which is why the field exists.
    #[test]
    fn a_node_version_overrides_the_graph_import() {
        let imports = HashMap::from([(String::new(), 13)]);
        assert_eq!(context_for(Some(24), &imports), 24);
    }

    /// Values that cannot be a version defer to the graph rather than being
    /// believed.
    ///
    /// Shape inference used to convert `Node::version` with a bare
    /// `u64::try_from`, so `Some(0)` became opset 0 here while
    /// `Graph::effective_opset` ignored it — the same node meant different
    /// things to the shape rules and to dispatch, and a rule gated on a
    /// minimum opset would silently return unknown shapes.
    #[test]
    fn implausible_versions_defer_to_the_graph() {
        let imports = HashMap::from([(String::new(), 13)]);
        for version in [-1, 0, i64::MAX, i64::from(i32::MAX) + 1] {
            assert_eq!(
                context_for(Some(version), &imports),
                13,
                "version {version} is not usable and must not override the graph"
            );
        }
    }
}
