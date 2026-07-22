//! The inference context handed to each op rule, plus the supporting type
//! model: [`TypeInfo`], [`TypedShape`], [`MergePolicy`], and the
//! [`SymbolInterner`] that lowers derived [`DimExpr`]s back to IR [`Dim`]s.

use std::collections::HashMap;

use onnx_runtime_ir::{DataType, Dim, Node, SymbolId, ValueId};

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

/// The resolved inference state of a single input or output slot: an optional
/// type and an optional [`ShapeData`] side-value.
#[derive(Clone, Debug, Default)]
pub struct NodeIo {
    pub type_info: Option<TypeInfo>,
    pub shape_data: Option<ShapeData>,
}

impl NodeIo {
    /// An i/o slot carrying only a type.
    pub fn typed(type_info: TypeInfo) -> Self {
        Self {
            type_info: Some(type_info),
            shape_data: None,
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
    cache: HashMap<DimExpr, SymbolId>,
    /// Symbols minted during inference, to be registered on the graph.
    fresh: Vec<SymbolId>,
}

impl SymbolInterner {
    /// A new interner that allocates symbol ids starting at `next` (which must
    /// be greater than every symbol id already present in the graph).
    pub fn new(next: u32) -> Self {
        Self {
            next,
            cache: HashMap::new(),
            fresh: Vec::new(),
        }
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
    pub fn lower(&mut self, expr: &DimExpr) -> Dim {
        // An overflowed (unknown) expression has no representable value and must
        // not alias other overflows via the cache: mint a distinct fresh symbol.
        if expr.is_overflow() {
            return Dim::Symbolic(self.fresh_symbol());
        }
        if let Some(n) = expr.as_const() {
            if n >= 0 {
                return Dim::Static(n as usize);
            }
            // A negative extent is nonsensical; degrade to a fresh symbol.
            return Dim::Symbolic(self.fresh_symbol());
        }
        if let Some(s) = expr.as_symbol() {
            return Dim::Symbolic(s);
        }
        if let Some(&id) = self.cache.get(expr) {
            return Dim::Symbolic(id);
        }
        let id = self.fresh_symbol();
        self.cache.insert(expr.clone(), id);
        Dim::Symbolic(id)
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

    /// The imported opset version for `domain` (the default `""`/`ai.onnx`
    /// domain falls back to the highest imported ai.onnx version, else `1`).
    pub fn opset(&self, domain: &str) -> u64 {
        if domain.is_empty() || domain == "ai.onnx" {
            self.opset_imports
                .get("")
                .or_else(|| self.opset_imports.get("ai.onnx"))
                .copied()
                .unwrap_or(1)
        } else {
            self.opset_imports.get(domain).copied().unwrap_or(1)
        }
    }

    /// Mint a fresh opaque dimension.
    pub fn fresh_dim(&mut self) -> DimExpr {
        self.interner.fresh_dim()
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
                (Some(sa), Some(sb)) => Ok(if sa.0 <= sb.0 { a.clone() } else { b.clone() }),
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
