//! Graph nodes (operations) and their attributes.

use std::collections::HashMap;

use crate::arena::ArenaKey;
use crate::device::DeviceId;
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::{SparseTensorData, TensorData, TypeProto};
use crate::value::ValueId;

/// The operator domain for operators this runtime defines itself.
///
/// Anything we invent goes here, and nothing we invent goes in `com.microsoft`
/// or the default ONNX domain: those namespaces belong to their owners, and an
/// operator placed in one of them claims a provenance and a specification it
/// does not have. Consumers reading a graph use the domain to decide whose
/// definition applies, so getting it wrong is a factual error about the model,
/// not a naming preference.
pub const RUNTIME_DOMAIN: &str = "pkg.nxrt";

/// Unique identifier for a [`Node`] within a [`Graph`](crate::Graph).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

impl ArenaKey for NodeId {
    fn from_raw(raw: u32) -> Self {
        NodeId(raw)
    }
    fn to_raw(self) -> u32 {
        self.0
    }
}

/// An operation in the graph.
///
/// Inputs are `Option<ValueId>` because ONNX ops may have optional (skipped)
/// inputs represented by empty names; a `None` slot preserves positional
/// arity. Outputs are always present (SSA values).
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    /// Optional ONNX node name (`""` means unnamed).
    pub name: String,
    pub op_type: String,
    /// Operator domain (`""` == the default ONNX domain).
    pub domain: String,
    /// Opset version of the operator called.
    ///
    /// If `None`, the version is unspecified and follows the owning graph's
    /// `opset_imports`. This property is special to ONNX IR to allow mixed opset
    /// usage in a graph for more flexible graph transformations; it does not
    /// exist in the ONNX protobuf spec. For example, a fusion may emit an
    /// opset-24 `Swish` into a graph whose other default-domain nodes still use
    /// the graph's older exported opset, avoiding a false graph-wide upgrade.
    pub version: Option<i64>,
    pub inputs: Vec<Option<ValueId>>,
    pub outputs: Vec<ValueId>,
    pub attributes: HashMap<String, Attribute>,
    pub doc_string: Option<String>,
    /// Device placement, filled in by the placement pass.
    pub device: Option<DeviceId>,
    /// Position in the final execution schedule, filled in by the scheduler.
    pub exec_order: Option<usize>,
}

impl Node {
    /// A new node with the given op type and edges, and no attributes.
    pub fn new(
        id: NodeId,
        op_type: impl Into<String>,
        inputs: Vec<Option<ValueId>>,
        outputs: Vec<ValueId>,
    ) -> Self {
        Self {
            id,
            name: String::new(),
            op_type: op_type.into(),
            domain: String::new(),
            version: None,
            inputs,
            outputs,
            attributes: HashMap::new(),
            doc_string: None,
            device: None,
            exec_order: None,
        }
    }

    /// Iterate over the present (non-skipped) input value ids.
    pub fn input_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.inputs.iter().filter_map(|slot| *slot)
    }

    /// Look up an attribute by name.
    pub fn attr(&self, name: &str) -> Option<&Attribute> {
        self.attributes.get(name)
    }

    /// Whether this node belongs to the default ONNX operator domain.
    ///
    /// Relies on the post-load invariant that the loader canonicalizes the
    /// default domain to `""` (see [`crate::normalize_domain`]), so this is a
    /// simple emptiness test — the `"ai.onnx"` spelling never reaches loaded IR.
    #[inline]
    #[must_use]
    pub fn is_default_domain(&self) -> bool {
        self.domain.is_empty()
    }

    /// This node's own opset version, if it names one that could be real.
    ///
    /// The single owner of that judgement, so every subsystem reading
    /// [`Node::version`] agrees about the same node. Values that cannot be an
    /// opset — negative, zero, or beyond what any opset could plausibly reach —
    /// yield `None`: a node claiming them describes IR that is already wrong,
    /// and the graph's own import is the better answer.
    ///
    /// Callers with a graph in hand should prefer
    /// [`Graph::effective_opset`](crate::Graph::effective_opset), which falls
    /// back to that import. Shape inference has only the node and a map of
    /// imports, so it uses this directly.
    #[inline]
    #[must_use]
    pub fn local_opset(&self) -> Option<u64> {
        match self.version {
            Some(version) if (1..=MAX_PLAUSIBLE_OPSET).contains(&version) => Some(version as u64),
            _ => None,
        }
    }
}

/// No ONNX opset will plausibly reach this, so a larger `Node::version` is a
/// mistake rather than a version we do not know yet.
pub(crate) const MAX_PLAUSIBLE_OPSET: i64 = 1_000;

/// An ONNX operator attribute. Covers all attribute value kinds.
#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Float(f32),
    /// An ONNX `STRING` attribute. Stored as **raw bytes**, not `String`, so
    /// that the load/dump path round-trips the payload byte-exactly: ONNX
    /// `STRING` attributes are arbitrary byte strings (e.g. an opaque compiled
    /// blob) that are not guaranteed to be valid UTF-8. Use [`Attribute::as_str`]
    /// to view the bytes as UTF-8 text when that is meaningful.
    String(Vec<u8>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    /// An ONNX `STRINGS` attribute — a list of raw byte strings (see
    /// [`Attribute::String`] for why bytes rather than `String`).
    Strings(Vec<Vec<u8>>),
    Tensor(TensorData),
    Tensors(Vec<TensorData>),
    SparseTensor(SparseTensorData),
    SparseTensors(Vec<SparseTensorData>),
    /// A subgraph body (control-flow ops: If/Loop/Scan). Stored inline; the
    /// owning [`Graph`] also indexes it in `subgraphs` for traversal.
    Graph(Box<Graph>),
    Graphs(Vec<Graph>),
    TypeProto(TypeProto),
    TypeProtos(Vec<TypeProto>),
}

impl Attribute {
    /// The `i64` value, if this is an [`Attribute::Int`].
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attribute::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The `f32` value, if this is an [`Attribute::Float`].
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Attribute::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// The value as UTF-8 text, if this is an [`Attribute::String`] whose bytes
    /// are valid UTF-8. Returns `None` for a non-string attribute or for string
    /// bytes that are not valid UTF-8 (e.g. an opaque binary payload).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attribute::String(v) => std::str::from_utf8(v).ok(),
            _ => None,
        }
    }

    /// The raw bytes of an [`Attribute::String`], regardless of whether they are
    /// valid UTF-8. Returns `None` for any other attribute kind.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Attribute::String(v) => Some(v),
            _ => None,
        }
    }

    /// The `&[i64]` slice, if this is an [`Attribute::Ints`].
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Attribute::Ints(v) => Some(v),
            _ => None,
        }
    }

    /// Interpret an `Ints` attribute as a shape of static dims.
    pub fn as_shape(&self) -> Option<Shape> {
        self.as_ints()
            .map(|v| v.iter().map(|&d| (d as usize).into()).collect())
    }
}
