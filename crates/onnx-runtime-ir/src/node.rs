//! Graph nodes (operations) and their attributes.

use std::collections::HashMap;

use crate::arena::ArenaKey;
use crate::device::DeviceId;
use crate::graph::Graph;
use crate::shape::Shape;
use crate::tensor::{SparseTensorData, TensorData, TypeProto};
use crate::value::ValueId;

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
    pub op_type: String,
    /// Operator domain (`""` == the default ONNX domain).
    pub domain: String,
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
            op_type: op_type.into(),
            domain: String::new(),
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
}

/// An ONNX operator attribute. Covers all attribute value kinds.
#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
    Tensor(TensorData),
    SparseTensor(SparseTensorData),
    /// A subgraph body (control-flow ops: If/Loop/Scan). Stored inline; the
    /// owning [`Graph`] also indexes it in `subgraphs` for traversal.
    Graph(Box<Graph>),
    Graphs(Vec<Graph>),
    TypeProto(TypeProto),
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

    /// The `&str` value, if this is an [`Attribute::String`].
    pub fn as_str(&self) -> Option<&str> {
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
