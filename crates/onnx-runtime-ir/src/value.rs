//! Graph values: the typed, shaped SSA edges between nodes.

use std::collections::HashSet;
use std::fmt;

use crate::TypeProto;
use crate::arena::ArenaKey;
use crate::device::DeviceId;
use crate::dtype::DataType;
use crate::layout::TensorLayout;
use crate::node::NodeId;
use crate::shape::Shape;

/// Unique identifier for a [`Value`] within a [`Graph`](crate::Graph).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ValueId(pub u32);

impl ArenaKey for ValueId {
    fn from_raw(raw: u32) -> Self {
        ValueId(raw)
    }
    fn to_raw(self) -> u32 {
        self.0
    }
}

/// A node input slot that uses a value.
pub type Usage = (NodeId, u32);

/// The unordered internal set of input slots that consume a value.
///
/// Iteration is intentionally exposed only through sorted snapshots so hash
/// iteration order cannot affect graph rewrites or serialized output.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Consumers {
    uses: HashSet<Usage>,
}

impl Consumers {
    pub(crate) fn insert(&mut self, node: NodeId, input_index: u32) {
        self.uses.insert((node, input_index));
    }

    pub(crate) fn remove(&mut self, node: NodeId, input_index: u32) -> bool {
        self.uses.remove(&(node, input_index))
    }

    pub(crate) fn contains(&self, node: NodeId, input_index: u32) -> bool {
        self.uses.contains(&(node, input_index))
    }

    /// Number of consuming input slots.
    pub fn len(&self) -> usize {
        self.uses.len()
    }

    /// Whether no node input slot consumes the value.
    pub fn is_empty(&self) -> bool {
        self.uses.is_empty()
    }

    /// Consuming input slots sorted by `(NodeId, input_index)`.
    pub fn uses(&self) -> Vec<Usage> {
        let mut uses: Vec<_> = self.uses.iter().copied().collect();
        uses.sort_unstable_by_key(|&(node, input_index)| (node.0, input_index));
        uses
    }

    /// Distinct consuming nodes sorted by ascending [`NodeId`].
    pub fn nodes(&self) -> Vec<NodeId> {
        let mut nodes: Vec<_> = self.uses.iter().map(|&(node, _)| node).collect();
        nodes.sort_unstable_by_key(|node| node.0);
        nodes.dedup();
        nodes
    }
}

impl fmt::Debug for Consumers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.uses()).finish()
    }
}

/// A value flowing through the graph — the output of one node and the input of
/// zero or more others (SSA form: at most one producer).
///
/// Carries a first-class [`TensorLayout`] and optional [`DeviceId`] placement,
/// beyond ONNX's plain type+shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub id: ValueId,
    /// Optional name (graph inputs/outputs and initializers are named; interior
    /// SSA values may be anonymous).
    pub name: Option<String>,
    /// Full declared ONNX type, before tensor inference. This retains recursive
    /// containers, opaque types, denotations and absent rank/dimension fields.
    /// `dtype`/`shape` below are tensor inference facts, not substitutes for this
    /// descriptor. Non-tensor and untyped values use `DataType::Undefined`.
    pub declared_type: Option<TypeProto>,
    pub dtype: DataType,
    pub shape: Shape,
    pub(crate) tensor_type_known: bool,
    pub(crate) tensor_shape_known: bool,
    pub layout: TensorLayout,
    /// Device placement, filled in by the placement pass.
    pub device: Option<DeviceId>,
    /// The node that produces this value, or `None` for graph inputs and
    /// initializers.
    pub producer: Option<NodeId>,
    /// Input slots that consume this value, keyed by `(node, input_index)`.
    pub consumers: Consumers,
    /// Whether this value is present in [`Graph::inputs`](crate::Graph::inputs).
    pub is_graph_input: bool,
    /// Whether this value is present in [`Graph::outputs`](crate::Graph::outputs).
    pub is_graph_output: bool,
}

impl Value {
    /// A new anonymous value with a contiguous default layout and no edges.
    pub fn new(id: ValueId, dtype: DataType, shape: Shape) -> Self {
        Self {
            id,
            name: None,
            declared_type: None,
            dtype,
            shape,
            tensor_type_known: dtype != DataType::Undefined,
            tensor_shape_known: true,
            layout: TensorLayout::contiguous(),
            device: None,
            producer: None,
            consumers: Consumers::default(),
            is_graph_input: false,
            is_graph_output: false,
        }
    }

    /// Whether this value has no producing node (graph input / initializer).
    pub fn is_source(&self) -> bool {
        self.producer.is_none()
    }

    /// Known tensor element type; never projects a container into a tensor.
    pub fn tensor_dtype(&self) -> Option<DataType> {
        (self.tensor_type_known && self.dtype != DataType::Undefined && self.permits_tensor_facts())
            .then_some(self.dtype)
    }

    /// Known-rank tensor shape. `None` differs from the scalar shape `Some([])`.
    pub fn tensor_shape(&self) -> Option<&Shape> {
        (self.tensor_shape_known && self.permits_tensor_facts()).then_some(&self.shape)
    }

    /// The tensor rank, or `None` for unknown-rank and non-tensor values.
    pub fn rank(&self) -> Option<usize> {
        self.tensor_shape().map(Vec::len)
    }

    fn permits_tensor_facts(&self) -> bool {
        self.declared_type.as_ref().is_none_or(|tp| {
            matches!(
                tp.value,
                None | Some(crate::onnx::type_proto::Value::TensorType(_))
            )
        })
    }
}
