//! Graph values: the typed, shaped SSA edges between nodes.

use crate::arena::ArenaKey;
use crate::device::DeviceId;
use crate::dtype::DataType;
use crate::layout::TensorLayout;
use crate::node::NodeId;
use crate::shape::Shape;

/// Unique identifier for a [`Value`] within a [`Graph`](crate::Graph).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub u32);

impl ArenaKey for ValueId {
    fn from_raw(raw: u32) -> Self {
        ValueId(raw)
    }
    fn to_raw(self) -> u32 {
        self.0
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
    pub dtype: DataType,
    pub shape: Shape,
    pub layout: TensorLayout,
    /// Device placement, filled in by the placement pass.
    pub device: Option<DeviceId>,
    /// The node that produces this value, or `None` for graph inputs and
    /// initializers.
    pub producer: Option<NodeId>,
    /// Nodes that consume this value (one entry per consuming input slot).
    pub consumers: Vec<NodeId>,
}

impl Value {
    /// A new anonymous value with a contiguous default layout and no edges.
    pub fn new(id: ValueId, dtype: DataType, shape: Shape) -> Self {
        Self {
            id,
            name: None,
            dtype,
            shape,
            layout: TensorLayout::contiguous(),
            device: None,
            producer: None,
            consumers: Vec::new(),
        }
    }

    /// Whether this value has no producing node (graph input / initializer).
    pub fn is_source(&self) -> bool {
        self.producer.is_none()
    }

    /// The static rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}
