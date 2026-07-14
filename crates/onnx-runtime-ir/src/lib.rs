//! # `onnx-runtime-ir`
//!
//! The Graph intermediate representation (IR) for the ORT 2.0 runtime.
//!
//! This crate is the **stable contract** that every downstream runtime crate
//! (`onnx-runtime-loader`, `onnx-runtime-ep-api`, `onnx-runtime-session`, …)
//! builds against. It is intentionally pure, safe Rust with no FFI and no
//! device dependencies so it compiles standalone on any target.
//!
//! It is a Rust port of the design captured in `docs/ORT2.md` §3 (Graph IR),
//! §5 (Striding & Layout) and §11 (Dynamic Shape), itself inspired by the
//! Python [`onnx-ir`](https://github.com/onnx/ir-py) package.
//!
//! ## What lives here
//!
//! | Concept | Type |
//! |---------|------|
//! | Element type | [`DataType`] |
//! | Symbolic / static shapes | [`Shape`], [`Dim`], [`SymbolConstraints`] |
//! | Physical strided layout | [`TensorLayout`], [`MemoryFormat`] |
//! | Device placement | [`DeviceType`], [`DeviceId`] |
//! | Graph values (SSA edges) | [`Value`], [`ValueId`] |
//! | Graph operations | [`Node`], [`NodeId`], [`Attribute`] |
//! | Constant / weight storage | [`TensorData`], [`SparseTensorData`], [`WeightRef`] |
//! | The graph itself | [`Graph`] |
//! | Errors | [`IrError`], [`GraphError`] |
//!
//! ## Design guarantees
//!
//! * **SSA-like:** every [`Value`] has at most one producer [`Node`]; node
//!   outputs are unique.
//! * **First-class layout & device:** every value carries a [`TensorLayout`]
//!   and an optional [`DeviceId`], unlike upstream ONNX / `onnx-ir`.
//! * **Mutable during optimization:** the [`Graph`] mutation API keeps
//!   producer/consumer edges consistent so optimization passes can rewrite it,
//!   then it is shared immutably via `Arc` once frozen.
//!
//! Deep algorithms whose full implementation belongs to a later task (e.g.
//! per-op shape inference) are represented here only by their data model; the
//! `Graph` operations that are cheap and foundational (topological ordering,
//! validation, edge rewiring, broadcasting, stride arithmetic) are fully
//! implemented and unit-tested so downstream crates compile against a stable,
//! working surface.

#![forbid(unsafe_code)]

mod arena;
mod device;
mod dtype;
mod error;
mod graph;
mod layout;
mod node;
mod shape;
mod tensor;
mod value;

pub use arena::{Arena, ArenaKey};
pub use device::{DeviceId, DeviceType};
pub use dtype::DataType;
pub use error::{GraphError, IrError, Result};
pub use graph::Graph;
pub use layout::{
    broadcast_shapes, compute_contiguous_strides, is_contiguous, MemoryFormat, TensorLayout,
};
pub use node::{Attribute, Node, NodeId};
pub use shape::{
    as_static_shape, is_fully_static, static_shape, Dim, Shape, SymbolConstraints, SymbolId,
};
pub use tensor::{SparseTensorData, TensorData, TypeProto, WeightRef};
pub use value::{Value, ValueId};
