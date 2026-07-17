//! # `onnx-rs`
//!
//! A pure-Rust ONNX **standard library** — model I/O, textual format, and an
//! extensible validator — built on the shared runtime IR. This is the first
//! wave of the design captured in `docs/ONNX_RS.md`.
//!
//! ## Relationship to the rest of the workspace
//!
//! `onnx-rs` does **not** define its own IR. Per ONNX_RS §4.1 ("Shared Crate"),
//! the `Graph` / `Node` / `Value` / `Tensor` / `WeightRef` types are reused
//! verbatim from [`onnx_runtime_ir`], and the protobuf parse/encode stack is
//! reused from [`onnx_runtime_loader`]. `onnx-rs` is the *standard-library*
//! layer on top: ergonomic model I/O, a human-readable text format, and a
//! checker — the concerns the design assigns to the ONNX library rather than the
//! inference runtime.
//!
//! ```text
//! onnx-rs  ──(standard lib: load/save/print/check)
//!    │
//!    └─ depends on ─▶ onnx-runtime-ir      (shared Graph/Node/Value IR)
//!    └─ depends on ─▶ onnx-runtime-loader  (protobuf serde + weight mmap)
//! ```
//!
//! ## What this wave covers
//!
//! | ONNX_RS §  | Feature | Status |
//! |------------|---------|--------|
//! | §3.4 / §4  | [`load_model`] / [`save_model`] round-trip | ✅ |
//! | §5         | [`Model::to_text`] / [`Model::from_text`] | ✅ |
//! | §7         | [`schema`] op schemas and opset-aware registry | ✅ |
//! | §8         | [`check::OnnxChecker`] extensible validator | ✅ (schema-aware) |
//! | §9         | [`shape::infer_shapes`] symbolic shape inference | ✅ |
//! | §10        | [`version::VersionConverter`] opset conversion | ✅ |
//! | §6.2       | [`textproto`] protobuf TextFormat I/O | ✅ |
//!
//! JSON and protobuf TextFormat are descriptor-driven from the exact vendored
//! ONNX proto, so every bound message, field, oneof, and enum is covered. The
//! readable text DSL appends a protobuf-TextFormat extension containing only
//! fields outside the graph syntax. The parsed DSL is authoritative for every
//! field it represents, while the explicit extension preserves the remainder.
//!
//! Deferred to later waves (see `// FOLLOW-UP` markers): the complete operator
//! catalog, remaining function/IR-gate checker rules, custom-op registration
//! (§11), and Python bindings (§12). Training semantics are out of scope.
//!
//! ## Example
//!
//! ```no_run
//! let model = onnx_rs::load_model("model.onnx")?;
//! println!("{}", model.to_text());
//! let report = model.validate();
//! assert!(report.is_valid());
//! onnx_rs::save_model(&model, "roundtrip.onnx")?;
//! # Ok::<(), onnx_rs::Error>(())
//! ```

#![forbid(unsafe_code)]

pub mod check;
mod codec;
mod error;
pub mod json;
mod model;
mod proto_serde;
pub mod schema;
pub mod shape;
pub mod text;
pub mod textproto;
pub mod version;

pub use error::{Error, Result};
pub use model::{
    DeviceConfigurationProto, IntIntListEntryProto, Model, NodeDeviceConfigurationProto,
    OpaqueProto, ShardedDimProto, ShardingSpecProto, SimpleShardedDimProto, load_model, save_model,
    simple_sharded_dim_proto,
};

// Re-export the shared IR and the metadata/weight types so downstream users can
// build and inspect models without depending on the runtime crates directly
// (ONNX_RS §4.1: the IR is shared, not re-defined here).
pub use onnx_runtime_ir as ir;
pub use onnx_runtime_loader::{ModelMetadata, WeightStore};

pub use check::{OnnxChecker, Severity, ValidationResult, ValidationRule, Violation};
pub use codec::{Json, Text, TextCodec, TextProto};
pub use schema::{
    AttributeDefault, AttributeSpec, AttributeType, InputSpec, OpSchema, OutputSpec, SchemaError,
    SchemaRegistry, TypeConstraint,
};
pub use shape::{ShapeError, ShapeInferenceResult, infer_shapes};
pub use text::{PrintOptions, from_text, to_text, to_text_with};
pub use version::{
    AdaptResult, ConvertError, ConvertReport, IncompatibleOp, OpAdapter, VersionConverter,
};
