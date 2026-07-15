//! # `onnx-rs`
//!
//! A pure-Rust ONNX **standard library** — model I/O, textual dump, and an
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
//! | §5         | [`Model::to_text`] readable dump | ✅ (dump only) |
//! | §7         | [`schema`] op schemas and opset-aware registry | ✅ |
//! | §8         | [`check::OnnxChecker`] extensible validator | ✅ (schema-aware) |
//!
//! Deferred to later waves (see `// FOLLOW-UP` markers): text *parse-back*
//! (§5.4), JSON / TextProto (§6), the remaining checker rule set (§8.2), shape
//! inference wrapping (§9), the version converter (§10),
//! custom-op registration (§11), and Python bindings (§12).
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
mod error;
mod model;
pub mod schema;
pub mod text;

pub use error::{Error, Result};
pub use model::{Model, load_model, save_model};

// Re-export the shared IR and the metadata/weight types so downstream users can
// build and inspect models without depending on the runtime crates directly
// (ONNX_RS §4.1: the IR is shared, not re-defined here).
pub use onnx_runtime_ir as ir;
pub use onnx_runtime_loader::{ModelMetadata, WeightStore};

pub use check::{OnnxChecker, Severity, ValidationResult, ValidationRule, Violation};
pub use schema::{
    AttributeDefault, AttributeSpec, AttributeType, InputSpec, OpSchema, OutputSpec, SchemaError,
    SchemaRegistry, TypeConstraint,
};
pub use text::PrintOptions;
