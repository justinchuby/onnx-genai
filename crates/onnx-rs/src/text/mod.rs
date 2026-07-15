//! Textual representation of ONNX models (ONNX_RS §5).
//!
//! This module implements the **dump** direction — `Model` → readable text —
//! following the ONNX textual format design (§5.2/§5.3): SSA-like node syntax,
//! `dtype[shape]` types, `<attr = value>` attribute blocks, and weights shown as
//! *references* (name + type) rather than inlined binary data.
//!
//! The parse-back direction (`text → Model`) is intentionally deferred to a
//! later wave; see the crate-level docs and the `// FOLLOW-UP` markers.

mod printer;

pub use printer::{PrintOptions, print, print_with};
