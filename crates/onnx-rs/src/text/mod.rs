//! Textual representation of ONNX models (ONNX_RS §5).
//!
//! Implements both directions of the readable format: [`print`] renders an
//! owned model and [`parse_model`] reconstructs one. Weight and tensor payloads
//! are intentionally represented by typed references/placeholders rather than
//! embedding binary data.

mod parser;
mod printer;

pub use parser::parse_model;
pub use printer::{PrintOptions, print, print_with};
