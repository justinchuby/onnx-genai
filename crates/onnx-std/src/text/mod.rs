//! Textual representation of ONNX models (ONNX_RS §5).
//!
//! Implements both directions of the readable format: [`to_text`] renders an
//! owned model and [`from_text`] reconstructs one. Weight and tensor payloads
//! are intentionally represented by typed references/placeholders rather than
//! embedding binary data.

mod de;
mod extensions;
mod ser;

pub use de::from_text;
pub use ser::{PrintOptions, to_text, to_text_with};
