//! Native model input preprocessing defined by DESIGN §35.
//!
//! This crate implements reusable image and audio preprocessing in Rust without
//! requiring ONNX Runtime Extensions. Model servers, CLIs, and other frontends
//! can share the same metadata-driven transformations before invoking a model.

pub mod audio;
pub mod image;
