//! Native model input preprocessing defined by DESIGN §35.
//!
//! This crate implements reusable image and audio preprocessing in Rust without
//! requiring ONNX Runtime Extensions. Model servers, CLIs, and other frontends
//! can share the same metadata-driven transformations before invoking a model.
//!
//! ## Prompt token expansion seam
//!
//! After image preprocessing and prompt tokenization, callers can pass
//! [`image::ImageTensor::tiling_summary`] to
//! [`image::expand_image_placeholders`]. This replaces one placeholder per image
//! before the engine computes sequence length and allocates KV cache pages.

pub mod audio;
pub mod image;
