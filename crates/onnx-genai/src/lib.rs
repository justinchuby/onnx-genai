//! # onnx-genai
//!
//! A Rust inference runtime for generative AI models built on ONNX Runtime.
//!
//! Reference implementation of the ONNX inference metadata standard
//! ([onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184)).

pub use onnx_genai_metadata as metadata;
pub use onnx_genai_kv as kv;
pub use onnx_genai_scheduler as scheduler;
pub use onnx_genai_engine as engine;
pub use onnx_genai_ort as ort;

pub use onnx_genai_engine::Engine;
