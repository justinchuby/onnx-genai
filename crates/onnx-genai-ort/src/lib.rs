//! Safe Rust wrapper over the ONNX Runtime C API.
//!
//! This provides a thin, safe layer over ORT's C API, giving us:
//! - Full control over IoBinding (for zero-copy KV cache passing)
//! - Latest ORT features (opset 24, tensor scatter)
//! - Support for all Execution Providers (CUDA, DirectML, QNN, CoreML, etc.)
//!
//! Design: reference the `ort` crate (pyke) for patterns, but use latest ORT directly.

pub mod allocator;
pub mod binding;
pub mod env;
pub mod error;
pub mod loader;
pub mod session;
pub mod tokenizer;
pub mod value;

pub use allocator::{Allocator, AllocatorType, MemoryInfo, MemoryType};
pub use binding::IoBinding;
pub use env::Environment;
pub use error::{OrtError, Result};
pub use loader::{ModelDirectory, PipelineModelDirectory, PipelineModels, PipelineTokenizerPaths};
pub use session::{Session, SessionOptions, TensorInfo};
pub use tokenizer::Tokenizer;
pub use value::{DataType, Value};
