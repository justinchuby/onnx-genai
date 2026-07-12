//! Safe Rust wrapper over the ONNX Runtime C API.
//!
//! This provides a thin, safe layer over ORT's C API, giving us:
//! - Full control over IoBinding (for zero-copy KV cache passing)
//! - Latest ORT features (opset 24, tensor scatter)
//! - Support for all Execution Providers (CUDA, DirectML, QNN, CoreML, etc.)
//!
//! Design: reference the `ort` crate (pyke) for patterns, but use latest ORT directly.

pub mod env;
pub mod session;
pub mod value;
pub mod binding;
pub mod allocator;
pub mod error;

pub use env::Environment;
pub use session::{Session, SessionOptions};
pub use value::{Value, DataType};
pub use binding::IoBinding;
pub use allocator::{Allocator, MemoryInfo, MemoryType, AllocatorType};
pub use error::{OrtError, Result};
