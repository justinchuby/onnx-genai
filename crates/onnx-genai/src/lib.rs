//! # onnx-genai
//!
//! A Rust inference runtime for generative AI models built on ONNX Runtime.
//!
//! This is the reference implementation of the ONNX inference metadata standard
//! proposed in [onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184).
//!
//! ## Architecture
//!
//! - **onnx-genai-metadata** — Inference metadata parser and types
//! - **onnx-genai-kv** — Paged KV cache with tiered storage and prefix sharing
//! - **onnx-genai-scheduler** — Continuous batching scheduler with preemption
//! - **onnx-genai-engine** — Generation engine tying everything together
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use onnx_genai::Engine;
//! use std::path::Path;
//!
//! let engine = Engine::from_dir(
//!     Path::new("./models/phi-4/"),
//!     Default::default(),
//! ).unwrap();
//! ```

pub use onnx_genai_metadata as metadata;
pub use onnx_genai_kv as kv;
pub use onnx_genai_scheduler as scheduler;
pub use onnx_genai_engine as engine;

pub use onnx_genai_engine::Engine;
