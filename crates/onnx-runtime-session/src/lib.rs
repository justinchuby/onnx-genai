//! # `onnx-runtime-session`
//!
//! The user-facing session and inference API for the ORT 2.0 runtime
//! (see `docs/ORT2.md` §20). Design goal: **zero-config by default** — the user
//! never has to know what an execution provider is; the runtime auto-detects
//! hardware and picks a strategy.
//!
//! **Phase 1 skeleton:** the intent-based [`SessionBuilder`] and
//! [`InferenceSession`] surfaces are defined; `build`/`run` bodies are
//! `todo!()` pending the sequential executor (Phase 1 task `ort2-session`).
//!
//! ```ignore
//! let mut session = onnx_runtime_session::load("model.onnx")?;
//! let outputs = session.run(&[("input_ids", &tensor)])?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx_runtime_ir::{DataType, DeviceType, Shape};

pub use error::SessionError;

mod error {
    /// Errors produced by the session layer.
    #[derive(Debug, thiserror::Error)]
    pub enum SessionError {
        #[error("session not initialized")]
        NotInitialized,

        #[error("input not found: {name}")]
        InputNotFound { name: String },

        #[error("unknown session option: {key}")]
        UnknownOption { key: String },

        #[error(transparent)]
        Load(#[from] onnx_runtime_loader::LoaderError),

        #[error(transparent)]
        Ep(#[from] onnx_runtime_ep_api::EpError),

        #[error(transparent)]
        Ir(#[from] onnx_runtime_ir::IrError),
    }

    /// Session `Result` alias.
    pub type Result<T> = std::result::Result<T, SessionError>;
}

use error::Result;

/// An owned tensor handed to / returned from [`InferenceSession::run`].
///
/// Placeholder owned tensor for the Phase 1 skeleton; the full device-aware
/// `Tensor` (DLPack import/export, strided layout — §5.3/§5.4) is a downstream
/// deliverable.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    /// Raw little-endian element bytes.
    pub data: Vec<u8>,
}

/// Metadata describing a model input or output (§20.2).
#[derive(Clone, Debug)]
pub struct IoMeta {
    pub name: String,
    pub dtype: DataType,
    pub shape: Shape,
}

/// Intent-based device preference (§20.4). The runtime maps this to concrete
/// EPs during `build`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DevicePreference {
    /// Pick the best available device automatically.
    #[default]
    Auto,
    /// Prefer CPU execution.
    Cpu,
    /// Prefer a GPU / accelerator, optionally by ordinal.
    Gpu { index: Option<u32> },
    /// Pin to a specific device class + ordinal.
    Explicit { device_type: DeviceType, index: u32 },
}

/// A shape to pre-compile kernels for at session init (§11.3).
#[derive(Clone, Debug)]
pub struct WarmupShape {
    pub input_name: String,
    pub shape: Vec<usize>,
}

/// Builder for advanced session configuration (§20.6).
#[derive(Default)]
pub struct SessionBuilder {
    model_path: Option<PathBuf>,
    model_bytes: Option<Vec<u8>>,
    device: DevicePreference,
    memory_limit: Option<usize>,
    enable_profiling: bool,
    warmup_shapes: Vec<WarmupShape>,
    options: HashMap<String, String>,
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn model_bytes(mut self, bytes: &[u8]) -> Self {
        self.model_bytes = Some(bytes.to_vec());
        self
    }

    pub fn device(mut self, pref: DevicePreference) -> Self {
        self.device = pref;
        self
    }

    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    pub fn profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    pub fn warmup(mut self, shapes: Vec<WarmupShape>) -> Self {
        self.warmup_shapes = shapes;
        self
    }

    /// Set a namespaced option. Unknown keys are rejected at [`Self::build`].
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the session: load → detect device → optimize → compile → allocate.
    pub fn build(self) -> Result<InferenceSession> {
        let _ = (
            self.model_path,
            self.model_bytes,
            self.device,
            self.memory_limit,
            self.enable_profiling,
            self.warmup_shapes,
            self.options,
        );
        todo!("ort2-session: load model, select EPs, optimize, compile, allocate buffers")
    }
}

/// A loaded model ready to run inference (§20.2).
pub struct InferenceSession {
    inputs: Vec<IoMeta>,
    outputs: Vec<IoMeta>,
}

impl InferenceSession {
    /// Primary entry point: load a model with auto device detection.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::builder().model(path).build()
    }

    /// Load a model from an in-memory buffer.
    pub fn load_bytes(bytes: &[u8]) -> Result<Self> {
        Self::builder().model_bytes(bytes).build()
    }

    /// Start a configuration builder.
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Run inference with named inputs.
    pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        let _ = inputs;
        todo!("ort2-session: sequential executor over the compiled graph")
    }

    /// Input metadata.
    pub fn inputs(&self) -> &[IoMeta] {
        &self.inputs
    }

    /// Output metadata.
    pub fn outputs(&self) -> &[IoMeta] {
        &self.outputs
    }

    /// Pre-compile kernels for common shapes to avoid first-inference latency.
    pub fn warmup(&mut self, shapes: &[WarmupShape]) -> Result<()> {
        let _ = shapes;
        todo!("ort2-session: run dummy inferences to populate the kernel cache")
    }
}

/// Load a model. Auto-detects the best available hardware (§20.2).
///
/// This is the primary entry point — no configuration required.
pub fn load(path: impl AsRef<Path>) -> Result<InferenceSession> {
    InferenceSession::load(path)
}
