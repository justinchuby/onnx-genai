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
pub use executor::CacheStats;
pub use tensor::Tensor;

mod executor;
mod tensor;

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

        #[error("no model source: set a path or bytes on the builder")]
        NoModelSource,

        #[error("op type not supported by any available EP: {op_type}")]
        UnsupportedOp { op_type: String },

        #[error("value has a non-static (symbolic) shape and no binding to resolve it: {value}")]
        DynamicShape { value: String },

        #[error(
            "symbol {symbol} bound to conflicting sizes {first} and {second} across bound inputs"
        )]
        SymbolConflict {
            symbol: String,
            first: usize,
            second: usize,
        },

        #[error("input {name}: rank mismatch (graph declares rank {expected}, got {got})")]
        RankMismatch {
            name: String,
            expected: usize,
            got: usize,
        },

        #[error("no inferred shape for value {value} produced by op {op}")]
        UnresolvedShape { value: String, op: String },

        #[error("shape element count overflows usize for value {value} (dims {dims:?})")]
        ShapeOverflow { value: String, dims: Vec<usize> },

        #[error(
            "op {op} produced {got} data-dependent output shape(s) but has {expected} output(s)"
        )]
        OutputShapeCountMismatch {
            op: String,
            expected: usize,
            got: usize,
        },

        #[error("input {name}: dtype mismatch (expected {expected}, got {got})")]
        DtypeMismatch {
            name: String,
            expected: String,
            got: String,
        },

        #[error("input {name}: shape mismatch (expected {expected:?}, got {got:?})")]
        ShapeMismatch {
            name: String,
            expected: Vec<usize>,
            got: Vec<usize>,
        },

        #[error("internal executor error: {0}")]
        Internal(String),

        #[error(transparent)]
        Load(#[from] onnx_runtime_loader::LoaderError),

        #[error(transparent)]
        Ep(#[from] onnx_runtime_ep_api::EpError),

        #[error(transparent)]
        Ir(#[from] onnx_runtime_ir::IrError),

        #[error(transparent)]
        Graph(#[from] onnx_runtime_ir::GraphError),
    }

    /// Session `Result` alias.
    pub type Result<T> = std::result::Result<T, SessionError>;
}

use error::Result;

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
    ///
    /// Phase 1: device selection is CPU-only (`auto_detect` yields the CPU EP),
    /// the optimize stage is a no-op, and "compile" resolves a kernel per node
    /// into the shape-keyed cache.
    pub fn build(self) -> Result<InferenceSession> {
        // Phase 1 recognizes no session options; reject any provided key so
        // typos surface instead of being silently ignored (no compat shim).
        if let Some(key) = self.options.keys().next() {
            return Err(SessionError::UnknownOption { key: key.clone() });
        }
        // `memory_limit`, `enable_profiling`, and non-CPU `device` preferences
        // are accepted but not yet acted on in Phase 1 (CPU-only executor).
        let _ = (self.device, self.memory_limit, self.enable_profiling);

        let (graph, weights) = match (self.model_path, self.model_bytes) {
            (Some(path), _) => onnx_runtime_loader::load_model_with_weights(path)?,
            (None, Some(bytes)) => {
                onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".")?
            }
            (None, None) => return Err(SessionError::NoModelSource),
        };

        let mut session = InferenceSession::from_parts(graph, weights)?;
        if !self.warmup_shapes.is_empty() {
            session.warmup(&self.warmup_shapes)?;
        }
        Ok(session)
    }
}

/// A loaded model ready to run inference (§20.2).
pub struct InferenceSession {
    inputs: Vec<IoMeta>,
    outputs: Vec<IoMeta>,
    exec: executor::Executor,
}

fn io_meta(graph: &onnx_runtime_ir::Graph, values: &[onnx_runtime_ir::ValueId]) -> Vec<IoMeta> {
    values
        .iter()
        .map(|&vid| {
            let v = graph.value(vid);
            IoMeta {
                name: v.name.clone().unwrap_or_default(),
                dtype: v.dtype,
                shape: v.shape.clone(),
            }
        })
        .collect()
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

    /// Build a session directly from an in-memory IR [`Graph`](onnx_runtime_ir::Graph).
    ///
    /// Initializer bytes are read from the graph's inline [`WeightRef`]s, so no
    /// on-disk model or weight store is required. Useful for programmatically
    /// constructed graphs and tests.
    pub fn from_graph(graph: onnx_runtime_ir::Graph) -> Result<Self> {
        Self::from_parts(graph, std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()))
    }

    fn from_parts(
        graph: onnx_runtime_ir::Graph,
        weights: std::sync::Arc<onnx_runtime_loader::WeightStore>,
    ) -> Result<Self> {
        let inputs = io_meta(&graph, &graph.inputs);
        let outputs = io_meta(&graph, &graph.outputs);
        let ep = executor::auto_detect_cpu_ep()?;
        let exec = executor::Executor::build(graph, weights, ep)?;
        Ok(Self {
            inputs,
            outputs,
            exec,
        })
    }

    /// Start a configuration builder.
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Run inference with named inputs, returning the graph outputs in order.
    pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        self.exec.run(inputs)
    }

    /// Input metadata.
    pub fn inputs(&self) -> &[IoMeta] {
        &self.inputs
    }

    /// Output metadata.
    pub fn outputs(&self) -> &[IoMeta] {
        &self.outputs
    }

    /// Kernel-cache statistics (§11.1); useful to observe warmup/run reuse.
    pub fn cache_stats(&self) -> CacheStats {
        self.exec.cache_stats()
    }

    /// Pre-compile kernels for common shapes to avoid first-inference latency
    /// (§11.3). Phase-1 minimal: the compiled plan's shapes already key the
    /// cache, so this repopulates it for the plan; `shapes` are validated to
    /// name real inputs.
    pub fn warmup(&mut self, shapes: &[WarmupShape]) -> Result<()> {
        for ws in shapes {
            if !self.inputs.iter().any(|m| m.name == ws.input_name) {
                return Err(SessionError::InputNotFound {
                    name: ws.input_name.clone(),
                });
            }
        }
        self.exec.warmup()
    }
}

/// Load a model. Auto-detects the best available hardware (§20.2).
///
/// This is the primary entry point — no configuration required.
pub fn load(path: impl AsRef<Path>) -> Result<InferenceSession> {
    InferenceSession::load(path)
}
