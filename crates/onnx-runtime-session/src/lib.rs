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

pub use epcontext::{
    CompiledPartition, EpContextPlacement, dump_session_ep_context, load_ep_context_nodes,
};
pub use onnx_runtime_loader::{EpContextDumpConfig, EpContextPartition, Model as EncoderModel};
pub use error::SessionError;
pub use executor::CacheStats;
pub use tensor::Tensor;

mod epcontext;
mod executor;
mod tensor;

/// Operator-set version associated with an operator dispatch failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpsetVersion {
    /// The model declares this version for the operator's domain.
    Known(u64),
    /// The model has no opset import for the operator's domain.
    Undeclared,
}

impl std::fmt::Display for OpsetVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Known(version) => version.fmt(f),
            Self::Undeclared => f.write_str("<undeclared>"),
        }
    }
}

mod error {
    use super::OpsetVersion;

    struct UnsupportedOpRemediation<'a> {
        opset: OpsetVersion,
        domain: &'a str,
    }

    impl std::fmt::Display for UnsupportedOpRemediation<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.opset == OpsetVersion::Undeclared {
                write!(
                    f,
                    "declare an opset_import for domain {:?} in the model, ",
                    self.domain
                )?;
            }
            f.write_str(
                "enable another EP that supports this operator and opset, convert or decompose \
                 the model operator, or file an nxrt issue with the model details",
            )
        }
    }

    fn unsupported_op_remediation(
        opset: OpsetVersion,
        domain: &str,
    ) -> UnsupportedOpRemediation<'_> {
        UnsupportedOpRemediation { opset, domain }
    }

    /// Errors produced by the session layer.
    #[derive(Debug, thiserror::Error)]
    pub enum SessionError {
        #[error("session not initialized")]
        NotInitialized,

        #[error("input not found: {name}")]
        InputNotFound { name: String },

        #[error("unknown session option: {key}")]
        UnknownOption { key: String },

        #[error("invalid value {value:?} for session option {key:?}: expected one of {expected}")]
        InvalidOption {
            key: String,
            value: String,
            expected: String,
        },

        #[error("no model source: set a path or bytes on the builder")]
        NoModelSource,

        #[error(
            "unsupported operator {domain}::{op_type}: no available execution provider has a \
             kernel; node {node}, opset {opset}; consulted execution providers (priority order): \
             {execution_providers}. To fix: {remediation}",
            remediation = unsupported_op_remediation(*.opset, .domain)
        )]
        UnsupportedOp {
            op_type: String,
            domain: String,
            node: String,
            opset: OpsetVersion,
            execution_providers: String,
        },

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

        #[error(
            "EPContext reference node (main_context=0) has no matching primary \
             (source={source_key:?}, partition_name={partition_name:?})"
        )]
        DanglingEpContext {
            source_key: Option<String>,
            partition_name: Option<String>,
        },

        #[error(transparent)]
        Load(#[from] onnx_runtime_loader::LoaderError),

        #[error(transparent)]
        Ep(#[from] onnx_runtime_ep_api::EpError),

        #[error(transparent)]
        Ir(#[from] onnx_runtime_ir::IrError),

        #[error(transparent)]
        Graph(#[from] onnx_runtime_ir::GraphError),

        #[error(transparent)]
        Optimize(#[from] onnx_runtime_optimizer::OptimizerError),

        #[error(transparent)]
        ShapeInfer(#[from] onnx_runtime_shape_inference::ShapeInferError),
    }

    impl SessionError {
        pub(crate) fn unsupported_op(
            node: &onnx_runtime_ir::Node,
            node_id: onnx_runtime_ir::NodeId,
            opset: u64,
            execution_providers: impl Into<String>,
        ) -> Self {
            let domain = if node.domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                node.domain.clone()
            };
            let node_display = if node.name.is_empty() {
                format!("<unnamed node #{}>", node_id.0)
            } else {
                format!("{:?}", node.name)
            };
            let opset = if opset == u64::MAX {
                OpsetVersion::Undeclared
            } else {
                OpsetVersion::Known(opset)
            };
            Self::UnsupportedOp {
                op_type: node.op_type.clone(),
                domain,
                node: node_display,
                opset,
                execution_providers: execution_providers.into(),
            }
        }
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

/// Graph-optimization level for the session's `optimize` pipeline stage
/// (`docs/ORT2.md` §18). Selected via the generic `"optimization"` session
/// option (see [`SessionBuilder::option`]).
///
/// The default is [`OptimizationLevel::None`]: with optimization off the graph
/// reaches the executor exactly as the loader produced it, so default runtime
/// behavior is byte-identical to a build with no optimizer wired in at all.
///
/// This is a generic, model-agnostic knob — no level ever special-cases a model
/// name or op. Higher levels simply enable more of the device-independent pass
/// pipeline from [`onnx_runtime_optimizer`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No passes — the `optimize` stage is a no-op (default).
    #[default]
    None,
    /// Structure-preserving passes only: constant folding then dead-node
    /// elimination. No operator fusion, so the op set the executor sees is a
    /// subset of the loaded graph's.
    Basic,
    /// The full device-independent pipeline: constant folding, dead-node
    /// elimination, and operator fusion (which can introduce fused
    /// `com.microsoft` contrib ops such as `LayerNormalization`).
    All,
}

impl OptimizationLevel {
    /// Parse the `"optimization"` option value. Accepts `"none"`, `"basic"`,
    /// and `"all"` (case-insensitive).
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "0" => Some(Self::None),
            "basic" => Some(Self::Basic),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    /// The optimizer passes this level enables, in pipeline order. Empty for
    /// [`OptimizationLevel::None`].
    fn passes(self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        use onnx_runtime_optimizer::{ConstantFolding, DeadNodeElimination, OpFusion};
        match self {
            Self::None => Vec::new(),
            Self::Basic => vec![Box::new(ConstantFolding), Box::new(DeadNodeElimination)],
            Self::All => vec![
                Box::new(ConstantFolding),
                Box::new(DeadNodeElimination),
                Box::new(OpFusion::new()),
            ],
        }
    }
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

    /// Set a namespaced option. Unknown keys — and unknown values for a known
    /// key — are rejected at [`Self::build`].
    ///
    /// # Recognized options
    ///
    /// | Key                     | Values                       | Default  | Effect |
    /// |-------------------------|------------------------------|----------|--------|
    /// | `"optimization"`        | `"none"`, `"basic"`, `"all"` | `"none"` | Graph optimization level (see [`OptimizationLevel`]). |
    /// | `"ep.context_enable"`   | `"0"`/`"1"`/`"false"`/`"true"` | `"0"`  | Dump a `*_ctx.onnx` EPContext model after compile (§21.4 / §55.4). |
    /// | `"ep.context_file_path"`| any path                     | `<orig>_ctx.onnx` | Output path for the generated context model. |
    /// | `"ep.context_embed_mode"`| `"0"` (external) / `"1"` (embed) | `"1"` | How the compiled blob is stored in each EPContext node. |
    ///
    /// `"optimization"` = `"none"` (the default) leaves the loaded graph
    /// untouched, so behavior is byte-identical to a runtime with no optimizer.
    /// `"basic"` runs constant folding + dead-node elimination; `"all"` adds
    /// operator fusion. When any pass runs, the session re-runs shape inference
    /// on the rewritten graph before compiling so fused/introduced nodes get
    /// inferred shapes.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Parse every set session option in a single pass, rejecting any unknown
    /// key or unparseable value up front (no silent compat shim — an
    /// unrecognized key is a typo, never a no-op). Returns the resolved
    /// [`OptimizationLevel`] and the EPContext dump config (§21.4 / §55.5)
    /// driven by the `ep.context_*` keys.
    ///
    /// # Recognized keys
    ///
    /// * `"optimization"` → [`OptimizationLevel`] (`none` / `basic` / `all`).
    /// * `"ep.context_enable"` → [`EpContextDumpConfig::enable`]
    ///   (`1`/`0`/`true`/`false`, case-insensitive).
    /// * `"ep.context_file_path"` → [`EpContextDumpConfig::file_path`] (an empty
    ///   value clears it back to the `<orig>_ctx.onnx` default).
    /// * `"ep.context_embed_mode"` → [`EpContextDumpConfig::embed_mode`]
    ///   (`0` external file / `1` embed; any other value is rejected).
    fn parse_options(
        options: &HashMap<String, String>,
    ) -> Result<(OptimizationLevel, EpContextDumpConfig)> {
        let mut level = OptimizationLevel::None;
        let mut ctx = EpContextDumpConfig::default();
        for (key, value) in options {
            match key.as_str() {
                "optimization" => {
                    level = OptimizationLevel::parse(value).ok_or_else(|| {
                        SessionError::InvalidOption {
                            key: key.clone(),
                            value: value.clone(),
                            expected: "none, basic, all".to_string(),
                        }
                    })?;
                }
                "ep.context_enable" => {
                    ctx.enable = parse_bool_option(key, value)?;
                }
                "ep.context_file_path" => {
                    // Empty/unset ⇒ None (fall back to `<orig>_ctx.onnx`).
                    ctx.file_path = if value.trim().is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(value))
                    };
                }
                "ep.context_embed_mode" => {
                    ctx.embed_mode = parse_embed_mode(key, value)?;
                }
                // No compat shim: an unrecognized key is a typo, not a silent
                // no-op.
                _ => return Err(SessionError::UnknownOption { key: key.clone() }),
            }
        }
        Ok((level, ctx))
    }

    /// Build the session: load → detect device → optimize → compile → allocate.
    ///
    /// The `optimize` stage is driven by the `"optimization"` session option and
    /// defaults to [`OptimizationLevel::None`] (a no-op), so the default path is
    /// byte-identical to loading straight into the executor. When optimization
    /// is enabled the pipeline is:
    ///
    /// ```text
    /// load (+ loader shape inference)
    ///   → run optimizer passes (constant-fold / DCE / fusion)
    ///   → re-run shape inference on the rewritten graph
    ///   → compile (kernel per node) → allocate
    /// ```
    ///
    /// The re-inference step is essential: fusion can replace a multi-op
    /// decomposition (e.g. the 9-op LayerNorm) with a single fused node whose
    /// output has no inferred shape yet, and the compile/execute stages require
    /// every value to carry a resolved shape.
    ///
    /// Device selection is CPU-only (`auto_detect` yields the CPU EP), and
    /// "compile" resolves a kernel per node into the shape-keyed cache.
    pub fn build(self) -> Result<InferenceSession> {
        let (level, ep_context_config) = Self::parse_options(&self.options)?;

        // `memory_limit`, `enable_profiling`, and non-CPU `device` preferences
        // are accepted but not yet acted on in Phase 1 (CPU-only executor).
        let _ = (self.device, self.memory_limit, self.enable_profiling);

        let (mut graph, weights, model_dir) = match (self.model_path, self.model_bytes) {
            (Some(path), _) => {
                // The EPContext load path resolves `embed_mode=0` external blob
                // paths relative to the model file's directory (§55.3), so
                // retain it (same base dir the loader used for external data).
                let model_dir = path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));
                let (g, w) = onnx_runtime_loader::load_model_with_weights(path)?;
                (g, w, model_dir)
            }
            (None, Some(bytes)) => {
                let (g, w) = onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".")?;
                (g, w, PathBuf::from("."))
            }
            (None, None) => return Err(SessionError::NoModelSource),
        };

        // Optimize stage. Off by default; only runs when a level is selected.
        optimize_graph(&mut graph, level)?;

        let mut session =
            InferenceSession::from_parts(graph, weights, &model_dir, ep_context_config)?;
        if !self.warmup_shapes.is_empty() {
            session.warmup(&self.warmup_shapes)?;
        }
        Ok(session)
    }
}

/// Parse a boolean-ish session-option value (§21.4). Accepts `1`/`0` and
/// `true`/`false` (case-insensitive), mirroring how ORT's C API treats its
/// `int`-typed `ep.context_enable` flag while also allowing the textual form.
/// Any other value is a typo, surfaced as [`SessionError::InvalidOption`].
fn parse_bool_option(key: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => Err(SessionError::InvalidOption {
            key: key.to_string(),
            value: value.to_string(),
            expected: "0, 1, true, false".to_string(),
        }),
    }
}

/// Parse the `ep.context_embed_mode` option (§21.4): `0` = external sidecar
/// file, `1` = embed the blob inline. Any other value is rejected with
/// [`SessionError::InvalidOption`] (mirroring [`OptimizationLevel::parse`]'s
/// fail-closed rejection rather than silently clamping).
fn parse_embed_mode(key: &str, value: &str) -> Result<u8> {
    match value.trim() {
        "0" => Ok(0),
        "1" => Ok(1),
        _ => Err(SessionError::InvalidOption {
            key: key.to_string(),
            value: value.to_string(),
            expected: "0, 1".to_string(),
        }),
    }
}

/// Run the optimizer passes selected by `level`, then re-run shape inference so
/// any node fusion introduced (whose outputs the loader never saw) gets a fully
/// inferred shape/dtype before compile.
///
/// A no-op when `level` is [`OptimizationLevel::None`] — the graph is returned
/// untouched and no re-inference runs, keeping the default path byte-identical.
fn optimize_graph(graph: &mut onnx_runtime_ir::Graph, level: OptimizationLevel) -> Result<()> {
    let passes = level.passes();
    if passes.is_empty() {
        return Ok(());
    }

    onnx_runtime_optimizer::run_passes(
        graph,
        &passes,
        &onnx_runtime_optimizer::PassContext::new(),
    )?;

    // Fusion emits fused ops in the `com.microsoft` contrib domain; make sure
    // that domain is imported so shape-inference and kernel dispatch pick the
    // contrib-registered rules (they register from opset 1, but recording the
    // import keeps the graph self-consistent and future-proofs versioned rules).
    graph
        .opset_imports
        .entry(onnx_runtime_optimizer::CONTRIB_DOMAIN.to_string())
        .or_insert(1);

    // Re-infer shapes over the rewritten graph: fused nodes' outputs (and any
    // value whose producer changed) must be re-resolved before compile.
    let registry = onnx_runtime_shape_inference::InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    registry.infer_graph(
        graph,
        &opset_imports,
        onnx_runtime_shape_inference::MergePolicy::Permissive,
    )?;

    Ok(())
}

/// A loaded model ready to run inference (§20.2).
pub struct InferenceSession {
    inputs: Vec<IoMeta>,
    outputs: Vec<IoMeta>,
    exec: executor::Executor,
    /// EPContext dump config parsed from the `ep.context_*` session options
    /// (§21.4). Drives [`InferenceSession::export_ep_context`]; disabled by
    /// default so an ordinary session never touches the dump path.
    ep_context_config: EpContextDumpConfig,
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
        // No on-disk model: `embed_mode=0` external EPContext blobs resolve
        // relative to the current directory (consistent with the loader's
        // in-memory `base_dir` default).
        Self::from_parts(
            graph,
            std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            EpContextDumpConfig::default(),
        )
    }

    fn from_parts(
        graph: onnx_runtime_ir::Graph,
        weights: std::sync::Arc<onnx_runtime_loader::WeightStore>,
        model_dir: &Path,
        ep_context_config: EpContextDumpConfig,
    ) -> Result<Self> {
        onnx_runtime_loader::validate_opset_imports(&graph)?;

        let inputs = io_meta(&graph, &graph.inputs);
        let outputs = io_meta(&graph, &graph.outputs);
        let ep = executor::auto_detect_cpu_ep()?;

        // EPContext consume path (§55.3): restore any pre-compiled EP contexts
        // before building the executor. Dispatch is a pure `source`-key lookup
        // over the session's registered EPs (Phase 1: the CPU EP only, which
        // declares no `source` keys — so a model that carries EPContext nodes
        // for an unloaded compiled EP fails with a clear `NoEpForContext`). The
        // executor then bypasses these nodes (they are pre-compiled, never run
        // as ordinary kernels).
        let eps: [(
            onnx_runtime_ep_api::EpId,
            &dyn onnx_runtime_ep_api::ExecutionProvider,
        ); 1] = [(onnx_runtime_ep_api::EpId(0), ep.as_ref())];
        epcontext::load_ep_context_nodes(&graph, model_dir, &eps)?;

        let exec = executor::Executor::build(graph, weights, ep)?;
        Ok(Self {
            inputs,
            outputs,
            exec,
            ep_context_config,
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

    /// The EPContext dump configuration parsed from the `ep.context_*` session
    /// options (§21.4). Disabled by default.
    pub fn ep_context_config(&self) -> &EpContextDumpConfig {
        &self.ep_context_config
    }

    /// The session's (post-optimize) compiled graph.
    ///
    /// This is the graph the executor runs and the same one
    /// [`Self::export_ep_context`] serialises — a caller identifying the
    /// [`NodeId`](onnx_runtime_ir::NodeId)s of a compiled partition (the
    /// [`CompiledPartition::covered_nodes`]) must read them from here so they
    /// reference the exact nodes the exporter will splice out. This is the
    /// compiler-integration seam: a real compiling EP inspects this graph to
    /// choose the subgraphs it claims.
    pub fn graph(&self) -> &onnx_runtime_ir::Graph {
        self.exec.graph()
    }

    /// Export a `com.microsoft::EPContext` context-cache model for this session
    /// (§55.4 dump path), driven by the `ep.context_*` session options
    /// ([`Self::ep_context_config`]).
    ///
    /// `orig_path` is the source model path the default output location
    /// (`<orig>_ctx.onnx`) is derived from when `ep.context_file_path` is unset.
    /// `partitions` are the EP-compiled partitions to serialise — each names the
    /// [`ExecutionProvider`](onnx_runtime_ep_api::ExecutionProvider) that
    /// compiled it, so the driver pulls the blob + SDK version via
    /// [`save_context`](onnx_runtime_ep_api::ExecutionProvider::save_context) and
    /// the `source` key via
    /// [`context_source_keys`](onnx_runtime_ep_api::ExecutionProvider::context_source_keys)
    /// (§55.6 — nothing is hardcoded).
    ///
    /// When `ep.context_enable` is `false` (the default) this is a **no-op**: no
    /// EP `save_context` is called and no files are written; it returns the path
    /// it *would* have written to.
    ///
    /// # Compiler-integration seam
    ///
    /// The Phase-1 CPU EP has **no compile step**, so no real EP yet yields
    /// [`CompiledPartition`]s — `partitions` is therefore supplied by the
    /// caller (proven end-to-end with a mock compiling EP in the crate tests).
    /// TODO(compiler): when a real compiling EP lands, collect its partitions
    /// from the compile/placement stage and call this internally at build time
    /// so a session created with `ep.context_enable=1` dumps automatically.
    pub fn export_ep_context(
        &self,
        orig_path: &Path,
        partitions: &[CompiledPartition],
    ) -> Result<PathBuf> {
        let model = EncoderModel::new(self.exec.graph()).with_weights(self.exec.weights().as_ref());
        dump_session_ep_context(&model, orig_path, partitions, &self.ep_context_config)
    }
}

/// Load a model. Auto-detects the best available hardware (§20.2).
///
/// This is the primary entry point — no configuration required.
pub fn load(path: impl AsRef<Path>) -> Result<InferenceSession> {
    InferenceSession::load(path)
}

#[cfg(test)]
mod option_tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn level_of(pairs: &[(&str, &str)]) -> Result<OptimizationLevel> {
        SessionBuilder::parse_options(&opts(pairs)).map(|(level, _)| level)
    }

    fn ctx_of(pairs: &[(&str, &str)]) -> Result<EpContextDumpConfig> {
        SessionBuilder::parse_options(&opts(pairs)).map(|(_, ctx)| ctx)
    }

    #[test]
    fn optimization_defaults_to_none_when_unset() {
        assert_eq!(level_of(&[]).unwrap(), OptimizationLevel::None);
    }

    #[test]
    fn optimization_parses_known_values() {
        for (v, want) in [
            ("none", OptimizationLevel::None),
            ("off", OptimizationLevel::None),
            ("BASIC", OptimizationLevel::Basic),
            ("All", OptimizationLevel::All),
        ] {
            assert_eq!(level_of(&[("optimization", v)]).unwrap(), want, "value {v:?}");
        }
    }

    #[test]
    fn unknown_option_key_is_rejected() {
        let err = level_of(&[("optimisation", "all")]).unwrap_err();
        assert!(matches!(err, SessionError::UnknownOption { key } if key == "optimisation"));
    }

    #[test]
    fn invalid_optimization_value_is_rejected() {
        let err = level_of(&[("optimization", "aggressive")]).unwrap_err();
        assert!(matches!(
            err,
            SessionError::InvalidOption { key, value, .. } if key == "optimization" && value == "aggressive"
        ));
    }

    #[test]
    fn none_level_selects_no_passes() {
        assert!(OptimizationLevel::None.passes().is_empty());
        assert_eq!(OptimizationLevel::Basic.passes().len(), 2);
        assert_eq!(OptimizationLevel::All.passes().len(), 3);
    }

    // ── EPContext dump options (§21.4 / §55.5) ────────────────────────────────

    #[test]
    fn ep_context_defaults_to_disabled() {
        let ctx = ctx_of(&[]).unwrap();
        assert_eq!(ctx, EpContextDumpConfig::default());
        assert!(!ctx.enable);
        assert_eq!(ctx.file_path, None);
        assert_eq!(ctx.embed_mode, 1);
    }

    #[test]
    fn ep_context_enable_parses_bool_forms() {
        for (v, want) in [
            ("1", true),
            ("0", false),
            ("true", true),
            ("TRUE", true),
            ("false", false),
            ("False", false),
        ] {
            let ctx = ctx_of(&[("ep.context_enable", v)]).unwrap();
            assert_eq!(ctx.enable, want, "value {v:?}");
        }
    }

    #[test]
    fn ep_context_enable_rejects_garbage() {
        let err = ctx_of(&[("ep.context_enable", "yes")]).unwrap_err();
        assert!(matches!(
            err,
            SessionError::InvalidOption { key, value, .. }
                if key == "ep.context_enable" && value == "yes"
        ));
    }

    #[test]
    fn ep_context_file_path_parses_and_empty_clears() {
        let ctx = ctx_of(&[("ep.context_file_path", "/out/net_ctx.onnx")]).unwrap();
        assert_eq!(ctx.file_path, Some(PathBuf::from("/out/net_ctx.onnx")));

        // Empty value falls back to the `<orig>_ctx.onnx` default (None).
        let ctx = ctx_of(&[("ep.context_file_path", "")]).unwrap();
        assert_eq!(ctx.file_path, None);
    }

    #[test]
    fn ep_context_embed_mode_parses_and_rejects() {
        assert_eq!(ctx_of(&[("ep.context_embed_mode", "0")]).unwrap().embed_mode, 0);
        assert_eq!(ctx_of(&[("ep.context_embed_mode", "1")]).unwrap().embed_mode, 1);

        let err = ctx_of(&[("ep.context_embed_mode", "2")]).unwrap_err();
        assert!(matches!(
            err,
            SessionError::InvalidOption { key, value, expected }
                if key == "ep.context_embed_mode" && value == "2" && expected == "0, 1"
        ));
    }

    #[test]
    fn ep_context_options_combine_with_optimization() {
        let (level, ctx) = SessionBuilder::parse_options(&opts(&[
            ("optimization", "all"),
            ("ep.context_enable", "1"),
            ("ep.context_file_path", "/tmp/out_ctx.onnx"),
            ("ep.context_embed_mode", "0"),
        ]))
        .unwrap();
        assert_eq!(level, OptimizationLevel::All);
        assert!(ctx.enable);
        assert_eq!(ctx.file_path, Some(PathBuf::from("/tmp/out_ctx.onnx")));
        assert_eq!(ctx.embed_mode, 0);
    }
}
