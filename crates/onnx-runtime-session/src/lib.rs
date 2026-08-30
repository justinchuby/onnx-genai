//! # `onnx-runtime-session`
//!
//! The user-facing session and inference API for the ORT 2.0 runtime
//! (see `docs/architecture/ORT2.md` §20). Design goal: **zero-config by default** — the user
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

// SessionError intentionally preserves rich, structured diagnostics in the
// public API; boxing it would be an API and behavior change rather than a lint fix.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx_runtime_ir::{DataType, DeviceType, Shape};
use onnx_runtime_tracer::{Args, SpanGuard};

pub use epcontext::{
    CompiledPartition, EpContextPlacement, dump_session_ep_context, load_ep_context_nodes,
};
pub use error::SessionError;
pub use executor::{
    ActivationMemoryPlanStats, CacheStats, CaptureDecline, CaptureDeclineReport, CapturePathKind,
    ControlFlowStats, DENSE_WEIGHT_PREFETCH_LOOKAHEAD_ENV, DensePrefetchGapStats,
    DeviceAllocationCounts, DeviceGraphCaptureResult, ExecutionProviderDecline,
    ExecutionProviderFallbackReport, PrefetchStep, SeamReason, dense_prefetch_gap_stats,
    dense_weight_prefetch_lookahead_nodes, drive_double_buffer,
    enable_activation_memory_plan_for_process, enable_exec_phase_profile_for_process,
    exec_phase_stats, plan_double_buffer, print_exec_phase_profile, reset_dense_prefetch_gap_stats,
    reset_exec_phase_profile,
};
pub use onnx_runtime_ep_api::DeviceBuffer;
pub use onnx_runtime_ep_api::DeviceGraphSlot;
pub use onnx_runtime_ep_api::WorkspaceRequirement;
pub use onnx_runtime_loader::{
    EpContextDumpConfig, EpContextPartition, Model as EncoderModel, ModelMetadata,
};
pub use plugin_provider::{PluginExecutionProvider, is_plugin_fused_node};
pub use tensor::{
    DeviceBindingTransferStats, DeviceIoBinding, ExternalMemorySpec, Tensor, cpu_allocator,
};
/// Device-bound graph outputs, stored inline for the common fixed-output case.
pub type DeviceBindingOutputs = smallvec::SmallVec<[Option<Tensor>; 16]>;

mod epcontext;
mod executor;
mod fp16_decode;
pub mod hetero;
mod plugin_provider;
pub mod sequence;
mod tensor;

fn trace_span(name: &'static str, cat: &'static str) -> Option<SpanGuard> {
    onnx_runtime_tracer::global_context()
        .filter(|trace| trace.is_enabled())
        .map(|trace| trace.span(name, cat))
}

/// A graph output produced by the runtime.
#[derive(Debug)]
pub enum SessionOutput {
    Tensor(Tensor),
    Sequence(sequence::SequenceValue),
}

impl SessionOutput {
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::Sequence(_) => None,
        }
    }

    pub fn as_sequence(&self) -> Option<&sequence::SequenceValue> {
        match self {
            Self::Tensor(_) => None,
            Self::Sequence(sequence) => Some(sequence),
        }
    }

    pub fn into_tensor(self) -> Option<Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::Sequence(_) => None,
        }
    }

    pub fn into_sequence(self) -> Option<sequence::SequenceValue> {
        match self {
            Self::Tensor(_) => None,
            Self::Sequence(sequence) => Some(sequence),
        }
    }
}

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

        #[error("execution provider unavailable: {0}")]
        ExecutionProviderUnavailable(String),

        #[error(
            "CUDA execution required by ONNX_GENAI_REQUIRE_CUDA=1, but CPU fallback is needed: \
             {unsupported_nodes}"
        )]
        HeterogeneousPlacementRequired { unsupported_nodes: String },

        #[error(
            "the opt-in heterogeneous executor cannot run this graph or API safely: \
             {placement_summary}. The first #603 execution slice supports fully-static, \
             tensor-only acyclic graphs through ordinary run/run_outputs; unset \
             ONNX_GENAI_HETERO for legacy whole-session CPU fallback"
        )]
        HeterogeneousExecutionUnsupported { placement_summary: String },

        #[error(
            "unsupported operator {domain}::{op_type}: no available execution provider has a \
             kernel; node {node}, opset {opset}; decline reason: {reason}; consulted execution \
             providers (priority order): {execution_providers}. To fix: {remediation}",
            remediation = unsupported_op_remediation(*.opset, .domain)
        )]
        UnsupportedOp {
            op_type: String,
            domain: String,
            node: String,
            opset: OpsetVersion,
            reason: String,
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

        /// A caller-supplied buffer cannot back the device binding it was
        /// offered for.
        #[error("external buffer for device binding '{binding}' cannot be used: {reason}")]
        ExternalBuffer { binding: String, reason: String },

        #[error(
            "op {op} produced {got} data-dependent output shape(s) but has {expected} output(s)"
        )]
        OutputShapeCountMismatch {
            op: String,
            expected: usize,
            got: usize,
        },

        #[error(
            "runtime broadcast shape resolution failed for node {node} ({domain}::{op_type}): \
             concrete input shapes {input_shapes:?} are not broadcast-compatible, so no valid \
             elementwise output shape exists. To fix: update the model or runtime inputs so each \
             aligned dimension is equal or one of them is 1"
        )]
        RuntimeBroadcastIncompatible {
            node: String,
            domain: String,
            op_type: String,
            input_shapes: Vec<Vec<usize>>,
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

        #[error("Sequence op {op}: {reason}")]
        SequenceOp { op: String, reason: String },

        #[error("control-flow op {op}: {reason}")]
        ControlFlow { op: String, reason: String },

        #[error(
            "EPContext reference node (main_context=0) has no matching primary \
             (source={source_key:?}, partition_name={partition_name:?})"
        )]
        DanglingEpContext {
            source_key: Option<String>,
            partition_name: Option<String>,
        },

        #[error(
            "EPContext export failed: operator {domain}::{op_type} at node {node} has \
             in-memory IR version {node_version}, but the graph-level opset_import for \
             domain '{domain}' is {graph_version}. Why: Node.version supports mixed-opset \
             graph rewrites in memory, but ONNX protobuf has no per-node version field, so \
             writing this graph would silently claim the wrong operator version. How to fix: \
             export before EP fusion, or disable the fusion for an export run"
        )]
        EpContextMixedNodeVersion {
            op_type: String,
            node: String,
            domain: String,
            node_version: u64,
            graph_version: String,
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
            reason: impl Into<String>,
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
                reason: reason.into(),
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

/// Decoder-wide numeric precision for the session's decode graph.
///
/// This is a generic, model-agnostic knob selected via
/// [`SessionBuilder::decode_precision`]. The default,
/// [`DecodePrecision::Model`], runs the graph exactly as authored, so default
/// runtime behaviour is byte-identical to a build with no precision knob at all.
///
/// [`DecodePrecision::Fp16`] requests a whole-decoder fp32→fp16 rewrite (see
/// [`fp16_decode`]). It only takes effect on a GPU device and only for an
/// fp32-activation int4/block-32 quantized decoder (fp32-scale `MatMulNBits`);
/// for every other model — including native fp16-activation models — it is a
/// strict no-op, leaving the graph bit-identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodePrecision {
    /// Run the decoder at the precision authored in the model graph (default).
    #[default]
    Model,
    /// Cast an fp32-activation int4/block-32 decoder to a fully fp16 graph so a
    /// GPU backend runs it through the fp16-fused decode kernels. No-op unless
    /// the session targets a GPU and the graph is fp32-activation quantized.
    Fp16,
}

/// Graph-optimization level for the session's `optimize` pipeline stage
/// (`docs/architecture/ORT2.md` §18). Selected via the generic `"optimization"` session
/// option (see [`SessionBuilder::option`]).
///
/// The default is [`OptimizationLevel::None`]: with optimization off the graph
/// reaches the executor exactly as the loader produced it, so default runtime
/// behavior is byte-identical to a build with no optimizer wired in at all.
///
/// This is a generic, model-agnostic knob — no level ever special-cases a model
/// name or op. Higher levels simply enable more of the device-independent pass
/// pipeline from [`onnx_runtime_optimizer`]. Operator fusion is deliberately
/// excluded here because provider-specific fused operators must be introduced
/// only by the execution provider that can run them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No passes — the `optimize` stage is a no-op (default).
    #[default]
    None,
    /// Structure-preserving passes only: constant folding then dead-node
    /// elimination. No operator fusion, so the op set the executor sees is a
    /// subset of the loaded graph's.
    Basic,
    /// The full device-independent pipeline. This currently matches
    /// [`OptimizationLevel::Basic`]; provider-scoped fusion runs later through
    /// [`onnx_runtime_ep_api::ExecutionProvider::custom_passes`].
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
        use onnx_runtime_optimizer::{ConstantFolding, DeadNodeElimination};
        match self {
            Self::None => Vec::new(),
            Self::Basic => vec![Box::new(ConstantFolding), Box::new(DeadNodeElimination)],
            Self::All => vec![Box::new(ConstantFolding), Box::new(DeadNodeElimination)],
        }
    }
}

/// Builder for advanced session configuration (§20.6).
#[derive(Default)]
pub struct SessionBuilder {
    model_path: Option<PathBuf>,
    model_bytes: Option<Vec<u8>>,
    device: DevicePreference,
    execution_provider: Option<std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>>,
    memory_limit: Option<usize>,
    enable_profiling: bool,
    warmup_shapes: Vec<WarmupShape>,
    decode_precision: DecodePrecision,
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

    /// Use an explicitly constructed execution provider instead of device auto-selection.
    pub fn execution_provider(
        mut self,
        execution_provider: std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>,
    ) -> Self {
        self.execution_provider = Some(execution_provider);
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

    /// Select the decoder-wide numeric precision (see [`DecodePrecision`]).
    /// Defaults to [`DecodePrecision::Model`] (graph as authored); selecting
    /// [`DecodePrecision::Fp16`] opts into the fp32→fp16 decode rewrite, which
    /// only takes effect on a GPU fp32-activation quantized decoder.
    pub fn decode_precision(mut self, precision: DecodePrecision) -> Self {
        self.decode_precision = precision;
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
    /// Device selection keeps CPU as the default and selects CUDA only when
    /// explicitly requested in a CUDA-enabled build. "Compile" resolves a
    /// kernel per node into the shape-keyed cache.
    pub fn build(self) -> Result<InferenceSession> {
        let (level, ep_context_config) = Self::parse_options(&self.options)?;

        // Memory limits and profiling remain reserved builder intents.
        let _ = (self.memory_limit, self.enable_profiling);

        // EP selection is graph-independent (it only inspects the requested
        // device / explicit override), so resolve it *before* loading. The
        // loader then keeps a function-call node as an op — instead of inlining
        // it into its body — whenever this EP's claim gate reports a fused kernel
        // for it (the general "keep-as-op iff a kernel claims it, else inline"
        // policy). With no fused kernel the predicate declines and inlining is
        // byte-identical to the default path.
        let ep = {
            let mut span = trace_span("session.select_execution_provider", "session");
            let ep = match self.execution_provider {
                Some(ep) => ep,
                None => select_execution_provider(&self.device)?,
            };
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("provider", ep.name().to_string())
                        .with("device", ep.device_type().trace_name().into_owned())
                        .with("device_index", ep.device_id().index as u64),
                );
            }
            ep
        };

        let (mut graph, weights, model_dir, model_metadata) = {
            let keep_as_op =
                |node: &onnx_runtime_ir::Node, opset: u64, dtypes: &[onnx_runtime_ir::DataType]| {
                    ep.supports_op(node, opset, &[], dtypes, &[]).is_supported()
                };
            match (self.model_path, self.model_bytes) {
                (Some(path), _) => {
                    // The EPContext load path resolves `embed_mode=0` external blob
                    // paths relative to the model file's directory (§55.3), so
                    // retain it (same base dir the loader used for external data).
                    let model_dir = path
                        .parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| PathBuf::from("."));
                    let bytes = onnx_runtime_loader::read_model_binary(&path)?;
                    let metadata = {
                        let mut span = trace_span("load.model_metadata", "load");
                        let metadata = model_metadata_from_bytes(&bytes)?;
                        if let Some(span) = span.as_mut() {
                            span.set_args(
                                Args::new()
                                    .bytes(bytes.len() as u64)
                                    .with("metadata_props", metadata.metadata_props.len() as u64),
                            );
                        }
                        metadata
                    };
                    let (g, w) = onnx_runtime_loader::load_model_bytes_with_weights_filtered(
                        &bytes,
                        &model_dir,
                        &keep_as_op,
                    )?;
                    (g, w, model_dir, metadata)
                }
                (None, Some(bytes)) => {
                    let metadata = {
                        let mut span = trace_span("load.model_metadata", "load");
                        let metadata = model_metadata_from_bytes(&bytes)?;
                        if let Some(span) = span.as_mut() {
                            span.set_args(
                                Args::new()
                                    .bytes(bytes.len() as u64)
                                    .with("metadata_props", metadata.metadata_props.len() as u64),
                            );
                        }
                        metadata
                    };
                    let (g, w) = onnx_runtime_loader::load_model_bytes_with_weights_filtered(
                        &bytes,
                        ".",
                        &keep_as_op,
                    )?;
                    (g, w, PathBuf::from("."), metadata)
                }
                (None, None) => return Err(SessionError::NoModelSource),
            }
        };

        // Decoder precision rewrite (opt-in). Applied here — on the freshly
        // loaded graph, before the I/O signature (`IoMeta`) is computed and
        // before EP optimization — so the KV/logits buffers and the executor
        // graph agree on the rewritten dtype. A strict no-op for the default
        // `DecodePrecision::Model`, for non-GPU devices, and for any graph that
        // is not an fp32-activation quantized decoder, so the default path and
        // native fp16 models stay bit-identical.
        {
            let nodes_before = graph.num_nodes();
            let mut span = trace_span("session.fp16_decode", "session");
            fp16_decode::maybe_convert_decode_fp16(
                &mut graph,
                &weights,
                self.decode_precision,
                device_preference_is_gpu(&self.device),
            );
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("nodes_before", nodes_before as u64)
                        .with("nodes_after", graph.num_nodes() as u64)
                        .with("device_gpu", device_preference_is_gpu(&self.device)),
                );
            }
        }

        // Optimize stage. Off by default; only runs when a level is selected.
        optimize_graph(&mut graph, level)?;

        let mut session = InferenceSession::from_parts(
            graph,
            weights,
            &model_dir,
            ep_context_config,
            model_metadata,
            ep,
        )?;
        if !self.warmup_shapes.is_empty() {
            let mut span = trace_span("session.warmup", "session");
            session.warmup(&self.warmup_shapes)?;
            if let Some(span) = span.as_mut() {
                span.set_args(Args::new().with("shape_count", self.warmup_shapes.len() as u64));
            }
        }
        Ok(session)
    }
}

/// Whether a [`DevicePreference`] targets a GPU / accelerator (non-host) device.
/// Used to gate GPU-only graph rewrites such as the fp16 decode precision mode.
fn device_preference_is_gpu(preference: &DevicePreference) -> bool {
    match preference {
        DevicePreference::Gpu { .. } => true,
        DevicePreference::Explicit { device_type, .. } => !device_type.is_host_accessible(),
        DevicePreference::Auto | DevicePreference::Cpu => false,
    }
}

fn select_execution_provider(
    preference: &DevicePreference,
) -> Result<std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>> {
    match preference {
        // Keep the zero-config/default behavior CPU-only. CUDA is an explicit
        // opt-in until heterogeneous placement and fallback exist.
        DevicePreference::Auto | DevicePreference::Cpu => executor::auto_detect_cpu_ep(),
        DevicePreference::Explicit {
            device_type: DeviceType::Cpu,
            index: 0,
        } => executor::auto_detect_cpu_ep(),
        DevicePreference::Gpu { index } => cuda_execution_provider(index.unwrap_or(0)),
        DevicePreference::Explicit {
            device_type: DeviceType::Cuda,
            index,
        } => cuda_execution_provider(*index),
        DevicePreference::Explicit { device_type, index } => {
            Err(SessionError::ExecutionProviderUnavailable(format!(
                "{device_type:?}:{index} is not implemented by onnx-runtime-session"
            )))
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_execution_provider(
    index: u32,
) -> Result<std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>> {
    let mut ep = onnx_runtime_ep_cuda::CudaExecutionProvider::new(index)?;
    onnx_runtime_ep_api::ExecutionProvider::initialize(&mut ep, &Default::default())?;
    Ok(std::sync::Arc::new(ep))
}

#[cfg(not(feature = "cuda"))]
fn cuda_execution_provider(
    index: u32,
) -> Result<std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>> {
    Err(SessionError::ExecutionProviderUnavailable(format!(
        "CUDA:{index} requested, but onnx-runtime-session was built without the `cuda` feature"
    )))
}

fn model_metadata_from_bytes(bytes: &[u8]) -> Result<ModelMetadata> {
    let model = onnx_runtime_loader::proto::decode_model(bytes)?;
    Ok(ModelMetadata {
        ir_version: model.ir_version,
        producer_name: model.producer_name,
        producer_version: model.producer_version,
        domain: model.domain,
        model_version: model.model_version,
        doc_string: (!model.doc_string.is_empty()).then_some(model.doc_string),
        graph_name: model.graph.map(|graph| graph.name).unwrap_or_default(),
        metadata_props: model
            .metadata_props
            .into_iter()
            .map(|entry| (entry.key, entry.value))
            .collect(),
    })
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
/// any rewritten values get a fully inferred shape/dtype before compile.
///
/// A no-op when `level` is [`OptimizationLevel::None`] — the graph is returned
/// untouched and no re-inference runs, keeping the default path byte-identical.
fn optimize_graph(graph: &mut onnx_runtime_ir::Graph, level: OptimizationLevel) -> Result<()> {
    let passes = level.passes();
    if passes.is_empty() {
        return Ok(());
    }

    {
        let nodes_before = graph.num_nodes();
        let mut span = trace_span("session.optimize_graph", "session");
        onnx_runtime_optimizer::run_passes(
            graph,
            &passes,
            &onnx_runtime_optimizer::PassContext::new(),
        )?;
        if let Some(span) = span.as_mut() {
            span.set_args(
                Args::new()
                    .with("passes", passes.len() as u64)
                    .with("nodes_before", nodes_before as u64)
                    .with("nodes_after_passes", graph.num_nodes() as u64),
            );
        }
    }

    // Re-infer shapes over the rewritten graph: fused nodes' outputs (and any
    // value whose producer changed) must be re-resolved before compile.
    let registry = onnx_runtime_shape_inference::InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    {
        let mut span = trace_span("session.optimize_shape_inference", "session");
        registry.infer_graph(
            graph,
            &opset_imports,
            onnx_runtime_shape_inference::MergePolicy::Permissive,
        )?;
        if let Some(span) = span.as_mut() {
            span.set_args(
                Args::new()
                    .with("nodes", graph.num_nodes() as u64)
                    .with("opset_domains", opset_imports.len() as u64),
            );
        }
    }

    Ok(())
}

/// Coarse traffic phase selected by a BlockQuantizedMoE observer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockQuantizedMoeTrafficPhase {
    Load,
    Warmup,
    Prefill,
    Decode,
}

/// Request identity used to arm session-owned BlockQuantizedMoE traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQuantizedMoeTrafficConfig {
    pub request_id: u32,
}

/// One explicit phase snapshot from the production device record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQuantizedMoeTrafficSnapshot {
    pub phase: BlockQuantizedMoeTrafficPhase,
    pub request_id: u32,
    pub traffic: onnx_runtime_ep_api::BlockQuantizedMoeTraffic,
}

/// Exclusive session/request owner for FreeToken-style BlockQuantizedMoE
/// observability. The borrow prevents arm/disarm from racing execution, while
/// captured graphs retain the exact immutable device record they reference.
///
/// ```compile_fail
/// # use onnx_runtime_session::{
/// #     BlockQuantizedMoeTrafficConfig, InferenceSession,
/// # };
/// # fn concurrent_reconfiguration(session: &mut InferenceSession) {
/// let observer = session
///     .observe_block_quantized_moe_traffic(BlockQuantizedMoeTrafficConfig {
///         request_id: 1,
///     })
///     .unwrap();
/// let _ = session.run_with_device_bindings(&[], &mut []);
/// drop(observer);
/// # }
/// ```
pub struct BlockQuantizedMoeTrafficObserver<'a> {
    session: &'a mut InferenceSession,
    phase: BlockQuantizedMoeTrafficPhase,
    request_id: u32,
    active: bool,
}

impl BlockQuantizedMoeTrafficObserver<'_> {
    pub fn reset_phase(&mut self, phase: BlockQuantizedMoeTrafficPhase) -> Result<()> {
        self.session.exec.finish_device_validation_boundary()?;
        self.session.exec.reset_block_quantized_moe_traffic()?;
        self.phase = phase;
        Ok(())
    }

    pub fn snapshot(&mut self) -> Result<BlockQuantizedMoeTrafficSnapshot> {
        self.session.exec.finish_device_validation_boundary()?;
        Ok(BlockQuantizedMoeTrafficSnapshot {
            phase: self.phase,
            request_id: self.request_id,
            traffic: self.session.exec.snapshot_block_quantized_moe_traffic()?,
        })
    }

    #[cfg(feature = "gpu-tests")]
    pub fn inject_fault_for_test(
        &self,
        fault: onnx_runtime_ep_cuda::kernels::block_quantized_moe::BlockQuantizedMoeTrafficFaultForTest,
    ) -> Result<()> {
        self.session
            .exec
            .inject_block_quantized_moe_traffic_fault_for_test(fault)
    }

    pub fn prepare_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<onnx_runtime_ep_api::WorkspaceRequirement> {
        self.session
            .exec
            .prepare_with_device_bindings(inputs, bindings)
    }

    pub fn warmup(&mut self, shapes: &[WarmupShape]) -> Result<()> {
        self.session.warmup(shapes)
    }

    pub fn run_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceBindingOutputs> {
        self.session.exec.run_with_device_bindings(inputs, bindings)
    }

    pub fn try_capture_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        self.session
            .exec
            .try_capture_with_device_bindings(inputs, bindings)
    }

    pub fn replay_device_graph(&mut self, bindings: &mut [DeviceIoBinding]) -> Result<bool> {
        self.session.exec.replay_device_graph(bindings)
    }

    pub fn reset_device_graph(&mut self) -> Result<bool> {
        self.session.exec.reset_device_graph()
    }

    pub fn finish(mut self) -> Result<()> {
        self.cleanup()
    }

    fn cleanup(&mut self) -> Result<()> {
        let validation = self.session.exec.finish_device_validation_boundary();
        let disarm = self.session.exec.disarm_block_quantized_moe_traffic();
        if disarm.is_ok() {
            self.active = false;
        }
        match (validation, disarm) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(validation), Err(disarm)) => Err(SessionError::Internal(format!(
                "BlockQuantizedMoE observer validation failed: {validation}; cleanup also failed: \
                 {disarm}"
            ))),
        }
    }
}

impl Drop for BlockQuantizedMoeTrafficObserver<'_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.cleanup()
        {
            eprintln!(
                "[onnx-runtime-session] BlockQuantizedMoE observer drop observed a cleanup \
                 failure after attempting validation and graph-safe disarm: {error}"
            );
        }
    }
}

/// A loaded model ready to run inference (§20.2).
pub struct InferenceSession {
    inputs: Vec<IoMeta>,
    outputs: Vec<IoMeta>,
    model_metadata: ModelMetadata,
    exec: executor::Executor,
    /// Decode-specialized sibling executor whose single-trip recurrent `Scan`
    /// bodies are inlined into the parent graph (Inc-1b PR-2). Built lazily and
    /// only when [`InferenceSession::enable_decode_inline`] is called on a model
    /// that is on the hybrid single-trip Scan path; `None` otherwise (the
    /// default), so an ordinary session is byte-identical and pays nothing. It
    /// shares `exec`'s `Arc<WeightStore>` and `Arc<dyn ExecutionProvider>`, so a
    /// decode step routed to it binds the identical persistent state buffers.
    decode_inline_exec: Option<executor::Executor>,
    /// Verify-dedicated sibling executor for native MTP self-speculative decode
    /// (built lazily by [`InferenceSession::enable_verify_sibling`]). Runs the
    /// fixed M=k+1 verify forward with its OWN interior device-buffer arena so
    /// the M=2 verify's JIT-sized interior scratch is never resized by the
    /// interleaved M=1 base decode on `exec` (the shared-arena clobber that made
    /// the verify capture decline). Shares `exec`'s `Arc<WeightStore>` and
    /// `Arc<dyn ExecutionProvider>`, so a verify step routed here binds the
    /// identical persistent external KV/recurrent-state buffers; it drives the
    /// EP's [`DeviceGraphSlot::Verify`] slot, independent of `exec`'s `Primary`
    /// M=1 decode graph. `None` (the default) for every non-MTP session, so an
    /// ordinary session is byte-identical and pays nothing.
    verify_exec: Option<executor::Executor>,
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
            ModelMetadata::default(),
            executor::auto_detect_cpu_ep()?,
        )
    }

    /// Build a session from an in-memory IR [`Graph`](onnx_runtime_ir::Graph)
    /// with a caller-supplied execution provider and weight store.
    ///
    /// Like [`Self::from_graph`] but lets the caller choose the execution
    /// provider (e.g. a native CUDA EP) and supply the [`WeightStore`] whose
    /// mmap backs the graph's initializer [`WeightRef`]s. This is how the MTP
    /// draft LM-head projection reuses the target model's already-loaded int4
    /// `MatMulNBits` initializers zero-copy on the native CUDA EP, without
    /// re-exporting a sidecar or dequantising the weight host-side.
    pub fn from_graph_with_provider(
        graph: onnx_runtime_ir::Graph,
        weights: std::sync::Arc<onnx_runtime_loader::WeightStore>,
        model_dir: &Path,
        provider: std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>,
    ) -> Result<Self> {
        Self::from_parts(
            graph,
            weights,
            model_dir,
            EpContextDumpConfig::default(),
            ModelMetadata::default(),
            provider,
        )
    }

    fn from_parts(
        graph: onnx_runtime_ir::Graph,
        weights: std::sync::Arc<onnx_runtime_loader::WeightStore>,
        model_dir: &Path,
        ep_context_config: EpContextDumpConfig,
        model_metadata: ModelMetadata,
        ep: std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>,
    ) -> Result<Self> {
        // Establish the canonical-domain invariant for programmatically built
        // graphs (the loader already normalizes at proto-materialization time):
        // the default ONNX domain is `""`, never `"ai.onnx"`. The executor and
        // validators rely on this, comparing `domain.is_empty()` directly.
        let mut graph = graph;
        {
            let mut span = trace_span("session.normalize_validate", "session");
            graph.normalize_domains();

            onnx_runtime_loader::validate_model(&graph)?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("nodes", graph.num_nodes() as u64)
                        .with("values", graph.values.len() as u64),
                );
            }
        }

        let (inputs, outputs) = {
            let mut span = trace_span("session.io_meta", "session");
            let inputs = io_meta(&graph, &graph.inputs);
            let outputs = io_meta(&graph, &graph.outputs);
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("inputs", inputs.len() as u64)
                        .with("outputs", outputs.len() as u64),
                );
            }
            (inputs, outputs)
        };
        // EPContext consume path (§55.3): restore any pre-compiled EP contexts
        // before building the executor. Dispatch is a pure `source`-key lookup
        // over the session's selected EP, so a model carrying EPContext nodes
        // for an unloaded compiled EP fails with a clear `NoEpForContext`. The
        // executor then bypasses these nodes (they are pre-compiled, never run
        // as ordinary kernels).
        let eps: [(
            onnx_runtime_ep_api::EpId,
            &dyn onnx_runtime_ep_api::ExecutionProvider,
        ); 1] = [(onnx_runtime_ep_api::EpId(0), ep.as_ref())];
        {
            let mut span = trace_span("session.epcontext_restore", "session");
            epcontext::load_ep_context_nodes(&graph, model_dir, &eps)?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("provider", ep.name().to_string())
                        .with("model_dir", model_dir.display().to_string()),
                );
            }
        }

        let exec = {
            let mut span = trace_span("session.executor_build", "session");
            let exec = executor::Executor::build(graph, weights, ep)?;
            if let Some(span) = span.as_mut() {
                span.set_args(Args::new().with("cache_entries", exec.cache_stats().entries as u64));
            }
            exec
        };
        Ok(Self {
            inputs,
            outputs,
            model_metadata,
            exec,
            decode_inline_exec: None,
            verify_exec: None,
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

    /// Attach the shared runtime trace context. When enabled, the executor opens
    /// one span per executed op so kernels can attach kernel-variant and
    /// capture-rejection reasons to a live span. Defaults to a disabled no-op
    /// context, so untraced runs pay only a single relaxed atomic load per op.
    pub fn set_trace_context(&mut self, trace: onnx_runtime_tracer::TraceContext) {
        self.exec.set_trace_context(trace);
    }

    /// Run inference and preserve tensor or sequence graph-output types.
    pub fn run_outputs(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<SessionOutput>> {
        self.exec.run_outputs(inputs)
    }

    /// F5 Stage 1 decode-plan memo activity counters `(primed, rebuilt, replayed,
    /// ineligible)` over this session's lifetime. `replayed > 0` after a decode
    /// run proves the memo actually engaged on the real (persistent-KV-binding)
    /// path; the coordinator's on-model A/B reads this to reject a vacuous pass.
    pub fn decode_memo_counts(&self) -> (u64, u64, u64, u64) {
        self.exec.decode_memo_counts()
    }

    /// F5 Stage 2 view-plan activity counters `(views_reused, dispatch_elided)`
    /// over this session's lifetime. Both `> 0` after a decode run prove the
    /// invariant zero-copy view reuse and pure-view dispatch elision actually
    /// fired on the real path (not a vacuous pass); an on-model A/B reads this
    /// alongside [`Self::decode_memo_counts`].
    pub fn decode_view_plan_counts(&self) -> (u64, u64) {
        self.exec.decode_view_plan_counts()
    }

    /// Most recent activation-memory planner measurement.
    ///
    /// Populated only by measured top-level eager runs, after concrete shapes
    /// and zero-copy view aliases are known. Stage-2 replay and nested runs skip
    /// re-planning and leave the last measured result. The planner's
    /// `naive_bytes` is an upper-bound activation-owner baseline, not the
    /// executor's exact current allocation behavior (in-place aliases and
    /// sequences can allocate less).
    pub fn activation_memory_plan_stats(&self) -> Option<ActivationMemoryPlanStats> {
        self.exec.activation_memory_plan_stats()
    }

    /// How many times the single-trip `Scan` inline dual-path
    /// (`ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP`) engaged over this session's
    /// lifetime. `> 0` after a decode run proves the runtime `trip_count == 1`
    /// inline path actually fired (not a silently gated-out pass); an on-model
    /// flag-on/flag-off A/B reads this alongside the token stream to prove the
    /// dual-path is both engaged and byte-exact.
    pub fn scan_inline_single_trip_count(&self) -> u64 {
        self.exec.scan_inline_single_trip_count()
    }

    /// Run with persistent device allocations supplying graph inputs and,
    /// optionally, aliasing graph outputs. Bound outputs are returned as `None`
    /// because their bytes remain resident in the caller-owned allocation.
    pub fn run_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceBindingOutputs> {
        self.exec.run_with_device_bindings(inputs, bindings)
    }

    /// Arm production BlockQuantizedMoE traffic before capture and hold
    /// exclusive session ownership through all observed phases.
    pub fn observe_block_quantized_moe_traffic(
        &mut self,
        config: BlockQuantizedMoeTrafficConfig,
    ) -> Result<BlockQuantizedMoeTrafficObserver<'_>> {
        let armed = self
            .exec
            .arm_block_quantized_moe_traffic(config.request_id)?;
        if armed == 0 {
            return Err(SessionError::Internal(
                "session has no prepared BlockQuantizedMoE kernel to observe".into(),
            ));
        }
        Ok(BlockQuantizedMoeTrafficObserver {
            session: self,
            phase: BlockQuantizedMoeTrafficPhase::Load,
            request_id: config.request_id,
            active: true,
        })
    }

    /// Prepare exact kernel workspace for a bound run without launching kernels.
    pub fn prepare_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<onnx_runtime_ep_api::WorkspaceRequirement> {
        self.exec.prepare_with_device_bindings(inputs, bindings)
    }

    pub fn prepare_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MappedGrowthGrant>> {
        self.exec.prepare_mapped_growth(bytes, role)
    }

    pub fn release_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) {
        self.exec.release_mapped_growth(bytes, role);
    }

    /// Locations of graph nodes that declare owned kernel workspace.
    pub fn workspace_node_locations(&self) -> Vec<String> {
        self.exec.workspace_node_locations()
    }

    /// Lazily build the decode-specialized inlined-body sibling executor
    /// (Inc-1b PR-2, `cohaagen-27b-inc1b-design.md` §1). Idempotent: a second
    /// call is a no-op. Returns `true` when a decode-inline plan is now
    /// available (the model is on the hybrid single-trip recurrent `Scan`
    /// path), or `false` when the model has no such `Scan` and the caller must
    /// keep today's Scan child-session path. The main/prefill executor is left
    /// byte-identical either way.
    ///
    /// The caller probes this automatically at the first decode step; the
    /// session itself reads no env and takes no user knob — the graph property
    /// (an inlineable single-trip recurrent `Scan`) is the only gate.
    pub fn enable_decode_inline(&mut self) -> Result<bool> {
        if self.decode_inline_exec.is_none() {
            self.decode_inline_exec = self.exec.build_decode_inline_sibling()?;
        }
        Ok(self.decode_inline_exec.is_some())
    }

    /// Whether a decode-inline sibling executor is built and ready to run.
    pub fn decode_inline_ready(&self) -> bool {
        self.decode_inline_exec.is_some()
    }

    /// Run one decode step through the decode-inline sibling executor (eager),
    /// binding the identical persistent device buffers `bindings` supplies so
    /// recurrent-state continuity across the prefill→decode boundary is
    /// preserved (design §3). Only valid for a **single-trip** (scan-axis
    /// extent 1) decode step: the caller must route any extent≠1 step to
    /// [`Self::run_with_device_bindings`] (the main exec) instead — the
    /// decode-inline graph is specialized to one iteration and would otherwise
    /// run a wrongly-collapsed graph.
    ///
    /// Errors if [`Self::enable_decode_inline`] has not built a sibling.
    pub fn run_decode_inline_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceBindingOutputs> {
        let exec = self.decode_inline_exec.as_mut().ok_or_else(|| {
            SessionError::Internal(
                "decode-inline executor requested but not built; call enable_decode_inline first"
                    .into(),
            )
        })?;
        exec.run_with_device_bindings(inputs, bindings)
    }

    /// Inc-1b PR-3: capture-record one decode step of the **decode-inline
    /// sibling** into a device graph (`cohaagen-inc1b-pr3-scope.md`, bucket-A).
    /// Mirrors [`Self::try_capture_with_device_bindings`] but drives the sibling
    /// executor instead of the main one, so the inlined-body ops fold into the
    /// CUDA graph exactly like any other kernel — the same segmenter/warm-seeded
    /// machinery, no #443/#543 shared-capture-code change. The sibling shares the
    /// main executor's `Arc<dyn ExecutionProvider>` (one EP graph slot + one
    /// capture-error latch), so callers MUST keep the main decode capture machine
    /// dormant while routing single-token decode here (no double-capture).
    ///
    /// Errors if [`Self::enable_decode_inline`] has not built a sibling.
    pub fn try_capture_decode_inline_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        let exec = self.decode_inline_exec.as_mut().ok_or_else(|| {
            SessionError::Internal(
                "decode-inline capture requested but no sibling built; call enable_decode_inline first"
                    .into(),
            )
        })?;
        exec.try_capture_with_device_bindings(inputs, bindings)
    }

    /// Replay the decode-inline sibling's installed device graph. Mirrors
    /// [`Self::replay_device_graph`] on the sibling executor. Returns `false`
    /// when a control-flow seam retired the graph this step (token produced
    /// eagerly; caller re-warms/re-captures).
    ///
    /// Errors if [`Self::enable_decode_inline`] has not built a sibling.
    pub fn replay_decode_inline_device_graph(
        &mut self,
        bindings: &mut [DeviceIoBinding],
    ) -> Result<bool> {
        let exec = self.decode_inline_exec.as_mut().ok_or_else(|| {
            SessionError::Internal(
                "decode-inline replay requested but no sibling built; call enable_decode_inline first"
                    .into(),
            )
        })?;
        exec.replay_device_graph(bindings)
    }

    /// Invalidate the decode-inline sibling's installed device graph (before
    /// reset, KV-bucket growth, shape change, or binding destruction). A no-op
    /// returning `false` when no sibling has been built. Because the sibling
    /// shares the main executor's EP, the underlying EP graph + capture-error
    /// latch are also cleared by [`Self::reset_device_graph`]; this additionally
    /// clears the sibling executor's host-side capture schedule so it re-warms
    /// rather than replaying a stale graph.
    pub fn reset_decode_inline_device_graph(&mut self) -> Result<bool> {
        match self.decode_inline_exec.as_mut() {
            Some(exec) => exec.reset_device_graph(),
            None => Ok(false),
        }
    }

    /// Number of captured device-graph segments installed by the most recent
    /// [`Self::try_capture_decode_inline_with_device_bindings`] call on the
    /// sibling (`0` when no sibling, or nothing captured; `>= 1` once the
    /// inlined body folds; `>= 2` when eager seams split it). Backs the PR-3
    /// capture-engagement test.
    pub fn decode_inline_captured_graph_segment_count(&self) -> usize {
        self.decode_inline_exec
            .as_ref()
            .map(executor::Executor::captured_segment_count)
            .unwrap_or(0)
    }

    /// Structured segment boundaries from the sibling's most recent capture (one
    /// entry per eager seam node between captured segments, with its seam kind and
    /// `CaptureSupport` decline reason). Empty for a whole-subgraph capture or no
    /// sibling.
    pub fn decode_inline_capture_segmentation(&self) -> &[CaptureDecline] {
        self.decode_inline_exec
            .as_ref()
            .map(executor::Executor::capture_segmentation)
            .unwrap_or(&[])
    }

    /// Lazily build the verify-dedicated sibling executor for native MTP
    /// self-speculative decode. Idempotent: a second call is a no-op. The
    /// sibling is a structural clone of the main executor's graph with its own
    /// interior device-buffer arena (so the fixed M=k+1 verify's interior scratch
    /// is never resized by the interleaved M=1 decode on the main executor), and
    /// it drives the [`DeviceGraphSlot::Verify`] slot. Also pins the sibling's
    /// fixed-capacity KV sequence-axis symbols so its captured verify graph
    /// admits the attention nodes, mirroring the main executor. The main/prefill
    /// executor is left byte-identical.
    pub fn enable_verify_sibling(&mut self) -> Result<bool> {
        if self.verify_exec.is_none() {
            let mut sibling = self.exec.build_verify_sibling()?;
            sibling.pin_fixed_capacity_kv_capture_symbols();
            self.verify_exec = Some(sibling);
        }
        Ok(self.verify_exec.is_some())
    }

    /// Whether the verify-dedicated sibling executor is built and ready to run.
    pub fn verify_sibling_ready(&self) -> bool {
        self.verify_exec.is_some()
    }

    /// Run one fixed M=k+1 verify forward through the verify-dedicated sibling
    /// executor (eager), binding the identical persistent external device buffers
    /// `bindings` supplies (KV cache, recurrent/conv state) while the sibling's
    /// interior scratch stays private. Errors if
    /// [`Self::enable_verify_sibling`] has not built a sibling.
    pub fn run_verify_sibling_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceBindingOutputs> {
        let exec = self.verify_exec.as_mut().ok_or_else(|| {
            SessionError::Internal(
                "verify sibling requested but not built; call enable_verify_sibling first".into(),
            )
        })?;
        exec.run_with_device_bindings(inputs, bindings)
    }

    /// Capture-record one fixed M=k+1 verify forward of the verify-dedicated
    /// sibling into its `Verify`-slot device graph. Mirrors
    /// [`Self::try_capture_with_device_bindings`] but drives the sibling executor,
    /// whose interior arena is private so the captured graph's baked interior
    /// pointers are not resized by the interleaved M=1 decode. Errors if
    /// [`Self::enable_verify_sibling`] has not built a sibling.
    pub fn try_capture_verify_sibling_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        let exec = self.verify_exec.as_mut().ok_or_else(|| {
            SessionError::Internal(
                "verify sibling capture requested but no sibling built; call enable_verify_sibling first".into(),
            )
        })?;
        exec.try_capture_with_device_bindings(inputs, bindings)
    }

    /// Replay the verify-dedicated sibling's installed `Verify`-slot device graph.
    /// Returns `false` when a seam retired the graph this step (logits produced
    /// eagerly; caller re-warms/re-captures). Errors if
    /// [`Self::enable_verify_sibling`] has not built a sibling.
    pub fn replay_verify_sibling_device_graph(
        &mut self,
        bindings: &mut [DeviceIoBinding],
    ) -> Result<bool> {
        let exec = self.verify_exec.as_mut().ok_or_else(|| {
            SessionError::Internal(
                "verify sibling replay requested but no sibling built; call enable_verify_sibling first".into(),
            )
        })?;
        exec.replay_device_graph(bindings)
    }

    /// Invalidate the verify-dedicated sibling's installed device graph. A no-op
    /// returning `false` when no sibling has been built.
    pub fn reset_verify_sibling_device_graph(&mut self) -> Result<bool> {
        match self.verify_exec.as_mut() {
            Some(exec) => exec.reset_device_graph(),
            None => Ok(false),
        }
    }

    /// Number of captured device-graph segments installed by the most recent
    /// verify-sibling capture (`0` when no sibling or nothing captured).
    pub fn verify_sibling_captured_graph_segment_count(&self) -> usize {
        self.verify_exec
            .as_ref()
            .map(executor::Executor::captured_segment_count)
            .unwrap_or(0)
    }

    /// Allocate a persistent buffer on this session's execution device.
    pub fn allocate_device_binding(
        &self,
        input_name: impl Into<String>,
        output_name: Option<impl Into<String>>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        self.exec.allocate_device_binding(
            input_name.into(),
            output_name.map(Into::into),
            dtype,
            physical_shape,
            logical_shape,
        )
    }

    /// Allocate a persistent binding whose virtual allocation is larger than
    /// the shape currently exposed to kernels.
    ///
    /// Lazy device allocators use `committed_ranges` to map only the parts that
    /// are live. Eager allocators preserve the old behaviour and commit the
    /// whole allocation, so callers must gate memory-sensitive use on
    /// [`InferenceSession::commits_on_demand`].
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_device_binding_committed(
        &self,
        input_name: impl Into<String>,
        output_name: Option<impl Into<String>>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
        allocation_bytes: usize,
        committed_ranges: Vec<std::ops::Range<usize>>,
    ) -> Result<DeviceIoBinding> {
        self.exec.allocate_device_binding_committed(
            input_name.into(),
            output_name.map(Into::into),
            dtype,
            physical_shape,
            logical_shape,
            allocation_bytes,
            committed_ranges,
        )
    }

    /// Bind a persistent buffer the **caller** allocated on this session's
    /// execution device.
    ///
    /// [`Session::allocate_device_binding`] has the execution provider allocate,
    /// which puts the bytes outside any budget kept elsewhere. This entry point
    /// is for the opposite arrangement: a memory manager owns the allocation and
    /// lends it to the session, so device memory can be leased, pooled, or
    /// migrated by code this crate knows nothing about.
    ///
    /// The session **borrows**; dropping the binding does not free the buffer.
    ///
    /// # Safety
    ///
    /// * `ptr` must be non-null, on this session's device, correctly aligned,
    ///   and at least `len_bytes` long; `len_bytes` must cover
    ///   `physical_shape` (this part is checked and reported).
    /// * The allocation must outlive the returned binding and every run that
    ///   reads or writes it, including a captured device graph that recorded
    ///   its address.
    /// * Nothing else may write to the memory while the binding is live.
    pub unsafe fn device_binding_from_external_memory(
        &self,
        spec: crate::tensor::ExternalMemorySpec,
    ) -> Result<DeviceIoBinding> {
        // SAFETY: delegated to this function's contract.
        unsafe { self.exec.device_binding_from_external_memory(spec) }
    }

    /// Allocate a persistent buffer for a graph output without also binding it
    /// as an input.
    pub fn allocate_device_output_binding(
        &self,
        output_name: impl Into<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        self.exec.allocate_device_output_binding(
            output_name.into(),
            dtype,
            physical_shape,
            logical_shape,
        )
    }

    /// Execute once while recording the kernel launches into a device graph.
    ///
    /// `NotCapturable` means the mandatory all-kernel audit rejected the run
    /// before stream capture began, so callers may safely retry eagerly.
    pub fn try_capture_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        self.exec.try_capture_with_device_bindings(inputs, bindings)
    }

    /// Replay the installed device graph after the caller has refreshed any
    /// persistent scalar inputs. Returns `true` when the graph is still valid for
    /// the next step, or `false` when a control-flow branch flip retired it this
    /// step (the token was produced correctly via eager fallback) and the caller
    /// should re-warm and re-capture.
    pub fn replay_device_graph(&mut self, bindings: &mut [DeviceIoBinding]) -> Result<bool> {
        self.exec.replay_device_graph(bindings)
    }

    /// Invalidate the installed device graph before reset, rewind, shape change,
    /// or binding destruction.
    pub fn reset_device_graph(&mut self) -> Result<bool> {
        self.exec.reset_device_graph()
    }

    /// Which captured-graph slot the main executor currently drives.
    pub fn main_exec_graph_slot(&self) -> DeviceGraphSlot {
        self.exec.graph_slot()
    }

    /// Route the **main** executor's CUDA-graph capture/replay/reset to the given
    /// slot. The main executor defaults to [`DeviceGraphSlot::Primary`]; native
    /// MTP self-speculative decode retargets it to [`DeviceGraphSlot::Verify`]
    /// around each fixed M=k+1 verify forward so that forward captures/replays
    /// into an independent slot, then switches back to `Primary` for the M=1
    /// decode. Because the executor now holds **per-slot** host capture state,
    /// the switch is a pure retarget — it does NOT reset the other slot's
    /// installed graph — so `Primary` (M=1) and `Verify` (M=k+1) graphs coexist
    /// and each replays across steps.
    ///
    /// Safe to leave at Primary (the default) for every non-MTP path, which keeps
    /// greedy byte-identical: all main-exec graph calls then route to the same
    /// single slot they always did.
    pub fn set_main_exec_graph_slot(&mut self, slot: DeviceGraphSlot) -> Result<()> {
        self.exec.set_graph_slot(slot)
    }

    /// Whether the main executor's StepScoped workspace is pinned across runs.
    pub fn main_exec_step_workspace_pinned(&self) -> bool {
        self.exec.step_workspace_pinned()
    }

    /// Pin (or unpin) the **main** executor's StepScoped workspace across runs.
    /// Native MTP verify capture pins it so the captured fixed-M verify graph
    /// replays against a stable scratch pointer even though the M=1 decode step
    /// (on the sibling executor) reserves a smaller scratch in between (#1647).
    /// Inert by default; leaving it unpinned keeps every non-verify path
    /// byte-identical.
    pub fn set_main_exec_pin_step_workspace(&mut self, pin: bool) {
        self.exec.set_pin_step_workspace(pin);
    }

    /// Pin the fixed-capacity KV sequence-axis symbols CONSTANT so CUDA-graph
    /// capture ADMITS the attention nodes (`GroupQueryAttention` in particular)
    /// instead of vetoing each as a growing-seq eager seam. Returns the total
    /// number of symbols pinned across this session's executors.
    ///
    /// Call this ONLY once the engine has bound fixed-capacity, device-resident
    /// KV (physical `[.., max_len, ..]`, valid length read on-device) and CUDA
    /// graphs are enabled — see
    /// [`executor::Executor::pin_fixed_capacity_kv_capture_symbols`] for the full
    /// correctness argument. A growing/paged KV decoder must NOT call this.
    pub fn pin_fixed_capacity_kv_capture_symbols(&mut self) -> usize {
        let mut pinned = self.exec.pin_fixed_capacity_kv_capture_symbols();
        if let Some(inline) = self.decode_inline_exec.as_mut() {
            pinned += inline.pin_fixed_capacity_kv_capture_symbols();
        }
        if let Some(verify) = self.verify_exec.as_mut() {
            pinned += verify.pin_fixed_capacity_kv_capture_symbols();
        }
        pinned
    }

    /// Number of captured device-graph segments installed by the most recent
    /// [`Self::try_capture_with_device_bindings`] call.
    ///
    /// `1` for a whole-subgraph capture; `>= 2` when the CUDA EP claimed the
    /// subgraph but split it into segments around non-capturable seam nodes.
    pub fn captured_graph_segment_count(&self) -> usize {
        self.exec.captured_segment_count()
    }

    /// Structured, transparent segment boundaries from the most recent capture:
    /// one entry per non-capturable seam node the EP ran eagerly between captured
    /// segments (with its structural seam kind and `CaptureSupport` decline reason).
    /// Empty for a whole-subgraph capture.
    pub fn capture_segmentation(&self) -> &[CaptureDecline] {
        self.exec.capture_segmentation()
    }

    /// Read (without clearing) any latching device capture-safety error recorded
    /// during graph replay, as a raw violation bitmask (zero when none). Callers
    /// poll this at the per-step logits sync to fail before consuming a token
    /// produced from an out-of-range captured replay.
    pub fn check_device_capture_error(&self) -> Result<u32> {
        self.exec.check_device_capture_error()
    }

    pub fn device_allocation_counts(&self) -> Option<DeviceAllocationCounts> {
        self.exec.device_allocation_counts()
    }

    pub fn raw_device_allocation_site_stats(
        &self,
    ) -> Vec<onnx_runtime_ep_api::RawDeviceAllocationSiteStats> {
        self.exec.raw_device_allocation_site_stats()
    }

    /// Place any long-lived device memory this session's provider holds under
    /// `governor`.
    ///
    /// A provider that keeps a standing pool -- the CUDA weight-residency cache
    /// is one -- otherwise sizes it for itself, which is a second claim on
    /// memory the governor is already dividing up. Returns the bytes now
    /// governed; zero means this provider holds no standing pool.
    /// Whether the memory behind this session commits physically as it is used.
    ///
    /// Answered by the allocator rather than by the backend, so a caller does
    /// not need to know whether it is holding a native session or an ONNX
    /// Runtime one -- both reach the same `DeviceAllocator` seam.
    pub fn commits_on_demand(&self) -> bool {
        self.exec.commits_on_demand()
    }

    pub fn adopt_memory_governor(
        &self,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> Result<u64> {
        Ok(self.exec.adopt_memory_governor(governor, tier, holder)?)
    }

    /// Release each intermediate value's buffer once its last consumer has run.
    ///
    /// Off by default: a session that records a device graph needs stable buffer
    /// addresses across the recording run and its replays, and freeing lets the
    /// allocator hand out a different address next time. Enable it on graphs that
    /// never capture -- a prompt-phase vision encoder is the motivating case,
    /// where holding all 2545 intermediates at once cost ~20x the live set.
    pub fn set_release_dead_values(&mut self, enabled: bool) {
        self.exec.set_release_dead_values(enabled);
    }

    pub fn set_weight_residency_budget(&self, budget_bytes: u64) -> Result<Option<u64>> {
        Ok(self.exec.set_weight_residency_budget(budget_bytes)?)
    }

    pub fn max_lazy_weight_working_set_bytes(&self) -> u64 {
        self.exec.max_lazy_weight_working_set_bytes()
    }

    pub fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.exec.device_id()
    }

    /// Report why an explicitly requested accelerator session was assigned to
    /// CPU instead. `None` means the requested EP serves the whole graph.
    pub fn execution_provider_fallback_report(&self) -> Option<&ExecutionProviderFallbackReport> {
        self.exec.execution_provider_fallback_report()
    }

    /// Per-provider node counts and cross-provider transfer count for an
    /// opt-in heterogeneous session. `None` for the unchanged single-EP path.
    pub fn heterogeneous_placement_report(&self) -> Option<&str> {
        self.exec.heterogeneous_placement_report()
    }

    /// Input metadata.
    pub fn inputs(&self) -> &[IoMeta] {
        &self.inputs
    }

    /// Output metadata.
    pub fn outputs(&self) -> &[IoMeta] {
        &self.outputs
    }

    /// Model-level metadata from the source `ModelProto`.
    pub fn model_metadata(&self) -> &ModelMetadata {
        &self.model_metadata
    }

    /// Kernel-cache statistics (§11.1); useful to observe warmup/run reuse.
    pub fn cache_stats(&self) -> CacheStats {
        self.exec.cache_stats()
    }

    /// Control-flow subgraph build/run statistics. A Loop or Scan body with a
    /// stable input-shape signature should build once and run many times.
    pub fn control_flow_stats(&self) -> ControlFlowStats {
        self.exec.control_flow_stats()
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
        if self.ep_context_config.enable {
            reject_mixed_versions_for_ep_context_export(self.exec.graph())?;
        }
        let model = EncoderModel::new(self.exec.graph()).with_weights(self.exec.weights().as_ref());
        dump_session_ep_context(&model, orig_path, partitions, &self.ep_context_config)
    }
}

fn reject_mixed_versions_for_ep_context_export(graph: &onnx_runtime_ir::Graph) -> Result<()> {
    for node in graph.nodes.values() {
        // A version no reader honours is not lost by an export that drops it,
        // so judge representability with the same rule everything else uses.
        let Some(node_version) = node.local_opset() else {
            continue;
        };
        let graph_version = graph.opset_imports.get(node.domain.as_str()).copied();
        if graph_version == Some(node_version) {
            continue;
        }
        return Err(SessionError::EpContextMixedNodeVersion {
            op_type: node.op_type.clone(),
            node: if node.name.is_empty() {
                format!("#{}", node.id.0)
            } else {
                node.name.clone()
            },
            domain: if node.domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                node.domain.clone()
            },
            node_version,
            graph_version: graph_version.map_or_else(|| "missing".to_string(), |v| v.to_string()),
        });
    }
    Ok(())
}

/// Load a model. Auto-detects the best available hardware (§20.2).
///
/// This is the primary entry point — no configuration required.
pub fn load(path: impl AsRef<Path>) -> Result<InferenceSession> {
    InferenceSession::load(path)
}

#[cfg(test)]
mod device_binding_tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use onnx_runtime_ir::Attribute;
    #[cfg(feature = "cuda")]
    use onnx_runtime_ir::static_shape;
    use onnx_runtime_ir::{Graph, Node, NodeId};

    #[test]
    fn persistent_binding_aliases_input_output_and_suppresses_materialization() {
        let mut graph = Graph::new();
        graph.opset_imports.insert("".into(), 13);
        let length = graph.intern_symbol("length");
        let input = graph.create_named_value("input", DataType::Float32, vec![length.into()]);
        graph.add_input(input);
        let output = graph.create_named_value("output", DataType::Float32, vec![length.into()]);
        graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        graph.add_output(output);
        let mut session = InferenceSession::from_graph(graph).unwrap();
        let mut binding = session
            .allocate_device_binding("input", Some("output"), DataType::Float32, vec![4], vec![2])
            .unwrap();
        let ptr = binding.device_ptr();
        let bytes = [-2.0f32, 3.0, -4.0, 5.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        binding.write_bytes(0, &bytes).unwrap();

        let outputs = session
            .run_with_device_bindings(&[], std::slice::from_mut(&mut binding))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_none());
        assert_eq!(binding.device_ptr(), ptr);
        assert_eq!(binding.logical_shape(), &[2]);
        let values = binding
            .read_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0.0, 3.0, -4.0, 5.0]);
        assert_eq!(
            binding.transfer_stats(),
            DeviceBindingTransferStats {
                host_upload_calls: 1,
                host_upload_bytes: 16,
                host_download_calls: 1,
                host_download_bytes: 16,
            }
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_graph_replay_uses_persistent_io_without_device_allocations() {
        let Ok(mut ep) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) else {
            eprintln!("skipping session CUDA graph test: CUDA runtime unavailable");
            return;
        };
        onnx_runtime_ep_api::ExecutionProvider::initialize(&mut ep, &Default::default()).unwrap();

        let mut graph = Graph::new();
        graph.opset_imports.insert("".into(), 13);
        let input = graph.create_named_value("input", DataType::Int64, static_shape([1]));
        graph.add_input(input);
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
        let mut cast = Node::new(NodeId(0), "Cast", vec![Some(input)], vec![output]);
        cast.attributes
            .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
        graph.insert_node(cast);
        graph.add_output(output);

        let mut session = InferenceSession::from_parts(
            graph,
            std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            EpContextDumpConfig::default(),
            ModelMetadata::default(),
            std::sync::Arc::new(ep),
        )
        .unwrap();
        let mut input = session
            .allocate_device_binding("input", None::<String>, DataType::Int64, vec![1], vec![1])
            .unwrap();
        let output = session
            .allocate_device_output_binding("output", DataType::Float32, vec![1], vec![1])
            .unwrap();
        input.write_bytes(0, &7i64.to_le_bytes()).unwrap();
        let mut bindings = vec![input, output];
        session
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        assert_eq!(read_bound_f32(&mut bindings[1]), 7.0);

        input_write(&mut bindings[0], 11);
        assert!(matches!(
            session
                .try_capture_with_device_bindings(&[], &mut bindings)
                .unwrap(),
            DeviceGraphCaptureResult::Captured(_)
        ));
        assert_eq!(read_bound_f32(&mut bindings[1]), 11.0);

        let before = session.device_allocation_counts().unwrap();
        input_write(&mut bindings[0], 23);
        session.replay_device_graph(&mut bindings).unwrap();
        assert_eq!(read_bound_f32(&mut bindings[1]), 23.0);
        assert_eq!(session.device_allocation_counts().unwrap(), before);
        assert!(session.reset_device_graph().unwrap());

        input_write(&mut bindings[0], 31);
        assert!(matches!(
            session
                .try_capture_with_device_bindings(&[], &mut bindings)
                .unwrap(),
            DeviceGraphCaptureResult::Captured(_)
        ));
        assert_eq!(read_bound_f32(&mut bindings[1]), 31.0);
        input_write(&mut bindings[0], 47);
        session.replay_device_graph(&mut bindings).unwrap();
        assert_eq!(read_bound_f32(&mut bindings[1]), 47.0);
        assert!(session.reset_device_graph().unwrap());
    }

    /// The main executor can drive the **Verify** captured-graph slot (the second
    /// independent slot added for option-c) end-to-end through the session +
    /// executor layers, not just at the raw EP: retargeting the slot, capturing,
    /// replaying with persistent I/O, and resetting all succeed on the Verify
    /// slot exactly as they do on Primary. This is the executor-level lift of the
    /// EP-level `primary_and_verify_graph_slots_are_independent` proof — the
    /// plumbing native MTP verify capture will drive.
    #[cfg(feature = "cuda")]
    #[test]
    fn main_exec_drives_verify_graph_slot_end_to_end() {
        let Ok(mut ep) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) else {
            eprintln!("skipping session Verify-slot test: CUDA runtime unavailable");
            return;
        };
        onnx_runtime_ep_api::ExecutionProvider::initialize(&mut ep, &Default::default()).unwrap();

        let mut graph = Graph::new();
        graph.opset_imports.insert("".into(), 13);
        let input = graph.create_named_value("input", DataType::Int64, static_shape([1]));
        graph.add_input(input);
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
        let mut cast = Node::new(NodeId(0), "Cast", vec![Some(input)], vec![output]);
        cast.attributes
            .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
        graph.insert_node(cast);
        graph.add_output(output);

        let mut session = InferenceSession::from_parts(
            graph,
            std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            EpContextDumpConfig::default(),
            ModelMetadata::default(),
            std::sync::Arc::new(ep),
        )
        .unwrap();

        // Default routing is Primary; retarget the main executor to Verify.
        assert_eq!(session.main_exec_graph_slot(), DeviceGraphSlot::Primary);
        session
            .set_main_exec_graph_slot(DeviceGraphSlot::Verify)
            .unwrap();
        assert_eq!(session.main_exec_graph_slot(), DeviceGraphSlot::Verify);

        let mut input = session
            .allocate_device_binding("input", None::<String>, DataType::Int64, vec![1], vec![1])
            .unwrap();
        let output = session
            .allocate_device_output_binding("output", DataType::Float32, vec![1], vec![1])
            .unwrap();
        input.write_bytes(0, &7i64.to_le_bytes()).unwrap();
        let mut bindings = vec![input, output];
        session
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        assert_eq!(read_bound_f32(&mut bindings[1]), 7.0);

        // Capture into the Verify slot and replay with persistent I/O (no new
        // device allocations across the replay) — the same guarantees Primary has.
        input_write(&mut bindings[0], 11);
        assert!(matches!(
            session
                .try_capture_with_device_bindings(&[], &mut bindings)
                .unwrap(),
            DeviceGraphCaptureResult::Captured(_)
        ));
        assert_eq!(read_bound_f32(&mut bindings[1]), 11.0);

        let before = session.device_allocation_counts().unwrap();
        input_write(&mut bindings[0], 23);
        session.replay_device_graph(&mut bindings).unwrap();
        assert_eq!(read_bound_f32(&mut bindings[1]), 23.0);
        assert_eq!(session.device_allocation_counts().unwrap(), before);
        assert!(session.reset_device_graph().unwrap());

        // Switching back to Primary keeps the API working (resets the Verify slot
        // on the way out) — proving the retarget is reversible and inert.
        session
            .set_main_exec_graph_slot(DeviceGraphSlot::Primary)
            .unwrap();
        assert_eq!(session.main_exec_graph_slot(), DeviceGraphSlot::Primary);
        input_write(&mut bindings[0], 31);
        assert!(matches!(
            session
                .try_capture_with_device_bindings(&[], &mut bindings)
                .unwrap(),
            DeviceGraphCaptureResult::Captured(_)
        ));
        assert_eq!(read_bound_f32(&mut bindings[1]), 31.0);
        input_write(&mut bindings[0], 47);
        session.replay_device_graph(&mut bindings).unwrap();
        assert_eq!(read_bound_f32(&mut bindings[1]), 47.0);
        assert!(session.reset_device_graph().unwrap());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn segmented_cuda_graph_claims_whole_subgraph_around_eager_seam() {
        let Ok(mut ep) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) else {
            eprintln!("skipping segmented session CUDA graph test: CUDA runtime unavailable");
            return;
        };
        onnx_runtime_ep_api::ExecutionProvider::initialize(&mut ep, &Default::default()).unwrap();

        // A decoder-like chain with a deliberately non-capturable node in the
        // middle: input -> Cast(f32) -> Clip(min/max attrs) -> Cast(i64) -> out.
        // Cast is CUDA-graph capture-safe (it skips its trailing sync while the
        // stream is capturing); Clip declines capture, so it forces a segment
        // boundary while remaining CUDA-placed. Over integer inputs and a wide
        // clip the chain round-trips to the identity.
        let n = 4usize;
        let mut graph = Graph::new();
        graph.opset_imports.insert("".into(), 13);
        let input = graph.create_named_value("input", DataType::Int64, static_shape([n]));
        graph.add_input(input);
        let as_float = graph.create_named_value("as_float", DataType::Float32, static_shape([n]));
        let mut cast_in = Node::new(NodeId(0), "Cast", vec![Some(input)], vec![as_float]);
        cast_in
            .attributes
            .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
        graph.insert_node(cast_in);
        let clipped = graph.create_named_value("clipped", DataType::Float32, static_shape([n]));
        let mut clip = Node::new(NodeId(1), "Clip", vec![Some(as_float)], vec![clipped]);
        clip.attributes
            .insert("min".into(), Attribute::Float(-1000.0));
        clip.attributes
            .insert("max".into(), Attribute::Float(1000.0));
        graph.insert_node(clip);
        let output = graph.create_named_value("output", DataType::Int64, static_shape([n]));
        let mut cast_out = Node::new(NodeId(2), "Cast", vec![Some(clipped)], vec![output]);
        cast_out
            .attributes
            .insert("to".into(), Attribute::Int(DataType::Int64 as i64));
        graph.insert_node(cast_out);
        graph.add_output(output);

        let mut session = InferenceSession::from_parts(
            graph,
            std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            EpContextDumpConfig::default(),
            ModelMetadata::default(),
            std::sync::Arc::new(ep),
        )
        .unwrap();

        let input_binding = session
            .allocate_device_binding("input", None::<String>, DataType::Int64, vec![n], vec![n])
            .unwrap();
        let output_binding = session
            .allocate_device_output_binding("output", DataType::Int64, vec![n], vec![n])
            .unwrap();
        let mut bindings = vec![input_binding, output_binding];

        // Warmup / eager reference for input A (also warms the shape-keyed
        // kernels the capture pass requires).
        let values_a = [-2i64, 3, -4, 5];
        write_bound_i64(&mut bindings[0], &values_a);
        session
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        let eager_a = read_bound_i64_vec(&mut bindings[1]);
        assert_eq!(
            eager_a,
            values_a.to_vec(),
            "Cast∘Clip∘Cast round-trips ints"
        );

        // Segmented capture: the whole subgraph is still claimed and run on the
        // CUDA EP even though Clip is not capturable.
        match session
            .try_capture_with_device_bindings(&[], &mut bindings)
            .unwrap()
        {
            DeviceGraphCaptureResult::Captured(outputs) => {
                assert!(
                    outputs.iter().all(Option::is_none),
                    "device-bound outputs must not materialize to host"
                );
            }
            DeviceGraphCaptureResult::NotCapturable(report) => {
                panic!(
                    "expected the CUDA EP to claim the whole subgraph via segmented capture, \
                     got a full decline: {report}"
                );
            }
        }
        // Whole-subgraph claim, split into two captured segments around one seam.
        assert_eq!(
            session.captured_graph_segment_count(),
            2,
            "Clip should split the plan into two captured Cast segments"
        );
        let seams = session.capture_segmentation();
        assert_eq!(seams.len(), 1, "exactly one eager seam node (Clip)");
        assert_eq!(seams[0].op_type, "Clip");
        // Token-exact: the segmented capture pass matches the eager reference.
        assert_eq!(read_bound_i64_vec(&mut bindings[1]), eager_a);

        // Segmented replay for a new input B interleaves the two captured Cast
        // segment graphs with the eager Clip seam, and stays token-exact.
        let values_b = [7i64, -1, 0, -8];
        write_bound_i64(&mut bindings[0], &values_b);
        session.replay_device_graph(&mut bindings).unwrap();
        let replay_b = read_bound_i64_vec(&mut bindings[1]);

        // Independent eager reference for input B.
        assert!(session.reset_device_graph().unwrap());
        write_bound_i64(&mut bindings[0], &values_b);
        session
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        let eager_b = read_bound_i64_vec(&mut bindings[1]);
        assert_eq!(
            replay_b, eager_b,
            "segmented replay must be bit-identical to eager execution"
        );
        assert_eq!(replay_b, values_b.to_vec());
    }

    /// Inc-1b PR-3 capture-engagement (design §4 / Harry #588 rec #2, #3): the
    /// decode-inline sibling folds its inlined body ops into a captured device
    /// graph (`decode_inline_captured_graph_segment_count() >= 1`) while staying
    /// byte-exact with the eager sibling run — and the shared-EP invalidation path
    /// (`reset_decode_inline_device_graph`) drops that graph cleanly. Distinct
    /// (non-aliased) present/past bindings per design §3, so each run is idempotent
    /// for a fixed input and the eager vs captured comparison is exact.
    #[cfg(feature = "cuda")]
    #[test]
    fn decode_inline_sibling_folds_body_into_captured_graph_byte_exact() {
        use onnx_runtime_ir::Dim;

        let Ok(mut ep) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) else {
            eprintln!("skipping decode-inline capture-engagement test: CUDA runtime unavailable");
            return;
        };
        onnx_runtime_ep_api::ExecutionProvider::initialize(&mut ep, &Default::default()).unwrap();

        const W: usize = 3;
        // Hybrid single-trip recurrent Scan: state threaded through the body,
        // one scan input at decode extent 1. Body: present = state + scan_in;
        // scan_out = present * scan_in. Distinct present/past (no in-place alias).
        let mut body = Graph::new();
        body.opset_imports.insert(String::new(), 17);
        let state = body.create_named_value("state", DataType::Float32, static_shape([W]));
        let scan_in = body.create_named_value("scan_in", DataType::Float32, static_shape([W]));
        body.add_input(state);
        body.add_input(scan_in);
        let present = body.create_named_value("present", DataType::Float32, static_shape([W]));
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(state), Some(scan_in)],
            vec![present],
        ));
        let y = body.create_named_value("y", DataType::Float32, static_shape([W]));
        body.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(present), Some(scan_in)],
            vec![y],
        ));
        body.add_output(present);
        body.add_output(y);

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let seq = graph.intern_symbol("seq");
        let past_state =
            graph.create_named_value("past_state", DataType::Float32, static_shape([W]));
        graph.add_input(past_state);
        let x =
            graph.create_named_value("x", DataType::Float32, vec![Dim::from(seq), Dim::Static(W)]);
        graph.add_input(x);
        let present_state =
            graph.create_named_value("present_state", DataType::Float32, static_shape([W]));
        let scan_out = graph.create_named_value(
            "scan_out",
            DataType::Float32,
            vec![Dim::from(seq), Dim::Static(W)],
        );
        let mut scan = Node::new(
            NodeId(0),
            "Scan",
            vec![Some(past_state), Some(x)],
            vec![present_state, scan_out],
        );
        scan.attributes
            .insert("num_scan_inputs".to_string(), Attribute::Int(1));
        let scan_id = graph.insert_node(scan);
        graph.subgraphs.insert((scan_id, "body".to_string()), body);
        graph.add_output(present_state);
        graph.add_output(scan_out);

        let mut session = InferenceSession::from_parts(
            graph,
            std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            EpContextDumpConfig::default(),
            ModelMetadata::default(),
            std::sync::Arc::new(ep),
        )
        .unwrap();

        assert!(
            session.enable_decode_inline().unwrap(),
            "single-trip recurrent Scan must yield a decode-inline sibling"
        );

        // Persistent device bindings: past_state + x as inputs, present_state +
        // scan_out as outputs (every graph output MUST be persistently bound for
        // capture). Decode extent 1, so x is [1, W].
        let mut past = session
            .allocate_device_binding(
                "past_state",
                None::<String>,
                DataType::Float32,
                vec![W],
                vec![W],
            )
            .unwrap();
        let mut x_bind = session
            .allocate_device_binding(
                "x",
                None::<String>,
                DataType::Float32,
                vec![1, W],
                vec![1, W],
            )
            .unwrap();
        let present_bind = session
            .allocate_device_output_binding("present_state", DataType::Float32, vec![W], vec![W])
            .unwrap();
        let scan_bind = session
            .allocate_device_output_binding("scan_out", DataType::Float32, vec![1, W], vec![1, W])
            .unwrap();
        let write_f32 = |b: &mut DeviceIoBinding, v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            b.write_bytes(0, &bytes).unwrap();
        };
        let read_f32 = |b: &mut DeviceIoBinding| -> Vec<f32> {
            b.read_bytes()
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        write_f32(&mut past, &[1.0, 2.0, 3.0]);
        write_f32(&mut x_bind, &[0.5, 1.5, 2.5]);
        let mut bindings = vec![past, x_bind, present_bind, scan_bind];

        // Eager warmup (also warms the shape-keyed kernels capture requires).
        session
            .run_decode_inline_with_device_bindings(&[], &mut bindings)
            .unwrap();
        let eager_present = read_f32(&mut bindings[2]);
        let eager_scan = read_f32(&mut bindings[3]);
        // present = past + x ; scan_out = present * x.
        assert_eq!(eager_present, vec![1.5, 3.5, 5.5]);
        assert_eq!(eager_scan, vec![0.75, 5.25, 13.75]);

        // Capture pass: the inlined body (Add + Mul) must fold into >= 1 captured
        // device-graph segment (Harry #2: segment growth over the body) while
        // staying byte-exact with the eager reference (Harry #3: the inlined
        // interior shapes were warm-seeded so capture engages).
        match session
            .try_capture_decode_inline_with_device_bindings(&[], &mut bindings)
            .unwrap()
        {
            DeviceGraphCaptureResult::Captured(outputs) => {
                assert!(
                    outputs.iter().all(Option::is_none),
                    "device-bound outputs must not materialize to host"
                );
            }
            DeviceGraphCaptureResult::NotCapturable(report) => {
                panic!("decode-inline body did not fold into a captured graph: {report}");
            }
        }
        assert!(
            session.decode_inline_captured_graph_segment_count() >= 1,
            "the inlined body must fold into at least one captured segment (engagement proof)"
        );
        assert_eq!(
            read_f32(&mut bindings[2]),
            eager_present,
            "captured present-state must be byte-exact"
        );
        assert_eq!(
            read_f32(&mut bindings[3]),
            eager_scan,
            "captured scan-out must be byte-exact"
        );

        // Replay stays byte-exact for the same inputs.
        session
            .replay_decode_inline_device_graph(&mut bindings)
            .unwrap();
        assert_eq!(read_f32(&mut bindings[2]), eager_present);
        assert_eq!(read_f32(&mut bindings[3]), eager_scan);

        // Shared-EP invalidation drops the sibling graph cleanly.
        assert!(session.reset_decode_inline_device_graph().unwrap());
        assert_eq!(
            session.decode_inline_captured_graph_segment_count(),
            0,
            "reset must clear the sibling's captured schedule"
        );
    }

    #[cfg(feature = "cuda")]
    fn write_bound_i64(binding: &mut DeviceIoBinding, values: &[i64]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        binding.write_bytes(0, &bytes).unwrap();
    }

    #[cfg(feature = "cuda")]
    fn read_bound_i64_vec(binding: &mut DeviceIoBinding) -> Vec<i64> {
        binding
            .read_bytes()
            .unwrap()
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn input_write(binding: &mut DeviceIoBinding, value: i64) {
        binding.write_bytes(0, &value.to_le_bytes()).unwrap();
    }

    #[cfg(feature = "cuda")]
    fn read_bound_f32(binding: &mut DeviceIoBinding) -> f32 {
        let bytes = binding.read_bytes().unwrap();
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
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
    fn ep_context_export_rejects_mixed_node_version() {
        let mut graph = onnx_runtime_ir::Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let x = graph.create_named_value(
            "x",
            onnx_runtime_ir::DataType::Float32,
            onnx_runtime_ir::static_shape([2]),
        );
        let y = graph.create_named_value(
            "y",
            onnx_runtime_ir::DataType::Float32,
            onnx_runtime_ir::static_shape([2]),
        );
        graph.add_input(x);
        graph.add_output(y);
        let mut node =
            onnx_runtime_ir::Node::new(onnx_runtime_ir::NodeId(0), "Swish", vec![Some(x)], vec![y]);
        node.version = Some(24);
        graph.insert_node(node);

        let err = reject_mixed_versions_for_ep_context_export(&graph).unwrap_err();
        assert!(matches!(
            err,
            SessionError::EpContextMixedNodeVersion {
                op_type,
                node_version: 24,
                graph_version,
                ..
            } if op_type == "Swish" && graph_version == "21"
        ));
    }

    #[test]
    fn optimization_defaults_to_none_when_unset() {
        assert_eq!(level_of(&[]).unwrap(), OptimizationLevel::None);
    }

    #[test]
    fn explicit_execution_provider_is_retained_by_builder() {
        let builder =
            SessionBuilder::new().execution_provider(executor::auto_detect_cpu_ep().unwrap());
        assert!(builder.execution_provider.is_some());
    }

    #[test]
    fn optimization_parses_known_values() {
        for (v, want) in [
            ("none", OptimizationLevel::None),
            ("off", OptimizationLevel::None),
            ("BASIC", OptimizationLevel::Basic),
            ("All", OptimizationLevel::All),
        ] {
            assert_eq!(
                level_of(&[("optimization", v)]).unwrap(),
                want,
                "value {v:?}"
            );
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
        assert_eq!(OptimizationLevel::All.passes().len(), 2);
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
        assert_eq!(
            ctx_of(&[("ep.context_embed_mode", "0")])
                .unwrap()
                .embed_mode,
            0
        );
        assert_eq!(
            ctx_of(&[("ep.context_embed_mode", "1")])
                .unwrap()
                .embed_mode,
            1
        );

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
