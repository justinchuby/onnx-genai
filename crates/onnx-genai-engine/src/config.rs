//! Public generation API types and configuration.

use crate::logits::{StopSequence, TokenId};
use onnx_genai_kv::{CachePriority, DEFAULT_CHUNK_SIZE, KvDType, LocalTieredConfig, SequenceId};
use onnx_genai_metadata::{GenerationContract, GenerationDefaults};
// The sidecar-descriptor mapping is native-only; an ORT-only build imports none
// of these.
#[cfg(feature = "native-backend")]
use onnx_genai_metadata::{
    MtpHiddenLayout as MetadataMtpHiddenLayout, MtpKvMode as MetadataMtpKvMode, MtpProposerSpec,
};
use onnx_genai_ort::{Eagle3DraftKvMode, MtpDraftKvMode};
use onnx_genai_scheduler::{Priority, ResourceLimit, ResourceLimits, SchedulerConfig};
use serde::Deserialize;
use std::path::PathBuf;

/// Error returned when a user-facing resource limit cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid resource limit {input:?}: {reason}; use a byte count (for example 8589934592), \
     a binary/decimal byte string (for example 8GiB or 8GB), a fraction in [0, 1] \
     (for example 0.9), or \"auto\""
)]
pub struct LimitParseError {
    input: String,
    reason: String,
}

/// Parse a user-facing resource ceiling.
///
/// Integers without a suffix are bytes. Decimal values without a suffix are
/// fractions, while suffixed values may be integral or decimal byte quantities.
pub fn parse_resource_limit(input: &str) -> Result<ResourceLimit, LimitParseError> {
    let input = input.trim();
    if input.eq_ignore_ascii_case("auto") {
        return Ok(ResourceLimit::Auto);
    }
    if let Ok(bytes) = input.parse::<u64>() {
        return Ok(ResourceLimit::Bytes(bytes));
    }

    let unit_start = input
        .find(|character: char| character.is_ascii_alphabetic())
        .unwrap_or(input.len());
    let (number, unit) = input.split_at(unit_start);
    if unit.is_empty() {
        let fraction = number.parse::<f64>().map_err(|_| {
            limit_error(
                input,
                "the value is neither an integer byte count nor a numeric fraction",
            )
        })?;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(limit_error(
                input,
                "a unitless decimal is a fraction, but this value is outside [0, 1]",
            ));
        }
        return Ok(ResourceLimit::Fraction(fraction as f32));
    }

    let multiplier = match unit.to_ascii_uppercase().as_str() {
        "KIB" => 1_u64 << 10,
        "MIB" => 1_u64 << 20,
        "GIB" => 1_u64 << 30,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        _ => {
            return Err(limit_error(
                input,
                format!("the unit {unit:?} is not supported"),
            ));
        }
    };
    if number.chars().all(|character| character.is_ascii_digit()) {
        let quantity = number.parse::<u64>().map_err(|_| {
            limit_error(
                input,
                format!("the integral byte quantity {number:?} does not fit in u64"),
            )
        })?;
        let bytes = quantity.checked_mul(multiplier).ok_or_else(|| {
            limit_error(
                input,
                format!(
                    "multiplying {quantity} by the {unit} unit size ({multiplier} bytes) \
                     overflows u64; use a smaller byte quantity or a smaller unit"
                ),
            )
        })?;
        return Ok(ResourceLimit::Bytes(bytes));
    }
    let quantity = number.parse::<f64>().map_err(|_| {
        limit_error(
            input,
            format!("the numeric part {number:?} is not a valid non-negative number"),
        )
    })?;
    let bytes = quantity * multiplier as f64;
    if !quantity.is_finite() || quantity < 0.0 || !bytes.is_finite() || bytes >= u64::MAX as f64 {
        return Err(limit_error(
            input,
            "the byte quantity is negative, non-finite, or exceeds u64",
        ));
    }
    Ok(ResourceLimit::Bytes(bytes.round() as u64))
}

fn limit_error(input: impl Into<String>, reason: impl Into<String>) -> LimitParseError {
    LimitParseError {
        input: input.into(),
        reason: reason.into(),
    }
}

/// Error returned while decoding the resource-governor YAML surface.
#[derive(Debug, thiserror::Error)]
pub enum EngineConfigError {
    #[error("failed to parse engine YAML resource limits: {0}; check serving.memory.limits syntax")]
    Yaml(#[from] serde_yaml::Error),
    #[error("failed to parse serving.memory.limits.{field}: {source}")]
    Limit {
        field: &'static str,
        #[source]
        source: LimitParseError,
    },
    #[error("failed to parse serving.memory.weights.device_policy: {source}")]
    DevicePolicy {
        #[source]
        source: DevicePolicyParseError,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LimitValue {
    String(String),
    Integer(u64),
    Float(f64),
}

impl LimitValue {
    fn parse(self, field: &'static str) -> Result<ResourceLimit, EngineConfigError> {
        let parsed = match self {
            Self::String(value) => parse_resource_limit(&value),
            Self::Integer(value) => Ok(ResourceLimit::Bytes(value)),
            Self::Float(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {
                Ok(ResourceLimit::Fraction(value as f32))
            }
            Self::Float(value) => Err(limit_error(
                value.to_string(),
                "a YAML floating-point limit is a fraction, but this value is outside [0, 1]",
            )),
        };
        parsed.map_err(|source| EngineConfigError::Limit { field, source })
    }
}

#[derive(Debug, Default, Deserialize)]
struct LimitsYaml {
    vram_limit: Option<LimitValue>,
    host_ram_limit: Option<LimitValue>,
    disk_spill_limit: Option<LimitValue>,
    #[serde(default)]
    allow_runtime_override: bool,
}

#[derive(Debug, Default, Deserialize)]
struct MemoryYaml {
    #[serde(default)]
    limits: LimitsYaml,
    #[serde(default)]
    weights: WeightsYaml,
}

#[derive(Debug, Default, Deserialize)]
struct WeightsYaml {
    device_policy: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ServingYaml {
    #[serde(default)]
    memory: MemoryYaml,
}

#[derive(Debug, Default, Deserialize)]
struct EngineConfigYaml {
    #[serde(default)]
    serving: ServingYaml,
}

/// Source of a target weight shared with an MTP sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MtpWeightSource {
    /// Standalone raw little-endian f32 matrix used by legacy/manual configs.
    File(PathBuf),
    /// Exact initializer name in the target model package.
    TargetInitializer(String),
}

impl From<PathBuf> for MtpWeightSource {
    fn from(path: PathBuf) -> Self {
        Self::File(path)
    }
}

impl From<&str> for MtpWeightSource {
    fn from(path: &str) -> Self {
        Self::File(path.into())
    }
}

impl From<String> for MtpWeightSource {
    fn from(path: String) -> Self {
        Self::File(path.into())
    }
}

/// Target hidden-state layout consumed by an MTP sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpHiddenLayout {
    /// Legacy `[batch, sequence, hidden]` state.
    Bsh,
    /// Mobius `[batch, sequence, hc_mult, hidden]` Hyper-Connection state.
    Bshc,
}

/// Lifetime of an MTP sidecar's private KV state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpCacheScope {
    /// Reset sidecar KV at each target verification iteration.
    ProposalLocal,
    /// Retain KV corresponding to the accepted draft prefix.
    AcceptedPrefix,
}

/// Files and target-model outputs required for MTP.
///
/// The target decoder must emit both logits and the configured last-layer
/// hidden-state output on every forward. The embedding and LM-head files must
/// contain the exact target weights as little-endian f32 matrices; mismatched
/// weights remain greedy-correct because every candidate is target-verified,
/// but will reduce acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpConfig {
    /// ONNX model containing the MTP head.
    pub head_model: PathBuf,
    /// Target decoder output containing `[batch, sequence, hidden]` states.
    pub target_hidden_output: String,
    /// Raw little-endian f32 target embedding weights in `[vocab, hidden]` order.
    pub embedding_weights: PathBuf,
    /// Raw little-endian f32 target LM-head weights in `[hidden, vocab]` order.
    pub lm_head_weights: PathBuf,
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Target hidden size.
    pub hidden_size: usize,
    /// MTP-head cache strategy.
    pub kv_mode: MtpDraftKvMode,
    /// Number of speculative tokens produced after the guaranteed target token.
    pub num_speculative_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedMtpConfig {
    pub(crate) public_config: MtpConfig,
    pub(crate) target_hidden_layout: MtpHiddenLayout,
    pub(crate) embedding_weights: MtpWeightSource,
    pub(crate) lm_head_weights: MtpWeightSource,
    pub(crate) hc_mult: usize,
    pub(crate) mtp_hidden_output: String,
    pub(crate) mtp_state_output: Option<String>,
    pub(crate) cache_scope: MtpCacheScope,
}

impl ResolvedMtpConfig {
    pub(crate) fn from_manual(config: MtpConfig) -> Self {
        Self {
            embedding_weights: MtpWeightSource::File(config.embedding_weights.clone()),
            lm_head_weights: MtpWeightSource::File(config.lm_head_weights.clone()),
            public_config: config,
            target_hidden_layout: MtpHiddenLayout::Bsh,
            hc_mult: 1,
            mtp_hidden_output: "mtp_hidden".into(),
            mtp_state_output: None,
            cache_scope: MtpCacheScope::ProposalLocal,
        }
    }

    /// Resolve sidecar-discovered MTP settings without expanding the stable
    /// public hand-authored [`MtpConfig`] surface.
    ///
    /// Everything except the vocabulary comes from the sidecar's own
    /// declaration; the vocabulary belongs to the *target* (the head borrows the
    /// target's LM-head initializer), so the caller supplies it from the
    /// package's declared model capabilities.
    ///
    /// Only the native decode path resolves a sidecar proposer, so this stays
    /// gated with its single consumer rather than sitting dead in an ORT-only
    /// build.
    #[cfg(feature = "native-backend")]
    pub(crate) fn from_sidecar_descriptor(spec: &MtpProposerSpec, vocab_size: usize) -> Self {
        let public_config = MtpConfig {
            head_model: spec.model.clone(),
            target_hidden_output: spec.target_hidden_output.clone(),
            embedding_weights: PathBuf::from(&spec.embedding_initializer),
            lm_head_weights: PathBuf::from(&spec.lm_head_initializer),
            vocab_size,
            hidden_size: spec.target_hidden_size,
            kv_mode: MtpDraftKvMode::GrowCache,
            num_speculative_tokens: spec.num_speculative_tokens,
        };
        Self {
            public_config,
            target_hidden_layout: match spec.target_hidden_layout {
                MetadataMtpHiddenLayout::Bsh => MtpHiddenLayout::Bsh,
                MetadataMtpHiddenLayout::Bshc => MtpHiddenLayout::Bshc,
            },
            embedding_weights: MtpWeightSource::TargetInitializer(
                spec.embedding_initializer.clone(),
            ),
            lm_head_weights: MtpWeightSource::TargetInitializer(spec.lm_head_initializer.clone()),
            hc_mult: spec.hc_mult,
            mtp_hidden_output: spec.mtp_hidden_output.clone(),
            // A head that threads a recurrent state declares its output name; a
            // pure-attention (proposal-local) head declares none. The sidecar
            // schema keeps this optional, so honor exactly what was declared
            // rather than inventing a phantom "mtp_state" output the head does
            // not expose (which would make `MtpDecodeSession` reject it).
            mtp_state_output: spec.mtp_state_output.clone(),
            cache_scope: match spec.kv_mode {
                MetadataMtpKvMode::ProposalLocal => MtpCacheScope::ProposalLocal,
                MetadataMtpKvMode::AcceptedPrefix => MtpCacheScope::AcceptedPrefix,
            },
        }
    }
}

/// Files and target-model outputs required for EAGLE-3 speculation.
///
/// EAGLE-3 consumes exactly three target hidden-state outputs (low, middle,
/// high), concatenates their last-token rows, and autoregressively recycles the
/// draft head's own hidden output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Eagle3Config {
    /// ONNX model containing the EAGLE-3 draft head.
    pub head_model: PathBuf,
    /// Low, middle, and high target hidden-state output names, in that order.
    pub target_hidden_outputs: Vec<String>,
    /// Raw little-endian f32 target embedding weights in `[vocab, hidden]` order.
    pub embedding_weights: PathBuf,
    /// Optional raw little-endian i64 table mapping each proposer token id to
    /// the corresponding target token id. Absent means identical vocabularies.
    pub token_map: Option<PathBuf>,
    /// Target vocabulary size used by the shared embedding table.
    pub vocab_size: usize,
    /// Width of each target hidden state and token embedding.
    pub hidden_size: usize,
    /// EAGLE-3 head cache strategy.
    pub kv_mode: Eagle3DraftKvMode,
    /// Number of speculative tokens produced after the guaranteed target token.
    pub num_speculative_tokens: usize,
}

/// Built-in speculative candidate source.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SpeculativeMode {
    /// Disable speculative decoding.
    #[default]
    None,
    /// Propose tokens with the configured draft model.
    DraftModel,
    /// Copy continuations from the most recent matching context n-gram.
    PromptLookup {
        /// Number of trailing context tokens used as the lookup key.
        ngram: usize,
        /// Maximum copied continuation length per verification step.
        max_tokens: usize,
    },
    /// Propose from a target hidden state with an external MTP head.
    Mtp(MtpConfig),
    /// Propose autoregressively from fused low/middle/high target hidden states.
    Eagle3(Eagle3Config),
}

/// Identifier for a persistent generation session.
pub type SessionId = SequenceId;

/// Absolute logical token position within a persistent session.
///
/// Newtyping token positions keeps APIs from accepting an arbitrary `usize`
/// where a session boundary is required. Use [`SessionPosition::new`] at the
/// boundary where a caller intentionally converts from a raw count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SessionPosition(usize);

impl SessionPosition {
    /// Construct an absolute logical token position.
    pub const fn new(position: usize) -> Self {
        Self(position)
    }

    /// Raw zero-based token boundary.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Relative logical-token rewind distance for a persistent session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RewindTokenCount(usize);

impl RewindTokenCount {
    /// Construct a rewind distance.
    pub const fn new(tokens: usize) -> Self {
        Self(tokens)
    }

    /// Raw token count to rewind.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Opaque checkpoint for a persistent generation session.
///
/// The checkpoint records the logical token boundary that can later be passed to
/// [`Engine::restore_session`](crate::Engine::restore_session). It is only valid
/// for the session that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionCheckpoint {
    /// Session whose logical token stream was checkpointed.
    pub session_id: SessionId,
    /// Logical token position retained by the checkpoint.
    pub position: SessionPosition,
}

/// Capability token required to request a session fork.
///
/// Engines return this token only for decode configurations that can fork
/// without deep-copying KV or aliasing mutable decoder state. Current backends
/// return `None`, so unsupported engines cannot be asked to fork through the
/// typed API.
#[derive(Debug, Clone)]
pub struct SessionForkCapability {
    pub(crate) _private: (),
}

/// Distributed KV connector backend selection (DESIGN §38, K3).
///
/// Model-agnostic by construction: a backend carries only its own generic
/// settings, never per-model branches. `Null` is the default and reproduces the
/// engine's in-process-only prefix reuse exactly.
#[derive(Debug, Clone, Default)]
pub enum KvConnectorBackend {
    /// No external connector: KV lives only in the local paged cache.
    #[default]
    Null,
    /// Single-node tiered (GPU→CPU, optional disk) connector.
    LocalTiered(LocalTieredConfig),
}

/// Generic configuration for wiring a [`KvCacheConnector`](onnx_genai_kv::KvCacheConnector)
/// into the engine (DESIGN §38, K3).
///
/// Every field is a backend-neutral parameter. `model_id` only namespaces cache
/// keys (opaque; never interpreted); when `None` the engine derives a stable id
/// from the model directory.
#[derive(Debug, Clone)]
pub struct KvConnectorConfig {
    /// Which connector backend to use. Defaults to [`KvConnectorBackend::Null`].
    pub backend: KvConnectorBackend,
    /// Opaque model identity used to namespace cache keys. `None` => derived
    /// from the model directory name.
    pub model_id: Option<String>,
    /// Tokens per cached chunk for keying. `0` => [`DEFAULT_CHUNK_SIZE`].
    pub chunk_size: usize,
    /// Priority applied to chunks stored to the connector.
    pub store_priority: CachePriority,
    /// Estimated prefill recompute cost per token (ms), used as the
    /// fetch-vs-recompute baseline against a location's `estimated_load_ms`.
    pub recompute_ms_per_token: f64,
}

impl Default for KvConnectorConfig {
    fn default() -> Self {
        Self {
            backend: KvConnectorBackend::Null,
            model_id: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
            store_priority: CachePriority::Session,
            recompute_ms_per_token: 0.05,
        }
    }
}

/// Model-execution backend selected for decoder generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EngineDecodeBackend {
    /// Use the native runtime for models containing native-only operators;
    /// otherwise use ONNX Runtime.
    #[default]
    Auto,
    /// Always use ONNX Runtime.
    Ort,
    /// Always use the native runtime.
    Native,
}

/// User-selected static weight placement policy.
///
/// This is deliberately a loud parse surface. A misspelled `gpu_layers` setting
/// used to be accepted by nobody and ignored by everybody, which is the exact
/// "documented but unreachable" failure mode this configuration prevents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DevicePolicy {
    /// Let the engine use the governor-coordinated device weight budget.
    #[default]
    Auto,
    /// Keep planned layers on the host.
    Cpu,
    /// Compatibility override matching llama.cpp's `-ngl`: translate the layer
    /// prefix to bytes, then cap it by the governor's weight budget.
    GpuLayers(usize),
    /// Request a byte ceiling for static placement, still capped by the
    /// governor's coordinated weight budget.
    DeviceBytes(u64),
}

/// Error returned when `device_policy` cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid device_policy {input:?}: {reason}; use auto, cpu, gpu_layers:<N>, \
     or device_bytes:<SIZE>"
)]
pub struct DevicePolicyParseError {
    input: String,
    reason: String,
}

impl std::str::FromStr for DevicePolicy {
    type Err = DevicePolicyParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        parse_device_policy(input)
    }
}

/// Parse the `serving.memory.weights.device_policy` value.
pub fn parse_device_policy(input: &str) -> Result<DevicePolicy, DevicePolicyParseError> {
    let input = input.trim();
    if input.eq_ignore_ascii_case("auto") {
        return Ok(DevicePolicy::Auto);
    }
    if input.eq_ignore_ascii_case("cpu") {
        return Ok(DevicePolicy::Cpu);
    }
    if let Some(value) = input.strip_prefix("gpu_layers:") {
        let layers = value.trim().parse::<usize>().map_err(|_| {
            device_policy_error(
                input,
                "gpu_layers requires a non-negative integer layer count",
            )
        })?;
        return Ok(DevicePolicy::GpuLayers(layers));
    }
    if let Some(value) = input.strip_prefix("device_bytes:") {
        let bytes = match parse_resource_limit(value).map_err(|error| {
            device_policy_error(
                input,
                format!("device_bytes has invalid size syntax: {error}"),
            )
        })? {
            ResourceLimit::Bytes(bytes) => bytes,
            ResourceLimit::Auto => {
                return Err(device_policy_error(
                    input,
                    "device_bytes requires an explicit size, not auto",
                ));
            }
            ResourceLimit::Fraction(_) => {
                return Err(device_policy_error(
                    input,
                    "device_bytes requires a byte size, not a fraction",
                ));
            }
        };
        return Ok(DevicePolicy::DeviceBytes(bytes));
    }
    Err(device_policy_error(
        input,
        "the value does not match any supported policy",
    ))
}

fn device_policy_error(
    input: impl Into<String>,
    reason: impl Into<String>,
) -> DevicePolicyParseError {
    DevicePolicyParseError {
        input: input.into(),
        reason: reason.into(),
    }
}

/// Profile-facing summary of the static weight placement computed at load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPlacementReport {
    pub coordinated_weight_budget_bytes: u64,
    pub effective_budget_bytes: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub explanation: String,
}

/// Load-time memory strategy selected from graph/model evidence and overrides.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MemoryStrategyPlan {
    /// Effective strategy selected for the current runtime. Policy construction
    /// must consume this value rather than resolving the same inputs again.
    pub strategy: MemoryStrategy,
    /// Unconditionally inferred strategy before the current activation gate or
    /// an explicit override is applied.
    pub inferred_strategy: MemoryStrategy,
    pub weight_access_pattern: WeightAccessPattern,
    pub total_weight_bytes: u64,
    /// Resident side-buffer bytes folded into [`Self::total_weight_bytes`] when
    /// admitted: the dequantized-f32 decode cache (#971) and/or the int4
    /// `accuracy_level == 0` MLAS SQNBit packed buffer (#1027), both held for the
    /// session beside the on-disk weights. Zero on backends/models that take
    /// neither native CPU path.
    pub resident_f32_cache_bytes: u64,
    /// Whether the plan admitted the resident side buffers above. When `false`
    /// the runtime declined them (expanded footprint over budget): the f32 cache
    /// dequantizes on the fly and the MLAS int4 route falls back to the borrowed
    /// zero-copy path, so only the on-disk weights are held (#971, #1027). Always
    /// `true` when [`Self::resident_f32_cache_bytes`] is zero.
    pub f32_weight_cache_admitted: bool,
    pub kv_bytes_per_token: Option<u64>,
    pub per_layer_weight_bytes: Vec<LayerWeightBytes>,
    pub resolved_device_budget_bytes: Option<u64>,
    pub fits_resolved_device_budget: Option<bool>,
    pub application: MemoryPolicyApplication,
    /// True when the backend can report the plan but cannot safely enforce
    /// weight residency with its current provider lifecycle.
    pub advisory_only: bool,
    pub decisions: Vec<MemoryStrategyDecision>,
}

impl MemoryStrategyPlan {
    pub fn unknown(
        total_weight_bytes: u64,
        kv_bytes_per_token: Option<u64>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            strategy: MemoryStrategy::Unknown,
            inferred_strategy: MemoryStrategy::Unknown,
            weight_access_pattern: WeightAccessPattern::Unknown,
            total_weight_bytes,
            resident_f32_cache_bytes: 0,
            f32_weight_cache_admitted: true,
            kv_bytes_per_token,
            per_layer_weight_bytes: Vec::new(),
            resolved_device_budget_bytes: None,
            fits_resolved_device_budget: None,
            application: MemoryPolicyApplication::default(),
            advisory_only: true,
            decisions: vec![MemoryStrategyDecision::new(
                "strategy",
                "Unknown",
                DecisionSource::Unknown,
                reason,
                format!(
                    "total_weight_bytes={total_weight_bytes} kv_bytes_per_token={kv_bytes_per_token:?}"
                ),
            )],
        }
    }

    /// Concrete policy the runtime applies for this plan.
    ///
    /// The effective strategy remains authoritative: changing it changes
    /// whether paging is active instead of leaving behavior hidden in a
    /// separately resolved boolean.
    pub fn runtime_application(&self) -> MemoryPolicyApplication {
        let mut application = self.application.clone();
        application.weight_offload_enabled = match self.strategy {
            MemoryStrategy::FullResident => false,
            MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeRoutingAware => true,
            MemoryStrategy::Compatibility | MemoryStrategy::Unknown => {
                self.application.weight_offload_enabled
            }
        };
        application
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum MemoryStrategy {
    Compatibility,
    FullResident,
    DynamicWeightResidency,
    MoeRoutingAware,
    Unknown,
}

/// Concrete policy fields derived from [`MemoryStrategyPlan::strategy`].
///
/// Native CUDA provider construction consumes this object directly. Keeping it
/// in the serialized plan makes the applied behavior and its evidence the same
/// source of truth.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct MemoryPolicyApplication {
    pub weight_offload_enabled: bool,
    pub device_budget_bytes: Option<u64>,
    pub scan_resistant_dense: bool,
    pub managed_no_spill: bool,
    pub managed_limit_bytes: Option<u64>,
    pub device_budget_is_override: bool,
    pub auto_enabled_from_vram_limit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum WeightAccessPattern {
    SequentialDense,
    MoeRouted,
    Iterative,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LayerWeightBytes {
    pub layer_index: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MemoryStrategyDecision {
    pub field: &'static str,
    pub value: String,
    pub source: DecisionSource,
    /// Inferred value retained when an override or compatibility gate changes
    /// the effective value.
    pub inferred_value: Option<String>,
    pub reason: String,
    pub evidence: String,
}

impl MemoryStrategyDecision {
    pub fn new(
        field: &'static str,
        value: impl Into<String>,
        source: DecisionSource,
        reason: impl Into<String>,
        evidence: impl Into<String>,
    ) -> Self {
        Self {
            field,
            value: value.into(),
            source,
            inferred_value: None,
            reason: reason.into(),
            evidence: evidence.into(),
        }
    }

    pub fn with_inferred_value(mut self, inferred_value: impl Into<String>) -> Self {
        self.inferred_value = Some(inferred_value.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DecisionSource {
    Inference,
    ExplicitOverride,
    CompatibilityDefault,
    Unknown,
    /// The value could not be measured on this platform (e.g. no device-capacity
    /// query is available), so no number was resolved. Distinct from `Unknown`,
    /// which marks an inference the evidence could not decide (#947).
    Unavailable,
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Decoder execution backend. [`EngineDecodeBackend::Auto`] preserves ORT
    /// for existing models and selects native execution only when required.
    ///
    /// When this remains `Auto`, `ONNX_GENAI_BACKEND` may select `auto`, `ort`,
    /// or `native` instead. An explicit `Ort` or `Native` value always takes
    /// precedence over the environment variable.
    pub decode_backend: EngineDecodeBackend,
    /// Native decoder device override. `None` follows the execution provider in
    /// [`onnx_genai_ort::SessionOptions`], including `ONNX_GENAI_EP`.
    pub native_device: Option<crate::native_decode_device::NativeDecodeDevice>,
    /// Persistent native decode batch extent: how many sequences one fused
    /// forward advances (#750). `None` defers to
    /// `ONNX_GENAI_NATIVE_DECODE_BATCH`, which defaults to `1`.
    ///
    /// Setting it is what makes batch-N *requestable* rather than only
    /// environment-enabled. The server's `--max-batch` sets it, so a caller who
    /// asks for concurrent decoding either gets a session shaped for it or gets
    /// an error -- previously the request was refused because the capability was
    /// read from a session nobody had asked to build in batch shape (#1064).
    pub native_decode_batch: Option<usize>,
    /// Decoder-wide numeric precision for the native decode session
    /// (see [`onnx_runtime_session::DecodePrecision`]). Defaults to
    /// [`DecodePrecision::Model`](onnx_runtime_session::DecodePrecision::Model)
    /// (graph as authored); selecting
    /// [`Fp16`](onnx_runtime_session::DecodePrecision::Fp16) opts an
    /// fp32-activation int4/block-32 GPU decoder onto the fp16-fused kernels.
    /// A strict no-op for every other model, so the default path is unchanged.
    #[cfg(feature = "native-backend")]
    pub decode_precision: onnx_runtime_session::DecodePrecision,
    /// Tokens per KV page.
    pub page_size: usize,
    /// Scheduler config.
    pub scheduler: SchedulerConfig,
    /// Optional draft model directory used for greedy speculative decoding.
    pub draft_model: Option<PathBuf>,
    /// Number of draft tokens proposed per speculative step.
    pub num_speculative_tokens: usize,
    /// Default speculative source. For compatibility, a configured
    /// `draft_model` selects `DraftModel` when this remains `None`.
    pub speculative_mode: SpeculativeMode,
    /// Storage dtype for the host-side paged KV cache mirror.
    ///
    /// Controls how KV tensors are stored in the paged cache after being
    /// written from model outputs. The model's own I/O dtype (Float32 /
    /// Float16) is independent of this setting; the cache quantises/
    /// dequantises internally.  Defaults to `KvDType::F32` (no quantisation).
    pub kv_cache_dtype: KvDType,
    /// Optional distributed KV connector (DESIGN §38). Defaults to
    /// [`KvConnectorBackend::Null`], which preserves in-process-only behavior.
    pub kv_connector: KvConnectorConfig,
    /// Vendor-neutral hot, warm, and cold resource ceilings (DESIGN §26.11).
    pub limits: ResourceLimits,
    /// Permit programmatic resource-limit changes after engine initialization.
    pub allow_runtime_override: bool,
    /// Static device/host weight placement policy.
    pub device_policy: DevicePolicy,
    /// Byte budget for a pipeline's memoized prompt-phase (encoder) outputs.
    ///
    /// Re-asking about the same picture should not re-run the vision encoder,
    /// so its outputs are kept, keyed by the exact bytes that produced them.
    /// `0` disables the cache and every turn recomputes.
    pub pipeline_cache_bytes: u64,
    /// Maximum number of native session token histories to retain before the
    /// least-recently-used one is dropped. Defaults to 8. `0` disables the
    /// limit.
    ///
    /// This bounds retained *history*, not KV memory. Native sessions do not
    /// hold KV memory of their own: one KV cache exists and switching sessions
    /// resets it. Bounding KV bytes belongs to `EngineResourceGovernor` once
    /// native sessions hold real leases on the central KV manager, so that it
    /// applies to both backends rather than being a native-only knob.
    pub native_max_sessions: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            decode_backend: EngineDecodeBackend::Auto,
            native_device: None,
            native_decode_batch: None,
            #[cfg(feature = "native-backend")]
            decode_precision: onnx_runtime_session::DecodePrecision::Model,
            page_size: 16,
            scheduler: SchedulerConfig::default(),
            draft_model: None,
            num_speculative_tokens: 4,
            speculative_mode: SpeculativeMode::None,
            kv_cache_dtype: KvDType::F32,
            kv_connector: KvConnectorConfig::default(),
            limits: ResourceLimits::default(),
            allow_runtime_override: false,
            device_policy: DevicePolicy::Auto,
            // One encoder output for a handful of attachments. Big enough that
            // a conversation about a few images keeps all of them, small enough
            // to be an unremarkable line in a process's memory profile.
            pipeline_cache_bytes: 512 * 1024 * 1024,
            native_max_sessions: 8,
        }
    }
}

impl EngineConfig {
    /// Decode the `serving.memory.limits` YAML surface documented in §26.11.4.
    ///
    /// Engine settings outside that block retain their programmatic defaults.
    pub fn from_yaml(yaml: &str) -> Result<Self, EngineConfigError> {
        let document: EngineConfigYaml = serde_yaml::from_str(yaml)?;
        let yaml_limits = document.serving.memory.limits;
        let yaml_weights = document.serving.memory.weights;
        let mut config = Self::default();
        if let Some(limit) = yaml_limits.vram_limit {
            config.limits.vram_limit = limit.parse("vram_limit")?;
        }
        if let Some(limit) = yaml_limits.host_ram_limit {
            config.limits.host_ram_limit = limit.parse("host_ram_limit")?;
        }
        if let Some(limit) = yaml_limits.disk_spill_limit {
            config.limits.disk_spill_limit = Some(limit.parse("disk_spill_limit")?);
        }
        config.allow_runtime_override = yaml_limits.allow_runtime_override;
        if let Some(device_policy) = yaml_weights.device_policy {
            config.device_policy = parse_device_policy(&device_policy)
                .map_err(|source| EngineConfigError::DevicePolicy { source })?;
        }
        Ok(config)
    }
}

/// Prompt input accepted by Phase 1 generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratePrompt {
    /// Raw prompt text.
    Text(String),
    /// Already-tokenized prompt ids.
    TokenIds(Vec<TokenId>),
    /// Equal-length token rows for workflows with internal branches such as
    /// conditional/unconditional guidance. Native text decoders reject this form.
    TokenRows(Vec<Vec<TokenId>>),
}

impl From<String> for GeneratePrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for GeneratePrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn parses_integer_bytes_and_all_supported_units() {
        let cases = [
            ("42", 42),
            ("2KiB", 2 * 1024),
            ("2MiB", 2 * 1024 * 1024),
            ("2GiB", 2 * 1024 * 1024 * 1024),
            ("2KB", 2_000),
            ("2MB", 2_000_000),
            ("2GB", 2_000_000_000),
            ("1.5KiB", 1536),
        ];
        for (input, expected) in cases {
            assert_eq!(
                parse_resource_limit(input).unwrap(),
                ResourceLimit::Bytes(expected),
                "{input}"
            );
        }
    }

    #[test]
    fn parses_fraction_and_case_insensitive_auto() {
        assert_eq!(
            parse_resource_limit("0.5").unwrap(),
            ResourceLimit::Fraction(0.5)
        );
        assert_eq!(
            parse_resource_limit("1.0").unwrap(),
            ResourceLimit::Fraction(1.0)
        );
        assert_eq!(parse_resource_limit("AuTo").unwrap(), ResourceLimit::Auto);
    }

    #[test]
    fn rejects_out_of_range_fractions_unknown_units_and_invalid_numbers() {
        for input in ["1.01", "-0.1", "NaN", "inf"] {
            let error = parse_resource_limit(input).unwrap_err().to_string();
            assert!(error.contains("invalid resource limit"), "{error}");
            assert!(error.contains("use a byte count"), "{error}");
        }
        for input in ["8TiB", "8G", "8Gi", "8XB"] {
            let error = parse_resource_limit(input).unwrap_err().to_string();
            assert!(error.contains("not supported"), "{input}: {error}");
            assert!(error.contains("8GiB"), "{input}: {error}");
        }
        for input in ["GiB", "oneGiB", "-1GiB", "1e100GiB"] {
            let error = parse_resource_limit(input).unwrap_err().to_string();
            assert!(error.contains("invalid resource limit"), "{input}: {error}");
        }
    }

    #[test]
    fn rejects_integral_unit_overflow_at_exact_boundary() {
        let error = parse_resource_limit("17179869184GiB")
            .unwrap_err()
            .to_string();
        assert!(error.contains("overflows u64"), "{error}");
        assert!(error.contains("use a smaller byte quantity"), "{error}");
    }

    #[test]
    fn engine_config_defaults_to_scheduler_resource_defaults() {
        let config = EngineConfig::default();
        assert_eq!(config.decode_backend, EngineDecodeBackend::Auto);
        assert_eq!(config.limits, ResourceLimits::default());
        assert!(!config.allow_runtime_override);
    }

    #[test]
    fn yaml_limits_parse_fraction_bytes_auto_null_and_override() {
        let config = EngineConfig::from_yaml(
            r#"
    serving:
      memory:
        limits:
          vram_limit: "0.5"
          host_ram_limit: "8GiB"
          disk_spill_limit: "auto"
          allow_runtime_override: true
    "#,
        )
        .unwrap();
        assert_eq!(config.limits.vram_limit, ResourceLimit::Fraction(0.5));
        assert_eq!(
            config.limits.host_ram_limit,
            ResourceLimit::Bytes(8_u64 << 30)
        );
        assert_eq!(config.limits.disk_spill_limit, Some(ResourceLimit::Auto));
        assert!(config.allow_runtime_override);

        let disabled = EngineConfig::from_yaml(
            "serving:\n  memory:\n    limits:\n      disk_spill_limit: null\n",
        )
        .unwrap();
        assert_eq!(disabled.limits.disk_spill_limit, None);
    }

    #[test]
    fn yaml_device_policy_is_loud_and_reaches_config() {
        let config = EngineConfig::from_yaml(
            "serving:\n  memory:\n    weights:\n      device_policy: gpu_layers:2\n",
        )
        .unwrap();
        assert_eq!(config.device_policy, DevicePolicy::GpuLayers(2));

        let error = EngineConfig::from_yaml(
            "serving:\n  memory:\n    weights:\n      device_policy: gpu_layerz:2\n",
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("failed to parse serving.memory.weights.device_policy"),
            "{error}"
        );
    }

    #[test]
    fn yaml_accepts_numeric_fraction_and_reports_field_context() {
        for (value, expected) in [("1.0", 1.0), ("0.5", 0.5)] {
            let config = EngineConfig::from_yaml(&format!(
                "serving:\n  memory:\n    limits:\n      vram_limit: {value}\n"
            ))
            .unwrap();
            assert_eq!(config.limits.vram_limit, ResourceLimit::Fraction(expected));
        }

        let error =
            EngineConfig::from_yaml("serving:\n  memory:\n    limits:\n      vram_limit: 1.5\n")
                .unwrap_err()
                .to_string();
        assert!(error.contains("vram_limit"), "{error}");
        assert!(error.contains("outside [0, 1]"), "{error}");
    }
}

impl From<Vec<TokenId>> for GeneratePrompt {
    fn from(value: Vec<TokenId>) -> Self {
        Self::TokenIds(value)
    }
}

/// DRY (Don't Repeat Yourself) n-gram repetition controls.
#[derive(Debug, Clone, PartialEq)]
pub struct DryConfig {
    /// Penalty applied when a token would extend a repeated sequence.
    pub multiplier: f32,
    /// Exponential growth base for repetitions beyond `allowed_length`.
    pub base: f32,
    /// Repeated prefix length allowed before penalties begin.
    pub allowed_length: usize,
    /// Tokens that stop matching across semantic sequence boundaries.
    pub sequence_breakers: Vec<TokenId>,
}

/// Mirostat feedback algorithm version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirostatVersion {
    V1,
    V2,
}

/// Adaptive Mirostat surprise controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirostatConfig {
    /// Target surprise, in bits.
    pub tau: f32,
    /// Feedback learning rate.
    pub eta: f32,
    /// Mirostat algorithm version.
    pub version: MirostatVersion,
}

/// XTC (eXclude Top Choices) diversity controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XtcConfig {
    /// Probability of applying XTC on each sampling step.
    pub probability: f32,
    /// Minimum token probability considered a top choice.
    pub threshold: f32,
}

/// User-controllable decoding options for Phase 1 generation.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Maximum tokens to produce after the prompt.
    pub max_new_tokens: usize,
    /// Temperature applied before sampling. Zero forces greedy selection.
    pub temperature: f32,
    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    pub top_p: f32,
    /// Keep only the top-k logits before sampling. Zero disables top-k filtering.
    pub top_k: usize,
    /// Min-p sampling threshold. Zero disables min-p filtering.
    pub min_p: f32,
    /// Top-A sampling coefficient. Zero disables Top-A filtering.
    pub top_a: f32,
    /// Locally typical cumulative probability. One disables typical filtering.
    pub typical_p: f32,
    /// Repetition penalty applied to prompt and generated tokens. Values <= 1 disable it.
    pub repetition_penalty: f32,
    /// When `Some(n)`, the repetition penalty only considers the most recent `n`
    /// tokens of the combined prompt+generated stream. `None` uses the whole history.
    pub repetition_window: Option<usize>,
    /// OpenAI-style count penalty: logit[t] -= frequency_penalty * count(t).
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty: logit[t] -= presence_penalty once if seen.
    pub presence_penalty: f32,
    /// Optional DRY n-gram repetition penalty.
    pub dry: Option<DryConfig>,
    /// Optional adaptive Mirostat sampler.
    pub mirostat: Option<MirostatConfig>,
    /// Optional XTC top-choice exclusion.
    pub xtc: Option<XtcConfig>,
    /// If true, choose argmax after processors; otherwise sample categorically.
    pub greedy: bool,
    /// Optional seed for reproducible categorical sampling.
    pub seed: Option<u64>,
    /// Text or token sequences that terminate generation when matched as a suffix.
    pub stop_sequences: Vec<StopSequence>,
    /// A caller's optional single-id EOS override.
    ///
    /// A model may have several package-default end tokens, so callers that
    /// need a multi-id override use [`Self::eos_token_ids`].
    pub eos_token_id: Option<TokenId>,
    /// A caller's optional multi-id EOS override.
    ///
    /// When neither request field is set, the engine copies the package default
    /// from top-level `tokens.eos_token_id`. Tokenizer assets never contribute
    /// numeric ids.
    pub eos_token_ids: Vec<TokenId>,
    /// Whether an end token terminates generation.
    pub stop_on_eos: bool,
    /// Optional maximum total context length (prompt + generated tokens).
    /// Used when model metadata does not declare `model.max_sequence_length`.
    pub max_context: Option<usize>,
    /// Optional per-request override for speculative draft width K.
    pub num_speculative_tokens: Option<usize>,
    /// Optional per-request speculative mode override.
    pub speculative_mode: Option<SpeculativeMode>,
    /// Optional constrained decoding grammar. None preserves unconstrained generation.
    pub constraint: Option<GenerateConstraint>,
    /// Return per-token log probabilities and this many highest-probability alternatives.
    ///
    /// Values are computed from the final post-processor distribution used for sampling.
    /// The chosen token is always included in `TokenLogprob::top`, in addition to the
    /// requested alternatives when it is not already among them.
    pub top_logprobs: Option<usize>,
    /// Force a cold start (full KV reset) even when session-persistent KV reuse
    /// is available. Default is `false`, which allows the engine to reuse cached
    /// KV state for matching prompt prefixes. Set to `true` for A/B measurement
    /// or when guaranteed-cold behaviour is required.
    pub cold_start: bool,
    /// Token length of a declared semantic prefix boundary, such as the end of a
    /// system prompt. Native hybrid decoders may snapshot loop-carried state at
    /// this boundary and reuse it for later prompts that share the prefix.
    pub semantic_prefix_len: Option<usize>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            top_a: 0.0,
            typical_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry: None,
            mirostat: None,
            xtc: None,
            greedy: true,
            seed: None,
            stop_sequences: Vec::new(),
            eos_token_id: None,
            eos_token_ids: Vec::new(),
            stop_on_eos: true,
            max_context: None,
            num_speculative_tokens: None,
            speculative_mode: None,
            constraint: None,
            top_logprobs: None,
            cold_start: false,
            semantic_prefix_len: None,
        }
    }
}

/// Observable activity for semantic recurrent-prefix reuse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecurrentPrefixCacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub stores: u64,
    pub restored_tokens: u64,
}

/// Sampling controls a caller explicitly requested.
///
/// Each `None` means the caller did not specify that control, so
/// [`GenerateOptions::resolve_sampling_defaults`] falls back to the model's
/// author-declared defaults and then to the runtime fallback already held in
/// [`GenerateOptions`]. This type carries the "explicit flag wins" half of the
/// precedence contract; the model-declared half lives in
/// [`GenerationDefaults`](onnx_genai_metadata::GenerationDefaults).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverrides {
    /// Explicit greedy decision: `Some(true)` forces deterministic argmax,
    /// `Some(false)` forces stochastic sampling, `None` defers to the model's
    /// declared `do_sample` (then the runtime fallback).
    pub greedy: Option<bool>,
    /// Explicit sampling temperature.
    pub temperature: Option<f32>,
    /// Explicit nucleus (top-p) threshold.
    pub top_p: Option<f32>,
    /// Explicit top-k cutoff.
    pub top_k: Option<usize>,
}

impl GenerateOptions {
    /// Whether `token` ends generation.
    ///
    /// The single place that question is answered, so the fast path, the
    /// batched path and the speculative verifier cannot disagree about whether
    /// a model has finished. The engine resolves either the request override or
    /// the package default into these fields before execution.
    pub fn terminates(&self, token: TokenId) -> bool {
        self.stop_on_eos
            && (self.eos_token_ids.contains(&token) || self.eos_token_id == Some(token))
    }

    /// Whether token selection is greedy: the maximum logit wins, deterministically.
    ///
    /// `temperature == 0.0` means the same thing as `greedy` -- a zero-temperature
    /// softmax is a point mass on the maximum -- and callers use both spellings,
    /// so every decision that turns on "is this greedy" must ask here rather than
    /// re-spell the disjunction. Three places had spelled it out inline and one
    /// of them was already drifting.
    ///
    /// This is the predicate that decides whether the sampling warpers
    /// (temperature, top-k, top-p, min-p, top-a, typical-p, mirostat, XTC) are
    /// built at all: under greedy selection they cannot affect the outcome that
    /// the caller asked for, so applying them would be a silent contradiction of
    /// the request.
    pub fn selects_greedily(&self) -> bool {
        self.greedy || self.temperature == 0.0
    }

    /// Resolve sampling controls against a model author's declared generation
    /// defaults, applying a strict precedence.
    ///
    /// Highest priority first:
    /// 1. an explicit caller override in `overrides`;
    /// 2. the author's declared default in `declared` — the `do_sample`,
    ///    `temperature`, `top_p`, and `top_k` values a model publishes in
    ///    inference metadata (or a compatible `genai_config.json` `search` block);
    /// 3. the runtime fallback already stored in `self` (greedy).
    ///
    /// The runtime's hardcoded `greedy: true` is therefore used only when the
    /// caller is silent *and* the model declares nothing. It never overrides a
    /// value the model actually published — a reasoning model that ships
    /// `do_sample: true, temperature: 0.6` (precisely because greedy decoding
    /// makes it loop) now decodes stochastically by default. This keeps the
    /// runtime from baking in a decoding assumption the model contradicts
    /// (RULES.md rule 2).
    ///
    /// After the four controls are resolved, a *resolved* temperature of `0.0`
    /// forces `greedy = true`, regardless of where that zero came from (an
    /// explicit caller override, or a model that declares `do_sample: true`
    /// alongside `temperature: 0.0`). Temperature zero has no stochastic meaning
    /// — the sampler already collapses it to argmax — so the resolved
    /// `GenerateOptions` is made self-consistent here rather than leaving a
    /// `greedy: false, temperature: 0.0` state that reads as "sample at zero".
    /// Every consumer (not just the CLI) therefore inherits the
    /// `temperature 0 -> greedy` mapping by construction.
    pub fn resolve_sampling_defaults(
        &mut self,
        declared: Option<&GenerationDefaults>,
        overrides: &SamplingOverrides,
    ) {
        self.greedy = match overrides.greedy {
            Some(greedy) => greedy,
            None => match declared.and_then(|declared| declared.do_sample) {
                Some(do_sample) => !do_sample,
                None => self.greedy,
            },
        };
        if let Some(temperature) = overrides
            .temperature
            .or_else(|| declared.and_then(|declared| declared.temperature))
        {
            self.temperature = temperature;
        }
        if let Some(top_p) = overrides
            .top_p
            .or_else(|| declared.and_then(|declared| declared.top_p))
        {
            self.top_p = top_p;
        }
        if let Some(top_k) = overrides
            .top_k
            .or_else(|| declared.and_then(|declared| declared.top_k))
        {
            self.top_k = top_k;
        }
        // A resolved temperature of zero is greedy by definition; keep the flag
        // and the value consistent so the decision is inspectable rather than
        // implicit in the sampler (RULES.md rule 5).
        if self.temperature == 0.0 {
            self.greedy = true;
        }
    }

    /// Resolve sampling controls against a package's generation *contract*.
    ///
    /// The contract's defaults are authoritative. A caller may override only the
    /// fields the package structurally exposes as request-sourced workflow
    /// inputs, within the bounds those inputs declare. Every other override is
    /// rejected here rather than silently dropped: a request that asks for a
    /// temperature the package never wired to an input would otherwise decode at
    /// the package default while the caller believed their value took effect.
    ///
    /// A package with no generation contract at all has no declared override
    /// surface, so this behaves exactly like
    /// [`Self::resolve_sampling_defaults`] with no declared defaults — callers
    /// keep the runtime fallbacks, and nothing is silently discarded.
    pub fn resolve_generation_contract(
        &mut self,
        contract: Option<&GenerationContract>,
        overrides: &SamplingOverrides,
    ) -> anyhow::Result<()> {
        if let Some(contract) = contract {
            for (field, requested) in requested_overrides(overrides) {
                let Some(declared) = contract.overrides.get(field) else {
                    anyhow::bail!(
                        "generation override '{field}' is not supported by this package. \
                         Why: a package may only be overridden through fields it declares as \
                         request-sourced workflow inputs, and this one declares {}. How to fix: \
                         drop the override, or re-export the package with a workflow input for \
                         '{field}' listed under generation.overrides",
                        describe_supported(contract)
                    );
                };
                if let (Some(value), Some(constraint)) = (requested, &declared.constraint)
                    && (constraint.minimum.is_some_and(|minimum| value < minimum)
                        || constraint.maximum.is_some_and(|maximum| value > maximum))
                {
                    anyhow::bail!(
                        "generation override '{field}' = {value} is outside the range this \
                             package declares ({}..={}). Why: the request-sourced workflow input \
                             '{}' declares bounds the runtime enforces before execution. How to \
                             fix: choose a value inside the declared range",
                        constraint
                            .minimum
                            .map_or_else(|| "-inf".to_owned(), |bound| bound.to_string()),
                        constraint
                            .maximum
                            .map_or_else(|| "+inf".to_owned(), |bound| bound.to_string()),
                        declared.input,
                    );
                }
            }
        }
        self.resolve_sampling_defaults(
            contract.and_then(|contract| contract.defaults.as_ref()),
            overrides,
        );
        Ok(())
    }
}

/// The generation fields a caller actually asked to override, with the numeric
/// value when the field has one (`do_sample` is a flag, so it has none).
fn requested_overrides(overrides: &SamplingOverrides) -> Vec<(&'static str, Option<f64>)> {
    let mut requested = Vec::new();
    if overrides.greedy.is_some() {
        requested.push(("do_sample", None));
    }
    if let Some(temperature) = overrides.temperature {
        requested.push(("temperature", Some(f64::from(temperature))));
    }
    if let Some(top_p) = overrides.top_p {
        requested.push(("top_p", Some(f64::from(top_p))));
    }
    if let Some(top_k) = overrides.top_k {
        requested.push(("top_k", Some(top_k as f64)));
    }
    requested
}

fn describe_supported(contract: &GenerationContract) -> String {
    if contract.overrides.is_empty() {
        return "no overridable fields".to_owned();
    }
    format!(
        "only {}",
        contract
            .overrides
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Built-in constrained decoding grammars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateConstraint {
    /// Constrain output to one complete, well-formed JSON value.
    Json,
    /// Constrain output to a JSON value accepted by the provided JSON Schema.
    JsonSchema(String),
    /// Constrain output to text matching the provided Rust regular expression.
    Regex(String),
    /// Constrain output to the provided llguidance Lark grammar.
    Lark(String),
}

impl GenerateOptions {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.max_new_tokens == 0 {
            anyhow::bail!("max_new_tokens must be greater than zero");
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            anyhow::bail!("temperature must be finite and non-negative");
        }
        if !self.top_p.is_finite() || self.top_p < 0.0 {
            anyhow::bail!("top_p must be finite and non-negative");
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            anyhow::bail!("min_p must be finite and between 0 and 1");
        }
        if !self.top_a.is_finite() || !(0.0..=1.0).contains(&self.top_a) {
            anyhow::bail!("top_a must be finite and between 0 and 1");
        }
        if !self.typical_p.is_finite() || !(0.0..=1.0).contains(&self.typical_p) {
            anyhow::bail!("typical_p must be finite and between 0 and 1");
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            anyhow::bail!("repetition_penalty must be finite and greater than zero");
        }
        if !self.frequency_penalty.is_finite() {
            anyhow::bail!("frequency_penalty must be finite");
        }
        if !self.presence_penalty.is_finite() {
            anyhow::bail!("presence_penalty must be finite");
        }
        if let Some(dry) = &self.dry {
            if !dry.multiplier.is_finite() || dry.multiplier < 0.0 {
                anyhow::bail!("dry multiplier must be finite and non-negative");
            }
            if !dry.base.is_finite() || dry.base < 1.0 {
                anyhow::bail!("dry base must be finite and at least 1");
            }
            if dry.allowed_length == 0 {
                anyhow::bail!("dry allowed_length must be greater than zero");
            }
        }
        if let Some(mirostat) = self.mirostat {
            if !mirostat.tau.is_finite() || mirostat.tau <= 0.0 {
                anyhow::bail!("mirostat tau must be finite and greater than zero");
            }
            if !mirostat.eta.is_finite() || mirostat.eta <= 0.0 {
                anyhow::bail!("mirostat eta must be finite and greater than zero");
            }
        }
        if let Some(xtc) = self.xtc {
            if !xtc.probability.is_finite() || !(0.0..=1.0).contains(&xtc.probability) {
                anyhow::bail!("xtc probability must be finite and between 0 and 1");
            }
            if !xtc.threshold.is_finite() || !(0.0..=1.0).contains(&xtc.threshold) {
                anyhow::bail!("xtc threshold must be finite and between 0 and 1");
            }
        }
        if self.max_context == Some(0) {
            anyhow::bail!("max_context must be greater than zero when provided");
        }
        if self.num_speculative_tokens == Some(0) {
            anyhow::bail!("num_speculative_tokens must be greater than zero when provided");
        }
        if let Some(SpeculativeMode::PromptLookup { ngram, max_tokens }) = &self.speculative_mode {
            if *ngram == 0 {
                anyhow::bail!("prompt-lookup ngram must be greater than zero");
            }
            if *max_tokens == 0 {
                anyhow::bail!("prompt-lookup max_tokens must be greater than zero");
            }
        }
        if let Some(SpeculativeMode::Mtp(config)) = &self.speculative_mode {
            validate_mtp_config(config)?;
        }
        if let Some(SpeculativeMode::Eagle3(config)) = &self.speculative_mode {
            validate_eagle3_config(config)?;
        }
        Ok(())
    }
}

pub(crate) fn validate_mtp_config(config: &MtpConfig) -> anyhow::Result<()> {
    if config.head_model.as_os_str().is_empty() {
        anyhow::bail!("MTP head_model must not be empty");
    }
    if config.target_hidden_output.is_empty() {
        anyhow::bail!("MTP target_hidden_output must not be empty");
    }
    if config.embedding_weights.as_os_str().is_empty()
        || config.lm_head_weights.as_os_str().is_empty()
    {
        anyhow::bail!("MTP embedding_weights and lm_head_weights must not be empty");
    }
    if config.vocab_size == 0 || config.hidden_size == 0 {
        anyhow::bail!("MTP vocab_size and hidden_size must be greater than zero");
    }
    if config.num_speculative_tokens == 0 {
        anyhow::bail!("MTP num_speculative_tokens must be greater than zero");
    }
    Ok(())
}

pub(crate) fn validate_resolved_mtp_config(config: &ResolvedMtpConfig) -> anyhow::Result<()> {
    validate_mtp_config(&config.public_config)?;
    if config.mtp_hidden_output.is_empty() {
        anyhow::bail!("MTP mtp_hidden_output must not be empty");
    }
    if config
        .mtp_state_output
        .as_ref()
        .is_some_and(|name| name.is_empty())
    {
        anyhow::bail!("MTP mtp_state_output must not be empty when provided");
    }
    if config.hc_mult == 0 {
        anyhow::bail!("MTP hc_mult must be greater than zero");
    }
    if config.target_hidden_layout == MtpHiddenLayout::Bsh && config.hc_mult != 1 {
        anyhow::bail!("MTP BSH target_hidden_layout requires hc_mult == 1");
    }
    if config.hc_mult > 1 && config.mtp_state_output.is_none() {
        anyhow::bail!("MTP hc_mult > 1 requires mtp_state_output");
    }
    for (field, source) in [
        ("embedding_weights", &config.embedding_weights),
        ("lm_head_weights", &config.lm_head_weights),
    ] {
        let empty = match source {
            MtpWeightSource::File(path) => path.as_os_str().is_empty(),
            MtpWeightSource::TargetInitializer(name) => name.is_empty(),
        };
        if empty {
            anyhow::bail!("MTP {field} must not be empty");
        }
    }
    Ok(())
}

pub(crate) fn validate_eagle3_config(config: &Eagle3Config) -> anyhow::Result<()> {
    if config.target_hidden_outputs.len() != 3
        || config
            .target_hidden_outputs
            .iter()
            .any(|name| name.is_empty())
    {
        anyhow::bail!(
            "EAGLE-3 target_hidden_outputs must contain exactly three non-empty low/middle/high output names"
        );
    }
    if config.vocab_size == 0 || config.hidden_size == 0 {
        anyhow::bail!("EAGLE-3 vocab_size and hidden_size must be greater than zero");
    }
    if config.num_speculative_tokens == 0 {
        anyhow::bail!("EAGLE-3 num_speculative_tokens must be greater than zero");
    }
    Ok(())
}

#[cfg(test)]
mod mtp_config_tests {
    use super::*;

    /// A discovered sidecar spec must survive the hop into the engine's
    /// resolved config without losing a declared fact.
    ///
    /// The native loader no longer reads a metadata speculation block, so this
    /// mapping is the only place the sidecar's own declarations become
    /// executable settings: a silent drop here (a layout, an optional state
    /// output, or the KV lifetime) would surface much later as a head that
    /// refuses to run.
    #[cfg(feature = "native-backend")]
    #[test]
    fn a_discovered_sidecar_maps_onto_the_resolved_config_intact() {
        let spec = MtpProposerSpec {
            model: "/models/target/mtp/model.onnx".into(),
            num_speculative_tokens: 4,
            target_hidden_output: "hidden_states".into(),
            target_hidden_layout: MetadataMtpHiddenLayout::Bshc,
            target_hidden_size: 4096,
            hc_mult: 4,
            mtp_hidden_output: "mtp_hidden".into(),
            mtp_state_output: Some("mtp_state".into()),
            kv_mode: MetadataMtpKvMode::ProposalLocal,
            embedding_initializer: "model.embed_tokens.weight".into(),
            lm_head_initializer: "lm_head.weight".into(),
        };

        // The vocabulary is the target's, not the sidecar's, so it arrives from
        // the caller rather than from the spec.
        let config = ResolvedMtpConfig::from_sidecar_descriptor(&spec, 129_280);

        assert_eq!(
            config.public_config.head_model,
            std::path::Path::new("/models/target/mtp/model.onnx")
        );
        assert_eq!(config.public_config.target_hidden_output, "hidden_states");
        assert_eq!(config.target_hidden_layout, MtpHiddenLayout::Bshc);
        assert_eq!(
            config.embedding_weights,
            MtpWeightSource::TargetInitializer("model.embed_tokens.weight".into())
        );
        assert_eq!(
            config.lm_head_weights,
            MtpWeightSource::TargetInitializer("lm_head.weight".into())
        );
        assert_eq!(config.public_config.vocab_size, 129_280);
        assert_eq!(config.public_config.hidden_size, 4096);
        assert_eq!(config.hc_mult, 4);
        assert_eq!(config.mtp_hidden_output, "mtp_hidden");
        assert_eq!(config.mtp_state_output.as_deref(), Some("mtp_state"));
        assert_eq!(config.public_config.num_speculative_tokens, 4);
        assert_eq!(config.cache_scope, MtpCacheScope::ProposalLocal);
        validate_resolved_mtp_config(&config).expect("resolved config validates");
    }

    /// A pure-attention (proposal-local) head declares no recurrent state
    /// output. The `None` has to stay `None`: inventing a phantom `mtp_state`
    /// name would make the decode session demand an output the head does not
    /// expose.
    #[cfg(feature = "native-backend")]
    #[test]
    fn a_head_that_threads_no_state_keeps_its_absent_state_output() {
        let spec = MtpProposerSpec {
            model: "/models/target/mtp/model.onnx".into(),
            num_speculative_tokens: 1,
            target_hidden_output: "hidden_states.63".into(),
            target_hidden_layout: MetadataMtpHiddenLayout::Bsh,
            target_hidden_size: 5120,
            hc_mult: 1,
            mtp_hidden_output: "mtp_hidden".into(),
            mtp_state_output: None,
            kv_mode: MetadataMtpKvMode::ProposalLocal,
            embedding_initializer: "model.embed_tokens.weight".into(),
            lm_head_initializer: "lm_head.weight".into(),
        };

        let config = ResolvedMtpConfig::from_sidecar_descriptor(&spec, 248_320);

        assert_eq!(config.mtp_state_output, None);
        assert_eq!(config.target_hidden_layout, MtpHiddenLayout::Bsh);
        assert_eq!(config.hc_mult, 1);
        assert_eq!(config.cache_scope, MtpCacheScope::ProposalLocal);
        validate_resolved_mtp_config(&config).expect("resolved config validates");
    }

    /// `accepted_prefix` is declarable but not executable, so the lifetime must
    /// arrive intact at the loader that rejects it rather than being flattened
    /// into the proposal-local default here.
    #[cfg(feature = "native-backend")]
    #[test]
    fn an_accepted_prefix_lifetime_reaches_the_loader_that_refuses_it() {
        let spec = MtpProposerSpec {
            model: "/models/target/mtp/model.onnx".into(),
            num_speculative_tokens: 2,
            target_hidden_output: "hidden_states".into(),
            target_hidden_layout: MetadataMtpHiddenLayout::Bsh,
            target_hidden_size: 2048,
            hc_mult: 1,
            mtp_hidden_output: "mtp_hidden".into(),
            mtp_state_output: None,
            kv_mode: MetadataMtpKvMode::AcceptedPrefix,
            embedding_initializer: "model.embed_tokens.weight".into(),
            lm_head_initializer: "lm_head.weight".into(),
        };

        let config = ResolvedMtpConfig::from_sidecar_descriptor(&spec, 1_024);

        assert_eq!(config.cache_scope, MtpCacheScope::AcceptedPrefix);
    }

    #[test]
    fn mtp_validation_enforces_hc_layout_and_state_contract() {
        let mut config = ResolvedMtpConfig {
            public_config: MtpConfig {
                head_model: "mtp/model.onnx".into(),
                target_hidden_output: "hidden_states".into(),
                embedding_weights: "embedding.f32".into(),
                lm_head_weights: "lm_head.f32".into(),
                vocab_size: 32,
                hidden_size: 16,
                kv_mode: MtpDraftKvMode::HiddenThreaded,
                num_speculative_tokens: 4,
            },
            target_hidden_layout: MtpHiddenLayout::Bshc,
            embedding_weights: MtpWeightSource::File("embedding.f32".into()),
            lm_head_weights: MtpWeightSource::File("lm_head.f32".into()),
            hc_mult: 0,
            mtp_hidden_output: "mtp_hidden".into(),
            mtp_state_output: Some("mtp_state".into()),
            cache_scope: MtpCacheScope::ProposalLocal,
        };
        assert!(
            validate_resolved_mtp_config(&config)
                .unwrap_err()
                .to_string()
                .contains("hc_mult")
        );

        config.hc_mult = 2;
        config.mtp_state_output = None;
        assert!(
            validate_resolved_mtp_config(&config)
                .unwrap_err()
                .to_string()
                .contains("mtp_state_output")
        );

        config.target_hidden_layout = MtpHiddenLayout::Bsh;
        config.mtp_state_output = Some("mtp_state".into());
        assert!(
            validate_resolved_mtp_config(&config)
                .unwrap_err()
                .to_string()
                .contains("requires hc_mult == 1")
        );
    }
}

/// A single generation request.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt text or token ids.
    pub prompt: GeneratePrompt,
    /// Decoding options.
    pub options: GenerateOptions,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<GeneratePrompt>) -> Self {
        Self {
            prompt: prompt.into(),
            options: GenerateOptions::default(),
        }
    }
}

/// A generation request with an explicit scheduler priority.
#[derive(Debug, Clone)]
pub struct PrioritizedGenerateRequest {
    pub session_id: SessionId,
    pub request: GenerateRequest,
    pub priority: Priority,
}

/// A prioritized request that becomes visible to the engine after a decode-step count.
#[derive(Debug, Clone)]
pub struct ScheduledGenerateArrival {
    pub arrival_step: usize,
    pub request: PrioritizedGenerateRequest,
}

/// Result for one request driven through the priority scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct PrioritizedGenerateResult {
    pub session_id: SessionId,
    pub result: GenerateResult,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The configured maximum number of new tokens was reached.
    MaxTokens,
    /// The configured EOS token was generated.
    EosToken,
    /// A stop sequence matched; index refers to `GenerateOptions::stop_sequences`.
    StopSequence { index: usize },
    /// The model context window was reached before another decode step could run.
    Length,
}

/// Scheduler admission reduced the requested generation ceiling to preserve the
/// shared KV byte-budget guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationBudgetCap {
    pub requested_max_new_tokens: usize,
    pub admitted_max_new_tokens: usize,
    pub requested_bytes: u64,
    pub admitted_bytes: u64,
    pub available_bytes: u64,
}

/// Final generation output.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateResult {
    /// Detokenized generated text.
    pub text: String,
    /// Generated token ids, excluding prompt tokens.
    pub token_ids: Vec<TokenId>,
    /// Termination reason.
    pub finish_reason: FinishReason,
    /// Number of prompt/context tokens whose KV state was reused from the prefix cache.
    pub prefix_cache_hit_len: usize,
    /// Per-generated-token log probabilities, or `None` when not requested.
    pub logprobs: Option<Vec<TokenLogprob>>,
    /// Present when scheduler admission capped `max_new_tokens` below the
    /// requested value to keep the conservative KV byte reservation valid.
    pub budget_cap: Option<GenerationBudgetCap>,
}

/// Log-probability metadata for one generated token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The selected token id.
    pub token_id: TokenId,
    /// Natural-log probability of the selected token.
    pub logprob: f32,
    /// Highest-probability tokens and their natural-log probabilities, sorted descending.
    pub top: Vec<(TokenId, f32)>,
}

/// Per-token streaming event shape for future callback/iterator APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateToken {
    pub token_id: TokenId,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
}

/// Streaming callback shape. Returning an error aborts generation.
pub type GenerateTokenCallback<'a> = dyn FnMut(GenerateToken) -> anyhow::Result<()> + Send + 'a;

#[cfg(test)]
mod sampling_defaults_tests {
    use super::*;
    use onnx_genai_metadata::GenerationDefaults;

    fn declared(do_sample: Option<bool>, temperature: Option<f32>) -> GenerationDefaults {
        GenerationDefaults {
            do_sample,
            temperature,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            num_beams: None,
            num_return_sequences: None,
            min_length: None,
            max_length: None,
            length_penalty: None,
            no_repeat_ngram_size: None,
            diversity_penalty: None,
            early_stopping: None,
        }
    }

    // Row 1: model declares sampling, caller is silent -> the model's regime is used.
    #[test]
    fn model_sampling_used_when_no_flags() {
        let mut options = GenerateOptions::default();
        let model = GenerationDefaults {
            top_p: Some(0.95),
            top_k: Some(40),
            ..declared(Some(true), Some(0.6))
        };
        options.resolve_sampling_defaults(Some(&model), &SamplingOverrides::default());
        assert!(!options.greedy, "model do_sample=true must disable greedy");
        assert_eq!(options.temperature, 0.6);
        assert_eq!(options.top_p, 0.95);
        assert_eq!(options.top_k, 40);
    }

    // Row 2: explicit --greedy wins even when the model asks to sample.
    #[test]
    fn explicit_greedy_overrides_model_sampling() {
        let mut options = GenerateOptions::default();
        let model = declared(Some(true), Some(0.6));
        let overrides = SamplingOverrides {
            greedy: Some(true),
            ..SamplingOverrides::default()
        };
        options.resolve_sampling_defaults(Some(&model), &overrides);
        assert!(
            options.greedy,
            "explicit greedy must beat model do_sample=true"
        );
    }

    // Row 3: model declares nothing -> greedy fallback is preserved.
    #[test]
    fn greedy_fallback_when_model_declares_nothing() {
        let mut options = GenerateOptions::default();
        assert!(options.greedy);
        options.resolve_sampling_defaults(None, &SamplingOverrides::default());
        assert!(options.greedy, "no declaration and no flags stays greedy");
        // A declaration that omits do_sample is equally silent on the greedy question.
        let mut options = GenerateOptions::default();
        options
            .resolve_sampling_defaults(Some(&declared(None, None)), &SamplingOverrides::default());
        assert!(options.greedy);
    }

    // An explicit `--greedy` override is applied and the caller's explicit
    // temperature travels with it. This pins override precedence and value
    // pass-through; it does *not* exercise temperature-0 handling, because the
    // greedy result here is driven by `greedy: Some(true)`, not by the zero.
    // (The `temperature == 0 -> greedy` mapping is pinned separately below.)
    #[test]
    fn explicit_greedy_override_is_applied_and_keeps_its_temperature() {
        let mut options = GenerateOptions::default();
        let model = declared(Some(true), Some(0.6));
        let overrides = SamplingOverrides {
            greedy: Some(true),
            temperature: Some(0.0),
            ..SamplingOverrides::default()
        };
        options.resolve_sampling_defaults(Some(&model), &overrides);
        assert!(
            options.greedy,
            "explicit greedy=Some(true) must win over model do_sample=true"
        );
        assert_eq!(
            options.temperature, 0.0,
            "the caller's explicit temperature must be carried through"
        );
    }

    // The resolver itself maps a resolved `temperature == 0.0` to greedy, even
    // when the caller left `greedy` unspecified and the model asks to sample.
    // This is the property the old test name falsely claimed: here the greedy
    // result is driven purely by the zero temperature, not by an explicit
    // greedy flag. Answers "what does `temperature: Some(0.0)` without `greedy`
    // do?" — deterministic argmax, never stochastic sampling at zero.
    #[test]
    fn resolved_temperature_zero_forces_greedy_without_explicit_greedy() {
        let mut options = GenerateOptions::default();
        let model = declared(Some(true), Some(0.6));
        let overrides = SamplingOverrides {
            greedy: None,
            temperature: Some(0.0),
            ..SamplingOverrides::default()
        };
        options.resolve_sampling_defaults(Some(&model), &overrides);
        assert!(
            options.greedy,
            "temperature 0 must collapse to greedy even against model do_sample=true"
        );
        assert_eq!(options.temperature, 0.0);

        // A model that itself declares temperature 0 alongside do_sample=true is
        // likewise resolved to greedy, with no caller override at all.
        let mut options = GenerateOptions::default();
        let model = declared(Some(true), Some(0.0));
        options.resolve_sampling_defaults(Some(&model), &SamplingOverrides::default());
        assert!(
            options.greedy,
            "a model-declared temperature of 0 also collapses to greedy"
        );
    }

    // Explicit sampling flags win over a model that declares greedy, and the
    // caller's value is kept while unspecified controls fall back to the model.
    #[test]
    fn explicit_sampling_overrides_model_greedy_and_keeps_caller_values() {
        let mut options = GenerateOptions::default();
        let model = GenerationDefaults {
            top_p: Some(0.9),
            ..declared(Some(false), Some(0.3))
        };
        let overrides = SamplingOverrides {
            greedy: Some(false),
            temperature: Some(0.8),
            ..SamplingOverrides::default()
        };
        options.resolve_sampling_defaults(Some(&model), &overrides);
        assert!(
            !options.greedy,
            "explicit sampling must beat model do_sample=false"
        );
        assert_eq!(options.temperature, 0.8, "caller temperature wins");
        assert_eq!(
            options.top_p, 0.9,
            "unspecified top_p falls back to the model"
        );
    }

    // do_sample=false is honored as an explicit greedy declaration by the model.
    #[test]
    fn model_do_sample_false_selects_greedy() {
        let mut options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        options.resolve_sampling_defaults(
            Some(&declared(Some(false), None)),
            &SamplingOverrides::default(),
        );
        assert!(options.greedy, "model do_sample=false must select greedy");
    }
}

#[cfg(test)]
mod generation_contract_tests {
    use super::*;
    use onnx_genai_metadata::{GenerationOverride, GenerationOverrideConstraint};

    fn contract() -> GenerationContract {
        GenerationContract {
            defaults: Some(GenerationDefaults {
                do_sample: Some(true),
                temperature: Some(0.6),
                ..GenerationDefaults::default()
            }),
            overrides: [(
                "temperature".to_owned(),
                GenerationOverride {
                    input: "request.temperature".to_owned(),
                    constraint: Some(GenerationOverrideConstraint {
                        minimum: Some(0.0),
                        maximum: Some(2.0),
                    }),
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn a_declared_override_within_its_declared_bounds_is_applied() {
        let mut options = GenerateOptions::default();
        options
            .resolve_generation_contract(
                Some(&contract()),
                &SamplingOverrides {
                    temperature: Some(1.25),
                    ..SamplingOverrides::default()
                },
            )
            .expect("declared override");
        assert_eq!(options.temperature, 1.25);
        assert!(!options.greedy);
    }

    #[test]
    fn an_undeclared_override_fails_loudly_instead_of_being_dropped() {
        let mut options = GenerateOptions::default();
        let error = options
            .resolve_generation_contract(
                Some(&contract()),
                &SamplingOverrides {
                    top_k: Some(40),
                    ..SamplingOverrides::default()
                },
            )
            .expect_err("top_k is not wired to a request input");
        let message = error.to_string();
        assert!(message.contains("top_k"), "{message}");
        assert!(message.contains("only temperature"), "{message}");
    }

    #[test]
    fn an_override_outside_its_declared_range_fails_loudly() {
        let mut options = GenerateOptions::default();
        let error = options
            .resolve_generation_contract(
                Some(&contract()),
                &SamplingOverrides {
                    temperature: Some(9.0),
                    ..SamplingOverrides::default()
                },
            )
            .expect_err("9.0 exceeds the declared maximum");
        let message = error.to_string();
        assert!(message.contains("request.temperature"), "{message}");
        assert!(message.contains("0..=2"), "{message}");
    }

    #[test]
    fn package_defaults_are_authoritative_when_the_caller_is_silent() {
        let mut options = GenerateOptions::default();
        options
            .resolve_generation_contract(Some(&contract()), &SamplingOverrides::default())
            .expect("no override requested");
        assert_eq!(options.temperature, 0.6);
        assert!(
            !options.greedy,
            "declared do_sample: true must win over the runtime fallback"
        );
    }

    #[test]
    fn a_package_without_a_contract_keeps_the_runtime_fallbacks() {
        let mut options = GenerateOptions::default();
        options
            .resolve_generation_contract(
                None,
                &SamplingOverrides {
                    temperature: Some(0.9),
                    ..SamplingOverrides::default()
                },
            )
            .expect("no contract to violate");
        assert_eq!(options.temperature, 0.9);
    }
}
