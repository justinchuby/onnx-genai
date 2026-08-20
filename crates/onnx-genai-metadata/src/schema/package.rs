//! Package-level facts, task profiles, sharding, and speculative contracts.
//!
//! These sections carry the portable facts a caller needs to interpret a
//! request correctly. They never carry deployment policy: budgets, placement,
//! execution providers, transfers, cache identity, and trust remain runtime and
//! distribution concerns.

use super::*;
use std::collections::BTreeSet;

/// Exact package facts required to interpret request data correctly.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PackageFacts {
    /// Exact tokenizer, vocabulary, and special-token facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerFacts>,

    /// Constraint/grammar dialects this package's parser can interpret.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraint_languages: Vec<ConstraintLanguageFacts>,
}

/// Byte-exact tokenizer facts.
///
/// A request carries text, tokens, grammars, and JSON Schemas. Interpreting any
/// of them requires the exact vocabulary the package was built against, so the
/// artifact bytes are pinned rather than described by a family name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizerFacts {
    /// Tokenizer algorithm identifier, e.g. `bpe`, `unigram`, `wordpiece`.
    #[schemars(length(min = 1))]
    pub algorithm: String,

    /// Number of entries in the vocabulary, including added tokens.
    #[schemars(range(min = 1))]
    pub vocab_size: usize,

    /// Whether the tokenizer operates on raw bytes rather than Unicode scalars.
    #[serde(default)]
    pub byte_level: bool,

    /// Package-relative tokenizer artifacts pinned by exact content hash.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(min = 1))]
    pub artifacts: Vec<TokenizerArtifact>,

    /// Special tokens by semantic role, e.g. `bos`, `eos`, `pad`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub special_tokens: BTreeMap<String, SpecialTokenFact>,
}

/// One tokenizer artifact pinned to its exact bytes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizerArtifact {
    /// Package-relative path of the artifact.
    #[schemars(length(min = 1))]
    pub location: String,
    /// Lowercase SHA-256 of the exact artifact bytes.
    #[schemars(length(min = 64, max = 64))]
    pub sha256: String,
}

/// One special token, pinned by id and exact surface bytes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpecialTokenFact {
    /// Vocabulary id of the token.
    pub id: u32,
    /// Exact UTF-8 surface form of the token.
    #[schemars(length(min = 1))]
    pub content: String,
}

/// A constraint language the package's parser accepts.
///
/// The parser implementation may be native; only the dialect and version are
/// portable facts. A request carries the grammar or JSON Schema itself.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConstraintLanguageFacts {
    /// Dialect identifier, e.g. `json_schema`, `ebnf`, `regex`.
    #[schemars(length(min = 1))]
    pub dialect: String,
    /// Exact dialect version, e.g. `2020-12` for JSON Schema.
    #[schemars(length(min = 1))]
    pub version: String,
    /// Workflow component that interprets this dialect.
    #[schemars(length(min = 1))]
    pub component: String,
}

/// Authoritative generation defaults and the structural override surface.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerationContract {
    /// Authoritative package defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<GenerationDefaults>,

    /// Overridable fields, each bound to a request-sourced workflow input.
    ///
    /// A caller may override exactly these fields and no others. An override of
    /// any unlisted field must fail loudly rather than being silently dropped.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overrides: BTreeMap<String, GenerationOverride>,
}

/// One caller-overridable generation field.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerationOverride {
    /// Request-sourced workflow input that carries the override value.
    #[schemars(length(min = 1))]
    pub input: String,
    /// Declared bounds the runtime enforces before executing the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint: Option<GenerationOverrideConstraint>,
}

/// Declared bounds on one overridable generation field.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerationOverrideConstraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
}

/// One executable task profile that shares the package's common facts.
///
/// Generative and non-generative tasks live in one document. Each profile
/// carries its own version and a requirement class so a strict reader can skip
/// an optional profile it does not understand while still rejecting unknown
/// core fields.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskProfile {
    /// Task kind identifier, e.g. `generation`, `embedding`, `reranking`.
    #[schemars(length(min = 1))]
    pub kind: String,

    /// Version of this profile's own contract.
    #[schemars(length(min = 1))]
    pub version: String,

    /// Whether a reader that does not understand this profile may skip it.
    #[serde(default)]
    pub requirement: ProfileRequirement,

    /// Workflow outputs this profile consumes, by semantic role.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub outputs: BTreeMap<String, String>,

    /// Pooling applied to a sequence-valued output, when the task needs it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooling: Option<PoolingSpec>,

    /// Whether this profile changes generated output and therefore participates
    /// in cache correctness dependencies.
    #[serde(default)]
    pub generation_affecting: bool,
}

/// Whether a reader may ignore a profile it does not understand.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProfileRequirement {
    /// A reader that cannot execute this profile must refuse to load the package.
    #[default]
    Required,
    /// A reader that does not understand this profile may skip it.
    Ignorable,
}

/// Pooling reduction applied to a sequence-valued task output.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PoolingSpec {
    pub kind: PoolingKind,
    /// Axis reduced by the pooling operation.
    pub axis: usize,
    /// Whether the pooled vector is L2-normalized.
    #[serde(default)]
    pub normalize: bool,
}

/// Pooling reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PoolingKind {
    Mean,
    Max,
    Cls,
    LastToken,
}

/// Legal sharding and replication facts for distributed execution.
///
/// The caller and runtime choose degree, device mapping, placement, and
/// collective backend. Metadata only declares what is legal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShardingContract {
    /// Legal tensor-parallel shard axes by logical parameter group.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tensor_parallel: BTreeMap<String, TensorShardFacts>,

    /// Legal pipeline-parallel stage boundaries and their cross-stage state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pipeline_parallel: Vec<PipelineStageFacts>,

    /// Legal expert-parallel facts for sparse mixture-of-experts layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_parallel: Option<ExpertShardFacts>,
}

/// Shard axis and replication facts of one logical parameter group.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TensorShardFacts {
    /// Axis of the parameter that may be split across ranks.
    pub shard_axis: usize,
    /// Largest legal shard count; the caller may choose any divisor.
    #[schemars(range(min = 1))]
    pub max_shards: usize,
    /// Values that must be replicated identically on every rank.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub replicated: BTreeSet<String>,
}

/// One legal pipeline stage boundary.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineStageFacts {
    /// Workflow components executed by this stage, in order.
    #[schemars(length(min = 1))]
    pub components: Vec<String>,
    /// Typed values crossing into the next stage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<String>,
    /// State groups this stage owns.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub state_groups: BTreeSet<String>,
}

/// Expert identity and routing facts for expert parallelism.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExpertShardFacts {
    /// Total number of experts per sparse layer.
    #[schemars(range(min = 1))]
    pub expert_count: usize,
    /// Whether experts may be split across ranks in arbitrary contiguous groups.
    #[serde(default)]
    pub contiguous_groups_only: bool,
    /// Values replicated on every expert-parallel rank, such as the router.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub replicated: BTreeSet<String>,
}

/// Portable compatibility facts for speculative decoding.
///
/// Proposal width, tree shape, scheduling, kernels, and whether speculation is
/// enabled at all are runtime decisions.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeContract {
    /// Workflow component that proposes tokens.
    #[schemars(length(min = 1))]
    pub proposer: String,
    /// Workflow component that verifies proposals.
    #[schemars(length(min = 1))]
    pub target: String,
    /// Proposer ports bound to target-owned values, by semantic role.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub port_bindings: BTreeMap<String, String>,
    /// State groups the proposer shares with the target.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub shared_state: BTreeSet<String>,
    /// Target initializers the proposer borrows.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub shared_weights: BTreeSet<String>,
    /// How the proposer's vocabulary relates to the target's.
    pub vocabulary: SpeculativeVocabulary,
    /// Maximum number of proposed positions this package can undo.
    ///
    /// This is a rollback bound, not a choice of proposal width: the runtime
    /// picks any K up to this bound. A validator rejects a package whose state
    /// or speculative effects cannot be undone this far.
    #[schemars(range(min = 1))]
    pub max_proposal_width: usize,
    /// Whether accepting proposals preserves the target's output distribution.
    ///
    /// A runtime may auto-enable speculation only when this is true; otherwise
    /// the caller must opt in explicitly.
    #[serde(default)]
    pub distribution_preserving: bool,
    /// State groups that must roll back when a proposal is rejected.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub rollback_state: BTreeSet<String>,
}

/// Vocabulary relationship between a proposer and its target.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculativeVocabulary {
    /// Proposer and target share one identical vocabulary.
    Identical,
    /// The proposer's vocabulary is a prefix-compatible subset of the target's.
    Subset { proposer_vocab_size: usize },
    /// A declared mapping artifact translates proposer ids into target ids.
    Mapped { artifact: String },
}
