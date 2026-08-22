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

/// Tokenizer facts and package-relative artifacts.
///
/// A request carries text, tokens, grammars, and JSON Schemas. Interpreting any
/// of them requires the vocabulary contract the package was built against.
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

    /// Package-relative tokenizer artifacts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(min = 1))]
    pub artifacts: Vec<TokenizerArtifact>,

    /// Special tokens by semantic role, e.g. `bos`, `eos`, `pad`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub special_tokens: BTreeMap<String, SpecialTokenFact>,
}

/// One package-relative tokenizer artifact.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizerArtifact {
    /// Package-relative path of the artifact.
    #[schemars(length(min = 1))]
    pub location: String,
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

    /// How a non-generative sequence output is decoded into discrete tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoding: Option<SequenceDecodingSpec>,

    /// Whether a row's outputs depend on the other rows batched with it.
    ///
    /// `row_independent` means a row produces identical values whether it is
    /// run alone or co-batched with rows of any other length, so a runtime may
    /// batch freely. `padding_sensitive` means padding a row to the batch width
    /// changes its values — for example when a normalization reduces over the
    /// padded time axis — so a runtime that batches trades accuracy for
    /// throughput and must not treat batched results as equal to solo results.
    ///
    /// Absent means unstated, not `row_independent`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::BatchInvariance>")]
    pub batch_invariance: Option<String>,

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

/// Frame-synchronous decoding contract for non-autoregressive sequence models.
///
/// A CTC acoustic model emits one class distribution per encoder frame. The
/// transcript is recovered by taking the per-frame argmax, collapsing runs of
/// repeated ids, and dropping the blank id — no generation loop and no
/// autoregressive state are involved.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SequenceDecodingSpec {
    /// Generic decoding algorithm selector (e.g. `ctc`).
    #[schemars(with = "schema_vocabulary::SequenceDecodingKind")]
    pub kind: String,

    /// Class id reserved for the CTC blank symbol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blank_id: Option<u32>,

    /// Whether runs of identical consecutive ids collapse to one id.
    #[serde(default)]
    pub collapse_repeats: bool,

    /// Axis of the logits tensor that enumerates frames.
    #[schemars(range(min = 0))]
    pub time_axis: usize,

    /// Axis of the logits tensor that enumerates classes.
    #[schemars(range(min = 0))]
    pub class_axis: usize,

    /// Profile output role naming the per-row count of valid frames.
    ///
    /// Present when the package is batched with padding: rows are decoded only
    /// over their own valid frame prefix so a padded batch produces the same
    /// transcript per row as an unpadded single-row run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lengths: Option<String>,

    /// Where the class-id -> string mapping comes from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocabulary: Option<DecodingVocabulary>,
}

/// Source of the class-id -> string mapping used to render a transcript.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecodingVocabulary {
    /// Generic source selector (e.g. `tokenizer`, `inline`).
    #[schemars(with = "schema_vocabulary::DecodingVocabularySource")]
    pub source: String,

    /// Number of classes in the decoding vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<usize>,

    /// Token string that separates words when rendering (e.g. `|`).
    ///
    /// A delimiter *separates* words, so it never contributes whitespace of its
    /// own: a reader splits the decoded token run on this token, discards empty
    /// groups, and joins the remaining groups with a single U+0020. Leading,
    /// trailing, and repeated delimiters therefore produce no empty words and no
    /// leading or trailing space. When absent, tokens are concatenated verbatim.
    ///
    /// Must be present in `tokens` when `source` is `inline`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub word_delimiter: Option<String>,

    /// Token strings dropped before rendering (e.g. `<pad>`, `<s>`).
    ///
    /// Removal happens after CTC collapsing and before word splitting, so an
    /// ignored token never joins or separates words.
    ///
    /// Every entry must be present in `tokens` when `source` is `inline`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored_tokens: Vec<String>,

    /// Inline class-id -> string table, ordered by class id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<String>,
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
    /// How the proposer materializes one candidate block.
    ///
    /// `block` components emit the complete proposal in one invocation.
    /// `chained` components emit one distribution and recurrence update per
    /// invocation; the runtime repeatedly invokes the same typed component up
    /// to `max_proposal_width`.
    #[serde(default)]
    pub proposal_execution: SpeculativeProposalExecution,
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

/// Execution shape of a speculative proposer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculativeProposalExecution {
    /// One proposer invocation returns the complete token block.
    #[default]
    Block,
    /// Repeated proposer invocations form an autoregressive proposal chain.
    Chained {
        /// Proposer input port receiving the previous selected token embedding.
        #[schemars(length(min = 1))]
        token_embedding_input: String,
        /// Proposer output port carrying the next-token distribution.
        #[schemars(length(min = 1))]
        logits_output: String,
        /// Loop-carried hidden/cache state updated by every proposer invocation
        /// through its own input port.
        #[serde(default)]
        recurrent: Vec<SpeculativeRecurrenceBinding>,
        /// A loop-carried activation the proposer emits but re-consumes WITHOUT
        /// a separate input port: it re-enters as the trailing segment of
        /// `token_embedding_input`. The proposer's fused input is
        /// `concat(token_embedding, carry)`, so the carry has no port and no
        /// workflow state cell of its own. Its first-step value is the target
        /// context named by `port_bindings.target_hidden_context`; each step
        /// replaces it with this output. Because a folded carry is recomputed
        /// from committed tokens on rejection rather than restored, it does not
        /// appear in `rollback_state`. A chained proposer declares at least one
        /// of `recurrent` or `folded_carry_output`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        folded_carry_output: Option<String>,
    },
}

/// One loop-carried proposer value in a chained proposal.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeRecurrenceBinding {
    /// Workflow state cell checkpointed before proposal and restored on rejection.
    #[schemars(length(min = 1))]
    pub state: String,
    /// Proposer input port receiving the current value.
    #[schemars(length(min = 1))]
    pub input: String,
    /// Proposer output port producing the next value.
    #[schemars(length(min = 1))]
    pub output: String,
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
