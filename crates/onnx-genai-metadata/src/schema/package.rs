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
    /// Exact tokenizer and vocabulary facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerFacts>,

    /// Constraint/grammar dialects this package's parser can interpret.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraint_languages: Vec<ConstraintLanguageFacts>,

    /// Exact, versioned tool-call protocol used to render caller-owned tools
    /// and parse model-produced envelopes.  Absence means this package does
    /// not support tool calls; it is intentionally not a boolean capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_protocol: Option<ToolProtocolDeclaration>,
}

/// A portable tool-call protocol identity.  Implementations select only this
/// exact pair and never infer a protocol from a model name or emitted text.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolProtocolDeclaration {
    /// Stable protocol identity, independent of model family or runtime.
    #[schemars(length(min = 1))]
    pub identity: String,
    /// Exact protocol version, including its envelope and streaming semantics.
    #[schemars(length(min = 1))]
    pub version: String,
}

/// Tokenizer facts and package-relative artifacts.
///
/// A request carries text, tokens, grammars, and JSON Schemas. Interpreting any
/// of them requires the vocabulary contract the package was built against.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizerFacts {
    /// Tokenizer algorithm identifier, e.g. `bpe`, `unigram`, `wordpiece`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub algorithm: Option<String>,

    /// Number of entries in the vocabulary, including added tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub vocab_size: Option<usize>,

    /// Whether the tokenizer operates on raw bytes rather than Unicode scalars.
    #[serde(default)]
    pub byte_level: bool,

    /// Package-relative tokenizer artifacts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(min = 1))]
    pub artifacts: Vec<TokenizerArtifact>,

    /// Numeric model and control-token facts for this tokenizer vocabulary.
    ///
    /// Token strings, added-token mappings, and chat templates remain in the
    /// tokenizer assets. Request EOS inputs may override these defaults, but
    /// workflow literals and termination components do not own another copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub special_tokens: Option<TokenFacts>,
}

/// One package-relative tokenizer artifact.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizerArtifact {
    /// Package-relative path of the artifact.
    #[schemars(length(min = 1))]
    pub location: String,
}

/// Numeric model and control-token facts.
///
/// These ids are model/package facts. Token spellings, added-token maps, and
/// chat templates remain in tokenizer assets and are not repeated here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenFacts {
    /// Padding token used by package-authored tensor contracts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_token_id: Option<u32>,
    /// Beginning-of-sequence token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u32>,
    /// Every token id that terminates package-default autoregressive generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eos_token_id: Vec<u32>,
    /// Separator token used by sequence-pair models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sep_token_id: Option<u32>,
    /// First token fed to an encoder-decoder's autoregressive decoder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_start_token_id: Option<u32>,
    /// Prompt placeholder replaced by image features.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_token_id: Option<u32>,
    /// Prompt placeholder replaced by video features.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_token_id: Option<u32>,
    /// Prompt placeholder replaced by audio features.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_token_id: Option<u32>,
    /// Token that opens a vision segment in a multimodal prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_start_token_id: Option<u32>,
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
    /// Required when the logits output pads `time_axis`. The role must map to
    /// the exact workflow output named by that dimension's
    /// `padding.valid_lengths`; CTC then decodes only that valid prefix. An
    /// unpadded time axis needs no lengths binding.
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
/// This is the only portable authority for a speculative step.  In particular,
/// a runtime must not supplement it from a HuggingFace sidecar, a filename, a
/// model family, or an inferred graph convention. Proposal width, scheduling,
/// batching, kernels, and enablement remain runtime decisions.
// Not `Eq`: a chained proposer's declared embedding normalizer is a real number.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeContract {
    /// Exact built-in contract identity. Unknown identities are not a
    /// best-effort proposal format; they are an unsupported execution contract.
    #[schemars(length(min = 1))]
    pub identity: String,
    /// Exact identity version understood by the runtime.
    #[schemars(length(min = 1))]
    pub version: String,
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
    /// Proposer input ports bound to target-owned values, by semantic role.
    ///
    /// The keys are protocol roles and the values are declared proposer ports;
    /// neither side is recovered from a name or tensor shape at runtime.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub port_bindings: BTreeMap<String, String>,
    /// Target ports bound for verification, by protocol role.
    ///
    /// This is intentionally separate from [`Self::port_bindings`]: a target
    /// port and a proposer port may have the same spelling while carrying
    /// unrelated values.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub target_port_bindings: BTreeMap<String, String>,
    /// State groups the proposer shares with the target.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub shared_state: BTreeSet<String>,
    /// Immutable target initializers the proposer borrows.
    ///
    /// The component and initializer are both required. A bare string used to
    /// force a loader to guess which target artifact owned an initializer.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub shared_weights: BTreeSet<SpeculativeInitializerRef>,
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
    /// Exact values the verifier produces and the declared accepted-path
    /// publication binding. Sampling is legal only when `probabilities` is
    /// present; this is not inferred from `distribution_preserving`.
    pub verification: SpeculativeVerification,
    /// State groups that must roll back when a proposal is rejected.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub rollback_state: BTreeSet<String>,
}

impl SpeculativeContract {
    /// Admit distribution-preserving sampling only when both distributions were
    /// declared. Greedy verification is intentionally a separate structural
    /// path and needs no probability tensor.
    pub fn admit_sampling(&self) -> Result<(), String> {
        if self.verification.probabilities.is_none() {
            return Err("speculative sampling was requested, but \
                 speculative.verification.probabilities is absent. Declare both \
                 proposal and target probability outputs for \
                 distribution-preserving rejection sampling, or use greedy verification."
                .to_string());
        }
        if !self.distribution_preserving {
            return Err(
                "speculative sampling was requested, but this contract does not claim \
                 distribution_preserving. Use greedy verification or re-export a proposer \
                 with exact proposal/target probability correction."
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// A component-owned immutable initializer used by a speculative proposer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeInitializerRef {
    /// Component whose ONNX artifact owns the initializer.
    #[schemars(length(min = 1))]
    pub component: String,
    /// Exact ONNX initializer name.
    #[schemars(length(min = 1))]
    pub initializer: String,
}

/// Explicit verifier outputs for a speculative proposal.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeVerification {
    /// Target output used by the declared acceptance rule.
    pub target_output: SpeculativeValueRef,
    /// Explicit accepted-path output binding.
    ///
    /// A runtime-owned acceptance implementation names a stable binding such as
    /// `accepted_prefix`; a graph-owned implementation names the component
    /// output that provides it. Either way the binding is declared, rather than
    /// selected from an output name or emission order.
    pub accepted_path: SpeculativeAcceptedPath,
    /// Proposal and target probability outputs used for exact rejection
    /// sampling. Omission means this contract is greedy-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<SpeculativeProbabilityOutputs>,
}

/// The authority that produces the accepted candidate path.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculativeAcceptedPath {
    /// The registered speculative executor publishes the accepted path under a
    /// protocol binding. It remains an explicit output of the contract.
    Runtime {
        #[schemars(length(min = 1))]
        binding: String,
    },
    /// A workflow component produces the accepted path.
    Component { value: SpeculativeValueRef },
}

/// Probability outputs needed for distribution-preserving sampling.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeProbabilityOutputs {
    /// Per-candidate probabilities emitted by the proposer.
    pub proposal: SpeculativeValueRef,
    /// Target probabilities for the corresponding candidate positions.
    pub target: SpeculativeValueRef,
}

/// Execution shape of a speculative proposer.
// Not `Eq`: a chained proposer's declared embedding normalizer is a real number.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculativeProposalExecution {
    /// One proposer invocation returns the complete token block.
    #[default]
    Block,
    /// One DFlash invocation predicts every masked position of a flat block in
    /// parallel from explicitly bound target hidden features.
    ///
    /// Version 1 is the base DFlash architecture. Version 2 adds the DFlash 2
    /// selector and block-local convolution through [`DFlashStructure`];
    /// neither version is inferred from optional ports or tensor names.
    DflashFlatBlock {
        /// Exact structural ABI version implemented by the reader.
        #[schemars(length(min = 1))]
        version: String,
        /// Target hidden features and how they become one proposer input.
        conditioning: Box<DFlashConditioning>,
        /// Anchor/masked-position layout and the proposer inputs carrying it.
        block: Box<DFlashBlockLayout>,
        /// Proposer and verifier outputs consumed by the generic runtime.
        outputs: Box<DFlashOutputs>,
        /// Immutable target initializers reused by the proposer.
        shared_weights: Box<DFlashSharedWeights>,
        /// Mutable state cells owned only by the drafter.
        ///
        /// Every cell is also present in the enclosing contract's
        /// `rollback_state`; the separate set distinguishes private draft
        /// state from target state without creating another rollback authority.
        #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
        draft_private_state: BTreeSet<String>,
        /// How every rollback participant commits exactly one accepted prefix.
        ///
        /// The keys must equal the enclosing contract's `rollback_state`.
        /// Sequence state is truncated on its declared state-group sequence
        /// axis. Fixed recurrent state selects an explicitly emitted
        /// per-prefix snapshot; a runtime never guesses from a port name.
        accepted_prefix_state: Box<BTreeMap<String, DFlashStateCommit>>,
        /// Base DFlash or an exact versioned structural extension.
        structure: Box<DFlashStructure>,
    },
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
        /// Proposer OUTPUT port producing the folded carry for the NEXT step
        /// (carry_k, k>=1). The carry has no dedicated input port: it re-enters
        /// as the trailing segment of the proposer's fused
        /// `token_embedding_input` (`concat(embed(last_token), carry)`), so it
        /// owns no workflow state cell. Three ports pin the fold EXPLICITLY, so
        /// a runtime never infers by convention:
        ///
        /// * DESTINATION: `port_bindings.target_hidden_context` names the
        ///   proposer input port the carry lands in. For a folded carry it must
        ///   equal `token_embedding_input`: the carry occupies the fused input's
        ///   trailing half, so the destination is that fused input, never a
        ///   separate port.
        /// * carry_0 SOURCE: `folded_carry_seed` names the target output read
        ///   as the carry on the first step.
        /// * carry_k SOURCE: this field, the proposer output on every later
        ///   step.
        ///
        /// Because a folded carry is recomputed from committed tokens on
        /// rejection rather than restored, it does not appear in
        /// `rollback_state`. A chained proposer declares at least one of
        /// `recurrent` or `folded_carry_output`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        folded_carry_output: Option<String>,
        /// carry_0 seed: the target component OUTPUT read as the folded carry on
        /// the FIRST step, before the proposer has produced a carry. Named
        /// explicitly (`component` + `output`) so a runtime reads it rather than
        /// inferring "the target hidden output" by convention. Its `component`
        /// must be the speculative target, since carry_0 is the target's own
        /// per-token hidden output. Required whenever `folded_carry_output` is
        /// present.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        folded_carry_seed: Option<SpeculativeValueRef>,
        /// Where a runtime obtains `embed(last_token)` for the LEADING half of
        /// the fused `token_embedding_input`. A folded-carry proposer graph
        /// consumes only the fused input, so it reads no embedding initializer of
        /// its own and `speculative.shared_weights` stays empty; this names the
        /// model-agnostic embedding table the runtime gathers the leading half
        /// from (never extracted heuristically from a graph). Its `component`
        /// must be the speculative target — an ONNX model that owns the named
        /// `table` initializer — so the table resolves to a real initializer in
        /// the target model/artifact. Required whenever `folded_carry_output` is
        /// present.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_embedding: Option<TokenEmbeddingSource>,
    },
    /// A multi-token-prediction head conditioned on an explicitly named target
    /// hidden-state output. Its head inputs, outputs, shared weights, and
    /// private-state lifetime are all declared here; it is not selected by a
    /// model family or a legacy sidecar.
    Mtp {
        /// Target output that seeds the first MTP step.
        target_hidden: SpeculativeValueRef,
        /// MTP-head input receiving `target_hidden`.
        #[schemars(length(min = 1))]
        target_hidden_input: String,
        /// MTP-head input receiving the selected token embedding.
        #[schemars(length(min = 1))]
        token_embedding_input: String,
        /// MTP-head output projected through the declared shared LM head.
        #[schemars(length(min = 1))]
        hidden_output: String,
        /// Shape of the target hidden state and MTP-head hidden-state input.
        /// `bsh` is `[batch, sequence, hidden]`; `bshc` additionally carries
        /// `hc_mult` before the feature axis.
        #[serde(default)]
        hidden_layout: MtpHiddenStateLayout,
        /// Feature width of the declared hidden-state binding.
        #[schemars(range(min = 1))]
        hidden_size: usize,
        /// Number of lanes in a `bshc` hidden state. The value is one for
        /// `bsh`; declaring it prevents a runtime from guessing a lane axis.
        #[serde(default = "one")]
        #[schemars(range(min = 1))]
        hc_mult: usize,
        /// Optional MTP-head replacement-state output.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state_output: Option<String>,
        /// The target embedding and LM-head initializers MTP shares. Their
        /// component ownership is explicit so no loader searches artifacts.
        weights: MtpSharedWeights,
        /// MTP private-state behavior between proposal blocks.
        #[serde(default)]
        state: MtpProposalState,
    },
    /// A generic branching candidate tree.
    ///
    /// The tree shape is proposal data: a runtime may pick a width and schedule
    /// it, but it must receive the candidate ids and exactly one declared
    /// topology representation from this proposer.
    CandidateTree {
        /// Proposer output carrying flattened candidate token IDs.
        #[schemars(length(min = 1))]
        candidate_tokens: String,
        /// Parent-pointer or ancestor-mask topology over `candidate_tokens`.
        topology: CandidateTreeTopology,
    },
}

fn one() -> usize {
    1
}

/// Layout of the target activation consumed by an MTP head.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MtpHiddenStateLayout {
    /// `[batch, sequence, hidden]`.
    #[default]
    Bsh,
    /// `[batch, sequence, hc_mult, hidden]`.
    Bshc,
}

/// Exact immutable weight sharing used by an MTP head.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MtpSharedWeights {
    pub embedding: SpeculativeInitializerRef,
    pub lm_head: SpeculativeInitializerRef,
}

/// Whether MTP private state is reset or retained after a proposal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MtpProposalState {
    /// The head has no state that survives a proposal. It is reset on every
    /// outcome and therefore cannot become a hidden transaction participant.
    #[default]
    ProposalLocal,
    /// The head state is retained only through the accepted prefix. Every
    /// named cell must be a rollback participant.
    AcceptedPrefix {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        recurrent: Vec<SpeculativeRecurrenceBinding>,
    },
}

/// One of the two non-interchangeable candidate-tree topology encodings.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateTreeTopology {
    /// One parent index per flattened candidate (`-1`/a declared root sentinel
    /// is represented by the producer's tensor contract).
    ParentIndices {
        #[schemars(length(min = 1))]
        output: String,
    },
    /// Boolean ancestor matrix over flattened candidates.
    AncestorMask {
        #[schemars(length(min = 1))]
        output: String,
    },
}

/// Explicit target-feature conditioning for a DFlash proposer.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashConditioning {
    /// Target component outputs, in concatenation order.
    ///
    /// A source must be a floating rank-3 `[batch, sequence, hidden]` output
    /// with semantic role `hidden_states`. Layer selection is therefore
    /// producer-authored provenance, not a runtime model-family lookup.
    #[schemars(length(min = 1))]
    pub sources: Vec<SpeculativeValueRef>,
    /// Proposer input receiving the fused target features.
    #[schemars(length(min = 1))]
    pub proposer_input: String,
    /// Structural operation used to combine multiple sources.
    pub combination: DFlashFeatureCombination,
}

/// How target hidden features are fused before DFlash consumes them.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DFlashFeatureCombination {
    /// Concatenate sources in declaration order along `axis`.
    Concatenate { axis: usize },
}

/// Flat anchor-plus-mask block presented to one DFlash invocation.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashBlockLayout {
    /// Proposer input receiving target-embedding rows for the anchor followed
    /// by the masked candidate positions.
    #[schemars(length(min = 1))]
    pub noise_embeddings_input: String,
    /// Proposer input receiving a boolean tensor with `true` exactly at masked
    /// candidate positions and `false` at the anchor.
    #[schemars(length(min = 1))]
    pub masked_positions_input: String,
    /// Proposer input receiving absolute positions for target context followed
    /// by the flat block.
    #[schemars(length(min = 1))]
    pub position_ids_input: String,
    /// Proposer input receiving validity for target context followed by the
    /// flat block.
    #[schemars(length(min = 1))]
    pub attention_mask_input: String,
    /// Position holding the verifier-produced anchor token.
    pub anchor_position: usize,
    /// First position predicted in parallel. Validation requires this to be
    /// exactly one past `anchor_position`.
    pub first_candidate_position: usize,
    /// Token whose shared target embedding initializes masked positions.
    pub mask_token_id: u32,
}

/// Typed outputs of DFlash proposal and verification.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashOutputs {
    /// Proposer output containing selected candidate token ids
    /// `[batch, proposal]`.
    #[schemars(length(min = 1))]
    pub candidate_tokens: String,
    /// Full-vocabulary proposal probabilities `[batch, proposal, vocabulary]`.
    ///
    /// Absence is valid for greedy-only execution. A sampling request must be
    /// rejected before invoking the proposer when this field is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_probabilities: Option<String>,
    /// Target output containing exact verification logits
    /// `[batch, proposal_plus_bonus, vocabulary]`.
    pub verifier_logits: SpeculativeValueRef,
}

/// Immutable target weights DFlash reuses.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashSharedWeights {
    /// Target token embedding table used for anchor and mask rows.
    pub input_embedding: TokenEmbeddingSource,
    /// Target output projection passed read-only to the proposer.
    pub output_projection: DFlashOutputProjection,
}

/// Target LM-head initializer and the proposer input that borrows it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashOutputProjection {
    /// Component owning the immutable initializer.
    #[schemars(length(min = 1))]
    pub component: String,
    /// Exact initializer name.
    #[schemars(length(min = 1))]
    pub initializer: String,
    /// Proposer input receiving the initializer as a read-only tensor.
    #[schemars(length(min = 1))]
    pub proposer_input: String,
    /// Matrix layout required by the proposer graph.
    pub layout: DFlashProjectionLayout,
}

/// Layout of the shared output projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DFlashProjectionLayout {
    HiddenVocabulary,
    VocabularyHidden,
}

/// How one speculative state cell is reduced to the accepted prefix.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DFlashStateCommit {
    /// The component's ordinary state output grows on its declared sequence
    /// axis and is truncated to `baseline + accepted`.
    Sequence { source: SpeculativeValueRef },
    /// The component emits one complete fixed-state snapshot for each possible
    /// accepted length, including prefix zero.
    PrefixSnapshots {
        source: SpeculativeValueRef,
        /// Axis enumerating prefix lengths `0..=proposal`.
        axis: usize,
    },
}

/// Exact DFlash structural family.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DFlashStructure {
    /// Base DFlash: parallel masked-block prediction with target KV injection.
    Base,
    /// DFlash 2: base DFlash plus an explicit adjacent-candidate selector and
    /// block-local dynamic convolution.
    SelectorConvolutionV1 {
        selector: DFlashSelectorContract,
        convolution: DFlashConvolutionContract,
    },
}

/// DFlash 2 adjacent-candidate selector ABI.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashSelectorContract {
    /// Proposer output containing the selected path `[batch, proposal]`.
    #[schemars(length(min = 1))]
    pub selected_tokens_output: String,
    /// Proposer output containing top-k candidate ids
    /// `[batch, proposal, candidates]`.
    #[schemars(length(min = 1))]
    pub candidate_ids_output: String,
    /// Proposer output containing the selected conditional distribution over
    /// `candidate_ids_output`, required for sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditional_probabilities_output: Option<String>,
    #[schemars(range(min = 1))]
    pub top_k: usize,
    #[schemars(range(min = 1))]
    pub rank: usize,
}

/// DFlash 2 block-local grouped dynamic convolution ABI.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DFlashConvolutionContract {
    #[schemars(range(min = 2))]
    pub kernel_size: usize,
    #[schemars(range(min = 1))]
    pub group_size: usize,
    /// The first candidate reads the anchor representation as its predecessor.
    pub first_position_reads_anchor: bool,
}

/// An explicit reference to a value a workflow component produces.
///
/// Both halves are named so a speculative runtime resolves the value from the
/// declared graph I/O, never by string convention or shape guessing.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeValueRef {
    /// The workflow component that produces the value.
    #[schemars(length(min = 1))]
    pub component: String,
    /// The declared output port of that component carrying the value.
    #[schemars(length(min = 1))]
    pub output: String,
}

/// Explicit, model-agnostic source of the token embedding a chained proposer
/// folds into the leading half of its fused input.
///
/// A folded-carry proposer never reads an embedding initializer inside its own
/// graph, so the table cannot be recovered from the proposer graph. This names
/// the component whose embedding the runtime reuses and the initializer that
/// holds it, so gathering `embed(last_token)` is a declared contract rather than
/// a per-model heuristic.
// Not `Eq`: `scale` is a real number, and a package that declares 39.19 is not
// meaningfully "equal" to one that declares 39.190000000000005.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenEmbeddingSource {
    /// The workflow component whose token-embedding table is reused. For a
    /// folded carry this must be the speculative target, whose vocabulary the
    /// proposer shares and whose ONNX model owns the `table` initializer.
    #[schemars(length(min = 1))]
    pub component: String,
    /// The named embedding table (graph initializer) on that component, e.g.
    /// `model.embed_tokens.weight`. A `[vocab, hidden]` row-major matrix. It
    /// must name a real initializer in the target model/artifact.
    #[schemars(length(min = 1))]
    pub table: String,
    /// Normalizer the target applies to a looked-up row before its backbone
    /// reads it.
    ///
    /// A gathered row is not always the tensor a graph feeds its first block:
    /// several architectures scale the embedding by a constant folded into the
    /// graph (`sqrt(hidden_size)` is the common one), so the initializer alone
    /// is not what a proposer's fused input needs. The proposer stands in for
    /// the target's own embedding step, so it must see the *scaled* row — and
    /// the factor has to be declared rather than guessed: nothing in a
    /// `[vocab, hidden]` initializer says whether one was applied, and a
    /// proposer fed the unscaled row drafts fluent, plausible, uniformly
    /// rejected tokens. It is the acceptance rate that collapses, not the
    /// output, so the failure is invisible to a token-parity check.
    ///
    /// Absent means 1.0: a package whose graph gathers and feeds directly.
    ///
    /// Single precision because that is the precision it is *used* in: the
    /// factor multiplies a float16 or float32 table, so declaring more would
    /// promise an accuracy the application of it discards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
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
