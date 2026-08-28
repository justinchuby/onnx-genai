//! Load inference metadata from YAML or JSON files.

use crate::schema::{
    InferenceMetadata, MtpHiddenLayout, MtpKvMode, PipelineSpec, ProposalType, SpeculatorConfig,
};
use std::path::{Path, PathBuf};

/// Source used to discover a speculator declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculatorConfigSource {
    HuggingFaceConfig,
}

/// Proposer implementation that will eventually back a detected speculator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculatorProposerKind {
    Eagle3,
    PEagle,
    Mtp,
    DFlash,
}

/// Resolved Mobius MTP sidecar descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpProposerSpec {
    /// Absolute path to the MTP sidecar ONNX model.
    pub model: PathBuf,
    /// Number of speculative tokens after the guaranteed target token.
    pub num_speculative_tokens: usize,
    /// Target decoder output carrying the recurrent MTP state.
    pub target_hidden_output: String,
    /// Target hidden-state layout.
    pub target_hidden_layout: MtpHiddenLayout,
    /// Target hidden width `H`.
    pub target_hidden_size: usize,
    /// Hyper-Connection multiplier `C`.
    pub hc_mult: usize,
    /// Sidecar output consumed by the shared target LM head.
    pub mtp_hidden_output: String,
    /// Sidecar recurrent HC-state output, if the head threads one. A
    /// pure-attention (proposal-local) head declares none.
    pub mtp_state_output: Option<String>,
    /// Sidecar KV lifetime.
    pub kv_mode: MtpKvMode,
    /// Exact target embedding initializer name.
    pub embedding_initializer: String,
    /// Exact target LM-head initializer name.
    pub lm_head_initializer: String,
}

/// Current construction status for the engine-facing proposer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculatorProposerStatus {
    /// A fully resolved Mobius MTP sidecar.
    Mtp(MtpProposerSpec),
    NotYetSupported(SpeculatorProposerKind),
    Unknown(String),
}

/// Resolved speculator declaration for a model directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculatorDescriptor {
    pub model_dir: PathBuf,
    pub proposal_type: ProposalType,
    pub num_speculative_tokens: usize,
    pub verifier: Option<crate::schema::SpeculatorVerifier>,
    pub source: SpeculatorConfigSource,
    pub proposer: SpeculatorProposerStatus,
}

impl SpeculatorDescriptor {
    fn from_config(
        model_dir: &Path,
        config: SpeculatorConfig,
        source: SpeculatorConfigSource,
    ) -> Self {
        let proposer = match &config.proposal_type {
            ProposalType::Eagle3 => {
                SpeculatorProposerStatus::NotYetSupported(SpeculatorProposerKind::Eagle3)
            }
            ProposalType::PEagle => {
                SpeculatorProposerStatus::NotYetSupported(SpeculatorProposerKind::PEagle)
            }
            ProposalType::Mtp => Self::resolve_mtp(model_dir, &config),
            ProposalType::DFlash => SpeculatorProposerStatus::Unknown(
                "legacy `proposal_type: dflash` is not an executable authority. Re-export one \
                 canonical `speculative.proposal_execution: { kind: dflash_flat_block, version: \
                 \"1\" | \"2\", ... }` contract with explicit target-hidden provenance, block \
                 layout, probabilities, shared initializers, verifier outputs, and accepted-prefix \
                 state participants"
                    .into(),
            ),
            // A legacy `shared_kv` speculator no longer selects a runtime path.
            // Borrowed-KV drafting is declared by the package's
            // `speculative.proposal_execution: {kind: chained}` contract and
            // driven by the workflow interpreter, so a descriptor that only says
            // `shared_kv` names no executable proposer — say so instead of
            // resolving one nothing can run.
            ProposalType::SharedKv => SpeculatorProposerStatus::Unknown(
                "shared_kv speculators are declared by the package's \
                 `speculative.proposal_execution: {kind: chained}` workflow contract, not by a \
                 `proposal_type: shared_kv` block"
                    .into(),
            ),
            ProposalType::Unknown(value) => SpeculatorProposerStatus::Unknown(value.clone()),
        };

        Self {
            model_dir: model_dir.to_path_buf(),
            proposal_type: config.proposal_type,
            num_speculative_tokens: config.num_speculative_tokens,
            verifier: config.verifier,
            source,
            proposer,
        }
    }

    fn resolve_mtp(model_dir: &Path, config: &SpeculatorConfig) -> SpeculatorProposerStatus {
        let missing = |field: &str| {
            SpeculatorProposerStatus::Unknown(format!("mtp metadata is missing `{field}`"))
        };
        let Some(model) = config.model.as_ref().filter(|value| !value.is_empty()) else {
            return missing("model");
        };
        let Some(target_hidden_size) = config.target_hidden_size.filter(|&value| value > 0) else {
            return missing("target_hidden_size");
        };
        let Some(hc_mult) = config.hc_mult.filter(|&value| value > 0) else {
            return missing("hc_mult");
        };
        let Some(embedding) = config.embedding.as_ref() else {
            return missing("embedding");
        };
        if embedding.name.is_empty() {
            return SpeculatorProposerStatus::Unknown(
                "mtp metadata `embedding.name` must not be empty".into(),
            );
        }
        let Some(lm_head) = config.lm_head.as_ref() else {
            return missing("lm_head");
        };
        if lm_head.name.is_empty() {
            return SpeculatorProposerStatus::Unknown(
                "mtp metadata `lm_head.name` must not be empty".into(),
            );
        }
        if config.num_speculative_tokens == 0 {
            return SpeculatorProposerStatus::Unknown(
                "mtp metadata `num_speculative_tokens` must be greater than zero".into(),
            );
        }

        SpeculatorProposerStatus::Mtp(MtpProposerSpec {
            model: model_dir.join(model),
            num_speculative_tokens: config.num_speculative_tokens,
            target_hidden_output: config
                .target_hidden_output
                .clone()
                .unwrap_or_else(|| "hidden_states".into()),
            target_hidden_layout: config.target_hidden_layout.unwrap_or(MtpHiddenLayout::Bshc),
            target_hidden_size,
            hc_mult,
            mtp_hidden_output: config
                .mtp_hidden_output
                .clone()
                .unwrap_or_else(|| "mtp_hidden".into()),
            mtp_state_output: config.mtp_state_output.clone(),
            kv_mode: config.kv_mode.unwrap_or(MtpKvMode::ProposalLocal),
            embedding_initializer: embedding.name.clone(),
            lm_head_initializer: lm_head.name.clone(),
        })
    }
}

/// The inference-metadata sidecar in `model_dir`, if the package ships one.
///
/// Which filenames count as inference metadata, and which wins when a package
/// ships more than one, is a property of the format rather than of any one
/// loader. Every caller asks here so a package cannot be read as having
/// metadata by one loader and as having none by another.
pub fn find_metadata_path(model_dir: &Path) -> Option<PathBuf> {
    METADATA_FILE_NAMES
        .iter()
        .map(|name| model_dir.join(name))
        .find(|path| path.is_file())
}

/// Load the inference-metadata sidecar from `model_dir`, if there is one.
///
/// `Ok(None)` means the package ships no metadata. A metadata file that exists
/// but cannot be read is an error rather than a `None`: silently treating a
/// malformed sidecar as an absent one is how a model comes to run with every
/// declared setting -- context length, chunked prefill, EOS ids, sampling
/// defaults -- quietly ignored.
pub fn load_metadata_from_dir(
    model_dir: &Path,
) -> Result<Option<InferenceMetadata>, crate::MetadataError> {
    find_metadata_path(model_dir)
        .map(|path| load_metadata(&path))
        .transpose()
}

/// Recognized inference-metadata filenames, in the order they are preferred.
const METADATA_FILE_NAMES: [&str; 3] = [
    "inference_metadata.yaml",
    "inference_metadata.yml",
    "inference_metadata.json",
];

/// Load inference metadata from a file (YAML or JSON based on extension).
pub fn load_metadata(path: &Path) -> Result<InferenceMetadata, crate::MetadataError> {
    let content = std::fs::read_to_string(path).map_err(crate::MetadataError::Io)?;
    parse_metadata(&content, path.extension().and_then(|e| e.to_str()))
}

/// The one way a document becomes an [`InferenceMetadata`].
///
/// Every loader goes through here rather than calling `serde` itself, because
/// migrations have to happen before a typed reader sees the bytes. Retired
/// shapes and batching hints are recognized so a package gets an actionable
/// conversion instead of an unknown-field error. The schema version is compared
/// first for migrations whose meaning is scoped to the v1 vocabulary, because every
/// structure in this schema denies unknown fields: a document from a newer
/// runtime is not malformed, and reporting the first field this build happens
/// not to know would send a reader hunting for a typo that is not there.
///
/// `extension` is a hint, not a requirement — `None` tries YAML and then JSON,
/// which is what a document with no filename gets.
pub fn parse_metadata(
    content: &str,
    extension: Option<&str>,
) -> Result<InferenceMetadata, crate::MetadataError> {
    preparse(content)?;
    match extension {
        Some("yaml" | "yml") => {
            serde_yaml::from_str(content).map_err(|e| crate::MetadataError::Parse(e.to_string()))
        }
        Some("json") => {
            serde_json::from_str(content).map_err(|e| crate::MetadataError::Parse(e.to_string()))
        }
        _ => {
            // YAML is a superset of JSON, but its error for a JSON document that
            // is wrong in a JSON way is worse, so fall back rather than insist.
            if let Ok(metadata) = serde_yaml::from_str::<InferenceMetadata>(content) {
                Ok(metadata)
            } else {
                serde_json::from_str::<InferenceMetadata>(content)
                    .map_err(|e| crate::MetadataError::Parse(e.to_string()))
            }
        }
    }
}

/// Like [`parse_metadata`], for a document a caller already holds as JSON.
///
/// A lowering that builds a document in memory gets the same two checks a file
/// gets. It would otherwise be the one path on which a package could declare a
/// version nothing verified.
pub fn parse_metadata_json(
    document: &serde_json::Value,
) -> Result<InferenceMetadata, crate::MetadataError> {
    let value = serde_yaml::to_value(document)
        .map_err(|error| crate::MetadataError::Parse(error.to_string()))?;
    gate_document(&value)?;
    serde_json::from_value(document.clone())
        .map_err(|error| crate::MetadataError::Parse(error.to_string()))
}

/// Checks that read a document as a tree, before anything reads it as a type.
fn preparse(content: &str) -> Result<(), crate::MetadataError> {
    let Ok(document) = serde_yaml::from_str::<serde_yaml::Value>(content) else {
        // Not parseable as a tree at all; the typed reader reports why.
        return Ok(());
    };
    gate_document(&document)
}

fn gate_document(document: &serde_yaml::Value) -> Result<(), crate::MetadataError> {
    reject_retired_model_io(document)?;
    reject_retired_top_level_tokens(document)?;
    reject_invalid_tensor_contract_shapes(document, String::new())?;
    let declared = crate::version::declared_in(document).map_err(crate::MetadataError::Parse)?;
    let version = crate::version::gate(declared).map_err(crate::MetadataError::Parse)?;
    require_output_families(document, version)?;
    reject_retired_streaming_emit(document)?;
    // After the version, deliberately. The flat packed spelling is a reshape
    // within the v1 line, so refusing it presumes the document belongs to that
    // line. A document from a version this build does not support is refused for
    // being from that version, and this crate does not tell its author what a
    // spelling it has never read is supposed to mean.
    //
    // It also cannot become part of the gate, which is the tempting
    // simplification to reach for later. A stale document here is well-formed
    // and declares a version this reader supports; both spellings belong to the
    // same line, so no comparison of version numbers can tell them apart. A
    // retired spelling is only ever found by recognizing its shape.
    reject_flat_token_packed(document, String::new())
        .and_then(|_| reject_retired_batching_hints(document))
}

/// Output protocols are a v1.5 addition. Older packages retain their original
/// materialized-value interpretation, while a v1.5 package must make the
/// family explicit so an older reader cannot accidentally choose one.
fn require_output_families(
    document: &serde_yaml::Value,
    version: crate::version::SchemaVersion,
) -> Result<(), crate::MetadataError> {
    if version < crate::version::OUTPUT_PROTOCOL_SCHEMA_VERSION {
        return Ok(());
    }
    let Some(outputs) = document
        .get("pipeline")
        .and_then(|pipeline| pipeline.get("workflow"))
        .and_then(|workflow| workflow.get("outputs"))
        .and_then(serde_yaml::Value::as_mapping)
    else {
        return Ok(());
    };
    for (name, output) in outputs {
        let name = name.as_str().unwrap_or("?");
        if output.get("family").is_none() {
            return Err(crate::MetadataError::Parse(format!(
                "pipeline.workflow.outputs.{name} is missing required `family` in schema \
                 version {version}; declare exactly one of `{{ kind: materialized }}`, \
                 `{{ kind: events }}`, or `{{ kind: revisions, version: \"1\" }}`"
            )));
        }
    }
    Ok(())
}

/// `streaming_emit` was a redundant capability spelling: the output family
/// alone decides whether a publication is an event or a revision. Refuse it
/// before typed parsing so a producer gets a migration rather than a generic
/// unsupported-capability error.
fn reject_retired_streaming_emit(document: &serde_yaml::Value) -> Result<(), crate::MetadataError> {
    fn walk(value: &serde_yaml::Value, path: String) -> Option<String> {
        match value {
            serde_yaml::Value::Mapping(mapping) => {
                for (key, child) in mapping {
                    let key = key.as_str().unwrap_or("?");
                    let child_path = if path.is_empty() {
                        key.to_string()
                    } else {
                        format!("{path}.{key}")
                    };
                    if key == "capabilities"
                        && child.as_sequence().is_some_and(|items| {
                            items
                                .iter()
                                .any(|item| item.as_str() == Some("streaming_emit"))
                        })
                    {
                        return Some(child_path);
                    }
                    if let Some(found) = walk(child, child_path) {
                        return Some(found);
                    }
                }
                None
            }
            serde_yaml::Value::Sequence(items) => items
                .iter()
                .enumerate()
                .find_map(|(index, child)| walk(child, format!("{path}[{index}]"))),
            _ => None,
        }
    }

    let Some(path) = walk(document, String::new()) else {
        return Ok(());
    };
    Err(crate::MetadataError::Parse(format!(
        "`{path}` declares retired capability `streaming_emit`. Remove it and select the \
         workflow output's canonical `family`: `events` for ordered occurrences or \
         `revisions` with an exact protocol version for replaceable output. `typed_emit` \
         remains the capability for workflow output publication."
    )))
}

/// Refuse tensor contracts that retain the duplicate rank authority or omit
/// their sole shape authority.
///
/// This is a diagnostic recognizer, not a compatibility reader: it never
/// rewrites the document. Typed deserialization would reject both forms, but it
/// cannot reliably name the nested tensor/port path in YAML.
fn reject_invalid_tensor_contract_shapes(
    document: &serde_yaml::Value,
    path: String,
) -> Result<(), crate::MetadataError> {
    match document {
        serde_yaml::Value::Mapping(mapping) => {
            let has_dtype = mapping.contains_key(serde_yaml::Value::from("dtype"));
            let contract_fields = ["dtype", "shape", "optional", "batch_layout", "padding"];
            let looks_like_contract = has_dtype
                && mapping.keys().all(|key| {
                    key.as_str()
                        .is_some_and(|key| contract_fields.contains(&key) || key == "rank")
                });
            if looks_like_contract {
                let at = if path.is_empty() { "<root>" } else { &path };
                if mapping.contains_key(serde_yaml::Value::from("rank")) {
                    return Err(crate::MetadataError::Parse(format!(
                        "tensor contract at `{at}` declares retired field `rank`; remove it and \
                         provide required `shape` explicitly. The tensor rank is `shape.len()`, \
                         `shape: []` is scalar, and use `Any` for each independently unconstrained \
                         dimension"
                    )));
                }
                if !mapping.contains_key(serde_yaml::Value::from("shape")) {
                    return Err(crate::MetadataError::Parse(format!(
                        "tensor contract at `{at}` is missing required `shape`; provide the complete \
                         shape because its length is the tensor rank. Use `shape: []` for a scalar \
                         or `Any` for each independently unconstrained dimension"
                    )));
                }
            }

            for (key, value) in mapping {
                let segment = key.as_str().unwrap_or("?");
                let child = if path.is_empty() {
                    segment.to_string()
                } else {
                    format!("{path}.{segment}")
                };
                reject_invalid_tensor_contract_shapes(value, child)?;
            }
            Ok(())
        }
        serde_yaml::Value::Sequence(items) => {
            for (index, item) in items.iter().enumerate() {
                reject_invalid_tensor_contract_shapes(item, format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Refuse the short-lived top-level spelling before typed parsing obscures the
/// migration with a generic unknown-field error.
fn reject_retired_top_level_tokens(
    document: &serde_yaml::Value,
) -> Result<(), crate::MetadataError> {
    if document.get("tokens").is_none() {
        return Ok(());
    }
    Err(crate::MetadataError::Parse(
        "top-level `tokens` is retired. Put numeric package/model ids in \
         `package.tokenizer.special_tokens` (`eos_token_id` is a list); keep token strings, \
         added-token mappings, and chat templates only in tokenizer.json/tokenizer_config.json. \
         A workflow termination component consumes the resolved numeric values but does not own \
         them."
            .to_string(),
    ))
}

/// Refuse a document that still declares the retired `model.io` block.
///
/// The schema has no field for it, so `serde` would simply drop the key and the
/// package would fail later with a puzzled "declares no workflow". Recognizing
/// the retired shape here — and *only* to explain it — turns that into an
/// actionable error naming the conversion. This is the one place the old spelling
/// appears, and it never produces a value: it produces a refusal.
fn reject_retired_model_io(document: &serde_yaml::Value) -> Result<(), crate::MetadataError> {
    let declares_retired_io = document
        .get("model")
        .and_then(|model| model.get("io"))
        .is_some_and(|io| !io.is_null());
    if !declares_retired_io {
        return Ok(());
    }
    Err(crate::MetadataError::Parse(
        "this package declares the retired `model.io` block, which is no longer a way to state \
         a graph ABI. A single decoder is declared exactly like every other pipeline: a \
         `pipeline.workflow` with one ONNX component whose ports carry roles, a state_service \
         group owning its KV cache, and a generation loop. Convert the package once, offline, \
         with `migrate_model_io <package-dir>`."
            .to_string(),
    ))
}

/// Refuse a document that still spells `token_packed` as one flat pair.
///
/// The layout used to carry `offsets` and `owner` directly on the batch layout,
/// which could say only that a packed axis had exactly one level of ownership.
/// It now carries `levels`, because an item can itself be a group -- frames in
/// clips in rows -- and a single pair cannot say that.
///
/// A one-level chain is exactly what the old spelling meant, so a shim would be
/// expressible. It is refused rather than translated because this schema is
/// pre-release and [rule 3](../../../RULES.md) is to reshape it completely
/// rather than keep a second spelling alive: no in-tree package uses
/// `token_packed`, so nothing is being broken except a document written against
/// an unreleased shape, and the alternative is two ways to say one thing for as
/// long as the crate exists.
///
/// What that costs is a good error, which is why this is here. `serde` reports
/// the old shape as an unknown field, and an unknown field reads like a typo --
/// a reader goes looking for a misspelling rather than for a migration. Like
/// `model.io`, this recognizer never produces a value; it produces a refusal
/// that names the change.
fn reject_flat_token_packed(
    document: &serde_yaml::Value,
    path: String,
) -> Result<(), crate::MetadataError> {
    match document {
        serde_yaml::Value::Mapping(mapping) => {
            let is_packed = mapping
                .get(serde_yaml::Value::from("kind"))
                .and_then(serde_yaml::Value::as_str)
                == Some("token_packed");
            let retired = ["offsets", "owner"]
                .into_iter()
                .filter(|field| mapping.contains_key(serde_yaml::Value::from(*field)))
                .collect::<Vec<_>>();
            if is_packed && !retired.is_empty() {
                let at = if path.is_empty() {
                    String::new()
                } else {
                    format!(" at `{path}`")
                };
                return Err(crate::MetadataError::Parse(format!(
                    "the `token_packed` batch layout{at} declares `{}` directly, which is the \
                     retired flat spelling of ownership. A packed axis now declares `levels`, an \
                     ownership chain innermost first, because an item can itself be a group. The \
                     flat pair is the one-level case: replace `offsets: <o>, owner: <w>` with \
                     `levels: [{{ offsets: <o>, owner: <w> }}]`, and add an outer entry if the \
                     items are themselves grouped. An emitted level also states `extent`. This \
                     spelling is not read, and no document is converted for you.",
                    retired.join("` and `")
                )));
            }

            for (key, value) in mapping {
                let segment = key.as_str().unwrap_or("?");
                let child = if path.is_empty() {
                    segment.to_string()
                } else {
                    format!("{path}.{segment}")
                };
                reject_flat_token_packed(value, child)?;
            }
            Ok(())
        }
        serde_yaml::Value::Sequence(items) => {
            for (index, item) in items.iter().enumerate() {
                reject_flat_token_packed(item, format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Refuse batching hints that duplicated structural contracts or runtime policy.
///
/// These fields were never consumed by execution. Keeping them would allow a
/// profile, model hint, capability list, and component contract to disagree
/// about the same optimization. The component's `batch_capacity` is now the
/// sole authored claim that grouped execution preserves semantics; graph/state
/// contracts and the resolved backend determine feasibility, and deployment
/// configuration decides whether to group.
fn reject_retired_batching_hints(document: &serde_yaml::Value) -> Result<(), crate::MetadataError> {
    if document
        .get("model")
        .and_then(|model| model.get("runtime_configurable"))
        .and_then(|runtime| runtime.get("continuous_batching"))
        .is_some()
    {
        return Err(crate::MetadataError::Parse(
            "`model.runtime_configurable.continuous_batching` is retired. Remove it: whether a \
             resolved graph/backend can share a forward is derived structurally, while enabling \
             and sizing batches is deployment policy. For an encoder component, declare \
             `batch_capacity` only when grouped execution is semantically equivalent to solo \
             execution."
                .to_string(),
        ));
    }

    if let Some(profiles) = document
        .get("profiles")
        .and_then(serde_yaml::Value::as_mapping)
    {
        for (name, profile) in profiles {
            if profile.get("batch_invariance").is_some() {
                let name = name.as_str().unwrap_or("?");
                return Err(crate::MetadataError::Parse(format!(
                    "`profiles.{name}.batch_invariance` is retired. Grouping correctness belongs \
                     to the component that executes the grouped call: declare `batch_capacity` \
                     when grouped and solo results are equivalent, and omit `batch_capacity` when \
                     padding or co-batching changes the answer. Keep validity provenance in \
                     `TensorContract.padding` or the profile's decoding lengths binding."
                )));
            }
        }
    }

    for path in [
        "required_capabilities",
        "pipeline.workflow.manifest.capabilities",
    ] {
        let value = path
            .split('.')
            .try_fold(document, |value, segment| value.get(segment));
        if value
            .and_then(serde_yaml::Value::as_sequence)
            .is_some_and(|capabilities| {
                capabilities
                    .iter()
                    .any(|capability| capability.as_str() == Some("continuous_batching"))
            })
        {
            return Err(crate::MetadataError::Parse(format!(
                "`{path}` contains retired capability `continuous_batching`. Remove it: \
                 continuous batching is an optimization, not a correctness requirement. Typed \
                 workflow/state contracts and the resolved backend determine feasibility; \
                 deployment policy decides whether and how much to batch."
            )));
        }
    }

    Ok(())
}

/// Load inference metadata together with its canonical semantic identity.
///
/// The identity binds disposable artifacts -- compiled plans, memory plans,
/// state checkpoints -- to the metadata semantics they were produced against.
/// It is not integrity, not provenance, and not a trust decision.
pub fn load_metadata_with_identity(
    path: &Path,
) -> Result<(InferenceMetadata, String), crate::MetadataError> {
    let content = std::fs::read_to_string(path).map_err(crate::MetadataError::Io)?;
    let identity = crate::identity::semantic_identity_of_str(&content)?;
    Ok((load_metadata(path)?, identity))
}

/// Load and semantically validate a metadata document or package directory.
///
/// Package-relative artifact references are checked for existence and may not
/// escape the package root. ONNX signature admission remains the runtime's
/// responsibility because it depends on the selected execution provider.
pub fn load_metadata_package(path: &Path) -> Result<InferenceMetadata, crate::MetadataError> {
    let metadata_path = if path.is_dir() {
        [
            "inference_metadata.yaml",
            "inference_metadata.yml",
            "inference_metadata.json",
        ]
        .into_iter()
        .map(|name| path.join(name))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            crate::MetadataError::Parse(format!(
                "package '{}' has no inference_metadata.yaml, .yml, or .json",
                path.display()
            ))
        })?
    } else {
        path.to_path_buf()
    };
    let metadata = load_metadata(&metadata_path)?;
    let Some(pipeline) = &metadata.pipeline else {
        return Err(crate::MetadataError::Parse(
            "metadata has no pipeline section".to_string(),
        ));
    };
    crate::validation::validate_pipeline_spec(
        pipeline,
        crate::version::normalize(metadata.schema_version.as_deref())
            .unwrap_or(crate::version::INITIAL_SCHEMA_VERSION),
    )
    .map_err(|error| crate::MetadataError::Parse(error.to_string()))?;
    // Document-level invariants are not pipeline-scoped, so `validate_pipeline_spec`
    // cannot see them. Without this call they hold only for callers who reach for
    // `validate_metadata` directly — which is nobody loading a package from disk,
    // including the `validate_metadata` binary. A guarantee that a producer's own
    // validation run cannot observe is not a guarantee.
    crate::validation::validate_metadata(&metadata)
        .map_err(|errors| crate::MetadataError::Parse(errors.join("; ")))?;
    validate_package_artifacts(
        &pipeline.workflow,
        metadata_path.parent().unwrap_or_else(|| Path::new(".")),
    )?;
    if let Some(adapters) = &metadata.adapters {
        validate_adapter_artifacts(
            adapters,
            metadata_path.parent().unwrap_or_else(|| Path::new(".")),
        )?;
    }
    Ok(metadata)
}

fn validate_adapter_artifacts(
    service: &crate::schema::AdapterServiceContract,
    root: &Path,
) -> Result<(), crate::MetadataError> {
    for (alias, artifact) in &service.artifacts {
        for (index, source) in artifact.weights.iter().enumerate() {
            let mut files = vec![("weights", source.location.as_str())];
            if let Some(location) = &source.config_location {
                files.push(("config", location.as_str()));
            }
            for (kind, location) in files {
                resolve_package_artifact(
                    root,
                    location,
                    &format!("adapter '{alias}' source {index} {kind}"),
                )?;
            }
        }
    }
    Ok(())
}

/// Resolve one package-relative file.
///
/// Canonicalizing both paths rejects symlink and `..` escapes. Callers receive
/// the canonical file path so they load the exact confined file admitted here.
pub fn resolve_package_artifact(
    root: &Path,
    location: &str,
    description: &str,
) -> Result<PathBuf, crate::MetadataError> {
    let root = root.canonicalize().map_err(crate::MetadataError::Io)?;
    let candidate = root.join(location);
    let resolved = candidate.canonicalize().map_err(|error| {
        crate::MetadataError::Parse(format!(
            "{description} artifact '{}' cannot be opened: {error}",
            candidate.display()
        ))
    })?;
    if !resolved.starts_with(&root) {
        return Err(crate::MetadataError::Parse(format!(
            "{description} artifact '{location}' escapes package root '{}'",
            root.display()
        )));
    }
    if !resolved.is_file() {
        return Err(crate::MetadataError::Parse(format!(
            "{description} artifact '{}' is not a file",
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn validate_package_artifacts(
    workflow: &crate::schema::WorkflowSpec,
    root: &Path,
) -> Result<(), crate::MetadataError> {
    let root = root.canonicalize().map_err(crate::MetadataError::Io)?;
    let mut artifacts = Vec::new();
    for (component, declaration) in &workflow.components {
        match &declaration.implementation {
            crate::schema::ComponentImplementation::Onnx { artifact } => {
                artifacts.push((format!("component '{component}'"), artifact.as_str()));
            }
            crate::schema::ComponentImplementation::Adapter {
                artifact: Some(artifact),
                ..
            } => {
                artifacts.push((
                    format!("adapter component '{component}'"),
                    artifact.as_str(),
                ));
            }
            crate::schema::ComponentImplementation::Adapter { artifact: None, .. }
            | crate::schema::ComponentImplementation::Binding => {}
        }
    }
    for (name, input) in &workflow.inputs {
        if let crate::schema::WorkflowInputSource::Artifact { path } = &input.source {
            artifacts.push((format!("workflow input '{name}'"), path.as_str()));
        }
    }
    for (owner, artifact) in artifacts {
        let candidate = root.join(artifact);
        let resolved = candidate.canonicalize().map_err(|error| {
            crate::MetadataError::Parse(format!(
                "{owner} artifact '{}' cannot be opened: {error}",
                candidate.display()
            ))
        })?;
        if !resolved.starts_with(&root) {
            return Err(crate::MetadataError::Parse(format!(
                "{owner} artifact '{artifact}' escapes package root '{}'",
                root.display()
            )));
        }
        if !resolved.is_file() {
            return Err(crate::MetadataError::Parse(format!(
                "{owner} artifact '{}' is not a file",
                resolved.display()
            )));
        }
    }
    Ok(())
}

/// Load and validate a metadata file's `pipeline` section.
pub fn load_pipeline_spec(path: &Path) -> Result<PipelineSpec, crate::MetadataError> {
    let metadata = load_metadata(path)?;
    let version = crate::version::normalize(metadata.schema_version.as_deref())
        .unwrap_or(crate::version::INITIAL_SCHEMA_VERSION);
    let spec = metadata
        .pipeline
        .ok_or_else(|| crate::MetadataError::Parse("metadata has no pipeline section".into()))?;
    crate::validation::validate_pipeline_spec(&spec, version)
        .map_err(|err| crate::MetadataError::Parse(err.to_string()))?;
    Ok(spec)
}

/// Detect a legacy HuggingFace speculator package from `config.json`.
///
/// Detection is best-effort so malformed or unrelated external configuration
/// does not change normal model-directory loading behavior.
pub fn detect_speculator(model_dir: &Path) -> Option<SpeculatorDescriptor> {
    let config_path = model_dir.join("config.json");
    let content = std::fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<HuggingFaceModelConfig>(&content)
        .ok()?
        .speculator_config?;
    Some(SpeculatorDescriptor::from_config(
        model_dir,
        config,
        SpeculatorConfigSource::HuggingFaceConfig,
    ))
}

#[derive(serde::Deserialize)]
struct HuggingFaceModelConfig {
    #[serde(default)]
    speculator_config: Option<SpeculatorConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct LegacySpeculatorDocument {
        speculative: SpeculatorConfig,
    }

    fn parse_legacy_speculator(document: &str) -> SpeculatorConfig {
        serde_yaml::from_str::<LegacySpeculatorDocument>(document)
            .expect("legacy speculator config parses")
            .speculative
    }

    /// A directory of this test's own, so parallel tests cannot see each
    /// other's sidecars.
    fn metadata_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("metadata-discovery-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the test directory must be created");
        dir
    }

    #[test]
    fn a_directory_without_a_sidecar_has_no_metadata() {
        let dir = metadata_dir("absent");
        assert_eq!(find_metadata_path(&dir), None);
        assert!(
            load_metadata_from_dir(&dir)
                .expect("an absent sidecar is not an error")
                .is_none()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn yaml_wins_over_the_other_spellings_of_the_same_sidecar() {
        let dir = metadata_dir("preference");
        for name in METADATA_FILE_NAMES {
            std::fs::write(dir.join(name), "version: v1\n").unwrap();
        }
        assert_eq!(
            find_metadata_path(&dir),
            Some(dir.join("inference_metadata.yaml")),
            "a package that ships more than one spelling must resolve the same way everywhere"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // A malformed sidecar used to read as an absent one, which is how a model
    // came to run with every declared setting -- context length, chunked
    // prefill, EOS ids, sampling defaults -- quietly ignored.
    #[test]
    fn a_sidecar_that_cannot_be_parsed_is_an_error_not_an_absence() {
        let dir = metadata_dir("malformed");
        std::fs::write(
            dir.join("inference_metadata.yaml"),
            "model: [this is not a mapping",
        )
        .unwrap();

        let error = load_metadata_from_dir(&dir).expect_err("a malformed sidecar must be reported");
        assert!(
            matches!(error, crate::MetadataError::Parse(_)),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tensor_contract_rank_is_rejected_with_its_full_path_and_required_shape() {
        let document: serde_yaml::Value = serde_yaml::from_str(
            "\
pipeline:
  workflow:
    components:
      decoder:
        ports:
          inputs:
            input_ids:
              dtype: int64
              rank: 2
",
        )
        .expect("test document parses");

        let error = gate_document(&document).expect_err("retired tensor rank must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("pipeline.workflow.components.decoder.ports.inputs.input_ids")
                && message.contains("retired field `rank`")
                && message.contains("provide required `shape` explicitly")
                && message.contains("shape.len()"),
            "unexpected diagnostic: {message}"
        );
    }

    #[test]
    fn tensor_contract_without_shape_is_rejected_with_its_full_path() {
        let document: serde_yaml::Value = serde_yaml::from_str(
            "\
pipeline:
  workflow:
    inputs:
      request.input_ids:
        contract:
          dtype: int64
",
        )
        .expect("test document parses");

        let error = gate_document(&document).expect_err("missing tensor shape must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("pipeline.workflow.inputs.request.input_ids.contract")
                && message.contains("missing required `shape`")
                && message.contains("shape: []")
                && message.contains("independently unconstrained"),
            "unexpected diagnostic: {message}"
        );
    }

    const SHARED_KV_YAML: &str = "\
speculative:
  proposal_type: shared_kv
  num_speculative_tokens: 3
  model: assistant/model.onnx
  backbone_hidden_size: 16
  vocab_size: 32
  projected_state_output: projected_state
  logits_output: logits
  input_embedding: input_embedding.f32
  shared_kv:
    - name: sliding_attention
      target_layers: [0]
    - name: full_attention
      target_layers: [1]
";

    /// A legacy `proposal_type: shared_kv` block still parses — third-party
    /// configs in the wild carry it — but it no longer resolves to a runnable
    /// proposer, because borrowed-KV drafting is declared by the package's
    /// `speculative.proposal_execution: {kind: chained}` workflow contract and
    /// driven by the interpreter. The diagnostic has to say so, or a package
    /// author reading "unknown" would try to fix the wrong block.
    #[test]
    fn legacy_shared_kv_speculator_degrades_to_unknown_naming_the_workflow_contract() {
        let config = parse_legacy_speculator(SHARED_KV_YAML);
        assert_eq!(config.proposal_type, ProposalType::SharedKv);
        assert_eq!(config.num_speculative_tokens, 3);
        assert_eq!(config.shared_kv.len(), 2);

        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/shared-kv"),
            config,
            SpeculatorConfigSource::HuggingFaceConfig,
        );
        let SpeculatorProposerStatus::Unknown(reason) = descriptor.proposer else {
            panic!("a legacy shared_kv descriptor must not resolve to a runnable proposer");
        };
        assert!(
            reason.contains("proposal_execution") && reason.contains("chained"),
            "the diagnostic must point at the workflow contract: {reason}"
        );
    }

    #[test]
    fn legacy_dflash_is_migration_input_not_parallel_runtime_authority() {
        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/dflash"),
            SpeculatorConfig {
                proposal_type: ProposalType::DFlash,
                num_speculative_tokens: 8,
                verifier: None,
                model: None,
                io: None,
                backbone_hidden_size: None,
                vocab_size: None,
                projected_state_output: None,
                logits_output: None,
                input_embedding: None,
                shared_kv: Vec::new(),
                target_hidden_output: None,
                target_hidden_layout: None,
                target_hidden_size: None,
                hc_mult: None,
                mtp_hidden_output: None,
                mtp_state_output: None,
                kv_mode: None,
                embedding: None,
                lm_head: None,
            },
            SpeculatorConfigSource::HuggingFaceConfig,
        );
        let SpeculatorProposerStatus::Unknown(reason) = descriptor.proposer else {
            panic!("legacy DFlash must not resolve beside the canonical workflow contract");
        };
        assert!(
            reason.contains("dflash_flat_block")
                && reason.contains("target-hidden provenance")
                && reason.contains("accepted-prefix"),
            "{reason}"
        );
    }

    /// A legacy `gemma4_assistant` proposal_type (pre-generalization name) no
    /// longer resolves to SharedKv — it degrades gracefully to Unknown instead
    /// of hard-failing model loading.
    #[test]
    fn legacy_gemma4_assistant_proposal_type_degrades_to_unknown() {
        for legacy in &["gemma4_assistant", "gemma4-assistant"] {
            let yaml = format!(
                "\
speculative:
  proposal_type: {legacy}
  num_speculative_tokens: 3
  model: assistant/model.onnx
  backbone_hidden_size: 16
  vocab_size: 32
  shared_kv:
    - name: sliding_attention
      target_layers: [0]
"
            );
            let config = parse_legacy_speculator(&yaml);
            assert!(
                matches!(config.proposal_type, ProposalType::Unknown(_)),
                "expected Unknown for legacy value '{legacy}', got {:?}",
                config.proposal_type
            );
            let descriptor = SpeculatorDescriptor::from_config(
                Path::new("/models/shared-kv"),
                config,
                SpeculatorConfigSource::HuggingFaceConfig,
            );
            assert!(
                matches!(descriptor.proposer, SpeculatorProposerStatus::Unknown(_)),
                "expected proposer Unknown for legacy value '{legacy}'"
            );
        }
    }
}
