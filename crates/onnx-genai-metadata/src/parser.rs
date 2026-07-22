//! Load inference metadata from YAML or JSON files.

use crate::schema::{
    InferenceMetadata, MtpHiddenLayout, MtpKvMode, PipelineSpec, ProposalType, SharedKvGroup,
    SpeculatorConfig,
};
use std::path::{Path, PathBuf};

/// Source used to discover a speculator declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculatorConfigSource {
    InferenceMetadata,
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

/// Resolved shared-KV proposer descriptor.
///
/// Every field is resolved from the `speculative` metadata section, with
/// output-name defaults applied. `model` is resolved relative to the model
/// directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedKvProposerSpec {
    /// Absolute path to the assistant ONNX model.
    pub model: PathBuf,
    /// Number of speculative tokens proposed after the guaranteed target token.
    pub num_speculative_tokens: usize,
    /// Target backbone hidden size `H`.
    pub backbone_hidden_size: usize,
    /// Vocabulary size of the assistant's own `logits` output.
    pub vocab_size: usize,
    /// Name of the assistant output threaded forward between steps.
    pub projected_state_output: String,
    /// Name of the assistant's draft-distribution output.
    pub logits_output: String,
    /// Absolute path to the target model's raw input-token embedding table
    /// (`[vocab_size, backbone_hidden_size]` little-endian f32).
    pub input_embedding: PathBuf,
    /// Shared-KV binding groups consumed by the assistant.
    pub shared_kv: Vec<SharedKvGroup>,
    /// Fully resolved proposer execution contract.
    pub io: crate::schema::ModelIoSpec,
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
    /// Sidecar recurrent HC-state output.
    pub mtp_state_output: String,
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
    /// A fully resolved shared-KV proposer.
    SharedKv(Box<SharedKvProposerSpec>),
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
            ProposalType::DFlash => {
                SpeculatorProposerStatus::NotYetSupported(SpeculatorProposerKind::DFlash)
            }
            ProposalType::SharedKv => resolve_shared_kv(model_dir, &config),
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
            mtp_state_output: config
                .mtp_state_output
                .clone()
                .unwrap_or_else(|| "mtp_state".into()),
            kv_mode: config.kv_mode.unwrap_or(MtpKvMode::ProposalLocal),
            embedding_initializer: embedding.name.clone(),
            lm_head_initializer: lm_head.name.clone(),
        })
    }
}

/// Resolve a parsed speculative declaration without re-reading metadata.
pub fn resolve_speculator_config(
    model_dir: &Path,
    config: SpeculatorConfig,
) -> SpeculatorDescriptor {
    SpeculatorDescriptor::from_config(model_dir, config, SpeculatorConfigSource::InferenceMetadata)
}

/// Resolve a `shared_kv` speculator into a supported proposer status.
///
/// Missing or malformed required fields — `model`, `backbone_hidden_size`,
/// `vocab_size`, an empty `shared_kv` list, or any group with empty
/// `target_layers` — degrade to [`SpeculatorProposerStatus::Unknown`] so a
/// malformed descriptor never aborts model loading; the engine treats such
/// descriptors as absent.
fn resolve_shared_kv(model_dir: &Path, config: &SpeculatorConfig) -> SpeculatorProposerStatus {
    let Some(model) = config.model.as_ref() else {
        return SpeculatorProposerStatus::Unknown("shared_kv metadata is missing `model`".into());
    };
    let Some(backbone_hidden_size) = config.backbone_hidden_size else {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata is missing `backbone_hidden_size`".into(),
        );
    };
    let Some(vocab_size) = config.vocab_size else {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata is missing `vocab_size`".into(),
        );
    };
    let Some(input_embedding) = config.input_embedding.as_ref() else {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata is missing `input_embedding` (target input-token \
             embedding table required to build the assistant's inputs_embeds)"
                .into(),
        );
    };
    if config.shared_kv.is_empty() {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata declares no `shared_kv` binding groups".into(),
        );
    }
    if let Some(group) = config
        .shared_kv
        .iter()
        .find(|group| group.target_layers.is_empty())
    {
        return SpeculatorProposerStatus::Unknown(format!(
            "shared_kv group '{}' lists no `target_layers`",
            group.name
        ));
    }
    let io = config
        .io
        .clone()
        .unwrap_or_else(|| crate::schema::ModelIoSpec {
            sequence_source: Some(crate::schema::SequenceInputKind::InputsEmbeds),
            kv_ownership: Some(crate::schema::KvOwnership::Shared),
            token_input: None,
            inputs_embeds_input: Some("inputs_embeds".into()),
            attention_mask_input: Some("attention_mask".into()),
            position_ids_input: Some("position_ids".into()),
            logits_output: Some(
                config
                    .logits_output
                    .clone()
                    .unwrap_or_else(|| "logits".into()),
            ),
            hidden_output: Some(
                config
                    .projected_state_output
                    .clone()
                    .unwrap_or_else(|| "projected_state".into()),
            ),
            kv_inputs: None,
            kv_outputs: None,
            encoder_hidden_states_input: None,
            cross_kv_inputs: None,
            cross_kv_outputs: None,
            kv_update: None,
            state_pairs: None,
            optional_inputs: std::collections::BTreeMap::new(),
        });
    if io
        .sequence_source
        .unwrap_or(crate::schema::SequenceInputKind::TokenIds)
        != crate::schema::SequenceInputKind::InputsEmbeds
    {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata `io.sequence_source` must be `inputs_embeds`".into(),
        );
    }
    if io.kv_ownership.unwrap_or(crate::schema::KvOwnership::Owned)
        != crate::schema::KvOwnership::Shared
    {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata `io.kv_ownership` must be `shared`".into(),
        );
    }
    if io.inputs_embeds_input.as_deref().is_none_or(str::is_empty) {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata is missing `io.inputs_embeds_input`".into(),
        );
    }
    if io.logits_output.as_deref().is_none_or(str::is_empty)
        && io.hidden_output.as_deref().is_none_or(str::is_empty)
    {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata must declare at least one output role: `io.logits_output` or `io.hidden_output`"
                .into(),
        );
    }
    SpeculatorProposerStatus::SharedKv(Box::new(SharedKvProposerSpec {
        model: model_dir.join(model),
        num_speculative_tokens: config.num_speculative_tokens,
        backbone_hidden_size,
        vocab_size,
        projected_state_output: config
            .projected_state_output
            .clone()
            .unwrap_or_else(|| "projected_state".to_string()),
        logits_output: config
            .logits_output
            .clone()
            .unwrap_or_else(|| "logits".to_string()),
        input_embedding: model_dir.join(input_embedding),
        shared_kv: config.shared_kv.clone(),
        io,
    }))
}

/// Load inference metadata from a file (YAML or JSON based on extension).
pub fn load_metadata(path: &Path) -> Result<InferenceMetadata, crate::MetadataError> {
    let content = std::fs::read_to_string(path).map_err(crate::MetadataError::Io)?;

    let metadata: InferenceMetadata = match path.extension().and_then(|e| e.to_str()) {
        Some("yaml" | "yml") => serde_yaml::from_str(&content)
            .map_err(|e| crate::MetadataError::Parse(e.to_string()))?,
        Some("json") => serde_json::from_str(&content)
            .map_err(|e| crate::MetadataError::Parse(e.to_string()))?,
        _ => {
            // Try YAML first, then JSON
            if let Ok(m) = serde_yaml::from_str::<InferenceMetadata>(&content) {
                m
            } else {
                serde_json::from_str::<InferenceMetadata>(&content)
                    .map_err(|e| crate::MetadataError::Parse(e.to_string()))?
            }
        }
    };

    Ok(metadata)
}

/// Load and validate a metadata file's `pipeline` section.
pub fn load_pipeline_spec(path: &Path) -> Result<PipelineSpec, crate::MetadataError> {
    let metadata = load_metadata(path)?;
    let spec = metadata
        .pipeline
        .ok_or_else(|| crate::MetadataError::Parse("metadata has no pipeline section".into()))?;
    crate::validation::validate_pipeline_spec(&spec)
        .map_err(|err| crate::MetadataError::Parse(err.to_string()))?;
    Ok(spec)
}

/// Detect a speculator package, preferring native inference metadata over the
/// HuggingFace `config.json` compatibility format.
///
/// Detection is best-effort so malformed or unrelated external configuration
/// does not change normal model-directory loading behavior.
pub fn detect_speculator(model_dir: &Path) -> Option<SpeculatorDescriptor> {
    for name in [
        "inference_metadata.yaml",
        "inference_metadata.yml",
        "inference_metadata.json",
    ] {
        let path = model_dir.join(name);
        if !path.is_file() {
            continue;
        }
        if let Ok(metadata) = load_metadata(&path)
            && let Some(config) = metadata.speculative
        {
            return Some(SpeculatorDescriptor::from_config(
                model_dir,
                config,
                SpeculatorConfigSource::InferenceMetadata,
            ));
        }
    }

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
    use crate::schema::InferenceMetadata;

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

    #[test]
    fn shared_kv_metadata_round_trips_into_supported_descriptor() {
        let metadata: InferenceMetadata =
            serde_yaml::from_str(SHARED_KV_YAML).expect("shared_kv metadata parses");
        let config = metadata.speculative.expect("speculative section present");
        assert_eq!(config.proposal_type, ProposalType::SharedKv);
        assert_eq!(config.num_speculative_tokens, 3);
        assert_eq!(config.backbone_hidden_size, Some(16));
        assert_eq!(config.vocab_size, Some(32));
        assert_eq!(config.shared_kv.len(), 2);
        assert_eq!(config.shared_kv[0].name, "sliding_attention");
        assert_eq!(config.shared_kv[0].target_layers, vec![0]);
        assert_eq!(config.shared_kv[1].name, "full_attention");
        assert_eq!(config.shared_kv[1].target_layers, vec![1]);

        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/shared-kv"),
            config,
            SpeculatorConfigSource::InferenceMetadata,
        );
        let SpeculatorProposerStatus::SharedKv(spec) = descriptor.proposer else {
            panic!("expected a supported shared_kv proposer");
        };
        assert_eq!(
            spec.model,
            Path::new("/models/shared-kv/assistant/model.onnx")
        );
        assert_eq!(spec.num_speculative_tokens, 3);
        assert_eq!(spec.backbone_hidden_size, 16);
        assert_eq!(spec.vocab_size, 32);
        assert_eq!(spec.projected_state_output, "projected_state");
        assert_eq!(spec.logits_output, "logits");
        assert_eq!(
            spec.input_embedding,
            Path::new("/models/shared-kv/input_embedding.f32")
        );
        assert_eq!(spec.shared_kv.len(), 2);
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
            let metadata: InferenceMetadata = serde_yaml::from_str(&yaml).expect("metadata parses");
            let config = metadata.speculative.expect("speculative section present");
            assert!(
                matches!(config.proposal_type, ProposalType::Unknown(_)),
                "expected Unknown for legacy value '{legacy}', got {:?}",
                config.proposal_type
            );
            let descriptor = SpeculatorDescriptor::from_config(
                Path::new("/models/shared-kv"),
                config,
                SpeculatorConfigSource::InferenceMetadata,
            );
            assert!(
                matches!(descriptor.proposer, SpeculatorProposerStatus::Unknown(_)),
                "expected proposer Unknown for legacy value '{legacy}'"
            );
        }
    }

    #[test]
    fn shared_kv_defaults_output_names() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            "\
speculative:
  proposal_type: shared-kv
  model: assistant/model.onnx
  backbone_hidden_size: 8
  vocab_size: 16
  input_embedding: input_embedding.f32
  shared_kv:
    - name: sliding_attention
      target_layers: [0]
",
        )
        .expect("shared_kv metadata parses");
        let config = metadata.speculative.expect("speculative section present");
        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/shared-kv"),
            config,
            SpeculatorConfigSource::InferenceMetadata,
        );
        let SpeculatorProposerStatus::SharedKv(spec) = descriptor.proposer else {
            panic!("expected a supported shared_kv proposer");
        };
        assert_eq!(spec.projected_state_output, "projected_state");
        assert_eq!(spec.logits_output, "logits");
        assert_eq!(
            spec.input_embedding,
            Path::new("/models/shared-kv/input_embedding.f32")
        );
        assert_eq!(spec.num_speculative_tokens, 4);
        assert_eq!(spec.shared_kv.len(), 1);
        assert_eq!(
            spec.io.sequence_source,
            Some(crate::schema::SequenceInputKind::InputsEmbeds)
        );
        assert_eq!(
            spec.io.kv_ownership,
            Some(crate::schema::KvOwnership::Shared)
        );
        assert_eq!(
            spec.io.inputs_embeds_input.as_deref(),
            Some("inputs_embeds")
        );
        assert_eq!(spec.io.logits_output.as_deref(), Some("logits"));
        assert_eq!(spec.io.hidden_output.as_deref(), Some("projected_state"));
    }

    #[test]
    fn shared_kv_explicit_execution_contract_and_ports_are_preserved() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            "\
speculative:
  proposal_type: shared_kv
  model: proposer.onnx
  backbone_hidden_size: 6
  vocab_size: 10
  input_embedding: embedding.f32
  io:
    sequence_source: inputs_embeds
    kv_ownership: shared
    inputs_embeds_input: proposer_embeddings
    logits_output: draft_scores
    hidden_output: recurrent_projection
  shared_kv:
    - name: local
      target_layers: [0]
      key_input: proposer_cache_key
      value_input: proposer_cache_value
      target_key_input: target_cache_key
      target_value_input: target_cache_value
",
        )
        .expect("explicit proposer metadata parses");
        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/explicit"),
            metadata.speculative.expect("speculative section"),
            SpeculatorConfigSource::InferenceMetadata,
        );
        let SpeculatorProposerStatus::SharedKv(spec) = descriptor.proposer else {
            panic!("expected shared-KV proposer");
        };
        assert_eq!(
            spec.io.sequence_source,
            Some(crate::schema::SequenceInputKind::InputsEmbeds)
        );
        assert_eq!(
            spec.io.kv_ownership,
            Some(crate::schema::KvOwnership::Shared)
        );
        assert_eq!(
            spec.io.inputs_embeds_input.as_deref(),
            Some("proposer_embeddings")
        );
        assert_eq!(spec.io.logits_output.as_deref(), Some("draft_scores"));
        assert_eq!(
            spec.io.hidden_output.as_deref(),
            Some("recurrent_projection")
        );
        assert_eq!(
            spec.shared_kv[0].target_key_input.as_deref(),
            Some("target_cache_key")
        );
    }

    #[test]
    fn shared_kv_missing_required_field_is_unknown() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            "\
speculative:
  proposal_type: shared_kv
  model: assistant/model.onnx
",
        )
        .expect("shared_kv metadata parses");
        let config = metadata.speculative.expect("speculative section present");
        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/shared-kv"),
            config,
            SpeculatorConfigSource::InferenceMetadata,
        );
        assert!(matches!(
            descriptor.proposer,
            SpeculatorProposerStatus::Unknown(_)
        ));
    }

    /// A malformed shared-KV block (empty `shared_kv`, or a group with empty
    /// `target_layers`) must degrade to `Unknown` rather than resolve, so it
    /// never aborts model loading — the engine treats it as absent.
    #[test]
    fn shared_kv_empty_binding_groups_degrade_to_unknown() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            "\
speculative:
  proposal_type: shared_kv
  model: assistant/model.onnx
  backbone_hidden_size: 8
  vocab_size: 16
",
        )
        .expect("shared_kv metadata parses");
        let config = metadata.speculative.expect("speculative section present");
        assert!(config.shared_kv.is_empty());
        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/shared-kv"),
            config,
            SpeculatorConfigSource::InferenceMetadata,
        );
        assert!(matches!(
            descriptor.proposer,
            SpeculatorProposerStatus::Unknown(_)
        ));
    }

    #[test]
    fn shared_kv_empty_target_layers_degrade_to_unknown() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            "\
speculative:
  proposal_type: shared_kv
  model: assistant/model.onnx
  backbone_hidden_size: 8
  vocab_size: 16
  shared_kv:
    - name: sliding_attention
",
        )
        .expect("shared_kv metadata parses");
        let config = metadata.speculative.expect("speculative section present");
        assert!(config.shared_kv[0].target_layers.is_empty());
        let descriptor = SpeculatorDescriptor::from_config(
            Path::new("/models/shared-kv"),
            config,
            SpeculatorConfigSource::InferenceMetadata,
        );
        assert!(matches!(
            descriptor.proposer,
            SpeculatorProposerStatus::Unknown(_)
        ));
    }
}
