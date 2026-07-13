//! Load inference metadata from YAML or JSON files.

use crate::schema::{
    InferenceMetadata, PipelineSpec, ProposalType, SharedKvGroup, SpeculatorConfig,
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
    /// Shared-KV binding groups consumed by the assistant.
    pub shared_kv: Vec<SharedKvGroup>,
}

/// Current construction status for the engine-facing proposer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculatorProposerStatus {
    /// A fully resolved shared-KV proposer.
    SharedKv(SharedKvProposerSpec),
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
            ProposalType::Mtp => {
                SpeculatorProposerStatus::NotYetSupported(SpeculatorProposerKind::Mtp)
            }
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
}

/// Resolve a `shared_kv` speculator into a supported proposer status.
///
/// Missing or malformed required fields — `model`, `backbone_hidden_size`,
/// `vocab_size`, an empty `shared_kv` list, or any group with empty
/// `target_layers` — degrade to [`SpeculatorProposerStatus::Unknown`] so a
/// malformed descriptor never aborts model loading; the engine treats such
/// descriptors as absent.
fn resolve_shared_kv(
    model_dir: &Path,
    config: &SpeculatorConfig,
) -> SpeculatorProposerStatus {
    let Some(model) = config.model.as_ref() else {
        return SpeculatorProposerStatus::Unknown(
            "shared_kv metadata is missing `model`".into(),
        );
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
    SpeculatorProposerStatus::SharedKv(SharedKvProposerSpec {
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
        shared_kv: config.shared_kv.clone(),
    })
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
            let metadata: InferenceMetadata =
                serde_yaml::from_str(&yaml).expect("metadata parses");
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
        assert_eq!(spec.num_speculative_tokens, 4);
        assert_eq!(spec.shared_kv.len(), 1);
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
