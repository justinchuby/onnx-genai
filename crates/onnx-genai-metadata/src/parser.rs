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
            ProposalType::DFlash => {
                SpeculatorProposerStatus::NotYetSupported(SpeculatorProposerKind::DFlash)
            }
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
    reject_retired_model_io(&content)?;

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

/// Refuse a document that still declares the retired `model.io` block.
///
/// The schema has no field for it, so `serde` would simply drop the key and the
/// package would fail later with a puzzled "declares no workflow". Recognizing
/// the retired shape here — and *only* to explain it — turns that into an
/// actionable error naming the conversion. This is the one place the old spelling
/// appears, and it never produces a value: it produces a refusal.
fn reject_retired_model_io(content: &str) -> Result<(), crate::MetadataError> {
    let Ok(document) = serde_yaml::from_str::<serde_yaml::Value>(content) else {
        // Not parseable as YAML at all; the real parser reports why.
        return Ok(());
    };
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
    crate::validation::validate_pipeline_spec(pipeline)
        .map_err(|error| crate::MetadataError::Parse(error.to_string()))?;
    // Document-level invariants are not pipeline-scoped, so `validate_pipeline_spec`
    // cannot see them. Without this call the rule that forbids a package from
    // carrying both `model.io` and a workflow holds only for callers who reach
    // for `validate_metadata` directly — which is nobody loading a package from
    // disk, including the `validate_metadata` binary. A guarantee that a
    // producer's own validation run cannot observe is not a guarantee.
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
    let spec = metadata
        .pipeline
        .ok_or_else(|| crate::MetadataError::Parse("metadata has no pipeline section".into()))?;
    crate::validation::validate_pipeline_spec(&spec)
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
