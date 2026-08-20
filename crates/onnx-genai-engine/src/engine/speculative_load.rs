//! Weight readers and speculative proposer (MTP / shared-KV) loading.

use super::*;

pub(crate) fn read_f32_weights(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read f32 weights from '{}'", path.display()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!(
            "f32 weight file '{}' has byte length {}, which is not divisible by 4",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

#[cfg(feature = "native-backend")]
pub(crate) fn load_native_shared_kv_proposer(
    metadata: &InferenceMetadata,
    model_dir: &Path,
    device: crate::native_decode::NativeDecodeDevice,
) -> anyhow::Result<(Option<NativeSharedKvProposerModel>, SpeculativeMode)> {
    let Some(config) = metadata.speculative.as_ref() else {
        return Ok((None, SpeculativeMode::None));
    };
    if config.proposal_type != ProposalType::SharedKv {
        return Ok((None, SpeculativeMode::None));
    }
    if config.io.is_none() {
        tracing::warn!(
            "shared-KV proposer metadata has no explicit speculative.io execution contract; native target decode remains available, but the proposer stays disabled until sequence_source, kv_ownership, and output roles are declared"
        );
        return Ok((None, SpeculativeMode::None));
    }
    let descriptor = onnx_genai_metadata::resolve_speculator_config(model_dir, config.clone());
    let spec = match descriptor.proposer {
        SpeculatorProposerStatus::SharedKv(spec) => spec,
        SpeculatorProposerStatus::Unknown(reason) => {
            anyhow::bail!("invalid native shared-KV proposer metadata: {reason}")
        }
        other => {
            anyhow::bail!("shared-KV metadata resolved to unexpected proposer status {other:?}")
        }
    };
    let target_hidden_output = metadata
            .model
            .as_ref()
            .and_then(|model| model.io.as_ref())
            .and_then(|io| io.hidden_output.clone())
            .context(
                "native shared-KV speculation requires model.io.hidden_output to name the target decoder hidden-state output; add the exact graph output name to inference metadata",
            )?;
    for group in &spec.shared_kv {
        for (field, value) in [
            ("key_input", group.key_input.as_deref()),
            ("value_input", group.value_input.as_deref()),
            ("target_key_input", group.target_key_input.as_deref()),
            ("target_value_input", group.target_value_input.as_deref()),
        ] {
            if value.is_none_or(str::is_empty) {
                anyhow::bail!(
                    "native shared-KV group '{}' is missing `{field}`; declare exact proposer and target KV port names so the runtime never infers cache roles from model or tensor names",
                    group.name
                );
            }
        }
    }
    let weights = read_f32_weights(&spec.input_embedding)?;
    let embedder = LinearEmbedder::new(weights, spec.vocab_size, spec.backbone_hidden_size)
        .context("build native shared-KV target embedding lookup")?;
    let session =
        crate::native_decode::NativeProposerSession::load(&spec.model, device, Some(&spec.io))
            .with_context(|| {
                format!(
                    "load native shared-KV proposer graph '{}'",
                    spec.model.display()
                )
            })?;
    let mode = SpeculativeMode::SharedKv(SharedKvProposerConfig {
        assistant_model: spec.model,
        target_hidden_output,
        input_embedding_weights: spec.input_embedding,
        backbone_hidden_size: spec.backbone_hidden_size,
        vocab_size: spec.vocab_size,
        num_speculative_tokens: spec.num_speculative_tokens,
        shared_kv: spec
            .shared_kv
            .iter()
            .map(|group| SharedKvBinding {
                name: group.name.clone(),
                target_layers: group.target_layers.clone(),
            })
            .collect(),
    });
    Ok((
        Some(NativeSharedKvProposerModel {
            session,
            embedder,
            groups: spec.shared_kv,
            hidden_size: spec.backbone_hidden_size,
        }),
        mode,
    ))
}

/// Resolve a native MTP runtime configuration from the already-loaded metadata.
///
/// The target vocabulary is read from the target `logits` signature; exact
/// embedding and LM-head initializer names remain package references until the
/// MTP model is initialized.
pub(crate) fn mtp_config_from_metadata(
    metadata: &InferenceMetadata,
    model_dir: &Path,
    session: &Session,
) -> anyhow::Result<Option<ResolvedMtpConfig>> {
    let Some(config) = metadata.speculative.as_ref() else {
        return Ok(None);
    };
    if config.proposal_type != ProposalType::Mtp {
        return Ok(None);
    }
    let descriptor = onnx_genai_metadata::resolve_speculator_config(model_dir, config.clone());
    let spec = match descriptor.proposer {
        SpeculatorProposerStatus::Mtp(spec) => spec,
        SpeculatorProposerStatus::Unknown(reason) => {
            anyhow::bail!("Invalid MTP sidecar metadata: {reason}")
        }
        other => anyhow::bail!("MTP metadata resolved to unexpected proposer status {other:?}"),
    };
    let vocab_size = session
        .outputs()
        .iter()
        .find(|output| output.name == "logits")
        .and_then(|output| output.shape.last().copied())
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .context("MTP metadata requires a target logits output with static vocabulary size")?;
    let config = ResolvedMtpConfig::from_sidecar_descriptor(&spec, vocab_size);
    validate_resolved_mtp_config(&config)?;
    Ok(Some(config))
}

/// Build a [`SpeculativeMode::SharedKv`] from a model directory's native
/// inference metadata, or `None` when no supported assistant is advertised.
///
/// The target hidden output name is not part of the shared metadata contract,
/// so it is auto-detected: the first Float32 output whose last dimension equals
/// the advertised backbone hidden size (excluding `logits`).
pub(crate) fn shared_kv_mode_from_metadata(
    model_dir: &Path,
    session: &Session,
) -> Option<SpeculativeMode> {
    let descriptor = onnx_genai_metadata::detect_speculator(model_dir)?;
    let onnx_genai_metadata::SpeculatorProposerStatus::SharedKv(spec) = descriptor.proposer else {
        return None;
    };
    let target_hidden_output = detect_target_hidden_output(session, spec.backbone_hidden_size)?;
    let shared_kv = spec
        .shared_kv
        .into_iter()
        .map(|group| SharedKvBinding {
            name: group.name,
            target_layers: group.target_layers,
        })
        .collect();
    Some(SpeculativeMode::SharedKv(SharedKvProposerConfig {
        assistant_model: spec.model,
        target_hidden_output,
        input_embedding_weights: spec.input_embedding,
        backbone_hidden_size: spec.backbone_hidden_size,
        vocab_size: spec.vocab_size,
        num_speculative_tokens: spec.num_speculative_tokens,
        shared_kv,
    }))
}

/// Find a Float32 hidden-state output ending in `hidden_size` (not `logits`).
pub(crate) fn detect_target_hidden_output(session: &Session, hidden_size: usize) -> Option<String> {
    session
        .outputs()
        .iter()
        .find(|output| {
            output.dtype == DataType::Float32
                && !output.name.to_ascii_lowercase().contains("logits")
                && output.shape.last().copied().filter(|dim| *dim > 0) == Some(hidden_size as i64)
        })
        .map(|output| output.name.clone())
}

/// Stable, opaque model identity derived from the model directory name.
///
/// Used only to namespace connector cache keys when the caller does not supply
/// an explicit `model_id`. It is never interpreted or branched on.
pub(crate) fn default_connector_model_id(model_directory: &ModelDirectory) -> String {
    model_directory
        .root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "onnx-genai-model".to_string())
}

/// Build the engine's KV connector bridge from generic, model-agnostic config.
pub(crate) fn build_connector_bridge(
    config: &KvConnectorConfig,
    model_directory: &ModelDirectory,
    kv_model: Option<&KvModelInfo>,
) -> anyhow::Result<ConnectorBridge> {
    match &config.backend {
        KvConnectorBackend::Null => Ok(ConnectorBridge::null()),
        KvConnectorBackend::LocalTiered(local_config) => {
            let connector = LocalTieredConnector::new(local_config.clone()).map_err(|error| {
                anyhow::anyhow!("failed to build LocalTiered KV connector: {error}")
            })?;
            let model_id = config
                .model_id
                .clone()
                .unwrap_or_else(|| default_connector_model_id(model_directory));
            let chunk_size = if config.chunk_size == 0 {
                onnx_genai_kv::DEFAULT_CHUNK_SIZE
            } else {
                config.chunk_size
            };
            let num_layers = kv_model
                .map(|model| model.tensor_config.num_layers)
                .unwrap_or(1)
                .max(1);
            ConnectorBridge::new(
                Arc::new(connector),
                model_id,
                chunk_size,
                0..num_layers,
                config.store_priority,
                config.recompute_ms_per_token,
            )
        }
    }
}
