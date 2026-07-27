//! Main generation engine.

pub(crate) use crate::FimConfig;
pub(crate) use crate::config::{ResolvedMtpConfig, validate_resolved_mtp_config};
pub(crate) use crate::connector_bridge::ConnectorBridge;
pub(crate) use crate::decode::{
    DecodeState, ModelDecodePath, detect_model_decode_path, next_session_token_argmax,
    next_session_token_logits, next_session_token_sampled,
};
pub(crate) use crate::decode_loop::{
    DecodeLoopBackend, DecodeLoopState, exceeded_context_limit, run_decode_loop, step_decode_loop,
};
pub(crate) use crate::kv_bridge::{
    KvModelInfo, PlacedPayload, attach_pages_to_sequence, chunk_payload_from_exported,
    common_prefix_len, exported_layers_from_runner, infer_kv_model_info, kv_model_past_is_f32,
    load_materialized_past, past_kv_from_payloads, sequence_pages_for_len,
};
pub(crate) use crate::logits::{StopSequence, TokenId};
pub(crate) use crate::processors::{
    build_processor_chain, ensure_constrained_finish, load_fim_config_from_model_dir,
    push_unique_stop_sequence,
};
pub(crate) use crate::sampling::{Sampler, SamplingRng};
pub(crate) use crate::session::{ActiveGenerate, DraftModel, DraftSession, EngineSession};
pub(crate) use anyhow::Context;
pub(crate) use onnx_genai_kv::{Device, KvCacheOps, KvDType, LocalTieredConnector, PagedKvCache, PrefixCache};
pub(crate) use onnx_genai_metadata::{InferenceMetadata, ProposalType, SpeculatorProposerStatus};
pub(crate) use onnx_genai_ort::{
    DataType, Eagle3DecodeSession, Environment, ModelDirectory, MtpDecodeSession, Session,
    SessionOptions, SharedKvProposerSession, Tokenizer,
};
pub(crate) use onnx_genai_scheduler::{
    CapacityProvider, CapacityProviders, FixedCapacity, GovernorReconfigureOutcome,
    GovernorSnapshot, ModelKvConfig, Priority, ResourceError, ResourceGovernor, ResourceLimit,
    ResourceLimits, Scheduler, VramBreakdown,
};
pub(crate) use std::collections::HashMap;
pub(crate) use std::path::Path;
pub(crate) use std::sync::Arc;

pub use crate::config::{
    Eagle3Config, EngineConfig, EngineConfigError, EngineDecodeBackend, FinishReason,
    GenerateConstraint, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateToken, GenerateTokenCallback, KvConnectorBackend, KvConnectorConfig, LimitParseError,
    MtpCacheScope, MtpConfig, MtpHiddenLayout, MtpWeightSource, PrioritizedGenerateRequest,
    PrioritizedGenerateResult, ScheduledGenerateArrival, SessionId, SharedKvBinding,
    SharedKvProposerConfig, SpeculativeMode, TokenLogprob, parse_resource_limit,
};
pub use crate::connector_bridge::{ConnectorLookupOutcome, ConnectorStats};
pub(crate) use crate::speculative::{
    LinearEmbedder, LinearLmHead, MtpEmbedder, MtpLmHead, SpeculativeStats,
    load_target_initializer_adapters,
};

mod decode_backend;
mod governor;
mod load;
mod metadata;
mod model;
mod runtime;

pub use governor::{EngineGovernorError, EngineResourceGovernor};
pub use model::Engine;
pub(crate) use decode_backend::*;
pub(crate) use governor::*;
pub(crate) use metadata::*;
pub(crate) use model::*;


fn read_f32_weights(path: &Path) -> anyhow::Result<Vec<f32>> {
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
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(feature = "native-backend")]
fn load_native_shared_kv_proposer(
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
fn mtp_config_from_metadata(
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
fn shared_kv_mode_from_metadata(model_dir: &Path, session: &Session) -> Option<SpeculativeMode> {
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
fn detect_target_hidden_output(session: &Session, hidden_size: usize) -> Option<String> {
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
fn default_connector_model_id(model_directory: &ModelDirectory) -> String {
    model_directory
        .root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "onnx-genai-model".to_string())
}

/// Build the engine's KV connector bridge from generic, model-agnostic config.
fn build_connector_bridge(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_loop::logprob_for_token;
    use crate::logits::ProcessorContext;
    use crate::processors::{
        finish_reason_after_token, select_next_token, select_next_token_with_sampler,
    };
    use crate::sampling::Sampler;

    #[test]
    fn cap_kv_len_uncapped_returns_model_max() {
        assert_eq!(cap_kv_len(32_768, None), 32_768);
    }

    #[test]
    fn cap_kv_len_caps_when_smaller() {
        assert_eq!(cap_kv_len(40_960, Some(512)), 512);
    }

    #[test]
    fn cap_kv_len_ignores_cap_larger_than_model_max() {
        assert_eq!(cap_kv_len(512, Some(40_960)), 512);
    }

    fn test_model_path(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_PATH_ID: AtomicUsize = AtomicUsize::new(0);
        std::env::current_dir()
            .expect("current directory")
            .join(format!(
                ".onnx-genai-{label}-{}-{}.onnx",
                std::process::id(),
                NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed)
            ))
    }

    fn write_scan_model(nodes: &[(&str, &str)]) -> std::path::PathBuf {
        write_scan_model_with_weights(nodes, 0)
    }

    /// Build a valid ONNX model whose graph has the given `(domain, op_type)`
    /// nodes, plus `weight_floats` f32 elements of inline initializer data to
    /// simulate an inline-weight export (the qwen3 case). The prost scan must
    /// skip past this initializer (via `Buf::advance`) rather than reading it.
    fn write_scan_model_with_weights(
        nodes: &[(&str, &str)],
        weight_floats: usize,
    ) -> std::path::PathBuf {
        use onnx::ir::{DataType, Dim, Graph, Node, NodeId, TensorData, WeightRef, static_shape};
        use onnx_std as onnx;
        use prost::Message;

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        for (index, &(domain, op_type)) in nodes.iter().enumerate() {
            let output = graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape([]),
            );
            let mut node = Node::new(NodeId(index as u32), op_type, vec![], vec![output]);
            node.domain = domain.to_string();
            graph.insert_node(node);
            graph.add_output(output);
        }

        if weight_floats > 0 {
            let weight = graph.create_named_value(
                "inline_weight",
                DataType::Float32,
                vec![Dim::from(weight_floats)],
            );
            let bytes = vec![0u8; weight_floats * std::mem::size_of::<f32>()];
            graph.set_initializer(
                weight,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    vec![weight_floats],
                    bytes,
                )),
            );
        }

        let path = test_model_path("control-flow");
        std::fs::write(
            &path,
            onnx::Model::new(graph)
                .to_proto()
                .expect("serialize test model")
                .encode_to_vec(),
        )
        .expect("write test model");
        path
    }

    #[test]
    fn control_flow_scan_ignores_regular_ops() {
        let plain = write_scan_model(&[("", "MatMul"), ("", "Add"), ("", "GroupQueryAttention")]);
        assert!(!model_has_control_flow_nodes(&plain));
        std::fs::remove_file(&plain).ok();
    }

    #[test]
    fn ort_cuda_graph_configuration_is_opt_in() {
        let model = write_scan_model(&[("", "MatMul")]);
        let mut options =
            SessionOptions::with_execution_provider(onnx_genai_ort::ep_selection("cuda"));

        options.graph_capture = false;
        configure_ort_cuda_graph(&mut options, &model);
        assert!(
            !options.graph_capture,
            "ORT capture must stay off by default"
        );

        options.graph_capture = true;
        configure_ort_cuda_graph(&mut options, &model);
        assert!(
            options.graph_capture,
            "an explicit ORT capture opt-in must be preserved"
        );
        std::fs::remove_file(&model).ok();
    }

    #[test]
    fn control_flow_scan_detects_standard_onnx_control_flow_ops() {
        for domain in ["", "ai.onnx"] {
            for op_type in ["If", "Loop", "Scan"] {
                let path = write_scan_model(&[(domain, op_type)]);
                assert!(
                    model_has_control_flow_nodes(&path),
                    "expected standard-domain control-flow op '{domain}:{op_type}' to be detected"
                );
                std::fs::remove_file(&path).ok();
            }
        }
    }

    #[test]
    fn control_flow_scan_ignores_custom_domain_control_flow_names() {
        for op_type in ["If", "Loop", "Scan"] {
            let path = write_scan_model(&[("com.example", op_type)]);
            assert!(
                !model_has_control_flow_nodes(&path),
                "custom-domain op 'com.example:{op_type}' must not disable capture"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn control_flow_scan_conservatively_skips_uninspectable_models() {
        let missing = std::path::Path::new("does-not-exist-onnx-genai.onnx");
        assert!(model_has_control_flow_nodes(missing));

        let garbage = test_model_path("garbage");
        std::fs::write(&garbage, b"not a protobuf").expect("write garbage model");
        assert!(model_has_control_flow_nodes(&garbage));
        std::fs::remove_file(&garbage).ok();
    }

    #[test]
    fn control_flow_scan_reads_nodes_past_large_inline_weights() {
        // Simulate an inline-weight export (like the qwen3 models, whose
        // `model.onnx` embeds >1 GB of weights): the graph carries a large
        // initializer alongside its nodes. The prost scan must still find
        // the control-flow op — and, for a plain graph, must NOT be fooled into
        // conservatively reporting control flow just because the file is large.
        // 4 Mi f32 elements = 16 MiB of inline initializer data.
        let weight_floats = 4 * 1024 * 1024;

        let with_control_flow =
            write_scan_model_with_weights(&[("", "MatMul"), ("", "If")], weight_floats);
        assert!(
            model_has_control_flow_nodes(&with_control_flow),
            "control-flow op must be detected even behind a large inline initializer"
        );
        std::fs::remove_file(&with_control_flow).ok();

        let plain = write_scan_model_with_weights(
            &[("", "MatMul"), ("", "GroupQueryAttention")],
            weight_floats,
        );
        assert!(
            !model_has_control_flow_nodes(&plain),
            "a large inline-weight model without control flow must remain capture-eligible"
        );
        std::fs::remove_file(&plain).ok();
    }

    #[test]
    fn control_flow_scan_conservatively_handles_truncated_control_flow_model() {
        // A control-flow model whose bytes are truncated anywhere must never
        // parse cleanly into a "no control flow" verdict (which would wrongly
        // enable CUDA graph capture and trigger ORT's ~6x slower per-Run path).
        // Truncation either cuts the graph payload (its length header points past
        // EOF -> None) or stops before the graph is ever seen (no-graph -> None),
        // so every prefix (including the empty file) must fall back to
        // conservative `true`.
        let full = write_scan_model(&[("", "MatMul"), ("", "If")]);
        let bytes = std::fs::read(&full).expect("read full model");
        std::fs::remove_file(&full).ok();

        for truncated_len in 0..bytes.len() {
            let truncated = test_model_path(&format!("truncated-{truncated_len}"));
            std::fs::write(&truncated, &bytes[..truncated_len]).expect("write truncated model");
            assert!(
                model_has_control_flow_nodes(&truncated),
                "a truncated control-flow model (len {truncated_len}) must stay conservative"
            );
            std::fs::remove_file(&truncated).ok();
        }
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn backend_and_control_flow_scans_parse_textproto_fixture() -> anyhow::Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/model.onnx.textproto");

        assert!(!model_requires_native_backend(&path)?);
        assert!(!model_has_control_flow_nodes(&path));
        Ok(())
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn auto_backend_detection_reads_onnx_node_types_not_incidental_strings() {
        use onnx_runtime_loader::proto::{
            ModelProto,
            onnx::{GraphProto, NodeProto, OperatorSetIdProto},
        };

        let mut model = ModelProto {
            producer_name: "BlockQuantizedMatMul appears only in metadata".to_string(),
            graph: Some(GraphProto::default()),
            ..ModelProto::default()
        };
        assert!(!model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node.push(NodeProto {
            domain: "pkg.nxrt".to_string(),
            op_type: "BlockQuantizedMatMul".to_string(),
            ..NodeProto::default()
        });
        model.opset_import.push(OperatorSetIdProto {
            domain: "pkg.nxrt".to_string(),
            version: 1,
        });
        assert!(model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node[0].domain = "example.wrong.domain".to_string();
        assert!(!model_proto_requires_native_backend(&model));

        model.graph.as_mut().unwrap().node[0].domain = "pkg.nxrt".to_string();
        model.opset_import[0].version = 2;
        assert!(!model_proto_requires_native_backend(&model));
    }

    #[test]
    fn backend_env_values_are_case_insensitive_and_reject_unknown_values() {
        assert_eq!(
            parse_backend_env("AuTo").unwrap(),
            EngineDecodeBackend::Auto
        );
        assert_eq!(parse_backend_env("ORT").unwrap(), EngineDecodeBackend::Ort);
        assert_eq!(
            parse_backend_env("native").unwrap(),
            EngineDecodeBackend::Native
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || {
                Ok(Some("nAtIvE".to_owned()))
            })
            .unwrap(),
            EngineDecodeBackend::Native
        );

        let error = parse_backend_env("cuda").unwrap_err().to_string();
        assert!(error.contains("ONNX_GENAI_BACKEND"), "{error}");
        assert!(error.contains("auto, ort, native"), "{error}");
    }

    #[test]
    fn explicit_backend_ignores_env_and_auto_honors_it() {
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Ort, || {
                Err(anyhow::anyhow!("unreadable environment value"))
            })
            .unwrap(),
            EngineDecodeBackend::Ort
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Native, || {
                panic!("explicit backend must not read ONNX_GENAI_BACKEND")
            })
            .unwrap(),
            EngineDecodeBackend::Native
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || {
                Ok(Some("ort".to_owned()))
            })
            .unwrap(),
            EngineDecodeBackend::Ort
        );
        assert_eq!(
            requested_decode_backend_with_env(EngineDecodeBackend::Auto, || Ok(None)).unwrap(),
            EngineDecodeBackend::Auto
        );
    }

    #[test]
    fn forced_ort_load_failure_includes_native_switch_hint() {
        let error = augment_backend_error::<()>(
            Err(anyhow::anyhow!("simulated native-only model load failure")),
            EngineDecodeBackend::Ort,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("EngineDecodeBackend::Native"), "{error}");
        assert!(error.contains("ONNX_GENAI_BACKEND=native"), "{error}");
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn forced_native_load_or_run_failure_includes_ort_switch_hint() {
        let error = augment_backend_error::<()>(
            Err(anyhow::anyhow!("simulated native decoder load/run failure")),
            EngineDecodeBackend::Native,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("EngineDecodeBackend::Ort"), "{error}");
        assert!(error.contains("ONNX_GENAI_BACKEND=ort"), "{error}");
    }

    #[cfg(not(feature = "native-backend"))]
    #[test]
    fn forced_native_without_feature_reports_how_to_switch() {
        let error = resolve_decode_backend(Path::new("unused.onnx"), EngineDecodeBackend::Native)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ONNX_GENAI_BACKEND=ort"), "{error}");
        assert!(error.contains("EngineDecodeBackend::Ort"), "{error}");
    }

    fn test_capacities() -> CapacityProviders {
        CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 1_000)),
            host_ram: Arc::new(FixedCapacity::new(2_000, 2_000)),
            disk_spill: None,
        }
    }

    #[test]
    fn governor_handle_reflects_configured_limits_without_a_model() {
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Fraction(0.5),
            host_ram_limit: ResourceLimit::Auto,
            disk_spill_limit: None,
        };
        let governor = EngineResourceGovernor::new_with_capacities(
            limits.clone(),
            true,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
            0,
        )
        .unwrap();
        let snapshot = governor.snapshot();
        assert_eq!(snapshot.configured_limits, limits);
        assert_eq!(snapshot.resolved_limits.vram_bytes, 500);
        assert_eq!(snapshot.derived_budget.total_pages, 5);
        assert_eq!(snapshot.vram.headroom, 500);
        assert_eq!(snapshot.host_ram.used, 0);
        assert_eq!(snapshot.host_ram.limit, 500);
        assert_eq!(snapshot.host_ram.headroom, 500);
        assert_eq!(snapshot.disk_spill, None);

        let outcome = governor.set_vram_limit(ResourceLimit::Bytes(800)).unwrap();
        assert_eq!(outcome.new_limits.vram_bytes, 800);
        assert_eq!(
            governor.snapshot().configured_limits.vram_limit,
            ResourceLimit::Bytes(800)
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn two_engine_governors_keep_independent_host_cache_budgets() {
        let first = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(400),
                ..ResourceLimits::default()
            },
            false,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
            0,
        )
        .unwrap();
        let second = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                host_ram_limit: ResourceLimit::Bytes(900),
                ..ResourceLimits::default()
            },
            false,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
            0,
        )
        .unwrap();

        assert_eq!(
            first.weight_offload_host_cache().configured_budget_bytes(),
            400
        );
        assert_eq!(
            second.weight_offload_host_cache().configured_budget_bytes(),
            900
        );
    }

    #[test]
    fn an_explicit_byte_limit_is_honored_above_the_provisional_capacity() {
        // The device-capacity provider is still a fixed constant, not a probe.
        // Clamping an explicit limit to it would cap every machine at the
        // constant — a 40 GB GPU could not be told about its own memory — so an
        // absolute byte limit is taken as the caller's authoritative statement.
        // Fractions and `auto` remain relative to the reported capacity.
        let limits = ResourceLimits {
            vram_limit: ResourceLimit::Bytes(PROVISIONAL_VRAM_CAPACITY_BYTES + 1),
            host_ram_limit: ResourceLimit::Fraction(0.5),
            disk_spill_limit: Some(ResourceLimit::Auto),
        };
        let governor = EngineResourceGovernor::new(
            limits,
            false,
            ModelKvConfig {
                page_size_bytes: 1,
                tokens_per_page: 1,
            },
            0,
        )
        .unwrap();
        let snapshot = governor.snapshot();
        assert_eq!(
            snapshot.resolved_limits.vram_bytes,
            PROVISIONAL_VRAM_CAPACITY_BYTES + 1,
            "an explicit byte limit must not be clamped to a provisional constant"
        );
        assert_eq!(
            snapshot.resolved_limits.host_ram_bytes,
            PROVISIONAL_HOST_RAM_CAPACITY_BYTES / 2,
            "a fraction stays relative to the reported capacity"
        );
        assert_eq!(
            snapshot.resolved_limits.disk_spill_bytes,
            Some(PROVISIONAL_DISK_CAPACITY_BYTES)
        );
    }

    #[test]
    fn governor_snapshot_reports_usage_limit_and_headroom_for_each_enabled_tier() {
        let capacities = CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 900)),
            host_ram: Arc::new(FixedCapacity::new(2_000, 1_500)),
            disk_spill: Some(Arc::new(FixedCapacity::new(4_000, 3_000))),
        };
        let governor = EngineResourceGovernor::new_with_capacities(
            ResourceLimits {
                vram_limit: ResourceLimit::Bytes(800),
                host_ram_limit: ResourceLimit::Bytes(1_200),
                disk_spill_limit: Some(ResourceLimit::Bytes(3_000)),
            },
            false,
            capacities,
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
            0,
        )
        .unwrap();
        governor.byte_budget().try_reserve(300).unwrap();

        let snapshot = governor.snapshot();
        assert_eq!(
            snapshot.vram,
            onnx_genai_scheduler::TierSnapshot {
                used: 300,
                limit: 800,
                headroom: 500,
            }
        );
        assert_eq!(
            snapshot.host_ram,
            onnx_genai_scheduler::TierSnapshot {
                used: 500,
                limit: 1_200,
                headroom: 700,
            }
        );
        assert_eq!(
            snapshot.disk_spill,
            Some(onnx_genai_scheduler::TierSnapshot {
                used: 1_000,
                limit: 3_000,
                headroom: 2_000,
            })
        );
    }

    #[test]
    fn governor_handle_rejects_disabled_runtime_override() {
        let governor = EngineResourceGovernor::new_with_capacities(
            ResourceLimits::default(),
            false,
            test_capacities(),
            ModelKvConfig {
                page_size_bytes: 100,
                tokens_per_page: 16,
            },
            0,
        )
        .unwrap();
        assert!(matches!(
            governor.set_vram_limit(ResourceLimit::Bytes(800)),
            Err(EngineGovernorError::RuntimeOverrideDisabled)
        ));
    }

    #[test]
    fn token_logprobs_use_log_softmax_and_sorted_top_tokens() {
        let logits = [1.0, f32::NEG_INFINITY, 3.0, 2.0];
        let result = logprob_for_token(&logits, 3, 2);
        let logsumexp = 3.0 + ((1.0_f32 - 3.0).exp() + 1.0 + (2.0_f32 - 3.0).exp()).ln();

        assert_eq!(result.token_id, 3);
        assert_eq!(result.logprob, 2.0 - logsumexp);
        assert!(result.logprob <= 0.0);
        assert!(result.top.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        assert!(result.top.iter().any(|(token_id, _)| *token_id == 3));
        assert!(result.top.iter().all(|(token_id, _)| *token_id != 1));
    }

    #[test]
    fn processor_chain_uses_documented_order() {
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            stop_sequences: vec![StopSequence::Tokens(vec![42])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "stop_sequence",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
    }

    #[test]
    fn processor_chain_includes_json_constraint_before_sampling_filters() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        let tokenizer = Tokenizer::from_file(&fixture)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let options = GenerateOptions {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };

        let chain = build_processor_chain(&options, Some(&tokenizer))?;

        assert_eq!(
            chain.names(),
            vec![
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
                "json_constraint",
                "temperature",
                "top_k",
                "top_p",
                "min_p"
            ]
        );
        Ok(())
    }

    #[test]
    fn greedy_selection_uses_argmax_after_processors() {
        let options = GenerateOptions {
            greedy: true,
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.0),
            2
        );
    }

    #[test]
    fn sampled_selection_can_pick_non_argmax() {
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 0.0];
        assert_eq!(
            select_next_token(&mut logits, &context, &options, &chain, 0.75),
            1
        );
    }

    struct LastTokenSampler;

    impl Sampler for LastTokenSampler {
        fn sample(&mut self, logits: &[f32], _context: &ProcessorContext) -> TokenId {
            logits.len().saturating_sub(1) as TokenId
        }

        fn name(&self) -> &str {
            "last_token"
        }
    }

    #[test]
    fn custom_sampler_can_select_after_default_processors() {
        let options = GenerateOptions {
            top_k: 2,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext::default();
        let mut logits = vec![0.0, 2.0, 4.0, 3.0];
        let mut sampler = LastTokenSampler;

        assert_eq!(
            select_next_token_with_sampler(&mut logits, &context, &chain, &mut sampler),
            3
        );
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn default_processor_chain_is_empty_for_unchanged_defaults() {
        let options = GenerateOptions::default();
        let chain = build_processor_chain(&options, None).unwrap();
        assert!(chain.names().is_empty());
    }

    #[test]
    fn finish_reason_detects_eos_before_stop_sequence() {
        let options = GenerateOptions {
            eos_token_id: Some(7),
            stop_sequences: vec![StopSequence::Tokens(vec![7])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![7],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(7, &options, &chain, &context),
            Some(FinishReason::EosToken)
        );
    }

    #[test]
    fn finish_reason_detects_stop_sequence() {
        let options = GenerateOptions {
            stop_sequences: vec![StopSequence::Tokens(vec![2, 3])],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None).unwrap();
        let context = ProcessorContext {
            generated_tokens: vec![1, 2, 3],
            ..Default::default()
        };
        assert_eq!(
            finish_reason_after_token(3, &options, &chain, &context),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn json_constraint_defers_stop_until_value_is_complete() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            stop_sequences: vec![StopSequence::Text("}".to_string())],
            ..Default::default()
        };
        let chain_options = GenerateOptions {
            stop_sequences: options.stop_sequences.clone(),
            ..Default::default()
        };
        let chain = build_processor_chain(&chain_options, None).unwrap();
        let incomplete = ProcessorContext {
            generated_text: "{\"value\":".to_string(),
            ..Default::default()
        };
        let complete = ProcessorContext {
            generated_text: "{\"value\":1}".to_string(),
            ..Default::default()
        };

        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &incomplete),
            None
        );
        assert_eq!(
            finish_reason_after_token(1, &options, &chain, &complete),
            Some(FinishReason::StopSequence { index: 0 })
        );
    }

    #[test]
    fn incomplete_json_constraint_rejects_length_finishes() {
        let options = GenerateOptions {
            constraint: Some(GenerateConstraint::Json),
            ..Default::default()
        };
        for reason in [FinishReason::MaxTokens, FinishReason::Length] {
            let error = ensure_constrained_finish(&options, "{\"value\":", reason).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("stopped before a complete JSON value")
            );
        }
        ensure_constrained_finish(&options, "{\"value\":1}", FinishReason::MaxTokens).unwrap();
        ensure_constrained_finish(&options, "", FinishReason::EosToken).unwrap();
    }

    #[test]
    fn common_prefix_len_stops_before_rejected_draft_token() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(common_prefix_len(&[7], &[8]), 0);
    }

    #[test]
    fn tiny_fixture_generates_requested_tokens_end_to_end() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 3);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_returns_opt_in_per_token_logprobs() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir_with_session_options(
            &fixture,
            EngineConfig::default(),
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.top_logprobs = Some(3);

        let result = engine.generate(request)?;
        let logprobs = result.logprobs.as_ref().expect("logprobs requested");

        assert_eq!(logprobs.len(), result.token_ids.len());
        for (token_id, token_logprob) in result.token_ids.iter().zip(logprobs) {
            assert_eq!(*token_id, token_logprob.token_id);
            assert!(token_logprob.logprob <= 0.0);
            assert!(
                token_logprob
                    .top
                    .windows(2)
                    .all(|pair| pair[0].1 >= pair[1].1)
            );
            assert!(
                token_logprob
                    .top
                    .iter()
                    .any(|(top_token_id, _)| top_token_id == token_id)
            );
        }

        let mut disabled = GenerateRequest::new("hello");
        disabled.options.max_new_tokens = 1;
        disabled.options.temperature = 0.0;
        disabled.options.stop_on_eos = false;
        assert!(engine.generate(disabled)?.logprobs.is_none());
        Ok(())
    }

    #[test]
    fn tiny_fixture_uses_past_present_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                ..
            }
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![22, 22, 20]);
        Ok(())
    }

    #[test]
    fn scatter_fixture_uses_static_cache_decode_session_with_stable_greedy_output()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm-scatter")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(matches!(
            engine.decode_path,
            ModelDecodePath::StaticCache { max_len } if max_len > 0
        ));
        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids, vec![23, 15, 28]);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        Ok(())
    }

    #[test]
    fn tiny_fixture_speculative_matches_plain_greedy_with_k_gt_one() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut speculative = Engine::from_dir(
            &fixture,
            EngineConfig {
                draft_model: Some(fixture.clone()),
                num_speculative_tokens: 3,
                ..Default::default()
            },
        )?;

        let mut request = GenerateRequest::new("hello");
        request.options.max_new_tokens = 6;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.num_speculative_tokens = Some(3);

        let baseline_result = baseline.generate(request.clone())?;
        let speculative_result = speculative.generate(request)?;

        assert_eq!(speculative_result.token_ids, baseline_result.token_ids);
        assert_eq!(
            speculative_result.finish_reason,
            baseline_result.finish_reason
        );
        assert_eq!(speculative_result.token_ids.len(), 6);
        Ok(())
    }

    #[test]
    fn tiny_fixture_stops_at_explicit_context_length_without_ort_error() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate(request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_stops_at_explicit_context_length_without_ort_error()
    -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3]));
        request.options.max_new_tokens = 32;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;
        request.options.max_context = Some(16);

        let result = engine.generate_in_session(session_id, request)?;

        assert_eq!(result.token_ids.len(), 13);
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert_eq!(engine.session_token_count(session_id)?, 16);
        engine.close_session(session_id)?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_session_persists_context_across_turns() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, EngineConfig::default())?;
        let session_id = engine.create_session()?;

        let mut first = GenerateRequest::new("hello");
        first.options.max_new_tokens = 2;
        first.options.temperature = 0.0;
        first.options.stop_on_eos = false;
        let first_result = engine.generate_in_session(session_id, first)?;
        let first_count = engine.session_token_count(session_id)?;

        let mut second = GenerateRequest::new(" world");
        second.options.max_new_tokens = 2;
        second.options.temperature = 0.0;
        second.options.stop_on_eos = false;
        let second_result = engine.generate_in_session(session_id, second)?;
        let second_count = engine.session_token_count(session_id)?;

        assert_eq!(first_result.token_ids.len(), 2);
        assert_eq!(second_result.token_ids.len(), 2);
        assert!(second_count > first_count);
        assert!(engine.sessions[&session_id].kv_token_count > 0);
        engine.close_session(session_id)?;
        assert!(engine.sessions.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires ONNX_GENAI_FIM_MODEL_DIR to point at a FIM-capable coder model"]
    fn fim_generation_runs_with_fim_capable_model() -> anyhow::Result<()> {
        let Some(model_dir) = onnx_genai_runtime_config::runtime_config()
            .fim_model_dir
            .as_deref()
        else {
            eprintln!("set ONNX_GENAI_FIM_MODEL_DIR to a Qwen2.5-Coder/StarCoder-style model");
            return Ok(());
        };
        let mut engine = Engine::from_dir(model_dir, EngineConfig::default())?;
        assert!(
            engine.fim_config().is_some(),
            "model tokenizer_config.json must expose recognized FIM tokens"
        );

        let mut options = GenerateOptions {
            max_new_tokens: 16,
            temperature: 0.0,
            ..Default::default()
        };
        options
            .stop_sequences
            .push(StopSequence::Text("\n\n".into()));

        let result =
            engine.generate_fim("fn add(a: i32, b: i32) -> i32 {\n    ", "\n}", options)?;

        assert!(!result.token_ids.is_empty());
        Ok(())
    }

    fn local_tiered_engine_config(chunk_size: usize) -> EngineConfig {
        EngineConfig {
            kv_connector: KvConnectorConfig {
                backend: KvConnectorBackend::LocalTiered(onnx_genai_kv::LocalTieredConfig {
                    chunk_size,
                    page_size: chunk_size,
                    ..onnx_genai_kv::LocalTieredConfig::default()
                }),
                chunk_size,
                ..KvConnectorConfig::default()
            },
            ..EngineConfig::default()
        }
    }

    #[test]
    fn null_connector_default_leaves_behavior_unchanged() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
        assert!(!baseline.connector.is_active());

        let mut request =
            GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3, 5, 6, 7, 8, 9]));
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let result = baseline.generate(request)?;
        // With the default Null connector, no external activity happens at all.
        assert_eq!(baseline.last_connector_stats(), ConnectorStats::default());
        assert_eq!(result.token_ids.len(), 3);
        Ok(())
    }

    #[test]
    fn local_tiered_connector_stores_prefill_chunks() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, local_tiered_engine_config(2))?;
        assert!(engine.connector.is_active());

        let mut request =
            GenerateRequest::new(GeneratePrompt::TokenIds(vec![2, 4, 3, 5, 6, 7, 8, 9]));
        request.options.max_new_tokens = 3;
        request.options.temperature = 0.0;
        request.options.stop_on_eos = false;

        let baseline_ids = {
            let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
            baseline.generate(request.clone())?.token_ids
        };

        let result = engine.generate(request)?;

        // Store-after-prefill ran: complete chunks were pushed to the connector.
        assert!(
            engine.last_connector_stats().stores > 0,
            "expected connector store path to push chunks, got {:?}",
            engine.last_connector_stats()
        );
        // The store path is a pure side effect for a first, unseen request:
        // nothing is resident to fetch, so output matches full recompute.
        assert_eq!(result.token_ids, baseline_ids);
        Ok(())
    }

    #[test]
    fn local_tiered_connector_fetch_reuse_is_token_identical() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let mut engine = Engine::from_dir(&fixture, local_tiered_engine_config(2))?;

        // Request 1 populates the connector with the prompt's KV chunks.
        let prompt = vec![10, 11, 12, 13, 14, 15];
        let mut warm = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        warm.options.max_new_tokens = 1;
        warm.options.temperature = 0.0;
        warm.options.stop_on_eos = false;
        engine.generate(warm)?;
        assert!(engine.last_connector_stats().stores > 0);

        // Drop the in-process caches so the connector is the ONLY source of
        // cross-session reuse — simulating a fresh process / different node that
        // shares nothing but the connector.
        engine.token_prefix_cache.clear();
        engine.prefix_cache = PrefixCache::new();

        // Request 2 shares the whole prefix (≥ 1 chunk) with request 1.
        let mut reuse = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
        reuse.options.max_new_tokens = 4;
        reuse.options.temperature = 0.0;
        reuse.options.stop_on_eos = false;
        let reuse_result = engine.generate(reuse)?;
        let stats = engine.last_connector_stats();

        // (a) Prefill was genuinely shortened: real KV bytes were fetched and
        // injected into the runner.
        assert!(
            stats.fetched_tokens > 0 && stats.chunk_hits > 0,
            "expected connector fetch to materialize KV, got {stats:?}"
        );
        // At least one prompt token is always left to feed the decoder.
        assert!(stats.fetched_tokens < prompt.len());

        // (b) Output is byte-for-byte identical to full recompute with a Null
        // connector — proving the materialized KV is correct, not just present.
        let baseline_ids = {
            let mut baseline = Engine::from_dir(&fixture, EngineConfig::default())?;
            let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt.clone()));
            request.options.max_new_tokens = 4;
            request.options.temperature = 0.0;
            request.options.stop_on_eos = false;
            baseline.generate(request)?.token_ids
        };
        assert_eq!(
            reuse_result.token_ids, baseline_ids,
            "connector-reuse output must match full recompute exactly"
        );
        Ok(())
    }
}
