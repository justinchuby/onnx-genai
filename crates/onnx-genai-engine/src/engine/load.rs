//! `Engine` construction: ORT and native model directory constructors.

use super::*;

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::from_dir_impl(model_dir, config, SessionOptions::default(), false)
    }

    /// Load a model from a directory with explicit ORT session options.
    pub fn from_dir_with_session_options(
        model_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        Self::from_dir_impl(model_dir, config, session_options, true)
    }

    fn from_dir_impl(
        model_dir: &Path,
        mut config: EngineConfig,
        mut session_options: SessionOptions,
        session_options_are_programmatic: bool,
    ) -> anyhow::Result<Self> {
        let model_directory = {
            let _span = onnx_genai_ort::prof_span!("engine.resolve_model_directory");
            let package_selection = package_selection_from_session_options(&session_options);
            ModelDirectory::load_with_package_selection(model_dir, &package_selection)
                .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {e}"))?
        };
        let metadata_hints = load_model_metadata_hints(&model_directory.model_path)?;
        report_metadata_hint_warnings(&metadata_hints);
        if metadata_hints.has_errors() {
            anyhow::bail!(
                "ONNX model metadata contains conflicting forced placement hints; remove one of the contradictory onnx_runtime.device declarations"
            );
        }
        apply_model_memory_hints(&mut config, &metadata_hints)?;
        apply_model_placement_hints(
            &mut session_options,
            &metadata_hints,
            session_options_are_programmatic,
        )?;
        let decode_backend = {
            let _span = onnx_genai_ort::prof_span!("engine.resolve_decode_backend");
            resolve_decode_backend(&model_directory.model_path, config.decode_backend)?
        };
        if decode_backend == EngineDecodeBackend::Native {
            return augment_backend_error(
                Self::from_native_model_directory(
                    model_directory,
                    config,
                    &session_options,
                    metadata_hints,
                ),
                EngineDecodeBackend::Native,
            );
        }

        // ORT CUDA graph capture is opt-in: it fails with unconstructed OrtValue
        // outputs on some Foundry exports. SessionOptions still honors an explicit
        // ONNX_GENAI_CUDA_GRAPH=1 request; native whole-step capture is separate.
        configure_ort_cuda_graph(&mut session_options, &model_directory.model_path);

        let environment = {
            let _span = onnx_genai_ort::prof_span!("engine.ort_environment");
            Environment::new("onnx-genai-engine")
                .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {e}"))?
        };
        let session = {
            let _span = onnx_genai_ort::prof_span!("engine.ort_session_load");
            augment_backend_error(
                Session::new(
                    &environment,
                    &model_directory.model_path,
                    session_options.clone(),
                )
                .map_err(|e| anyhow::anyhow!("Failed to load ORT session: {e}")),
                EngineDecodeBackend::Ort,
            )?
        };

        // Stage: metadata and decode-path resolution.
        let MetadataResolution {
            metadata,
            metadata_max_context,
            decode_path,
        } = resolve_metadata_and_decode_path(&model_directory, &session)?;

        let tokenizer = {
            let _span = onnx_genai_ort::prof_span!("engine.tokenizer_load");
            Tokenizer::from_file(&model_directory.tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
        };
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        let kv_model = {
            let _span = onnx_genai_ort::prof_span!("engine.kv_model_info");
            infer_kv_model_info(
                &session,
                metadata.model.as_ref().and_then(|model| model.io.as_ref()),
                config.page_size,
                config.kv_cache_dtype,
            )?
        };

        // Stage: resource governor and batch scheduler.
        let (governor, scheduler) =
            build_governor_and_scheduler(&config, &model_directory, kv_model.as_ref())?;

        // Stage: draft-model loading. Kept before KV-cache allocation to preserve
        // the original constructor's fallible-step ordering.
        let draft = load_draft_model(
            &config,
            &environment,
            &session_options,
            metadata_max_context,
        )?;

        // Stage: runtime KV-cache allocation.
        let kv_cache = allocate_kv_cache(&config, kv_model.as_ref());

        // Stage: speculative-assistant loading (mode resolution then per-mode heads).
        let (speculative_mode, resolved_mtp_config) = resolve_speculative_mode(
            config.speculative_mode.clone(),
            &metadata,
            &model_directory,
            &session,
            draft.is_some(),
        )?;
        let mtp = load_mtp_model(
            resolved_mtp_config,
            &session,
            &environment,
            &session_options,
            &model_directory,
        )?;
        let eagle3 =
            load_eagle3_model(&speculative_mode, &session, &environment, &session_options)?;
        let shared_kv_proposer =
            load_shared_kv_proposer(&speculative_mode, &session, &environment, &session_options)?;

        let connector = {
            let _span = onnx_genai_ort::prof_span!("engine.connector_bridge");
            build_connector_bridge(&config.kv_connector, &model_directory, kv_model.as_ref())?
        };

        Ok(Self {
            decode_backend,
            metadata,
            metadata_hints,
            kv_cache,
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model,
            decode_path,
            scheduler,
            governor,
            sessions: HashMap::new(),
            _environment: Some(environment),
            session: Some(Box::new(session)),
            #[cfg(feature = "native-backend")]
            native_session: None,
            #[cfg(feature = "native-backend")]
            native_sessions: HashMap::new(),
            #[cfg(feature = "native-backend")]
            native_active_session: None,
            #[cfg(feature = "native-backend")]
            native_session_counter: 0,
            #[cfg(feature = "native-backend")]
            native_shared_kv_proposer: None,
            draft,
            mtp,
            eagle3,
            shared_kv_proposer,
            tokenizer,
            fim_config,
            num_speculative_tokens: config.num_speculative_tokens.max(1),
            speculative_mode,
            last_speculative_stats: SpeculativeStats::default(),
            connector,
        })
    }

    #[cfg(feature = "native-backend")]
    fn from_native_model_directory(
        model_directory: ModelDirectory,
        config: EngineConfig,
        session_options: &SessionOptions,
        metadata_hints: MetadataHints,
    ) -> anyhow::Result<Self> {
        if config.draft_model.is_some() || !matches!(config.speculative_mode, SpeculativeMode::None)
        {
            anyhow::bail!(
                "native decoder backend does not yet support speculative, MTP, EAGLE-3, or shared-KV generation"
            );
        }
        if !matches!(&config.kv_connector.backend, KvConnectorBackend::Null) {
            anyhow::bail!("native decoder backend does not yet support external KV connectors");
        }
        let native_device =
            resolve_native_decode_device(config.native_device.clone(), session_options)?;

        let metadata = {
            let _span = onnx_genai_ort::prof_span!("engine.metadata_load");
            if let Some(metadata_path) = &model_directory.metadata_path {
                onnx_genai_metadata::load_metadata(metadata_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load metadata: {e}"))?
            } else if let Some(compat) = genai_config_compat_metadata_from_model_path(
                model_directory.genai_config_path.as_deref(),
                &model_directory.model_path,
            )? {
                compat
            } else {
                tracing::warn!("No inference metadata found, using defaults");
                default_inference_metadata()
            }
        };
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {unsupported:?}");
        }

        let tokenizer = {
            let _span = onnx_genai_ort::prof_span!("engine.tokenizer_load");
            Tokenizer::from_file(&model_directory.tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
        };
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        let governor_kv_config = governor_kv_config(None, &config)?;
        let governor = {
            let _span = onnx_genai_ort::prof_span!("engine.resource_governor");
            EngineResourceGovernor::new(
                config.limits.clone(),
                config.allow_runtime_override,
                governor_kv_config,
                onnx_genai_ort::model_weight_bytes(&model_directory.model_path),
            )
            .map_err(|error| anyhow::anyhow!("Failed to initialize Resource Governor: {error}"))?
        };
        let mut scheduler_config = config.scheduler.clone();
        if scheduler_config.bytes_per_token.is_none() {
            scheduler_config.bytes_per_token = Some(
                governor_kv_config
                    .page_size_bytes
                    .div_ceil(governor_kv_config.tokens_per_page),
            );
        }
        let scheduler = Scheduler::with_byte_budget(scheduler_config, governor.byte_budget());
        let connector = {
            let _span = onnx_genai_ort::prof_span!("engine.connector_bridge");
            build_connector_bridge(&config.kv_connector, &model_directory, None)?
        };
        let startup_trace = onnx_genai_ort::profile::tracing_enabled()
            .then(crate::runtime_trace::context)
            .flatten();
        let native_session = {
            let _span = onnx_genai_ort::prof_span!("engine.native_session_load");
            crate::native_decode::NativeDecodeSession::load_with_weight_offload_host_cache(
                &model_directory.model_path,
                native_device.clone(),
                governor.weight_offload_host_cache(),
                metadata.model.as_ref().and_then(|model| model.io.as_ref()),
                metadata
                    .model
                    .as_ref()
                    .and_then(|model| model.max_sequence_length),
                crate::decode::key_sequence_lengths_policy(&metadata),
                config.decode_precision,
            )
            .map_err(|error| anyhow::anyhow!("Failed to load native decoder session: {error:#}"))?
        };
        let mut native_session = native_session;
        // Join the runtime and its execution providers to the engine's timeline.
        // Without this their spans are recorded into a disabled context and
        // `native.session_run` exports as one opaque block.
        let trace = startup_trace.or_else(crate::runtime_trace::context);
        if let Some(trace) = trace {
            native_session.set_trace_context(trace);
        }
        let (native_shared_kv_proposer, speculative_mode) =
            load_native_shared_kv_proposer(&metadata, &model_directory.root, native_device)?;
        let environment = {
            let _span = onnx_genai_ort::prof_span!("engine.ort_environment");
            Environment::new("onnx-genai-engine")
                .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {e}"))?
        };

        Ok(Self {
            decode_backend: EngineDecodeBackend::Native,
            metadata,
            metadata_hints,
            kv_cache: PagedKvCache::new(config.page_size, config.num_gpu_pages),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::Legacy,
            scheduler,
            governor,
            sessions: HashMap::new(),
            session: None,
            native_session: Some(native_session),
            native_sessions: HashMap::new(),
            native_active_session: None,
            native_session_counter: 0,
            native_shared_kv_proposer,
            draft: None,
            mtp: None,
            eagle3: None,
            shared_kv_proposer: None,
            tokenizer,
            fim_config,
            num_speculative_tokens: config.num_speculative_tokens.max(1),
            speculative_mode,
            last_speculative_stats: SpeculativeStats::default(),
            connector,
            _environment: Some(environment),
        })
    }

    #[cfg(not(feature = "native-backend"))]
    fn from_native_model_directory(
        _model_directory: ModelDirectory,
        _config: EngineConfig,
        _session_options: &SessionOptions,
        _metadata_hints: MetadataHints,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "native decoder backend requires building onnx-genai-engine with the 'native-backend' feature"
        )
    }
}

fn package_selection_from_session_options(
    session_options: &SessionOptions,
) -> onnx_genai_ort::ModelPackageSelection {
    let execution_provider = session_options.execution_providers.first().map(|provider| {
        match provider.selection.name.as_str() {
            "cpu" => "CPUExecutionProvider".to_string(),
            "cuda" => "CUDAExecutionProvider".to_string(),
            "coreml" | "core-ml" | "core_ml" => "CoreMLExecutionProvider".to_string(),
            "webgpu" | "web-gpu" | "web_gpu" => "WebGpuExecutionProvider".to_string(),
            "metal" => "MlxExecutionProvider".to_string(),
            name if name.ends_with("ExecutionProvider") => name.to_string(),
            name => format!("{name}ExecutionProvider"),
        }
    });
    onnx_genai_ort::ModelPackageSelection {
        execution_provider,
        ..Default::default()
    }
}

/// Resolved model inference metadata plus the decode-path derived from it.
/// Produced by [`resolve_metadata_and_decode_path`] as the first construction stage.
struct MetadataResolution {
    metadata: InferenceMetadata,
    metadata_max_context: Option<usize>,
    decode_path: ModelDecodePath,
}

fn resolve_metadata_and_decode_path(
    model_directory: &ModelDirectory,
    session: &Session,
) -> anyhow::Result<MetadataResolution> {
    // Resolve inference metadata. Our own `inference_metadata.yaml` is the
    // canonical source of truth. When a model ships without it (e.g. the
    // onnxruntime-genai / Foundry Local models, which carry only a
    // `genai_config.json`), fall back to converting that config into native
    // metadata so share-buffer-capable GQA models still get the O(1)/token
    // decode path instead of the growing rebind path.
    let metadata = {
        let _span = onnx_genai_ort::prof_span!("engine.metadata_load");
        if let Some(metadata_path) = &model_directory.metadata_path {
            onnx_genai_metadata::load_metadata(metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {e}"))?
        } else if let Some(compat) =
            genai_config_compat_metadata(model_directory.genai_config_path.as_deref(), session)?
        {
            tracing::info!(
                "No inference_metadata.yaml found; derived inference metadata from genai_config.json (onnxruntime-genai compatibility)"
            );
            compat
        } else {
            tracing::warn!("No inference metadata found, using defaults");
            default_inference_metadata()
        }
    };

    // Validate capabilities
    let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
    if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
        anyhow::bail!("Unsupported capabilities: {unsupported:?}");
    }

    // Optional explicit cap on runtime-owned KV growth. Foundry /
    // onnxruntime-genai `genai_config.json` models advertise the model's full
    // `context_length` (e.g. 32k-131k) as their max sequence length. The
    // shared-buffer decode path pre-allocates at an initial power-of-two bucket
    // (256 tokens by default, overridden by `ONNX_GENAI_KV_MIN_BUCKET`) and grows
    // on demand up to the model's declared `max_length`; it does not pre-allocate
    // the full context. `ONNX_GENAI_KV_MAX_LEN` caps growth below the model
    // maximum, mirroring the native path's `ONNX_GENAI_CUDA_KV_MAX_LEN`.
    // Unset = model metadata is the factual ceiling.
    let kv_shared_buffer_cap = shared_buffer_cap_from_env();
    let metadata_max_context = metadata
        .model
        .as_ref()
        .and_then(|model| model.max_sequence_length)
        .map(|max_len| cap_kv_len(max_len, kv_shared_buffer_cap));
    // Our own inference metadata (inference_metadata.yaml), not
    // onnxruntime-genai's genai_config.json, drives the runtime-owned
    // share-buffer KV path for GQA models.
    let shared_kv_max_len = crate::decode::shared_kv_buffer_len_from_metadata(&metadata)
        .map(|max_len| cap_kv_len(max_len, kv_shared_buffer_cap));
    let sliding_window = crate::decode::sliding_window_from_metadata(&metadata)?;
    let sink_tokens = crate::decode::sink_tokens_from_metadata(&metadata);
    let decode_path = {
        let _span = onnx_genai_ort::prof_span!("engine.detect_decode_path");
        detect_model_decode_path(
            session,
            metadata.model.as_ref().and_then(|model| model.io.as_ref()),
            metadata_max_context,
            shared_kv_max_len,
            sliding_window,
            sink_tokens,
        )?
    };
    Ok(MetadataResolution {
        metadata,
        metadata_max_context,
        decode_path,
    })
}

fn build_governor_and_scheduler(
    config: &EngineConfig,
    model_directory: &ModelDirectory,
    kv_model: Option<&KvModelInfo>,
) -> anyhow::Result<(EngineResourceGovernor, Scheduler)> {
    let governor_kv_config = governor_kv_config(kv_model, config)?;
    let governor = {
        let _span = onnx_genai_ort::prof_span!("engine.resource_governor");
        EngineResourceGovernor::new(
            config.limits.clone(),
            config.allow_runtime_override,
            governor_kv_config,
            onnx_genai_ort::model_weight_bytes(&model_directory.model_path),
        )
        .map_err(|error| anyhow::anyhow!("Failed to initialize Resource Governor: {error}"))?
    };
    let mut scheduler_config = config.scheduler.clone();
    if scheduler_config.bytes_per_token.is_none() {
        scheduler_config.bytes_per_token = Some(
            governor_kv_config
                .page_size_bytes
                .div_ceil(governor_kv_config.tokens_per_page),
        );
    }
    let scheduler = Scheduler::with_byte_budget(scheduler_config, governor.byte_budget());
    Ok((governor, scheduler))
}

fn load_draft_model(
    config: &EngineConfig,
    environment: &Environment,
    session_options: &SessionOptions,
    metadata_max_context: Option<usize>,
) -> anyhow::Result<Option<DraftModel>> {
    let draft = if let Some(draft_model_path) = &config.draft_model {
        let draft_directory = ModelDirectory::load(draft_model_path)
            .map_err(|e| anyhow::anyhow!("Failed to resolve draft model directory: {e}"))?;
        let draft_io = draft_directory
            .metadata_path
            .as_deref()
            .map(onnx_genai_metadata::load_metadata)
            .transpose()?
            .and_then(|metadata| metadata.model.and_then(|model| model.io));
        let draft_session = Session::new(
            environment,
            &draft_directory.model_path,
            session_options.clone(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to load draft ORT session: {e}"))?;
        let draft_decode_path =
            // Draft models are loaded with sliding_window=None and sink_tokens=0:
            // draft architectures are typically distinct from the target (e.g. a
            // smaller model with a full KV cache) and declare their own attention
            // constraints through their own inference metadata. Even if the target
            // uses SWA + attention sinks (sink_tokens > 0), propagating the
            // target's sink_tokens to the draft would be a silent no-op — all
            // sink/window management is gated on `sliding_window.is_some()`, which
            // is None here — and it would mask a future bug if a windowed draft
            // path were introduced without explicitly loading draft metadata.
            // If a draft model needs its own SWA + sinks, load its
            // inference_metadata.yaml and pass the values from there.
            detect_model_decode_path(&draft_session, None, metadata_max_context, None, None, 0)?;
        let draft_kv_model = infer_kv_model_info(
            &draft_session,
            draft_io.as_ref(),
            config.page_size,
            onnx_genai_kv::KvDType::F32,
        )?;
        let draft_kv_cache = if let Some(kv_model) = &draft_kv_model {
            PagedKvCache::new_with_layer_tensor_configs(
                kv_model.tensor_config.page_size,
                kv_model.tensor_config.dtype,
                kv_model.layer_configs.clone(),
                config.num_gpu_pages,
            )
        } else {
            PagedKvCache::new(config.page_size, config.num_gpu_pages)
        };
        Some(DraftModel {
            session: Box::new(draft_session),
            decode_path: draft_decode_path,
            io: draft_io,
            kv_model: draft_kv_model,
            kv_cache: draft_kv_cache,
        })
    } else {
        None
    };
    Ok(draft)
}

fn allocate_kv_cache(config: &EngineConfig, kv_model: Option<&KvModelInfo>) -> PagedKvCache {
    if let Some(kv_model) = kv_model {
        let mut span = onnx_genai_ort::prof_span!("engine.kv_cache_alloc");
        span.set_arg("page_size", kv_model.tensor_config.page_size as u64);
        span.set_arg("num_gpu_pages", config.num_gpu_pages as u64);
        span.set_arg("layers", kv_model.layer_configs.len() as u64);
        // The paged tensor layout is derived from present-KV outputs: each
        // layer has key/value tensors shaped like [batch, kv_heads, seq, head_dim].
        // Per-layer geometry (heterogeneous head_dim across layers, e.g. the
        // Gemma-4 sliding/full split) is fed from the model's own KV output
        // shapes so mixed-geometry models page correctly.
        PagedKvCache::new_with_layer_tensor_configs(
            kv_model.tensor_config.page_size,
            kv_model.tensor_config.dtype,
            kv_model.layer_configs.clone(),
            config.num_gpu_pages,
        )
    } else {
        let mut span = onnx_genai_ort::prof_span!("engine.kv_cache_alloc");
        span.set_arg("page_size", config.page_size as u64);
        span.set_arg("num_gpu_pages", config.num_gpu_pages as u64);
        PagedKvCache::new(config.page_size, config.num_gpu_pages)
    }
}

fn resolve_speculative_mode(
    requested_mode: SpeculativeMode,
    metadata: &InferenceMetadata,
    model_directory: &ModelDirectory,
    session: &Session,
    draft_present: bool,
) -> anyhow::Result<(SpeculativeMode, Option<ResolvedMtpConfig>)> {
    let (speculative_mode, resolved_mtp_config) = match requested_mode {
        SpeculativeMode::None if draft_present => (SpeculativeMode::DraftModel, None),
        // No explicit mode: adopt a shared-KV draft proposer advertised by
        // the model's own inference metadata, if the target exposes an f32
        // hidden output the assistant can be seeded from.
        SpeculativeMode::None => {
            if let Some(config) =
                mtp_config_from_metadata(metadata, &model_directory.root, session)?
            {
                (
                    SpeculativeMode::Mtp(config.public_config.clone()),
                    Some(config),
                )
            } else {
                (
                    shared_kv_mode_from_metadata(&model_directory.root, session)
                        .unwrap_or(SpeculativeMode::None),
                    None,
                )
            }
        }
        SpeculativeMode::Mtp(config) => (
            SpeculativeMode::Mtp(config.clone()),
            Some(ResolvedMtpConfig::from_manual(config)),
        ),
        mode => (mode, None),
    };
    if let SpeculativeMode::PromptLookup { ngram, max_tokens } = &speculative_mode
        && (*ngram == 0 || *max_tokens == 0)
    {
        anyhow::bail!("prompt-lookup ngram and max_tokens must be greater than zero");
    }
    Ok((speculative_mode, resolved_mtp_config))
}

fn load_mtp_model(
    resolved_mtp_config: Option<ResolvedMtpConfig>,
    session: &Session,
    environment: &Environment,
    session_options: &SessionOptions,
    model_directory: &ModelDirectory,
) -> anyhow::Result<Option<MtpModel>> {
    let mtp = if let Some(mtp_config) = resolved_mtp_config {
        validate_resolved_mtp_config(&mtp_config)?;
        if mtp_config.cache_scope == MtpCacheScope::AcceptedPrefix {
            anyhow::bail!(
                "MTP kv_mode accepted_prefix is declared but not executable: the frozen Mobius contract does not define correction-token/cache alignment"
            );
        }
        let hidden_output = session
            .outputs()
            .iter()
            .find(|output| output.name == mtp_config.public_config.target_hidden_output)
            .with_context(|| {
                format!(
                    "MTP target model must expose hidden-state output '{}'",
                    mtp_config.public_config.target_hidden_output
                )
            })?;
        if !matches!(
            hidden_output.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            anyhow::bail!(
                "MTP target hidden-state output '{}' must be Float32, Float16, or BFloat16, got {:?}",
                hidden_output.name,
                hidden_output.dtype
            );
        }
        match mtp_config.target_hidden_layout {
            MtpHiddenLayout::Bsh
                if hidden_output.shape.len() == 3
                    && hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                        == Some(mtp_config.public_config.hidden_size as i64) => {}
            MtpHiddenLayout::Bshc
                if hidden_output.shape.len() == 4
                    && hidden_output.shape[2] == mtp_config.hc_mult as i64
                    && hidden_output.shape[3] == mtp_config.public_config.hidden_size as i64 => {}
            _ => anyhow::bail!(
                "MTP target hidden-state output '{}' shape {:?} does not match configured {:?} with hc_mult {} and hidden size {}",
                hidden_output.name,
                hidden_output.shape,
                mtp_config.target_hidden_layout,
                mtp_config.hc_mult,
                mtp_config.public_config.hidden_size
            ),
        }
        let head_session = Session::new(
            environment,
            &mtp_config.public_config.head_model,
            session_options.clone(),
        )
        .map_err(|error| anyhow::anyhow!("Failed to load MTP head: {error}"))?;
        let decode_options = onnx_genai_ort::MtpDecodeOptions {
            kv_mode: mtp_config.public_config.kv_mode,
            batch_size: 1,
            hc_mult: mtp_config.hc_mult,
            hidden_state_rank4: mtp_config.target_hidden_layout == MtpHiddenLayout::Bshc,
            hidden_output: mtp_config.mtp_hidden_output.clone(),
            state_output: mtp_config.mtp_state_output.clone(),
        };
        let head_signature = MtpDecodeSession::new(&head_session, decode_options)
            .map_err(|error| anyhow::anyhow!("Failed to inspect MTP head: {error}"))?
            .signature()
            .clone();
        if head_signature.hidden_size != mtp_config.public_config.hidden_size {
            anyhow::bail!(
                "MTP head hidden size {} does not match configured target hidden size {}",
                head_signature.hidden_size,
                mtp_config.public_config.hidden_size
            );
        }
        let (embedder, lm_head) = match (&mtp_config.embedding_weights, &mtp_config.lm_head_weights)
        {
            (MtpWeightSource::File(embedding), MtpWeightSource::File(lm_head)) => (
                MtpEmbedder::Linear(
                    LinearEmbedder::new(
                        read_f32_weights(embedding)?,
                        mtp_config.public_config.vocab_size,
                        mtp_config.public_config.hidden_size,
                    )
                    .map_err(|error| anyhow::anyhow!("Invalid MTP embedding weights: {error}"))?,
                ),
                MtpLmHead::Linear(
                    LinearLmHead::new(
                        read_f32_weights(lm_head)?,
                        mtp_config.public_config.hidden_size,
                        mtp_config.public_config.vocab_size,
                    )
                    .map_err(|error| anyhow::anyhow!("Invalid MTP LM-head weights: {error}"))?,
                ),
            ),
            (
                MtpWeightSource::TargetInitializer(embedding),
                MtpWeightSource::TargetInitializer(lm_head),
            ) => {
                let (embedder, lm_head, vocab_size) = load_target_initializer_adapters(
                    &model_directory.model_path,
                    embedding,
                    lm_head,
                    mtp_config.public_config.hidden_size,
                )?;
                if vocab_size != mtp_config.public_config.vocab_size {
                    anyhow::bail!(
                        "MTP target initializer vocabulary {vocab_size} does not match configured vocabulary {}",
                        mtp_config.public_config.vocab_size
                    );
                }
                (embedder, lm_head)
            }
            _ => anyhow::bail!(
                "MTP embedding_weights and lm_head_weights must both use files or both use target initializers"
            ),
        };
        Some(MtpModel {
            config: mtp_config.public_config.clone(),
            runtime_config: mtp_config.clone(),
            session: Arc::new(head_session),
            embedder,
            lm_head,
            hidden_output: mtp_config.public_config.target_hidden_output.clone(),
            kv_mode: mtp_config.public_config.kv_mode,
            num_speculative_tokens: mtp_config.public_config.num_speculative_tokens,
        })
    } else {
        None
    };
    Ok(mtp)
}

fn load_eagle3_model(
    speculative_mode: &SpeculativeMode,
    session: &Session,
    environment: &Environment,
    session_options: &SessionOptions,
) -> anyhow::Result<Option<Eagle3Model>> {
    let eagle3 = if let SpeculativeMode::Eagle3(eagle_config) = speculative_mode {
        crate::config::validate_eagle3_config(eagle_config)?;
        for output_name in &eagle_config.target_hidden_outputs {
            let hidden_output = session
                .outputs()
                .iter()
                .find(|output| output.name == *output_name)
                .with_context(|| {
                    format!("EAGLE-3 target model must expose hidden-state output '{output_name}'")
                })?;
            if hidden_output.dtype != DataType::Float32 {
                anyhow::bail!(
                    "EAGLE-3 target hidden-state output '{}' must be Float32, got {:?}",
                    hidden_output.name,
                    hidden_output.dtype
                );
            }
            if hidden_output.shape.last().copied().filter(|dim| *dim > 0)
                != Some(eagle_config.hidden_size as i64)
            {
                anyhow::bail!(
                    "EAGLE-3 target hidden-state output '{}' shape {:?} does not end in configured hidden size {}",
                    hidden_output.name,
                    hidden_output.shape,
                    eagle_config.hidden_size
                );
            }
        }
        let head_session = Session::new(
            environment,
            &eagle_config.head_model,
            session_options.clone(),
        )
        .map_err(|error| anyhow::anyhow!("Failed to load EAGLE-3 head: {error}"))?;
        let head_signature = Eagle3DecodeSession::detect(&head_session)
            .map_err(|error| anyhow::anyhow!("Failed to inspect EAGLE-3 head: {error}"))?
            .context("configured EAGLE-3 head model does not expose EAGLE-3 head I/O")?;
        if head_signature.hidden_size != eagle_config.hidden_size {
            anyhow::bail!(
                "EAGLE-3 head hidden size {} does not match configured target hidden size {}",
                head_signature.hidden_size,
                eagle_config.hidden_size
            );
        }
        let expected_fused = eagle_config.hidden_size * eagle_config.target_hidden_outputs.len();
        if head_signature.fused_hidden_size != expected_fused {
            anyhow::bail!(
                "EAGLE-3 head fused hidden size {} does not match three target layers totaling {}",
                head_signature.fused_hidden_size,
                expected_fused
            );
        }
        if head_signature.draft_vocab_size > eagle_config.vocab_size {
            anyhow::bail!(
                "EAGLE-3 draft vocabulary {} exceeds target embedding vocabulary {}",
                head_signature.draft_vocab_size,
                eagle_config.vocab_size
            );
        }
        let embedding = read_f32_weights(&eagle_config.embedding_weights)?;
        Some(Eagle3Model {
            config: eagle_config.clone(),
            session: Box::new(head_session),
            embedder: LinearEmbedder::new(
                embedding,
                eagle_config.vocab_size,
                eagle_config.hidden_size,
            )
            .map_err(|error| anyhow::anyhow!("Invalid EAGLE-3 embedding weights: {error}"))?,
            hidden_outputs: eagle_config.target_hidden_outputs.clone(),
            kv_mode: eagle_config.kv_mode,
            num_speculative_tokens: eagle_config.num_speculative_tokens,
        })
    } else {
        None
    };
    Ok(eagle3)
}

fn load_shared_kv_proposer(
    speculative_mode: &SpeculativeMode,
    session: &Session,
    environment: &Environment,
    session_options: &SessionOptions,
) -> anyhow::Result<Option<SharedKvProposerModel>> {
    let shared_kv_proposer = if let SpeculativeMode::SharedKv(assistant_config) = speculative_mode {
        crate::config::validate_shared_kv_proposer_config(assistant_config)?;
        let hidden_output = session
            .outputs()
            .iter()
            .find(|output| output.name == assistant_config.target_hidden_output)
            .with_context(|| {
                format!(
                    "shared-KV proposer target model must expose hidden-state output '{}'",
                    assistant_config.target_hidden_output
                )
            })?;
        if hidden_output.dtype != DataType::Float32 {
            anyhow::bail!(
                "shared-KV proposer target hidden-state output '{}' must be Float32, got {:?}",
                hidden_output.name,
                hidden_output.dtype
            );
        }
        if hidden_output.shape.last().copied().filter(|dim| *dim > 0)
            != Some(assistant_config.backbone_hidden_size as i64)
        {
            anyhow::bail!(
                "shared-KV proposer target hidden-state output '{}' shape {:?} does not end in configured backbone hidden size {}",
                hidden_output.name,
                hidden_output.shape,
                assistant_config.backbone_hidden_size
            );
        }
        let assistant_session = Session::new(
            environment,
            &assistant_config.assistant_model,
            session_options.clone(),
        )
        .map_err(|error| anyhow::anyhow!("Failed to load shared-KV proposer model: {error}"))?;
        let signature = SharedKvProposerSession::detect(&assistant_session)
            .map_err(|error| {
                anyhow::anyhow!("Failed to inspect shared-KV proposer model: {error}")
            })?
            .context("configured shared-KV proposer model does not expose proposer I/O")?;
        if signature.backbone_hidden_size != assistant_config.backbone_hidden_size {
            anyhow::bail!(
                "shared-KV proposer hidden size {} does not match configured backbone hidden size {}",
                signature.backbone_hidden_size,
                assistant_config.backbone_hidden_size
            );
        }
        if signature.vocab_size != assistant_config.vocab_size {
            anyhow::bail!(
                "shared-KV proposer vocabulary {} does not match configured vocab size {}",
                signature.vocab_size,
                assistant_config.vocab_size
            );
        }
        for group in &assistant_config.shared_kv {
            if !signature
                .shared_kv
                .iter()
                .any(|spec| spec.name == group.name)
            {
                anyhow::bail!(
                    "shared-KV proposer model does not expose shared_kv group '{}'",
                    group.name
                );
            }
        }
        let embedding = read_f32_weights(&assistant_config.input_embedding_weights)?;
        let embedder = LinearEmbedder::new(
            embedding,
            assistant_config.vocab_size,
            assistant_config.backbone_hidden_size,
        )
        .map_err(|error| {
            anyhow::anyhow!("Invalid shared-KV proposer input embedding weights: {error}")
        })?;
        Some(SharedKvProposerModel {
            config: assistant_config.clone(),
            session: Box::new(assistant_session),
            embedder,
            num_speculative_tokens: assistant_config.num_speculative_tokens,
        })
    } else {
        None
    };
    Ok(shared_kv_proposer)
}
