//! `Engine` construction: ORT and native model directory constructors.

use super::*;
use crate::engine::memory_plan::Holder;

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
        fail_explicit_vram_limit_without_offload(
            &config,
            device_weight_package_bytes(&model_directory.model_path),
            session_options
                .execution_providers
                .iter()
                .any(|provider| !provider.caps.is_host()),
        )?;

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
            &governor,
        )?;

        // Stage: runtime KV-cache allocation, granted by the governor built above.
        let kv_cache = allocate_kv_cache(&config, kv_model.as_ref(), &governor)?;

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
            weight_placement: None,
            #[cfg(feature = "native-backend")]
            native_sessions: HashMap::new(),
            #[cfg(feature = "native-backend")]
            native_active_session: None,
            #[cfg(feature = "native-backend")]
            native_session_counter: 0,
            #[cfg(feature = "native-backend")]
            native_access_counter: 0,
            #[cfg(feature = "native-backend")]
            native_default_session: None,
            #[cfg(feature = "native-backend")]
            native_max_sessions: config.native_max_sessions,
            #[cfg(feature = "native-backend")]
            native_shared_kv_proposer: None,
            #[cfg(feature = "native-backend")]
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
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
        let model_weight_bytes = device_weight_package_bytes(&model_directory.model_path);
        #[cfg(feature = "cuda")]
        let cuda_offload_resolution =
            resolve_cuda_offload_policy(&native_device, &config.limits, model_weight_bytes);
        #[cfg(feature = "cuda")]
        let cuda_offload_policy = cuda_offload_resolution.map(|resolution| resolution.policy);
        #[cfg(feature = "cuda")]
        let weight_reservation_bytes = device_weight_reservation_for(
            model_weight_bytes,
            cuda_offload_resolution.and_then(|resolution| {
                resolution.policy.enabled.then(|| {
                    resolution
                        .policy
                        .device_budget_bytes
                        .unwrap_or(onnx_runtime_ep_cuda::DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES)
                })
            }),
            governor_kv_config.page_size_bytes,
        );
        #[cfg(not(feature = "cuda"))]
        let weight_reservation_bytes = device_weight_reservation_for(
            model_weight_bytes,
            None,
            governor_kv_config.page_size_bytes,
        );
        let governor = {
            let _span = onnx_genai_ort::prof_span!("engine.resource_governor");
            EngineResourceGovernor::new(
                config.limits.clone(),
                config.allow_runtime_override,
                governor_kv_config,
                weight_reservation_bytes,
            )
            .map_err(|error| anyhow::anyhow!("Failed to initialize Resource Governor: {error}"))?
        };
        let mut scheduler_config = config.scheduler.clone();
        // The native pool carries no per-layer geometry, so it holds only
        // bookkeeping. Its size is a fixed bound rather than a budget
        // derivation: the table pre-creates one `Page` per slot, so deriving
        // the count from a memory budget would build hundreds of millions of
        // empty structs for storage that is never allocated.
        let native_kv_pages = BOOKKEEPING_POOL_PAGES;
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
                crate::native_decode::NativeDecodeLoadOptions {
                    host_cache: governor.weight_offload_host_cache(),
                    #[cfg(feature = "cuda")]
                    cuda_offload_policy,
                    io: metadata.model.as_ref().and_then(|model| model.io.as_ref()),
                    metadata_max_len: metadata
                        .model
                        .as_ref()
                        .and_then(|model| model.max_sequence_length),
                    key_sequence_lengths_policy: crate::decode::key_sequence_lengths_policy(
                        &metadata,
                    ),
                    decode_precision: config.decode_precision,
                },
            )
            .map_err(|error| anyhow::anyhow!("Failed to load native decoder session: {error:#}"))?
        };
        let mut native_session = native_session;
        #[cfg(feature = "cuda")]
        let cuda_offload_policy = reconcile_cuda_offload_budget_after_native_load(
            &native_session,
            metadata
                .model
                .as_ref()
                .and_then(|model| model.max_sequence_length),
            governor.snapshot().resolved_limits.vram_bytes,
            model_weight_bytes,
            cuda_offload_resolution,
        )?;
        // End the second budget.
        //
        // The execution provider is built before this governor exists, so a
        // standing pool it keeps -- the CUDA weight-residency cache -- sized
        // itself from an operator figure or a default that answered to nobody.
        // Grant the KV pool most of a card, let residency default to a fraction
        // of it, and both are individually satisfied while the device is
        // oversubscribed.
        //
        // This is only correct because the ledger's device tier is now the
        // device rather than the KV sub-budget. Charging a weights pool to a
        // tier that meant "bytes KV may have" would have taken the room out of
        // KV's allowance and counted weights twice.
        //
        // A refusal here is the point: it says the model does not fit while
        // there is still a load to fail, rather than at an allocation somewhere
        // unrelated later.
        // End the allocator double count.
        //
        // The fixed reservation covers weights *and* activations *and* runtime
        // overhead, taken before any session existed because nothing else
        // accounted for them. An allocator that commits on demand accounts for
        // the weights itself, granule by granule, as they are actually
        // allocated -- so holding a reservation for them too charges the same
        // memory twice and the tier reads high by the weight size.
        //
        // Only the weight portion is released. Activations and ONNX Runtime's
        // internal overhead do not flow through our allocator, so nothing else
        // is accounting for those and the reservation is still the only thing
        // that knows about them.
        if native_session.commits_on_demand() {
            let weights = governor.snapshot().breakdown.model_weights_bytes;
            let released = governor
                .plan()
                .release(Holder::FixedDeviceReservation, weights);
            if released > 0 {
                tracing::debug!(
                    "released {released} bytes of the fixed device reservation: the allocator \
                     commits on demand and charges the ledger for the model's weights as it \
                     maps them, so reserving for them as well counted the same memory twice"
                );
            }
        }
        // End the residency double count.
        //
        // With CUDA weight offload and the VMM arena off, the "model weights"
        // portion of the fixed reservation is the residency budget: the device
        // will hold at most that much of the package at once. The CUDA EP then
        // adopts the same budget as the residency cache lease it owns. Leaving
        // the fixed claim live while adoption calls `reserve` asks the ledger
        // for the exact bytes it already charged, producing the #704 refusal.
        //
        // Release the startup placeholder only on the CUDA offload path, where
        // the provider is about to take over the same device claim. Non-CUDA
        // providers do not hold this residency cache, and VMM has already
        // released the weight portion above because its allocator records real
        // commits instead of a standing weight budget.
        #[cfg(feature = "cuda")]
        if matches!(
            native_device,
            crate::native_decode::NativeDecodeDevice::Cuda { .. }
        ) && cuda_offload_policy.is_some_and(|policy| policy.enabled)
            && !native_session.commits_on_demand()
        {
            let weights = governor.snapshot().breakdown.model_weights_bytes;
            let released = governor
                .plan()
                .release_fixed_device_reservation_for_provider_pool(weights);
            if released > 0 {
                tracing::debug!(
                    "released {released} bytes of the fixed device reservation: CUDA weight \
                     offload will adopt the same bytes as the residency cache lease, so keeping \
                     both claims would double-count the cache budget"
                );
            }
        }
        let governed_pool_bytes = governor
            .plan()
            .adopt_provider_pool(&native_session, Holder::WeightResidency)
            .context(
                "the native execution provider holds a standing device pool the governor cannot \
                 grant alongside the model's other claims; lower \
                 ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES or raise the device limit",
            )?;

        if governed_pool_bytes > 0 {
            tracing::debug!(
                bytes = governed_pool_bytes,
                "native execution provider pool is now governed"
            );
        }
        let weight_placement = plan_static_weight_placement(
            native_session.inference_session(),
            config.device_policy,
            governed_pool_bytes,
        )?;
        if let Some(report) = &weight_placement {
            tracing::info!(
                device_bytes = report.device_bytes,
                host_bytes = report.host_bytes,
                "computed static weight placement plan; enforcement is not wired yet"
            );
        }
        // Charge the fixed-size recurrent state a hybrid decoder keeps --
        // `conv_state` and `recurrent_state` for the linear-attention layers.
        //
        // Unlike KV it cannot be rewound, recomputed or shared, so a sequence
        // either keeps it or ends. It was allocated and charged to nothing.
        //
        // One instance, not one per concurrent sequence: native decode runs a
        // single serialized session, and other sequences retain tokens and are
        // re-prefilled rather than each holding a live state tensor. Multiplying
        // by the scheduler's batch size would reserve up to 32x memory that is
        // never allocated, and refuse models that fit.
        //
        // The tier comes from the session, because it is a fact about the
        // running system rather than about the holder: CPU state is host memory,
        // a CUDA session's fixed-state bindings are on the device.
        //
        // Zero for a decoder with no recurrent layers, which is most of them,
        // and takes no lease.
        let (recurrent_bytes, recurrent_tier) = native_session
            .recurrent_state_reservation()
            .context("sizing the decoder's fixed recurrent state")?;
        governor
            .plan()
            .reserve_on(Holder::RecurrentState, recurrent_tier, recurrent_bytes)
            .map_err(|error| {
                anyhow::anyhow!(
                    "cannot reserve {recurrent_bytes} bytes of recurrent state on \
                     {recurrent_tier:?}: {error}; raise the limit for that tier"
                )
            })?;
        // The KV the native path actually uses. Its page table carries no
        // storage, so unlike the ONNX Runtime path nothing else leases this.
        // Sized at full context because a lease is a reservation: charging the
        // current length would admit a sequence the device cannot carry to its
        // limit and discover that mid-generation.
        match metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length)
        {
            Some(max_context) => {
                let (native_kv_bytes, native_kv_tier) = native_session
                    .kv_reservation(max_context)
                    .context("sizing the decoder's KV tensors")?;
                if native_session.commits_on_demand() {
                    // The allocator maps physical memory as the sequence grows
                    // and charges the ledger for each granule, so the worst
                    // case is a ceiling to check rather than a claim to hold.
                    //
                    // Holding it would refuse models a short conversation never
                    // grows into -- 768 MiB per sequence on a 32K-context 0.5B
                    // model -- which on a small machine is the difference
                    // between running and not.
                    let headroom = governor.plan().available_on(native_kv_tier);
                    if native_kv_bytes > headroom {
                        tracing::warn!(
                            "one sequence at {max_context} tokens of context needs \
                             {native_kv_bytes} bytes of KV on {native_kv_tier:?} but only \
                             {headroom} bytes are free; the allocator commits on demand so this \
                             loads, and a conversation that grows into the full context will be \
                             refused a page at a time rather than at load"
                        );
                    }
                } else {
                    governor
                        .plan()
                        .reserve_on(Holder::NativeKvCache, native_kv_tier, native_kv_bytes)
                        .map_err(|error| {
                            anyhow::anyhow!(
                                "cannot reserve {native_kv_bytes} bytes of KV for one sequence at \
                                 {max_context} tokens of context on {native_kv_tier:?}: {error}; \
                                 {max_context} is the model's declared max_sequence_length, and \
                                 --max-context will not lower it because the declared value takes \
                                 precedence -- raise the limit for that tier, or re-export the \
                                 model with a shorter declared context"
                            )
                        })?;
                }
            }
            // Refusing to load would be the strict reading, but the model runs
            // fine and the only loss is that this holder is missing from the
            // ledger -- which is where it was before this existed. Warn rather
            // than reserve a guessed figure, which would be worse than a known
            // gap because it would look accounted for.
            None => tracing::warn!(
                "inference metadata declares no max_sequence_length, so the decoder's KV \
                 tensors cannot be sized and are not charged to the memory ledger; tier \
                 totals will understate device use by one sequence's KV"
            ),
        }
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
            kv_cache: PagedKvCache::new(config.page_size, native_kv_pages),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model: None,
            decode_path: ModelDecodePath::Legacy,
            scheduler,
            governor,
            sessions: HashMap::new(),
            session: None,
            native_session: Some(native_session),
            weight_placement,
            native_sessions: HashMap::new(),
            native_active_session: None,
            native_session_counter: 0,
            native_access_counter: 0,
            native_default_session: None,
            native_max_sessions: config.native_max_sessions,
            native_shared_kv_proposer,
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
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
            device_weight_package_bytes(&model_directory.model_path),
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
    governor: &EngineResourceGovernor,
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
        let draft_kv_cache =
            if let Some(kv_model) = &draft_kv_model {
                let draft_pages = kv_pages_for_budget(
                    governor.snapshot().derived_budget.kv_bytes,
                    governor.snapshot().resolved_limits.host_ram_bytes,
                    config.scheduler.max_total_tokens,
                    kv_model.tensor_config.page_size,
                    kv_model.tensor_config.dtype,
                    &kv_model.layer_configs,
                );
                governor.plan().kv_pool(
                Holder::DraftKvPool,
                kv_model.tensor_config.page_size,
                kv_model.tensor_config.dtype,
                kv_model.layer_configs.clone(),
                draft_pages,
            )
            .context(
                "cannot allocate the draft model's KV page pool within the device KV budget; a \
                 draft model needs its own pages alongside the target model's",
            )?
            } else {
                PagedKvCache::new(config.page_size, BOOKKEEPING_POOL_PAGES)
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

/// Pages retained by a pool that holds no KV data.
///
/// Without per-layer geometry a page carries no storage, so this bounds only
/// the page-table bookkeeping. It still needs a bound: the table pre-creates
/// one `Page` per slot, so a count derived from a memory budget would build
/// hundreds of millions of empty structs and exhaust the machine before any KV
/// existed.
const BOOKKEEPING_POOL_PAGES: usize = 1024;

/// How many pages of real KV storage fit in the governor's KV budget.
///
/// Deliberately **not** `derived_budget.total_pages`. That figure divides the
/// budget by the governor's own `page_size_bytes`, which is a placeholder when
/// no KV model has been inferred — on a machine with 8 GiB of device memory it
/// resolves to hundreds of millions of pages, and the pool would try to
/// allocate a `Page` for every one of them. The page count has to come from the
/// geometry the pages will actually have.
pub(crate) fn kv_pages_for_budget(
    kv_budget_bytes: u64,
    host_ram_bytes: u64,
    working_set_tokens: usize,
    page_size: usize,
    dtype: onnx_genai_kv::KvDType,
    layer_configs: &[onnx_genai_kv::LayerTensorConfig],
) -> usize {
    let quant = onnx_genai_kv::KvQuantConfig::homogeneous(dtype, layer_configs.len());
    let per_page =
        onnx_genai_kv::PageTable::planned_pool_bytes(page_size, 1, layer_configs, Some(&quant));
    if per_page == 0 {
        return BOOKKEEPING_POOL_PAGES;
    }
    // Two ceilings, both binding: the KV budget is policy, host RAM is physics.
    // A page is a `Vec<f32>`, so a pool that fits the KV budget but not host
    // memory still cannot be allocated.
    let ceiling =
        usize::try_from(kv_budget_bytes.min(host_ram_bytes) / per_page).unwrap_or(usize::MAX);
    let wanted = working_set_tokens.div_ceil(page_size.max(1));
    // At least one page, or the pool cannot hold a single token and the failure
    // surfaces later as a decode that mysteriously caches nothing.
    //
    // Note what this does *not* promise: when a ceiling is the binding term the
    // pool still eagerly allocates up to it. The working set bounds the pool
    // only while it is the smallest of the three.
    wanted.min(ceiling).max(1)
}

/// Build the KV page pool, sized and granted by the governor.
///
/// The page count comes from `derived_budget.total_pages`, which the governor
/// already computes from the device ceiling minus the fixed reservation. It
/// used to come from `EngineConfig::num_gpu_pages` as well, and two sources of
/// truth for one quantity is how a budget ends up describing memory nobody
/// allocated.
///
/// Only the configured path holds storage worth leasing. Without per-layer
/// geometry a pool is pure bookkeeping and occupies nothing, so it is built
/// ungoverned rather than taking a lease of zero that implies otherwise.
/// Device bytes to reserve for model weights.
///
/// The whole package, normally: with every weight resident, that is what the
/// device holds.
///
/// **Not** the whole package when weight offload is on. Offload exists because
/// the weights do not fit, and it keeps only a bounded residency cache on the
/// device while the rest stays on the host. Reserving the full package *and*
/// letting that cache hold a slice of it counts the same bytes twice: on an
/// 8 GiB card a 6 GiB model reserves 6 GiB, leaving 2 GiB for KV, while the
/// residency cache separately holds up to its own budget of the same weights.
///
/// So with offload on, the reservation is what the device will actually hold --
/// the residency budget -- capped at the package size, because a budget larger
/// than the model cannot be filled.
///
/// For native CUDA, an explicit VRAM limit can derive the offload budget before
/// the provider exists; other paths still reserve the whole package.
fn device_weight_package_bytes(model_path: &std::path::Path) -> u64 {
    onnx_genai_ort::model_weight_bytes(model_path)
}

/// The temporary startup reservation, given the package size and offload budget.
///
/// Native CUDA offload replaces this placeholder with the provider-owned
/// residency lease after session load. Leaving one KV page unreserved prevents
/// the governor from taking the "reservation does not fit; drop it" warning path
/// that hid #712 while preserving the later ledger-enforced admission point.
fn device_weight_reservation_for(
    package_bytes: u64,
    offload_budget: Option<u64>,
    kv_page_size_bytes: u64,
) -> u64 {
    match offload_budget {
        // A budget larger than the model cannot be filled, so the device still
        // holds at most the package.
        Some(budget) => {
            let reservation = budget.min(package_bytes);
            reservation.saturating_sub(kv_page_size_bytes.min(reservation))
        }
        None => package_bytes,
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
struct CudaOffloadResolution {
    policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
    device_budget_is_override: bool,
    auto_enabled_from_vram_limit: bool,
}

#[cfg(feature = "cuda")]
fn resolve_cuda_offload_policy(
    native_device: &crate::native_decode::NativeDecodeDevice,
    limits: &ResourceLimits,
    package_bytes: u64,
) -> Option<CudaOffloadResolution> {
    resolve_cuda_offload_policy_from_env_policy(
        native_device,
        limits,
        package_bytes,
        onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env(),
    )
}

#[cfg(feature = "cuda")]
fn resolve_cuda_offload_policy_from_env_policy(
    native_device: &crate::native_decode::NativeDecodeDevice,
    limits: &ResourceLimits,
    package_bytes: u64,
    env_policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
) -> Option<CudaOffloadResolution> {
    if !matches!(
        native_device,
        crate::native_decode::NativeDecodeDevice::Cuda { .. }
    ) {
        return None;
    }

    if env_policy.enabled {
        return Some(CudaOffloadResolution {
            policy: env_policy,
            device_budget_is_override: env_policy.device_budget_bytes.is_some(),
            auto_enabled_from_vram_limit: false,
        });
    }

    let ResourceLimit::Bytes(resolved_vram_bytes) = limits.vram_limit else {
        return None;
    };
    if package_bytes <= resolved_vram_bytes {
        return None;
    }

    let offload_device_budget_bytes = env_policy
        .device_budget_bytes
        .unwrap_or(resolved_vram_bytes);
    Some(CudaOffloadResolution {
        policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy {
            enabled: true,
            device_budget_bytes: Some(offload_device_budget_bytes),
            async_pagein: env_policy.async_pagein,
        },
        device_budget_is_override: env_policy.device_budget_bytes.is_some(),
        auto_enabled_from_vram_limit: true,
    })
}

#[cfg(feature = "cuda")]
fn reconcile_cuda_offload_budget_after_native_load(
    native_session: &crate::native_decode::NativeDecodeSession,
    max_context: Option<usize>,
    resolved_vram_bytes: u64,
    model_weight_bytes: u64,
    resolution: Option<CudaOffloadResolution>,
) -> anyhow::Result<Option<onnx_runtime_ep_cuda::DeviceOffloadPolicy>> {
    let Some(mut resolution) = resolution else {
        return Ok(None);
    };
    if !resolution.policy.enabled {
        return Ok(Some(resolution.policy));
    }

    let (native_kv_bytes, native_kv_tier) = match max_context {
        Some(max_context) => native_session
            .kv_reservation(max_context)
            .context("sizing native CUDA KV before deriving the weight-offload budget")?,
        None => (0, onnx_runtime_memory_governor::Tier::Device),
    };
    let native_kv_device_bytes = if native_kv_tier == onnx_runtime_memory_governor::Tier::Device {
        native_kv_bytes
    } else {
        0
    };
    let (recurrent_state_bytes, recurrent_tier) = native_session
        .recurrent_state_reservation()
        .context("sizing native CUDA recurrent state before deriving the weight-offload budget")?;
    let recurrent_device_bytes = if recurrent_tier == onnx_runtime_memory_governor::Tier::Device {
        recurrent_state_bytes
    } else {
        0
    };
    let required_device_non_weight_bytes =
        native_kv_device_bytes.saturating_add(recurrent_device_bytes);
    let available_weight_offload_budget_bytes =
        resolved_vram_bytes.saturating_sub(required_device_non_weight_bytes);
    let minimum_useful_weight_budget_bytes = native_session.max_lazy_weight_working_set_bytes();
    let requested_weight_offload_budget_bytes = resolution
        .policy
        .device_budget_bytes
        .unwrap_or(onnx_runtime_ep_cuda::DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES);

    if available_weight_offload_budget_bytes == 0
        || available_weight_offload_budget_bytes < minimum_useful_weight_budget_bytes
        || requested_weight_offload_budget_bytes < minimum_useful_weight_budget_bytes
        || (resolution.device_budget_is_override
            && requested_weight_offload_budget_bytes > available_weight_offload_budget_bytes)
    {
        anyhow::bail!(
            "explicit VRAM limit cannot fit native CUDA weights plus required device state: \
             model_weight_bytes={model_weight_bytes} resolved_vram_bytes={resolved_vram_bytes} \
             required_device_non_weight_bytes={required_device_non_weight_bytes} \
             native_kv_bytes={native_kv_device_bytes} recurrent_state_bytes={recurrent_device_bytes} \
             activations_bytes=unknown runtime_overhead_bytes=unknown \
             minimum_useful_weight_budget_bytes={minimum_useful_weight_budget_bytes} \
             available_weight_offload_budget_bytes={available_weight_offload_budget_bytes} \
             requested_weight_offload_budget_bytes={requested_weight_offload_budget_bytes}"
        );
    }

    let offload_device_budget_bytes = if resolution.device_budget_is_override {
        requested_weight_offload_budget_bytes
    } else {
        requested_weight_offload_budget_bytes.min(available_weight_offload_budget_bytes)
    };
    let adopted = native_session
        .set_weight_residency_budget(offload_device_budget_bytes)
        .context("setting native CUDA weight-offload budget after reserving room for KV")?
        .unwrap_or(offload_device_budget_bytes);
    resolution.policy.device_budget_bytes = Some(adopted);

    if resolution.auto_enabled_from_vram_limit {
        tracing::info!(
            model_weight_bytes,
            resolved_vram_bytes,
            required_device_non_weight_bytes,
            native_kv_bytes = native_kv_device_bytes,
            recurrent_state_bytes = recurrent_device_bytes,
            minimum_useful_weight_budget_bytes,
            offload_device_budget_bytes = adopted,
            "enabled CUDA weight offload because model weights exceed the VRAM limit"
        );
    }

    Ok(Some(resolution.policy))
}

fn fail_explicit_vram_limit_without_offload(
    config: &EngineConfig,
    package_bytes: u64,
    device_weights_are_selected: bool,
) -> anyhow::Result<()> {
    if !device_weights_are_selected {
        return Ok(());
    }
    let ResourceLimit::Bytes(resolved_vram_bytes) = config.limits.vram_limit else {
        return Ok(());
    };
    if package_bytes <= resolved_vram_bytes {
        return Ok(());
    }
    anyhow::bail!(
        "model weights require {package_bytes} bytes of device memory, but --vram-limit / \
         serving.memory.limits.vram_limit allows {resolved_vram_bytes} bytes and this backend \
         cannot automatically offload weights; select the native CUDA backend for automatic \
         weight offload or raise the VRAM limit"
    );
}

fn allocate_kv_cache(
    config: &EngineConfig,
    kv_model: Option<&KvModelInfo>,
    governor: &EngineResourceGovernor,
) -> anyhow::Result<PagedKvCache> {
    let budget = governor.snapshot().derived_budget;
    if let Some(kv_model) = kv_model {
        let num_pages = kv_pages_for_budget(
            budget.kv_bytes,
            governor.snapshot().resolved_limits.host_ram_bytes,
            config.scheduler.max_total_tokens,
            kv_model.tensor_config.page_size,
            kv_model.tensor_config.dtype,
            &kv_model.layer_configs,
        );
        let mut span = onnx_genai_ort::prof_span!("engine.kv_cache_alloc");
        span.set_arg("page_size", kv_model.tensor_config.page_size as u64);
        span.set_arg("num_gpu_pages", num_pages as u64);
        span.set_arg("kv_budget_bytes", budget.kv_bytes);
        span.set_arg("layers", kv_model.layer_configs.len() as u64);
        // The paged tensor layout is derived from present-KV outputs: each
        // layer has key/value tensors shaped like [batch, kv_heads, seq, head_dim].
        // Per-layer geometry (heterogeneous head_dim across layers, e.g. the
        // Gemma-4 sliding/full split) is fed from the model's own KV output
        // shapes so mixed-geometry models page correctly.
        governor.plan().kv_pool(
            Holder::KvPool,
            kv_model.tensor_config.page_size,
            kv_model.tensor_config.dtype,
            kv_model.layer_configs.clone(),
            num_pages,
        )
        .with_context(|| {
            format!(
                "cannot allocate the KV page pool: {num_pages} page(s) of {} token(s) across {} \
                 layer(s) do not fit the {} byte KV budget; lower the context length, raise the \
                 device limit, or use a smaller KV precision",
                kv_model.tensor_config.page_size,
                kv_model.layer_configs.len(),
                budget.kv_bytes,
            )
        })
    } else {
        let mut span = onnx_genai_ort::prof_span!("engine.kv_cache_alloc");
        span.set_arg("page_size", config.page_size as u64);
        span.set_arg("num_gpu_pages", BOOKKEEPING_POOL_PAGES as u64);
        Ok(PagedKvCache::new(config.page_size, BOOKKEEPING_POOL_PAGES))
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

#[cfg(test)]
mod pool_sizing_tests {
    use super::*;

    fn geometry(layers: usize) -> Vec<onnx_genai_kv::LayerTensorConfig> {
        (0..layers)
            .map(|_| onnx_genai_kv::LayerTensorConfig {
                num_kv_heads: 8,
                head_dim: 128,
            })
            .collect()
    }

    /// The pool must never be sized by a budget divided by the wrong page size.
    ///
    /// `derived_budget.total_pages` divides the KV budget by the governor's own
    /// `page_size_bytes`, which is a placeholder when no KV model has been
    /// inferred. On an 8 GiB device it comes to ~483 million pages, and because
    /// the table pre-creates one `Page` per slot, building that pool exhausts
    /// the machine before any KV exists -- which is exactly how this was found,
    /// as a CI runner dying with SIGTERM mid-test.
    ///
    /// So the count has to come from the geometry the pages will really have,
    /// and the resulting pool has to fit the budget it was derived from.
    #[test]
    fn pages_derived_from_a_budget_produce_a_pool_that_fits_that_budget() {
        let configs = geometry(32);
        let dtype = onnx_genai_kv::KvDType::F32;
        for budget in [
            64 * 1024 * 1024u64,
            1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
        ] {
            let pages = kv_pages_for_budget(budget, u64::MAX, 1 << 20, 16, dtype, &configs);
            let quant = onnx_genai_kv::KvQuantConfig::homogeneous(dtype, configs.len());
            let planned =
                onnx_genai_kv::PageTable::planned_pool_bytes(16, pages, &configs, Some(&quant));
            assert!(
                planned <= budget,
                "a {budget} byte budget produced {pages} pages needing {planned} bytes"
            );
            assert!(pages >= 1, "a usable budget produced a pool with no pages");
        }
    }

    /// A budget too small for even one page still yields a usable pool.
    ///
    /// Returning zero pages would defer the failure to the first decode, which
    /// would simply cache nothing and look like a mysterious slowdown.
    #[test]
    fn a_budget_below_one_page_still_yields_one_page() {
        assert_eq!(
            kv_pages_for_budget(
                1,
                u64::MAX,
                65536,
                16,
                onnx_genai_kv::KvDType::F32,
                &geometry(32)
            ),
            1
        );
    }

    /// Without geometry the pool is bookkeeping only and takes a fixed bound.
    #[test]
    fn a_pool_without_geometry_is_bounded_rather_than_derived() {
        assert_eq!(
            kv_pages_for_budget(
                8 * 1024 * 1024 * 1024,
                u64::MAX,
                65536,
                16,
                onnx_genai_kv::KvDType::F32,
                &[]
            ),
            BOOKKEEPING_POOL_PAGES
        );
    }
    /// A large budget must not become a large eager allocation.
    ///
    /// The page table materialises every page at construction, so sizing the
    /// pool by the *ceiling* claims the whole KV budget before a token is
    /// generated -- 8 GiB on an 8 GiB device. The budget is a limit, not a
    /// target; the scheduler's working set is the target. This caught a second
    /// bug after the first fix: engine tests went from ~2s to 872s because
    /// every engine built a pool sized to the entire device.
    #[test]
    fn a_huge_budget_does_not_produce_a_huge_pool() {
        let configs = geometry(32);
        let dtype = onnx_genai_kv::KvDType::F32;
        let working_set = 65_536;
        let pages = kv_pages_for_budget(
            64 * 1024 * 1024 * 1024,
            u64::MAX,
            working_set,
            16,
            dtype,
            &configs,
        );

        assert_eq!(
            pages,
            working_set / 16,
            "the pool should hold the scheduler's working set, not the whole budget"
        );

        let quant = onnx_genai_kv::KvQuantConfig::homogeneous(dtype, configs.len());
        let planned =
            onnx_genai_kv::PageTable::planned_pool_bytes(16, pages, &configs, Some(&quant));
        assert!(
            planned < 64 * 1024 * 1024 * 1024,
            "a 64 GiB budget produced a {planned} byte pool"
        );
    }

    /// A budget smaller than the working set still caps the pool.
    #[test]
    fn the_budget_still_caps_a_working_set_that_does_not_fit() {
        let configs = geometry(32);
        let dtype = onnx_genai_kv::KvDType::F32;
        let quant = onnx_genai_kv::KvQuantConfig::homogeneous(dtype, configs.len());
        let per_page = onnx_genai_kv::PageTable::planned_pool_bytes(16, 1, &configs, Some(&quant));
        let budget = per_page * 4;

        let pages = kv_pages_for_budget(budget, u64::MAX, 1 << 20, 16, dtype, &configs);
        assert_eq!(pages, 4, "the budget must cap a working set it cannot hold");
    }
    /// Host RAM caps the pool even when the KV budget would allow more.
    ///
    /// A page is a `Vec<f32>`, so a pool that fits the KV policy budget but not
    /// physical host memory still cannot be allocated. Charging only the KV
    /// budget would let a CPU deployment OOM while every counter reported
    /// headroom.
    #[test]
    fn host_memory_caps_the_pool_even_when_the_kv_budget_would_allow_more() {
        let configs = geometry(32);
        let dtype = onnx_genai_kv::KvDType::F32;
        let quant = onnx_genai_kv::KvQuantConfig::homogeneous(dtype, configs.len());
        let per_page = onnx_genai_kv::PageTable::planned_pool_bytes(16, 1, &configs, Some(&quant));

        let pages = kv_pages_for_budget(
            u64::MAX,     // KV policy budget: unlimited
            per_page * 3, // host RAM: three pages
            1 << 20,      // working set: far more than three pages
            16,
            dtype,
            &configs,
        );
        assert_eq!(pages, 3, "host memory did not cap the pool");
    }

    /// The working set bounds the pool only while it is the smallest term.
    ///
    /// Stated explicitly because the sibling test's name invites the opposite
    /// reading: when a ceiling binds, the pool still eagerly allocates up to
    /// that ceiling. That is the intended behaviour for a pre-allocated pool,
    /// but it is not the unconditional bound the phrase "does not produce a
    /// huge pool" suggests, so it is pinned rather than left to the reader.
    #[test]
    fn a_binding_ceiling_is_still_allocated_eagerly() {
        let configs = geometry(32);
        let dtype = onnx_genai_kv::KvDType::F32;
        let quant = onnx_genai_kv::KvQuantConfig::homogeneous(dtype, configs.len());
        let per_page = onnx_genai_kv::PageTable::planned_pool_bytes(16, 1, &configs, Some(&quant));

        let pages = kv_pages_for_budget(per_page * 64, u64::MAX, 1 << 30, 16, dtype, &configs);
        assert_eq!(
            pages, 64,
            "a binding budget should be taken in full, not reduced further"
        );
    }

    /// With offload off, the device holds the whole package.
    ///
    /// With it on, it does not -- that is what offload is for -- so reserving
    /// the package *and* letting the residency cache hold a slice of it counts
    /// the same bytes twice. On an 8 GiB card a 6 GiB model would reserve 6 GiB,
    /// leaving 2 GiB for KV, while the residency cache separately held up to
    /// its own budget of the same weights.
    #[test]
    fn offload_reserves_what_the_device_holds_not_the_whole_package() {
        let package = 6u64 << 30;

        assert_eq!(
            device_weight_reservation_for(package, None, 0),
            package,
            "with offload off every weight is resident"
        );
        assert_eq!(
            device_weight_reservation_for(package, Some(2 << 30), 0),
            2 << 30,
            "with offload on the device holds the residency budget, not the package"
        );
    }

    /// A budget larger than the model cannot be filled.
    ///
    /// The default is 4 GiB, so any model smaller than that would otherwise
    /// reserve more than it could possibly occupy -- and the reservation comes
    /// straight out of what KV is offered.
    #[test]
    fn a_budget_larger_than_the_model_reserves_only_the_model() {
        let package = 1u64 << 30;
        assert_eq!(
            device_weight_reservation_for(package, Some(4 << 30), 0),
            package
        );
    }

    #[test]
    fn offload_startup_reservation_leaves_one_kv_page_unwarned() {
        assert_eq!(
            device_weight_reservation_for(10_000, Some(6_000), 128),
            5_872,
            "the temporary startup claim must leave the governor a one-page KV floor; \
             the CUDA provider later adopts the full offload budget as the real claim"
        );
    }

    #[test]
    fn explicit_vram_limit_fails_without_an_offload_capable_backend() {
        let error = fail_explicit_vram_limit_without_offload(
            &EngineConfig {
                limits: ResourceLimits {
                    vram_limit: ResourceLimit::Bytes(6_000),
                    ..ResourceLimits::default()
                },
                ..EngineConfig::default()
            },
            10_000,
            true,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("require 10000 bytes"), "{error}");
        assert!(error.contains("allows 6000 bytes"), "{error}");
        assert!(error.contains("automatically offload weights"), "{error}");
    }

    #[test]
    fn explicit_vram_limit_does_not_reject_host_only_execution() {
        fail_explicit_vram_limit_without_offload(
            &EngineConfig {
                limits: ResourceLimits {
                    vram_limit: ResourceLimit::Bytes(6_000),
                    ..ResourceLimits::default()
                },
                ..EngineConfig::default()
            },
            10_000,
            false,
        )
        .expect("a VRAM limit should not reject a host-only execution provider");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn explicit_vram_limit_auto_enables_cuda_weight_offload() {
        let policy = resolve_cuda_offload_policy_from_env_policy(
            &crate::native_decode::NativeDecodeDevice::Cuda { index: Some(0) },
            &ResourceLimits {
                vram_limit: ResourceLimit::Bytes(6_000),
                ..ResourceLimits::default()
            },
            10_000,
            onnx_runtime_ep_cuda::DeviceOffloadPolicy {
                enabled: false,
                device_budget_bytes: None,
                async_pagein: true,
            },
        )
        .expect("weights above an explicit CUDA VRAM limit should enable offload");

        assert!(policy.policy.enabled);
        assert_eq!(policy.policy.device_budget_bytes, Some(6_000));
        assert!(policy.policy.async_pagein);
        assert!(policy.auto_enabled_from_vram_limit);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn explicit_weight_offload_device_bytes_overrides_vram_limit_derivation() {
        let policy = resolve_cuda_offload_policy_from_env_policy(
            &crate::native_decode::NativeDecodeDevice::Cuda { index: Some(0) },
            &ResourceLimits {
                vram_limit: ResourceLimit::Bytes(6_000),
                ..ResourceLimits::default()
            },
            10_000,
            onnx_runtime_ep_cuda::DeviceOffloadPolicy {
                enabled: false,
                device_budget_bytes: Some(4_000),
                async_pagein: false,
            },
        )
        .expect("the explicit limit still triggers offload");

        assert_eq!(policy.policy.device_budget_bytes, Some(4_000));
        assert!(policy.device_budget_is_override);
    }
}
