//! `Engine` construction: ORT and native model directory constructors.

use super::*;
use crate::engine::memory_plan::Holder;
use crate::memory_authority::{
    DeviceCompatibilityDomain, MemoryAuthorityProvider, SharedMemoryAuthorityProvider,
};

impl Engine {
    /// Whether `model_dir` declares a `pipeline.workflow` package.
    ///
    /// Resolved from the package itself so a caller never has to know — and can
    /// never disagree about — which shape it loaded. This is the one place the
    /// distinction is made; every constructor below routes on it.
    fn declares_workflow(model_dir: &Path) -> anyhow::Result<bool> {
        Ok(
            onnx_genai_ort::PipelineModelDirectory::load_if_declared(model_dir)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "Failed to inspect package '{}': {error}",
                        model_dir.display()
                    )
                })?
                .is_some(),
        )
    }

    /// Load a package from a directory.
    ///
    /// One entry point for both package shapes: a package that declares
    /// `pipeline.workflow` is loaded as workflow interpreter state, a package
    /// that declares a bare decoder as decode-core state. The caller gets the
    /// same type either way.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        if Self::declares_workflow(model_dir)? {
            return Self::from_pipeline_dir(model_dir, config);
        }
        Self::from_dir_impl(model_dir, config, SessionOptions::default(), false, None)
    }

    /// Load a package using a caller-owned device authority provider.
    pub fn from_dir_with_memory_authority_provider(
        model_dir: &Path,
        config: EngineConfig,
        provider: Arc<dyn MemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        if Self::declares_workflow(model_dir)? {
            return Self::from_pipeline_dir_with_memory_authority_provider(
                model_dir, config, provider,
            );
        }
        Self::from_dir_impl(
            model_dir,
            config,
            SessionOptions::default(),
            false,
            Some(provider),
        )
    }

    /// Load a package from a directory with explicit ORT session options.
    pub fn from_dir_with_session_options(
        model_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        if Self::declares_workflow(model_dir)? {
            return crate::pipeline::WorkflowRuntime::from_dir_with_session_options(
                model_dir,
                config,
                session_options,
            )
            .and_then(Self::from_workflow);
        }
        Self::from_dir_impl(model_dir, config, session_options, true, None)
    }

    fn from_dir_impl(
        model_dir: &Path,
        mut config: EngineConfig,
        mut session_options: SessionOptions,
        session_options_are_programmatic: bool,
        authority_provider: Option<SharedMemoryAuthorityProvider>,
    ) -> anyhow::Result<Self> {
        let model_directory = {
            let _span = onnx_genai_ort::prof_span!("engine.resolve_model_directory");
            let package_selection = package_selection_from_session_options(&session_options);
            ModelDirectory::load_with_package_selection(model_dir, &package_selection)
                .map_err(|e| anyhow::anyhow!("Failed to resolve model directory: {e}"))?
        };
        let operator_vram_limit = config.limits.vram_limit;
        let metadata_hints = load_model_metadata_hints(&model_directory.model_path)?;
        report_metadata_hint_warnings(&metadata_hints);
        if metadata_hints.has_errors() {
            anyhow::bail!(
                "ONNX model metadata contains conflicting forced placement hints; remove one of the contradictory onnx_runtime.device declarations"
            );
        }
        apply_model_memory_hints(&mut config, &metadata_hints)?;
        // A shared production authority is process policy. Model metadata may
        // advise placement inside that policy, but must never establish or
        // lower the device ceiling (especially when the operator selected
        // Auto). Standalone engines keep the historical metadata behavior.
        if authority_provider.is_some() {
            config.limits.vram_limit = operator_vram_limit;
        }
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
            #[cfg(feature = "native-backend")]
            {
                let native_device =
                    resolve_native_decode_device(config.native_device.clone(), &session_options)?;
                let domain = native_device_domain(&native_device);
                validate_shared_authority_limit(
                    authority_provider.as_ref(),
                    &domain,
                    config.limits.vram_limit,
                )?;
                return augment_backend_error(
                    Self::from_native_model_directory(
                        model_directory,
                        config,
                        &session_options,
                        metadata_hints,
                        native_device,
                        authority_provider.as_ref(),
                        &domain,
                    ),
                    EngineDecodeBackend::Native,
                );
            }
            #[cfg(not(feature = "native-backend"))]
            {
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
        }
        let domain = session_device_domain(&session_options)?;
        validate_shared_authority_limit(
            authority_provider.as_ref(),
            &domain,
            config.limits.vram_limit,
        )?;
        let metadata = load_inference_metadata(&model_directory)?;
        let model_io = metadata.decoder_io();
        let kv_inputs = model_io
            .and_then(|io| io.kv_inputs.clone())
            .unwrap_or_default();
        let kv_outputs = model_io
            .and_then(|io| io.kv_outputs.clone())
            .unwrap_or_default();
        let graph_io = onnx_genai_ort::graph_io_from_model_path_for_kv_pairs(
            &model_directory.model_path,
            &kv_inputs,
            &kv_outputs,
        )
        .map_err(|e| anyhow::anyhow!("Failed to read decoder graph I/O for KV geometry: {e}"))?;
        let kv_model =
            infer_kv_model_info(&graph_io, model_io, config.page_size, config.kv_cache_dtype)?;
        let plan_kv_config = match kv_model.as_ref() {
            Some(kv_model) => governor_kv_config(Some(kv_model), &config)?,
            None if model_io_declares_only_fixed_state(model_io) => {
                governor_no_paged_kv_config(&config)?
            }
            None => governor_kv_config(None, &config)?,
        };
        let model_weight_bytes = device_weight_package_bytes(&model_directory.model_path);
        // ORT backend manages its own device memory (advisory-only governor),
        // so it has no native CUDA ordinal to resolve the VRAM fraction against.
        // The device (VRAM) capacity stays honestly `None` when it cannot be
        // measured (#947): it is reported verbatim as `resolved_device_budget`
        // and never borrows the host tier. The residency verdict is a separate
        // fact, sized against the physical hot tier the weights actually occupy
        // -- the measured VRAM budget when a device is queryable, else the
        // measured host-RAM ceiling -- so a fitting model reads `FullResident`
        // instead of `Unknown` without fabricating a device number.
        let resolved_vram_bytes = resolve_vram_limit_bytes(&config.limits, None)?;
        let residency_ceiling_bytes = resolve_memory_strategy_hot_tier_bytes(&config.limits, None)?;
        let graph_memory = analyze_model_memory(&model_directory.model_path);
        let minimum_useful_weight_budget_bytes = graph_memory
            .per_layer_weight_bytes
            .iter()
            .map(|layer| layer.bytes)
            .max()
            .unwrap_or(0);
        let memory_strategy_plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes,
            residency_ceiling_bytes,
            model_weight_bytes,
            // ORT backend uses its own kernels, not the CPU EP MatMulNBits f32
            // decode cache, so there is no #971 expansion to account for here.
            resident_f32_cache_bytes: 0,
            kv_config: plan_kv_config,
            graph: graph_memory,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes,
            default_dynamic_device_budget_bytes: None,
            // ORT backend: managed no-spill VMM is not available here, so the
            // plan stays keyed on an explicit byte limit as before #755.
            inferred_policy_enabled: matches!(config.limits.vram_limit, ResourceLimit::Bytes(_)),
            managed_vmm: matches!(config.limits.vram_limit, ResourceLimit::Bytes(_)),
            overrides: MemoryStrategyOverrides::default(),
            advisory_only: true,
            // ORT backend: no managed offload happens here (advisory only), but
            // keep the platform signal consistent with the native path.
            shared_memory_weight_fallback: cfg!(windows),
            force_managed_weight_streaming: force_managed_weight_streaming_enabled(),
        });
        log_memory_strategy_plan(&memory_strategy_plan, "single_model_ort");
        fail_explicit_vram_limit_without_offload(
            &memory_strategy_plan,
            session_options
                .execution_providers
                .iter()
                .any(|provider| !provider.caps.is_host()),
        )?;

        // ORT CUDA graph capture is opt-in via ONNX_GENAI_CUDA_GRAPH (default
        // off, resolved per-EP as `ResolvedEp::graph_capture_env`): it fails with
        // unconstructed OrtValue outputs on some Foundry exports. Note that
        // requesting it is not sufficient — the decode session only captures on
        // `DecodeKvMode::SharedBuffer`, so a model on the zero-copy-rebind path
        // runs uncaptured regardless. Native whole-step capture is separate.
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
        let shared_kv = shared_kv_offer(&session, &metadata, &model_directory.model_path);
        let MetadataResolution {
            metadata,
            decode_path,
        } = resolve_metadata_and_decode_path(metadata, shared_kv, session.graph_capture())?;

        let tokenizer = {
            let _span = onnx_genai_ort::prof_span!("engine.tokenizer_load");
            Tokenizer::from_file(&model_directory.tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
        };
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        // Stage: resource governor and batch scheduler.
        let (governor, scheduler) = build_governor_and_scheduler(
            &config,
            &model_directory,
            kv_model.as_ref(),
            &decode_path,
            authority_provider.as_ref(),
            &domain,
        )?;
        // Stage: draft-model loading. Kept before KV-cache allocation to preserve
        // the original constructor's fallible-step ordering.
        let draft = load_draft_model(&config, &environment, &session_options, &governor)?;

        // Stage: runtime KV-cache allocation, granted by the governor built above.
        let kv_cache = allocate_kv_cache(&config, kv_model.as_ref(), &governor)?;

        // Stage: speculative-assistant loading (mode resolution then per-mode heads).
        let (speculative_mode, resolved_mtp_config) =
            resolve_speculative_mode(config.speculative_mode.clone(), draft.is_some())?;
        let mtp = load_mtp_model(
            resolved_mtp_config,
            &session,
            &environment,
            &session_options,
            &model_directory,
        )?;
        let eagle3 =
            load_eagle3_model(&speculative_mode, &session, &environment, &session_options)?;

        let connector = {
            let _span = onnx_genai_ort::prof_span!("engine.connector_bridge");
            build_connector_bridge(&config.kv_connector, &model_directory, kv_model.as_ref())?
        };

        Ok(Self {
            workflow: None,
            decode_backend,
            metadata,
            metadata_hints,
            kv_cache,
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model,
            decode_path,
            scheduler,
            governor: Some(governor),
            sessions: HashMap::new(),
            _environment: Some(environment),
            session: Some(Box::new(session)),
            #[cfg(feature = "native-backend")]
            native_session: None,
            #[cfg(feature = "native-backend")]
            weight_placement: None,
            memory_strategy_plan,
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
            #[cfg(feature = "native-backend")]
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
            draft,
            mtp,
            eagle3,
            tokenizer: Some(tokenizer),
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
        native_device: crate::native_decode::NativeDecodeDevice,
        authority_provider: Option<&SharedMemoryAuthorityProvider>,
        authority_domain: &DeviceCompatibilityDomain,
    ) -> anyhow::Result<Self> {
        if config.draft_model.is_some() || !matches!(config.speculative_mode, SpeculativeMode::None)
        {
            // User-configured speculation (draft model / explicit mode) is still
            // unsupported on the native backend. Metadata-advertised proposers
            // (shared-KV, MTP) are resolved below and do NOT set
            // `config.speculative_mode`, so they never trip this guard.
            anyhow::bail!(
                "native decoder backend does not yet support user-configured speculative, MTP, EAGLE-3, or shared-KV generation; declare the proposer in inference metadata instead"
            );
        }
        if !matches!(&config.kv_connector.backend, KvConnectorBackend::Null) {
            anyhow::bail!("native decoder backend does not yet support external KV connectors");
        }
        let mut metadata = {
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
        // Auto-derive the decoder `io` port contract from the ONNX graph when the
        // package's `inference_metadata.yaml` declares none. This engages ONLY
        // for hybrid linear-attention decoders — graphs that expose recurrent
        // `conv_state`/`recurrent_state` state pairs alongside sparse dense KV
        // (qwen3.5/3.6). Their thin sidecars declare only `grouped_query_attention`
        // and no `io` block, so the KV-geometry / Resource-Governor sizing below
        // (and the native decode step driver) have no port contract and the load
        // fails `per-layer KV page geometry is unknown`. The derivation is
        // attribute/shape-driven (via `derive_decoder_io_from_graph`), never
        // model-name-gated, and mirrors the native decode driver's own
        // `derive_fallback_io`: a declared `io` block always wins, and pure-dense
        // decoders (no recurrent state pairs) yield no derived spec and keep their
        // existing load path unchanged. See #384 and the qwen3.5-27B enablement.
        maybe_fill_hybrid_io_from_graph(&mut metadata, &model_directory.model_path);
        admit_inference_metadata(&metadata)?;
        // Native MTP self-speculation seeds its draft head from a target hidden
        // output. The native decode session only records that hidden state when
        // the decode ABI names it, but the seed is declared by the MTP sidecar
        // rather than by the target's own graph contract, so it has to be
        // carried across. Install it as a *derived* ABI: the package keeps one
        // serialized representation and gains no second writable statement of
        // its own ports. An ABI that already names a hidden output wins, and a
        // directory that declares no MTP sidecar is left exactly as it was.
        if let Some(seeded) = mtp_seeded_decoder_io(&metadata, &model_directory.root) {
            metadata.set_derived_decoder_io(seeded);
        }
        let tokenizer = {
            let _span = onnx_genai_ort::prof_span!("engine.tokenizer_load");
            Tokenizer::from_file(&model_directory.tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
        };
        let fim_config = load_fim_config_from_model_dir(&model_directory.root)?;
        let kv_model = {
            let _span = onnx_genai_ort::prof_span!("engine.native_kv_model_info");
            let model_io = metadata.decoder_io();
            let kv_inputs = model_io
                .and_then(|io| io.kv_inputs.clone())
                .unwrap_or_default();
            let kv_outputs = model_io
                .and_then(|io| io.kv_outputs.clone())
                .unwrap_or_default();
            let graph_io = onnx_genai_ort::graph_io_from_model_path_for_kv_pairs(
                &model_directory.model_path,
                &kv_inputs,
                &kv_outputs,
            )
            .map_err(|e| {
                anyhow::anyhow!("Failed to read native decoder graph I/O for KV geometry: {e}")
            })?;
            infer_kv_model_info(&graph_io, model_io, config.page_size, config.kv_cache_dtype)
                .context("failed to infer native decoder KV geometry from model graph I/O")?
        };
        let model_io = metadata.decoder_io();
        let governor_kv_config = match kv_model.as_ref() {
            Some(kv_model) => governor_native_kv_config(Some(kv_model), &config)?,
            None if model_io_declares_only_fixed_state(model_io) => {
                governor_no_paged_kv_config(&config)?
            }
            None => governor_kv_config(None, &config)?,
        };
        let model_weight_bytes = device_weight_package_bytes(&model_directory.model_path);
        let graph_memory = analyze_model_memory(&model_directory.model_path);
        let minimum_useful_weight_budget_bytes = graph_memory
            .per_layer_weight_bytes
            .iter()
            .map(|layer| layer.bytes)
            .max()
            .unwrap_or(0);
        // The device (VRAM) capacity stays honestly `None` when it cannot be
        // measured (#947): it is reported verbatim as `resolved_device_budget`
        // and never borrows the host tier. The residency verdict is a separate,
        // still-knowable fact, sized against the physical hot tier the weights
        // will really occupy: the measured VRAM budget when the native load
        // targets a queryable CUDA device, else the measured host-RAM ceiling
        // the CPU-native weights actually live in. A model that plainly fits the
        // host RAM it will occupy must read `FullResident`, not `Unknown`, and no
        // device number is fabricated to make it so (an explicit `--vram-limit`
        // still resolves to `Some`; if host RAM itself is unmeasurable, the
        // residency ceiling stays `None`).
        let resolved_vram_bytes =
            resolve_vram_limit_bytes(&config.limits, native_device.cuda_index())?;
        let residency_ceiling_bytes =
            resolve_memory_strategy_hot_tier_bytes(&config.limits, native_device.cuda_index())?;
        #[cfg(feature = "native-cuda")]
        let required_device_non_weight_bytes = if matches!(
            native_device,
            crate::native_decode::NativeDecodeDevice::Cuda { .. }
        ) {
            let metadata_max_len = metadata
                .model
                .as_ref()
                .and_then(|model| model.max_sequence_length);
            let max_context = crate::native_decode::configured_cuda_kv_max_len()?
                .map(|limit| metadata_max_len.map_or(limit, |metadata| limit.min(metadata)))
                .or(metadata_max_len);
            let kv_bytes = governor_kv_config
                .bytes_per_token()
                .zip(max_context)
                .map_or(0, |(bytes, max_context)| {
                    bytes.saturating_mul(max_context as u64)
                });
            let graph = onnx_runtime_loader::load_model(&model_directory.model_path)
                .context("loading native graph for recurrent-state memory planning")?;
            let recurrent_bytes =
                crate::native_decode::recurrent_state_bytes_from_graph(&graph, model_io)?;
            kv_bytes.saturating_add(recurrent_bytes)
        } else {
            0
        };
        #[cfg(not(feature = "native-cuda"))]
        let required_device_non_weight_bytes = 0;
        #[cfg(feature = "native-cuda")]
        let cuda_env_policy = onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env();
        #[cfg(feature = "native-cuda")]
        let memory_strategy_overrides = memory_strategy_overrides_from_cuda_env(cuda_env_policy);
        #[cfg(not(feature = "native-cuda"))]
        let memory_strategy_overrides = MemoryStrategyOverrides::default();
        let native_cuda_load = matches!(
            native_device,
            crate::native_decode::NativeDecodeDevice::Cuda { .. }
        );
        // #971: on the native CPU path the MatMulNBits kernel builds a resident
        // dequantised f32 weight cache for models whose quantisation takes the
        // f32 decode path. That cache is held for the whole session, so the plan
        // must account for it as real resident weight bytes. We ask the CPU EP
        // (which owns the kernel dispatch) how many extra bytes the cache costs,
        // rather than re-deriving the rule here where it would drift from the
        // kernel (#947). CUDA decode uses different kernels, so it never applies.
        //
        // #1056: the weight-transpose cache is the third such buffer. It holds
        // one full `K x N` f32/f16 copy per transposed constant weight for the
        // session (populated on all platforms by `Gemm` with `transB`, and on
        // Apple also by `MatMul`). We ask the CPU EP to predict its bytes from
        // the same graph and fold them into the resident total, so the single
        // admission verdict below governs it alongside the other two.
        let resident_f32_cache_bytes = if native_cuda_load {
            0
        } else {
            match onnx_runtime_loader::load_model(&model_directory.model_path) {
                Ok(graph) => onnx_runtime_ep_cpu::resident_dequant_f32_cache_bytes(&graph)
                    .saturating_add(onnx_runtime_ep_cpu::weight_transpose_cache_predicted_bytes(
                        &graph,
                    ))
                    .saturating_add(onnx_runtime_ep_cpu::matmul_dense_cache_predicted_bytes(
                        &graph,
                    ))
                    .saturating_add(
                        onnx_runtime_ep_cpu::qlinear_accumulator_budget_predicted_bytes(&graph),
                    )
                    .saturating_add(onnx_runtime_ep_cpu::qlinear_packed_b_predicted_bytes(
                        &graph,
                    )),
                Err(_) => 0,
            }
        };
        let explicit_vram_bytes = matches!(config.limits.vram_limit, ResourceLimit::Bytes(_));
        // #755: managed no-spill VMM is the default on the native CUDA path
        // (unless the legacy allocator opt-out is set). On other backends the
        // managed path is unavailable, so it stays keyed on an explicit byte
        // limit as before.
        #[cfg(feature = "native-cuda")]
        let managed_vmm = if native_cuda_load {
            managed_vmm_default_enabled()
        } else {
            explicit_vram_bytes
        };
        #[cfg(not(feature = "native-cuda"))]
        let managed_vmm = explicit_vram_bytes;
        let memory_strategy_plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes,
            residency_ceiling_bytes,
            model_weight_bytes,
            resident_f32_cache_bytes,
            kv_config: governor_kv_config,
            graph: graph_memory,
            required_device_non_weight_bytes,
            minimum_useful_weight_budget_bytes,
            #[cfg(feature = "native-cuda")]
            default_dynamic_device_budget_bytes: Some(
                onnx_runtime_ep_cuda::DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES,
            ),
            #[cfg(not(feature = "native-cuda"))]
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: managed_vmm || explicit_vram_bytes,
            managed_vmm,
            overrides: memory_strategy_overrides,
            advisory_only: !native_cuda_load,
            // #864: on Windows/WDDM the OS shared-memory fallback pages over-budget
            // weights from host RAM over PCIe, ~30x faster than managed streaming
            // for the single-touch decode pattern. Gate the auto-disable on that
            // platform property (cfg!(windows)); on Linux there is no such fallback
            // so managed streaming stays the only over-budget path (#783).
            shared_memory_weight_fallback: cfg!(windows),
            force_managed_weight_streaming: force_managed_weight_streaming_enabled(),
        });
        log_memory_strategy_plan(&memory_strategy_plan, "single_model_native");
        // #971: tell the CPU EP whether the governor admitted the resident f32
        // decode cache. When declined (expanded footprint would not fit the
        // budget), the kernel dequantises on the fly per call instead of holding
        // the ~8x cache resident — slower per token but avoids paging.
        onnx_runtime_ep_cpu::set_resident_dequant_f32_cache_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        // #1027: the int4 accuracy_level=0 MLAS SQNBit route holds a packed
        // buffer (CompFp32 packs to ~1x the int4 bytes) plus its retained scale
        // copy, and the shape-keyed kernel cache materializes one per prefill
        // and decode activation shape -- ~2.5x the int4 bytes beside the mapped
        // weight for the session (#1051 corrected the earlier per-copy under-
        // count). Its bytes are folded into `resident_f32_cache_bytes`, so the
        // admission verdict governs it: when declined, the kernel keeps the
        // borrowed zero-copy int4 path (byte-identical, only slower on x86)
        // instead of doubling the weight footprint.
        onnx_runtime_ep_cpu::set_mlas_sqnbit_packing_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        // #1056: the weight-transpose cache (one resident `K x N` f32/f16 copy
        // per transposed constant weight) is the third buffer folded into
        // `resident_f32_cache_bytes`, so the same verdict governs it. When
        // declined, the `MatMul`/`Gemm` kernels transpose per call and retain
        // nothing instead of holding the session-lifetime copies resident over
        // budget. That is byte-identical everywhere except the x86 f16/bf16
        // decode GEMV, where admission picks the kernel too: a declined
        // session reads the [K, N] weight in place with a different (slower,
        // and further from an f64 reference) accumulation order, so its decode
        // output differs in the low bits. See
        // `set_weight_transpose_cache_enabled`.
        onnx_runtime_ep_cpu::set_weight_transpose_cache_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        // #1056: the per-kernel `MatMulPrepack::dense` widened-f32 cache (one
        // resident `4 * K * N` f32 copy per constant operand that is not already
        // contiguous f32 -- f16/bf16/f64 or strided) is the fourth buffer folded
        // into `resident_f32_cache_bytes`, so the same verdict governs it. When
        // declined, the `MatMul`/`FusedMatMulBias` kernels widen the operand per
        // call and retain nothing (byte-identical output, only slower) instead of
        // holding the session-lifetime copies resident over budget.
        onnx_runtime_ep_cpu::set_matmul_dense_cache_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        // #1133: the QLinearMatMul process-wide accumulator scratch budget and
        // the constant-`B` MLAS pre-pack are the fifth and sixth buffers folded
        // into `resident_f32_cache_bytes`, governed by the same verdict. When
        // declined, the kernel reallocates the `i32` accumulator per call and
        // takes the unpacked GEMM (densifying `B` per call) -- byte-identical
        // output, only slower -- instead of retaining either buffer over budget.
        onnx_runtime_ep_cpu::set_qlinear_accumulator_budget_admitted(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        onnx_runtime_ep_cpu::set_qlinear_packed_b_enabled(
            memory_strategy_plan.f32_weight_cache_admitted,
        );
        #[cfg(feature = "native-cuda")]
        let cuda_offload_resolution =
            cuda_offload_resolution_from_plan(&native_device, &memory_strategy_plan);
        #[cfg(feature = "native-cuda")]
        let cuda_offload_policy = cuda_offload_resolution.map(|resolution| resolution.policy);
        #[cfg(feature = "native-cuda")]
        let governed_physical_pool = uses_governed_physical_pool(
            cuda_offload_resolution,
            dynamic_lending_enabled(),
            onnx_runtime_ep_cuda::vmm_allocator::production_physical_pool_enabled(),
        );
        #[cfg(feature = "native-cuda")]
        let weight_reservation_bytes = cuda_weight_startup_reservation(
            model_weight_bytes,
            cuda_offload_resolution,
            governed_physical_pool,
            governor_kv_config.page_size_bytes,
        );
        #[cfg(feature = "native-cuda")]
        tracing::info!(
            // The device actually resolved, not the compiled-in feature. This
            // line used to say "CUDA" on every run of a CUDA-enabled build,
            // including ones that resolved to the CPU because the model declared
            // no execution provider -- which reads as confirmation that you are
            // on the GPU when you are not (#1064).
            device = ?native_device,
            managed_no_spill = cuda_offload_resolution
                .is_some_and(|resolution| resolution.policy.managed_no_spill),
            dynamic_lending = dynamic_lending_enabled(),
            governed_physical_pool,
            weight_reservation_bytes,
            "resolved device-memory strategy before governor creation"
        );
        #[cfg(not(feature = "native-cuda"))]
        let weight_reservation_bytes = device_weight_reservation_for(
            model_weight_bytes,
            None,
            governor_kv_config.page_size_bytes,
        );
        let governor = {
            let _span = onnx_genai_ort::prof_span!("engine.resource_governor");
            EngineResourceGovernor::new_with_authority_and_reservation(
                config.limits.clone(),
                config.allow_runtime_override,
                governor_kv_config,
                model_weight_bytes,
                weight_reservation_bytes,
                native_device.cuda_index(),
                authority_provider,
                Some(authority_domain),
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
        populate_scheduler_bytes_per_token(&mut scheduler_config, governor_kv_config)?;
        let connector = {
            let _span = onnx_genai_ort::prof_span!("engine.connector_bridge");
            build_connector_bridge(&config.kv_connector, &model_directory, kv_model.as_ref())?
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
                    #[cfg(feature = "native-cuda")]
                    cuda_offload_policy,
                    #[cfg(feature = "native-cuda")]
                    cuda_memory_governor: std::sync::Arc::new(governor.device_authority()),
                    #[cfg(feature = "native-cuda")]
                    process_memory_manager: governor.process_memory_manager(),
                    io: metadata.decoder_io(),
                    metadata_max_len: metadata
                        .model
                        .as_ref()
                        .and_then(|model| model.max_sequence_length),
                    key_sequence_lengths_policy: crate::decode::key_sequence_lengths_policy(
                        &metadata,
                    ),
                    decode_precision: config.decode_precision,
                    decode_batch: config.native_decode_batch,
                },
            )
            .map_err(|error| anyhow::anyhow!("Failed to load native decoder session: {error:#}"))?
        };
        let mut native_session = native_session;
        // #1362: a prefill forward's activations scale with the tokens in it, so
        // an unchunked prompt makes peak device memory a function of prompt
        // length. The flat ORT pipeline has read this metadata since chunked
        // prefill was introduced; the native backend ignored it, so a model that
        // declared a chunk size got it honored on one backend only.
        native_session.set_prefill_chunk_size(
            metadata
                .model
                .as_ref()
                .and_then(|model| model.runtime_configurable.as_ref())
                .and_then(|runtime| runtime.chunked_prefill.as_ref())
                .and_then(|chunked| chunked.chunk_size),
        );
        #[cfg(feature = "native-cuda")]
        let cuda_offload_policy = reconcile_cuda_offload_budget_after_native_load(
            &native_session,
            metadata
                .model
                .as_ref()
                .and_then(|model| model.max_sequence_length),
            // Native CUDA load always has a measured device capacity, so the
            // resolved VRAM budget is `Some`. `unwrap_or(0)` only guards the
            // impossible unknown-device-on-CUDA case.
            governor.snapshot().resolved_limits.vram_bytes.unwrap_or(0),
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
        #[cfg(feature = "native-cuda")]
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
        let native_kv_capacity = native_session
            .cuda_kv_debug_stats()
            .map(|stats| (stats.hard_max_len, stats.max_len_source));
        let native_kv_max_context = native_kv_reservation_max_context(
            native_kv_capacity,
            metadata
                .model
                .as_ref()
                .and_then(|model| model.max_sequence_length),
        );
        match native_kv_max_context {
            Some((max_context, max_context_source)) => {
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
                                 KV max length source: {max_context_source}; raise the limit for \
                                 that tier, set ONNX_GENAI_CUDA_KV_MAX_LEN lower when supported, \
                                 or re-export the model with a shorter declared context"
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
        // MTP is the one native proposer kind; the effective mode is decided
        // below once the MTP loader has reported.
        let shared_kv_mode = SpeculativeMode::None;
        let environment = {
            let _span = onnx_genai_ort::prof_span!("engine.ort_environment");
            Environment::new("onnx-genai-engine")
                .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {e}"))?
        };
        // Sidecar-declared MTP self-speculation: the pure-attention MTP head
        // loads on the ORT `environment` (ORT CUDA EP), seeded from the native
        // hybrid target's declared hidden output. Yields `None` for every
        // non-MTP model, preserving the plain native load path exactly.
        let (mtp, mtp_mode) = load_native_mtp_proposer(
            &metadata,
            &model_directory,
            &environment,
            session_options,
            native_device,
        )?;
        // A model runs at most one proposer kind; shared-KV drafting and MTP are
        // mutually exclusive, so at most one of these may be anything but
        // `None`.
        let speculative_mode = match (&shared_kv_mode, &mtp_mode) {
            (SpeculativeMode::None, mode) => mode.clone(),
            (mode, SpeculativeMode::None) => mode.clone(),
            _ => anyhow::bail!(
                "native metadata resolved both a shared-KV and an MTP proposer; only one speculative proposer is supported per model"
            ),
        };
        // Native CUDA can replace its conservative startup weight reservation
        // with the provider's smaller governed residency pool during load.
        // Size the scheduler only after that handoff so its static maximum does
        // not remain pinned to the single KV page deliberately left at startup.
        let scheduler =
            Scheduler::with_byte_budget(scheduler_config, governor.byte_budget_after_native_load());

        Ok(Self {
            workflow: None,
            decode_backend: EngineDecodeBackend::Native,
            metadata,
            metadata_hints,
            kv_cache: PagedKvCache::new(config.page_size, native_kv_pages),
            prefix_cache: PrefixCache::new(),
            token_prefix_cache: Vec::new(),
            kv_model,
            decode_path: ModelDecodePath::Generic,
            scheduler,
            governor: Some(governor),
            sessions: HashMap::new(),
            session: None,
            native_session: Some(native_session),
            weight_placement,
            memory_strategy_plan,
            native_sessions: HashMap::new(),
            native_active_session: None,
            native_session_counter: 0,
            native_access_counter: 0,
            native_default_session: None,
            native_max_sessions: config.native_max_sessions,
            native_recurrent_prefix_stats: RecurrentPrefixCacheStats::default(),
            draft: None,
            mtp,
            eagle3: None,
            tokenizer: Some(tokenizer),
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

/// Fill the resolved decode ABI from the ONNX graph's declared ports when the
/// package's sidecar declares no `io` block AND the graph is a hybrid
/// linear-attention decoder (exposes recurrent `conv_state`/`recurrent_state`
/// state pairs).
///
/// Hybrid SSM/attention decoders (qwen3.5/3.6) ship a thin
/// `inference_metadata.yaml` that names only `grouped_query_attention` and no
/// `io` port contract. Without that contract the Resource Governor cannot derive
/// the per-layer KV page byte geometry (only the periodic full-attention layers
/// hold dense KV) and native load fails with `per-layer KV page geometry is
/// unknown`. The graph itself already encodes the exact topology, so this
/// derives the sparse dense `kv_inputs`/`kv_outputs` and the fixed recurrent
/// `state_pairs` directly from the port inventory via
/// [`GenAiConfig::derive_decoder_io_from_graph`].
///
/// This is deliberately attribute/shape-driven, never model-name-gated, and
/// engages only for the recurrent-hybrid case (the safety gate is a non-empty
/// derived `state_pairs`, mirroring the native decode driver's
/// `derive_fallback_io`). An already-resolved ABI always wins; pure-dense
/// decoders (no state pairs) are left untouched and keep their existing load
/// path. The result is installed as a *derived* ABI rather than written into
/// the deprecated serialized block, so it stays one representation.
#[cfg(feature = "native-backend")]
fn maybe_fill_hybrid_io_from_graph(metadata: &mut InferenceMetadata, model_path: &Path) {
    // A workflow-recognized or legacy-declared ABI is always authoritative.
    if metadata.decoder_io().is_some() {
        return;
    }
    let Some(graph_info) = crate::engine::decoder_graph_info_from_model_path(model_path) else {
        return;
    };
    let Some(io) =
        onnx_genai_genai_config::GenAiConfig::derive_model_io_spec_from_graph(&graph_info)
    else {
        return;
    };
    metadata.set_derived_decoder_io(io);
}

/// The resolved decode ABI with an MTP sidecar's target hidden output filled in.
///
/// `None` unless the package resolves an ABI that names no hidden output *and*
/// the model directory declares an MTP sidecar that names one, so every other
/// model keeps exactly the ABI it resolved.
#[cfg(feature = "native-backend")]
fn mtp_seeded_decoder_io(
    metadata: &InferenceMetadata,
    model_root: &Path,
) -> Option<onnx_genai_metadata::ModelIoSpec> {
    let io = metadata.decoder_io()?;
    let descriptor = onnx_genai_metadata::detect_speculator(model_root)?;
    let onnx_genai_metadata::SpeculatorProposerStatus::Mtp(spec) = descriptor.proposer else {
        return None;
    };
    decoder_io_seeded_with(io, &spec.target_hidden_output)
}

/// Name `target_hidden_output` as the ABI's hidden output, or decline.
///
/// Declines when the ABI already names one -- a declared port is authoritative
/// and a sidecar must not redirect the target's own contract -- and when the
/// sidecar names nothing, which would otherwise install an empty port name that
/// reads as "declared" everywhere downstream.
#[cfg(feature = "native-backend")]
fn decoder_io_seeded_with(
    io: &onnx_genai_metadata::ModelIoSpec,
    target_hidden_output: &str,
) -> Option<onnx_genai_metadata::ModelIoSpec> {
    if !io.hidden_output.as_deref().unwrap_or_default().is_empty() {
        return None;
    }
    if target_hidden_output.is_empty() {
        return None;
    }
    let mut seeded = io.clone();
    seeded.hidden_output = Some(target_hidden_output.to_string());
    Some(seeded)
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
    decode_path: ModelDecodePath,
}

fn load_inference_metadata(model_directory: &ModelDirectory) -> anyhow::Result<InferenceMetadata> {
    let _span = onnx_genai_ort::prof_span!("engine.metadata_load");
    if let Some(metadata_path) = &model_directory.metadata_path {
        return onnx_genai_metadata::load_metadata(metadata_path)
            .map_err(|e| anyhow::anyhow!("Failed to load metadata: {e}"));
    }
    if let Some(compat) = genai_config_compat_metadata_from_model_path(
        model_directory.genai_config_path.as_deref(),
        &model_directory.model_path,
    )? {
        tracing::info!(
            "No inference_metadata.yaml found; derived inference metadata from genai_config.json (onnxruntime-genai compatibility)"
        );
        return Ok(compat);
    }
    tracing::warn!("No inference metadata found, using defaults");
    Ok(default_inference_metadata())
}

fn admit_inference_metadata(metadata: &InferenceMetadata) -> anyhow::Result<()> {
    let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
    let report = onnx_genai_metadata::validate_structure_and_capabilities(metadata, &runtime_caps);
    if !report.structural.is_empty() {
        anyhow::bail!("Invalid inference metadata: {:?}", report.structural);
    }
    if !report.unsupported_capabilities.is_empty() {
        anyhow::bail!(
            "Unsupported inference metadata capabilities: {}",
            report.unsupported_capabilities.join(", ")
        );
    }
    Ok(())
}

fn resolve_metadata_and_decode_path(
    metadata: InferenceMetadata,
    shared_kv: crate::decode::SharedKvOffer,
    capture_requested: bool,
) -> anyhow::Result<MetadataResolution> {
    // Canonical metadata is the sole execution contract. A bare decoder may
    // not bypass workflow, serving, adapter, or policy requirements it cannot
    // execute; admission must fail before decode-path selection.
    admit_inference_metadata(&metadata)?;

    let sliding_window = crate::decode::sliding_window_from_metadata(&metadata)?;
    let sink_tokens = crate::decode::sink_tokens_from_metadata(&metadata);
    let decode_path = {
        let _span = onnx_genai_ort::prof_span!("engine.detect_decode_path");
        detect_model_decode_path(
            metadata.decoder_io(),
            sliding_window,
            sink_tokens,
            shared_kv,
        )?
    };
    // Report the resolved decode path together with whether the session even
    // asked for CUDA-graph capture. Both matter and neither was visible before:
    // ORT capture requires `DecodeKvMode::SharedBuffer` (see
    // `will_sample_on_device`), so a model that lands on zero-copy-rebind runs
    // uncaptured no matter what `enable_cuda_graph` says. A benchmark can
    // therefore compare two backends on two different KV strategies and read
    // like a backend comparison. Emitting this once at load makes it visible in
    // every CLI, server and benchmark run.
    tracing::info!(
        decode_path = %decode_path.summary(),
        graph_capture_requested = capture_requested,
        "resolved decode path"
    );
    eprintln!(
        "decode_path: {} graph_capture_requested={capture_requested}",
        decode_path.summary()
    );
    Ok(MetadataResolution {
        metadata,
        decode_path,
    })
}

/// Resolve what this deployment can offer for an aliased KV buffer.
///
/// A shared buffer is a fixed reservation, so it needs a capacity, and the only
/// capacity known at load time is the one the package declares as its context
/// window. Without that bound the runtime declines to reserve rather than
/// guessing a size.
fn shared_kv_offer(
    session: &Session,
    metadata: &InferenceMetadata,
    model_path: &Path,
) -> crate::decode::SharedKvOffer {
    let max_len = metadata
        .model
        .as_ref()
        .and_then(|model| model.max_sequence_length);
    // The EP capability is necessary but not sufficient. Whether a
    // capacity-padded past is legal is a property of the attention *operator*:
    // the standard opset `Attention` derives `total_sequence_length` from the
    // past tensor's own extent and cross-checks it against the mask, so a
    // capacity-sized past makes the two disagree and ORT refuses to run. Ask the
    // graph before offering a shared buffer, so such a model falls back to the
    // exact-length rebind path instead of failing at the first decode step.
    //
    // Best-effort and short-circuited: the graph is only read when a shared
    // buffer is otherwise on the table, and a graph that cannot be read leaves
    // the offer as the session alone described it.
    let ep_supports = session.supports_fixed_capacity_present_binding();
    let present_binding_supported = ep_supports
        && (max_len.is_none()
            || match onnx_runtime_loader::load_model(model_path) {
                Ok(graph) => {
                    let accepts = crate::decode::graph_accepts_padded_past(&graph);
                    let explicit_kv_length =
                        crate::decode::graph_uses_explicit_kv_length_attention(&graph);
                    if !accepts {
                        tracing::debug!(
                            "decoder graph uses the standard opset Attention op, which \
                             cross-checks the attention mask against the past KV extent; \
                             declining the shared KV buffer for this deployment"
                        );
                    } else if explicit_kv_length {
                        tracing::debug!(
                            "decoder graph uses attention with an explicit valid KV length; \
                             fixed-capacity present binding is graph-compatible"
                        );
                    }
                    accepts
                }
                Err(_) => true,
            });
    crate::decode::SharedKvOffer {
        present_binding_supported,
        max_len,
    }
}

fn build_governor_and_scheduler(
    config: &EngineConfig,
    model_directory: &ModelDirectory,
    kv_model: Option<&KvModelInfo>,
    decode_path: &ModelDecodePath,
    authority_provider: Option<&SharedMemoryAuthorityProvider>,
    authority_domain: &DeviceCompatibilityDomain,
) -> anyhow::Result<(EngineResourceGovernor, Scheduler)> {
    let governor_kv_config = match (kv_model, decode_path) {
        (None, ModelDecodePath::StaticCache { .. } | ModelDecodePath::Generic) => {
            governor_no_paged_kv_config(config)?
        }
        _ => governor_kv_config(kv_model, config)?,
    };
    let governor = {
        let _span = onnx_genai_ort::prof_span!("engine.resource_governor");
        EngineResourceGovernor::new_with_authority(
            config.limits.clone(),
            config.allow_runtime_override,
            governor_kv_config,
            device_weight_package_bytes(&model_directory.model_path),
            None,
            authority_provider,
            Some(authority_domain),
        )
        .map_err(|error| anyhow::anyhow!("Failed to initialize Resource Governor: {error}"))?
    };
    let mut scheduler_config = config.scheduler.clone();
    if scheduler_config.bytes_per_token.is_none() {
        scheduler_config.bytes_per_token = match governor_kv_config.bytes_per_token() {
            Some(bytes_per_token) => Some(bytes_per_token),
            None if governor_kv_config.page_geometry_required => {
                required_bytes_per_token_from_kv_config(governor_kv_config)?
            }
            None => None,
        };
    }

    let scheduler = Scheduler::with_byte_budget(scheduler_config, governor.byte_budget());
    Ok((governor, scheduler))
}

pub(crate) fn validate_shared_authority_limit(
    provider: Option<&SharedMemoryAuthorityProvider>,
    domain: &DeviceCompatibilityDomain,
    limit: ResourceLimit,
) -> anyhow::Result<()> {
    if let Some(provider) = provider {
        provider.validate_limit(domain, limit)?;
    }
    Ok(())
}

pub(crate) fn session_device_domain(
    session_options: &SessionOptions,
) -> anyhow::Result<DeviceCompatibilityDomain> {
    let Some(provider) = session_options
        .execution_providers
        .iter()
        .find(|provider| !provider.caps.is_host())
    else {
        return Ok(DeviceCompatibilityDomain::Host);
    };
    let index = provider.caps.device_id().unwrap_or(0);
    let index = u32::try_from(index).map_err(|_| {
        anyhow::anyhow!(
            "execution provider {} has negative device id {index}",
            provider.caps.name
        )
    })?;
    if provider.caps.is_nvidia() {
        Ok(DeviceCompatibilityDomain::Cuda(index))
    } else {
        Ok(DeviceCompatibilityDomain::Accelerator {
            backend: provider.caps.name.to_ascii_lowercase(),
            index,
        })
    }
}

#[cfg(feature = "native-backend")]
fn native_device_domain(
    device: &crate::native_decode::NativeDecodeDevice,
) -> DeviceCompatibilityDomain {
    match device {
        crate::native_decode::NativeDecodeDevice::Cpu => DeviceCompatibilityDomain::Host,
        crate::native_decode::NativeDecodeDevice::Cuda { index } => {
            DeviceCompatibilityDomain::Cuda(index.unwrap_or(0))
        }
        crate::native_decode::NativeDecodeDevice::Plugin { provider_name, .. } => {
            DeviceCompatibilityDomain::Accelerator {
                backend: provider_name.to_ascii_lowercase(),
                index: 0,
            }
        }
    }
}

fn required_bytes_per_token_from_kv_config(
    kv_config: ModelKvConfig,
) -> anyhow::Result<Option<u64>> {
    kv_config.bytes_per_token().map(Some).with_context(|| {
        format!(
            "cannot derive scheduler bytes_per_token because KV page byte geometry is unknown \
             for {} token(s) per page; fix by declaring kv_inputs and \
             kv_outputs so admission uses real KV memory costs",
            kv_config.tokens_per_page
        )
    })
}

#[cfg(feature = "native-backend")]
fn populate_scheduler_bytes_per_token(
    scheduler: &mut onnx_genai_scheduler::SchedulerConfig,
    kv_config: ModelKvConfig,
) -> anyhow::Result<()> {
    if scheduler.bytes_per_token.is_none() {
        scheduler.bytes_per_token = match kv_config.bytes_per_token() {
            Some(bytes_per_token) => Some(bytes_per_token),
            None if kv_config.page_geometry_required => {
                required_bytes_per_token_from_kv_config(kv_config)?
            }
            None => None,
        };
    }
    Ok(())
}

fn load_draft_model(
    config: &EngineConfig,
    environment: &Environment,
    session_options: &SessionOptions,
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
            .and_then(|metadata| metadata.decoder_io().cloned());
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
            detect_model_decode_path(None, None, 0, crate::decode::SharedKvOffer::default())?;
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
/// budget by an unavailable or stale page-size estimate — on a machine with 8
/// GiB of device memory, a token-count placeholder would resolve to hundreds of
/// millions of pages, and the pool would try to allocate a `Page` for every one
/// of them. The page count has to come from the geometry the pages will
/// actually have.
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

#[cfg(feature = "native-backend")]
fn native_kv_reservation_max_context(
    native_capacity: Option<(usize, String)>,
    metadata_max_len: Option<usize>,
) -> Option<(usize, String)> {
    native_capacity
        .filter(|(max_len, _)| *max_len != usize::MAX)
        .or_else(|| {
            metadata_max_len.map(|max_len| (max_len, "model.max_sequence_length".to_owned()))
        })
}

/// The temporary startup reservation, given the package size and offload budget.
///
/// Native CUDA offload replaces this placeholder with the provider-owned
/// residency lease after session load. Leaving one KV page unreserved prevents
/// the governor from taking the "reservation does not fit; drop it" warning path
/// that hid #712 while preserving the later ledger-enforced admission point.
#[cfg(any(feature = "native-backend", test))]
fn device_weight_reservation_for(
    package_bytes: u64,
    offload_budget: Option<u64>,
    kv_page_size_bytes: Option<u64>,
) -> u64 {
    match offload_budget {
        // A budget larger than the model cannot be filled, so the device still
        // holds at most the package.
        Some(budget) => {
            let reservation = budget.min(package_bytes);
            reservation.saturating_sub(kv_page_size_bytes.unwrap_or(0).min(reservation))
        }
        None => package_bytes,
    }
}

#[cfg(feature = "native-cuda")]
#[derive(Clone, Copy, Debug)]
struct CudaOffloadResolution {
    policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
    device_budget_is_override: bool,
    auto_enabled_from_vram_limit: bool,
}

#[cfg(feature = "native-cuda")]
fn dynamic_lending_enabled() -> bool {
    !std::env::var("ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Whether the managed no-spill VMM allocator is the default for this process.
///
/// Since #755 managed VMM is on without a flag on the native CUDA path. The
/// legacy ungoverned allocator remains reachable for one release through an
/// explicit opt-out: `ONNX_GENAI_LEGACY_ALLOCATOR=1` (the documented #755 knob),
/// or the pre-#755 `ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING=0`, which also restored
/// the compatibility fallback and is honored for back-compat.
/// Whether `ONNX_GENAI_MANAGED_WEIGHT_STREAMING` forces the managed
/// weight-streaming path even where the WDDM shared-memory fallback would
/// otherwise be auto-preferred (#864). Unconditional: read on every native load
/// (any backend) so the operator override is honored regardless of features.
pub(crate) fn force_managed_weight_streaming_enabled() -> bool {
    force_managed_weight_streaming_from_env_value(
        std::env::var(MANAGED_WEIGHT_STREAMING_ENV).ok().as_deref(),
    )
}

#[cfg(feature = "native-cuda")]
pub(crate) fn managed_vmm_default_enabled() -> bool {
    let legacy_allocator_opt_out =
        std::env::var("ONNX_GENAI_LEGACY_ALLOCATOR").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
    !legacy_allocator_opt_out && dynamic_lending_enabled()
}

#[cfg(feature = "native-cuda")]
fn uses_governed_physical_pool(
    resolution: Option<CudaOffloadResolution>,
    lending_enabled: bool,
    configured_pool_enabled: bool,
) -> bool {
    resolution.is_some_and(|resolution| {
        (resolution.policy.enabled && configured_pool_enabled)
            || (resolution.policy.managed_no_spill && lending_enabled)
    })
}

#[cfg(feature = "native-cuda")]
fn cuda_weight_startup_reservation(
    model_weight_bytes: u64,
    resolution: Option<CudaOffloadResolution>,
    governed_physical_pool: bool,
    kv_page_size_bytes: Option<u64>,
) -> u64 {
    if governed_physical_pool {
        // The physical pool charges each handle to this authority, including
        // resident weights when managed VMM does not need weight offload.
        // Reserving package bytes here would charge the same weights twice.
        return 0;
    }
    device_weight_reservation_for(
        model_weight_bytes,
        resolution.and_then(|resolution| {
            resolution.policy.enabled.then(|| {
                resolution
                    .policy
                    .device_budget_bytes
                    .unwrap_or(onnx_runtime_ep_cuda::DEFAULT_DEVICE_OFFLOAD_BUDGET_BYTES)
            })
        }),
        kv_page_size_bytes,
    )
}

#[cfg(feature = "native-cuda")]
fn cuda_offload_resolution_from_plan(
    native_device: &crate::native_decode::NativeDecodeDevice,
    plan: &MemoryStrategyPlan,
) -> Option<CudaOffloadResolution> {
    if !matches!(
        native_device,
        crate::native_decode::NativeDecodeDevice::Cuda { .. }
    ) {
        return None;
    }
    let application = plan.runtime_application();
    if !application.weight_offload_enabled
        && !application.managed_no_spill
        && application.device_budget_bytes.is_none()
    {
        return None;
    }
    Some(CudaOffloadResolution {
        policy: cuda_policy_from_memory_strategy_plan(plan),
        device_budget_is_override: application.device_budget_is_override,
        auto_enabled_from_vram_limit: application.auto_enabled_from_vram_limit,
    })
}
/// The device-tier KV bytes to withhold from the elastic weight budget: the
/// KV context length to keep statically reserved under the elastic weight
/// budget: the initial KV bucket the engine commits at load, capped at the
/// declared max context.
///
/// This is the KV a sequence can always commit without reclaiming a weight page
/// (prefill and the first decode step), so it is the safe floor to reserve
/// while everything above it is lent to weights and reclaimed on demand as the
/// sequence grows (issue #857).
#[cfg(feature = "native-cuda")]
fn elastic_kv_floor_context(max_context: usize) -> usize {
    // The engine commits KV in power-of-two buckets whose smallest value is the
    // configured minimum bucket; `kv_capacity_bucket(1, ..)` is that first
    // bucket, which is what load commits before any decode step runs.
    onnx_genai_kv::kv_capacity_bucket(1, max_context).min(max_context)
}

/// The elastic weight-offload budget: the elastic device availability (resolved
/// VRAM minus the KV floor and recurrent state) less a headroom margin, but
/// never below what the static full-context reservation would have granted, so
/// elastic lending is never a regression versus baseline (issue #857).
#[cfg(feature = "native-cuda")]
fn elastic_weight_budget_bytes(
    elastic_available_bytes: u64,
    static_baseline_budget_bytes: u64,
    headroom_bytes: u64,
) -> u64 {
    elastic_available_bytes
        .saturating_sub(headroom_bytes)
        .max(static_baseline_budget_bytes)
}

/// Environment override for the elastic-lending device headroom (issue #857).
#[cfg(feature = "native-cuda")]
const ELASTIC_LENDING_HEADROOM_BYTES_ENV: &str = "ONNX_GENAI_ELASTIC_LENDING_HEADROOM_BYTES";

/// Default device bytes kept *unlent* below the managed no-spill limit under
/// elastic weight lending (issue #857).
///
/// The memory governor's ledger guarantees no oversubscription of the resolved
/// device budget, but on WDDM a device pushed close to full can have its
/// physical granules paged out to host RAM by the OS under demand — a spill
/// that is invisible to our ledger, so `oversubscribed_bytes` would read 0
/// while the device is in fact spilling. It is not yet established whether our
/// VMM-mapped granules (`cuMemCreate`/`cuMemMap`) are subject to that eviction
/// or only ordinary `cuMemAlloc` allocations. Until that is settled, elastic
/// lending deliberately leaves a headroom margin rather than lending to the
/// last byte, so the OS has slack before it must evict anything of ours.
///
/// This is a conservative default, tunable via
/// [`ELASTIC_LENDING_HEADROOM_BYTES_ENV`]: once the WDDM question is answered in
/// our favour it is a cheap follow-up to lower it (or set it to 0) and lend more
/// aggressively; the reverse mistake — a hidden host-spill regression that only
/// shows up as wall-clock variance — is not cheap.
#[cfg(feature = "native-cuda")]
const DEFAULT_ELASTIC_LENDING_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// The device headroom to keep unlent below the managed no-spill limit under
/// elastic weight lending (issue #857), honouring the environment override.
#[cfg(feature = "native-cuda")]
fn elastic_lending_headroom_bytes() -> u64 {
    std::env::var(ELASTIC_LENDING_HEADROOM_BYTES_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_ELASTIC_LENDING_HEADROOM_BYTES)
}

/// Whether the weight budget may lend the unused full-context KV reservation to
/// weights (issue #857).
///
/// True only when the dynamic KV/weight reclaim path is guaranteed active: the
/// same three conditions that register the weight-residency cache as a
/// reclaimable mapped holder in `CudaExecutionProvider::adopt_memory_governor`.
/// Without that path the static full-context reservation is the only thing that
/// guarantees a sequence can reach its declared max context, so lending is
/// refused.
#[cfg(feature = "native-cuda")]
fn elastic_weight_lending_active(
    managed_no_spill: bool,
    commits_on_demand: bool,
    lending_enabled: bool,
) -> bool {
    managed_no_spill && commits_on_demand && lending_enabled
}

/// The device-tier KV bytes to withhold from the elastic weight budget.
///
/// Host-tier KV (a host-accessible EP) contributes nothing to the device weight
/// budget and returns zero.
#[cfg(feature = "native-cuda")]
fn elastic_kv_floor_device_bytes(
    native_session: &crate::native_decode::NativeDecodeSession,
    max_context: Option<usize>,
) -> anyhow::Result<u64> {
    let Some(max_context) = max_context else {
        return Ok(0);
    };
    let floor_context = elastic_kv_floor_context(max_context);
    let (bytes, tier) = native_session
        .kv_reservation(floor_context)
        .context("sizing the elastic native CUDA KV floor")?;
    Ok(if tier == onnx_runtime_memory_governor::Tier::Device {
        bytes
    } else {
        0
    })
}

#[cfg(feature = "native-cuda")]
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
    let native_kv_full_context_device_bytes =
        if native_kv_tier == onnx_runtime_memory_governor::Tier::Device {
            native_kv_bytes
        } else {
            0
        };
    // Elastic weight budget (#857). The weight residency budget was historically
    // derived by subtracting the *full declared-max-context* KV reservation from
    // the device budget, once, at load, and never revisited. On a not-fit model
    // that permanently withholds `kv_bytes_per_token × max_context` from weights
    // (1.611 GB on qwen14b-zp) even for a short request that commits a fraction
    // of it, so weights stream bytes that would have fit in the idle reservation.
    //
    // When the dynamic KV/weight reclaim path is guaranteed active — managed
    // no-spill, an on-demand-committing arena, and lending enabled, which is
    // exactly the condition that registers the weight cache as a reclaimable
    // mapped holder in `CudaExecutionProvider::adopt_memory_governor` — hold back
    // only an initial KV *floor* instead. The bytes above the floor are lent to
    // the weight cache; the registered holder gives them back to KV as the
    // sequence grows, one bucket at a time, through the transactional
    // `MappedGrowthAuthority` reclaim (KV growth reclaims from the weight zone).
    // A sequence can therefore still reach its declared max context: KV reclaims
    // the lent budget rather than finding it statically reserved.
    //
    // Without that reclaim path the static full-context reservation is the only
    // thing that guarantees max context, so keep it unchanged.
    let elastic_lending = elastic_weight_lending_active(
        resolution.policy.managed_no_spill,
        native_session.commits_on_demand(),
        onnx_runtime_ep_cuda::dynamic_kv_weight_lending_enabled(),
    );
    let native_kv_device_bytes = if elastic_lending {
        let floor = elastic_kv_floor_device_bytes(native_session, max_context)?
            .min(native_kv_full_context_device_bytes);
        tracing::info!(
            native_kv_full_context_device_bytes,
            native_kv_floor_device_bytes = floor,
            lent_to_weights_bytes = native_kv_full_context_device_bytes.saturating_sub(floor),
            "elastic weight budget: lending the unused full-context KV reservation to weights; \
             KV reclaims it on demand as the sequence grows (issue #857)"
        );
        floor
    } else {
        native_kv_full_context_device_bytes
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
    } else if elastic_lending {
        // The auto budget in `resolution.policy.device_budget_bytes` was derived
        // by `build_memory_strategy_plan` by subtracting the *full declared-max-
        // context* KV reservation from the resolved VRAM. Under elastic lending
        // we withhold only the KV floor, so that plan-time `requested` value is a
        // stale, too-low cap: `requested.min(available)` would clamp the budget
        // straight back to the non-elastic value and lend nothing. The correct
        // auto budget is the elastic `available` (resolved VRAM minus the floor
        // and recurrent state); the bytes above the floor are lent to weights and
        // reclaimed by KV growth on demand (issue #857).
        //
        // But do not lend to the last byte: keep a device headroom margin unlent
        // as a high-water mark below the managed no-spill limit, because on WDDM
        // a device pushed near full may have physical granules paged to host RAM
        // by the OS — a spill our ledger cannot see. The margin never drops the
        // budget below what the static full-context reservation would have
        // granted, so elastic lending is never a regression versus baseline.
        let headroom = elastic_lending_headroom_bytes();
        let static_baseline_budget_bytes = resolved_vram_bytes.saturating_sub(
            native_kv_full_context_device_bytes.saturating_add(recurrent_device_bytes),
        );
        elastic_weight_budget_bytes(
            available_weight_offload_budget_bytes,
            static_baseline_budget_bytes,
            headroom,
        )
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
            elastic_lending,
            elastic_lending_headroom_bytes = if elastic_lending {
                elastic_lending_headroom_bytes()
            } else {
                0
            },
            offload_device_budget_bytes = adopted,
            "enabled CUDA weight offload because model weights exceed the resolved device budget"
        );
    }

    Ok(Some(resolution.policy))
}

fn fail_explicit_vram_limit_without_offload(
    plan: &MemoryStrategyPlan,
    device_weights_are_selected: bool,
) -> anyhow::Result<()> {
    if !device_weights_are_selected {
        return Ok(());
    }
    if !plan.runtime_application().weight_offload_enabled {
        return Ok(());
    }
    let resolved_vram_bytes = plan.resolved_device_budget_bytes.unwrap_or(0);
    let package_bytes = plan.total_weight_bytes;
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
    draft_present: bool,
) -> anyhow::Result<(SpeculativeMode, Option<ResolvedMtpConfig>)> {
    let (speculative_mode, resolved_mtp_config) = match requested_mode {
        SpeculativeMode::None if draft_present => (SpeculativeMode::DraftModel, None),
        SpeculativeMode::None => (SpeculativeMode::None, None),
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
        Some(build_mtp_model_from_resolved(
            mtp_config,
            environment,
            session_options,
            model_directory,
            // The ORT target path has no native projection device; a quantised
            // shared LM-head therefore still fails fast in the adapter loader,
            // preserving the existing ORT-backend behaviour exactly.
            None,
        )?)
    } else {
        None
    };
    Ok(mtp)
}

/// Build an [`MtpModel`] (ORT MTP-head session + shared target embedding / LM
/// head) from an already-resolved and validated [`ResolvedMtpConfig`].
///
/// The target hidden-state output existence/shape/dtype validation is the
/// caller's responsibility because it depends on the target backend: the ORT
/// path validates against the target [`Session`] outputs, the native path
/// validates against the target ONNX graph. Everything downstream of that
/// validation — loading the pure-attention MTP head on the ORT environment and
/// resolving the target-initializer or file-based embedding/LM-head adapters —
/// is backend-agnostic and shared here.
fn build_mtp_model_from_resolved(
    mtp_config: ResolvedMtpConfig,
    environment: &Environment,
    session_options: &SessionOptions,
    model_directory: &ModelDirectory,
    draft_projection: Option<crate::speculative::DraftProjectionDevice>,
) -> anyhow::Result<MtpModel> {
    if mtp_config.cache_scope == MtpCacheScope::AcceptedPrefix {
        anyhow::bail!(
            "MTP kv_mode accepted_prefix is declared but not executable: the frozen Mobius contract does not define correction-token/cache alignment"
        );
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
    let (embedder, lm_head) = match (&mtp_config.embedding_weights, &mtp_config.lm_head_weights) {
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
                draft_projection,
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
    Ok(MtpModel {
        config: mtp_config.public_config.clone(),
        runtime_config: mtp_config.clone(),
        session: Arc::new(head_session),
        embedder,
        lm_head,
        hidden_output: mtp_config.public_config.target_hidden_output.clone(),
        kv_mode: mtp_config.public_config.kv_mode,
        num_speculative_tokens: mtp_config.public_config.num_speculative_tokens,
    })
}

/// Resolve and build a native-backend MTP proposer from a model directory's
/// inference metadata, mirroring [`load_native_shared_kv_proposer`].
///
/// Returns the loaded [`MtpModel`] (the pure-attention MTP head runs on the ORT
/// `environment`; only the hybrid GDN target needs the native EP) together with
/// the [`SpeculativeMode::Mtp`] the engine should adopt. Yields `None` /
/// [`SpeculativeMode::None`] when the metadata advertises no MTP speculator, so
/// non-speculative models keep their exact existing load path.
#[cfg(feature = "native-backend")]
fn load_native_mtp_proposer(
    metadata: &InferenceMetadata,
    model_directory: &ModelDirectory,
    environment: &Environment,
    session_options: &SessionOptions,
    native_device: crate::native_decode::NativeDecodeDevice,
) -> anyhow::Result<(Option<MtpModel>, SpeculativeMode)> {
    // The package's workflow describes the *target* graph; the draft head is a
    // separate artifact the model directory declares for itself in `config.json`.
    // Discovery is therefore the signal here, and a directory that declares no
    // speculator simply loads unspeculated.
    let Some(descriptor) = onnx_genai_metadata::detect_speculator(&model_directory.root) else {
        return Ok((None, SpeculativeMode::None));
    };
    let spec = match descriptor.proposer {
        onnx_genai_metadata::SpeculatorProposerStatus::Mtp(spec) => spec,
        // A declaration that cannot be read is an authoring error worth
        // surfacing. Any other proposer kind is simply not this loader's
        // business and leaves the native path unspeculated.
        onnx_genai_metadata::SpeculatorProposerStatus::Unknown(reason) => {
            anyhow::bail!("Invalid native MTP sidecar metadata: {reason}")
        }
        _ => return Ok((None, SpeculativeMode::None)),
    };
    // The native target exposes no ORT `Session` to interrogate for the target
    // vocabulary (that is the point of the native EP). The head borrows the
    // target's own LM-head initializer, so the vocabulary that sizes it is the
    // target's: read the package's declared capability rather than restating the
    // same number in a second place that nothing forces to agree.
    let vocab_size = metadata
        .model
        .as_ref()
        .and_then(|model| model.vocab_size)
        .filter(|&value| value > 0)
        .context(
            "native MTP speculation requires the target vocabulary size; declare \
             `model.vocab_size` in the package metadata",
        )?;
    let resolved = ResolvedMtpConfig::from_sidecar_descriptor(&spec, vocab_size);
    validate_resolved_mtp_config(&resolved)?;
    validate_native_mtp_hidden_output(&model_directory.model_path, &resolved)?;
    // The pure-attention MTP head runs as an ORT session; when the native target
    // is on CUDA the head must load on the CUDA EP too (its mixed-precision graph
    // relies on CUDA kernels — the CPU EP rejects the bf16/f32 mix). Build head
    // session options that match the native decode device rather than inheriting
    // the native path's default (CPU) options. Allow CPU fallback so the handful
    // of mixed-precision norm nodes ORT cannot assign to CUDA still get a home
    // (the bulk of the head — attention and matmuls — stays on CUDA).
    let head_session_options = match native_device.cuda_index() {
        Some(index) => {
            let mut selection = onnx_genai_ort::ep_selection("cuda");
            selection
                .options
                .insert("device_id".to_string(), index.to_string());
            SessionOptions::with_execution_provider(selection).with_cpu_fallback(true)
        }
        None => session_options.clone(),
    };
    // The int4 shared LM-head is projected on the same device as the native
    // target: CUDA when the target is on CUDA, otherwise the CPU int4 kernel.
    // This projection runs during proposal, outside the captured decode step.
    let draft_projection = Some(match native_device.cuda_index() {
        Some(index) => crate::speculative::DraftProjectionDevice::Cuda { index },
        None => crate::speculative::DraftProjectionDevice::Cpu,
    });
    let mtp = build_mtp_model_from_resolved(
        resolved,
        environment,
        &head_session_options,
        model_directory,
        draft_projection,
    )?;
    let mode = SpeculativeMode::Mtp(mtp.config.clone());
    Ok((Some(mtp), mode))
}

/// Validate the target decoder's hidden-state seed output against the target
/// ONNX graph for the native MTP path — the analogue of [`load_mtp_model`]'s
/// ORT-`Session` validation, but reading only the declared hidden port from the
/// graph so the 17GB target is never loaded into ORT just to be inspected.
#[cfg(feature = "native-backend")]
fn validate_native_mtp_hidden_output(
    model_path: &Path,
    mtp_config: &ResolvedMtpConfig,
) -> anyhow::Result<()> {
    let hidden_name = mtp_config.public_config.target_hidden_output.clone();
    use onnx_genai_ort::GraphIo as _;
    let graph_io = onnx_genai_ort::graph_io_from_model_path_for_names(
        model_path,
        &[],
        std::slice::from_ref(&hidden_name),
    )
    .map_err(|error| {
        anyhow::anyhow!("read native MTP target hidden output '{hidden_name}': {error}")
    })?;
    let hidden_output = graph_io
        .outputs()
        .iter()
        .find(|output| output.name == hidden_name)
        .with_context(|| {
            format!("native MTP target model must expose hidden-state output '{hidden_name}'")
        })?;
    if !matches!(
        hidden_output.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        anyhow::bail!(
            "native MTP target hidden-state output '{hidden_name}' must be Float32, Float16, or BFloat16, got {:?}",
            hidden_output.dtype
        );
    }
    let hidden_size = mtp_config.public_config.hidden_size as i64;
    let matches_layout = match mtp_config.target_hidden_layout {
        MtpHiddenLayout::Bsh => {
            hidden_output.shape.len() == 3
                && hidden_output.shape.last().copied().filter(|dim| *dim > 0) == Some(hidden_size)
        }
        MtpHiddenLayout::Bshc => {
            hidden_output.shape.len() == 4
                && hidden_output.shape[2] == mtp_config.hc_mult as i64
                && hidden_output.shape[3] == hidden_size
        }
    };
    if !matches_layout {
        anyhow::bail!(
            "native MTP target hidden-state output '{hidden_name}' shape {:?} does not match configured {:?} with hc_mult {} and hidden size {}",
            hidden_output.shape,
            mtp_config.target_hidden_layout,
            mtp_config.hc_mult,
            mtp_config.public_config.hidden_size
        );
    }
    Ok(())
}

fn load_eagle3_model(
    speculative_mode: &SpeculativeMode,
    session: &Session,
    environment: &Environment,
    session_options: &SessionOptions,
) -> anyhow::Result<Option<Eagle3Model>> {
    let eagle3 = if let SpeculativeMode::Eagle3(eagle_config) = speculative_mode {
        crate::config::validate_eagle3_config(eagle_config)?;
        let mut target_activation_dtype = None;
        for output_name in &eagle_config.target_hidden_outputs {
            let hidden_output = session
                .outputs()
                .iter()
                .find(|output| output.name == *output_name)
                .with_context(|| {
                    format!("EAGLE-3 target model must expose hidden-state output '{output_name}'")
                })?;
            if !matches!(
                hidden_output.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) {
                anyhow::bail!(
                    "chained proposer target context '{}' must be Float32, Float16, or BFloat16, got {:?}",
                    hidden_output.name,
                    hidden_output.dtype
                );
            }
            if let Some(dtype) = target_activation_dtype {
                if hidden_output.dtype != dtype {
                    anyhow::bail!(
                        "chained proposer target context outputs must share one dtype; '{}' is {:?}, expected {:?}",
                        hidden_output.name,
                        hidden_output.dtype,
                        dtype
                    );
                }
            } else {
                target_activation_dtype = Some(hidden_output.dtype);
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
        if Some(head_signature.activation_dtype) != target_activation_dtype {
            anyhow::bail!(
                "chained proposer activation dtype {:?} does not match target context dtype {:?}",
                head_signature.activation_dtype,
                target_activation_dtype
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
        let token_map = eagle_config
            .token_map
            .as_ref()
            .map(|path| {
                let bytes = std::fs::read(path).with_context(|| {
                    format!("Failed to read proposer vocabulary map '{}'", path.display())
                })?;
                if bytes.len() % std::mem::size_of::<i64>() != 0 {
                    anyhow::bail!(
                        "proposer vocabulary map '{}' has byte length {}, which is not divisible by 8",
                        path.display(),
                        bytes.len()
                    );
                }
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|bytes| {
                        let token = i64::from_le_bytes(*bytes);
                        TokenId::try_from(token).with_context(|| {
                            format!("mapped proposer token id {token} is outside the target range")
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .transpose()?;
        if token_map
            .as_ref()
            .is_some_and(|map| map.len() < head_signature.draft_vocab_size)
        {
            anyhow::bail!("proposer vocabulary map has fewer entries than the draft vocabulary");
        }
        Some(Eagle3Model {
            config: eagle_config.clone(),
            session: Box::new(head_session),
            embedder: LinearEmbedder::new(
                embedding,
                eagle_config.vocab_size,
                eagle_config.hidden_size,
            )
            .map_err(|error| anyhow::anyhow!("Invalid EAGLE-3 embedding weights: {error}"))?,
            token_map,
            hidden_outputs: eagle_config.target_hidden_outputs.clone(),
            kv_mode: eagle_config.kv_mode,
            num_speculative_tokens: eagle_config.num_speculative_tokens,
        })
    } else {
        None
    };
    Ok(eagle3)
}

#[cfg(test)]
mod metadata_admission_tests {
    use super::*;

    #[test]
    fn unsupported_declared_capability_rejects_before_decode_selection() {
        let metadata: InferenceMetadata =
            serde_yaml::from_str("schema_version: v1\nrequired_capabilities: [vendor.future]\n")
                .expect("metadata parses");

        let error = admit_inference_metadata(&metadata)
            .expect_err("unsupported capability must fail closed")
            .to_string();
        assert!(
            error.contains("Unsupported inference metadata capabilities")
                && error.contains("vendor.future"),
            "{error}"
        );
    }

    #[test]
    fn unsupported_derived_workflow_capability_cannot_fall_back_to_bare_decoder() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa]
    components:
      decoder:
        implementation: { kind: binding }
        ports: {}
    steps:
      - kind: invoke
        component: decoder
"#,
        )
        .expect("metadata parses");

        let error = admit_inference_metadata(&metadata)
            .expect_err("workflow package must not fall back to bare decode")
            .to_string();
        assert!(
            error.contains("Unsupported inference metadata capabilities")
                && error.contains("workflow_ssa"),
            "{error}"
        );
    }
}

#[cfg(all(test, feature = "native-backend"))]
mod mtp_seed_tests {
    use super::*;

    fn io_with_hidden_output(hidden_output: Option<&str>) -> onnx_genai_metadata::ModelIoSpec {
        let mut value = serde_json::json!({});
        if let Some(name) = hidden_output {
            value["hidden_output"] = serde_json::json!(name);
        }
        serde_json::from_value(value).expect("model io spec parses")
    }

    /// The seed the sidecar names has to reach the ABI, or the native session
    /// never records the hidden state the draft head is supposed to consume.
    #[test]
    fn a_sidecar_seed_names_the_hidden_output_an_abi_left_unset() {
        let io = io_with_hidden_output(None);

        let seeded = decoder_io_seeded_with(&io, "hidden_states").expect("seed applies");

        assert_eq!(seeded.hidden_output.as_deref(), Some("hidden_states"));
    }

    /// A declared port is authoritative: a sidecar must not redirect the
    /// target's own contract at load time.
    #[test]
    fn a_declared_hidden_output_outranks_the_sidecar_seed() {
        let io = io_with_hidden_output(Some("last_hidden_state"));

        assert!(decoder_io_seeded_with(&io, "hidden_states").is_none());
    }

    /// An empty seed must decline rather than install `""`, which would read as
    /// a declared port everywhere downstream.
    #[test]
    fn an_unnamed_seed_declines_instead_of_declaring_an_empty_port() {
        let io = io_with_hidden_output(None);

        assert!(decoder_io_seeded_with(&io, "").is_none());
    }
}

#[cfg(test)]
mod pool_sizing_tests {
    use super::*;

    #[cfg(feature = "native-cuda")]
    fn cuda_plan(
        strategy: MemoryStrategy,
        total_weight_bytes: u64,
        resolved_device_budget_bytes: u64,
        application: MemoryPolicyApplication,
    ) -> MemoryStrategyPlan {
        let mut plan =
            MemoryStrategyPlan::unknown(total_weight_bytes, None, "CUDA policy test plan");
        plan.strategy = strategy;
        plan.inferred_strategy = strategy;
        plan.resolved_device_budget_bytes = Some(resolved_device_budget_bytes);
        plan.fits_resolved_device_budget = Some(total_weight_bytes <= resolved_device_budget_bytes);
        plan.application = application;
        plan
    }

    fn geometry(layers: usize) -> Vec<onnx_genai_kv::LayerTensorConfig> {
        (0..layers)
            .map(|_| onnx_genai_kv::LayerTensorConfig {
                num_kv_heads: 8,
                head_dim: 128,
            })
            .collect()
    }

    fn tiny_llm_model_path() -> anyhow::Result<std::path::PathBuf> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/model.onnx.textproto")
            .canonicalize()?)
    }

    fn tiny_llm_io() -> anyhow::Result<onnx_genai_metadata::ModelIoSpec> {
        let metadata_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/inference_metadata.yaml")
            .canonicalize()?;
        let metadata = onnx_genai_metadata::load_metadata(&metadata_path)?;
        metadata
            .decoder_io()
            .cloned()
            .context("tiny-llm fixture must declare a decode ABI")
    }

    fn profile_json(name: &str) -> anyhow::Result<serde_json::Value> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/profiles")
            .join(name);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read profile {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse profile {}", path.display()))
    }

    fn profile_u64(profile: &serde_json::Value, pointer: &str) -> anyhow::Result<u64> {
        profile
            .pointer(pointer)
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("profile field {pointer} must be a u64"))
    }

    #[test]
    fn native_graph_io_kv_page_cost_matches_ort_session_for_same_model() -> anyhow::Result<()> {
        let model_path = tiny_llm_model_path()?;
        let io = tiny_llm_io()?;
        let config = EngineConfig::default();

        let environment = Environment::new("onnx-genai-engine-native-kv-geometry-test")
            .map_err(|e| anyhow::anyhow!("Failed to create ORT environment: {e}"))?;
        let session = Session::new(&environment, &model_path, SessionOptions::default())
            .map_err(|e| anyhow::anyhow!("Failed to load ORT fixture session: {e}"))?;
        let ort_kv_model =
            infer_kv_model_info(&session, Some(&io), config.page_size, config.kv_cache_dtype)?
                .context("ORT fixture session should expose KV geometry")?;

        let native_graph_io =
            onnx_genai_ort::graph_io_from_model_path(&model_path).map_err(|e| {
                anyhow::anyhow!("Failed to read native fixture graph I/O metadata: {e}")
            })?;
        let native_kv_model = infer_kv_model_info(
            &native_graph_io,
            Some(&io),
            config.page_size,
            config.kv_cache_dtype,
        )?
        .context("native graph I/O should expose KV geometry")?;

        let ort_config = governor_kv_config(Some(&ort_kv_model), &config)?;
        let native_config = governor_native_kv_config(Some(&native_kv_model), &config)?;

        assert_eq!(native_config.page_size_bytes, ort_config.page_size_bytes);
        assert_eq!(
            native_config.bytes_per_token(),
            ort_config.bytes_per_token()
        );
        assert_ne!(
            native_config.page_size_bytes,
            Some(config.page_size as u64),
            "KV page byte size must not be the token count"
        );
        assert!(
            native_config
                .bytes_per_token()
                .is_some_and(|bytes| bytes > 1),
            "KV bytes/token must come from real geometry, not the old 1 B/token fallback"
        );
        Ok(())
    }

    #[test]
    fn committed_native_profiles_use_ort_kv_page_geometry() -> anyhow::Result<()> {
        for (ort_name, native_name) in [
            ("qwen2.5-0.5b-cpu.json", "qwen2.5-0.5b-native.json"),
            ("qwen2.5-0.5b-metal.json", "qwen2.5-0.5b-native-mlx.json"),
            ("qwen2.5-0.5b-f16-cpu.json", "qwen2.5-0.5b-f16-native.json"),
        ] {
            let ort = profile_json(ort_name)?;
            let native = profile_json(native_name)?;
            let native_page_bytes = profile_u64(&native, "/device_memory_breakdown/kv_page_bytes")?;
            let native_tokens = profile_u64(&native, "/kv_cache_max_tokens")?;

            assert_eq!(
                native_page_bytes,
                profile_u64(&ort, "/device_memory_breakdown/kv_page_bytes")?,
                "{native_name} must report the same KV page byte geometry as {ort_name}"
            );
            assert_eq!(
                profile_u64(&native, "/device_memory_breakdown/kv_pages")?,
                profile_u64(&ort, "/device_memory_breakdown/kv_pages")?,
                "{native_name} must report the same derived KV page count as {ort_name}"
            );
            assert_eq!(
                native_tokens,
                profile_u64(&ort, "/kv_cache_max_tokens")?,
                "{native_name} must report the same derived KV token budget as {ort_name}"
            );
            assert_ne!(
                native_page_bytes, 16,
                "{native_name} must not store a token count in kv_page_bytes"
            );
            assert!(
                native_page_bytes.div_ceil(16) > 1,
                "{native_name} must not imply 1 B/token admission"
            );
        }
        Ok(())
    }

    /// The pool must never be sized by a budget divided by the wrong page size.
    ///
    /// `derived_budget.total_pages` divides the KV budget by the governor's
    /// page byte size. If that byte size ever came from a token-count
    /// placeholder, an 8 GiB device would resolve to ~483 million pages, and
    /// because the table pre-creates one `Page` per slot, building that pool
    /// exhausts the machine before any KV exists -- which is exactly how this
    /// was found, as a CI runner dying with SIGTERM mid-test.
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
            device_weight_reservation_for(package, None, None),
            package,
            "with offload off every weight is resident"
        );
        assert_eq!(
            device_weight_reservation_for(package, Some(2 << 30), None),
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
            device_weight_reservation_for(package, Some(4 << 30), None),
            package
        );
    }

    #[test]
    fn offload_startup_reservation_leaves_one_kv_page_unwarned() {
        assert_eq!(
            device_weight_reservation_for(10_000, Some(6_000), Some(128)),
            5_872,
            "the temporary startup claim must leave the governor a one-page KV floor; \
             the CUDA provider later adopts the full offload budget as the real claim"
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_kv_reservation_uses_runtime_capacity_before_metadata() {
        let (max_len, source) = native_kv_reservation_max_context(
            Some((4096, "ONNX_GENAI_CUDA_KV_MAX_LEN".into())),
            Some(131_072),
        )
        .expect("runtime capacity should be available");

        assert_eq!(max_len, 4096);
        assert_eq!(source, "ONNX_GENAI_CUDA_KV_MAX_LEN");
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_kv_reservation_falls_back_to_metadata() {
        let (max_len, source) = native_kv_reservation_max_context(None, Some(131_072))
            .expect("metadata capacity should be available");

        assert_eq!(max_len, 131_072);
        assert_eq!(source, "model.max_sequence_length");
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_kv_reservation_ignores_unbounded_runtime_capacity_without_metadata() {
        assert_eq!(
            native_kv_reservation_max_context(Some((usize::MAX, "unbounded".into())), None),
            None
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn native_kv_reservation_uses_metadata_after_unbounded_runtime_capacity() {
        let (max_len, source) = native_kv_reservation_max_context(
            Some((usize::MAX, "unbounded".into())),
            Some(131_072),
        )
        .expect("metadata capacity should be available");

        assert_eq!(max_len, 131_072);
        assert_eq!(source, "model.max_sequence_length");
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn elastic_lending_requires_reclaim_path_to_be_guaranteed() {
        // Elastic lending is only safe when the reclaim path exists, which is
        // exactly managed no-spill + an on-demand-committing arena + lending on.
        assert!(elastic_weight_lending_active(true, true, true));
        // Any missing precondition falls back to the static full-context
        // reservation that guarantees max context on its own.
        assert!(!elastic_weight_lending_active(false, true, true));
        assert!(!elastic_weight_lending_active(true, false, true));
        assert!(!elastic_weight_lending_active(true, true, false));
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn elastic_kv_floor_is_the_first_bucket_and_never_exceeds_max_context() {
        // The floor is exactly the engine's first KV bucket, capped at the
        // declared maximum, so the two stay single-sourced.
        let expected_large = onnx_genai_kv::kv_capacity_bucket(1, 8192).min(8192);
        assert_eq!(elastic_kv_floor_context(8192), expected_large);
        assert!(
            elastic_kv_floor_context(8192) < 8192,
            "a large context must lend most of its reservation"
        );
        // A context at or below the first bucket cannot be lent below itself.
        assert_eq!(elastic_kv_floor_context(64), 64);
        // The floor never exceeds the declared maximum.
        for max in [1usize, 100, 256, 1024, 8192, 131_072] {
            assert!(elastic_kv_floor_context(max) <= max, "max={max}");
        }
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn elastic_weight_budget_leaves_headroom_and_never_regresses_below_baseline() {
        // resolved_vram=100, full-context KV reservation=40, floor=4.
        // elastic available = 100 - 4 = 96; static baseline = 100 - 40 = 60.
        let elastic_available = 96u64;
        let baseline = 60u64;

        // With a 10-byte headroom we lend up to the high-water mark, not the last
        // byte: 96 - 10 = 86, comfortably above the baseline and below available.
        let budget = elastic_weight_budget_bytes(elastic_available, baseline, 10);
        assert_eq!(budget, 86);
        assert!(budget < elastic_available, "headroom is kept unlent");
        assert!(
            budget > baseline,
            "still lends more than the static reservation"
        );

        // A headroom so large it would push below the static reservation is
        // clamped: elastic lending is never a regression versus baseline.
        let clamped = elastic_weight_budget_bytes(elastic_available, baseline, 90);
        assert_eq!(clamped, baseline);
        assert!(clamped >= baseline);

        // Zero headroom lends everything above the floor (opt-in maximal lending).
        assert_eq!(
            elastic_weight_budget_bytes(elastic_available, baseline, 0),
            elastic_available
        );
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn elastic_lending_headroom_defaults_to_a_conservative_nonzero_margin() {
        // The default must be non-zero so we never lend to the last byte while
        // the WDDM granule-eviction question is open.
        const {
            assert!(DEFAULT_ELASTIC_LENDING_HEADROOM_BYTES > 0);
        }
        // The helper returns the default when the override is absent or
        // unparseable; in an unset environment that is the conservative default.
        if std::env::var_os(ELASTIC_LENDING_HEADROOM_BYTES_ENV).is_none() {
            assert_eq!(
                elastic_lending_headroom_bytes(),
                DEFAULT_ELASTIC_LENDING_HEADROOM_BYTES
            );
        }
    }

    #[test]
    fn explicit_vram_limit_fails_without_an_offload_capable_backend() {
        let mut plan = MemoryStrategyPlan::unknown(10_000, None, "test plan");
        plan.strategy = MemoryStrategy::DynamicWeightResidency;
        plan.resolved_device_budget_bytes = Some(6_000);
        let error = fail_explicit_vram_limit_without_offload(&plan, true)
            .unwrap_err()
            .to_string();

        assert!(error.contains("require 10000 bytes"), "{error}");
        assert!(error.contains("allows 6000 bytes"), "{error}");
        assert!(error.contains("automatically offload weights"), "{error}");
    }

    #[test]
    fn explicit_vram_limit_does_not_reject_host_only_execution() {
        let mut plan = MemoryStrategyPlan::unknown(10_000, None, "test plan");
        plan.strategy = MemoryStrategy::DynamicWeightResidency;
        plan.resolved_device_budget_bytes = Some(6_000);
        fail_explicit_vram_limit_without_offload(&plan, false)
            .expect("a VRAM limit should not reject a host-only execution provider");
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn explicit_vram_limit_auto_enables_cuda_weight_offload() {
        let plan = cuda_plan(
            MemoryStrategy::DynamicWeightResidency,
            10_000,
            6_000,
            MemoryPolicyApplication {
                weight_offload_enabled: true,
                device_budget_bytes: Some(6_000),
                scan_resistant_dense: true,
                managed_no_spill: true,
                managed_limit_bytes: Some(6_000),
                device_budget_is_override: false,
                auto_enabled_from_vram_limit: true,
            },
        );
        let policy = cuda_offload_resolution_from_plan(
            &crate::native_decode::NativeDecodeDevice::Cuda { index: Some(0) },
            &plan,
        )
        .expect("weights above an explicit CUDA VRAM limit should enable offload");

        assert!(policy.policy.enabled);
        assert!(policy.policy.managed_no_spill);
        assert_eq!(policy.policy.managed_limit_bytes, Some(6_000));
        assert_eq!(policy.policy.device_budget_bytes, Some(6_000));
        assert!(!policy.policy.async_pagein, "prefetch remains disabled");
        assert!(policy.auto_enabled_from_vram_limit);
        assert!(uses_governed_physical_pool(Some(policy), true, false));
        assert_eq!(
            cuda_weight_startup_reservation(10_000, Some(policy), true, None),
            0
        );
        assert!(
            !uses_governed_physical_pool(Some(policy), false, false),
            "the lending opt-out must preserve the compatibility reservation path"
        );
        assert_eq!(
            cuda_weight_startup_reservation(10_000, Some(policy), false, None),
            6_000,
            "the non-pool offload path keeps its existing device-budget reservation"
        );
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn explicit_weight_offload_device_bytes_overrides_vram_limit_derivation() {
        let plan = cuda_plan(
            MemoryStrategy::DynamicWeightResidency,
            10_000,
            6_000,
            MemoryPolicyApplication {
                weight_offload_enabled: true,
                device_budget_bytes: Some(4_000),
                scan_resistant_dense: true,
                managed_no_spill: true,
                managed_limit_bytes: Some(6_000),
                device_budget_is_override: true,
                auto_enabled_from_vram_limit: false,
            },
        );
        let policy = cuda_offload_resolution_from_plan(
            &crate::native_decode::NativeDecodeDevice::Cuda { index: Some(0) },
            &plan,
        )
        .expect("the explicit limit still triggers offload");

        assert_eq!(policy.policy.device_budget_bytes, Some(4_000));
        assert_eq!(policy.policy.managed_limit_bytes, Some(6_000));
        assert!(policy.device_budget_is_override);
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn explicit_vram_limit_selects_managed_mode_even_when_weights_fit() {
        let plan = cuda_plan(
            MemoryStrategy::FullResident,
            10_000,
            12_000,
            MemoryPolicyApplication {
                weight_offload_enabled: false,
                device_budget_bytes: None,
                scan_resistant_dense: true,
                managed_no_spill: true,
                managed_limit_bytes: Some(12_000),
                device_budget_is_override: false,
                auto_enabled_from_vram_limit: false,
            },
        );
        let policy = cuda_offload_resolution_from_plan(
            &crate::native_decode::NativeDecodeDevice::Cuda { index: Some(0) },
            &plan,
        )
        .expect("explicit byte limit selects managed allocation");
        assert!(!policy.policy.enabled);
        assert!(policy.policy.managed_no_spill);
        assert!(
            uses_governed_physical_pool(Some(policy), true, false),
            "managed VMM owns physical weight handles even when offload is unnecessary"
        );
        assert_eq!(
            cuda_weight_startup_reservation(10_000, Some(policy), true, Some(256)),
            0,
            "pool-owned resident weights must not also reserve package bytes"
        );
        assert!(
            !uses_governed_physical_pool(Some(policy), false, false),
            "lending opt-out keeps the resident package-reservation path"
        );
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn resident_non_vmm_weights_keep_package_reservation() {
        let resolution = CudaOffloadResolution {
            policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy {
                enabled: false,
                managed_no_spill: false,
                ..onnx_runtime_ep_cuda::DeviceOffloadPolicy::default()
            },
            device_budget_is_override: false,
            auto_enabled_from_vram_limit: false,
        };

        assert_eq!(
            cuda_weight_startup_reservation(10_000, Some(resolution), false, Some(256)),
            10_000
        );
    }

    #[test]
    #[cfg(feature = "native-backend")]
    fn native_state_only_scheduler_does_not_require_kv_geometry() {
        let mut scheduler = onnx_genai_scheduler::SchedulerConfig::default();
        let kv_config =
            governor_no_paged_kv_config(&EngineConfig::default()).expect("state-only KV config");

        populate_scheduler_bytes_per_token(&mut scheduler, kv_config)
            .expect("state-only native load keeps absent KV geometry valid");

        assert!(!kv_config.page_geometry_required);
        assert_eq!(scheduler.bytes_per_token, None);
    }
}
