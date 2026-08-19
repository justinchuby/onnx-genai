use super::*;

pub(crate) struct NativeDecodeLoadOptions<'a> {
    pub(crate) host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_offload_policy: Option<onnx_runtime_ep_cuda::DeviceOffloadPolicy>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_memory_governor:
        Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    pub(crate) io: Option<&'a ModelIoSpec>,
    pub(crate) metadata_max_len: Option<usize>,
    pub(crate) key_sequence_lengths_policy: crate::decode::KeySequenceLengthsPolicy,
    pub(crate) decode_precision: DecodePrecision,
    /// Persistent decode batch extent requested by the caller (`--max-batch`),
    /// or `None` to defer to `ONNX_GENAI_NATIVE_DECODE_BATCH` (#1064).
    pub(crate) decode_batch: Option<usize>,
}

fn native_metadata_max_len_from_model_path(path: &Path) -> Option<usize> {
    let root = if path.is_dir() { path } else { path.parent()? };
    onnx_genai_metadata::load_metadata_from_dir(root)
        .ok()
        .flatten()
        .and_then(|metadata| metadata.model.and_then(|model| model.max_sequence_length))
}

/// Resolve a model directory's [`InferenceMetadata`] using the same precedence
/// as the engine's directory loader: a native `inference_metadata.{yaml,yml,json}`
/// sidecar first, then onnxruntime-genai `genai_config.json` compatibility
/// synthesis. Returns `None` when neither is present so callers fall back to
/// shape-based I/O inference exactly as before.
fn resolve_io_metadata_from_model_path(
    path: &Path,
) -> Option<onnx_genai_metadata::InferenceMetadata> {
    let root = if path.is_dir() { path } else { path.parent()? };
    if let Some(metadata) = onnx_genai_metadata::load_metadata_from_dir(root)
        .ok()
        .flatten()
    {
        return Some(metadata);
    }
    let genai_config = root.join("genai_config.json");
    if genai_config.is_file() {
        return crate::engine::genai_config_compat_metadata_from_model_path(
            Some(genai_config.as_path()),
            path,
        )
        .ok()
        .flatten();
    }
    None
}

impl NativeDecodeSession {
    /// Load a decoder-with-past model, resolving its [`ModelIoSpec`] from an
    /// adjacent `inference_metadata.{yaml,yml,json}` sidecar or (for
    /// onnxruntime-genai packages) `genai_config.json`, so genai_config decoders
    /// bind their token input from metadata instead of guessing from ambiguous
    /// tensor shapes. This is the observability entry point used by measurement
    /// tools so CUDA-graph capture counters and `--trace` reject reasons surface
    /// for genai_config decoders. Capture semantics are identical to [`load`]:
    /// `graph_capture` stays auto-decided and no defaults change; only I/O
    /// resolution is threaded.
    ///
    /// [`load`]: NativeDecodeSession::load
    pub fn load_with_resolved_io(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let metadata = resolve_io_metadata_from_model_path(path);
        let io = metadata
            .as_ref()
            .and_then(|metadata| metadata.model.as_ref())
            .and_then(|model| model.io.as_ref());
        Self::load_with_cuda_options_and_io(
            path,
            device,
            NativeDecodeCudaOptions::default(),
            io,
            None,
            None,
        )
    }

    pub(crate) fn load_with_weight_offload_host_cache(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        options: NativeDecodeLoadOptions<'_>,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
            NativeDecodeDevice::Plugin { .. } => DevicePreference::Cpu,
        };
        let mut builder = InferenceSession::builder()
            .model(path)
            .device(preference)
            .decode_precision(options.decode_precision);
        if device == NativeDecodeDevice::Cpu {
            let ep =
                onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_weight_offload_host_cache(
                    options.host_cache,
                )
                .context("initialize native CPU execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        #[cfg(feature = "cuda")]
        if let NativeDecodeDevice::Cuda { index } = device {
            let policy = options
                .cuda_offload_policy
                .unwrap_or_else(onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env);
            let ep = onnx_runtime_ep_cuda::CudaExecutionProvider::
                initialized_with_offload_policy_and_governor(
                    index.unwrap_or(0),
                    policy,
                    options.cuda_memory_governor,
                )
                .context("initialize native CUDA execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        if let NativeDecodeDevice::Plugin {
            library,
            registration_name,
            provider_name,
        } = device
        {
            let ep = onnx_runtime_session::PluginExecutionProvider::new(
                library,
                registration_name,
                provider_name.clone(),
                provider_name,
            )
            .context("initialize native plugin execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        let session = builder.build().context("load native decoder model")?;
        Self::validate_key_sequence_lengths_contract(
            &session,
            options.key_sequence_lengths_policy,
        )?;
        Self::from_session_with_cuda_options_and_io(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: None,
                metadata_max_len: options.metadata_max_len,
                graph_capture: None,
                weight_offload_enabled: {
                    #[cfg(feature = "cuda")]
                    {
                        options.cuda_offload_policy.map(|policy| policy.enabled)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        None
                    }
                },
                weight_offload_stable_va: {
                    #[cfg(feature = "cuda")]
                    {
                        options
                            .cuda_offload_policy
                            .map(|policy| policy.enabled && policy.managed_no_spill)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        None
                    }
                },
                decode_batch: options.decode_batch,
            },
            options.io,
        )
    }

    fn validate_key_sequence_lengths_contract(
        session: &InferenceSession,
        policy: crate::decode::KeySequenceLengthsPolicy,
    ) -> anyhow::Result<()> {
        for (_, node) in session.graph().nodes.iter() {
            // This registry is semantic, not name inference: each supported attention
            // schema declares the input position carrying key-sequence lengths.
            // Add future attention schemas here when their native kernel lands.
            let key_sequence_lengths_index = match (node.domain.as_str(), node.op_type.as_str()) {
                ("com.microsoft", "GroupQueryAttention") => 5,
                _ => continue,
            };
            let Some(value_id) = node
                .inputs
                .get(key_sequence_lengths_index)
                .and_then(|input| *input)
            else {
                continue;
            };
            let value = session.graph().value(value_id);
            if value.shape.is_empty()
                && policy == crate::decode::KeySequenceLengthsPolicy::Canonical
            {
                bail!(
                    "attention key-sequence lengths input '{}' is scalar, but metadata does not declare model.attention.key_sequence_lengths.scalar_broadcast: unit_batch; the default contract requires contiguous int32 [batch_size]",
                    value.name.as_deref().unwrap_or("<anonymous>")
                );
            }
        }
        Ok(())
    }

    /// Load with an explicit CUDA KV capacity. `None` resolves in order:
    /// `ONNX_GENAI_CUDA_KV_MAX_LEN`, then `model.max_sequence_length` from
    /// inference metadata, then unbounded growth until the device refuses
    /// allocation. No free-memory ceiling is derived; the hard maximum comes
    /// only from an explicit override or the model's declared context limit.
    pub fn load_with_cuda_kv_max_len(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        let metadata_max_len = native_metadata_max_len_from_model_path(path.as_ref());
        Self::load_with_cuda_options(
            path,
            device,
            NativeDecodeCudaOptions {
                decode_batch: None,
                kv_max_len: cuda_kv_max_len,
                metadata_max_len,
                graph_capture: None,
                weight_offload_enabled: None,
                weight_offload_stable_va: None,
            },
        )
    }

    /// Load with explicit native-CUDA decode options. When graph capture is
    /// unspecified (`None`) and `ONNX_GENAI_CUDA_GRAPH` is unset, capture is
    /// auto-enabled whenever the decode topology is structurally graph-safe
    /// (CUDA device with owned, device-resident KV), and transparently declines
    /// to eager execution otherwise. An explicit `ONNX_GENAI_CUDA_GRAPH=0`/`=1`
    /// or a programmatic `graph_capture` value always overrides the
    /// auto-decision.
    pub fn load_with_cuda_options(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        options: NativeDecodeCudaOptions,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options_and_io(path, device, options, None, None, None)
    }

    /// Load a decoder-with-past model, threading the pipeline-declared
    /// [`ModelIoSpec`] so `sequence_source` (e.g. `inputs_embeds`), the KV pairs,
    /// and routed step inputs are bound from metadata rather than guessed from
    /// tensor shapes. The pipeline's native device-KV decoder (inc2b) uses this so
    /// an `inputs_embeds` decoder with no token input loads correctly.
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn load_with_io(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        io: Option<&ModelIoSpec>,
        metadata_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options_and_io(
            path,
            device,
            NativeDecodeCudaOptions {
                metadata_max_len,
                ..NativeDecodeCudaOptions::default()
            },
            io,
            None,
            None,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn load_with_io_and_cuda_governor(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        io: Option<&ModelIoSpec>,
        metadata_max_len: Option<usize>,
        offload_policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
        governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options_and_io(
            path,
            device,
            NativeDecodeCudaOptions {
                metadata_max_len,
                ..NativeDecodeCudaOptions::default()
            },
            io,
            Some(governor),
            Some(offload_policy),
        )
    }

    /// Test-support (leverb-phase0): load with an explicit [`ModelIoSpec`] and
    /// custom CUDA options but no offload governor. Lets the `#[ignore]`d Lever-B
    /// probe drive a real metadata-declared decoder (e.g. glm-4-9b, whose two
    /// rank-2 int64 inputs are ambiguous under shape-only autoderive) directly at
    /// the `NativeDecodeSession` layer, without standing up a full pipeline.
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn load_with_cuda_options_and_io_spec(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        options: NativeDecodeCudaOptions,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options_and_io(path, device, options, io, None, None)
    }

    fn load_with_cuda_options_and_io(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        mut options: NativeDecodeCudaOptions,
        io: Option<&ModelIoSpec>,
        #[cfg(feature = "cuda")] cuda_governor: Option<
            Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
        >,
        #[cfg(not(feature = "cuda"))] _cuda_governor: Option<()>,
        #[cfg(feature = "cuda")] cuda_offload_policy: Option<
            onnx_runtime_ep_cuda::DeviceOffloadPolicy,
        >,
        #[cfg(not(feature = "cuda"))] _cuda_offload_policy: Option<()>,
    ) -> anyhow::Result<Self> {
        if options.metadata_max_len.is_none() {
            options.metadata_max_len = native_metadata_max_len_from_model_path(path.as_ref());
        }
        // Issue #716: the managed no-spill authority path installs the VMM arena
        // and physical granule pool, so weight page-ins run on reserved-once
        // stable virtual addresses. Record that here — where the effective
        // offload policy is known — so the decode session can keep whole-step
        // CUDA graph capture ON while offload is active.
        #[cfg(feature = "cuda")]
        if let Some(policy) = cuda_offload_policy {
            options.weight_offload_stable_va = Some(policy.enabled && policy.managed_no_spill);
        }
        let requested_cuda = matches!(&device, NativeDecodeDevice::Cuda { .. });
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
            NativeDecodeDevice::Plugin { .. } => DevicePreference::Cpu,
        };
        let path = path.as_ref();
        let mut builder = InferenceSession::builder().model(path).device(preference);
        #[cfg(feature = "cuda")]
        if let (NativeDecodeDevice::Cuda { index }, Some(governor)) = (&device, cuda_governor) {
            let ep = onnx_runtime_ep_cuda::CudaExecutionProvider::
                initialized_with_offload_policy_and_governor(
                    index.unwrap_or(0),
                    cuda_offload_policy
                        .unwrap_or_else(onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env),
                    governor,
                )
                .context("initialize governed native CUDA execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        if let NativeDecodeDevice::Plugin {
            library,
            registration_name,
            provider_name,
        } = device
        {
            let ep = onnx_runtime_session::PluginExecutionProvider::new(
                library,
                registration_name,
                provider_name.clone(),
                provider_name,
            )
            .context("initialize native plugin execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        let session = builder.build().context("load native decoder model")?;
        if requested_cuda && let Some(report) = session.execution_provider_fallback_report() {
            tracing::warn!(
                model = %path.display(),
                fallback = %report,
                "native CUDA decoder fell back to CPU"
            );
        }
        Self::from_session_with_cuda_options_and_io(session, options, io)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options(session, NativeDecodeCudaOptions::default())
    }

    /// Wrap an already-built native session with an explicit [`ModelIoSpec`],
    /// used when the graph's ports cannot be disambiguated by shape/dtype alone
    /// (e.g. the synthetic decoder whose `input_ids`/`attention_mask`/
    /// `position_ids` are all `[-1, -1]` Int64). The declared spec is
    /// authoritative.
    pub fn from_session_with_io(
        session: InferenceSession,
        io: &ModelIoSpec,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options_and_io(
            session,
            NativeDecodeCudaOptions::default(),
            Some(io),
        )
    }

    #[cfg(test)]
    pub(crate) fn from_session_with_cuda_kv_max_len_and_io(
        session: InferenceSession,
        cuda_kv_max_len: Option<usize>,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options_and_io(
            session,
            NativeDecodeCudaOptions {
                decode_batch: None,
                kv_max_len: cuda_kv_max_len,
                metadata_max_len: None,
                graph_capture: None,
                weight_offload_enabled: None,
                weight_offload_stable_va: None,
            },
            io,
        )
    }

    pub(crate) fn from_session_with_cuda_options(
        session: InferenceSession,
        cuda_options: NativeDecodeCudaOptions,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options_and_io(session, cuda_options, None)
    }

    /// Best-effort auto-derived [`ModelIoSpec`] for a stock export whose sidecar
    /// declares no `io` block, built purely from the session's graph ports.
    ///
    /// Reuses the guarded genai-config derivation
    /// ([`GenAiConfig::derive_decoder_io_from_graph`]) so recurrent
    /// `conv_state`/`recurrent_state` ports are classified as loop-carried
    /// `state_pairs` and never confused with growable KV. Returns `None` (leaving
    /// the caller's `io = None` shape-inference path untouched) unless the
    /// derivation yields at least one recurrent state pair — the exact case the
    /// shape-inference path cannot resolve. Non-KV ports (token/mask/position/
    /// logits) are bound by conventional-name presence in the graph interface.
    ///
    /// The state-pair condition is enforced *here* rather than inside the shared
    /// derivation. #1012 widened that derivation to fire on KV ports so a
    /// DeepSeek-V2 MLA package, whose scalar `decoder.head_size` cannot express
    /// asymmetric KV, gets a port contract instead of failing its load. That is
    /// right for the genai-config path, whose caller has already established that
    /// no contract exists — but it is wrong here, where returning `None` still
    /// leaves a working shape-inference path. Letting it fire on dense graphs
    /// silently auto-bound roles that this path is supposed to refuse, so a
    /// decoder with genuinely ambiguous ports loaded against guessed bindings
    /// instead of demanding `model.io`.
    fn derive_fallback_io(session: &InferenceSession) -> Option<ModelIoSpec> {
        let to_graph_tensor =
            |meta: &onnx_runtime_session::IoMeta| onnx_genai_genai_config::GraphTensorInfo {
                name: meta.name.clone(),
                dtype: crate::engine::ir_dtype_name(meta.dtype).to_owned(),
                dimensions: meta
                    .shape
                    .iter()
                    .map(|dim| match dim {
                        Dim::Static(value) => Some(*value),
                        Dim::Symbolic(_) => None,
                    })
                    .collect(),
            };
        let graph = onnx_genai_genai_config::ModelGraphInfo {
            inputs: session.inputs().iter().map(to_graph_tensor).collect(),
            outputs: session.outputs().iter().map(to_graph_tensor).collect(),
        };
        onnx_genai_genai_config::GenAiConfig::derive_model_io_spec_from_graph(&graph).filter(
            |derived| {
                derived
                    .state_pairs
                    .as_ref()
                    .is_some_and(|pairs| !pairs.is_empty())
            },
        )
    }

    pub(crate) fn from_session_with_cuda_options_and_io(
        mut session: InferenceSession,
        cuda_options: NativeDecodeCudaOptions,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        // Auto-derive a decoder I/O spec from the graph ports when the model
        // package declares none. Declared `io` always wins; this fallback is
        // additive and engages ONLY for hybrid linear-attention decoders (models
        // exposing recurrent `conv_state`/`recurrent_state` state pairs the
        // shape-inference path cannot classify). Pure-dense decoders derive no
        // state pairs and keep their existing `io = None` load path unchanged, so
        // no currently-loadable model changes behavior. See #384.
        let derived_io: Option<ModelIoSpec> = if io.is_none() {
            Self::derive_fallback_io(&session)
        } else {
            None
        };
        let io = io.or(derived_io.as_ref());
        let mut io_span = onnx_genai_ort::prof_span!("native.inspect_decode_io");
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let role_inputs = role_tensor_info(session.inputs());
        let role_outputs = role_tensor_info(session.outputs());

        let sequence_source = io
            .and_then(|io| io.sequence_source)
            .unwrap_or(SequenceInputKind::TokenIds);
        let token_input = if sequence_source == SequenceInputKind::TokenIds {
            Some(declared_or_detected_input(
                &role_inputs,
                io.and_then(|io| io.token_input.as_deref()),
                StructuralRole::IntegerSequence,
                "model.io",
                "token_input",
            )?)
        } else {
            io.and_then(|io| io.token_input.as_deref())
                .map(|name| {
                    declared_or_detected_input(
                        &role_inputs,
                        Some(name),
                        StructuralRole::IntegerSequence,
                        "model.io",
                        "token_input",
                    )
                })
                .transpose()?
        };
        let inputs_embeds_input = if sequence_source == SequenceInputKind::InputsEmbeds {
            Some(declared_or_detected_input(
                &role_inputs,
                io.and_then(|io| io.inputs_embeds_input.as_deref()),
                StructuralRole::EmbeddingSequence,
                "model.io",
                "inputs_embeds_input",
            )?)
        } else {
            io.and_then(|io| io.inputs_embeds_input.as_deref())
                .map(|name| {
                    declared_or_detected_input(
                        &role_inputs,
                        Some(name),
                        StructuralRole::EmbeddingSequence,
                        "model.io",
                        "inputs_embeds_input",
                    )
                })
                .transpose()?
        };
        let attention_mask = optional_declared_or_detected_input(
            &role_inputs,
            io.and_then(|io| io.attention_mask_input.as_deref()),
            StructuralRole::None,
            "model.io",
            "attention_mask_input",
        )?;
        let position_ids = optional_declared_or_detected_input(
            &role_inputs,
            io.and_then(|io| io.position_ids_input.as_deref()),
            StructuralRole::None,
            "model.io",
            "position_ids_input",
        )?;
        let position_rank = declared_position_rank(&role_inputs, position_ids.as_deref())?;
        let logits = declared_or_detected_output(
            &role_outputs,
            io.and_then(|io| io.logits_output.as_deref()),
            StructuralRole::ScoreOutput,
            "model.io",
            "logits_output",
        )?;
        let hidden_output = optional_declared_or_detected_output(
            &role_outputs,
            io.and_then(|io| io.hidden_output.as_deref()),
            StructuralRole::None,
            "model.io",
            "hidden_output",
        )?;
        let kv_ownership = io
            .and_then(|io| io.kv_ownership)
            .unwrap_or(KvOwnership::Owned);
        if kv_ownership != KvOwnership::Owned {
            bail!(
                "native target decoder requires metadata kv_ownership 'owned'; got '{kv_ownership:?}'. Shared KV is valid for proposer graphs that reference this target's cache"
            );
        }
        let (mut kv_inputs, mut present_outputs) = match io {
            Some(io) => match (&io.kv_inputs, &io.kv_outputs) {
                (Some(inputs), Some(outputs)) => (inputs.clone(), outputs.clone()),
                (None, None) => (Vec::new(), Vec::new()),
                _ => bail!(
                    "native target decoder metadata must declare model.io.kv_inputs and model.io.kv_outputs together"
                ),
            },
            None => (Vec::new(), Vec::new()),
        };

        // Fixed loop-carried recurrent states (hybrid linear-attention
        // `conv_state` / `recurrent_state`) are declared through `io.state_pairs`
        // rather than the growable `kv_inputs`/`kv_outputs` lists. The native
        // decode loop binds any past→present pair the same way — it seeds each
        // past input (recurrent states are seeded at their full static extent by
        // `make_empty_input_tensor`) and copies the present output back each step
        // (`replace` semantics fall out naturally from the wholesale tensor swap).
        // So fold the declared state pairs into the same positionally-paired
        // lists. This is what lets hybrid SSM/attention decoders (qwen3.5) decode:
        // their linear-attention layers carry state only through these pairs.
        // Appending to both lists in the same order keeps the positional zip below
        // correct; `present_to_past` also records each pair explicitly, so the
        // recurrent tail never depends on the zip.
        let mut state_pairs: Vec<(String, String)> = Vec::new();
        if let Some(pairs) = io.and_then(|io| io.state_pairs.as_ref()) {
            for pair in pairs {
                kv_inputs.push(pair.input.clone());
                present_outputs.push(pair.output.clone());
                state_pairs.push((pair.output.clone(), pair.input.clone()));
            }
        }
        let fixed_state_inputs = state_pairs
            .iter()
            .map(|(_, input)| input.clone())
            .collect::<HashSet<_>>();

        if kv_inputs.is_empty() || present_outputs.is_empty() {
            bail!(
                "native decode requires explicit decoder state; declare model.io.kv_inputs and model.io.kv_outputs (or model.io.state_pairs)"
            );
        }

        let mut present_to_past = HashMap::new();
        // KV lists pair positionally; state pairs carry explicit names.
        let kv_pair_count = kv_inputs.len() - state_pairs.len();
        present_to_past.extend(
            present_outputs
                .iter()
                .take(kv_pair_count)
                .cloned()
                .zip(kv_inputs.iter().take(kv_pair_count).cloned()),
        );
        present_to_past.extend(state_pairs.iter().cloned());
        if present_to_past.len() != kv_inputs.len() {
            bail!(
                "native decoder has incomplete past/present pairs; past inputs: {kv_inputs:?}, present outputs: {present_outputs:?}"
            );
        }

        let mut declared_sources = HashMap::new();
        for (name, source) in [
            (token_input.as_deref(), NativeStepInputSource::TokenIds),
            (
                inputs_embeds_input.as_deref(),
                NativeStepInputSource::InputsEmbeds,
            ),
            (
                attention_mask.as_deref(),
                NativeStepInputSource::AttentionMask,
            ),
            (position_ids.as_deref(), NativeStepInputSource::PositionIds),
        ] {
            let Some(name) = name else {
                continue;
            };
            if let Some(existing) = declared_sources.insert(name.to_owned(), source) {
                bail!(
                    "native target decoder metadata assigns input '{name}' to both {existing:?} and {source:?}; each generated step-input role must name a distinct graph port"
                );
            }
        }
        let step_inputs = input_names
            .iter()
            .filter(|name| !kv_inputs.contains(name))
            .map(|name| NativeStepInputBinding {
                name: name.clone(),
                source: declared_sources
                    .get(name)
                    .copied()
                    .unwrap_or(NativeStepInputSource::Routed),
            })
            .collect::<Vec<_>>();
        let required_sequence_source = match sequence_source {
            SequenceInputKind::TokenIds => NativeStepInputSource::TokenIds,
            SequenceInputKind::InputsEmbeds => NativeStepInputSource::InputsEmbeds,
        };
        if !step_inputs
            .iter()
            .any(|binding| binding.source == required_sequence_source)
        {
            bail!(
                "native target decoder metadata sequence_source '{sequence_source:?}' has no matching declared graph input"
            );
        }
        io_span.set_arg("inputs", input_names.len() as u64);
        io_span.set_arg("outputs", output_names.len() as u64);
        io_span.set_arg("kv_pairs", present_to_past.len() as u64);
        io_span.set_arg("step_inputs", step_inputs.len() as u64);
        drop(io_span);

        let cuda = if session.device_id().device_type == DeviceType::Cuda {
            // Inc3a: the CUDA native decoder accepts a metadata-declared
            // `inputs_embeds` sequence source (a fused VLM decoder). Inc3b: it
            // also accepts generic `Routed` non-KV ports — bound as owned per-
            // step device uploads (see `decode_cuda_eager_step_inputs`), so no
            // load-time refusal is needed. Vision cross-KV remains out of scope
            // (blocked on the vision Attention float-mask fixes), but those ports
            // are not declared on a text-decoder contract, so nothing to gate
            // here.
            let attention_mask = attention_mask
                .as_deref()
                .context("native CUDA target decode requires a declared attention-mask input")?;
            // Resolve the sequence input: token ids (Int64 `[1, 1]`) by default,
            // or a float `inputs_embeds` `[1, 1, hidden]` for a fused decoder.
            let inputs_embeds = if sequence_source == SequenceInputKind::InputsEmbeds {
                let name = inputs_embeds_input.as_deref().context(
                    "native CUDA target decode declares inputs_embeds but is missing its embedding input binding",
                )?;
                let meta = session
                    .inputs()
                    .iter()
                    .find(|meta| meta.name == name)
                    .with_context(|| {
                        format!("missing CUDA inputs_embeds input metadata for '{name}'")
                    })?;
                if !matches!(
                    meta.dtype,
                    DataType::Float32 | DataType::Float16 | DataType::BFloat16
                ) {
                    bail!(
                        "native CUDA inputs_embeds input '{name}' must be f32, f16 or bf16, got {:?} {:?}",
                        meta.dtype,
                        meta.shape
                    );
                }
                let hidden = match meta.shape.iter().copied().next_back() {
                    Some(Dim::Static(value)) => value,
                    _ => bail!(
                        "native CUDA inputs_embeds input '{name}' needs a static hidden dimension, got shape {:?}",
                        meta.shape
                    ),
                };
                Some(CudaEmbedsBinding {
                    name,
                    dtype: meta.dtype,
                    hidden,
                })
            } else {
                None
            };
            // Inc3c: resolve every routed (non-KV, non-generated) port to a fixed
            // single-token device binding so the capture path can replay the step.
            // Dynamic dims (batch / sequence) collapse to 1; static feature dims
            // are kept. The eager path (default) ignores these bindings.
            let routed = step_inputs
                .iter()
                .filter(|binding| binding.source == NativeStepInputSource::Routed)
                .map(|binding| {
                    let meta = session
                        .inputs()
                        .iter()
                        .find(|meta| meta.name == binding.name)
                        .with_context(|| {
                            format!("missing CUDA routed input metadata for '{}'", binding.name)
                        })?;
                    let shape = meta
                        .shape
                        .iter()
                        .map(|dim| match dim {
                            Dim::Static(value) => *value,
                            _ => 1,
                        })
                        .collect::<Vec<usize>>();
                    Ok(CudaRoutedBinding {
                        name: binding.name.as_str(),
                        dtype: meta.dtype,
                        shape,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let token_input = if inputs_embeds.is_some() {
                ""
            } else {
                token_input
                    .as_deref()
                    .context("native CUDA target decode is missing its token input binding")?
            };
            let bytes_per_token = DecodeCudaState::kv_bytes_per_token(
                &session,
                &present_to_past,
                &fixed_state_inputs,
            )?;
            let device_memory = cuda_device_memory_snapshot(session.device_id().index as i32).ok();
            let max_len = match cuda_options.kv_max_len {
                Some(0) => bail!("CUDA KV max length must be greater than zero"),
                Some(value) => Some(value),
                None => None,
            };
            let env_max_len = if max_len.is_some() {
                None
            } else {
                cuda_kv_max_len_from_env()?
            };
            let capacity = resolve_cuda_kv_capacity(
                max_len,
                env_max_len,
                cuda_options.metadata_max_len,
                bytes_per_token,
                device_memory,
            )?;
            let runtime_config = onnx_genai_runtime_config::runtime_config();
            // Live weight offload is a CUDA-EP feature and is mutually exclusive
            // with graph capture; when the CUDA EP isn't compiled in there is no
            // pager, so offload is unconditionally off here.
            #[cfg(feature = "cuda")]
            let weight_offload_enabled = cuda_options
                .weight_offload_enabled
                .unwrap_or_else(|| onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env().enabled);
            #[cfg(not(feature = "cuda"))]
            let weight_offload_enabled = false;
            // Issue #716: offload no longer forces capture OFF when it runs on
            // the stable-VA VMM paging path. Pass the three-state Option through
            // so the decline message can distinguish "policy proved unstable"
            // (`Some(false)`) from "no policy supplied, conservative default"
            // (`None`) — collapsing them to a bool here is what let a plumbing
            // gap read as a runtime limitation.
            let weight_offload_stable_va = cuda_options.weight_offload_stable_va;
            let graph_capture = resolve_graph_capture_decision(
                cuda_options.graph_capture,
                runtime_config.cuda_graph_explicit,
                runtime_config.cuda_graph,
                GraphCaptureStructuralSafety {
                    device_is_cuda: true,
                    kv_ownership,
                },
                weight_offload_enabled,
                weight_offload_stable_va,
            );
            let mut span = onnx_genai_ort::prof_span!("native.cuda_kv_alloc");
            span.set_arg("max_len", capacity.max_len as u64);
            span.set_arg("kv_pairs", present_to_past.len() as u64);
            span.set_arg("graph_capture", graph_capture.is_enabled());
            let kv_layout = crate::native_decode::cuda::resolve_cuda_kv_layout(
                io.and_then(|io| io.kv_layout.as_ref()),
            );
            Some(DecodeCudaState::new(
                &mut session,
                DecodeCudaIo {
                    input_ids: token_input,
                    inputs_embeds,
                    attention_mask,
                    position_ids: position_ids.as_deref(),
                    logits: &logits,
                    routed,
                },
                &present_to_past,
                &fixed_state_inputs,
                capacity,
                graph_capture,
                position_rank,
                kv_layout,
                cuda_options.decode_batch,
            )?)
        } else {
            None
        };

        // Persistent in-place CPU KV: eligible only for pure-attention decoders
        // (no recurrent state pairs, which are replaced wholesale each step and
        // cannot be appended in place) running on the CPU device. Gated behind
        // `ONNX_GENAI_CPU_INPLACE_KV` (default on; set to 0 to force the legacy
        // host round-trip). Any ineligible KV geometry disables it transparently.
        let cpu_kv =
            if session.device_id().device_type != DeviceType::Cuda && state_pairs.is_empty() {
                match cpu_inplace_kv_max_len_from_env()? {
                    Some(max_len) => {
                        let mut span = onnx_genai_ort::prof_span!("native.cpu_kv_alloc");
                        span.set_arg("max_len", max_len as u64);
                        span.set_arg("kv_pairs", present_to_past.len() as u64);
                        DecodeCpuKvState::new(&mut session, &present_to_past, max_len)?
                    }
                    None => None,
                }
            } else {
                None
            };

        let has_plugin_fused = graph_has_plugin_fused(session.graph());
        let uses_decode_pool = graph_uses_decode_pool(session.graph());
        Ok(Self {
            session,
            step_inputs,
            logits,
            hidden_output,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            cuda,
            cpu_kv,
            trace: TraceContext::noop(),
            current_len: 0,
            last_hidden: None,
            uses_decode_pool,
            has_plugin_fused,
            position_rank,
            decode_inline: DecodeInlineState::Untried,
            prefill_chunk_size: None,
            prefill_query_padding: true,
        })
    }
}
