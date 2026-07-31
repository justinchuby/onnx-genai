use super::*;

fn native_metadata_max_len_from_model_path(path: &Path) -> Option<usize> {
    let root = if path.is_dir() { path } else { path.parent()? };
    [
        "inference_metadata.yaml",
        "inference_metadata.yml",
        "inference_metadata.json",
    ]
    .iter()
    .map(|name| root.join(name))
    .find(|path| path.is_file())
    .and_then(|path| onnx_genai_metadata::load_metadata(&path).ok())
    .and_then(|metadata| metadata.model.and_then(|model| model.max_sequence_length))
}

impl NativeDecodeSession {
    pub(crate) fn load_with_weight_offload_host_cache(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
        io: Option<&ModelIoSpec>,
        metadata_max_len: Option<usize>,
        key_sequence_lengths_policy: crate::decode::KeySequenceLengthsPolicy,
        decode_precision: DecodePrecision,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
            NativeDecodeDevice::Plugin { .. } => DevicePreference::Cpu,
        };
        let mut builder = InferenceSession::builder()
            .model(path)
            .device(preference)
            .decode_precision(decode_precision);
        if device == NativeDecodeDevice::Cpu {
            let ep =
                onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_weight_offload_host_cache(
                    host_cache,
                )
                .context("initialize native CPU execution provider")?;
            builder = builder.execution_provider(Arc::new(ep));
        }
        #[cfg(feature = "cuda")]
        if let NativeDecodeDevice::Cuda { index } = device {
            let ep = onnx_runtime_ep_cuda::CudaExecutionProvider::initialized(index.unwrap_or(0))
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
        Self::validate_key_sequence_lengths_contract(&session, key_sequence_lengths_policy)?;
        Self::from_session_with_cuda_options_and_io(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: None,
                metadata_max_len,
                graph_capture: None,
            },
            io,
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
                kv_max_len: cuda_kv_max_len,
                metadata_max_len,
                graph_capture: None,
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
        Self::load_with_cuda_options_and_io(path, device, options, None)
    }

    /// Load a decoder-with-past model, threading the pipeline-declared
    /// [`ModelIoSpec`] so `sequence_source` (e.g. `inputs_embeds`), the KV pairs,
    /// and routed step inputs are bound from metadata rather than guessed from
    /// tensor shapes. The pipeline's native device-KV decoder (inc2b) uses this so
    /// an `inputs_embeds` decoder with no token input loads correctly.
    pub(crate) fn load_with_io(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options_and_io(path, device, NativeDecodeCudaOptions::default(), io)
    }

    fn load_with_cuda_options_and_io(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        mut options: NativeDecodeCudaOptions,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        if options.metadata_max_len.is_none() {
            options.metadata_max_len = native_metadata_max_len_from_model_path(path.as_ref());
        }
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
            NativeDecodeDevice::Plugin { .. } => DevicePreference::Cpu,
        };
        let mut builder = InferenceSession::builder().model(path).device(preference);
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
        Self::from_session_with_cuda_options_and_io(session, options, io)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options(session, NativeDecodeCudaOptions::default())
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
                kv_max_len: cuda_kv_max_len,
                metadata_max_len: None,
                graph_capture: None,
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

    fn from_session_with_cuda_options_and_io(
        mut session: InferenceSession,
        cuda_options: NativeDecodeCudaOptions,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
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
                if !matches!(meta.dtype, DataType::Float32 | DataType::Float16) {
                    bail!(
                        "native CUDA inputs_embeds input '{name}' must be f32 or f16, got {:?} {:?}",
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
            let weight_offload_enabled =
                onnx_runtime_ep_cuda::DeviceOffloadPolicy::from_env().enabled;
            #[cfg(not(feature = "cuda"))]
            let weight_offload_enabled = false;
            let graph_enabled = resolve_graph_capture_enabled(
                cuda_options.graph_capture,
                runtime_config.cuda_graph_explicit,
                runtime_config.cuda_graph,
                GraphCaptureStructuralSafety {
                    device_is_cuda: true,
                    kv_ownership,
                },
                weight_offload_enabled,
            );
            let mut span = onnx_genai_ort::prof_span!("native.cuda_kv_alloc");
            span.set_arg("max_len", capacity.max_len as u64);
            span.set_arg("kv_pairs", present_to_past.len() as u64);
            span.set_arg("graph_capture", graph_enabled);
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
                graph_enabled,
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
        })
    }
}
