use super::*;

impl NativeDecodeSession {
    pub(crate) fn load_with_weight_offload_host_cache(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        host_cache: onnx_runtime_ep_cpu::WeightOffloadHostCache,
        io: Option<&ModelIoSpec>,
        decode_precision: DecodePrecision,
        lora_adapter: Option<onnx_runtime_session::lora_inject::LoraAdapterSpec>,
        lora_target_manifest: Option<onnx_genai_metadata::LoraTargetManifest>,
    ) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
            NativeDecodeDevice::Plugin { .. } => DevicePreference::Cpu,
        };
        let has_lora_adapter = lora_adapter.is_some();
        let mut builder = InferenceSession::builder()
            .model(path)
            .device(preference)
            .decode_precision(decode_precision);
        if let Some(adapter) = lora_adapter {
            // CUDA phase (P5): native LoRA is CPU-only for Phase 1. Reject a GPU
            // device up front rather than silently injecting an inert branch.
            if !matches!(device, NativeDecodeDevice::Cpu) {
                anyhow::bail!(
                    "native LoRA adapters are only supported on the CPU device in this phase \
                     (CUDA is deferred to P5)"
                );
            }
            builder = builder.lora_adapter(adapter);
        }
        if let Some(manifest) = lora_target_manifest {
            builder = builder.lora_target_manifest(manifest);
        }
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
        let mut session = builder.build().context("load native decoder model")?;
        // Single fixed adapter per session (P4): the adapter was injected at
        // build time, so activation is a cheap toggle that feeds the override
        // buffers for every subsequent decode step until deactivated.
        if has_lora_adapter {
            session.set_lora_active(true);
        }
        Self::from_session_with_cuda_kv_max_len_and_io(session, None, io)
    }

    /// Load with an explicit CUDA KV capacity. `None` uses
    /// `ONNX_GENAI_CUDA_KV_MAX_LEN`, then the 4096-token default.
    pub fn load_with_cuda_kv_max_len(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        cuda_kv_max_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::load_with_cuda_options(
            path,
            device,
            NativeDecodeCudaOptions {
                kv_max_len: cuda_kv_max_len,
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
        Self::from_session_with_cuda_options(session, options)
    }

    /// Wrap an already-built native session, validating its decoder-with-past I/O.
    pub fn from_session(session: InferenceSession) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options(session, NativeDecodeCudaOptions::default())
    }

    pub(crate) fn from_session_with_cuda_kv_max_len_and_io(
        session: InferenceSession,
        cuda_kv_max_len: Option<usize>,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        Self::from_session_with_cuda_options_and_io(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: cuda_kv_max_len,
                graph_capture: None,
            },
            io,
        )
    }

    fn from_session_with_cuda_options(
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

        let sequence_source = io
            .and_then(|io| io.sequence_source)
            .unwrap_or(SequenceInputKind::TokenIds);
        let token_input = if sequence_source == SequenceInputKind::TokenIds {
            Some(declared_or_detected_input(
                &input_names,
                io.and_then(|io| io.token_input.as_deref()),
                &["input_ids", "decoder_input_ids"],
                "token_input",
            )?)
        } else {
            io.and_then(|io| io.token_input.as_deref())
                .map(|name| {
                    declared_or_detected_input(
                        &input_names,
                        Some(name),
                        &["input_ids", "decoder_input_ids"],
                        "token_input",
                    )
                })
                .transpose()?
        };
        let inputs_embeds_input = if sequence_source == SequenceInputKind::InputsEmbeds {
            Some(declared_or_detected_input(
                &input_names,
                io.and_then(|io| io.inputs_embeds_input.as_deref()),
                &["inputs_embeds"],
                "inputs_embeds_input",
            )?)
        } else {
            io.and_then(|io| io.inputs_embeds_input.as_deref())
                .map(|name| {
                    declared_or_detected_input(
                        &input_names,
                        Some(name),
                        &["inputs_embeds"],
                        "inputs_embeds_input",
                    )
                })
                .transpose()?
        };
        let attention_mask = optional_declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.attention_mask_input.as_deref()),
            &["attention_mask"],
            "attention_mask_input",
        )?;
        let position_ids = optional_declared_or_detected_input(
            &input_names,
            io.and_then(|io| io.position_ids_input.as_deref()),
            &["position_ids"],
            "position_ids_input",
        )?;
        let logits = declared_or_detected_output(
            &output_names,
            io.and_then(|io| io.logits_output.as_deref()),
            &["logits"],
            "logits_output",
        )?;
        let hidden_output = optional_declared_or_detected_output(
            &output_names,
            io.and_then(|io| io.hidden_output.as_deref()),
            &[],
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
                    "native target decoder metadata must declare io.kv_inputs and io.kv_outputs together"
                ),
            },
            None => (
                input_names
                    .iter()
                    .filter(|name| is_past_name(name))
                    .cloned()
                    .collect(),
                output_names
                    .iter()
                    .filter(|name| is_present_name(name))
                    .cloned()
                    .collect(),
            ),
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

        if kv_inputs.is_empty() || present_outputs.is_empty() {
            bail!(
                "native decode requires decoder-with-past I/O; past inputs: {kv_inputs:?}, present outputs: {present_outputs:?}"
            );
        }

        let mut present_to_past = HashMap::new();
        if io.is_some() {
            // KV pairs pair positionally; state pairs carry explicit names.
            let kv_pair_count = kv_inputs.len() - state_pairs.len();
            present_to_past.extend(
                present_outputs
                    .iter()
                    .take(kv_pair_count)
                    .cloned()
                    .zip(kv_inputs.iter().take(kv_pair_count).cloned()),
            );
            present_to_past.extend(state_pairs.iter().cloned());
        } else {
            for output in &present_outputs {
                let Some(input) = matching_past_name(output, &kv_inputs) else {
                    bail!(
                        "native decoder present output '{output}' has no matching past input; inputs: {kv_inputs:?}"
                    );
                };
                present_to_past.insert(output.clone(), input);
            }
        }
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
            if sequence_source != SequenceInputKind::TokenIds
                || step_inputs.iter().any(|binding| {
                    matches!(
                        binding.source,
                        NativeStepInputSource::InputsEmbeds | NativeStepInputSource::Routed
                    )
                })
            {
                bail!(
                    "native CUDA target decode does not yet support metadata-declared embedding or routed step inputs; use the CPU native device for this contract until generic device bindings are implemented"
                );
            }
            let token_input = token_input
                .as_deref()
                .context("native CUDA target decode is missing its token input binding")?;
            let attention_mask = attention_mask
                .as_deref()
                .context("native CUDA target decode requires a declared attention-mask input")?;
            let max_len = match cuda_options.kv_max_len {
                Some(0) => bail!("CUDA KV max length must be greater than zero"),
                Some(value) => value,
                None => cuda_kv_max_len_from_env()?,
            };
            let runtime_config = onnx_genai_runtime_config::runtime_config();
            let graph_enabled = resolve_graph_capture_enabled(
                cuda_options.graph_capture,
                runtime_config.cuda_graph_explicit,
                runtime_config.cuda_graph,
                GraphCaptureStructuralSafety {
                    device_is_cuda: true,
                    kv_ownership,
                },
            );
            let mut span = onnx_genai_ort::prof_span!("native.cuda_kv_alloc");
            span.set_arg("max_len", max_len as u64);
            span.set_arg("kv_pairs", present_to_past.len() as u64);
            span.set_arg("graph_capture", graph_enabled);
            Some(DecodeCudaState::new(
                &mut session,
                DecodeCudaIo {
                    input_ids: token_input,
                    attention_mask,
                    position_ids: position_ids.as_deref(),
                    logits: &logits,
                },
                &present_to_past,
                max_len,
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
