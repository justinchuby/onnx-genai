use super::*;

/// Semantic outputs of one native proposer forward.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeProposerOutput {
    pub logits: Option<Vec<Vec<f32>>>,
    pub projected_state: Option<Vec<f32>>,
}

/// Metadata-driven native execution adapter for speculative proposer graphs.
///
/// Unlike [`NativeDecodeSession`], this adapter accepts either token ids or
/// precomputed embeddings and supports both graph-owned past/present KV and
/// target-owned shared-KV inputs.
pub(crate) struct NativeProposerSession {
    session: InferenceSession,
    sequence_source: SequenceInputKind,
    sequence_input: String,
    attention_mask: Option<String>,
    position_ids: Option<String>,
    logits_output: Option<String>,
    projected_state_output: Option<String>,
    kv_ownership: KvOwnership,
    kv_inputs: Vec<String>,
    present_to_past: Vec<(String, String)>,
    past: HashMap<String, Tensor>,
    pub(crate) current_len: usize,
    uses_decode_pool: bool,
    has_plugin_fused: bool,
}

#[allow(dead_code)]
impl NativeProposerSession {
    pub(crate) fn load(
        path: impl AsRef<Path>,
        device: NativeDecodeDevice,
        io: Option<&ModelIoSpec>,
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
        let session = builder.build().context("load native proposer model")?;
        Self::from_session(session, io)
    }

    pub(crate) fn from_session(
        session: InferenceSession,
        io: Option<&ModelIoSpec>,
    ) -> anyhow::Result<Self> {
        let role_inputs = role_tensor_info(session.inputs());
        let role_outputs = role_tensor_info(session.outputs());
        let sequence_source = io
            .and_then(|io| io.sequence_source)
            .unwrap_or(SequenceInputKind::TokenIds);
        let sequence_input = match sequence_source {
            SequenceInputKind::TokenIds => declared_or_detected_input(
                &role_inputs,
                io.and_then(|io| io.token_input.as_deref()),
                StructuralRole::IntegerSequence,
                "speculative.io",
                "token_input",
            )?,
            SequenceInputKind::InputsEmbeds => declared_or_detected_input(
                &role_inputs,
                io.and_then(|io| io.inputs_embeds_input.as_deref()),
                StructuralRole::EmbeddingSequence,
                "speculative.io",
                "inputs_embeds_input",
            )?,
        };
        let attention_mask = optional_declared_or_detected_input(
            &role_inputs,
            io.and_then(|io| io.attention_mask_input.as_deref()),
            StructuralRole::None,
            "speculative.io",
            "attention_mask_input",
        )?;
        let position_ids = optional_declared_or_detected_input(
            &role_inputs,
            io.and_then(|io| io.position_ids_input.as_deref()),
            StructuralRole::None,
            "speculative.io",
            "position_ids_input",
        )?;
        let logits_output = optional_declared_or_detected_output(
            &role_outputs,
            io.and_then(|io| io.logits_output.as_deref()),
            StructuralRole::ScoreOutput,
            "speculative.io",
            "logits_output",
        )?;
        let projected_state_output = optional_declared_or_detected_output(
            &role_outputs,
            io.and_then(|io| io.hidden_output.as_deref()),
            StructuralRole::None,
            "speculative.io",
            "hidden_output",
        )?;
        if logits_output.is_none() && projected_state_output.is_none() {
            bail!(
                "native proposer metadata must declare at least one semantic output role: speculative.io.logits_output or speculative.io.hidden_output"
            );
        }

        let kv_ownership = io
            .and_then(|io| io.kv_ownership)
            .unwrap_or(KvOwnership::Owned);
        let (kv_inputs, present_to_past) = match kv_ownership {
            KvOwnership::Owned => {
                let (inputs, outputs) = match io {
                    Some(io) => match (&io.kv_inputs, &io.kv_outputs) {
                        (Some(inputs), Some(outputs)) => (inputs.clone(), outputs.clone()),
                        (None, None) => (Vec::new(), Vec::new()),
                        _ => bail!(
                            "native proposer metadata must declare io.kv_inputs and io.kv_outputs together for owned KV"
                        ),
                    },
                    None => (Vec::new(), Vec::new()),
                };
                if inputs.len() != outputs.len() {
                    bail!(
                        "native proposer owned-KV contract has {} past inputs but {} present outputs; declare equal positional lists",
                        inputs.len(),
                        outputs.len()
                    );
                }
                if inputs.is_empty() != outputs.is_empty() {
                    bail!(
                        "native proposer owned KV requires speculative.io.kv_inputs and speculative.io.kv_outputs"
                    );
                }
                let pairs = outputs.into_iter().zip(inputs.iter().cloned()).collect();
                (inputs, pairs)
            }
            KvOwnership::Shared => {
                if io.is_some_and(|io| io.kv_outputs.as_ref().is_some_and(|v| !v.is_empty())) {
                    bail!(
                        "native proposer metadata declares kv_ownership 'shared' but also declares io.kv_outputs; shared-KV proposers reference target cache and must not emit owned present KV"
                    );
                }
                (Vec::new(), Vec::new())
            }
        };

        let has_plugin_fused = graph_has_plugin_fused(session.graph());
        let uses_decode_pool = graph_uses_decode_pool(session.graph());
        Ok(Self {
            session,
            sequence_source,
            sequence_input,
            attention_mask,
            position_ids,
            logits_output,
            projected_state_output,
            kv_ownership,
            kv_inputs,
            present_to_past,
            past: HashMap::new(),
            current_len: 0,
            uses_decode_pool,
            has_plugin_fused,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.past.clear();
        self.current_len = 0;
    }

    #[cfg(test)]
    pub(crate) fn step_token_ids(
        &mut self,
        token_ids: &[TokenId],
    ) -> anyhow::Result<NativeProposerOutput> {
        if self.sequence_source != SequenceInputKind::TokenIds {
            bail!(
                "native proposer contract requires inputs_embeds, but token ids were supplied; build embeddings and call step_inputs_embeds"
            );
        }
        if token_ids.is_empty() {
            bail!("native proposer token input must contain at least one token");
        }
        let values = token_ids
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        let sequence = Tensor::from_i64(&[1, token_ids.len()], &values)?;
        self.run_step(sequence, token_ids.len(), self.current_len, &[])
    }

    pub(crate) fn step_inputs_embeds(
        &mut self,
        inputs_embeds: &[f32],
        position_start: usize,
        shared_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<NativeProposerOutput> {
        if self.sequence_source != SequenceInputKind::InputsEmbeds {
            bail!(
                "native proposer contract requires token_ids, but embeddings were supplied; call step_token_ids"
            );
        }
        let meta = self
            .session
            .inputs()
            .iter()
            .find(|meta| meta.name == self.sequence_input)
            .context("native proposer sequence input metadata disappeared")?;
        let width = match meta.shape.last() {
            Some(Dim::Static(width)) if *width > 0 => *width,
            _ => bail!(
                "native proposer inputs_embeds '{}' must declare a positive static final width, got {:?}; export the embedding width in the ONNX type",
                self.sequence_input,
                meta.shape
            ),
        };
        if inputs_embeds.is_empty() || !inputs_embeds.len().is_multiple_of(width) {
            bail!(
                "native proposer inputs_embeds length {} must be a non-zero multiple of declared width {width}",
                inputs_embeds.len()
            );
        }
        let sequence_len = inputs_embeds.len() / width;
        let sequence = tensor_from_f32_as(meta.dtype, &[1, sequence_len, width], inputs_embeds)
            .with_context(|| {
                format!(
                    "build native proposer embeddings for input '{}'",
                    self.sequence_input
                )
            })?;
        self.run_step(sequence, sequence_len, position_start, shared_inputs)
    }

    fn run_step(
        &mut self,
        sequence: Tensor,
        sequence_len: usize,
        position_start: usize,
        shared_inputs: &[(String, Tensor)],
    ) -> anyhow::Result<NativeProposerOutput> {
        let total_len = self
            .current_len
            .checked_add(sequence_len)
            .context("native proposer context length overflow")?;
        let shared_kv_len = shared_inputs
            .iter()
            .filter_map(|(_, tensor)| {
                tensor
                    .shape
                    .len()
                    .checked_sub(2)
                    .map(|axis| tensor.shape[axis])
            })
            .max()
            .unwrap_or(total_len);
        let mut owned = vec![(self.sequence_input.clone(), sequence)];
        if let Some(name) = &self.attention_mask {
            owned.push((
                name.clone(),
                Tensor::from_i64(&[1, shared_kv_len], &vec![1; shared_kv_len])?,
            ));
        }
        if let Some(name) = &self.position_ids {
            let position_end = position_start
                .checked_add(sequence_len)
                .context("native proposer position range overflow")?;
            let positions = (position_start..position_end)
                .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            owned.push((
                name.clone(),
                Tensor::from_i64(&[1, sequence_len], &positions)?,
            ));
        }
        match self.kv_ownership {
            KvOwnership::Owned => {
                for name in &self.kv_inputs {
                    let tensor = match self.past.remove(name) {
                        Some(tensor) => tensor,
                        None => make_empty_input_tensor(&self.session, name)?,
                    };
                    owned.push((name.clone(), tensor));
                }
            }
            KvOwnership::Shared => {
                for (name, tensor) in shared_inputs {
                    if !self.session.inputs().iter().any(|meta| meta.name == *name) {
                        bail!(
                            "native proposer shared-KV input '{name}' is not exposed by the graph; graph inputs: {:?}",
                            self.session
                                .inputs()
                                .iter()
                                .map(|meta| meta.name.as_str())
                                .collect::<Vec<_>>()
                        );
                    }
                    owned.push((name.clone(), tensor.clone()));
                }
            }
        }
        let bindings = owned
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let run_single_token = sequence_len == 1;
        let uses_decode_pool = self.uses_decode_pool;
        let outputs = if run_single_token && !self.has_plugin_fused {
            onnx_runtime_ep_cpu::with_decode_pool_scope(uses_decode_pool, || {
                self.session.run(&bindings).map_err(anyhow::Error::from)
            })
        } else {
            self.session.run(&bindings).map_err(anyhow::Error::from)
        }
        .context("native proposer forward pass failed; verify metadata port names, sequence_source, kv_ownership, and tensor shapes")?;
        let names = self
            .session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let mut named = names.into_iter().zip(outputs).collect::<HashMap<_, _>>();
        let logits = self
            .logits_output
            .as_ref()
            .map(|name| {
                let tensor = named.remove(name).with_context(|| {
                    format!("native proposer omitted declared logits output '{name}'")
                })?;
                extract_logits(&tensor)
            })
            .transpose()?;
        let projected_state = self
            .projected_state_output
            .as_ref()
            .map(|name| {
                let tensor = named.remove(name).with_context(|| {
                    format!("native proposer omitted declared hidden output '{name}'")
                })?;
                extract_last_row(&tensor)
            })
            .transpose()?;
        if self.kv_ownership == KvOwnership::Owned {
            let mut next = HashMap::with_capacity(self.present_to_past.len());
            for (present, past) in &self.present_to_past {
                let tensor = named.remove(present).with_context(|| {
                    format!("native proposer omitted declared present output '{present}'")
                })?;
                next.insert(past.clone(), tensor);
            }
            self.past = next;
            self.current_len = total_len;
        }
        Ok(NativeProposerOutput {
            logits,
            projected_state,
        })
    }
}
