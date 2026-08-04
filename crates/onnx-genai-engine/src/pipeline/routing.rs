//! Pipeline component routing and tensor preparation.
//!
//! Pure code motion from `pipeline.rs`: request-tensor preparation, prompt- and
//! step-phase component routing, decoder input-edge resolution, cross-attention
//! key/value binding, and the request-identity digest used for prefix reuse.

use super::*;

impl PipelineEngine {
    pub(crate) fn prepare_request_tensors(
        &self,
        inputs: PipelineTensors,
        present: &BTreeSet<String>,
    ) -> anyhow::Result<PipelineTensors> {
        if present.iter().any(String::is_empty) {
            anyhow::bail!("pipeline request presence keys must be non-empty");
        }
        let mut dimensions = HashMap::<String, i64>::new();

        for (component, model) in &self.models.directory.spec.models {
            let Some(io) = model.io.as_ref() else {
                continue;
            };
            let session = self
                .models
                .graph_io(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            for (port, optional) in &io.optional_inputs {
                let endpoint = format!("{component}.{port}");
                let route = self.plan.dataflow().iter().find(|edge| edge.to == endpoint);
                let supplied_endpoint = inputs
                    .get(&endpoint)
                    .map(|value| (endpoint.as_str(), value));
                let supplied_route = route.and_then(|edge| {
                    inputs
                        .get(&edge.from)
                        .map(|value| (edge.from.as_str(), value))
                });
                let supplied = supplied_endpoint.or(supplied_route);
                let is_present = present.contains(&optional.presence);

                if !is_present {
                    if let Some((supplied_name, _)) = supplied {
                        anyhow::bail!(
                            "pipeline input '{supplied_name}' is associated with presence key '{}' \
                             but that key was declared absent",
                            optional.presence
                        );
                    }
                } else if supplied.is_none() {
                    let active_route = route.is_some_and(|edge| {
                        endpoint_component(&edge.from).is_some_and(|producer| {
                            self.plan.component_is_present(producer, present)
                        })
                    });
                    if !active_route {
                        anyhow::bail!(
                            "missing optional-but-present pipeline input '{endpoint}' for presence \
                             key '{}': supply the destination endpoint or an active routed source",
                            optional.presence
                        );
                    }
                }

                let info = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == *port)
                    .with_context(|| {
                        format!(
                            "optional pipeline input '{endpoint}' is not exposed by its ONNX graph"
                        )
                    })?;
                if info.shape.len() != optional.absent.shape.len() {
                    anyhow::bail!(
                        "invalid fallback for optional pipeline input '{endpoint}': declared rank {} \
                         does not match graph rank {}",
                        optional.absent.shape.len(),
                        info.shape.len()
                    );
                }
                for (index, dimension) in optional.absent.shape.iter().enumerate() {
                    let TensorDimension::Symbol(symbol) = dimension else {
                        continue;
                    };
                    if info.shape[index] >= 0 {
                        bind_dimension(&mut dimensions, symbol, info.shape[index], &endpoint)?;
                    }
                    if let Some((_, value)) = supplied {
                        if value.shape().len() != optional.absent.shape.len() {
                            anyhow::bail!(
                                "pipeline input '{endpoint}' has rank {}, expected {} from its \
                                 optional-input contract",
                                value.shape().len(),
                                optional.absent.shape.len()
                            );
                        }
                        bind_dimension(&mut dimensions, symbol, value.shape()[index], &endpoint)?;
                    }
                }
            }
        }

        let mut tensors = inputs;
        for (component, model) in &self.models.directory.spec.models {
            let Some(io) = model.io.as_ref() else {
                continue;
            };
            let session = self
                .models
                .graph_io(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            for (port, optional) in &io.optional_inputs {
                if present.contains(&optional.presence) {
                    continue;
                }
                let endpoint = format!("{component}.{port}");
                if tensors.contains_key(&endpoint) {
                    continue;
                }
                let info = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == *port)
                    .with_context(|| {
                        format!(
                            "optional pipeline input '{endpoint}' is not exposed by its ONNX graph"
                        )
                    })?;
                let shape = optional
                    .absent
                    .shape
                    .iter()
                    .map(|dimension| match dimension {
                        TensorDimension::Fixed(value) => Ok(*value),
                        TensorDimension::Symbol(symbol) => {
                            dimensions.get(symbol).copied().with_context(|| {
                                format!(
                                    "unresolved fallback shape symbol '{symbol}' for optional \
                                     pipeline input '{endpoint}'"
                                )
                            })
                        }
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let value = match optional.absent.kind {
                    AbsentInputKind::Zeros => zero_value(&shape, info.dtype).with_context(|| {
                        format!(
                            "invalid fallback for optional pipeline input '{endpoint}' with dtype \
                             {:?} and shape {shape:?}",
                            info.dtype
                        )
                    })?,
                };
                tensors.insert(endpoint, value);
            }
        }
        Ok(tensors)
    }

    pub(crate) fn ensure_component_present(
        &self,
        component: &str,
        present: &BTreeSet<String>,
        role: &str,
    ) -> anyhow::Result<()> {
        if let Some(key) = self.plan.presence_condition(component)
            && !present.contains(key)
        {
            anyhow::bail!(
                "{role} '{component}' is gated by absent presence key '{key}' and cannot execute"
            );
        }
        Ok(())
    }

    pub(crate) fn missing_input_error(
        &self,
        component: &str,
        port: &str,
        present: &BTreeSet<String>,
    ) -> anyhow::Error {
        let endpoint = format!("{component}.{port}");
        let optional = self
            .models
            .directory
            .spec
            .models
            .get(component)
            .and_then(|model| model.io.as_ref())
            .and_then(|io| io.optional_inputs.get(port));
        match optional {
            Some(optional) if present.contains(&optional.presence) => anyhow::anyhow!(
                "missing optional-but-present pipeline input '{endpoint}' for presence key '{}'",
                optional.presence
            ),
            Some(optional) => anyhow::anyhow!(
                "missing or invalid fallback for absent optional pipeline input '{endpoint}' \
                 (presence key '{}')",
                optional.presence
            ),
            None => anyhow::anyhow!("missing required pipeline input '{endpoint}'"),
        }
    }

    pub(crate) fn run_prompt_phase_components(
        &self,
        components: &[String],
        tensors: &mut PipelineTensors,
        phase: &str,
        present: &BTreeSet<String>,
        mut timings: Option<&mut Vec<serde_json::Value>>,
    ) -> anyhow::Result<()> {
        for component in components {
            if !self.plan.component_is_present(component, present) {
                continue;
            }
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            let inputs = self.component_inputs(component, session, tensors, present)?;

            // A prompt-phase component is a pure function of its inputs — that
            // is what separates it from an `every_step` component — so identical
            // input bytes mean identical outputs. Re-asking about the same image
            // then costs a hash instead of a vision encoder forward pass.
            let memoizable = self.component_cache.borrow().is_enabled()
                && self.memoizable_components.contains(component);
            let key = memoizable.then(|| {
                digest_named_values(
                    component,
                    inputs.iter().map(|(name, value)| (name.as_str(), value)),
                )
            });
            let key = match key {
                Some(Some(key)) => Some(key),
                // Enabled but undigestible: run without touching the cache
                // rather than key on a partial description of the inputs.
                Some(None) => {
                    self.component_cache.borrow_mut().note_unkeyable();
                    None
                }
                None => None,
            };
            if let Some(key) = key
                && let Some(cached) = self.component_cache.borrow_mut().get(key)
            {
                if let Some(sink) = timings.as_deref_mut() {
                    sink.push(serde_json::json!({
                        "component": component,
                        "phase": phase,
                        "ms": 0.0,
                        "cached": true,
                    }));
                }
                for (name, value) in cached {
                    tensors.insert(format!("{component}.{name}"), value);
                }
                continue;
            }

            let refs = inputs
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect::<Vec<_>>();
            let started = std::time::Instant::now();
            let outputs = session
                .run(&refs)
                .map_err(|e| anyhow::anyhow!("ORT pipeline component '{component}' failed: {e}"))?;
            if let Some(sink) = timings.as_deref_mut() {
                sink.push(serde_json::json!({
                    "component": component,
                    "phase": phase,
                    "ms": started.elapsed().as_secs_f64() * 1e3,
                }));
            }
            let named = session
                .output_names()
                .iter()
                .cloned()
                .zip(outputs)
                .collect::<Vec<_>>();
            if let Some(key) = key {
                let mut cache = self.component_cache.borrow_mut();
                cache.note_miss();
                cache.insert(key, &named);
            }
            for (name, value) in named {
                tensors.insert(format!("{component}.{name}"), value);
            }
        }
        Ok(())
    }

    /// Digest everything about a request that changes what the decoder computes.
    ///
    /// This is the part of a multimodal prompt's identity that token ids cannot
    /// express: placeholder expansion turns any image into the same repeated
    /// token, so two different pictures produce byte-identical prompts. Without
    /// this digest in the key, retained KV for one photo would be served for
    /// another and the model would answer confidently about a picture it never
    /// saw.
    ///
    /// Covers the bound tensors, the presence keys, and the tile count.
    ///
    /// `None` when some input cannot be digested, which disables reuse for the
    /// request rather than keying it on an incomplete description.
    pub(crate) fn digest_request_identity(request: &PipelineGenerateRequest) -> Option<Digest> {
        let mut builder = DigestBuilder::new();

        // Presence keys gate which components run and which optional decoder
        // inputs are bound, so the same tensors under different presence keys
        // are a different computation and must not share KV.
        builder.absorb_u64(request.present.len() as u64);
        for key in &request.present {
            builder.absorb_str(key);
        }
        // Tile count drives placeholder expansion for encoder-free multimodal
        // pipelines, and so the meaning of the prompt's placeholder run.
        builder.absorb_u64(request.num_image_tiles.unwrap_or(0) as u64);

        let mut endpoints = request.inputs.keys().collect::<Vec<_>>();
        endpoints.sort();
        builder.absorb_u64(endpoints.len() as u64);
        for endpoint in endpoints {
            builder.absorb_str(endpoint);
            if !absorb_value(&mut builder, &request.inputs[endpoint]) {
                return None;
            }
        }
        Some(builder.finish())
    }

    pub(crate) fn component_inputs(
        &self,
        component: &str,
        session: &Session,
        tensors: &PipelineTensors,
        present: &BTreeSet<String>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut inputs = Vec::new();
        for info in session.inputs() {
            let endpoint = format!("{component}.{}", info.name);
            let routed = self
                .plan
                .dataflow()
                .iter()
                .find(|edge| {
                    edge.to == endpoint
                        && endpoint_component(&edge.from)
                            .is_none_or(|source| self.plan.component_is_present(source, present))
                })
                .and_then(|edge| tensors.get(&edge.from));
            let value = tensors
                .get(&endpoint)
                .or(routed)
                .ok_or_else(|| self.missing_input_error(component, &info.name, present))?;
            inputs.push((info.name.clone(), coerce_value_to_dtype(value, info.dtype)?));
        }
        Ok(inputs)
    }

    pub(crate) fn decoder_extra_inputs(
        &self,
        decoder: &str,
        tensors: &PipelineTensors,
        exclude_input: Option<&str>,
        present: &BTreeSet<String>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras = Vec::new();
        let mut bound = BTreeSet::new();
        for edge in self
            .plan
            .edges_to_component(decoder)
            .filter(|edge| endpoint_component(&edge.from).is_some_and(|from| from != decoder))
            .filter(|edge| {
                endpoint_component(&edge.from)
                    .is_none_or(|source| self.plan.component_is_present(source, present))
            })
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            // The per-step `inputs_embeds` edge is threaded dynamically by the
            // decode loop (re-embedding each step), not carried as a fixed extra.
            if exclude_input == Some(input) {
                continue;
            }
            let value = tensors
                .get(&edge.to)
                .or_else(|| tensors.get(&edge.from))
                .with_context(|| {
                    format!(
                        "missing pipeline tensor '{}' and routed source '{}'",
                        edge.to, edge.from
                    )
                })?;
            extras.push((input.to_string(), clone_value(value)?));
            bound.insert(input.to_string());
        }
        if let Some(optional_inputs) = self
            .models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|model| model.io.as_ref())
            .map(|io| &io.optional_inputs)
        {
            let session = self
                .models
                .graph_io(decoder)
                .with_context(|| format!("pipeline decoder '{decoder}' was not loaded"))?;
            for port in optional_inputs.keys() {
                if exclude_input == Some(port.as_str()) || bound.contains(port) {
                    continue;
                }
                let endpoint = format!("{decoder}.{port}");
                let value = tensors
                    .get(&endpoint)
                    .ok_or_else(|| self.missing_input_error(decoder, port, present))?;
                let dtype = session
                    .inputs()
                    .iter()
                    .find(|info| info.name == *port)
                    .with_context(|| {
                        format!("optional pipeline input '{endpoint}' is not exposed by its graph")
                    })?
                    .dtype;
                extras.push((port.clone(), coerce_value_to_dtype(value, dtype)?));
            }
        }
        Ok(extras)
    }

    /// Seed the prompt token ids into the shared pool for any prompt-phase
    /// component that consumes a token input (`input_ids`) which is neither
    /// supplied by the caller nor routed by a dataflow edge.
    ///
    /// The token port is taken only from explicit component `io.token_input`
    /// metadata. Components without that declaration are not implicitly seeded.
    pub(crate) fn seed_prompt_token_inputs(
        &self,
        components: &[String],
        prompt_tokens: &[TokenId],
        tensors: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        for component in components {
            let Some(token_input) = self
                .models
                .directory
                .spec
                .models
                .get(component)
                .and_then(|model| model.io.as_ref())
                .and_then(|io| io.token_input.as_deref())
            else {
                continue;
            };
            let endpoint = format!("{component}.{token_input}");
            let routed = self.plan.dataflow().iter().any(|edge| edge.to == endpoint);
            if routed || tensors.contains_key(&endpoint) {
                continue;
            }
            let ids: Vec<i64> = prompt_tokens.iter().map(|&t| i64::from(t)).collect();
            let value = Value::from_slice_i64(&ids, &[1, ids.len() as i64])?;
            tensors.insert(endpoint, value);
        }
        Ok(())
    }

    /// Resolve the STATIC encoder-produced cross-attention KV tensors that feed
    /// the autoregressive decoder on every step.
    ///
    /// For an encoder-decoder (e.g. Whisper) pipeline the encoder runs once as a
    /// prompt-phase prologue and publishes its `present_*_cross_%d` outputs into
    /// the shared pool as `{encoder}.present_*_cross_%d`. Those tensors encode
    /// the whole audio/text prompt and are STATIC for the entire decode: they
    /// never grow or change across autoregressive steps (unlike the decoder's
    /// self-attention KV cache). They are therefore cloned once here and re-bound
    /// verbatim to the decoder's `past_*_cross_%d` inputs on every step, rather
    /// than recomputed. The pairing comes from the decoder's declared
    /// `cross_kv_inputs`/`cross_kv_outputs` (resolved into `cross_kv_pairs`),
    /// keyed off the encoder-decoder pipeline shape, not any model name.
    pub(crate) fn static_cross_kv_bindings(
        &self,
        cross_kv_pairs: &[(String, String)],
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, Arc<Value>)>> {
        let mut bindings = Vec::with_capacity(cross_kv_pairs.len());
        for (decoder_input, encoder_output) in cross_kv_pairs {
            let suffix = format!(".{encoder_output}");
            let mut matches = tensors
                .iter()
                .filter(|(key, _)| key.ends_with(&suffix) || key.as_str() == encoder_output);
            let (_, value) = matches.next().with_context(|| {
                format!(
                    "encoder-decoder cross-attention: no pooled encoder output '{encoder_output}' \
                     to bind decoder input '{decoder_input}'; the encoder prologue must run and \
                     publish it before decode"
                )
            })?;
            if matches.next().is_some() {
                anyhow::bail!(
                    "encoder-decoder cross-attention: multiple pooled tensors match encoder output \
                     '{encoder_output}' for decoder input '{decoder_input}'; the producing \
                     component is ambiguous"
                );
            }
            // `Arc<Value>` mirrors the shared-ownership convention the ORT
            // decode paths already use for per-step-invariant tensors; `Value`
            // is neither `Send` nor `Sync`, so the lint is suppressed here as it
            // is in `onnx-genai-ort`.
            #[allow(clippy::arc_with_non_send_sync)]
            let shared = Arc::new(clone_value(value)?);
            bindings.push((decoder_input.clone(), shared));
        }
        Ok(bindings)
    }

    /// Precompute the static routing edges feeding the autoregressive `decoder`.
    ///
    /// Returns `(source_endpoint, decoder_input_port)` for every dataflow edge
    /// into the decoder whose source is a **different** component (self-edges are
    /// loop-carried KV / recurrent state, resolved inside the decode step). Both
    /// `every_step` producers and cached `prompt_only` conditioning route through
    /// this list; the values are re-read from the shared pool on every step, so
    /// per-step outputs stay fresh while fixed conditioning is simply reused.
    pub(crate) fn decoder_in_edges(
        &self,
        decoder: &str,
        present: &BTreeSet<String>,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, String)>> {
        let mut edges = Vec::new();
        let mut bound = BTreeSet::new();
        for edge in self
            .plan
            .edges_to_component(decoder)
            .filter(|edge| endpoint_component(&edge.from).is_some_and(|from| from != decoder))
            .filter(|edge| {
                endpoint_component(&edge.from)
                    .is_none_or(|source| self.plan.component_is_present(source, present))
            })
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            let source = if tensors.contains_key(&edge.to) {
                edge.to.clone()
            } else {
                edge.from.clone()
            };
            edges.push((source, input.to_string()));
            bound.insert(input.to_string());
        }
        if let Some(io) = self
            .models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|model| model.io.as_ref())
        {
            for port in io.optional_inputs.keys() {
                if bound.contains(port) {
                    continue;
                }
                let endpoint = format!("{decoder}.{port}");
                if tensors.contains_key(&endpoint) {
                    edges.push((endpoint, port.clone()));
                }
            }
        }
        Ok(edges)
    }

    /// Bind each declared `every_step` component to its generic input contract.
    ///
    /// The single running-token port comes from the component's explicit
    /// `io.token_input` metadata — never a tensor-name heuristic. Every other
    /// input is resolved from the shared pool on each step (directly by endpoint
    /// or through a dataflow edge), so cross-conditioning (e.g. image features)
    /// and chained per-step outputs both work without special-casing. This is
    /// the generic replacement for the former one-output `inputs_embeds` fusion
    /// binding: on prefill every component runs over the full prompt, on decode
    /// over the single running token, and all of its outputs are published back
    /// into the pool for routing into the decoder. Returns owned bindings so the
    /// caller can pair each with its loaded session without extending the borrow.
    pub(crate) fn build_step_bindings(
        &self,
        step_components: &[String],
        present: &BTreeSet<String>,
    ) -> anyhow::Result<Vec<StepComponentBinding>> {
        let mut bindings = Vec::with_capacity(step_components.len());
        for component in step_components {
            if !self.plan.component_is_present(component, present) {
                continue;
            }
            let session = self.models.graph_io(component).with_context(|| {
                format!("pipeline every_step component '{component}' was not loaded")
            })?;
            let token_input = self
                .models
                .directory
                .spec
                .models
                .get(component)
                .and_then(|spec| spec.io.as_ref())
                .and_then(|io| io.token_input.clone());
            if let Some(port) = &token_input
                && !session.inputs().iter().any(|info| &info.name == port)
            {
                anyhow::bail!(
                    "every_step component '{component}' declares io.token_input '{port}' but \
                     the graph does not expose it; graph inputs: {:?}",
                    session.input_names()
                );
            }
            let mut routed_inputs = Vec::new();
            for info in session.inputs() {
                if token_input.as_deref() == Some(info.name.as_str()) {
                    continue;
                }
                let endpoint = format!("{component}.{}", info.name);
                let routed_from = self
                    .plan
                    .dataflow()
                    .iter()
                    .find(|edge| {
                        edge.to == endpoint
                            && endpoint_component(&edge.from).is_none_or(|source| {
                                self.plan.component_is_present(source, present)
                            })
                    })
                    .map(|edge| edge.from.clone());
                routed_inputs.push(StepComponentInput {
                    port: info.name.clone(),
                    endpoint,
                    routed_from,
                    dtype: info.dtype,
                    missing_message: self
                        .missing_input_error(component, &info.name, present)
                        .to_string(),
                });
            }
            bindings.push(StepComponentBinding {
                component: component.clone(),
                token_input,
                routed_inputs,
            });
        }
        Ok(bindings)
    }
}
