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
            None => anyhow::anyhow!("{MISSING_REQUIRED_INPUT}'{endpoint}'"),
        }
    }

    /// True when every graph input of `component` can be resolved from the
    /// shared pool — directly by its `component.port` endpoint, or through an
    /// active dataflow edge whose source has published a value. A prompt-phase
    /// component whose inputs are unavailable is inactive for this request (for
    /// example a vision encoder on a text-only prompt) and is skipped rather
    /// than run, so a multimodal package decodes text without its image tower.
    pub(crate) fn prompt_component_inputs_available(
        &self,
        component: &str,
        tensors: &PipelineTensors,
        present: &BTreeSet<String>,
    ) -> bool {
        let Some(io) = self.models.graph_io(component) else {
            return false;
        };
        io.inputs().iter().all(|info| {
            let endpoint = format!("{component}.{}", info.name);
            if tensors.contains_key(&endpoint) {
                return true;
            }
            self.plan.dataflow().iter().any(|edge| {
                edge.to == endpoint
                    && endpoint_component(&edge.from)
                        .is_none_or(|source| self.plan.component_is_present(source, present))
                    && tensors.contains_key(&edge.from)
            })
        })
    }

    /// Run a prompt-phase `component` through a lazily-built native
    /// [`ComponentSession`], loading it on the pipeline's native device the first
    /// time it activates. Inputs are resolved from the shared pool exactly like
    /// the ORT prologue (by endpoint or active dataflow edge) and crossed into
    /// the backend-neutral `ComponentTensor` seam; outputs are published back
    /// into the pool as `component.output`. Native prompt components run without
    /// the ORT `component_cache` memoization.
    #[cfg(feature = "native-backend")]
    pub(crate) fn run_native_prompt_component(
        &self,
        component: &str,
        tensors: &mut PipelineTensors,
        present: &BTreeSet<String>,
        phase: &str,
        timings: Option<&mut Vec<serde_json::Value>>,
    ) -> anyhow::Result<()> {
        self.ensure_native_prompt_session(component)?;
        let mut sessions = self.native_prompt_sessions.borrow_mut();
        let session = sessions
            .get_mut(component)
            .expect("native prompt session was just ensured");
        let mut inputs: Vec<(String, onnx_genai_metadata::ComponentTensor)> = Vec::new();
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
            let coerced = coerce_value_to_dtype(value, DataType::from(info.dtype))?;
            inputs.push((info.name.clone(), value_to_component_tensor(&coerced)?));
        }
        let refs = inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let started = std::time::Instant::now();
        let outputs = session
            .run(&refs)
            .map_err(|e| anyhow::anyhow!("native pipeline component '{component}' failed: {e}"))?;
        if let Some(sink) = timings {
            sink.push(serde_json::json!({
                "component": component,
                "phase": phase,
                "ms": started.elapsed().as_secs_f64() * 1e3,
                "backend": "native",
            }));
        }
        for (name, tensor) in outputs {
            tensors.insert(
                format!("{component}.{name}"),
                component_tensor_to_value(&tensor)?,
            );
        }
        Ok(())
    }

    /// Load the native [`ComponentSession`] for a prompt-phase `component` on the
    /// pipeline's native device, unless it is already cached. Lazy so a
    /// multimodal package running a text-only prompt never materializes its
    /// vision tower's weights.
    #[cfg(feature = "native-backend")]
    fn ensure_native_prompt_session(&self, component: &str) -> anyhow::Result<()> {
        if self.native_prompt_sessions.borrow().contains_key(component) {
            return Ok(());
        }
        let path = self
            .models
            .directory
            .model_paths
            .get(component)
            .with_context(|| {
                format!("native pipeline prompt component '{component}' has no model path")
            })?;
        let device = super::native_decoder_device(self.native_device.as_ref());
        if matches!(device, crate::native_decode_device::NativeDecodeDevice::Cpu) {
            // The silent case is the expensive one: a vision tower on CPU still
            // produces correct embeddings, just two orders of magnitude slower,
            // so nothing fails and nothing is logged. Say it once, at load.
            tracing::warn!(
                component,
                model = %path.display(),
                "native pipeline prompt component is running on CPU; if this package has a CUDA \
                 execution provider available, prompt-phase encoders (a vision tower) will be far \
                 slower than the decoder that follows them"
            );
        }
        let session = crate::native_component::NativeComponentSession::load(
            path,
            device,
            Some(self.resource_governor.memory()),
        )
        .with_context(|| {
            format!("failed to load native pipeline prompt component '{component}'")
        })?;
        self.native_prompt_sessions
            .borrow_mut()
            .insert(component.to_string(), Box::new(session));
        Ok(())
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
            // On the native backend a prompt component's ORT session is
            // intentionally not built (its graph may use native-only operators).
            // Run it through a lazily-built native session instead. A component
            // whose inputs are unavailable for this request is inactive (e.g. a
            // vision encoder on a text-only prompt) and is skipped.
            if self.models.session(component).is_none() {
                if !self.prompt_component_inputs_available(component, tensors, present) {
                    continue;
                }
                #[cfg(feature = "native-backend")]
                {
                    self.run_native_prompt_component(
                        component,
                        tensors,
                        present,
                        phase,
                        timings.as_deref_mut(),
                    )?;
                    continue;
                }
                #[cfg(not(feature = "native-backend"))]
                {
                    anyhow::bail!("pipeline component '{component}' was not loaded");
                }
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

    /// Bind an empty tensor for any `every_step` component input whose declared
    /// producer did not run, when its declared shape has a dynamic axis (so an
    /// empty tensor is shape-valid).
    ///
    /// This is the multimodal image-features contract when no image is present:
    /// an embeds-driven decoder's embedder consumes both `input_ids` (seeded per
    /// step) and `image_features`, and for a text-only prompt the vision encoder
    /// never runs, so its `image_features` output is absent. The embedder still
    /// requires the graph input to be bound, so an empty `[0, hidden]` tensor is
    /// seeded once — exactly the empty image feed the `muse_decode` harness sends
    /// every step.
    ///
    /// Seeding is deliberately limited to inputs the pipeline **declares a
    /// producer for** via a `dataflow` edge. An input with no declared producer
    /// is a plain required graph input: substituting an empty tensor for it
    /// would silently run the model on nothing instead of reporting the missing
    /// binding, which is exactly what
    /// `undeclared_required_audio_input_never_receives_a_fallback` forbids. Such
    /// inputs are left untouched so `missing_input_error` names them at run.
    /// Inputs that are routed from an active producer, already present, or have
    /// a fully-static shape are also left untouched.
    pub(crate) fn seed_absent_step_component_inputs(
        &self,
        step_components: &[String],
        present: &BTreeSet<String>,
        tensors: &mut PipelineTensors,
    ) -> anyhow::Result<()> {
        for component in step_components {
            if !self.plan.component_is_present(component, present) {
                continue;
            }
            let Some(io) = self.models.graph_io(component) else {
                continue;
            };
            let token_input = self
                .models
                .directory
                .spec
                .models
                .get(component)
                .and_then(|model| model.io.as_ref())
                .and_then(|spec| spec.token_input.as_deref());
            let inputs: Vec<(String, DataType, Vec<i64>)> = io
                .inputs()
                .iter()
                .map(|info| (info.name.clone(), info.dtype, info.shape.clone()))
                .collect();
            for (name, dtype, shape) in inputs {
                if Some(name.as_str()) == token_input {
                    continue;
                }
                let endpoint = format!("{component}.{name}");
                if tensors.contains_key(&endpoint) {
                    continue;
                }
                // Only an input the pipeline declares a producer for may be
                // emptied when that producer is absent. Without this the empty
                // seed becomes a silent fallback for every dynamically-shaped
                // required input.
                let has_declared_producer =
                    self.plan.dataflow().iter().any(|edge| {
                        edge.to == endpoint && endpoint_component(&edge.from).is_some()
                    });
                if !has_declared_producer {
                    continue;
                }
                let routed = self.plan.dataflow().iter().any(|edge| {
                    edge.to == endpoint
                        && endpoint_component(&edge.from)
                            .is_some_and(|source| self.plan.component_is_present(source, present))
                        && tensors.contains_key(&edge.from)
                });
                if routed {
                    continue;
                }
                // Only emptyable when some axis is dynamic; a fully-static
                // required input is left to error with a precise message at run.
                if !shape.iter().any(|&dim| dim < 0) {
                    continue;
                }
                let empty_shape: Vec<i64> = shape
                    .iter()
                    .map(|&dim| if dim < 0 { 0 } else { dim })
                    .collect();
                let value = Value::from_raw_bytes(Vec::new(), &empty_shape, dtype)?;
                tensors.insert(endpoint, value);
            }
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

/// Prefix of the error a pipeline raises when a required input was never
/// produced.
///
/// A front end turns this into advice about the attachment the caller most
/// likely forgot, so the wording is shared rather than matched by hand: the
/// message and its recognizer cannot drift apart if they name the same
/// constant.
pub const MISSING_REQUIRED_INPUT: &str = "missing required pipeline input ";

/// True when `error`, or anything it wraps, is a required input that no
/// component produced.
pub fn is_missing_required_input(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().starts_with(MISSING_REQUIRED_INPUT))
}

#[cfg(test)]
mod missing_input_tests {
    use super::*;

    #[test]
    fn a_missing_required_input_is_recognized_through_the_context_wrapped_around_it() {
        let raw = anyhow::anyhow!("{MISSING_REQUIRED_INPUT}'encoder.pixel_values'");
        assert!(is_missing_required_input(&raw));
        assert!(is_missing_required_input(
            &raw.context("decoder forward failed")
        ));
    }

    #[test]
    fn an_unrelated_failure_is_not_mistaken_for_a_missing_input() {
        let oom = anyhow::anyhow!("cuda_ep: cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY")
            .context("decoder forward failed");
        assert!(!is_missing_required_input(&oom));

        // An optional input that was declared present is a package-authoring
        // fault, not a forgotten attachment, so it must not be advised as one.
        let optional = anyhow::anyhow!(
            "missing optional-but-present pipeline input 'encoder.pixel_values' for presence key 'has_image'"
        );
        assert!(!is_missing_required_input(&optional));
    }
}
