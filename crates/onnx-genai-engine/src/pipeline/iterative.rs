//! Iterative (diffusion) pipeline execution.
//!
//! Pure code motion from `pipeline.rs`: the iterative/diffusion denoiser
//! driver, its per-step denoiser pass and noise helpers, the composite and
//! single-pass tensor drivers, and the step/timing dump helpers.

use super::*;

/// Bundled inputs for [`PipelineEngine::run_denoiser_pass`]: the denoiser
/// session, the iterative plan and its start step, the constant conditioning
/// tensors and loop-carried values, the current step index and timestep, and
/// any per-port input overrides (input scaling or CFG unconditional
/// conditioning).
struct DenoiserPassContext<'a, 'o> {
    denoiser: &'a Session,
    plan: &'a IterativePlan,
    start_step: usize,
    constants: &'a PipelineTensors,
    carried: &'a HashMap<String, Value>,
    step: usize,
    timestep: f32,
    overrides: &'a [(&'o str, &'o Value)],
}

fn append_loop_conditioning(
    constants: &PipelineTensors,
    endpoint: &str,
    latent: &Value,
    is_loop: bool,
) -> anyhow::Result<Value> {
    let Some(conditioning) = is_loop
        .then(|| constants.get(&format!("{endpoint}.conditioning")))
        .flatten()
    else {
        return clone_value(latent);
    };
    let latent_shape = latent.shape();
    let conditioning_shape = conditioning.shape();
    if latent_shape.len() != 4
        || conditioning_shape.len() != 4
        || latent_shape[0] != conditioning_shape[0]
        || latent_shape[2..] != conditioning_shape[2..]
    {
        anyhow::bail!(
            "loop conditioning requires matching [batch, channels, height, width] tensors, got {latent_shape:?} and {conditioning_shape:?}"
        );
    }
    let batch = latent_shape[0] as usize;
    let latent_channels = latent_shape[1] as usize;
    let conditioning_channels = conditioning_shape[1] as usize;
    let plane = latent_shape[2] as usize * latent_shape[3] as usize;
    let latent = latent.to_vec_f32_lossy()?;
    let conditioning = conditioning.to_vec_f32_lossy()?;
    let mut combined = Vec::with_capacity(latent.len() + conditioning.len());
    for batch_index in 0..batch {
        let latent_start = batch_index * latent_channels * plane;
        let conditioning_start = batch_index * conditioning_channels * plane;
        combined.extend_from_slice(&latent[latent_start..latent_start + latent_channels * plane]);
        combined.extend_from_slice(
            &conditioning[conditioning_start..conditioning_start + conditioning_channels * plane],
        );
    }
    Value::from_slice_f32(
        &combined,
        &[
            latent_shape[0],
            latent_shape[1] + conditioning_shape[1],
            latent_shape[2],
            latent_shape[3],
        ],
    )
    .map_err(Into::into)
}

impl PipelineEngine {
    /// Run a bounded iterative (diffusion) denoise loop.
    ///
    /// Semantics: prompt-phase components run once; then the denoiser runs
    /// `num_steps` times, threading loop-carried state (its self-edges) from one
    /// step's output into the next step's input while constant conditioning
    /// (e.g. encoder hidden states) is re-supplied each step; then final-phase
    /// components run once. `guidance_scale` is carried but not yet applied —
    /// classifier-free guidance and timestep/sigma schedules are supplied by the
    /// scheduler-registry follow-up.
    pub(crate) fn run_iterative(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        self.run_iterative_with_callback(request, None)
    }

    pub(crate) fn run_iterative_with_callback(
        &self,
        request: PipelineGenerateRequest,
        mut callback: Option<&mut ImageStepCallback<'_>>,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Iterative(plan) = &self.plan else {
            anyhow::bail!("internal error: run_iterative on a non-iterative plan");
        };

        // Live overrides (ComfyUI-style): re-drive the already-loaded models with
        // different loop parameters, no reload. Seed / prompt / negative are
        // already live via per-request inputs, so only loop params are overridden.
        let present = request.present.clone();
        let overrides = &request.iterative_overrides;
        let num_steps = overrides.num_steps.unwrap_or(plan.num_steps);
        let start_step = overrides.start_step.unwrap_or(plan.start_step);
        if num_steps == 0 {
            anyhow::bail!("iterative override num_steps must be >= 1");
        }
        if start_step > num_steps {
            anyhow::bail!(
                "iterative override start_step ({start_step}) must be <= num_steps ({num_steps})"
            );
        }
        // Rebuild the scheduler when the step count changes (its schedule may be
        // baked at build time). An explicit per-step timestep schedule is tied to
        // the original step count, so reject a step-count override in that case.
        let rebuilt_scheduler = if num_steps != plan.num_steps {
            if plan.timesteps.is_some() {
                anyhow::bail!(
                    "cannot override num_steps for a pipeline with an explicit timestep schedule"
                );
            }
            match &plan.scheduler_spec {
                Some(spec) => Some(plan.scheduler_registry.build(spec, num_steps)?),
                None => None,
            }
        } else {
            None
        };
        let scheduler = rebuilt_scheduler.as_ref().or(plan.scheduler.as_ref());

        // Classifier-free guidance scale (active only when set and != 1.0).
        let guidance = overrides
            .guidance_scale
            .or(plan.guidance_scale)
            .filter(|s| *s != 1.0);
        // `constants` holds external inputs + prompt-phase outputs and is NOT
        // mutated by the loop, so a denoiser whose output port shares a name
        // with a conditioning input cannot clobber that conditioning. Denoiser
        // outputs live in a separate `loop_state`, keyed by output port.
        let mut constants = self.prepare_request_tensors(request.inputs, &present)?;
        let mut stage_timings: Vec<serde_json::Value> = Vec::new();
        {
            let _span = onnx_genai_ort::prof_span!("diffusion.text_encode");
            self.run_prompt_phase_components(
                &plan.prompt_components,
                &mut constants,
                "encode",
                &present,
                Some(&mut stage_timings),
            )?;
        }
        if !self.plan.component_is_present(&plan.denoiser, &present) {
            {
                let _span = onnx_genai_ort::prof_span!("diffusion.vae_decode");
                self.run_prompt_phase_components(
                    &plan.final_components,
                    &mut constants,
                    "decode",
                    &present,
                    Some(&mut stage_timings),
                )?;
            }
            dump_stage_timings(&stage_timings);
            return Ok(constants);
        }

        let denoiser = self
            .models
            .session(&plan.denoiser)
            .with_context(|| format!("pipeline denoiser '{}' was not loaded", plan.denoiser))?;

        // Precompute the CFG unconditional conditioning once. Any denoiser input
        // port with a supplied `{denoiser}.{port}.uncond` embedding is overridden
        // on the unconditional pass — this supports multi-conditioning models
        // (e.g. SDXL overrides both `encoder_hidden_states` and pooled
        // `text_embeds`, while sharing `time_ids`). The primary
        // `cfg_conditioning_input` is additionally zeroed when no `.uncond` is
        // supplied (the zeros fallback for a single-conditioning SD model).
        let cfg_uncond: Vec<(String, Value)> = if guidance.is_some() {
            if let Some(primary) = plan.cfg_conditioning_input.clone() {
                let mut overrides: Vec<(String, Value)> = Vec::new();
                let mut seen: BTreeSet<String> = BTreeSet::new();
                for info in denoiser.inputs() {
                    let port = info.name.as_str();
                    let uncond_endpoint = format!("{}.{}.uncond", plan.denoiser, port);
                    if let Some(u) = constants.get(&uncond_endpoint) {
                        overrides.push((port.to_string(), clone_value(u)?));
                        seen.insert(port.to_string());
                    }
                }
                if !seen.contains(&primary) {
                    let cond_endpoint = format!("{}.{}", plan.denoiser, primary);
                    let cond = constants
                        .get(&cond_endpoint)
                        .or_else(|| {
                            plan.dataflow
                                .iter()
                                .find(|e| e.to == cond_endpoint)
                                .and_then(|e| constants.get(&e.from))
                        })
                        .with_context(|| format!("cfg conditioning '{cond_endpoint}' not found"))?;
                    overrides.push((
                        primary.clone(),
                        Value::from_slice_f32(&vec![0.0f32; cond.numel()], cond.shape())?,
                    ));
                }
                overrides
            } else {
                // No static conditioning input: the unconditional pass is a
                // transform of the loop-carried sample (discrete language
                // diffusion re-masks the prompt via `cfg_uncond_sample`).
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // `carried` holds the value to feed each loop-carried INPUT port next
        // step (keyed by input port); `last_outputs` holds the denoiser's raw
        // outputs from the final step (keyed by output port). Keeping them
        // separate from the immutable `constants` pool prevents an output whose
        // name collides with a conditioning input from clobbering it.
        let mut carried: HashMap<String, Value> = HashMap::new();
        if start_step == num_steps {
            for (_, in_port) in &plan.loop_edges {
                let endpoint = format!("{}.{}", plan.denoiser, in_port);
                let seed = constants.get(&endpoint).with_context(|| {
                    format!("missing iterative pipeline seed '{endpoint}' at start step")
                })?;
                carried.insert(in_port.clone(), clone_value(seed)?);
            }
        }
        let mut last_outputs: HashMap<String, Value> = HashMap::new();
        // Reset any multistep scheduler state before the loop (img2img reuses a
        // plan whose scheduler may hold state from a previous run).
        if let Some(scheduler) = scheduler {
            scheduler.reset();
        }
        // Denoiser timestep schedule: prefer the plan's explicit `strategy.timesteps`,
        // otherwise fall back to the scheduler's own timesteps (so from-scratch
        // packages that omit the table still drive the denoiser with the correct
        // diffusion timesteps rather than the raw step index).
        let scheduler_timesteps: Option<Vec<f32>> = if plan.timesteps.is_some() {
            None
        } else {
            scheduler.and_then(|scheduler| scheduler.timesteps())
        };
        // Partial (img2img) loops start at `start_step`; the seed is then the
        // encoded image already noised to `timesteps[start_step]`.
        let denoise_start = std::time::Instant::now();
        {
            let _denoise_loop_span = onnx_genai_ort::prof_span!("diffusion.denoise_loop");
            for step in start_step..num_steps {
                let _step_span =
                    onnx_genai_ort::prof_span!("diffusion.denoise_step", "step" => step);
                let step_start = std::time::Instant::now();
                let is_first = step == start_step;
                // Timestep/sigma for this step: explicit plan schedule when provided,
                // else the scheduler's timesteps, else the 0-based step index.
                let timestep = plan
                    .timesteps
                    .as_ref()
                    .or(scheduler_timesteps.as_ref())
                    .and_then(|ts| ts.get(step).copied())
                    .unwrap_or(step as f32);

                // Raw (unscaled) loop-carried sample feeding each loop input this
                // step: the seed on the first step, otherwise the value carried from
                // the previous step. The scheduler's `step` consumes these raw samples.
                let mut raw_samples: HashMap<String, Value> = HashMap::new();
                for (_, in_port) in &plan.loop_edges {
                    let raw = if is_first {
                        let endpoint = format!("{}.{}", plan.denoiser, in_port);
                        constants.get(&endpoint).with_context(|| {
                            format!("missing iterative pipeline seed '{endpoint}' at start step")
                        })?
                    } else {
                        carried.get(in_port).with_context(|| {
                            format!(
                                "loop-carried input '{}.{in_port}' was not produced",
                                plan.denoiser
                            )
                        })?
                    };
                    raw_samples.insert(in_port.clone(), clone_value(raw)?);
                }

                // Some schedulers (e.g. Euler) scale the loop-carried sample before
                // it reaches the denoiser. Compute those scaled values once and feed
                // them as per-port overrides; schedulers that don't scale (DDIM,
                // masked diffusion) leave the raw sample untouched.
                let mut scaled_inputs: HashMap<String, Value> = HashMap::new();
                if let Some(scheduler) = scheduler {
                    for (_, in_port) in &plan.loop_edges {
                        let raw = &raw_samples[in_port];
                        if let Some(scaled) = scheduler.scale_input(step, num_steps, raw)? {
                            scaled_inputs.insert(in_port.clone(), scaled);
                        }
                    }
                }
                let scale_overrides: Vec<(&str, &Value)> = scaled_inputs
                    .iter()
                    .map(|(port, value)| (port.as_str(), value))
                    .collect();

                // Conditional pass (all inputs as declared, plus any input scaling).
                let cond_out = self.run_denoiser_pass(DenoiserPassContext {
                    denoiser,
                    plan,
                    start_step,
                    constants: &constants,
                    carried: &carried,
                    step,
                    timestep,
                    overrides: &scale_overrides,
                })?;

                // Classifier-free guidance: run an unconditional pass with the
                // conditioning replaced by the unconditional embedding, then combine
                // per output port:  pred = uncond + scale * (cond - uncond).
                let out_map = if let Some(scale) = guidance {
                    let mut cfg_overrides = scale_overrides.clone();
                    for (port, value) in &cfg_uncond {
                        cfg_overrides.retain(|(p, _)| *p != port.as_str());
                        cfg_overrides.push((port.as_str(), value));
                    }
                    // Language-diffusion CFG: the unconditional pass feeds the
                    // loop-carried input with its prompt tokens re-masked. Computed
                    // per step from the current sample (owned here so its references
                    // live through the unconditional denoiser pass).
                    let mut prompt_masked_inputs: Vec<(String, Value)> = Vec::new();
                    if let Some(scheduler) = scheduler {
                        for (_, in_port) in &plan.loop_edges {
                            let raw = &raw_samples[in_port];
                            if let Some(uncond_sample) = scheduler.cfg_uncond_sample(raw)? {
                                prompt_masked_inputs.push((in_port.clone(), uncond_sample));
                            }
                        }
                    }
                    for (port, value) in &prompt_masked_inputs {
                        cfg_overrides.retain(|(p, _)| *p != port.as_str());
                        cfg_overrides.push((port.as_str(), value));
                    }
                    let uncond_out = self.run_denoiser_pass(DenoiserPassContext {
                        denoiser,
                        plan,
                        start_step,
                        constants: &constants,
                        carried: &carried,
                        step,
                        timestep,
                        overrides: &cfg_overrides,
                    })?;
                    let mut combined: HashMap<String, Value> = HashMap::new();
                    for (port, cond_value) in &cond_out {
                        let uncond_value = uncond_out.get(port).with_context(|| {
                            format!(
                                "unconditional pass did not produce '{}.{port}'",
                                plan.denoiser
                            )
                        })?;
                        let cond_v = cond_value.to_vec_f32_lossy()?;
                        let uncond_v = uncond_value.to_vec_f32_lossy()?;
                        let guided: Vec<f32> = uncond_v
                            .iter()
                            .zip(&cond_v)
                            .map(|(u, c)| u + scale * (c - u))
                            .collect();
                        combined.insert(
                            port.clone(),
                            Value::from_slice_f32(&guided, cond_value.shape())?,
                        );
                    }
                    combined
                } else {
                    cond_out
                };

                // Compute the next value for each loop-carried input. Without a
                // scheduler this is identity feedback (output -> input). With a
                // scheduler the output is a noise prediction and the next sample is
                // `scheduler.step(raw_sample, prediction)` (raw = unscaled).
                for (out_port, in_port) in &plan.loop_edges {
                    let model_output = out_map.get(out_port).with_context(|| {
                        format!(
                            "denoiser did not produce loop output '{}.{out_port}'",
                            plan.denoiser
                        )
                    })?;
                    let next = if let Some(scheduler) = scheduler {
                        let _scheduler_span =
                            onnx_genai_ort::prof_span!("diffusion.scheduler_step", "step" => step);
                        let sample = raw_samples.get(in_port).with_context(|| {
                            format!(
                                "missing loop-carried sample for '{}.{in_port}'",
                                plan.denoiser
                            )
                        })?;
                        if scheduler.needs_noise() {
                            let noise = self
                                .step_noise(plan, num_steps, &constants, in_port, step, sample)?;
                            scheduler.step_with_noise(
                                step,
                                num_steps,
                                sample,
                                model_output,
                                Some(&noise),
                            )?
                        } else {
                            scheduler.step(step, num_steps, sample, model_output)?
                        }
                    } else {
                        clone_value(model_output)?
                    };
                    dump_iterative_step(
                        &plan.denoiser,
                        in_port,
                        step,
                        &next,
                        step_start.elapsed().as_secs_f64() * 1e3,
                    );
                    carried.insert(in_port.clone(), next);
                }
                if let Some(callback) = callback.as_deref_mut() {
                    let latents = plan
                        .loop_edges
                        .iter()
                        .map(|(_, input)| {
                            let endpoint = format!("{}.{}", plan.denoiser, input);
                            let value = carried.get(input).with_context(|| {
                                format!("iterative latent '{endpoint}' was not produced")
                            })?;
                            Ok((endpoint, clone_value(value)?))
                        })
                        .collect::<anyhow::Result<_>>()?;
                    callback(&ImageStep { step, latents })?;
                }
                last_outputs = out_map;
            }
        }
        let denoise_ms = denoise_start.elapsed().as_secs_f64() * 1e3;
        stage_timings.push(serde_json::json!({
            "component": plan.denoiser,
            "phase": "denoise",
            "ms": denoise_ms,
            "steps": num_steps - start_step,
        }));

        // Publish the final denoiser outputs (raw predictions) and the final
        // loop-carried samples, then run final-phase components once. A VAE can
        // route from either the output port or the (post-scheduler) sample port.
        let mut tensors = constants;
        for (out_port, value) in last_outputs {
            tensors.insert(format!("{}.{}", plan.denoiser, out_port), value);
        }
        for (in_port, value) in carried {
            tensors.insert(format!("{}.{}", plan.denoiser, in_port), value);
        }
        {
            let _span = onnx_genai_ort::prof_span!("diffusion.vae_decode");
            self.run_prompt_phase_components(
                &plan.final_components,
                &mut tensors,
                "decode",
                &present,
                Some(&mut stage_timings),
            )?;
        }
        dump_stage_timings(&stage_timings);
        Ok(tensors)
    }

    /// Run one denoiser invocation for `step`. Returns `(outputs, sample_in)`
    /// keyed by port. `override_input`, when set as `(port, value)`, substitutes
    /// that input's value — used to supply the unconditional conditioning on the
    /// CFG unconditional pass.
    fn run_denoiser_pass(
        &self,
        context: DenoiserPassContext<'_, '_>,
    ) -> anyhow::Result<HashMap<String, Value>> {
        let DenoiserPassContext {
            denoiser,
            plan,
            start_step,
            constants,
            carried,
            step,
            timestep,
            overrides,
        } = context;
        let mut inputs: Vec<(String, Value)> = Vec::new();
        for info in denoiser.inputs() {
            let port = info.name.as_str();
            let endpoint = format!("{}.{}", plan.denoiser, port);
            let is_loop = plan.loop_edges.iter().any(|(_, in_port)| in_port == port);
            // An override wins for its port. Two producers use overrides: the
            // scheduler's per-step input scaling (Euler) and CFG's unconditional
            // conditioning embedding.
            if let Some((_, over_value)) = overrides.iter().find(|(p, _)| *p == port) {
                let value = append_loop_conditioning(constants, &endpoint, over_value, is_loop)?;
                inputs.push((port.to_string(), coerce_value_to_dtype(&value, info.dtype)?));
                continue;
            }
            // Per-step timestep injection takes precedence for its port. Honor
            // the port dtype: real diffusion denoisers (DiT/UNet) declare an
            // INT64 timestep, while others take a float sigma.
            if plan.timestep_input.as_deref() == Some(port) {
                let ts = match info.dtype {
                    DataType::Int64 => Value::from_vec_i64(vec![timestep as i64], &[1])?,
                    _ => Value::from_slice_f32(&[timestep], &[1])?,
                };
                inputs.push((port.to_string(), ts));
                continue;
            }
            let value = if is_loop {
                if step == start_step {
                    constants.get(&endpoint).with_context(|| {
                        format!("missing iterative pipeline seed '{endpoint}' at start step")
                    })?
                } else {
                    carried.get(port).with_context(|| {
                        format!("loop-carried input '{endpoint}' was not produced")
                    })?
                }
            } else {
                let routed = plan
                    .dataflow
                    .iter()
                    .find(|edge| edge.to == endpoint)
                    .and_then(|edge| constants.get(&edge.from));
                constants
                    .get(&endpoint)
                    .or(routed)
                    .with_context(|| format!("missing pipeline input '{endpoint}'"))?
            };
            let value = append_loop_conditioning(constants, &endpoint, value, is_loop)?;
            inputs.push((port.to_string(), coerce_value_to_dtype(&value, info.dtype)?));
        }

        let refs = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let _span = onnx_genai_ort::prof_span!("diffusion.denoiser_pass", "step" => step);
        let outputs = denoiser.run(&refs).map_err(|e| {
            anyhow::anyhow!(
                "ORT denoiser '{}' failed at step {step}: {e}",
                plan.denoiser
            )
        })?;
        let mut out_map: HashMap<String, Value> = HashMap::new();
        for (name, value) in denoiser.output_names().iter().zip(outputs) {
            out_map.insert(name.clone(), value);
        }
        Ok(out_map)
    }

    /// Fetch the per-step Gaussian noise an ancestral scheduler needs at `step`.
    ///
    /// The caller supplies an external tensor `{denoiser}.{in_port}.noise` shaped
    /// `[num_steps, *sample_shape]` (so the noise sequence is reproducible and can
    /// match a reference generator); this slices out the `step`-th sample.
    fn step_noise(
        &self,
        plan: &IterativePlan,
        num_steps: usize,
        constants: &PipelineTensors,
        in_port: &str,
        step: usize,
        sample: &Value,
    ) -> anyhow::Result<Value> {
        let endpoint = format!("{}.{}.noise", plan.denoiser, in_port);
        let all = constants.get(&endpoint).with_context(|| {
            format!(
                "ancestral scheduler requires per-step noise tensor '{endpoint}' \
                 shaped [num_steps, ...]"
            )
        })?;
        let elem: usize = sample.shape().iter().map(|&d| d as usize).product();
        let data = all.to_vec_f32_lossy()?;
        let want = num_steps * elem;
        if data.len() != want {
            anyhow::bail!(
                "noise tensor '{endpoint}' has {} elements but expected {want} \
                 ({num_steps} steps x {elem})",
                data.len(),
            );
        }
        let slice = &data[step * elem..(step + 1) * elem];
        Value::from_slice_f32(slice, sample.shape()).map_err(Into::into)
    }

    /// Run a single-pass pipeline: prompt-phase components once, then one
    /// forward invocation of the strategy `model`.
    /// Execute a multi-stage composite pipeline (DESIGN.md §20): run each stage
    /// once, in declared order, over a shared tensor pool. A stage's model reads
    /// its inputs from the pool (routed by the pipeline dataflow) and writes its
    /// outputs back, so an earlier stage's outputs feed later stages.
    pub(crate) fn run_composite(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::Composite(plan) = &self.plan else {
            anyhow::bail!("internal error: run_composite on a non-composite plan");
        };
        if let Some(workflow) = &plan.workflow {
            let PipelineGenerateRequest {
                request,
                inputs,
                present: _,
                num_image_tiles: _,
                iterative_overrides: _,
                session_id,
            } = request;
            let mut values = self.bind_workflow_inputs(workflow, &request, inputs)?;
            for (cell, state) in &workflow.state {
                if state.scope != onnx_genai_metadata::WorkflowStateScope::Session {
                    continue;
                }
                let session_id = session_id.as_ref().with_context(|| {
                    format!("session-scoped workflow state '{cell}' requires a session id")
                })?;
                if let Some(value) = self
                    .workflow_session_state
                    .borrow()
                    .get(&(session_id.clone(), cell.clone()))
                {
                    values.insert(state.initializer.clone(), clone_value(value)?);
                }
            }
            let mut symbols = HashMap::new();
            let dynamic_symbols = workflow
                .state
                .values()
                .filter_map(|state| match &state.recurrence {
                    onnx_genai_metadata::ShapeRecurrence::Growing { axis, .. } => state
                        .contract
                        .shape
                        .as_ref()
                        .and_then(|shape| shape.get(*axis))
                        .and_then(|dimension| match dimension {
                            TensorDimension::Symbol(symbol) => Some(symbol.clone()),
                            TensorDimension::Fixed(_) => None,
                        }),
                    onnx_genai_metadata::ShapeRecurrence::Invariant => None,
                })
                .collect::<std::collections::HashSet<_>>();
            for (name, input) in &workflow.inputs {
                if let Some(value) = values.get(name) {
                    validate_workflow_value(
                        name,
                        value,
                        &input.contract,
                        &mut symbols,
                        &dynamic_symbols,
                    )?;
                }
            }
            let mut emit_counts = HashMap::new();
            let mut final_state_refs = HashMap::new();
            self.run_workflow_node(
                &workflow.graph,
                workflow,
                &mut values,
                &mut symbols,
                &dynamic_symbols,
                &mut emit_counts,
                &mut final_state_refs,
            )?;
            if let Some(session_id) = session_id {
                let mut updates = Vec::new();
                for (cell, state) in &workflow.state {
                    if state.scope != onnx_genai_metadata::WorkflowStateScope::Session {
                        continue;
                    }
                    let value_ref = final_state_refs
                        .get(cell)
                        .map(String::as_str)
                        .unwrap_or(&state.initializer);
                    let value = values.get(value_ref).with_context(|| {
                        format!(
                            "session-scoped workflow state '{cell}' has no final value '{value_ref}'"
                        )
                    })?;
                    updates.push(((session_id.clone(), cell.clone()), clone_value(value)?));
                }
                let mut session_state = self.workflow_session_state.borrow_mut();
                for (key, value) in updates {
                    session_state.insert(key, value);
                }
            }
            return Ok(values);
        }
        let present = request.present;
        let mut tensors = self.prepare_request_tensors(request.inputs, &present)?;
        for stage in &plan.stages {
            match &stage.kind {
                CompositeStageKind::SinglePass { model } => {
                    self.run_prompt_phase_components(
                        std::slice::from_ref(model),
                        &mut tensors,
                        &stage.name,
                        &present,
                        None,
                    )?;
                }
            }
        }
        for edge in &plan.dataflow {
            if !edge.to.contains('.')
                && let Some(value) = tensors.remove(&edge.from)
            {
                tensors.insert(edge.to.clone(), value);
            }
        }
        Ok(tensors)
    }

    fn run_workflow_node(
        &self,
        node: &WorkflowNode,
        workflow: &WorkflowSpec,
        values: &mut PipelineTensors,
        symbols: &mut HashMap<String, i64>,
        dynamic_symbols: &std::collections::HashSet<String>,
        emit_counts: &mut HashMap<String, usize>,
        final_state_refs: &mut HashMap<String, String>,
    ) -> anyhow::Result<()> {
        match node {
            WorkflowNode::Sequence { nodes } => {
                for node in nodes {
                    self.run_workflow_node(
                        node,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                    )?;
                }
            }
            WorkflowNode::Invoke {
                component,
                inputs,
                outputs,
                ..
            } => {
                let declaration = workflow
                    .components
                    .get(component)
                    .with_context(|| format!("workflow component '{component}' is undeclared"))?;
                match &declaration.implementation {
                    ComponentImplementation::Onnx { .. } => {
                        let session = self.models.session(component).with_context(|| {
                            format!("workflow ONNX component '{component}' was not loaded")
                        })?;
                        let resolved = inputs
                            .iter()
                            .map(|(port, value)| {
                                values
                                    .get(value)
                                    .with_context(|| {
                                        format!(
                                            "workflow component '{component}' input '{port}' \
                                             references unavailable value '{value}'"
                                        )
                                    })
                                    .and_then(|tensor| {
                                        let contract =
                                            declaration.ports.inputs.get(port).with_context(
                                                || {
                                                    format!(
                                                        "workflow component '{component}' has no \
                                                     declared input port '{port}'"
                                                    )
                                                },
                                            )?;
                                        validate_workflow_value(
                                            value,
                                            tensor,
                                            contract,
                                            symbols,
                                            dynamic_symbols,
                                        )?;
                                        Ok((port.as_str(), tensor))
                                    })
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        let produced = session.run(&resolved)?;
                        for (port, tensor) in session.output_names().iter().zip(produced) {
                            let value = outputs.get(port).with_context(|| {
                                format!(
                                    "workflow component '{component}' output '{port}' has no SSA binding"
                                )
                            })?;
                            let contract =
                                declaration.ports.outputs.get(port).with_context(|| {
                                    format!(
                                        "workflow component '{component}' has no declared output \
                                         port '{port}'"
                                    )
                                })?;
                            validate_workflow_value(
                                value,
                                &tensor,
                                contract,
                                symbols,
                                dynamic_symbols,
                            )?;
                            values.insert(value.clone(), tensor);
                        }
                    }
                    ComponentImplementation::Binding => {
                        for (port, output) in outputs {
                            let source = inputs.get(port).with_context(|| {
                                format!(
                                    "binding component '{component}' output '{port}' requires \
                                     an input with the same port name"
                                )
                            })?;
                            let tensor = values.get(source).with_context(|| {
                                format!("binding source value '{source}' is unavailable")
                            })?;
                            values.insert(output.clone(), clone_value(tensor)?);
                        }
                    }
                    ComponentImplementation::Adapter { abi, version, .. } => {
                        anyhow::bail!(
                            "workflow adapter '{component}' requires unsupported ABI {abi}@{version}"
                        );
                    }
                }
            }
            WorkflowNode::Loop {
                setup,
                body,
                condition,
                max_iterations,
                carried,
            } => {
                self.run_workflow_node(
                    setup,
                    workflow,
                    values,
                    symbols,
                    dynamic_symbols,
                    emit_counts,
                    final_state_refs,
                )?;
                for carry in carried {
                    final_state_refs.insert(carry.cell.clone(), carry.current.clone());
                }
                let limit = workflow_scalar_usize(values, max_iterations)?;
                for _ in 0..limit {
                    for carry in carried {
                        let current = values.get(&carry.current).with_context(|| {
                            format!("workflow loop value '{}' is unavailable", carry.current)
                        })?;
                        values.insert(carry.body_input.clone(), clone_value(current)?);
                    }
                    self.run_workflow_node(
                        body,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                    )?;
                    for carry in carried {
                        let current = values.get(&carry.current).with_context(|| {
                            format!("workflow loop value '{}' is unavailable", carry.current)
                        })?;
                        let next = values.get(&carry.body_output).with_context(|| {
                            format!("workflow loop body did not produce '{}'", carry.body_output)
                        })?;
                        let state = workflow.state.get(&carry.cell).with_context(|| {
                            format!("workflow loop carries undeclared state '{}'", carry.cell)
                        })?;
                        validate_state_recurrence(&carry.cell, current, next, state, values)?;
                        let next_value = clone_value(next)?;
                        values.insert(carry.current.clone(), clone_value(&next_value)?);
                        values.insert(carry.next.clone(), next_value);
                        final_state_refs.insert(carry.cell.clone(), carry.current.clone());
                    }
                    if !workflow_scalar_bool(values, condition)? {
                        break;
                    }
                }
            }
            WorkflowNode::Branch {
                predicate,
                cases,
                default,
            } => {
                let key = workflow_scalar_key(values, predicate)?;
                if let Some(case) = cases.get(&key) {
                    self.run_workflow_node(
                        case,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                    )?;
                } else if let Some(default) = default {
                    self.run_workflow_node(
                        default,
                        workflow,
                        values,
                        symbols,
                        dynamic_symbols,
                        emit_counts,
                        final_state_refs,
                    )?;
                } else {
                    anyhow::bail!("workflow branch has no case '{key}' and no default");
                }
            }
            WorkflowNode::Emit {
                value,
                output,
                mode,
                ..
            } => {
                let tensor = values
                    .get(value)
                    .with_context(|| format!("workflow emit value '{value}' is unavailable"))?;
                let output_contract = workflow.outputs.get(output).with_context(|| {
                    format!("workflow emit references undeclared output '{output}'")
                })?;
                validate_workflow_value(
                    value,
                    tensor,
                    &output_contract.contract,
                    symbols,
                    dynamic_symbols,
                )?;
                let emitted = clone_value(tensor)?;
                match mode {
                    WorkflowEmitMode::Replace => {
                        values.insert(output.clone(), emitted);
                    }
                    WorkflowEmitMode::Append => {
                        let appended = if let Some(previous) = values.get(output) {
                            append_workflow_value(previous, &emitted)?
                        } else {
                            emitted
                        };
                        values.insert(output.clone(), appended);
                    }
                    WorkflowEmitMode::Event => {
                        let index = emit_counts.entry(output.clone()).or_default();
                        values.insert(format!("{output}.{index}"), clone_value(&emitted)?);
                        *index += 1;
                        values.insert(output.clone(), emitted);
                    }
                }
            }
            WorkflowNode::Transfer {
                input,
                output,
                device,
            } => {
                if *device != DeviceKind::Cpu {
                    anyhow::bail!(
                        "workflow transfer to {device:?} requires a device allocator contract"
                    );
                }
                let tensor = values
                    .get(input)
                    .with_context(|| format!("workflow transfer value '{input}' is unavailable"))?;
                values.insert(output.clone(), clone_value(tensor)?);
            }
        }
        Ok(())
    }

    fn bind_workflow_inputs(
        &self,
        workflow: &WorkflowSpec,
        request: &GenerateRequest,
        mut provided: PipelineTensors,
    ) -> anyhow::Result<PipelineTensors> {
        let mut values = HashMap::new();
        for (name, input) in &workflow.inputs {
            let supplied = provided.remove(name).or_else(|| match &input.source {
                WorkflowInputSource::Application { name } => provided.remove(name),
                _ => None,
            });
            let value = if let Some(value) = supplied {
                Some(value)
            } else {
                match &input.source {
                    WorkflowInputSource::Request { field } => {
                        workflow_request_value(field, request, &input.contract)?
                    }
                    WorkflowInputSource::Literal => input
                        .default
                        .as_ref()
                        .map(|value| workflow_literal_value(value, &input.contract))
                        .transpose()?,
                    WorkflowInputSource::Application { .. } => input
                        .default
                        .as_ref()
                        .map(|value| workflow_literal_value(value, &input.contract))
                        .transpose()?,
                    WorkflowInputSource::Artifact { path } => {
                        anyhow::bail!(
                            "workflow input '{name}' requires artifact binding '{path}', which \
                             is not a tensor request input"
                        )
                    }
                }
            };
            match value {
                Some(value) => {
                    values.insert(name.clone(), value);
                }
                None if input.required => {
                    anyhow::bail!("required workflow package input '{name}' was not supplied")
                }
                None => {}
            }
        }
        if !provided.is_empty() {
            anyhow::bail!(
                "workflow request supplied undeclared application inputs: {:?}",
                provided.keys().collect::<Vec<_>>()
            );
        }
        Ok(values)
    }

    pub(crate) fn run_single_pass(
        &self,
        request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineTensors> {
        let PipelinePlan::SinglePass(plan) = &self.plan else {
            anyhow::bail!("internal error: run_single_pass on a non-single-pass plan");
        };
        let present = request.present;
        let mut tensors = self.prepare_request_tensors(request.inputs, &present)?;
        self.run_prompt_phase_components(
            &plan.prompt_components,
            &mut tensors,
            "prologue",
            &present,
            None,
        )?;

        if !self.plan.component_is_present(&plan.model, &present) {
            return Ok(tensors);
        }

        let model = self
            .models
            .session(&plan.model)
            .with_context(|| format!("pipeline model '{}' was not loaded", plan.model))?;
        let inputs = self.component_inputs(&plan.model, model, &tensors, &present)?;
        let refs = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let outputs = model
            .run(&refs)
            .map_err(|e| anyhow::anyhow!("ORT pipeline model '{}' failed: {e}", plan.model))?;
        for (name, value) in model.output_names().iter().zip(outputs) {
            tensors.insert(format!("{}.{}", plan.model, name), value);
        }
        Ok(tensors)
    }
}

fn workflow_request_value(
    field: &RuntimeInputRole,
    request: &GenerateRequest,
    contract: &TensorContract,
) -> anyhow::Result<Option<Value>> {
    let scalar_i64 = |value: i64| {
        let shape = scalar_or_batch_shape(contract)?;
        Value::from_slice_i64(&vec![value; shape_numel(&shape)], &shape).map_err(Into::into)
    };
    let scalar_f32 = |value: f32| {
        let shape = scalar_or_batch_shape(contract)?;
        Value::from_slice_f32(&vec![value; shape_numel(&shape)], &shape).map_err(Into::into)
    };
    match field {
        RuntimeInputRole::PromptTokens => match &request.prompt {
            GeneratePrompt::TokenIds(tokens) => {
                let data = tokens
                    .iter()
                    .map(|token| i64::from(*token))
                    .collect::<Vec<_>>();
                let shape = match contract.rank {
                    1 => vec![data.len() as i64],
                    2 => vec![1, data.len() as i64],
                    rank => anyhow::bail!(
                        "prompt token workflow input must have rank 1 or 2, got {rank}"
                    ),
                };
                Ok(Some(Value::from_slice_i64(&data, &shape)?))
            }
            GeneratePrompt::Text(_) => anyhow::bail!(
                "prompt_tokens request binding requires token ids; use a tokenizer adapter for text"
            ),
        },
        RuntimeInputRole::PromptText => {
            anyhow::bail!("prompt_text request binding requires a versioned tokenizer adapter")
        }
        RuntimeInputRole::MaxIterations | RuntimeInputRole::MaxOutputTokens => {
            scalar_i64(request.options.max_new_tokens as i64).map(Some)
        }
        RuntimeInputRole::Seed => {
            scalar_i64(request.options.seed.unwrap_or_default() as i64).map(Some)
        }
        RuntimeInputRole::SamplingTemperature => scalar_f32(request.options.temperature).map(Some),
        RuntimeInputRole::SamplingTopK => scalar_i64(request.options.top_k as i64).map(Some),
        RuntimeInputRole::SamplingTopP => scalar_f32(request.options.top_p).map(Some),
        RuntimeInputRole::Media | RuntimeInputRole::Constraint | RuntimeInputRole::SessionId => {
            Ok(None)
        }
    }
}

fn workflow_literal_value(
    scalar: &ScalarValue,
    contract: &TensorContract,
) -> anyhow::Result<Value> {
    let shape = literal_shape(contract)?;
    let numel = shape_numel(&shape);
    match scalar {
        ScalarValue::Integer(value) => {
            let (bytes, dtype) = match contract.dtype.as_str() {
                "int64" => (value.to_le_bytes().repeat(numel), DataType::Int64),
                "int32" => (
                    i32::try_from(*value)
                        .context("integer literal exceeds int32")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Int32,
                ),
                "int16" => (
                    i16::try_from(*value)
                        .context("integer literal exceeds int16")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Int16,
                ),
                "int8" => (
                    vec![
                        i8::try_from(*value).context("integer literal exceeds int8")? as u8;
                        numel
                    ],
                    DataType::Int8,
                ),
                "uint64" => (
                    u64::try_from(*value)
                        .context("integer literal is negative")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Uint64,
                ),
                "uint32" => (
                    u32::try_from(*value)
                        .context("integer literal exceeds uint32")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Uint32,
                ),
                "uint16" => (
                    u16::try_from(*value)
                        .context("integer literal exceeds uint16")?
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Uint16,
                ),
                "uint8" => (
                    vec![u8::try_from(*value).context("integer literal exceeds uint8")?; numel],
                    DataType::Uint8,
                ),
                _ => anyhow::bail!(
                    "integer workflow literal is incompatible with declared dtype '{}'",
                    contract.dtype
                ),
            };
            Value::from_raw_bytes(bytes, &shape, dtype).map_err(Into::into)
        }
        ScalarValue::Float(value) => {
            let (bytes, dtype) = match contract.dtype.as_str() {
                "float32" | "fp32" => (
                    (*value as f32).to_le_bytes().repeat(numel),
                    DataType::Float32,
                ),
                "float16" | "fp16" => (
                    half::f16::from_f64(*value)
                        .to_bits()
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::Float16,
                ),
                "bfloat16" | "bf16" => (
                    half::bf16::from_f64(*value)
                        .to_bits()
                        .to_le_bytes()
                        .repeat(numel),
                    DataType::BFloat16,
                ),
                _ => anyhow::bail!(
                    "floating-point workflow literal is incompatible with declared dtype '{}'",
                    contract.dtype
                ),
            };
            Value::from_raw_bytes(bytes, &shape, dtype).map_err(Into::into)
        }
        ScalarValue::Bool(value) if contract.dtype == "bool" => {
            Value::from_raw_bytes(vec![u8::from(*value); numel], &shape, DataType::Bool)
                .map_err(Into::into)
        }
        ScalarValue::String(_) => {
            anyhow::bail!("string literal workflow inputs require an adapter binding")
        }
        _ => anyhow::bail!(
            "workflow literal is incompatible with declared dtype '{}'",
            contract.dtype
        ),
    }
}

fn scalar_or_batch_shape(contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    match contract.rank {
        0 => Ok(Vec::new()),
        1 => Ok(vec![1]),
        rank => anyhow::bail!("request scalar binding requires rank 0 or 1, got {rank}"),
    }
}

fn literal_shape(contract: &TensorContract) -> anyhow::Result<Vec<i64>> {
    let Some(shape) = &contract.shape else {
        if contract.rank == 0 {
            return Ok(Vec::new());
        }
        anyhow::bail!("literal workflow input requires a fully declared shape");
    };
    shape
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Fixed(value) => Ok(*value),
            TensorDimension::Symbol(symbol) if symbol == "batch" => Ok(1),
            TensorDimension::Symbol(symbol) => {
                anyhow::bail!("literal workflow input has unresolved dimension '{symbol}'")
            }
        })
        .collect()
}

fn shape_numel(shape: &[i64]) -> usize {
    shape.iter().map(|dimension| *dimension as usize).product()
}

fn validate_workflow_value(
    name: &str,
    value: &Value,
    contract: &TensorContract,
    symbols: &mut HashMap<String, i64>,
    dynamic_symbols: &std::collections::HashSet<String>,
) -> anyhow::Result<()> {
    let expected_dtype = match contract.dtype.as_str() {
        "float32" | "fp32" => DataType::Float32,
        "float16" | "fp16" => DataType::Float16,
        "bfloat16" | "bf16" => DataType::BFloat16,
        "int64" => DataType::Int64,
        "int32" => DataType::Int32,
        "int16" => DataType::Int16,
        "int8" => DataType::Int8,
        "uint64" => DataType::Uint64,
        "uint32" => DataType::Uint32,
        "uint16" => DataType::Uint16,
        "uint8" => DataType::Uint8,
        "bool" => DataType::Bool,
        dtype => anyhow::bail!("workflow value '{name}' uses unsupported dtype '{dtype}'"),
    };
    if value.dtype() != expected_dtype {
        anyhow::bail!(
            "workflow value '{name}' has dtype {:?}, expected {}",
            value.dtype(),
            contract.dtype
        );
    }
    if value.shape().len() != contract.rank {
        anyhow::bail!(
            "workflow value '{name}' has rank {}, expected {}",
            value.shape().len(),
            contract.rank
        );
    }
    if let Some(shape) = &contract.shape {
        for (axis, (declared, actual)) in shape.iter().zip(value.shape()).enumerate() {
            match declared {
                TensorDimension::Fixed(expected) if expected != actual => anyhow::bail!(
                    "workflow value '{name}' axis {axis} is {actual}, expected {expected}"
                ),
                TensorDimension::Symbol(symbol) if dynamic_symbols.contains(symbol) => {}
                TensorDimension::Symbol(symbol) => match symbols.get(symbol) {
                    Some(expected) if expected != actual => anyhow::bail!(
                        "workflow value '{name}' axis {axis} binds symbol '{symbol}' to {actual}, \
                         but it was already {expected}"
                    ),
                    Some(_) => {}
                    None => {
                        symbols.insert(symbol.clone(), *actual);
                    }
                },
                TensorDimension::Fixed(_) => {}
            }
        }
    }
    Ok(())
}

fn validate_state_recurrence(
    cell: &str,
    current: &Value,
    next: &Value,
    state: &onnx_genai_metadata::WorkflowStateCell,
    values: &PipelineTensors,
) -> anyhow::Result<()> {
    if current.dtype() != next.dtype() || current.shape().len() != next.shape().len() {
        anyhow::bail!("workflow state '{cell}' update must preserve dtype and rank");
    }
    match &state.recurrence {
        onnx_genai_metadata::ShapeRecurrence::Invariant => {
            if current.shape() != next.shape() {
                anyhow::bail!(
                    "workflow state '{cell}' is invariant but changed shape from {:?} to {:?}",
                    current.shape(),
                    next.shape()
                );
            }
        }
        onnx_genai_metadata::ShapeRecurrence::Growing {
            axis,
            increment,
            max,
        } => {
            for (index, (before, after)) in current.shape().iter().zip(next.shape()).enumerate() {
                if index != *axis && before != after {
                    anyhow::bail!(
                        "workflow state '{cell}' changed non-growing axis {index} from {before} \
                         to {after}"
                    );
                }
            }
            let growth = i64::try_from(workflow_scalar_usize(values, increment)?)
                .context("workflow state growth increment exceeds i64")?;
            let limit = i64::try_from(workflow_scalar_usize(values, max)?)
                .context("workflow state growth limit exceeds i64")?;
            let before = *current.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' grows outside its tensor rank")
            })?;
            let after = *next.shape().get(*axis).with_context(|| {
                format!("workflow state '{cell}' grows outside its tensor rank")
            })?;
            let expected = before
                .checked_add(growth)
                .with_context(|| format!("workflow state '{cell}' shape growth overflowed"))?;
            if after != expected {
                anyhow::bail!(
                    "workflow state '{cell}' growing axis {axis} changed from {before} to {after}, \
                     expected {expected}"
                );
            }
            if after > limit {
                anyhow::bail!(
                    "workflow state '{cell}' growing axis {axis} reached {after}, above maximum \
                     {limit}"
                );
            }
        }
    }
    Ok(())
}

fn append_workflow_value(previous: &Value, next: &Value) -> anyhow::Result<Value> {
    if previous.dtype() != next.dtype() || previous.shape().len() != next.shape().len() {
        anyhow::bail!("workflow append emit requires matching dtype and rank");
    }
    let mut shape = previous.shape().to_vec();
    let Some(last) = shape.last_mut() else {
        anyhow::bail!("workflow append emit requires rank >= 1");
    };
    for (left, right) in previous
        .shape()
        .iter()
        .zip(next.shape())
        .take(previous.shape().len() - 1)
    {
        if left != right {
            anyhow::bail!("workflow append emit requires equal non-appended dimensions");
        }
    }
    let left_width = *last as usize;
    let right_width = next.shape().last().copied().unwrap_or_default() as usize;
    let outer = previous.shape()[..previous.shape().len() - 1]
        .iter()
        .map(|dimension| *dimension as usize)
        .product::<usize>();
    *last += right_width as i64;
    let dtype = previous.dtype();
    let element_size = dtype.size_of();
    let left = previous.to_raw_bytes()?;
    let right = next.to_raw_bytes()?;
    let mut data = Vec::with_capacity(left.len() + right.len());
    for row in 0..outer {
        data.extend_from_slice(
            &left[row * left_width * element_size..(row + 1) * left_width * element_size],
        );
        data.extend_from_slice(
            &right[row * right_width * element_size..(row + 1) * right_width * element_size],
        );
    }
    Value::from_raw_bytes(data, &shape, dtype).map_err(Into::into)
}

fn workflow_scalar_usize(values: &PipelineTensors, name: &str) -> anyhow::Result<usize> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow scalar value '{name}' is unavailable"))?;
    let data = value
        .to_vec_i64()
        .with_context(|| format!("workflow scalar '{name}' must be an integer tensor"))?;
    let [scalar] = data.as_slice() else {
        anyhow::bail!("workflow scalar '{name}' must contain exactly one value");
    };
    usize::try_from(*scalar)
        .with_context(|| format!("workflow scalar '{name}' must be non-negative"))
}

fn workflow_scalar_bool(values: &PipelineTensors, name: &str) -> anyhow::Result<bool> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow predicate value '{name}' is unavailable"))?;
    match value.dtype() {
        DataType::Bool => {
            let data = value.to_raw_bytes()?;
            let [scalar] = data.as_slice() else {
                anyhow::bail!("workflow bool predicate '{name}' must contain exactly one value");
            };
            Ok(*scalar != 0)
        }
        _ => {
            let data = value.to_vec_i64()?;
            let [scalar] = data.as_slice() else {
                anyhow::bail!("workflow integer predicate '{name}' must contain exactly one value");
            };
            Ok(*scalar != 0)
        }
    }
}

fn workflow_scalar_key(values: &PipelineTensors, name: &str) -> anyhow::Result<String> {
    let value = values
        .get(name)
        .with_context(|| format!("workflow branch value '{name}' is unavailable"))?;
    if value.dtype() == DataType::Bool {
        return Ok(workflow_scalar_bool(values, name)?.to_string());
    }
    let data = value.to_vec_i64()?;
    let [scalar] = data.as_slice() else {
        anyhow::bail!("workflow branch tensor '{name}' must contain exactly one value");
    };
    Ok(scalar.to_string())
}

#[cfg(test)]
mod workflow_scalar_tests {
    use super::*;

    #[test]
    fn batched_predicate_is_not_silently_reduced() {
        let mut values = PipelineTensors::new();
        values.insert(
            "done".to_string(),
            Value::from_raw_bytes(vec![0, 1], &[2], DataType::Bool).expect("bool tensor"),
        );

        let error = workflow_scalar_bool(&values, "done").expect_err("batched predicate fails");
        assert!(error.to_string().contains("exactly one value"));
    }

    #[test]
    fn batched_branch_key_is_not_silently_reduced() {
        let mut values = PipelineTensors::new();
        values.insert(
            "case".to_string(),
            Value::from_slice_i64(&[0, 1], &[2]).expect("integer tensor"),
        );

        let error = workflow_scalar_key(&values, "case").expect_err("batched branch key fails");
        assert!(error.to_string().contains("exactly one value"));
    }

    #[test]
    fn growing_state_uses_recurrence_instead_of_freezing_its_symbol() {
        let contract: TensorContract =
            serde_yaml::from_str("{ dtype: int64, rank: 2, shape: [batch, sequence] }")
                .expect("contract");
        let mut symbols = HashMap::new();
        let dynamic_symbols = std::collections::HashSet::from(["sequence".to_string()]);
        let current = Value::from_slice_i64(&[1, 2], &[1, 2]).expect("current");
        let next = Value::from_slice_i64(&[1, 2, 3], &[1, 3]).expect("next");
        validate_workflow_value(
            "current",
            &current,
            &contract,
            &mut symbols,
            &dynamic_symbols,
        )
        .expect("current contract");
        validate_workflow_value("next", &next, &contract, &mut symbols, &dynamic_symbols)
            .expect("growing symbol remains dynamic");

        let state: onnx_genai_metadata::WorkflowStateCell = serde_yaml::from_str(
            r#"
contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
scope: invocation
initializer: initial
recurrence: { kind: growing, axis: 1, increment: accepted, max: max_context }
"#,
        )
        .expect("state");
        let mut values = PipelineTensors::new();
        values.insert(
            "accepted".to_string(),
            Value::from_slice_i64(&[1], &[]).expect("increment"),
        );
        values.insert(
            "max_context".to_string(),
            Value::from_slice_i64(&[4], &[]).expect("limit"),
        );
        validate_state_recurrence("tokens", &current, &next, &state, &values)
            .expect("bounded growth validates");

        let invalid = Value::from_slice_i64(&[1, 2, 3, 4], &[1, 4]).expect("invalid next");
        let error = validate_state_recurrence("tokens", &current, &invalid, &state, &values)
            .expect_err("wrong increment fails");
        assert!(error.to_string().contains("expected 3"));
    }

    #[test]
    fn literals_and_append_support_declared_runtime_dtypes() {
        let int_contract: TensorContract =
            serde_yaml::from_str("{ dtype: int16, rank: 1, shape: [2] }").expect("contract");
        let integer =
            workflow_literal_value(&ScalarValue::Integer(7), &int_contract).expect("int16 literal");
        assert_eq!(integer.dtype(), DataType::Int16);

        let half_contract: TensorContract =
            serde_yaml::from_str("{ dtype: float16, rank: 1, shape: [2] }").expect("contract");
        let left =
            workflow_literal_value(&ScalarValue::Float(1.0), &half_contract).expect("half literal");
        let right =
            workflow_literal_value(&ScalarValue::Float(2.0), &half_contract).expect("half literal");
        let appended = append_workflow_value(&left, &right).expect("half append");
        assert_eq!(appended.dtype(), DataType::Float16);
        assert_eq!(appended.shape(), &[4]);
    }
}

/// Dump one iterative step's loop-carried tensor to `ONNX_GENAI_STEP_DUMP_DIR`
/// (when set) as `step_{i}_{port}.json` — used by the diffusion demo to animate
/// the reverse process. Best-effort; failures are ignored (never affects a run).
fn dump_iterative_step(denoiser: &str, port: &str, step: usize, value: &Value, step_ms: f64) {
    let Ok(dir) = std::env::var("ONNX_GENAI_STEP_DUMP_DIR") else {
        return;
    };
    let shape: Vec<i64> = value.shape().to_vec();
    // Emit int64 token sequences as integers (language diffusion) and everything
    // else as f32 (image latents). `step_ms` is this step's wall-clock time.
    let payload = match value.dtype() {
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => value
            .to_vec_i64()
            .ok()
            .map(|data| serde_json::json!({"dtype": "i64", "shape": shape, "data": data, "step_ms": step_ms})),
        _ => value
            .to_vec_f32()
            .ok()
            .map(|data| serde_json::json!({"dtype": "f32", "shape": shape, "data": data, "step_ms": step_ms})),
    };
    if let Some(payload) = payload {
        let path =
            std::path::Path::new(&dir).join(format!("step_{step:04}_{denoiser}_{port}.json"));
        let _ = std::fs::write(path, payload.to_string());
    }
}

/// Write the per-pipeline-stage timing report (`stages.json`) to the step-dump
/// directory when `ONNX_GENAI_STEP_DUMP_DIR` is set. Each entry is
/// `{component, phase, ms[, steps]}`, covering the prompt encoders (`encode`),
/// the denoiser loop total (`denoise`), and the final VAE-style pass (`decode`).
fn dump_stage_timings(stages: &[serde_json::Value]) {
    let Ok(dir) = std::env::var("ONNX_GENAI_STEP_DUMP_DIR") else {
        return;
    };
    let path = std::path::Path::new(&dir).join("stages.json");
    let _ = std::fs::write(path, serde_json::json!({ "stages": stages }).to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inpaint_loop_input_is_nine_channels_in_declared_order() {
        let latent =
            Value::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 4, 1, 2])
                .unwrap();
        let conditioning = Value::from_slice_f32(
            &[
                10.0, 11.0, // mask
                20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, // masked latent
            ],
            &[1, 5, 1, 2],
        )
        .unwrap();
        let mut constants = PipelineTensors::new();
        constants.insert("denoiser.sample.conditioning".to_string(), conditioning);

        let combined =
            append_loop_conditioning(&constants, "denoiser.sample", &latent, true).unwrap();

        assert_eq!(combined.shape(), &[1, 9, 1, 2]);
        assert_eq!(
            combined.to_vec_f32_lossy().unwrap(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // noisy latent
                10.0, 11.0, // mask
                20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, // masked latent
            ]
        );
    }
}
