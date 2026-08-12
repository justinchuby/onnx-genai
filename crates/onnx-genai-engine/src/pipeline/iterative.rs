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
