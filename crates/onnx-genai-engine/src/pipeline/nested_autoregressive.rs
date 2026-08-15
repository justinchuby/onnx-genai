//! Nested autoregressive (multi-decoder text-to-speech) pipeline execution.
//!
//! Pure code motion from `pipeline.rs`: the dual outer/inner autoregressive
//! decode driver, its post-decode synthesis entry point, and the pre-embedder
//! and named-output resolution helpers it relies on.

use super::*;

impl PipelineEngine {
    /// Post-decode counterpart to [`synthesize`](Self::synthesize) for a
    /// nested-AR (multi-decoder TTS) pipeline: drive the outer/inner loops (which
    /// publish `{outer}.output_codes` into the pool), then run the `final_only`
    /// vocoder stage over the pool to produce the waveform.
    pub(crate) fn synthesize_nested(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineSynthesis> {
        let present = pipeline_request.present.clone();
        let post_decode_components = match &self.plan {
            PipelinePlan::NestedAutoregressive(plan) => plan.post_decode_components.clone(),
            _ => anyhow::bail!("internal error: synthesize_nested on a non-nested plan"),
        };
        let (generation, mut tensors) = self.run_nested_autoregressive(pipeline_request)?;
        self.run_prompt_phase_components(
            &post_decode_components,
            &mut tensors,
            "postlogue",
            &present,
            None,
        )?;
        Ok(PipelineSynthesis {
            generation,
            tensors,
        })
    }

    /// Drive a **dual, hierarchically-nested autoregressive** pipeline for the
    /// multi-decoder TTS shape in DESIGN.md §20.3.
    ///
    /// The **outer** decoder (talker) runs up to `max_frames` frames; each outer
    /// step (one audio frame) produces a `last_hidden_state` that seeds the
    /// **inner** decoder (code_predictor) AR loop of `num_code_groups` steps.
    /// The inner loop threads the outer hidden state at inner step 0 and the
    /// inner decoder's own per-code embedding output on later steps. Every code
    /// group is assembled into the synthetic pool tensor `{outer}.output_codes`
    /// of shape `[1, frames, num_code_groups]` (int64), and the flattened codes
    /// are returned as the [`GenerateResult`]'s token ids.
    pub(crate) fn run_nested_autoregressive(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<(GenerateResult, PipelineTensors)> {
        let plan = match &self.plan {
            PipelinePlan::NestedAutoregressive(plan) => plan.clone(),
            _ => anyhow::bail!(
                "synthesize()/generate() on a nested pipeline requires a nested_autoregressive plan"
            ),
        };
        let present = pipeline_request.present.clone();
        self.ensure_component_present(&plan.outer, &present, "nested outer decoder")?;
        self.ensure_component_present(&plan.inner, &present, "nested inner decoder")?;

        let options = pipeline_request.request.options.clone();
        options.validate()?;
        let prompt_tokens = tokenize_with(self.tokenizer()?, &pipeline_request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }

        let mut tensors = self.prepare_request_tensors(pipeline_request.inputs, &present)?;
        self.seed_prompt_token_inputs(&plan.prompt_components, &prompt_tokens, &mut tensors)?;
        // Explicitly seed the prefill embedder's metadata-declared prompt input
        // with the tokenized prompt (int64 `[1, L]`) unless a dataflow edge
        // already routes it. This does NOT rely on `is_token_input_name` — the
        // prompt port is declared in the PrefillEmbedderSpec, never guessed.
        if let Some(prefill) = plan
            .prefill_embedder
            .as_ref()
            .filter(|binding| self.plan.component_is_present(&binding.component, &present))
        {
            let endpoint = format!("{}.{}", prefill.component, prefill.prompt_input);
            let routed = plan.dataflow.iter().any(|edge| edge.to == endpoint);
            if !routed && !tensors.contains_key(&endpoint) {
                let ids: Vec<i64> = prompt_tokens.iter().map(|&t| i64::from(t)).collect();
                let value = Value::from_slice_i64(&ids, &[1, ids.len() as i64])?;
                tensors.insert(endpoint, value);
            }
        }
        self.run_prompt_phase_components(
            &plan.prompt_components,
            &mut tensors,
            "prologue",
            &present,
            None,
        )?;

        // Fixed routed extras for each decoder (encoder conditioning etc.). The
        // inner decoder's seed input is threaded per inner step, so exclude it.
        // In pre-embedder mode the outer decoder's per-step `inputs_embeds` is
        // built each frame (not a fixed routed extra), so exclude it too.
        let outer_extra_exclude = plan.pre_embedder.as_ref().map(|p| p.outer_input.as_str());
        let outer_extras =
            self.decoder_extra_inputs(&plan.outer, &tensors, outer_extra_exclude, &present)?;
        let inner_extras = self.decoder_extra_inputs(
            &plan.inner,
            &tensors,
            Some(&plan.inner_embeds_input),
            &present,
        )?;

        let outer_session = self
            .models
            .session(&plan.outer)
            .with_context(|| format!("nested outer decoder '{}' was not loaded", plan.outer))?;
        let inner_session = self
            .models
            .session(&plan.inner)
            .with_context(|| format!("nested inner decoder '{}' was not loaded", plan.inner))?;

        let pre_embed = self.resolve_pre_embedder(&plan, &present, outer_session)?;
        let prefill = self.resolve_prefill(&plan, &present, pre_embed.as_ref(), &tensors)?;
        let inner_embed_output = resolve_inner_embedding_output(inner_session, &plan)?;
        let mut outer_state = DecodeState::new_with_io(outer_session, plan.outer_io.as_ref())?;
        let mut codes: Vec<i64> = Vec::with_capacity(plan.max_frames * plan.num_code_groups);
        // The outer loop feeds the full prompt on frame 0 (prefill), then the
        // previous frame's outer argmax token on each subsequent frame.
        let mut outer_input_tokens = prompt_tokens.clone();
        let mut outer_past_len = 0usize;
        // Pre-embedder mode only: the previous frame's assembled code tuple
        // `[outer_code_0, inner_code_1, ..., inner_code_{G-1}]`, used to build the
        // next frame's `inputs_embeds`. `None` on frame 0 (prefill).
        let mut prev_frame_codes: Option<Vec<i64>> = None;

        for frame in 0..plan.max_frames {
            let outer_step = run_outer_talker_step(
                frame,
                &plan,
                outer_session,
                &mut outer_state,
                &outer_extras,
                pre_embed.as_ref(),
                prefill.as_ref(),
                &outer_input_tokens,
                &mut outer_past_len,
                prev_frame_codes.as_deref(),
            )?;
            // The talker autoregresses on its own per-frame prediction.
            outer_input_tokens = vec![u32::try_from(outer_step.outer_token).unwrap_or(0)];

            let frame_inner_codes = run_inner_code_loop(
                &plan,
                inner_session,
                &inner_extras,
                &inner_embed_output,
                outer_step.seed,
                &mut codes,
            )?;

            // Pre-embedder mode: remember this frame's code tuple for the next
            // frame's `frame_codes`: the talker's own code as group 0 and the
            // inner residuals for groups 1..G-1, where code_0 comes from the
            // talker rather than the code predictor.
            if pre_embed.is_some() {
                let mut tuple = Vec::with_capacity(plan.num_code_groups);
                tuple.push(outer_step.outer_token);
                tuple.extend_from_slice(&frame_inner_codes[1..]);
                prev_frame_codes = Some(tuple);
            }
        }

        let result = publish_generation_result(&plan, codes, &mut tensors)?;
        Ok((result, tensors))
    }

    /// Resolve the optional pre-embedder against its loaded session, confirming
    /// every metadata-declared port exists (sessions are not available at
    /// plan-build time) — there is NO name/dtype guessing here.
    fn resolve_pre_embedder<'a>(
        &'a self,
        plan: &NestedAutoregressivePlan,
        present: &BTreeSet<String>,
        outer_session: &'a Session,
    ) -> anyhow::Result<Option<ResolvedPreEmbedder<'a>>> {
        // All port names come from the `PreEmbedderSpec` / the required dataflow
        // edge — there is NO name/dtype guessing here.
        let pre_embed = match plan
            .pre_embedder
            .as_ref()
            .filter(|binding| self.plan.component_is_present(&binding.component, present))
        {
            Some(binding) => {
                let session = self.models.session(&binding.component).with_context(|| {
                    format!("nested pre_embedder '{}' was not loaded", binding.component)
                })?;
                let frame_codes_input = binding.frame_codes_input.clone();
                if !session
                    .inputs()
                    .iter()
                    .any(|info| info.name == frame_codes_input)
                {
                    anyhow::bail!(
                        "nested pre_embedder '{}' has no declared frame_codes input '{}'",
                        binding.component,
                        frame_codes_input
                    );
                }
                let text_embed_input = binding.text_embed_input.clone();
                if let Some(name) = &text_embed_input
                    && !session.inputs().iter().any(|info| &info.name == name)
                {
                    anyhow::bail!(
                        "nested pre_embedder '{}' has no declared text_embed input '{}'",
                        binding.component,
                        name
                    );
                }
                if !session
                    .output_names()
                    .iter()
                    .any(|name| name == &binding.output_port)
                {
                    anyhow::bail!(
                        "nested pre_embedder '{}' has no declared output port '{}'",
                        binding.component,
                        binding.output_port
                    );
                }
                // Hidden size for the per-step embedding / zero `text_embed`:
                // prefer the outer decoder's `inputs_embeds` input (a metadata
                // port captured from the edge `to`), fall back to the
                // pre-embedder's declared output port.
                let hidden = outer_session
                    .inputs()
                    .iter()
                    .find(|info| info.name == binding.outer_input)
                    .and_then(|info| info.shape.last().copied())
                    .filter(|dim| *dim > 0)
                    .or_else(|| {
                        session
                            .outputs()
                            .iter()
                            .find(|info| info.name == binding.output_port)
                            .and_then(|info| info.shape.last().copied())
                            .filter(|dim| *dim > 0)
                    })
                    .map(|dim| dim as usize)
                    .with_context(|| {
                        format!(
                            "could not determine hidden size for nested pre_embedder '{}' \
                             (outer '{}' input '{}' has no static last dim)",
                            binding.component, plan.outer, binding.outer_input
                        )
                    })?;
                Some(ResolvedPreEmbedder {
                    session,
                    outer_input: binding.outer_input.clone(),
                    output_port: binding.output_port.clone(),
                    frame_codes_input,
                    text_embed_input,
                    hidden,
                })
            }
            None => None,
        };
        Ok(pre_embed)
    }

    /// Resolve the optional prefill embedder's pooled outputs (it ran as a
    /// prompt-phase component, seeded with the tokenized prompt via
    /// `seed_prompt_token_inputs`). `prefill_embeds` [1, prefill_len, hidden]
    /// seeds the talker's frame-0 `inputs_embeds` DIRECTLY (multi-position
    /// PREFILL); `trailing_text_embeds` [1, trailing_len, hidden] supplies one
    /// `text_embed` vector per outer frame `k >= 1` (fed through the
    /// pre-embedder). Only valid alongside `pre_embedder`.
    fn resolve_prefill(
        &self,
        plan: &NestedAutoregressivePlan,
        present: &BTreeSet<String>,
        pre_embed: Option<&ResolvedPreEmbedder<'_>>,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Option<ResolvedPrefill>> {
        let prefill = match plan
            .prefill_embedder
            .as_ref()
            .filter(|binding| self.plan.component_is_present(&binding.component, present))
        {
            Some(binding) => {
                let component = binding.component.as_str();
                let pre = pre_embed.with_context(|| {
                    format!(
                        "nested prefill_embedder '{component}' requires a pre_embedder to be set"
                    )
                })?;
                let _ = self.models.session(component).with_context(|| {
                    format!("nested prefill_embedder '{component}' was not loaded")
                })?;
                // The prefill component's two float outputs are metadata-declared
                // (`prefill_output` / `trailing_output`) — no name/dtype guessing.
                let prefill_name = binding.prefill_output.as_str();
                let trailing_name = binding.trailing_output.as_str();
                let prefill_value = tensors
                    .get(&format!("{component}.{prefill_name}"))
                    .with_context(|| {
                        format!(
                            "nested prefill_embedder '{component}' produced no pooled \
                             '{prefill_name}' output (did it run in the prompt phase?)"
                        )
                    })?;
                let prefill_len = match prefill_value.shape() {
                    [1, p, _] if *p > 0 => *p as usize,
                    other => anyhow::bail!(
                        "nested prefill_embedder '{component}' '{prefill_name}' must be \
                         [1, prefill_len, hidden]; got {other:?}"
                    ),
                };
                let prefill_embeds = clone_value(prefill_value)?;
                let trailing_value = tensors
                    .get(&format!("{component}.{trailing_name}"))
                    .with_context(|| {
                        format!(
                            "nested prefill_embedder '{component}' produced no pooled \
                             '{trailing_name}' output (did it run in the prompt phase?)"
                        )
                    })?;
                let trailing_len = match trailing_value.shape() {
                    [1, t, h] if *h as usize == pre.hidden => *t as usize,
                    other => anyhow::bail!(
                        "nested prefill_embedder '{component}' '{trailing_name}' must be \
                         [1, trailing_len, {}]; got {other:?}",
                        pre.hidden
                    ),
                };
                let trailing = trailing_value.to_vec_f32_lossy().map_err(|e| {
                    anyhow::anyhow!("failed to read trailing_text_embeds tensor: {e}")
                })?;
                Some(ResolvedPrefill {
                    prefill_embeds,
                    prefill_len,
                    trailing,
                    trailing_len,
                    hidden: pre.hidden,
                })
            }
            None => None,
        };
        Ok(prefill)
    }
}

/// The inner decoder's per-code embedding output, declared explicitly on the
/// plan and threaded into the next inner step's seed input. Validated to be an
/// actual graph output so a misconfigured contract fails with a clear error
/// rather than a silent binding failure.
fn resolve_inner_embedding_output(
    inner_session: &Session,
    plan: &NestedAutoregressivePlan,
) -> anyhow::Result<String> {
    let declared = plan.inner_embedding_output.clone();
    if inner_session
        .output_names()
        .iter()
        .any(|name| name == &declared)
    {
        Ok(declared)
    } else {
        anyhow::bail!(
            "nested inner decoder '{}' does not expose the declared \
             pipeline.strategy.inner_embedding_output '{}'",
            plan.inner,
            declared
        )
    }
}

/// The talker's per-frame result: its argmax token (code group 0 in
/// pre-embedder mode) and the `[1, 1, H]` hidden-state seed for the inner loop.
struct OuterStepOutcome {
    outer_token: i64,
    seed: Value,
}

/// Run one outer talker step (one audio frame): build or look up the talker's
/// per-step `inputs_embeds` (pre-embedder mode) or feed the token ids directly,
/// advance the KV past, then read the argmax token and the inner-loop seed.
#[allow(clippy::too_many_arguments)]
fn run_outer_talker_step(
    frame: usize,
    plan: &NestedAutoregressivePlan,
    outer_session: &Session,
    outer_state: &mut DecodeState,
    outer_extras: &[(String, Value)],
    pre_embed: Option<&ResolvedPreEmbedder<'_>>,
    prefill: Option<&ResolvedPrefill>,
    outer_input_tokens: &[TokenId],
    outer_past_len: &mut usize,
    prev_frame_codes: Option<&[i64]>,
) -> anyhow::Result<OuterStepOutcome> {
    let outer_outputs = if let Some(pre) = pre_embed {
        // Build (or, on frame 0 with a prefill embedder, look up) the talker's
        // per-step `inputs_embeds`.
        let (inputs_embeds, positions) = if let Some(prefill) = prefill.filter(|_| frame == 0) {
            // Frame 0 PREFILL: feed the prefill embedder's multi-position
            // `prefill_embeds` DIRECTLY to the talker (do NOT run the
            // pre-embedder), advancing the KV past by `prefill_len`.
            (clone_value(&prefill.prefill_embeds)?, prefill.prefill_len)
        } else {
            // Build this frame's `frame_codes` from the previous frame's code
            // tuple (frame 0 without a prefill embedder uses a zero seed), run
            // the pre-embedder to materialize a single-position `inputs_embeds`.
            // With a prefill embedder, frames `k >= 1` feed
            // `text_embed = trailing_text_embeds[:, k-1, :]` (zeros once the
            // trailing text is exhausted — a close stand-in for the reference's
            // tts_pad embedding; exact tts_pad is a documented refinement).
            let frame_codes = prev_frame_codes
                .map(<[i64]>::to_vec)
                .unwrap_or_else(|| vec![0i64; plan.num_code_groups]);
            let text_embed = match prefill {
                Some(prefill) => {
                    let idx = frame - 1;
                    let hidden = prefill.hidden;
                    let slice = if idx < prefill.trailing_len {
                        prefill.trailing[idx * hidden..(idx + 1) * hidden].to_vec()
                    } else {
                        vec![0.0f32; hidden]
                    };
                    Some(slice)
                }
                None => None,
            };
            (
                run_pre_embedder(pre, &frame_codes, text_embed.as_deref())?,
                1,
            )
        };
        let mut step_extras = Vec::with_capacity(outer_extras.len() + 1);
        for (name, value) in outer_extras {
            step_extras.push((name.clone(), clone_value(value)?));
        }
        step_extras.push((pre.outer_input.clone(), inputs_embeds));
        // Match the token-position count to the fed `inputs_embeds` sequence
        // length so any position_ids/attention_mask the talker exposes stay
        // consistent (the talker itself is embeds-driven and ignores the token
        // ids).
        let position_tokens = vec![0u32; positions];
        let outputs = run_decode_step_with_extra(
            outer_session,
            outer_state,
            &position_tokens,
            *outer_past_len,
            &step_extras,
        )?;
        *outer_past_len += positions;
        outputs
    } else {
        let outputs = run_decode_step_with_extra(
            outer_session,
            outer_state,
            outer_input_tokens,
            *outer_past_len,
            outer_extras,
        )?;
        *outer_past_len += outer_input_tokens.len();
        outputs
    };

    let outer_logits = named_output(outer_session, &outer_outputs, &plan.outer_logits_output)?;
    let outer_token = argmax_last_row(outer_logits)?;
    let hidden = named_output(outer_session, &outer_outputs, &plan.outer_hidden_output)?;
    let seed = last_position_hidden(hidden)?;
    Ok(OuterStepOutcome { outer_token, seed })
}

/// Run the inner code_predictor loop for one frame: `num_code_groups` residual
/// codes threaded through the inner decoder's own per-code embedding output.
/// Appends each code to `codes` and returns this frame's code tuple.
fn run_inner_code_loop(
    plan: &NestedAutoregressivePlan,
    inner_session: &Session,
    inner_extras: &[(String, Value)],
    inner_embed_output: &str,
    seed: Value,
    codes: &mut Vec<i64>,
) -> anyhow::Result<Vec<i64>> {
    let mut inner_state = DecodeState::new_with_io(inner_session, plan.inner_io.as_ref())?;
    let mut inner_embeds = seed;
    let mut frame_inner_codes: Vec<i64> = Vec::with_capacity(plan.num_code_groups);
    for step in 0..plan.num_code_groups {
        let mut step_extras = Vec::with_capacity(inner_extras.len() + 1);
        for (name, value) in inner_extras {
            step_extras.push((name.clone(), clone_value(value)?));
        }
        step_extras.push((plan.inner_embeds_input.clone(), inner_embeds));

        let inner_outputs =
            run_decode_step_with_extra(inner_session, &mut inner_state, &[0], step, &step_extras)?;
        let inner_logits = named_output(inner_session, &inner_outputs, &plan.inner_logits_output)?;
        let inner_code = argmax_last_row(inner_logits)?;
        codes.push(inner_code);
        frame_inner_codes.push(inner_code);
        // Thread the inner decoder's per-code embedding into the next step.
        inner_embeds = clone_value(named_output(
            inner_session,
            &inner_outputs,
            inner_embed_output,
        )?)?;
    }
    Ok(frame_inner_codes)
}

/// Publish the assembled per-frame codes as `{outer}.output_codes`
/// [1, frames, num_code_groups] (int64) for the post-decode vocoder stage and
/// build the flattened-code [`GenerateResult`].
fn publish_generation_result(
    plan: &NestedAutoregressivePlan,
    codes: Vec<i64>,
    tensors: &mut PipelineTensors,
) -> anyhow::Result<GenerateResult> {
    let codes_endpoint = format!("{}.output_codes", plan.outer);
    let codes_value = Value::from_slice_i64(
        &codes,
        &[1, plan.max_frames as i64, plan.num_code_groups as i64],
    )
    .with_context(|| format!("failed to build generated-codes tensor '{codes_endpoint}'"))?;
    tensors.insert(codes_endpoint, codes_value);

    let token_ids: Vec<TokenId> = codes
        .iter()
        .map(|&c| u32::try_from(c).unwrap_or(0))
        .collect();
    Ok(GenerateResult {
        text: String::new(),
        token_ids,
        finish_reason: crate::FinishReason::MaxTokens,
        prefix_cache_hit_len: 0,
        logprobs: None,
        budget_cap: None,
    })
}

/// Locate a named session output by index and return a reference to its value.
///
/// The `name` must match a declared output port EXACTLY. Every nested-AR output
/// (logits, hidden, per-code embedding) is bound from an explicit metadata port
/// name, so no substring or case-insensitive fallback is consulted.
fn named_output<'a>(
    session: &Session,
    outputs: &'a [Value],
    name: &str,
) -> anyhow::Result<&'a Value> {
    let index = session
        .output_names()
        .iter()
        .position(|out| out == name)
        .with_context(|| format!("model did not expose output '{name}'"))?;
    outputs
        .get(index)
        .with_context(|| format!("output '{name}' index was out of range"))
}

/// Argmax over the last sequence row of a logits tensor (`[V]`, `[S, V]`, or
/// `[1, S, V]`), returning the winning vocabulary index. Ties take the lowest
/// index, matching greedy decoding.
fn argmax_last_row(logits: &Value) -> anyhow::Result<i64> {
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("failed to read logits tensor: {e}"))?;
    let vocab = match shape {
        [vocab] if *vocab > 0 => *vocab as usize,
        [seq, vocab] if *seq > 0 && *vocab > 0 => *vocab as usize,
        [batch, seq, vocab] if *batch == 1 && *seq > 0 && *vocab > 0 => *vocab as usize,
        other => anyhow::bail!("unsupported logits tensor shape: {other:?}"),
    };
    let start = data.len() - vocab;
    let row = &data[start..];
    let mut best = 0usize;
    for (i, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = i;
        }
    }
    Ok(best as i64)
}

/// Slice the last sequence position of a hidden-state tensor (`[H]`, `[S, H]`,
/// or `[1, S, H]`) into a `[1, 1, H]` `float32` seed for the inner decoder.
fn last_position_hidden(hidden: &Value) -> anyhow::Result<Value> {
    let shape = hidden.shape();
    let data = hidden
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("failed to read hidden-state tensor: {e}"))?;
    let hidden_dim = match shape {
        [h] if *h > 0 => *h as usize,
        [seq, h] if *seq > 0 && *h > 0 => *h as usize,
        [batch, seq, h] if *batch == 1 && *seq > 0 && *h > 0 => *h as usize,
        other => anyhow::bail!("unsupported hidden-state tensor shape: {other:?}"),
    };
    let start = data.len() - hidden_dim;
    Value::from_slice_f32(&data[start..], &[1, 1, hidden_dim as i64])
        .map_err(|e| anyhow::anyhow!("failed to build inner seed embedding: {e}"))
}

/// A [`PreEmbedderBinding`] resolved against its loaded session for driving —
/// the codec-sum pre-embedder that materializes the outer talker's per-step
/// `inputs_embeds` from the previous frame's codes.
struct ResolvedPreEmbedder<'a> {
    /// Loaded pre-embedder session.
    session: &'a Session,
    /// Outer decoder input port fed the per-step embeddings (`inputs_embeds`).
    outer_input: String,
    /// Pre-embedder output port feeding the outer decoder. Metadata-declared
    /// (from the required dataflow edge's `from` side) — never guessed.
    output_port: String,
    /// Pre-embedder input receiving the previous frame's codes (int64 `[1, G]`).
    /// Metadata-declared via `PreEmbedderSpec::frame_codes_input`.
    frame_codes_input: String,
    /// Optional trailing-text input. Fed the prefill embedder's per-frame
    /// `trailing_text_embeds` slice when a `prefill_embedder` is set, else zeros.
    /// Metadata-declared via `PreEmbedderSpec::text_embed_input`.
    text_embed_input: Option<String>,
    /// Embedding hidden size for the emitted `inputs_embeds` / `text_embed`.
    hidden: usize,
}

/// The optional prefill embedder's resolved, pooled outputs: the talker's
/// frame-0 multi-position PREFILL sequence and the per-frame trailing-text
/// vectors consumed as the pre-embedder's `text_embed` on frames `k >= 1`.
struct ResolvedPrefill {
    /// `prefill_embeds` [1, prefill_len, hidden]: the talker's frame-0 seed.
    prefill_embeds: Value,
    /// Number of prefill positions (`prefill_embeds.shape()[1]`).
    prefill_len: usize,
    /// Flattened `trailing_text_embeds` [1, trailing_len, hidden] as row-major
    /// f32 (`trailing[i*hidden..(i+1)*hidden]` is the vector for frame `i + 1`).
    trailing: Vec<f32>,
    /// Number of trailing-text vectors (`trailing_text_embeds.shape()[1]`).
    trailing_len: usize,
    /// Embedding hidden size (matches the pre-embedder's `hidden`).
    hidden: usize,
}

/// Build the outer talker's per-step `inputs_embeds` by running the codec-sum
/// pre-embedder over one frame's `frame_codes` (`[outer_code_0, inner_code_1,
/// ..., inner_code_{G-1}]`). Returns a `[1, 1, hidden]` embedding.
///
/// When `text_embed` is `Some`, that `[hidden]` slice is fed as the
/// trailing-text conditioning input (the prefill embedder's per-frame
/// `trailing_text_embeds` vector). When `None`, a zero `[1, 1, hidden]` tensor is
/// fed (the backward-compatible no-prefill_embedder path).
///
/// Every port used here (`frame_codes_input`, `text_embed_input`, `output_port`)
/// is metadata-declared on [`ResolvedPreEmbedder`] — there is NO name/dtype
/// guessing of the pre-embedder's ports.
fn run_pre_embedder(
    pre: &ResolvedPreEmbedder<'_>,
    frame_codes: &[i64],
    text_embed: Option<&[f32]>,
) -> anyhow::Result<Value> {
    let mut inputs: Vec<(String, Value)> = Vec::with_capacity(2);
    inputs.push((
        pre.frame_codes_input.clone(),
        Value::from_slice_i64(frame_codes, &[1, frame_codes.len() as i64])?,
    ));
    if let Some(name) = &pre.text_embed_input {
        let dtype = pre
            .session
            .inputs()
            .iter()
            .find(|info| &info.name == name)
            .map(|info| info.dtype)
            .unwrap_or(DataType::Float32);
        let data = match text_embed {
            Some(slice) => slice.to_vec(),
            None => vec![0.0f32; pre.hidden],
        };
        inputs.push((
            name.clone(),
            Value::from_f32_slice_as(&data, &[1, 1, pre.hidden as i64], dtype)
                .map_err(|e| anyhow::anyhow!("failed to build text_embed: {e}"))?,
        ));
    }
    let refs = inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let outputs = pre
        .session
        .run(&refs)
        .map_err(|e| anyhow::anyhow!("ORT pre-embedder run failed: {e}"))?;
    // Select the metadata-declared output port (never guessed by name).
    let index = pre
        .session
        .output_names()
        .iter()
        .position(|name| name == &pre.output_port)
        .with_context(|| {
            format!(
                "pre-embedder has no declared output port '{}'",
                pre.output_port
            )
        })?;
    let value = outputs
        .get(index)
        .context("pre-embedder produced no output for its declared port")?;
    clone_value(value)
}
