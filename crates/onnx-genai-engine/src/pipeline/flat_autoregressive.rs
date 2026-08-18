//! Flat autoregressive pipeline execution.
//!
//! Pure code motion from `pipeline.rs`: the single-decoder autoregressive
//! decode driver and its paged-sequence and prefix-reuse helpers.

use super::paged_decode::{PagedMirror, PipelineDecodeLoopBackend};
use super::prefix_reuse::{KvPrefixStore, ReusedPrefix, apply_prefix_reuse};
use super::*;

/// A native decoder's session-resident KV, as a [`KvPrefixStore`].
struct NativeKvStore<'a>(&'a mut dyn PipelineDecoderComponent);

impl KvPrefixStore for NativeKvStore<'_> {
    fn current_kv_len(&self) -> usize {
        // `None` means the backend keeps nothing across turns, which is the
        // same starting point as an empty cache.
        self.0.current_kv_len().unwrap_or(0)
    }

    fn use_kv(&self) -> bool {
        self.0.use_kv()
    }

    fn rewind_to(&mut self, target: usize) -> anyhow::Result<bool> {
        self.0.rewind_kv(target)
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        // A decoder that cannot rewind to empty must not keep a cache the next
        // turn would read as its prefix, so a decline is an error here rather
        // than a silent carry-over.
        if self.0.rewind_kv(0)? {
            return Ok(());
        }
        anyhow::bail!("native decoder declined to reset its KV cache")
    }
}

/// ORT's past tensors, held by the pipeline as an `Option<DecodeState>` rather
/// than by the decoder, as a [`KvPrefixStore`]. The state now owns its KV
/// length, so this adapter is the same shape as [`NativeKvStore`]: it reads and
/// rewinds through `current_kv_len` / `rewind_kv` without the pipeline having to
/// thread the length in. The one essential difference is `reset`: ORT rebuilds a
/// fresh `DecodeState` per turn, so emptying the cache means dropping to `None`.
struct OrtKvStore<'a>(&'a mut Option<DecodeState>);

impl KvPrefixStore for OrtKvStore<'_> {
    fn current_kv_len(&self) -> usize {
        self.0.as_ref().map_or(0, DecodeState::current_kv_len)
    }

    fn use_kv(&self) -> bool {
        self.0.as_ref().is_some_and(|state| state.use_kv)
    }

    fn rewind_to(&mut self, target: usize) -> anyhow::Result<bool> {
        match self.0.as_mut() {
            Some(state) => state.rewind_kv(target),
            None => Ok(false),
        }
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        // Dropping the state is how ORT empties its cache: a fresh one is built
        // below for every turn that reuses nothing.
        *self.0 = None;
        Ok(())
    }
}

impl PipelineEngine {
    /// Resolve which components a flat autoregressive decode drives natively,
    /// unifying the pure-native backend and the hybrid env-flag injection so both
    /// funnel through the same component / decoder builders (DRY). A `Native`
    /// backend selects every component; an `Ort` backend consults the env flags,
    /// leaving the default ORT decode path byte-for-byte unchanged.
    fn native_component_selection(
        &self,
        decoder: &str,
        step_components: &[String],
    ) -> NativeComponentSelection {
        if self.decode_backend == EngineDecodeBackend::Native {
            NativeComponentSelection {
                decoder: true,
                step_components: step_components.iter().cloned().collect(),
            }
        } else {
            NativeComponentSelection {
                decoder: native_decoder_selected(decoder),
                step_components: native_step_component_set(),
            }
        }
    }

    /// Core autoregressive execution shared by [`generate_with_callback`] and
    /// [`synthesize`]: run the prompt-phase components, drive the decode loop,
    /// and return the generated tokens alongside the shared tensor pool (external
    /// inputs + prompt-phase outputs) so a caller can run post-decode stages.
    pub(crate) fn run_autoregressive(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
        admission: Option<&mut dyn FnMut()>,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<(GenerateResult, PipelineTensors)> {
        // Guard first: a non-autoregressive pipeline (single-pass / iterative
        // diffusion) has no token decode loop, so surface the actionable error
        // before touching the tokenizer or options.
        let ar = self
            .plan
            .autoregressive_plan()
            .context(
                "generate() requires an autoregressive pipeline; use run_pipeline() for \
                 single-pass or iterative (diffusion) pipelines",
            )?
            .clone();
        let present = pipeline_request.present.clone();
        self.ensure_component_present(&ar.decoder, &present, "autoregressive decoder")?;

        let mut options = pipeline_request.request.options.clone();
        options.validate()?;
        if options.stop_on_eos {
            let eos_token_ids = merge_eos_token_ids(
                &self.eos_token_ids,
                &self.tokenizer()?.eos_token_ids(),
                options.eos_token_id,
            );
            if options.eos_token_id.is_none() {
                options.eos_token_id = eos_token_ids.first().copied();
            }
            for id in eos_token_ids {
                let stop = StopSequence::Tokens(vec![id]);
                if !options.stop_sequences.contains(&stop) {
                    options.stop_sequences.push(stop);
                }
            }
        }
        let prompt_tokens = tokenize_with(self.tokenizer()?, &pipeline_request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if pipeline_request.num_image_tiles == Some(0) {
            anyhow::bail!("image tile count must be greater than zero");
        }
        // TODO(#14): Pipeline metadata must declare the image placeholder token
        // and tokens-per-tile contract. Expand that placeholder here using
        // `num_image_tiles` before DecodeState/KV allocation. The server vision
        // seam should pass ImageTensor::num_tiles via with_image_tile_count().

        let prompt_tokens = expand_image_placeholders_count_based(
            prompt_tokens,
            pipeline_request.num_image_tiles,
            self.models.directory.spec.vision.as_ref(),
        )?;

        // Decide how much of the previous turn's decoder KV this prompt can
        // keep before anything is rebuilt, because the answer decides whether
        // the decode state is recreated or carried over.
        let inputs_digest = Self::digest_request_identity(&pipeline_request);
        // Native selection is resolved once here (GAP-3 Inc-A): a `Native`
        // backend drives every component natively, while an `Ort` backend keeps
        // the hybrid env-flag behaviour — both through the same builders below.
        let native_selection = self.native_component_selection(&ar.decoder, &ar.step_components);
        let use_native_decoder = native_selection.decoder;
        // Build the native decoder up front (GAP-3 Inc-C) so paged prefix reuse
        // can seed its KV before the decode loop starts. It borrows nothing from
        // the decode state, unlike the ORT decoder built later. A native decoder
        // only joins the paged path when its KV is host-resident and f32
        // (`supports_paged_kv`); a device-resident / in-place / f16 store keeps
        // the non-paged fresh-decode path (Inc-A behaviour) — no regression, and
        // the still-unwired case is reported as Inc-D.
        let mut native_decoder_component: Option<Box<dyn PipelineDecoderComponent + 'static>> =
            if use_native_decoder {
                match self.native_retained_decoder.take() {
                    Some(decoder) => Some(decoder),
                    None => Some(build_native_pipeline_decoder(
                        &self.models,
                        &ar.decoder,
                        self.native_device.as_ref(),
                        &self.memory_strategy_plan,
                        #[cfg(feature = "cuda")]
                        std::sync::Arc::new(
                            self.native_cuda_authority
                                .clone()
                                .unwrap_or_else(|| self.resource_governor.device_authority()),
                        ),
                    )?),
                }
            } else {
                None
            };
        let native_supports_paging = native_decoder_component
            .as_ref()
            .is_some_and(|decoder| decoder.supports_paged_kv());
        // The paged cache supersedes the single retained context wherever it is
        // available: it holds many prefixes rather than only the last one. A
        // paged-capable native decoder now mirrors its present KV and reuses a
        // shared prefix through the same paged machinery as ORT.
        let paged_enabled = self.paged.is_some()
            && inputs_digest.is_some()
            && (!use_native_decoder || native_supports_paging);
        // One prefix-reuse policy for every backend (`prefix_reuse`): the
        // shared-prefix question and the decision it feeds are identical, so
        // only the cache mechanism is chosen here. A native decoder rewinds its
        // session-resident KV; ORT slices pipeline-held past tensors.
        let shared = match (inputs_digest, self.retained.as_ref()) {
            (Some(inputs), Some(retained)) => retained.reusable_prefix(inputs, &prompt_tokens),
            _ => 0,
        };
        let positions_are_linear = self.positions_are_linear();
        let reused = if paged_enabled {
            ReusedPrefix::NONE
        } else if use_native_decoder {
            match native_decoder_component.as_mut() {
                Some(decoder) => apply_prefix_reuse(
                    &mut NativeKvStore(decoder.as_mut()),
                    shared,
                    positions_are_linear,
                )?,
                None => ReusedPrefix::NONE,
            }
        } else {
            // The state owns its KV length, so the pipeline no longer threads
            // `retained.tokens.len()` in as the ORT cache's current length.
            apply_prefix_reuse(
                &mut OrtKvStore(&mut self.decoder_state),
                shared,
                positions_are_linear,
            )?
        };
        // Any failure below leaves the decoder KV in an unknown state, so the
        // retention is dropped now and only re-established on success.
        self.retained = None;

        let mut tensors = self.prepare_request_tensors(pipeline_request.inputs, &present)?;
        // Seed the prompt token ids into the shared pool so a prompt-phase
        // component that consumes `input_ids` (e.g. a text encoder) can run.
        self.seed_prompt_token_inputs(&ar.prompt_components, &prompt_tokens, &mut tensors)?;
        self.run_prompt_phase_components(
            &ar.prompt_components,
            &mut tensors,
            "prologue",
            &present,
            None,
        )?;
        // Bind an empty tensor for any every_step component input left absent by
        // an inactive prompt producer (e.g. the embedder's `image_features` when
        // a text-only prompt never ran the vision encoder), so the per-step
        // component still has every graph input bound — the empty image feed the
        // muse_decode harness sends each step.
        self.seed_absent_step_component_inputs(&ar.step_components, &present, &mut tensors)?;

        // Static routing from prompt-phase and per-step producers into the
        // decoder. Every non-self edge into the decoder is recomputed from the
        // shared pool on each step, so `every_step` outputs are always fresh and
        // `prompt_only` conditioning stays cached (it is simply re-read).
        let decoder_in_edges = self.decoder_in_edges(&ar.decoder, &present, &tensors)?;
        // Owned per-step component bindings (paired with their sessions below).
        // Built before `decoder_state` is taken mutably so the immutable borrow
        // used to enumerate graph ports is released first.
        let step_bindings = self.build_step_bindings(&ar.step_components, &present)?;

        // A decoder whose position ids arrive over a dataflow edge receives one
        // tensor covering the whole prompt. Prefilling only a suffix would hand
        // it positions for tokens it is not being given, so such a pipeline
        // recomputes rather than reuses.
        let reused = if !reused.is_empty() && self.decoder_positions_are_routed(&decoder_in_edges) {
            ReusedPrefix::NONE
        } else {
            reused
        };

        let positions_routed = self.decoder_positions_are_routed(&decoder_in_edges);
        // A paged sequence starts from a fresh decode state, because the shared
        // prefix is loaded into it wholesale rather than carried over.
        let mut paged_session = None;
        let mut reused = reused;
        if paged_enabled && !positions_routed {
            self.decoder_state = Some(Self::new_decoder_state(
                &self.models,
                &ar.decoder,
                self.fixed_state_budget_bytes,
            )?);
            let inputs = inputs_digest.expect("paged_enabled implies a digest");
            // A paged-capable native decoder seeds its own session-resident KV
            // from the shared prefix (GAP-3 Inc-C); the ORT decoder loads it into
            // the host `DecodeState`. Both claim the sequence through the same
            // `claim_paged_prefix` helper, so only the KV *sink* differs.
            let (seq, shared) = if let Some(native_decoder) = native_decoder_component.as_mut() {
                let paged = self.paged.as_mut().expect("paged_enabled implies storage");
                Self::admit_native_paged_sequence(
                    paged,
                    native_decoder.as_mut(),
                    inputs,
                    &prompt_tokens,
                )?
            } else {
                let decoder = self
                    .models
                    .session(&ar.decoder)
                    .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
                let paged = self.paged.as_mut().expect("paged_enabled implies storage");
                let state = self
                    .decoder_state
                    .as_mut()
                    .expect("the decode state was just built");
                Self::admit_paged_sequence(paged, state, decoder, inputs, &prompt_tokens)?
            };
            paged_session = Some((seq, inputs));
            reused = ReusedPrefix::from_paged_admission(shared);
        }

        let chain = build_processor_chain(&options, Some(self.tokenizer()?))?;
        if reused.is_empty() && paged_session.is_none() {
            self.decoder_state = Some(Self::new_decoder_state(
                &self.models,
                &ar.decoder,
                self.fixed_state_budget_bytes,
            )?);
        }

        // Encoder-decoder pipelines bind the encoder's static cross-attention KV
        // to the decoder every step. Resolve it once here, after the prompt-phase
        // encoder prologue has published its `present_*_cross_%d` outputs into the
        // shared pool; the tensors are invariant across the decode loop.
        let cross_kv_pairs = self
            .decoder_state
            .as_ref()
            .expect("autoregressive pipeline has decode state")
            .io
            .cross_kv_pairs
            .clone();
        let static_cross_kv = self.static_cross_kv_bindings(&cross_kv_pairs, &tensors)?;

        // Pair every `every_step` binding with a backend-neutral component
        // session. By default this borrows the already-loaded ORT session
        // (behaviour unchanged); components selected natively — every component
        // under a `Native` backend, or those named in
        // `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS` under `Ort` — are instead
        // loaded and driven through the native nxrt backend, so the same decode
        // loop drives both backends through the trait with no forked code path.
        let native_step_components = &native_selection.step_components;
        // Native every_step components load on the same device the native
        // decoder targets, so an embeds-driven pipeline runs its embedder on the
        // CUDA EP next to the decoder (matching `muse_decode`), not on CPU.
        #[cfg(feature = "native-backend")]
        let native_step_device = native_decoder_device(self.native_device.as_ref());
        #[cfg(not(feature = "native-backend"))]
        let native_step_device = crate::native_decode_device::NativeDecodeDevice::Cpu;
        let step_components = step_bindings
            .into_iter()
            .map(|binding| {
                let component: Box<dyn onnx_genai_metadata::ComponentSession> =
                    build_step_component_session(
                        &self.models,
                        &binding.component,
                        native_step_components,
                        &native_step_device,
                    )?;
                Ok((binding, component))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tokenizer = self
            .models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| {
                format!("no tokenizer available for '{}'", self.tokenizer_component)
            })?;
        let paged_mirror = match (paged_session, self.paged.as_mut()) {
            (Some((seq, _)), Some(paged)) => Some(PagedMirror {
                mirrored_tokens: 0,
                exhausted: false,
                windowed: false,
                kv_model: &paged.kv_model,
                cache: &mut paged.cache,
                seq,
            }),
            _ => None,
        };
        let mut ort_decoder_component: Option<Box<dyn PipelineDecoderComponent + '_>> = None;
        let decoder_component: &mut dyn PipelineDecoderComponent =
            if let Some(native) = native_decoder_component.as_deref_mut() {
                // Built (and, on a shared prefix, already KV-seeded) up front.
                native
            } else {
                // The ORT decode path needs the component's ORT session. Native
                // decoders run without one (the loader skips it), so the session
                // is only resolved on this branch to avoid requiring a session a
                // native-only artifact never built.
                let decoder = self
                    .models
                    .session(&ar.decoder)
                    .with_context(|| format!("pipeline decoder '{}' was not loaded", ar.decoder))?;
                ort_decoder_component = Some(Box::new(OrtPipelineDecoder::new(
                    decoder,
                    self.decoder_state
                        .as_mut()
                        .expect("autoregressive pipeline has decode state"),
                )));
                ort_decoder_component
                    .as_deref_mut()
                    .expect("ORT decoder component was just initialized")
            };
        let mut backend = PipelineDecodeLoopBackend {
            decoder: decoder_component,
            admission,
            paged: paged_mirror,
            pool: &mut tensors,
            step_components,
            decoder_in_edges,
            static_cross_kv,
            context_tokens: prompt_tokens,
            retained_len: reused.len(),
            prompt_len: 0,
            generated_count: 0,
            kv_len: reused.len(),
            prefill_chunk_size: self.prefill_chunk_size,
        };
        // Prefill only what the retained KV does not already cover.
        backend.prompt_len = backend.context_tokens.len() - backend.retained_len;
        let prefilled = backend.prompt_len;
        let mut loop_state = DecodeLoopState::new(reused.len(), options.seed, options.top_logprobs);
        // Taken without `?` so a failed generation still releases its sequence
        // below: an abandoned sequence holds its pages out of the pool for the
        // life of the process.
        let result = run_decode_loop(
            &mut backend,
            &mut loop_state,
            &options,
            &chain,
            tokenizer,
            None,
            callback,
        );
        // Exactly the tokens whose KV the decoder now holds. Truncated to
        // `kv_len` rather than taken whole: the last sampled token was committed
        // to the context but never fed to the decoder, so its KV does not exist
        // and the next turn must prefill it.
        let mut final_context = backend.context_tokens.clone();
        final_context.truncate(backend.kv_len);
        let retains_kv = backend.decoder.use_kv();
        // How far mirroring actually got. Equal to the context length unless
        // the page pool ran dry, in which case only this prefix may be
        // published for reuse.
        let mirrored_tokens = backend.paged.as_ref().map_or(0, |mirror| {
            if mirror.windowed {
                0
            } else {
                mirror.mirrored_tokens
            }
        });
        // The backend now owns its every_step component sessions behind
        // `Box<dyn ComponentSession>`, so it carries drop glue and its borrows of
        // `tensors` / `self` would otherwise live to end of scope. Everything
        // needed downstream has been copied out above, so release it explicitly
        // before the paged-sequence retirement and the `tensors` move below.
        drop(backend);
        drop(ort_decoder_component);
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.native_retained_decoder = None;
                if let Some(paged) = self.paged.as_mut() {
                    paged.discard_active();
                }
                return Err(error);
            }
        };

        self.component_cache
            .borrow_mut()
            .note_prefix_reuse(reused.len(), prefilled);
        match paged_session {
            // Publish what this generation computed so the next request can
            // attach to it, then let go of the sequence.
            Some((seq, inputs)) => {
                self.retire_paged_sequence(seq, inputs, &final_context, mirrored_tokens)?
            }
            None => {
                if retains_kv && let Some(inputs) = inputs_digest {
                    self.retained = Some(RetainedContext {
                        inputs,
                        tokens: final_context,
                    });
                    // Retain the decoder only here, under exactly the condition
                    // that refreshed `retained`. The next request rewinds this
                    // KV to a prefix length derived from `retained`, so a
                    // decoder kept without a matching `retained` would have its
                    // contents described by a stale context -- reusing another
                    // request's KV as this one's prefix, silently and with
                    // plausible output. Reachable whenever a request is not
                    // digestible (`digest_request_identity` returns `None` for
                    // an input `absorb_value` cannot canonicalize) or does not
                    // use KV: those requests would leave `retained` untouched
                    // while overwriting the KV it claims to describe. Dropping
                    // the decoder only costs the reuse.
                    if use_native_decoder {
                        self.native_retained_decoder = native_decoder_component;
                    }
                }
            }
        }
        Ok((result, tensors))
    }

    /// A fresh decode state for `decoder`.
    fn new_decoder_state(
        models: &PipelineModels,
        decoder: &str,
        fixed_state_budget_bytes: u64,
    ) -> anyhow::Result<DecodeState> {
        let session = models
            .graph_io(decoder)
            .with_context(|| format!("pipeline decoder '{decoder}' was not loaded"))?;
        let decoder_io = models
            .directory
            .spec
            .models
            .get(decoder)
            .and_then(|component| component.io.as_ref());
        DecodeState::new_with_io_positions_and_state_budget(
            session,
            decoder_io,
            models.directory.spec.positions.as_ref(),
            fixed_state_budget_bytes,
        )
    }

    /// Claim a paged sequence for this request and, if its prompt shares a
    /// prefix with an earlier request, attach that prefix's pages and materialize
    /// them. Returns the sequence id, how many leading tokens it now holds KV for
    /// (always at least one token short of the prompt, since a decode step needs
    /// an input to produce logits from), and the materialized shared KV when
    /// there is a reusable prefix.
    ///
    /// Backend-neutral: the caller injects the materialized KV into whichever KV
    /// store its decoder uses (the ORT host `DecodeState` or a native session),
    /// so both backends share this claim/lookup logic (DRY).
    fn claim_paged_prefix(
        paged: &mut PipelinePagedKv,
        inputs: Digest,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<(SequenceId, usize, Option<onnx_genai_kv::MaterializedKv>)> {
        // Free anything a previous generation abandoned, then make room for this
        // one, before claiming any pages.
        paged.discard_active();
        paged.evict_until_free(
            prompt_tokens
                .len()
                .div_ceil(paged.cache.page_table.page_size),
        );
        let key = prefix_key(inputs, prompt_tokens);
        let seq = paged.cache.create_sequence();
        paged.active = Some(seq);
        let matched = paged
            .prefix
            .lookup_shared(&key, &mut paged.cache.page_table);

        // A match shorter than the preamble cannot happen — every stored key
        // begins with one — but treat it as no match rather than trusting it.
        let reusable = matched
            .matched_tokens
            .saturating_sub(PREFIX_KEY_PREAMBLE)
            .min(prompt_tokens.len().saturating_sub(1));
        if matched.matched_tokens > 0 {
            let pages = matched
                .page_ids
                .iter()
                .copied()
                .take(reusable.div_ceil(paged.cache.page_table.page_size))
                .collect::<Vec<_>>();
            for &page_id in &pages {
                paged.cache.page_table.retain(page_id);
            }
            paged
                .prefix
                .release_shared(&key, matched.matched_tokens, &mut paged.cache.page_table);
            if reusable > 0 {
                attach_pages_to_sequence(&mut paged.cache, seq, &pages, reusable)?;
                let materialized = paged
                    .cache
                    .materialize_sequence(seq)
                    .map_err(|e| anyhow::anyhow!("failed to materialize the shared prefix: {e}"))?;
                return Ok((seq, reusable, Some(materialized)));
            }
        }
        Ok((seq, 0, None))
    }

    /// Claim a paged sequence for an ORT decoder, loading any shared prefix into
    /// its host [`DecodeState`].
    fn admit_paged_sequence(
        paged: &mut PipelinePagedKv,
        state: &mut DecodeState,
        decoder: &Session,
        inputs: Digest,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<(SequenceId, usize)> {
        let (seq, reusable, materialized) = Self::claim_paged_prefix(paged, inputs, prompt_tokens)?;
        if let Some(materialized) = materialized {
            load_materialized_past(decoder, &paged.kv_model, state, &materialized)?;
        }
        Ok((seq, reusable))
    }

    /// Claim a paged sequence for a native decoder, seeding any shared prefix
    /// into its session-resident KV (GAP-3 Inc-C). The native decoder must report
    /// [`supports_paged_kv`](PipelineDecoderComponent::supports_paged_kv).
    fn admit_native_paged_sequence(
        paged: &mut PipelinePagedKv,
        decoder: &mut dyn PipelineDecoderComponent,
        inputs: Digest,
        prompt_tokens: &[TokenId],
    ) -> anyhow::Result<(SequenceId, usize)> {
        let (seq, reusable, materialized) = Self::claim_paged_prefix(paged, inputs, prompt_tokens)?;
        if let Some(materialized) = materialized {
            decoder.load_paged_prefix(&paged.kv_model, &materialized)?;
        }
        Ok((seq, reusable))
    }

    /// Record this generation's KV under its prefix key and release the
    /// sequence.
    ///
    /// Pages the prefix cache kept are retained by it, so freeing the sequence
    /// returns only what nothing else refers to.
    fn retire_paged_sequence(
        &mut self,
        seq: SequenceId,
        inputs: Digest,
        tokens: &[TokenId],
        mirrored_tokens: usize,
    ) -> anyhow::Result<()> {
        let Some(paged) = self.paged.as_mut() else {
            return Ok(());
        };
        // Never publish past what was mirrored. If the pool ran dry mid-decode
        // the later pages were never written, and a key covering them would
        // hand a future request KV that does not exist.
        let tokens = &tokens[..tokens.len().min(mirrored_tokens)];
        // Publish at every page boundary, not only at the full length.
        //
        // The trie only reports a match where something was inserted, so a
        // prompt that diverges from this one matches nothing unless the shared
        // part was itself published. Page boundaries are the natural granularity:
        // a page is the smallest unit the table can hand to another sequence, so
        // publishing there is what lets two conversations share the head they
        // have in common rather than only exact repeats.
        let page_size = paged.cache.page_table.page_size;
        let mut lengths = (1..)
            .map(|pages| pages * page_size)
            .take_while(|&len| len < tokens.len())
            .collect::<Vec<_>>();
        lengths.push(tokens.len());
        for len in lengths {
            if len == 0 {
                continue;
            }
            let key = prefix_key(inputs, &tokens[..len]);
            if paged.prefix.lookup(&key).0 == key.len() {
                continue;
            }
            let pages = sequence_pages_for_len(&paged.cache, seq, len)?;
            paged
                .prefix
                .insert_pages(&key, &pages, &mut paged.cache.page_table);
        }
        if paged.active == Some(seq) {
            paged.active = None;
        }
        for page_id in paged.cache.page_table.remove_sequence(seq) {
            paged.cache.page_table.free(page_id);
        }
        Ok(())
    }

    /// Whether position ids are a plain function of the absolute past length,
    /// and so can be rebuilt after the KV is truncated.
    fn positions_are_linear(&self) -> bool {
        self.models
            .directory
            .spec
            .positions
            .as_ref()
            .is_none_or(|program| {
                program
                    .continuation
                    .as_deref()
                    .is_none_or(|continuation| continuation == "linear_increment")
            })
    }

    /// Whether the decoder's position ids are supplied by a dataflow edge
    /// rather than derived from the absolute past length.
    fn decoder_positions_are_routed(&self, decoder_in_edges: &[(String, String)]) -> bool {
        let Some(position_input) = self
            .decoder_state
            .as_ref()
            .and_then(|state| state.io.position_ids_input.as_deref())
        else {
            return false;
        };
        decoder_in_edges
            .iter()
            .any(|(_, port)| port == position_input)
    }
}

/// Every end-of-turn token the request should stop on.
///
/// Three sources declare one and none of them is authoritative on its own: the
/// package metadata (which knows the chat template's end-of-turn marker), the
/// tokenizer (which knows the model's), and the caller (which may pin a
/// specific one). Taking only the caller's when it is set is what let a
/// package-declared marker be dropped -- the OpenAI server always supplies an
/// id, so every request through it ran past the end of its answer and stopped
/// only on the token budget. Union them; a superfluous stop token costs
/// nothing, a missing one costs the whole reply.
///
/// Order is meaningful: the declared ids come first so that a caller who left
/// `eos_token_id` unset inherits the package's own marker rather than the
/// tokenizer's.
fn merge_eos_token_ids(
    declared: &[TokenId],
    from_tokenizer: &[TokenId],
    from_caller: Option<TokenId>,
) -> Vec<TokenId> {
    let mut merged: Vec<TokenId> = declared.to_vec();
    for id in from_tokenizer.iter().copied().chain(from_caller) {
        if !merged.contains(&id) {
            merged.push(id);
        }
    }
    merged
}

#[cfg(test)]
mod eos_tests {
    use super::merge_eos_token_ids;

    #[test]
    fn a_caller_supplied_eos_does_not_displace_the_declared_one() {
        assert_eq!(
            merge_eos_token_ids(&[200012], &[199999], Some(200002)),
            vec![200012, 199999, 200002],
            "the package's declared end-of-turn marker must survive a caller that \
             pins its own id -- dropping it is what made server requests generate \
             past the end of the answer until they hit the token budget"
        );
    }

    #[test]
    fn the_declared_marker_leads_so_an_unset_caller_inherits_it() {
        let merged = merge_eos_token_ids(&[200012], &[199999], None);
        assert_eq!(
            merged.first().copied(),
            Some(200012),
            "callers that leave eos_token_id unset take the first id, so the \
             package's own marker has to lead the tokenizer's"
        );
    }

    #[test]
    fn duplicates_collapse_across_all_three_sources() {
        assert_eq!(
            merge_eos_token_ids(&[7, 7], &[7, 9], Some(9)),
            vec![7, 7, 9],
            "de-duplication is against the accumulated set, not within the \
             declared list, which is left exactly as the package wrote it"
        );
    }

    #[test]
    fn no_declared_marker_leaves_the_other_two_sources_intact() {
        assert_eq!(
            merge_eos_token_ids(&[], &[199999], Some(2)),
            vec![199999, 2]
        );
        assert!(merge_eos_token_ids(&[], &[], None).is_empty());
    }
}
